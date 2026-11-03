"""ATE-6431 — a held state.db write lock at TURN END must never cost the turn its reply.

Measured 2026-10-19 00:58-00:59 EDT: one writer held the SQLite write lock for >60 s, 16 Hermes turns
expired their 60 s transcript patience and 12 runs ended ``adapter_failed`` / "No reply: session
storage was busy" — although the model call had completed and only the final persist was missing.

Contract pinned here:

1. ``_db_flush_failed`` classifies a ``locked`` error as ``"queued"`` when the caller is a turn-end
   flush, and keeps ``None`` (fail closed) for every other caller — the mid-turn pre-tool flush must
   not run side effects from process-only state.
2. The turn-end flush returns success to its caller (the reply goes to the adapter) while the batch
   waits on the bounded retry queue.
3. The retried write lands the transcript rows EXACTLY ONCE: markers are stamped only on commit, so a
   replay after a late landing cannot duplicate a row.
4. The queue is bounded and drops the OLDEST batch with an error log, so a stalled database cannot
   grow memory.
5. A queued batch lands in the session it was COLLECTED for, even if the agent was rebound to a
   compression tip while it waited (B1, ATE-6552).
6. ``_LOCK_RETRY_MAX_AGE_S`` is the binding bound: the attempt cap can never expire a batch the age
   still allows, and the last sleep is clipped to the remaining age (M1).
7. An EXIT flush (gateway finalize, reaper, atexit) waits the lock out in-process instead of parking
   on a daemon thread that dies with the interpreter (M2).
8. The retry mirrors the shadowed-checkpoint drop on the batch's own live dicts, not on ``[]`` (m1).
9. The queue is the session's FIFO for BOTH writers: a live flush settles it before writing instead
   of overtaking a waiting batch (m2).
"""

import sqlite3
import threading
import time
import types

from unittest.mock import MagicMock

import pytest

from agent import session_persistence
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.session_persistence import (
    _db_flush_failed,
    _lock_retry_queue,
    _LockRetryQueue,
    pending_lock_retries,
)
from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """Keep the real backoff shape, shrink the clock so the test finishes in milliseconds."""
    monkeypatch.setattr(session_persistence, "_LOCK_RETRY_BASE_DELAY_S", 0.01)
    monkeypatch.setattr(session_persistence, "_LOCK_RETRY_MAX_DELAY_S", 0.02)
    _lock_retry_queue.reset()
    yield
    _lock_retry_queue.await_idle(5.0)
    _lock_retry_queue.reset()


def _flush_agent(db, key):
    """Agent shell owning the real flush (same shape the tui_gateway persistence tests use)."""
    agent = types.SimpleNamespace(
        _session_db=db, _session_db_created=True, _persist_disabled=False, session_id=key,
        _session_persist_lock=threading.RLock(), _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0, _persist_user_message_idx=None, _persist_user_message_override=None,
        _persist_user_message_timestamp=None, _pending_cli_user_message=None, _db_flush_scan_prefix=None,
        _session_row_replay_pending=None, _parent_session_id=None)
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    agent._flush_messages_to_session_db_unlocked = AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    return agent


def _stub_agent(session_id, *, cause="database is locked"):
    """Minimal shell for the classifier-level assertions (no store behind it)."""
    db = MagicMock()
    db.append_messages_batch.side_effect = sqlite3.OperationalError(cause)
    return types.SimpleNamespace(
        session_id=session_id, _session_db=db, _session_persist_lock=None,
        _last_persistence_error_cause=None, _db_flush_scan_prefix=None, _session_db_created=True,
        _flushed_db_message_ids=set(), _last_flushed_db_idx=0,
    )


def test_locked_error_at_turn_end_is_queued_not_failed():
    """(a)+(b) at the classifier seam, and the mid-turn default stays fail-closed."""
    agent = _stub_agent("sess-queued")
    msg = {"role": "assistant", "content": "the answer"}
    rows = [{"role": "assistant", "content": "the answer"}]

    retry = _db_flush_failed(agent, sqlite3.OperationalError("database is locked"), rows, 1, [msg],
                             batch_msgs=[msg], queue_on_lock=True)

    assert retry == "queued"
    assert agent._last_persistence_error_cause == "locked"
    assert pending_lock_retries() == 1

    # Mid-turn callers (pre-tool flush) keep the old contract: fail closed, never queue.
    mid = _stub_agent("sess-midturn")
    assert _db_flush_failed(mid, sqlite3.OperationalError("database is locked"), rows, 1, [msg],
                            batch_msgs=[msg]) is None


def test_non_lock_error_at_turn_end_still_fails_closed():
    """Only a lock is a wait: a disk/corruption class must not be queued behind a retry loop."""
    agent = _stub_agent("sess-disk")
    msg = {"role": "assistant", "content": "x"}
    assert _db_flush_failed(agent, sqlite3.OperationalError("disk I/O error"), [{"role": "assistant"}], 1,
                            [msg], batch_msgs=[msg], queue_on_lock=True) is None
    assert pending_lock_retries() == 0


def test_turn_end_locked_flush_returns_reply_and_lands_exactly_once(tmp_path):
    """The card's acceptance test: (a) turn result returned, (b) batch retried, (c) one row after landing."""
    db = SessionDB(db_path=tmp_path / "state.db")
    key = "sess-lock-retry"
    db.create_session(key, source="pytest")
    agent = _flush_agent(db, key)

    real_write = db.append_messages_batch
    calls = {"n": 0}

    def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:  # the live attempt and the first retry both hit the held lock
            raise sqlite3.OperationalError("database is locked")
        return real_write(*args, **kwargs)

    db.append_messages_batch = _flaky
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "the answer"}]

    # (a) the turn-end flush reports success, so the turn returns its reply instead of aborting
    assert agent._flush_messages_to_session_db(messages, [], queue_on_lock=True) is True
    live_calls = calls["n"]
    assert live_calls >= 1

    # (b) the batch was retried until it landed (the live attempt failed twice before the lock cleared)
    assert _lock_retry_queue.await_idle(10.0), "retry queue did not drain"
    assert calls["n"] >= 3
    assert pending_lock_retries() == 0

    # (c) the transcript holds the turn exactly once
    rows = db.get_messages_as_conversation(key)
    assert [m["role"] for m in rows] == ["user", "assistant"]
    assert [m["content"] for m in rows] == ["hello", "the answer"]

    # A replay of the same dicts (the finalize retry path) must not duplicate the landed rows.
    assert agent._flush_messages_to_session_db(messages, [], queue_on_lock=True) is True
    assert all(m.get(_DB_PERSISTED_MARKER) for m in messages)
    replay = db.get_messages_as_conversation(key)
    assert [m["content"] for m in replay] == ["hello", "the answer"]


def test_queued_batch_skipped_when_a_live_flush_already_landed_it(tmp_path):
    """Marker is the truth: a queued batch that a live flush overtook must not write a second row."""
    db = SessionDB(db_path=tmp_path / "state.db")
    key = "sess-overtaken"
    db.create_session(key, source="pytest")
    agent = _flush_agent(db, key)

    msg = {"role": "assistant", "content": "only once"}
    rows = [{"role": "assistant", "content": "only once"}]
    assert _db_flush_failed(agent, sqlite3.OperationalError("database is locked"), rows, 1, [msg],
                            batch_msgs=[msg], queue_on_lock=True) == "queued"
    # The live path wins the race and writes the same dict before the worker gets there.
    assert agent._flush_messages_to_session_db([msg], [], queue_on_lock=False) is True
    assert _lock_retry_queue.await_idle(10.0)

    assert [m["content"] for m in db.get_messages_as_conversation(key)] == ["only once"]


def test_mid_turn_locked_flush_waits_for_the_lock_instead_of_failing(tmp_path, monkeypatch):
    """The pre-tool flush cannot queue (tools need the row first) — it waits the lock out, bounded."""
    monkeypatch.setattr(session_persistence, "_LOCK_TURN_WAIT_S", 5.0)
    db = SessionDB(db_path=tmp_path / "state.db")
    key = "sess-midturn"
    db.create_session(key, source="pytest")
    agent = _flush_agent(db, key)

    real_write = db.append_messages_batch
    calls = {"n": 0}

    def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise sqlite3.OperationalError("database is locked")
        return real_write(*args, **kwargs)

    db.append_messages_batch = _flaky
    messages = [{"role": "user", "content": "run a tool"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "read_file", "arguments": "{}"}}]}]

    assert agent._flush_messages_to_session_db(messages, []) is True
    assert calls["n"] >= 3
    assert [m["role"] for m in db.get_messages_as_conversation(key)] == ["user", "assistant"]


def test_mid_turn_permanent_lock_still_fails_closed_within_budget(tmp_path, monkeypatch):
    """No side effects from process-only state: an unsurmountable lock still reports failure."""
    monkeypatch.setattr(session_persistence, "_LOCK_TURN_WAIT_S", 0.2)
    db = SessionDB(db_path=tmp_path / "state.db")
    key = "sess-midturn-perm"
    db.create_session(key, source="pytest")
    agent = _flush_agent(db, key)

    def _always_locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    db.append_messages_batch = _always_locked
    messages = [{"role": "user", "content": "hello"}]

    assert agent._flush_messages_to_session_db(messages, []) is False
    assert agent._last_persistence_error_cause == "locked"
    assert db.get_messages_as_conversation(key) == []


def test_retry_queue_is_bounded_and_drops_the_oldest(monkeypatch):
    """A stalled database must cost memory only up to the bound, and the drop must be loud."""
    queue = _LockRetryQueue(max_batches=2)
    monkeypatch.setattr(queue, "_ensure_worker_locked", lambda: None)  # drive enqueue/reset only

    agents = []
    for i in range(3):
        agent = _stub_agent(f"sess-bound-{i}")
        agents.append(agent)
        queue.enqueue(agent, f"sess-bound-{i}", [{"role": "assistant", "content": str(i)}],
                      [{"role": "assistant", "content": str(i)}])

    assert len(queue._entries) == 2
    assert queue.dropped == 1
    assert [e["session_id"] for e in queue._entries] == ["sess-bound-1", "sess-bound-2"]
    queue.reset()


def test_queued_batch_lands_in_the_session_it_was_collected_for(tmp_path):
    """B1 (ATE-6552 probe): a rebind after enqueue must not move the queued rows to the new session."""
    db = SessionDB(db_path=tmp_path / "state.db")
    original, rotated = "sess-ORIGINAL", "sess-ROTATED"
    db.create_session(original, source="pytest")
    db.create_session(rotated, source="pytest")
    agent = _flush_agent(db, original)

    real_write = db.append_messages_batch
    calls = {"n": 0}

    def _first_locked(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # the live turn-end attempt hits the held lock, so the batch is queued
            raise sqlite3.OperationalError("database is locked")
        return real_write(*args, **kwargs)

    db.append_messages_batch = _first_locked
    messages = [{"role": "assistant", "content": "belongs to ORIGINAL"}]
    assert agent._flush_messages_to_session_db(messages, [], queue_on_lock=True) is True
    assert pending_lock_retries() == 1

    # What _db_flush_adopt_compression_tip does while the batch waits.
    agent.session_id = rotated
    assert _lock_retry_queue.await_idle(10.0)

    assert [m["content"] for m in db.get_messages_as_conversation(original)] == ["belongs to ORIGINAL"]
    assert db.get_messages_as_conversation(rotated) == []


def test_retry_age_is_the_binding_bound(monkeypatch):
    """M1: the advertised 600 s must be reachable — the attempt cap must not expire a batch the age allows."""
    # The autouse fixture shrinks the clock; put the shipped backoff numbers back for this invariant.
    monkeypatch.setattr(session_persistence, "_LOCK_RETRY_BASE_DELAY_S", 0.25)
    monkeypatch.setattr(session_persistence, "_LOCK_RETRY_MAX_DELAY_S", 3.0)

    def _min_delay(attempt: int) -> float:
        base = min(session_persistence._LOCK_RETRY_BASE_DELAY_S * (1.7 ** max(0, attempt)),
                   session_persistence._LOCK_RETRY_MAX_DELAY_S)
        return base * 0.6  # the jitter floor

    reachable_s = sum(_min_delay(a) for a in range(session_persistence._LOCK_RETRY_MAX_ATTEMPTS))
    assert reachable_s >= session_persistence._LOCK_RETRY_MAX_AGE_S


def test_aged_retry_gives_up_on_the_age_and_clips_its_last_wait(monkeypatch):
    """M1: the AGE ends the retry (the attempt cap is only a net), and no sleep starts past the deadline."""
    monkeypatch.setattr(session_persistence, "_LOCK_RETRY_MAX_AGE_S", 10.0)
    agent = _stub_agent("sess-aged")
    msg = {"role": "assistant", "content": "x"}
    entry = {"agent": agent, "session_id": "sess-aged",
             "batch_rows": [{"role": "assistant", "content": "x"}], "batch_msgs": [msg],
             "attempts": 0, "enqueued_at": time.monotonic() - 9.8}

    wait_s = session_persistence._lock_retry_queue._attempt(entry)
    assert wait_s is not None and 0 < wait_s <= 0.2  # clipped to the remaining age, not the 3 s cap
    assert entry["attempts"] < session_persistence._LOCK_RETRY_MAX_ATTEMPTS

    entry["enqueued_at"] = time.monotonic() - 10.5
    assert session_persistence._lock_retry_queue._attempt(entry) is None
    assert pending_lock_retries() == 0


def test_exit_flush_waits_in_process_instead_of_parking_on_the_queue(tmp_path, monkeypatch):
    """M2: a shutdown/exit flush must not hand its batch to a daemon thread that dies with the interpreter."""
    monkeypatch.setattr(session_persistence, "_LOCK_TURN_WAIT_S", 0.2)

    def _agent_with(locked_db, key):
        agent = _flush_agent(locked_db, key)
        agent._persist_session = AIAgent._persist_session.__get__(agent, AIAgent)
        agent._drop_trailing_empty_response_scaffolding = (
            AIAgent._drop_trailing_empty_response_scaffolding.__get__(agent, AIAgent))
        return agent

    def _always_locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    db = SessionDB(db_path=tmp_path / "state.db")
    key = "sess-exit"
    db.create_session(key, source="pytest")
    db.append_messages_batch = _always_locked
    agent = _agent_with(db, key)
    messages = [{"role": "assistant", "content": "goodbye"}]

    agent._persist_session(messages, queue_on_lock=False)  # exit caller: waits, returns, never raises
    assert pending_lock_retries() == 0

    # The turn-end default still parks the batch (the reply exists, the worker outlives the turn).
    db2 = SessionDB(db_path=tmp_path / "state2.db")
    key2 = "sess-turn-end"
    db2.create_session(key2, source="pytest")
    db2.append_messages_batch = _always_locked
    agent2 = _agent_with(db2, key2)
    agent2._persist_session([{"role": "assistant", "content": "reply"}])
    assert pending_lock_retries() == 1


def test_retry_mirrors_the_checkpoint_drop_on_the_live_transcript(tmp_path, monkeypatch):
    """m1: the retry writes the batch dicts as the live transcript, so the checkpoint drop mirrors on them."""
    seen = []
    real_drop = session_persistence.drop_shadowed_checkpoints
    monkeypatch.setattr(session_persistence, "drop_shadowed_checkpoints",
                        lambda msgs: (seen.append(list(msgs)), real_drop(msgs))[1])

    db = SessionDB(db_path=tmp_path / "state.db")
    key = "sess-checkpoint"
    db.create_session(key, source="pytest")
    agent = _flush_agent(db, key)

    real_write = db.append_messages_batch
    calls = {"n": 0}

    def _first_locked(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_write(*args, **kwargs)

    db.append_messages_batch = _first_locked
    msg = {"role": "assistant", "content": "carries a checkpoint",
           "codex_reasoning_items": [{"type": "compaction", "id": "r1"}]}
    assert agent._flush_messages_to_session_db([msg], [], queue_on_lock=True) is True
    assert _lock_retry_queue.await_idle(10.0)

    assert seen, "the checkpoint drop never ran on the retry path"
    assert seen[-1] == [msg]  # was [] before the fix: the live dicts never saw the drop


def test_live_flush_does_not_overtake_a_queued_batch(tmp_path):
    """m2: the live writer must wait for an older queued batch, not race it into the transcript."""
    db = SessionDB(db_path=tmp_path / "state.db")
    key = "sess-fifo"
    db.create_session(key, source="pytest")
    agent = _flush_agent(db, key)

    order = []
    real_write = db.append_messages_batch
    state = {"fail_next_queued": True}

    def _record(*args, **kwargs):
        content = ((kwargs.get("messages") or [{}])[0]).get("content")
        if content == "queued" and state["fail_next_queued"]:
            state["fail_next_queued"] = False  # the live turn-end attempt is the one that sees the lock
            raise sqlite3.OperationalError("database is locked")
        order.append(content)
        return real_write(*args, **kwargs)

    db.append_messages_batch = _record
    assert agent._flush_messages_to_session_db([{"role": "assistant", "content": "queued"}], [],
                                               queue_on_lock=True) is True
    assert pending_lock_retries() == 1

    # An older batch is waiting: the live write must land AFTER it (before the fix it wrote first).
    assert agent._flush_messages_to_session_db([{"role": "assistant", "content": "live"}], []) is True
    assert _lock_retry_queue.await_idle(10.0)

    assert order == ["queued", "live"]
    assert [m["content"] for m in db.get_messages_as_conversation(key)] == ["queued", "live"]
