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
"""

import sqlite3
import threading
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
