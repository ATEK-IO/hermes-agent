"""``sessions.fts_trigram`` gate: the trigram family is optional (ATE-13813 / ATE-13351 change 2).

With the key ``false`` opening a store that carries ``messages_fts_trigram`` must remove it and
never rebuild it — the rebuild is a full scan of ``messages`` (~4 GB on the fleet store) taken
inside the shared admission gate. With the key absent or ``true`` nothing changes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_fts import trigram_fts_config_enabled

_TRIGRAM_SQL_PREFIX = "messages_fts_trigram"


def _family_objects(db_path: Path) -> set[str]:
    """Every sqlite object named after the trigram family: vtable, shadows, view, triggers."""
    with sqlite3.connect(db_path) as conn:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE ?", (f"{_TRIGRAM_SQL_PREFIX}%",)
            )
        }


def _message_ids(db_path: Path) -> list[int]:
    with sqlite3.connect(db_path) as conn:
        return [row[0] for row in conn.execute("SELECT id FROM messages ORDER BY id")]


@pytest.fixture()
def statements(monkeypatch) -> list[str]:
    """Every SQL statement the store executes, schema ensure included."""
    seen: list[str] = []
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(seen.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    return seen


@pytest.fixture()
def rebuild_calls(monkeypatch) -> list[object]:
    """Calls into the admission-gated startup rebuild (the fault path for a missing family)."""
    seen: list[object] = []
    real = SessionDB._run_admitted_startup_rebuild

    def wrapper(self, cursor, rebuild_fn):
        seen.append(rebuild_fn)
        return real(self, cursor, rebuild_fn)

    monkeypatch.setattr(SessionDB, "_run_admitted_startup_rebuild", wrapper)
    return seen


def _seed(db_path: Path) -> None:
    """A store built with the family on: one row per route the family serves."""
    db = SessionDB(db_path=db_path)
    if db._trigram_available is not True:
        db.close()
        pytest.skip("this build has no fts5 trigram tokenizer, so the family cannot exist")
    db.create_session(session_id="s1", source="cli", model="m")
    db.append_message("s1", role="user", content="alpha beta gamma")
    db.append_message("s1", role="assistant", content="refactoring the beta plan")
    db.append_message("s1", role="user", content="웅기가 프로필을 요청했다")
    db.close()


def test_gate_off_drops_family_without_rebuild(tmp_path, statements, rebuild_calls, monkeypatch):
    db_path = tmp_path / "state.db"
    monkeypatch.setenv("HERMES_FTS_TRIGRAM", "true")
    _seed(db_path)
    assert f"{_TRIGRAM_SQL_PREFIX}_docsize" in _family_objects(db_path)
    rows_before = _message_ids(db_path)

    statements.clear()
    rebuild_calls.clear()
    monkeypatch.setenv("HERMES_FTS_TRIGRAM", "false")
    db = SessionDB(db_path=db_path)
    try:
        assert db._trigram_available is False
        # Family removed: vtable, shadow tables, source view and all three triggers.
        assert _family_objects(db_path) == set()
        # The drop itself ran, so this is a removal and not a store that never had the family.
        assert any("drop table if exists messages_fts_trigram" in sql.lower() for sql in statements)
        # No rebuild anywhere: the 4 GB scan is exactly what the gate exists to avoid.
        assert not [sql for sql in statements if "values('rebuild')" in sql.lower()]
        assert rebuild_calls == [], "missing-trigger fault path must not run for a gated-off family"
        # The drop reads no message row and writes none.
        assert _message_ids(db_path) == rows_before
        # Trigram searches route to the scan the code already had.
        assert [row["id"] for row in db.search_messages("actoring")] == _expected_substring_ids(
            db_path
        )
    finally:
        db.close()

    # Second open: already absent, still no fault and still no rebuild.
    statements.clear()
    rebuild_calls.clear()
    db = SessionDB(db_path=db_path)
    try:
        assert db._trigram_available is False
        assert not [sql for sql in statements if "values('rebuild')" in sql.lower()]
        assert rebuild_calls == []
    finally:
        db.close()


def _expected_substring_ids(db_path: Path) -> list[int]:
    with sqlite3.connect(db_path) as conn:
        return [
            row[0]
            for row in conn.execute(
                "SELECT id FROM messages WHERE content LIKE '%actoring%' ORDER BY id"
            )
        ]


def test_gate_absent_or_true_is_todays_behaviour(tmp_path, statements, rebuild_calls, monkeypatch):
    db_path = tmp_path / "state.db"
    monkeypatch.delenv("HERMES_FTS_TRIGRAM", raising=False)
    assert trigram_fts_config_enabled() is True

    rebuilt: list[object] = []
    statements.clear()
    db = SessionDB(db_path=db_path)
    try:
        rebuilt[:] = rebuild_calls
        assert db._trigram_available is True
        names = _family_objects(db_path)
        assert {
            _TRIGRAM_SQL_PREFIX,
            f"{_TRIGRAM_SQL_PREFIX}_docsize",
            f"{_TRIGRAM_SQL_PREFIX}_src",
            f"{_TRIGRAM_SQL_PREFIX}_insert",
            f"{_TRIGRAM_SQL_PREFIX}_delete",
            f"{_TRIGRAM_SQL_PREFIX}_update",
        } <= names
        # The family exists because the DDL ran on this open (not because the file had it).
        assert any(_TRIGRAM_SQL_PREFIX in sql for sql in statements)
        db.create_session(session_id="s1", source="cli", model="m")
        db.append_message("s1", role="user", content="alpha beta gamma")
        # The incremental merge still runs over both index families.
        assert db._merge_fts_incrementally(max_pages=8) >= 1
        assert rebuilt, "a fresh store takes the admitted bootstrap path"
    finally:
        db.close()

    # Reopen with the key explicitly true: family kept, no drop.
    monkeypatch.setenv("HERMES_FTS_TRIGRAM", "true")
    statements.clear()
    db = SessionDB(db_path=db_path)
    try:
        assert db._trigram_available is True
        assert f"{_TRIGRAM_SQL_PREFIX}_docsize" in _family_objects(db_path)
        assert not [
            sql for sql in statements
            if sql.lower().startswith("drop") and "messages_fts_trigram" in sql.lower()
        ], "an enabled family must never be removed"
    finally:
        db.close()


def test_token_search_returns_same_hits_with_family_absent(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    monkeypatch.setenv("HERMES_FTS_TRIGRAM", "true")
    _seed(db_path)

    queries = ["alpha", "actoring", "웅기가"]
    db = SessionDB(db_path=db_path)
    try:
        with_family = {q: {row["id"] for row in db.search_messages(q)} for q in queries}
    finally:
        db.close()
    assert all(with_family[q] for q in queries), "every query must hit at least one row"

    monkeypatch.setenv("HERMES_FTS_TRIGRAM", "false")
    db = SessionDB(db_path=db_path)
    try:
        assert db._trigram_available is False
        without_family = {q: {row["id"] for row in db.search_messages(q)} for q in queries}
    finally:
        db.close()

    assert without_family == with_family
    assert _family_objects(db_path) == set()


@pytest.mark.parametrize("raw", ["0", "false", "off", "no", " FALSE "])
def test_gate_reads_false_spellings(monkeypatch, raw):
    monkeypatch.setenv("HERMES_FTS_TRIGRAM", raw)
    assert trigram_fts_config_enabled() is False


def test_gate_defaults_on_in_config_defaults():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert (DEFAULT_CONFIG.get("sessions") or {}).get("fts_trigram") is True
