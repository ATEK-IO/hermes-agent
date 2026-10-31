"""ATE-15509: an interrupt that lands on ``BEGIN IMMEDIATE`` poisons the writer.

Production evidence (T15, 2026-10-31 12:58:21Z, session ``20261031_125611_390fe4``,
``~/.hermes/logs/agent.log``): a soft interrupt (a new message arriving mid-run) was
delivered as ``BEGIN IMMEDIATE`` returned inside ``SessionDB._execute_write``.
SQLite had already opened the transaction at that point, but the statement sat
*outside* the ``try`` that rolls back, so the writer connection stayed inside an open
transaction. Every later write on that connection — starting with the turn-lease
release in the teardown path — then died with

    sqlite3.OperationalError: cannot start a transaction within a transaction

which masked the interrupt and dropped the transcript batch. The same window exists
in ``_enter_fts_fail_open`` (its own ``BEGIN IMMEDIATE``).

The two tests below simulate the only thing that matters about signal delivery: the
statement completes at the SQLite level and the ``KeyboardInterrupt`` is raised
immediately after, before the callback's ``try`` block is entered.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB


class _InterruptOnBegin:
    """Writer connection whose ``BEGIN`` completes, then raises ``KeyboardInterrupt``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.armed = True

    def execute(self, sql, *args):  # noqa: ANN001, ANN201 - sqlite wrapper
        result = self._conn.execute(sql, *args)
        if self.armed and sql.lstrip().upper().startswith("BEGIN"):
            self.armed = False
            raise KeyboardInterrupt
        return result

    def __getattr__(self, name):  # noqa: ANN001, ANN204 - transparent proxy
        return getattr(self._conn, name)


def _db(tmp_path: Path) -> SessionDB:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_meta("keep", "1")
    return db


def test_interrupted_begin_leaves_no_open_transaction(tmp_path: Path) -> None:
    db = _db(tmp_path)
    real = db._conn
    db._conn = _InterruptOnBegin(real)
    try:
        with pytest.raises(KeyboardInterrupt):
            db._execute_write(lambda conn: None)
        # The transaction BEGIN opened must not survive the interrupt.
        assert not real.in_transaction, (
            "BEGIN IMMEDIATE committed the transaction at the SQLite level and the "
            "interrupt was delivered after it returned; _execute_write must roll "
            "that transaction back before propagating, or the connection is poisoned"
        )
    finally:
        db._conn = real
        db.close()


def test_write_after_interrupted_begin_still_works(tmp_path: Path) -> None:
    db = _db(tmp_path)
    real = db._conn
    db._conn = _InterruptOnBegin(real)
    try:
        with pytest.raises(KeyboardInterrupt):
            db._execute_write(lambda conn: None)
    finally:
        db._conn = real
    try:
        # The next write on the same connection — in production the turn-lease
        # release that runs in a finally — must not fail on the leftover BEGIN.
        db._execute_write(lambda conn: conn.execute("UPDATE state_meta SET value='2' WHERE key='keep'"))
        assert db.get_meta("keep") == "2"
    finally:
        db.close()
