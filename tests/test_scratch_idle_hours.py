"""Scratch idle retention is a deployment knob, and a live process always wins over it.

ATE-7299 (D1 of ATE-7270): the window comes from ``HERMES_SCRATCH_MAX_IDLE_HOURS`` (default 24 h,
clamped to 1-168 h) and an entry that holds a same-user process is never doomed, so shortening
the window cannot kill a run that is alive but quiet.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hermes_constants as constants
import hermes_constants_scratch as scratch


def _age(path: Path, hours: float) -> None:
    """Backdate *path* and everything in it, deepest first, so nothing is "recently written"."""
    stamp = time.time() - hours * 3600
    for target in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        os.utime(target, (stamp, stamp))
    os.utime(path, (stamp, stamp))


def _sleeper(cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], cwd=str(cwd), stdin=subprocess.DEVNULL,
    )


def _prune_log(home: Path) -> str:
    log = home / "logs" / "scratch-prune.log"
    return log.read_text(encoding="utf-8-sig") if log.exists() else ""


def test_idle_entry_without_a_process_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The retention rule still deletes a tree that nothing is using."""
    monkeypatch.delenv(constants.SCRATCH_MAX_IDLE_HOURS_ENV, raising=False)
    root = constants.get_scratch_dir(tmp_path, prune=False)
    lane = root / "lane"
    lane.mkdir()
    (lane / "out.md").write_text("done", encoding="utf-8")
    _age(lane, 30)

    assert constants.prune_scratch_dir(root) == 1
    assert not lane.exists()


def test_idle_entry_holding_a_live_process_is_kept_and_never_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Busy is decided by the cwd, and a busy entry is not even a candidate for the reap."""
    monkeypatch.delenv(constants.SCRATCH_MAX_IDLE_HOURS_ENV, raising=False)
    root = constants.get_scratch_dir(tmp_path, prune=False)
    lane = root / "lane" / "build"
    lane.mkdir(parents=True)
    _age(root / "lane", 30)
    worker = _sleeper(lane)
    try:
        assert constants.prune_scratch_dir(root) == 0
        assert worker.poll() is None
        assert lane.exists()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)

    assert "sent TERM" not in _prune_log(tmp_path)


def test_busy_entries_names_the_top_level_entry_that_holds_a_process(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    (root / "busy" / "deep").mkdir(parents=True)
    (root / "quiet").mkdir()
    worker = _sleeper(root / "busy" / "deep")
    try:
        assert {os.path.realpath(p) for p in scratch.busy_entries(root)} == {
            os.path.realpath(root / "busy"),
        }
    finally:
        worker.kill()
        worker.wait(timeout=10)


def test_orphan_with_a_deleted_cwd_is_still_reaped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard spares live entries, not orphans: a cwd already removed still gets a TERM."""
    monkeypatch.delenv(constants.SCRATCH_MAX_IDLE_HOURS_ENV, raising=False)
    root = constants.get_scratch_dir(tmp_path, prune=False)
    orphan = root / "orphan" / "deep"
    orphan.mkdir(parents=True)
    worker = _sleeper(orphan)
    try:
        assert worker.poll() is None
        shutil.rmtree(root / "orphan")
        assert constants.prune_scratch_dir(root) == 0
        assert worker.wait(timeout=10) is not None
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)

    assert f"sent TERM pid={worker.pid} " in _prune_log(tmp_path)


def test_env_value_sets_the_effective_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """8 h means 8 h: a 10 h-old tree goes, a 6 h-old tree stays."""
    monkeypatch.setenv(constants.SCRATCH_MAX_IDLE_HOURS_ENV, "8")
    assert constants.scratch_max_idle_hours() == 8.0
    root = constants.get_scratch_dir(tmp_path, prune=False)
    for name, hours in (("stale", 10), ("fresh", 6)):
        entry = root / name
        entry.mkdir()
        (entry / "out.md").write_text(name, encoding="utf-8")
        _age(entry, hours)

    assert constants.prune_scratch_dir(root) == 1
    assert not (root / "stale").exists()
    assert (root / "fresh").exists()


def test_explicit_argument_wins_over_the_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit caller keeps its own number, so the env change is a default, not an override."""
    monkeypatch.setenv(constants.SCRATCH_MAX_IDLE_HOURS_ENV, "8")
    root = constants.get_scratch_dir(tmp_path, prune=False)
    entry = root / "lane"
    entry.mkdir()
    _age(entry, 10)

    assert constants.prune_scratch_dir(root, 24) == 0
    assert entry.exists()


@pytest.mark.parametrize(("raw", "expected"), [
    ("abc", 24.0),
    ("", 24.0),
    ("nan", 24.0),
    ("0.1", 1.0),
    ("999", 168.0),
    ("8.5", 8.5),
])
def test_env_value_is_clamped_or_falls_back(
    raw: str, expected: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(constants.SCRATCH_MAX_IDLE_HOURS_ENV, raw)
    assert constants.scratch_max_idle_hours() == expected


def test_default_window_is_still_24_hours(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(constants.SCRATCH_MAX_IDLE_HOURS_ENV, raising=False)
    assert constants.SCRATCH_MAX_IDLE_HOURS == 24.0
    assert constants.scratch_max_idle_hours() == 24.0
