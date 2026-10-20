"""Cleanup needs a complete idle scan before deleting a scratch tree."""

import logging
import os
import sys
import time

import psutil
import pytest

import hermes_constants_scratch as scratch


@pytest.mark.parametrize("failed_probe", ["root-stat", "directory-scan", "child-stat"])
def test_unreadable_subtree_is_preserved_until_it_can_be_scanned(tmp_path, monkeypatch, failed_probe):
    root = tmp_path / "scratch"
    entry = root / "lane"
    entry.mkdir(parents=True)
    result = entry / "result.txt"
    result.write_text("keep until activity can be checked", encoding="utf-8")
    old = time.time() - 48 * 3600
    os.utime(result, (old, old))
    os.utime(entry, (old, old))
    # Process inspection is unrelated to determining whether the files are idle.
    monkeypatch.setattr(scratch, "reap_processes_rooted_in", lambda *_: 0)
    real_lstat, real_scandir = os.lstat, os.scandir
    with monkeypatch.context() as probe:
        if failed_probe == "root-stat":
            def lstat(path, *args, **kwargs):
                if path == entry:
                    raise PermissionError("cannot inspect entry")
                return real_lstat(path, *args, **kwargs)
            probe.setattr(scratch.os, "lstat", lstat)
        else:
            def scandir(path):
                if path == str(entry):
                    if failed_probe == "directory-scan":
                        raise PermissionError("cannot inspect subtree")

                    class UnreadableChild:
                        name = "result.txt"

                        def stat(self, **_kwargs):
                            raise OSError("cannot inspect child activity")

                        def is_dir(self, **_kwargs):
                            return False

                    class Scan:
                        def __enter__(self):
                            return iter([UnreadableChild()])

                        def __exit__(self, *_args):
                            pass

                    return Scan()
                return real_scandir(path)
            probe.setattr(scratch.os, "scandir", scandir)
        assert scratch.prune_idle_entries(root, 24, frozenset()) == 0
        assert result.read_text(encoding="utf-8") == "keep until activity can be checked"
    # Once observation succeeds, the same genuinely idle tree is reclaimable.
    assert scratch.prune_idle_entries(root, 24, frozenset()) == 1
    assert not entry.exists()


class _UnreadableProcess:
    """A process psutil can list but not inspect: the "API drift / host guard" shape of failure."""

    pid = 424242

    def uids(self):
        raise AttributeError("psutil.Process.uids is not available on this build")

    def cwd(self):  # never reached: the uid read already failed
        raise AssertionError("cwd must not be consulted after an unreadable uid")


class _ElsewhereProcess:
    pid = 424243

    class _Uids:
        real = os.getuid() if hasattr(os, "getuid") else 0

    def uids(self):
        return self._Uids()

    def cwd(self):
        return os.path.abspath(os.sep)


def _idle_lane(tmp_path):
    root = tmp_path / "scratch"
    entry = root / "lane"
    entry.mkdir(parents=True)
    result = entry / "result.txt"
    result.write_text("a live run owns this", encoding="utf-8")
    old = time.time() - 48 * 3600
    for path in (entry, result):
        os.utime(path, (old, old))
    return root, entry, result


def test_an_unreadable_process_scan_keeps_every_entry(tmp_path, monkeypatch, caplog):
    """Fail closed: if the scan inspected no process, "not busy" is a guess, not a finding.

    The pass used to treat "I could not inspect this process" as "this entry is not busy", so
    psutil API drift made the busy set silently empty and the idle scan deleted trees under live
    processes with nothing but a ``removed`` line in the log (ATE-7332 M1).
    """
    root, entry, result = _idle_lane(tmp_path)
    unreadable = [_UnreadableProcess(), _UnreadableProcess()]
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **k: iter(unreadable))
    with caplog.at_level(logging.WARNING):
        assert scratch.prune_idle_entries(root, 24, frozenset()) == 0
    assert entry.is_dir() and result.read_text(encoding="utf-8") == "a live run owns this"
    assert "keeping all entries" in caplog.text


def test_one_unreadable_process_does_not_fail_the_pass(tmp_path, monkeypatch, caplog):
    """The distinction the fix rests on: one gone/denied process is skipped, not escalated."""
    root, entry, _ = _idle_lane(tmp_path)
    monkeypatch.setattr(
        psutil, "process_iter",
        lambda *a, **k: iter([_UnreadableProcess(), _ElsewhereProcess()]),
    )
    monkeypatch.setattr(scratch, "reap_processes_rooted_in", lambda *_: 0)
    with caplog.at_level(logging.WARNING):
        assert scratch.prune_idle_entries(root, 24, frozenset()) == 1
    assert not entry.exists()
    assert "keeping all entries" not in caplog.text


def test_a_missing_psutil_keeps_every_entry(tmp_path, monkeypatch, caplog):
    """Pin the no-psutil fail-closed path (ATE-7332 m1): it works, and nothing else holds it.

    ``import psutil`` lives inside ``busy_entries``; if a later refactor moves it out, this test
    turns the guard back on rather than leaving the deletion unguarded and green.
    """
    root, entry, result = _idle_lane(tmp_path)
    monkeypatch.setitem(sys.modules, "psutil", None)
    with caplog.at_level(logging.WARNING):
        assert scratch.prune_idle_entries(root, 24, frozenset()) == 0
    assert entry.is_dir() and result.read_text(encoding="utf-8") == "a live run owns this"
    assert "keeping all entries" in caplog.text
