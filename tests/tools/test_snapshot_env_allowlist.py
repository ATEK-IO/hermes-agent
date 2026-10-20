"""ATE-7650 / ATE-7625 R1: the run-snapshot writer keeps an ALLOW-LIST, never a whole env dump.

Measured on 2026-10-20: 25 ``hermes-snap-*.sh`` files held 26 literal ``glpat-`` values
(3 distinct). Every one of them came from this writer, because it dumped ``export -p`` and
only subtracted a handful of per-session names.

The authority for the NAME LIST is the fleet scrubber ``tools/_lib/redact.mjs``
(``SNAPSHOT_ENV_ALLOW`` / ``SNAPSHOT_ENV_ALLOW_PREFIXES``); the runtime is the second
enforcement point of the same rule, so this suite asserts that the runtime list is a strict
SUBSET of it whenever that file is readable. That parity check is how the two points stay one
rule set without the runtime taking a dependency on a path outside its own repo.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from tools.environments.base_session_env import (
    SNAPSHOT_ENV_ALLOW,
    SNAPSHOT_ENV_ALLOW_PREFIXES,
    _SNAPSHOT_EXCLUDED_ENV_REGEX,
    _export_dump_excluding_session_vars,
    _snapshot_allowlist_emitter,
)


def _dump(env: dict[str, str], excluded=()) -> str:
    """Run the real bootstrap snippet in bash and return the snapshot it wrote."""
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "hermes-snap-test.sh")
        script = _export_dump_excluding_session_vars(target, excluded)
        proc = subprocess.run(
            ["bash", "-c", script],
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", **env},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        with open(target, "r", encoding="utf-8") as fh:
            return fh.read()


# ---------------------------------------------------------------------------
# Unit: the writer no longer emits the whole environment.
# ---------------------------------------------------------------------------

def test_dump_does_not_call_export_p():
    script = _export_dump_excluding_session_vars("/tmp/x")
    assert " export -p" not in script


def test_allowlist_has_no_credential_shaped_name():
    credential = re.compile(
        r"(^|_)(TOKEN|TOKENS|SECRET|SECRETS|PASS|PASSWD|PASSWORD|PASSWORDS|PASSPHRASE|PWD|"
        r"KEY|KEYS|APIKEY|CREDENTIAL|CREDENTIALS|AUTH|PRIVATE|COOKIE|SESSION_KEY)(_|$)"
    )
    # Paths, not passwords. PWD/OLDPWD are the shell's own directories; SSH_AUTH_SOCK is the
    # ssh-agent socket path (ATE-7783) - it lives in the EXACT arm precisely because its name
    # trips this shape and the exact arm applies no shape test, so nothing else is carved out.
    path_shaped = {"PWD", "OLDPWD", "SSH_AUTH_SOCK"}
    for name in SNAPSHOT_ENV_ALLOW:
        if name in path_shaped:
            continue
        assert not credential.search(name), f"{name} reads as a credential and must not persist"


# ---------------------------------------------------------------------------
# Integration: a real snapshot of a really dirty environment.
# ---------------------------------------------------------------------------

def test_only_allowlisted_names_reach_the_snapshot():
    out = _dump(
        {
            "HOME": "/root",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "XDG_DATA_HOME": "/root/.local/share",
            "XDG_TOKEN": "t0ken-value",
            "GITLAB_TOKEN": "glpat-abcdefghijklmnopqrst",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG",
            "MY_APP_FLAG": "1",
            "GOPATH": "/root/go",
        }
    )
    assert "declare -x PATH=" in out
    assert "declare -x LANG=" in out
    assert "declare -x LC_ALL=" in out
    assert "declare -x XDG_DATA_HOME=" in out
    for leaked in ("GITLAB_TOKEN", "AWS_SECRET_ACCESS_KEY", "MY_APP_FLAG", "GOPATH", "XDG_TOKEN"):
        assert leaked not in out, f"{leaked} must not persist into the snapshot"


def test_consumer_names_are_emitted_and_credential_family_is_not():
    """ATE-7783 / M1: the subset check cannot see an OMISSION, only a surplus.

    A real ``export -p`` on T15 measured 124 lines; the merged emitter kept 27. The three
    families below are what every later command in a resumed session needs: the board URL,
    the ssh-agent socket for a push, and the terminal tool's own settings.
    """
    out = _dump(
        {
            "HOME": "/root",
            "PAPERCLIP_API_URL": "http://paperclip.atek.io:3100/api",
            "PAPERCLIP_RUN_ID": "run-1",
            "SSH_AUTH_SOCK": "/run/user/1000/ssh-agent.socket",
            "TERMINAL_PERSISTENT_SHELL": "/bin/bash",
            # Credential-shaped members of the same families must still be dropped.
            "PAPERCLIP_API_KEY": "not-a-real-key",
            "TERMINAL_SESSION_KEY": "not-a-real-key",
        }
    )
    for present in ("PAPERCLIP_API_URL", "SSH_AUTH_SOCK", "TERMINAL_PERSISTENT_SHELL"):
        assert f"declare -x {present}=" in out, f"{present} is needed by a resumed session"
    # F1 on PR #9: per-run identity is NOT a consumer name — a later run must never
    # inherit the run that happened to write the snapshot.
    for absent in ("PAPERCLIP_RUN_ID", "PAPERCLIP_API_KEY", "TERMINAL_SESSION_KEY", "not-a-real-key"):
        assert absent not in out, f"{absent} must not persist into the snapshot"


def test_unexported_shell_variable_is_not_dumped():
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "snap.sh")
        script = "set -a; TMPDIR=/tmp; set +a; " + _export_dump_excluding_session_vars(target)
        subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        body = Path(target).read_text(encoding="utf-8")
    assert "declare -x TMPDIR=" in body


def test_multiline_value_stays_quoted_and_sources_cleanly():
    """#71296: a value holding a newline must stay inside its quotes, not become a command."""
    evil = "x\n/tmp/ate7625-pwned"
    out = _dump({"TERM": evil})
    assert "/tmp/ate7625-pwned" in out
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, "snap.sh")
        Path(snap).write_text(out, encoding="utf-8")
        proc = subprocess.run(
            ["bash", "-c", f'source "{snap}"; printf %s "$TERM"'],
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == evil
    assert not os.path.exists("/tmp/ate7625-pwned")


# ---------------------------------------------------------------------------
# Parity: this list must stay a subset of the fleet's one rule set.
# ---------------------------------------------------------------------------

def _find_redact_mjs() -> Path | None:
    candidates = [
        os.environ.get("HERMES_REDACT_MJS"),
        str(Path.home() / "work" / "paperclip" / "tools" / "_lib" / "redact.mjs"),
        str(Path.home() / "agent-roots" / "tools" / "_lib" / "redact.mjs"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def test_runtime_allowlist_is_a_subset_of_the_fleet_rule():
    redact = _find_redact_mjs()
    if redact is None:
        pytest.skip("fleet scrubber tools/_lib/redact.mjs not present on this host")
    source = Path(redact).read_text(encoding="utf-8")

    names_block = source.split("SNAPSHOT_ENV_ALLOW = new Set([", 1)[1].split("]);", 1)[0]
    fleet_names = set(re.findall(r'"([A-Za-z0-9_]+)"', names_block))
    prefix_block = source.split("SNAPSHOT_ENV_ALLOW_PREFIXES = [", 1)[1].split("];", 1)[0]
    fleet_prefixes = tuple(re.findall(r'"([A-Za-z0-9_]+)"', prefix_block))

    assert SNAPSHOT_ENV_ALLOW <= fleet_names, (
        f"runtime allows names the fleet rule does not: {sorted(SNAPSHOT_ENV_ALLOW - fleet_names)}"
    )
    for prefix in SNAPSHOT_ENV_ALLOW_PREFIXES:
        assert prefix in fleet_prefixes, f"runtime prefix {prefix} is not in the fleet rule"


# ---------------------------------------------------------------------------
# F1 (PR #9 review): the PAPERCLIP_ prefix arm must not persist per-run identity.
# ---------------------------------------------------------------------------

def test_per_run_identity_names_are_in_the_exclusion_contract():
    """Python-side contract: the three identity names are excluded by name, not by value."""
    contract = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    for name in ("PAPERCLIP_RUN_ID", "PAPERCLIP_TASK_ID", "PAPERCLIP_WAKE_REASON"):
        assert contract.match(f"declare -x {name}='x'"), f"{name} must be excluded by name"
    # The non-identity member of the family stays in the allow-list path.
    assert not contract.match("declare -x PAPERCLIP_API_URL='http://x'")


def test_snapshot_written_by_run_a_does_not_clobber_run_b_identity():
    """F1 reproduction: snapshot from run A sourced in run B's shell.

    Before the fix the snapshot carried ``PAPERCLIP_RUN_ID='run-A'``; sourcing it in a
    command whose process env already said ``run-B`` overwrote the fresher value with the
    foreign run's identity. The snapshot must carry the board URL and no identity at all,
    so ``source`` leaves the per-command value intact.
    """
    snapshot = _dump(
        {
            "PAPERCLIP_API_URL": "http://paperclip.atek.io:3100/api",
            "PAPERCLIP_RUN_ID": "run-A",
            "PAPERCLIP_TASK_ID": "task-A",
            "PAPERCLIP_WAKE_REASON": "heartbeat-A",
        }
    )
    for name in ("PAPERCLIP_RUN_ID", "PAPERCLIP_TASK_ID", "PAPERCLIP_WAKE_REASON"):
        assert f"declare -x {name}=" not in snapshot, f"{name} from the writing run leaked into the snapshot"
    assert "declare -x PAPERCLIP_API_URL=" in snapshot

    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.sh")
        Path(snap_path).write_text(snapshot, encoding="utf-8")
        probe = (
            f'source "{snap_path}" >/dev/null 2>&1; '
            'printf "%s\\n" "$PAPERCLIP_RUN_ID" "$PAPERCLIP_TASK_ID" '
            '"$PAPERCLIP_WAKE_REASON" "$PAPERCLIP_API_URL"'
        )
        proc = subprocess.run(
            ["bash", "-c", probe],
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "PAPERCLIP_RUN_ID": "run-B",
                "PAPERCLIP_TASK_ID": "task-B",
                "PAPERCLIP_WAKE_REASON": "heartbeat-B",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr

    lines = proc.stdout.splitlines()
    assert lines[0] == "run-B", f"snapshot clobbered the live run id with {lines[0]!r}"
    assert lines[1] == "task-B", f"snapshot clobbered the live task id with {lines[1]!r}"
    assert lines[2] == "heartbeat-B", f"snapshot clobbered the live wake reason with {lines[2]!r}"
    assert lines[3] == "http://paperclip.atek.io:3100/api", "board URL must still arrive from the snapshot"

