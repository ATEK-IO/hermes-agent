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
    for name in SNAPSHOT_ENV_ALLOW:
        if name in {"PWD", "OLDPWD"}:
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
