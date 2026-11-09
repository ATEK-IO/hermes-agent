"""ATE-21375 — the run child names the sender of its SIGINT/SIGTERM.

The bash shim records ``source: unknown`` for a signalled run child because the
sender is gone by the time it looks (ATE-21346: 14 runs, exit 130, same two
seconds). The child itself receives ``si_pid``/``si_code`` from the kernel, so
these tests drive a real child process, signal it from a known pid, and assert
that (a) the sender is named and (b) the child's own handler still runs — the
tap must never change interrupt behaviour, only the record.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

CHILD = textwrap.dedent(
    """
    import logging, signal, sys, time
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                        format="LOG %(levelname)s %(name)s %(message)s")
    from agent.signal_source import install_signal_source_capture

    def handler(signum, frame):
        print("handler-ran", flush=True)
        time.sleep(1.0)          # let the watcher thread write the record
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handler)
    ok = install_signal_source_capture((signal.SIGINT,))
    print(f"installed={ok}", flush=True)
    print("ready", flush=True)
    time.sleep(30)
    print("never-interrupted", flush=True)
    """
)


def _run_child(tmp_path, hermes_signal_source: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, HERMES_SIGNAL_SOURCE=hermes_signal_source)
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD], cwd=str(REPO_ROOT), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        lines = []
        deadline = time.time() + 20
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            if line.strip() == "ready":
                break
        assert any(line.strip() == "ready" for line in lines), (
            f"child never became ready: {''.join(lines)}")
        os.kill(proc.pid, signal.SIGINT)
        rest, _ = proc.communicate(timeout=25)
    finally:
        if proc.poll() is None:
            proc.kill()
    return subprocess.CompletedProcess(proc.args, proc.returncode, "".join(lines) + (rest or ""))


@pytest.mark.skipif(os.name != "posix" or not hasattr(signal, "SIGINT"), reason="POSIX signals only")
def test_sigint_names_the_sender_and_still_runs_the_real_handler(tmp_path):
    result = _run_child(tmp_path, "1")
    output = result.stdout

    assert "installed=True" in output, output
    # The interrupt path is unchanged: the child's own handler ran and exited.
    assert "handler-ran" in output, output
    assert "never-interrupted" not in output, output
    assert result.returncode == 0, output
    # The record names the sender: this test process sent it with kill() => SI_USER.
    assert f"si_pid={os.getpid()}" in output, output
    assert "si_code=0" in output, output
    assert "SI_USER" in output, output


@pytest.mark.skipif(os.name != "posix" or not hasattr(signal, "SIGINT"), reason="POSIX signals only")
def test_capture_can_be_disabled_and_the_process_is_untouched(tmp_path):
    result = _run_child(tmp_path, "0")
    output = result.stdout

    assert "installed=False" in output, output
    assert "handler-ran" in output, output
    assert result.returncode == 0, output
    assert "si_pid=" not in output, output
