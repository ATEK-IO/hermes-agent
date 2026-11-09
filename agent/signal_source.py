"""Name the process that sent SIGINT/SIGTERM to a run child (ATE-21375).

On 2026-11-08T19:28:22Z fourteen Paperclip runs on r410 ended inside two
seconds. The run record said what could be seen — ``signal INT``,
``exit 130``, ``source: unknown``, ``role: child`` — and nothing named the
sender, so the incident could only be described, not attributed (ATE-21346).
The bash shim cannot see it: by the time the child is dead the sender is gone.
The Python child can: the kernel hands it ``siginfo_t`` on delivery
(``si_pid``, ``si_code``, ``si_uid``).

CPython's ``signal.signal`` installs the handler **without** ``SA_SIGINFO``, so a
normal Python handler never receives that structure. This module installs a
one-shot tap in front of the handler that is already installed:

1. the tap runs under ``SA_SIGINFO`` and copies the raw ``siginfo_t`` into a
   preallocated buffer, then writes one byte on a self-pipe;
2. it restores the previous action and re-raises the signal, so the run's
   existing handler (``_install_single_query_signal_handlers`` ->
   ``_interrupt_agent_for_signal``) runs exactly as before — the interrupt
   behaviour does not change, only the record gets the sender;
3. a watcher thread reads the byte, parses the buffer and logs ``si_code`` /
   ``si_pid`` / ``si_uid`` with the child's own pid and parent pid.

Everything the tap does is async-signal-safe in intent (``memmove``, ``write``,
``sigaction``, ``raise``), and installation is strictly best effort: an
unexpected struct layout, a platform without ``SA_SIGINFO``, a restricted
environment — any of them leaves the process exactly as it was. Set
``HERMES_SIGNAL_SOURCE=0`` to disable it.
"""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import threading

logger = logging.getLogger(__name__)

ENV_VAR = "HERMES_SIGNAL_SOURCE"

# glibc ``struct sigaction`` on x86_64/aarch64: sa_handler (8) + sa_mask (128),
# then sa_flags. Verified empirically at install time (see ``_verify_layout``),
# never assumed: a mismatch means we restore and do nothing.
_ACTION_SIZE = 256
_HANDLER_OFFSET = 0
_FLAGS_OFFSET = 136

_SA_SIGINFO = 0x00000004
_SA_RESTART = 0x10000000
_SA_RESTORER = 0x04000000  # glibc sets it itself; ignore it when comparing

# ``siginfo_t`` for the kill/rt unions (SIGINT/SIGTERM): signo, errno, code,
# then pid/uid. Offsets are the same on x86_64 and aarch64 Linux.
_SI_SIGNO = 0
_SI_CODE = 8
_SI_PID = 16
_SI_UID = 20
_SIGINFO_SIZE = 128

_installed = {"done": False, "signals": ()}
_state = threading.local()


def enabled() -> bool:
    raw = (os.environ.get(ENV_VAR) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def describe_siginfo(buf: bytes) -> dict:
    """Parse a captured ``siginfo_t`` into the fields worth logging."""
    if len(buf) < _SIGINFO_SIZE:
        return {}
    return {
        "si_signo": ctypes.c_int.from_buffer_copy(buf[_SI_SIGNO:_SI_SIGNO + 4]).value,
        "si_code": ctypes.c_int.from_buffer_copy(buf[_SI_CODE:_SI_CODE + 4]).value,
        "si_pid": ctypes.c_int.from_buffer_copy(buf[_SI_PID:_SI_PID + 4]).value,
        "si_uid": ctypes.c_uint32.from_buffer_copy(buf[_SI_UID:_SI_UID + 4]).value,
    }


def _code_name(code: int) -> str:
    """Kernel constants that actually appear on a run child's SIGINT."""
    return {
        0: "SI_USER(kill)",
        -1: "SI_QUEUE(sigqueue)",
        -2: "SI_TIMER",
        -3: "SI_MESGQ",
        -4: "SI_ASYNCIO",
        -6: "SI_TKILL(tgkill/pthread_kill)",
        128: "SI_KERNEL(terminal)",
    }.get(code, f"code:{code}")


def _verify_layout(libc, signum: int, ours) -> bool:
    """Read our own action back: same handler pointer, same flags, or no deal."""
    check = ctypes.create_string_buffer(_ACTION_SIZE)
    if libc.sigaction(signum, None, check) != 0:
        return False
    want_ptr = ctypes.c_void_p.from_buffer(ours).value
    got_ptr = ctypes.c_void_p.from_buffer(check).value
    want_flags = ctypes.c_uint32.from_buffer(ours, _FLAGS_OFFSET).value
    got_flags = ctypes.c_uint32.from_buffer(check, _FLAGS_OFFSET).value
    return (bool(want_ptr) and got_ptr == want_ptr
            and (got_flags & ~_SA_RESTORER) == (want_flags & ~_SA_RESTORER))


def install_signal_source_capture(signals_to_watch=None) -> bool:
    """Install the one-shot sender tap. True = installed, False = left alone.

    Call this **after** the run's own signal handlers are installed: the tap
    restores and re-raises into whatever action is in place when it fires.
    """
    if _installed["done"]:
        return True
    if not enabled():
        logger.debug("signal source capture disabled by %s", ENV_VAR)
        return False
    if not hasattr(signal, "SIGINT") or os.name != "posix":
        return False
    watch = tuple(signals_to_watch or (signal.SIGINT, signal.SIGTERM))

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.sigaction.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        libc.sigaction.restype = ctypes.c_int
        # ``raise`` is a Python keyword: reach the libc symbol by name.
        _raise = getattr(libc, "raise")
        _raise.argtypes = [ctypes.c_int]
        _raise.restype = ctypes.c_int
    except Exception:
        logger.debug("signal source capture unavailable: no usable libc", exc_info=True)
        return False

    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    capture = ctypes.create_string_buffer(_SIGINFO_SIZE)
    seen = {"signal": 0}
    actions: dict[int, object] = {}

    def _tap(signum, info_ptr, _ucontext):  # signal context: no allocation, no logging
        try:
            if info_ptr:
                ctypes.memmove(capture, info_ptr, _SIGINFO_SIZE)
            seen["signal"] = int(signum)
            os.write(write_fd, b"1")
        except Exception:
            pass
        # Chain: put the previous action back and re-raise. The signal is blocked
        # while a handler runs, so this one is delivered normally on return.
        try:
            previous = actions.get(int(signum))
            if previous is not None:
                libc.sigaction(int(signum), previous, None)
            _raise(int(signum))
        except Exception:
            pass

    callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(_tap)

    installed_any = False
    for signum in watch:
        try:
            previous = ctypes.create_string_buffer(_ACTION_SIZE)
            if libc.sigaction(signum, None, previous) != 0:
                continue
            ours = ctypes.create_string_buffer(_ACTION_SIZE)
            ctypes.c_void_p.from_buffer(ours).value = ctypes.cast(callback, ctypes.c_void_p).value
            # Keep every flag the previous action had (CPython installs with
            # SA_ONSTACK) and only add what the tap needs.
            _prev_flags = ctypes.c_uint32.from_buffer(previous, _FLAGS_OFFSET).value
            ctypes.c_uint32.from_buffer(ours, _FLAGS_OFFSET).value = (
                _prev_flags | _SA_SIGINFO | _SA_RESTART)
            if libc.sigaction(signum, ours, None) != 0:
                continue
            if not _verify_layout(libc, signum, ours):
                libc.sigaction(signum, previous, None)  # put it back, this platform differs
                logger.debug("signal source capture: struct sigaction layout differs, skipped (signum=%s)", signum)
                continue
            actions[int(signum)] = previous
            installed_any = True
        except Exception:
            logger.debug("signal source capture: install failed for signum=%s", signum, exc_info=True)

    if not installed_any:
        os.close(read_fd)
        os.close(write_fd)
        return False

    def _watch() -> None:
        while True:
            try:
                if not os.read(read_fd, 1):
                    return
            except OSError:
                return
            try:
                fields = describe_siginfo(bytes(capture.raw))
                logger.warning(
                    "Received %s from %s: si_signo=%s si_code=%s si_pid=%s si_uid=%s "
                    "(child pid=%s ppid=%s) — naming the sender before exit (ATE-21375)",
                    signal.Signals(seen["signal"]).name if seen["signal"] else "a signal",
                    _code_name(fields.get("si_code", 0)),
                    fields.get("si_signo"), fields.get("si_code"),
                    fields.get("si_pid"), fields.get("si_uid"), os.getpid(), os.getppid())
            except Exception:
                logger.debug("signal source capture: could not describe siginfo", exc_info=True)

    threading.Thread(target=_watch, name="signal-source-watch", daemon=True).start()
    _installed.update(done=True, signals=tuple(int(s) for s in watch))
    return True
