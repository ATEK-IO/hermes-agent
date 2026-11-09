"""Per-request wall-clock deadline for one chat-completion attempt (ATE-21375).

Why this exists
---------------
The streaming ``read`` timeout is a **byte-read** timeout: it is rearmed by every
byte that arrives, and the run banner's transport budget (``timeout=20000``) is
far larger than any useful model call. The stale detector is a **no-bytes**
timer: it only fires when the stream is completely silent. Between the two there
was no bound on a *single* call, so one stream that stalls while dribbling SSE
keepalive comments — or one whose read timeout is the whole run budget — spends
the entire run in that call.

Measured, ATE-21346: on 2026-11-08T19:28:22Z fourteen Paperclip runs ended in
the same two seconds and one logged ``Operation interrupted: waiting for model
response (2454.1s elapsed)``. The gateway was healthy throughout (the same model
answered another caller in 2.3-7.4 s), so nothing on the server side was down:
the call had no deadline of its own.

What this module adds
---------------------
``resolve_request_deadline_seconds`` resolves the wall-clock budget for **one
attempt** — ``HERMES_REQUEST_DEADLINE_SECONDS`` (default 600 s) or an
``agent.request_deadline_seconds`` attribute. ``0`` disables it.

``_StreamingCall`` arms a monotonic clock at the start of every attempt
and the stream monitor kills the attempt's socket when it expires, exactly like
the stale kill. The existing ``HERMES_STREAM_RETRIES`` retry loop then opens a
fresh attempt, and the cut surfaces as :class:`RequestDeadlineExceeded` so the
log, the status line and the run record all name the deadline rather than
``ReadTimeout``/``unknown``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

#: Default wall-clock budget for one chat-completion attempt, in seconds.
DEFAULT_REQUEST_DEADLINE_SECONDS = 600.0

#: Environment override; ``0`` (or a negative value) disables the deadline.
ENV_VAR = "HERMES_REQUEST_DEADLINE_SECONDS"


class RequestDeadlineExceeded(Exception):
    """One chat-completion attempt outlived its own wall-clock deadline.

    Transient by design: the retry loop retries it like a dropped stream, and
    only the last attempt's instance becomes the turn's error.
    """

    def __init__(self, elapsed: float, deadline: float, attempt: Optional[int] = None,
                 max_attempts: Optional[int] = None):
        self.elapsed = float(elapsed)
        self.deadline = float(deadline)
        self.attempt = attempt
        self.max_attempts = max_attempts
        where = (f"attempt {attempt}/{max_attempts}"
                 if attempt is not None and max_attempts is not None else "one attempt")
        super().__init__(
            f"model call exceeded its request deadline of {self.deadline:.0f}s "
            f"({where}, elapsed {self.elapsed:.0f}s)")


def _coerce_seconds(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return value


def resolve_request_deadline_seconds(agent: Any = None) -> float:
    """Wall-clock budget for one attempt. ``inf`` means "no deadline".

    Order: ``HERMES_REQUEST_DEADLINE_SECONDS`` env, then
    ``agent.request_deadline_seconds``, then :data:`DEFAULT_REQUEST_DEADLINE_SECONDS`.
    A non-positive value disables the deadline (``inf``) rather than making every
    attempt fail instantly.
    """
    value = None
    raw_env = os.environ.get(ENV_VAR)
    if raw_env is not None and raw_env.strip():
        value = _coerce_seconds(raw_env)
    if value is None and agent is not None:
        value = _coerce_seconds(getattr(agent, "request_deadline_seconds", None))
    if value is None:
        value = DEFAULT_REQUEST_DEADLINE_SECONDS
    return value if value > 0 else float("inf")
