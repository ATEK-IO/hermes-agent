"""ATE-21375 — one chat-completion attempt gets its own wall-clock deadline.

The streaming ``read`` timeout is a *byte-read* timeout (1800 s by default, and the
Paperclip run banner allows ``timeout=20000s``), and the stale detector only fires
when NO bytes arrive at all. A transport that accepts the request and then stalls —
or dribbles SSE keepalive comments, which rearm the read timeout on every byte —
therefore spends the whole run budget in one call. ATE-21346 measured exactly that:
one run logged ``Operation interrupted: waiting for model response (2454.1s
elapsed)`` and fourteen runs died together at 2026-11-08T19:28:22Z.

These tests pin the fix: each attempt is cut at
``HERMES_REQUEST_DEADLINE_SECONDS`` (600 s by default), the cut is reported as
``RequestDeadlineExceeded``, and the existing ``HERMES_STREAM_RETRIES`` retry loop
opens a fresh attempt instead of blocking on the byte-read timeout.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import run_agent
from agent import chat_completion_helpers as helpers
from agent.request_deadline import DEFAULT_REQUEST_DEADLINE_SECONDS, RequestDeadlineExceeded


def _agent(base_url="http://127.0.0.1:1/v1"):
    return run_agent.AIAgent(
        api_key="test-key", base_url=base_url, model="m", provider="custom",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )


def _call(agent):
    return helpers._StreamingCall(
        agent, {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, None)


_DALTA = (
    b'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"m",'
    b'"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}\n\n'
)


class _StalledWire:
    """A transport that accepts the request and then never produces a chunk.

    ``mode="silent"``: 200 + SSE headers, then no bytes at all (the classic
    stalled stream). ``mode="drip"``: one real content delta every 0.2 s, never a
    finish_reason and never ``[DONE]`` — a stream that keeps the byte-read timeout
    and the stale detector happy forever, so only a wall-clock deadline can end it.
    """

    def __init__(self, mode="silent", ping_seconds=0.2, hold_seconds=30.0):
        self.mode = mode
        self.ping_seconds = ping_seconds
        self.hold_seconds = hold_seconds
        self.completions = []
        self.writes = 0
        self.release = threading.Event()
        hits = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                hits.completions.append(json.loads(self.rfile.read(n) or b"{}"))
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                if hits.mode == "silent":
                    # Hold the body open with no bytes: the client parks in its
                    # body read. Bounded wait so teardown can never hang.
                    hits.release.wait(timeout=hits.hold_seconds)
                    return
                stop_at = time.time() + hits.hold_seconds
                while time.time() < stop_at and not hits.release.is_set():
                    try:
                        self.wfile.write(_DALTA if hits.mode == "drip" else b": keepalive\n\n")
                        self.wfile.flush()
                        hits.writes += 1
                    except OSError:
                        return  # the client aborted us (expected)
                    hits.release.wait(timeout=hits.ping_seconds)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.05), daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def silent_wire():
    wire = _StalledWire("silent")
    yield wire
    wire.close()


@pytest.fixture
def drip_wire():
    wire = _StalledWire("drip", hold_seconds=10.0)
    yield wire
    wire.close()


def test_silent_transport_is_cut_at_the_request_deadline_and_retried(silent_wire, monkeypatch, caplog):
    """A transport that yields nothing must hit the call deadline and retry.

    It must NOT sit on the byte-read timeout (20 s here, 20000 s in the run
    banner): with a 1 s deadline and 3 attempts the whole call is seconds.
    """
    monkeypatch.setenv("HERMES_REQUEST_DEADLINE_SECONDS", "1")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    monkeypatch.setenv("HERMES_STREAM_READ_TIMEOUT", "20")
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "999")  # the stale detector must not be the rescue
    caplog.set_level("WARNING")

    agent = _agent(silent_wire.base_url)
    started = time.time()
    with pytest.raises(RequestDeadlineExceeded):
        _call(agent).run()
    elapsed = time.time() - started

    assert len(silent_wire.completions) >= 3, (
        f"expected at least 3 attempts (1 + HERMES_STREAM_RETRIES=2), got {len(silent_wire.completions)}")
    assert len(silent_wire.completions) <= 6, (
        f"retry loop ran away: {len(silent_wire.completions)} requests for a 3-attempt budget")
    assert elapsed < 20.0, (
        f"one stalled model call blocked {elapsed:.1f}s; the read timeout was 20s and the "
        "deadline was 1s — the attempt did not get its own deadline")
    assert any("request deadline" in record.getMessage().lower() for record in caplog.records), (
        "no log line named the request deadline; the kill must say what happened")


def test_delta_dribble_stream_still_hits_the_wall_clock_deadline(drip_wire, monkeypatch, caplog):
    """A stream that keeps delivering tokens is not stale — but it is not bounded.

    Every delta rearms the byte-read timeout and the stale clock, so the existing
    detectors never fire; only the per-attempt wall-clock deadline can end this
    attempt. Tokens already reached the display, so the cut ends in the partial
    stub (no duplicate preamble) rather than an exception — that is what proves a
    later attempt was not needed and the run keeps its remaining budget.
    """
    monkeypatch.setenv("HERMES_REQUEST_DEADLINE_SECONDS", "1")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    monkeypatch.setenv("HERMES_STREAM_READ_TIMEOUT", "20")
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "999")
    caplog.set_level("WARNING")

    agent = _agent(drip_wire.base_url)
    started = time.time()
    with pytest.raises(RequestDeadlineExceeded):
        _call(agent).run()
    elapsed = time.time() - started

    assert len([b for b in drip_wire.completions if "messages" in b]) == 1
    assert drip_wire.writes >= 3, (
        f"the wire only wrote {drip_wire.writes} events: this test must prove the deadline fires "
        "while bytes are still arriving, not because the stream went silent")
    assert elapsed < 6.0, (
        f"a delta-dribbling stream held the attempt for {elapsed:.1f}s; the deadline was 1s")
    assert any("request deadline" in record.getMessage().lower() for record in caplog.records), (
        "no log line named the request deadline")


def test_deadline_resolution_defaults_to_600s_and_can_be_disabled(monkeypatch):
    from agent.request_deadline import resolve_request_deadline_seconds

    monkeypatch.delenv("HERMES_REQUEST_DEADLINE_SECONDS", raising=False)
    assert resolve_request_deadline_seconds(None) == DEFAULT_REQUEST_DEADLINE_SECONDS == 600.0

    monkeypatch.setenv("HERMES_REQUEST_DEADLINE_SECONDS", "42")
    assert resolve_request_deadline_seconds(None) == 42.0

    monkeypatch.setenv("HERMES_REQUEST_DEADLINE_SECONDS", "0")
    assert resolve_request_deadline_seconds(None) == float("inf"), "0 disables the deadline"

    monkeypatch.delenv("HERMES_REQUEST_DEADLINE_SECONDS", raising=False)
    assert resolve_request_deadline_seconds(type("A", (), {"request_deadline_seconds": 5.0})()) == 5.0
