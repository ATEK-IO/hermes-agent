"""A keyless ``/models`` probe that a keyed gateway refused is remembered, not repeated.

Why this exists (ATE-18791): ``fetch_endpoint_model_metadata`` was called with ``api_key=""`` from the
Langfuse cost path on every traced generation. The gateway answered 401, the probe ``break``ed and
returned ``{}`` with nothing remembered, so once the 300 s memo expired the same keyless request fired
again — forever, and every one of them wrote an "unauthorized" line into the operator's access log.

Covers:
- a keyless 401/403 short-circuits later keyless probes for the same host:port
- a keyed caller is never suppressed and never records the entry (a real credential fault stays visible)
- the entry expires, is per host:port, and is disabled by a zero TTL
- an explicit api_key on the request still wins over the suppression
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GATEWAY = "http://gateway.example.com:4200/v1"
GATEWAY_KEY = "gateway.example.com:4200"


@pytest.fixture(autouse=True)
def _clear_caches():
    """Module-level caches must not leak between tests."""
    from agent import model_metadata

    def _clear():
        model_metadata._endpoint_blackhole_cache.clear()
        model_metadata._endpoint_unauth_probe_cache.clear()
        model_metadata._endpoint_probe_path_cache.clear()
        model_metadata._endpoint_model_metadata_cache.clear()
        model_metadata._endpoint_model_metadata_cache_time.clear()

    _clear()
    yield
    _clear()


def _install_stream(monkeypatch, status: int, calls: list):
    """Answer every probe with *status* and record each attempt (url, headers)."""
    from agent import model_metadata

    @contextmanager
    def stream(url, headers=None, **kwargs):
        calls.append((url, dict(headers or {})))
        yield httpx.Response(status, request=httpx.Request("GET", url), json={})

    monkeypatch.setattr(model_metadata.model_metadata_http, "stream", stream)


class TestCachePrimitive:
    def test_keyed_on_host_port_and_from_nowhere_else(self):
        from agent.model_metadata import _endpoint_unauth_probe_refused, _note_endpoint_unauth_probe_refused

        _note_endpoint_unauth_probe_refused(GATEWAY)
        assert _endpoint_unauth_probe_refused(GATEWAY) is True
        assert _endpoint_unauth_probe_refused("http://gateway.example.com:4200/api/v1") is True
        assert _endpoint_unauth_probe_refused("http://gateway.example.com:4201/v1") is False
        assert _endpoint_unauth_probe_refused("") is False

    def test_entry_expires_after_ttl(self):
        from agent import model_metadata

        model_metadata._note_endpoint_unauth_probe_refused(GATEWAY)
        stale = (
            model_metadata._endpoint_unauth_probe_cache[GATEWAY_KEY]
            - model_metadata._ENDPOINT_UNAUTH_PROBE_TTL_SECONDS
            - 1
        )
        model_metadata._endpoint_unauth_probe_cache[GATEWAY_KEY] = stale
        assert model_metadata._endpoint_unauth_probe_refused(GATEWAY) is False

    def test_ttl_zero_disables_suppression(self):
        from agent import model_metadata

        model_metadata._note_endpoint_unauth_probe_refused(GATEWAY)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(model_metadata, "_ENDPOINT_UNAUTH_PROBE_TTL_SECONDS", 0.0)
            assert model_metadata._endpoint_unauth_probe_refused(GATEWAY) is False

    def test_connect_timeout_cache_is_untouched(self):
        """The two verdicts share a shape but never each other's entries."""
        from agent import model_metadata

        model_metadata._note_endpoint_unauth_probe_refused(GATEWAY)
        assert model_metadata._endpoint_blackholed(GATEWAY) is False
        model_metadata._note_endpoint_blackholed(GATEWAY)
        model_metadata._endpoint_unauth_probe_cache.clear()
        assert model_metadata._endpoint_unauth_probe_refused(GATEWAY) is False
        assert model_metadata._endpoint_blackholed(GATEWAY) is True


class TestKeylessProbeSuppression:
    def test_second_keyless_probe_makes_no_request(self, monkeypatch):
        from agent.model_metadata import fetch_endpoint_model_metadata

        calls: list = []
        _install_stream(monkeypatch, 401, calls)

        assert fetch_endpoint_model_metadata(GATEWAY, api_key="", force_refresh=True) == {}
        assert len(calls) == 1
        assert "authorization" not in {k.lower() for k in calls[0][1]}

        assert fetch_endpoint_model_metadata(GATEWAY, api_key="", force_refresh=True) == {}
        assert len(calls) == 1, "a refused keyless probe must not be repeated"

    def test_403_is_remembered_like_401(self, monkeypatch):
        from agent.model_metadata import fetch_endpoint_model_metadata

        calls: list = []
        _install_stream(monkeypatch, 403, calls)

        fetch_endpoint_model_metadata(GATEWAY, api_key="", force_refresh=True)
        fetch_endpoint_model_metadata(GATEWAY, api_key="", force_refresh=True)
        assert len(calls) == 1

    def test_suppression_is_per_host_port(self, monkeypatch):
        from agent.model_metadata import fetch_endpoint_model_metadata

        calls: list = []
        _install_stream(monkeypatch, 401, calls)

        fetch_endpoint_model_metadata(GATEWAY, api_key="", force_refresh=True)
        fetch_endpoint_model_metadata("http://gateway.example.com:4300/v1", api_key="", force_refresh=True)
        assert len(calls) == 2

    def test_expiry_lets_the_probe_through_again(self, monkeypatch):
        from agent import model_metadata

        calls: list = []
        _install_stream(monkeypatch, 401, calls)

        model_metadata.fetch_endpoint_model_metadata(GATEWAY, api_key="", force_refresh=True)
        model_metadata._endpoint_unauth_probe_cache[GATEWAY_KEY] -= (
            model_metadata._ENDPOINT_UNAUTH_PROBE_TTL_SECONDS + 1
        )
        model_metadata.fetch_endpoint_model_metadata(GATEWAY, api_key="", force_refresh=True)
        assert len(calls) == 2


class TestKeyedCallersUnaffected:
    def test_keyed_caller_still_probes_after_a_keyless_refusal(self, monkeypatch):
        from agent.model_metadata import fetch_endpoint_model_metadata

        calls: list = []
        _install_stream(monkeypatch, 401, calls)

        fetch_endpoint_model_metadata(GATEWAY, api_key="", force_refresh=True)
        fetch_endpoint_model_metadata(GATEWAY, api_key="sk-live", force_refresh=True)

        assert len(calls) == 2, "a keyed caller must keep probing: its answer can differ"
        assert calls[1][1].get("Authorization", "").endswith("sk-live")

    def test_keyed_refusal_is_not_remembered(self, monkeypatch):
        from agent import model_metadata

        calls: list = []
        _install_stream(monkeypatch, 401, calls)

        model_metadata.fetch_endpoint_model_metadata(GATEWAY, api_key="sk-wrong", force_refresh=True)
        model_metadata.fetch_endpoint_model_metadata(GATEWAY, api_key="sk-wrong", force_refresh=True)

        assert len(calls) == 2
        assert model_metadata._endpoint_unauth_probe_cache == {}, \
            "a keyed 401 is a credential fault; suppressing it would hide the fix"

    def test_explicit_key_wins_over_an_existing_suppression(self, monkeypatch):
        from agent import model_metadata

        calls: list = []
        _install_stream(monkeypatch, 401, calls)

        model_metadata.fetch_endpoint_model_metadata(GATEWAY, api_key="", force_refresh=True)
        assert model_metadata._endpoint_unauth_probe_refused(GATEWAY) is True
        model_metadata.fetch_endpoint_model_metadata(GATEWAY, api_key="sk-live", force_refresh=True)
        assert len(calls) == 2
