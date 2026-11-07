"""ATE-19359 — the picker probe must never go out without the endpoint's key.

LiteLLM answers a keyless ``GET /v1/models`` with 401 and logs "No api key passed in" plus a
traceback; because the empty result is cached per credential, a probe that cannot authenticate
repeats every TTL forever. The picker resolves ``providers.<name>.key_env`` from config, so a
process where that resolution comes up empty still sent the probe — keyless.

(The non-picker callers of the same endpoint — the usage-pricing path and the model-metadata probe —
are ATE-18791, PR #18.)

Covers:
- the picker adopting the configured key when the caller hands it none
- the picker skipping the probe when a configured key does not resolve here
- a credential-less endpoint (local llama.cpp / LM Studio) still being probed keyless
- a caller-supplied Authorization keeping the probe authenticated and untouched
- the synthesized Authorization reaching the wire on the generic branch
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import http.server
import json
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE = "http://litellm.example.test:4200/v1"


@pytest.fixture(autouse=True)
def _clear_metadata_caches():
    from agent import model_metadata
    model_metadata._endpoint_model_metadata_cache.clear()
    model_metadata._endpoint_model_metadata_cache_time.clear()
    model_metadata._endpoint_blackhole_cache.clear()
    yield
    model_metadata._endpoint_model_metadata_cache.clear()
    model_metadata._endpoint_model_metadata_cache_time.clear()
    model_metadata._endpoint_blackhole_cache.clear()


def _configure(monkeypatch, *, key_value: str, entry: dict | None = None,
               in_custom: bool = False) -> list[str]:
    """Install a config with one endpoint entry and a resolvable env key."""
    entry = entry if entry is not None else {"base_url": BASE, "key_env": "LITELLM_API_KEY"}
    cfg = {"custom_providers": [entry]} if in_custom else {"providers": {"litellm": entry}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    lookups: list[str] = []

    def _scoped(name: str) -> str:
        lookups.append(name)
        return key_value if name == "LITELLM_API_KEY" else ""

    monkeypatch.setattr("hermes_cli.model_switch._scoped_key_env", _scoped)
    return lookups


def _pic_only(monkeypatch) -> None:
    """Neutralise the Ollama-native detection so the generic branch is what runs."""
    monkeypatch.setattr("hermes_cli.models._get_ollama_native_headers", lambda *a, **k: {})
    monkeypatch.setattr("hermes_cli.models_local.should_use_ollama_native_catalog",
                        lambda *a, **k: False)


def _probe(api_key="", headers=None, *, api_mode=None, credential_identity=None):
    from hermes_cli.model_switch_providers import _fetch_picker_live_models
    return _fetch_picker_live_models(api_key, BASE, "custom", False,
                                     headers=headers, timeout=1.0, cache=False,
                                     api_mode=api_mode, credential_identity=credential_identity)


def _record_fetch(monkeypatch, result=("model-a", "model-b")):
    calls: list[dict] = []

    def _fetch(key, url, timeout=5.0, api_mode=None, headers=None):
        calls.append({"api_key": key, "url": url, "api_mode": api_mode, "headers": dict(headers or {})})
        return list(result) if result is not None else None

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", _fetch)
    return calls


def test_probe_adopts_configured_key_when_caller_passes_none(monkeypatch):
    _pic_only(monkeypatch)
    lookups = _configure(monkeypatch, key_value="sk-configured")
    calls = _record_fetch(monkeypatch)

    assert _probe() == ["model-a", "model-b"]

    assert calls, "the probe must still run — it just must authenticate"
    assert calls[0]["api_key"] == "sk-configured"
    assert "Authorization" not in calls[0]["headers"], (
        "the key rides on api_key; probe_api_models builds the bearer itself")
    assert "LITELLM_API_KEY" in lookups


def test_probe_is_skipped_when_configured_key_does_not_resolve(monkeypatch):
    _pic_only(monkeypatch)
    _configure(monkeypatch, key_value="")
    calls = _record_fetch(monkeypatch)

    assert _probe() is None

    assert calls == [], "a declared credential that did not resolve must not be probed keyless"


def test_probe_runs_keyless_for_endpoint_without_declared_credential(monkeypatch):
    """Local endpoints authenticate by reachability — the guard must not silence them."""
    _pic_only(monkeypatch)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    calls = _record_fetch(monkeypatch)

    assert _probe() == ["model-a", "model-b"]

    assert len(calls) == 1
    assert not calls[0]["headers"].get("Authorization")


def test_caller_authorization_is_kept_and_not_replaced(monkeypatch):
    _pic_only(monkeypatch)
    _configure(monkeypatch, key_value="sk-configured")
    calls = _record_fetch(monkeypatch)

    _probe(headers={"Authorization": "Bearer caller-key", "x-extra": "1"})

    assert calls[0]["headers"].get("Authorization") == "Bearer caller-key"
    assert calls[0]["headers"].get("x-extra") == "1"


def test_caller_extra_headers_survive_a_recovered_key(monkeypatch):
    """The generic branch keeps the caller's headers; the key travels as api_key."""
    _pic_only(monkeypatch)
    _configure(monkeypatch, key_value="sk-configured")
    calls = _record_fetch(monkeypatch)

    _probe(headers={"x-paperclip-agent": "abc"})

    assert calls[0]["api_key"] == "sk-configured"
    assert calls[0]["headers"].get("x-paperclip-agent") == "abc"
    assert "Authorization" not in calls[0]["headers"]


def test_custom_providers_row_declares_the_credential_too(monkeypatch):
    _pic_only(monkeypatch)
    _configure(monkeypatch, key_value="", in_custom=True)
    calls = _record_fetch(monkeypatch)

    assert _probe() is None
    assert calls == []


def test_configured_endpoint_credential_reports_declared_without_a_key(monkeypatch):
    from hermes_cli.model_switch_providers import configured_endpoint_credential

    _configure(monkeypatch, key_value="")
    assert configured_endpoint_credential(BASE) == ("", True)


def test_configured_endpoint_credential_is_empty_for_an_unknown_endpoint(monkeypatch):
    from hermes_cli.model_switch_providers import configured_endpoint_credential

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    assert configured_endpoint_credential("http://127.0.0.1:8080/v1") == ("", False)


# --- MAJOR 1 (Code Reviewer, PR #20): N rows may share one proxy URL with distinct keys -------
# A row whose own key did not resolve must not authenticate as its sibling and show that tenant's
# catalog as its own. The lookup therefore matches on (url, credential identity), not on the URL.

TWO_TENANTS = {
    "custom_providers": [
        {"name": "tenant-a", "base_url": BASE, "key_env": "TENANT_A_KEY"},
        {"name": "tenant-b", "base_url": BASE, "api_key": "tenant-b-key"},
    ]
}


def _configure_two_tenants(monkeypatch, *, tenant_a_key: str) -> None:
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: TWO_TENANTS)
    monkeypatch.setattr(
        "hermes_cli.model_switch._scoped_key_env",
        lambda name: tenant_a_key if name == "TENANT_A_KEY" else "")


def test_row_with_an_unresolved_key_does_not_adopt_a_sibling_rows_key(monkeypatch):
    _pic_only(monkeypatch)
    _configure_two_tenants(monkeypatch, tenant_a_key="")
    calls = _record_fetch(monkeypatch)

    assert _probe(credential_identity="env:TENANT_A_KEY") is None

    assert calls == [], "row-a must not be probed, and must never authenticate as tenant-b"


def test_row_uses_only_its_own_key_when_two_rows_share_the_url(monkeypatch):
    _pic_only(monkeypatch)
    _configure_two_tenants(monkeypatch, tenant_a_key="tenant-a-key")
    calls = _record_fetch(monkeypatch)

    assert _probe(credential_identity="env:TENANT_A_KEY") == ["model-a", "model-b"]

    assert calls[0]["api_key"] == "tenant-a-key"
    assert "tenant-b-key" not in json.dumps(calls[0], default=str)


def test_url_only_lookup_refuses_an_ambiguous_endpoint(monkeypatch):
    """No row to name: two declaring entries on one URL -> declared, no key, no probe."""
    _pic_only(monkeypatch)
    _configure_two_tenants(monkeypatch, tenant_a_key="tenant-a-key")
    calls = _record_fetch(monkeypatch)

    assert _probe() is None

    assert calls == []


def test_picker_row_passes_its_own_credential_identity_to_the_probe(monkeypatch):
    """The row builders must hand the lookup the identity they grouped the row on."""
    _pic_only(monkeypatch)
    _configure_two_tenants(monkeypatch, tenant_a_key="")
    from hermes_cli.model_switch_providers import _PickerBuild, _lap_custom_provider_rows

    build = _PickerBuild(
        current_provider="", current_base_url="", current_model="", max_models=None,
        for_picker=True, force_fresh_nous_tier=False, probe_custom_providers=True,
        probe_current_custom_provider=False, refresh=False, excluded=set(), curated={})
    seen: list[dict] = []

    def _discover(self, api_key, api_url, native_provider, has_explicit, **kw):
        seen.append({"api_key": api_key, "credential_identity": kw.get("credential_identity")})
        return None, False, True

    monkeypatch.setattr(_PickerBuild, "discover_endpoint", _discover)
    _lap_custom_provider_rows(build, TWO_TENANTS["custom_providers"])

    assert {row["credential_identity"] for row in seen} == {"env:TENANT_A_KEY", "tenant-b-key"}
    assert {row["api_key"] for row in seen} == {"", "tenant-b-key"}


# --- MAJOR 2 (Code Reviewer, PR #20): probe_api_models builds x-api-key, never a bearer --------
# ``anthropic_messages`` authenticates with x-api-key + anthropic-version in models.py; a
# synthesized Authorization would be a header that mode never sends.

def test_anthropic_messages_probe_sends_no_synthesized_bearer(monkeypatch):
    _pic_only(monkeypatch)
    _configure(monkeypatch, key_value="")
    calls = _record_fetch(monkeypatch)

    assert _probe(api_key="caller-key", api_mode="anthropic_messages") == ["model-a", "model-b"]

    assert "Authorization" not in calls[0]["headers"]
    assert calls[0]["api_key"] == "caller-key"


def test_anthropic_messages_probe_adopts_the_key_but_still_no_bearer(monkeypatch):
    _pic_only(monkeypatch)
    _configure(monkeypatch, key_value="configured-key")
    calls = _record_fetch(monkeypatch)

    assert _probe(api_mode="anthropic_messages") == ["model-a", "model-b"]

    assert "Authorization" not in calls[0]["headers"]
    assert calls[0]["api_key"] == "configured-key"


# --- MINOR 1 (Code Reviewer, PR #20): a disabled providers entry declares no credential --------

def test_disabled_provider_entry_does_not_declare_a_credential(monkeypatch):
    _pic_only(monkeypatch)
    cfg = {"providers": {"retired": {"base_url": BASE, "key_env": "RETIRED_KEY", "enabled": False}}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr("hermes_cli.model_switch._scoped_key_env", lambda name: "")
    calls = _record_fetch(monkeypatch)

    assert _probe() == ["model-a", "model-b"]

    assert len(calls) == 1
    assert "Authorization" not in calls[0]["headers"]


# --- Wire level: the header set that actually reaches the endpoint -----------------------------
# The reviewed defects were header-set bugs, and only the wire shows the set: a sibling row's key
# adopted as its own (MAJOR 1) and a bearer that ``anthropic_messages`` never sends (MAJOR 2).
# Real socket, real ``fetch_api_models`` — no monkeypatched probe.

class _ProbeRecorder(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.server.seen.append({k.lower(): v for k, v in self.headers.items()})
        body = json.dumps({"data": [{"id": "wire-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture
def wire_endpoint():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ProbeRecorder)
    server.daemon_threads = True
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_wire_recovered_key_is_sent_as_a_bearer(monkeypatch, wire_endpoint):
    _pic_only(monkeypatch)
    from hermes_cli.model_switch_providers import _fetch_picker_live_models

    url = f"http://127.0.0.1:{wire_endpoint.server_address[1]}/v1"
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "custom_providers": [{"name": "litellm", "base_url": url, "api_key": "sk-configured"}]})

    assert _fetch_picker_live_models("", url, "custom", False, timeout=3.0, cache=False,
                                     credential_identity="sk-configured") == ["wire-model"]

    assert wire_endpoint.seen[0].get("authorization") == "Bearer sk-configured"


def test_wire_anthropic_messages_sends_x_api_key_and_no_bearer(monkeypatch, wire_endpoint):
    _pic_only(monkeypatch)
    from hermes_cli.model_switch_providers import _fetch_picker_live_models

    url = f"http://127.0.0.1:{wire_endpoint.server_address[1]}/v1"
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

    assert _fetch_picker_live_models("sk-anthropic", url, "custom", False, timeout=3.0,
                                     cache=False, api_mode="anthropic_messages") == ["wire-model"]

    headers = wire_endpoint.seen[0]
    assert headers.get("x-api-key") == "sk-anthropic"
    assert headers.get("anthropic-version") == "2023-06-01"
    assert "authorization" not in headers
