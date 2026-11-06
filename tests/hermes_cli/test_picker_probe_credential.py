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


def _probe(api_key="", headers=None):
    from hermes_cli.model_switch_providers import _fetch_picker_live_models
    return _fetch_picker_live_models(api_key, BASE, "custom", False,
                                     headers=headers, timeout=1.0, cache=False)


def _record_fetch(monkeypatch, result=("model-a", "model-b")):
    calls: list[dict] = []

    def _fetch(key, url, timeout=5.0, api_mode=None, headers=None):
        calls.append({"api_key": key, "url": url, "headers": dict(headers or {})})
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
    assert calls[0]["headers"].get("Authorization") == "Bearer sk-configured"
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


def test_configured_key_reaches_the_wire_with_caller_extra_headers(monkeypatch):
    """The generic branch used to drop the synthesized Authorization; it must not."""
    _pic_only(monkeypatch)
    _configure(monkeypatch, key_value="sk-configured")
    calls = _record_fetch(monkeypatch)

    _probe(headers={"x-paperclip-agent": "abc"})

    assert calls[0]["headers"].get("Authorization") == "Bearer sk-configured"
    assert calls[0]["headers"].get("x-paperclip-agent") == "abc"


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
