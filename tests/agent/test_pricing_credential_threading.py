"""ATE-18791: cost lookups must authenticate the endpoint the request itself used.

The pricing catalogue behind a cost breakdown is read from ``<base_url>/models``. Called keyless
against a keyed gateway that read returns 401, the catalogue is empty, and every per-type cost
component silently disappears (the total survives from other sources, so nothing looks broken).

Covers:
- ``resolve_endpoint_api_key`` itself: same-origin only, callable passthrough, fail-open to ""
- the Langfuse plugin's ``_canonical_usage_and_cost`` (both the estimate and the rate lookup)
- ``agent.insights._estimate_cost`` and ``agent.aux_accounting.record_aux_usage``
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

GW = "http://gateway.example.com:4200/v1"


def _plugin():
    sys.modules.pop("plugins.observability.langfuse", None)
    return importlib.import_module("plugins.observability.langfuse")


def _canonical():
    from agent.usage_pricing import CanonicalUsage

    return CanonicalUsage(input_tokens=1000, output_tokens=10)


class TestResolveEndpointApiKey:
    def test_empty_base_url_never_reads_a_key(self):
        from agent.usage_pricing import resolve_endpoint_api_key

        assert resolve_endpoint_api_key("") == ""
        assert resolve_endpoint_api_key(None) == ""

    def test_returns_the_same_origin_reader_value(self, monkeypatch):
        from agent import auxiliary_client, usage_pricing

        monkeypatch.setattr(auxiliary_client, "endpoint_api_key_if_same_origin", lambda base_url: "sk-live")
        assert usage_pricing.resolve_endpoint_api_key(GW) == "sk-live"

    def test_callable_credential_passes_through_unchanged(self, monkeypatch):
        from agent import auxiliary_client, usage_pricing

        rotate = lambda: "sk-rotated"
        monkeypatch.setattr(auxiliary_client, "endpoint_api_key_if_same_origin", lambda base_url: rotate)
        assert usage_pricing.resolve_endpoint_api_key(GW) is rotate

    def test_reader_failure_is_fail_open(self, monkeypatch):
        from agent import auxiliary_client, usage_pricing

        def boom(base_url):
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(auxiliary_client, "endpoint_api_key_if_same_origin", boom)
        assert usage_pricing.resolve_endpoint_api_key(GW) == ""


class TestLangfuseCostThreading:
    def _patch_pricing(self, monkeypatch, seen):
        from agent import usage_pricing

        monkeypatch.setattr(
            usage_pricing, "resolve_billing_route",
            lambda model, provider=None, base_url=None: SimpleNamespace(billing_mode="", provider=provider or ""),
        )

        def fake_estimate(model, usage, *, provider=None, base_url=None, api_key=None):
            seen["estimate"] = api_key
            return SimpleNamespace(amount_usd=1.25, status="estimated")

        def fake_entry(model, *, provider=None, base_url=None, api_key=None):
            seen["entry"] = api_key

        monkeypatch.setattr(usage_pricing, "estimate_usage_cost", fake_estimate)
        monkeypatch.setattr(usage_pricing, "get_pricing_entry", fake_entry)

    def test_both_pricing_calls_carry_the_turn_credential(self, monkeypatch):
        from agent import usage_pricing

        mod = _plugin()
        seen: dict = {}
        self._patch_pricing(monkeypatch, seen)
        monkeypatch.setattr(usage_pricing, "resolve_endpoint_api_key", lambda base_url: "sk-live")

        mod._canonical_usage_and_cost(_canonical(), provider="custom", model="acme-large", base_url=GW)

        assert seen == {"estimate": "sk-live", "entry": "sk-live"}

    def test_explicit_request_key_skips_resolution(self, monkeypatch):
        from agent import usage_pricing

        mod = _plugin()
        seen: dict = {}
        self._patch_pricing(monkeypatch, seen)

        def _must_not_run(base_url):
            raise AssertionError("a supplied credential must not be re-resolved")

        monkeypatch.setattr(usage_pricing, "resolve_endpoint_api_key", _must_not_run)

        mod._canonical_usage_and_cost(_canonical(), provider="custom", model="acme-large", base_url=GW,
                                      api_key="sk-request")

        assert seen == {"estimate": "sk-request", "entry": "sk-request"}

    def test_subscription_route_still_exports_no_costs(self, monkeypatch):
        """The subscription short-circuit must run before any credential work."""
        from agent import usage_pricing

        mod = _plugin()
        monkeypatch.setattr(
            usage_pricing, "resolve_billing_route",
            lambda model, provider=None, base_url=None: SimpleNamespace(billing_mode="subscription_included"),
        )
        monkeypatch.setattr(
            usage_pricing, "resolve_endpoint_api_key",
            lambda base_url: pytest.fail("no pricing lookup, so no credential is needed"),
        )

        usage_details, cost_details = mod._canonical_usage_and_cost(
            _canonical(), provider="anthropic", model="claude-sonnet-4-6", base_url=GW
        )
        assert usage_details["input"] == 1000
        assert cost_details == {}


class TestInsightsAndAuxAccounting:
    def test_insights_estimate_passes_the_credential(self, monkeypatch):
        from agent import insights

        seen: dict = {}
        monkeypatch.setattr(insights, "resolve_endpoint_api_key", lambda base_url: "sk-live")
        monkeypatch.setattr(
            insights, "estimate_usage_cost",
            lambda model, usage, *, provider=None, base_url=None, api_key=None: (
                seen.update(api_key=api_key, base_url=base_url) or SimpleNamespace(amount_usd=0.5, status="estimated")
            ),
        )

        insights._estimate_cost("acme-large", 1000, 10, provider="custom", base_url=GW)

        assert seen == {"api_key": "sk-live", "base_url": GW}

    def test_aux_accounting_passes_the_credential(self, monkeypatch):
        from agent import aux_accounting, usage_pricing

        seen: dict = {}
        monkeypatch.setattr(usage_pricing, "resolve_endpoint_api_key", lambda base_url: "sk-live")
        monkeypatch.setattr(
            usage_pricing, "estimate_usage_cost",
            lambda model, usage, *, provider=None, base_url=None, api_key=None: (
                seen.update(api_key=api_key, base_url=base_url) or SimpleNamespace(amount_usd=0.25, status="estimated")
            ),
        )

        class _StubDB:
            def __init__(self):
                self.rows = []

            def record_auxiliary_usage(self, session_id, task, **kwargs):
                self.rows.append((session_id, task, kwargs))

        db = _StubDB()
        token = aux_accounting.set_accounting_context(db, "session-1")
        try:
            aux_accounting.record_aux_usage(
                SimpleNamespace(model="aux-model", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20)),
                "vision", provider="custom", base_url=GW,
            )
        finally:
            aux_accounting.reset_accounting_context(token)

        assert seen == {"api_key": "sk-live", "base_url": GW}
        assert len(db.rows) == 1
        assert db.rows[0][2]["estimated_cost_usd"] == pytest.approx(0.25)


class TestConfigProviderCredentialFallback:
    """No bound runtime (fresh worker, background flush): the config-declared provider's own key.

    ATE-18791 measured this on r410: with no turn bound, the same-origin reader returns "" because
    ``model.api_key`` is empty and the gateway key lives in ``providers.litellm.key_env``. Left at that,
    the cost path would keep probing keyless in exactly the processes that generate the 401s.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        from agent import auxiliary_client

        auxiliary_client._config_provider_key_cache.clear()
        yield
        auxiliary_client._config_provider_key_cache.clear()

    def _bind_config(self, monkeypatch, providers, *, calls=None, error=None):
        from hermes_cli import config as cfg_mod

        def fake_load_config():
            if calls is not None:
                calls.append(1)
            if error is not None:
                raise error
            return {"providers": providers}

        monkeypatch.setattr(cfg_mod, "load_config", fake_load_config)

    def _unbound(self, monkeypatch):
        from agent import auxiliary_client

        monkeypatch.setattr(auxiliary_client, "endpoint_api_key_if_same_origin", lambda base_url: "")

    def test_key_env_credential_is_used_when_no_runtime_is_bound(self, monkeypatch):
        from agent import auxiliary_client, usage_pricing

        self._unbound(monkeypatch)
        self._bind_config(monkeypatch, {"litellm": {"base_url": GW, "key_env": "ATEK_GW_KEY"}})
        monkeypatch.setattr(
            auxiliary_client, "_scoped_key_env",
            lambda name: "sk-config-gateway" if name == "ATEK_GW_KEY" else "",
        )

        assert usage_pricing.resolve_endpoint_api_key(GW) == "sk-config-gateway"

    def test_inline_api_key_is_used(self, monkeypatch):
        from agent import auxiliary_client, usage_pricing

        self._unbound(monkeypatch)
        self._bind_config(monkeypatch, {"litellm": {"base_url": GW, "api_key": "sk-inline"}})
        monkeypatch.setattr(auxiliary_client, "_scoped_key_env", lambda name: "")

        assert usage_pricing.resolve_endpoint_api_key(GW) == "sk-inline"

    def test_a_different_port_on_the_same_host_never_borrows_the_key(self, monkeypatch):
        from agent import auxiliary_client, usage_pricing

        self._unbound(monkeypatch)
        self._bind_config(monkeypatch, {"litellm": {"base_url": GW, "api_key": "sk-inline"}})
        monkeypatch.setattr(auxiliary_client, "_scoped_key_env", lambda name: "")

        assert usage_pricing.resolve_endpoint_api_key("http://gateway.example.com:4201/v1") == ""

    def test_a_matching_provider_with_no_credential_stays_keyless(self, monkeypatch):
        from agent import auxiliary_client, usage_pricing

        self._unbound(monkeypatch)
        self._bind_config(monkeypatch, {"litellm": {"base_url": GW}})
        monkeypatch.setattr(auxiliary_client, "_scoped_key_env", lambda name: "")

        assert usage_pricing.resolve_endpoint_api_key(GW) == ""

    def test_unknown_origin_stays_keyless(self, monkeypatch):
        from agent import auxiliary_client, usage_pricing

        self._unbound(monkeypatch)
        self._bind_config(monkeypatch, {"litellm": {"base_url": GW, "api_key": "sk-inline"}})

        assert usage_pricing.resolve_endpoint_api_key("https://api.other-vendor.com/v1") == ""

    def test_the_live_runtime_credential_wins_and_config_is_not_read(self, monkeypatch):
        from agent import auxiliary_client, usage_pricing

        calls: list = []
        monkeypatch.setattr(auxiliary_client, "endpoint_api_key_if_same_origin", lambda base_url: "sk-live")
        self._bind_config(monkeypatch, {"litellm": {"base_url": GW, "api_key": "sk-inline"}}, calls=calls)

        assert usage_pricing.resolve_endpoint_api_key(GW) == "sk-live"
        assert calls == []

    def test_one_config_read_per_window(self, monkeypatch):
        from agent import auxiliary_client, usage_pricing

        calls: list = []
        self._unbound(monkeypatch)
        self._bind_config(monkeypatch, {"litellm": {"base_url": GW, "api_key": "sk-inline"}}, calls=calls)
        monkeypatch.setattr(auxiliary_client, "_scoped_key_env", lambda name: "")

        assert usage_pricing.resolve_endpoint_api_key(GW) == "sk-inline"
        assert usage_pricing.resolve_endpoint_api_key(GW) == "sk-inline"
        assert len(calls) == 1

    def test_unreadable_config_is_fail_open(self, monkeypatch):
        from agent import auxiliary_client, usage_pricing

        self._unbound(monkeypatch)
        self._bind_config(monkeypatch, {}, error=RuntimeError("config unreadable"))

        assert usage_pricing.resolve_endpoint_api_key(GW) == ""
