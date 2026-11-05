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
