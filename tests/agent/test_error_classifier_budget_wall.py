"""ATE-4494 — a LiteLLM budget wall on HTTP 429 is terminal, not a rate limit.

LiteLLM answers an exhausted per-key / per-user budget with HTTP 429,
``type: budget_exceeded``, ``code: 429``. Before ATE-4494 that fell through to the
generic rate-limit verdict (``retryable=True``, ``should_rotate_credential=True``),
so the runtime made three backed-off attempts per API call and retried into the
wall for the whole run (measured: 4,242 ``ExceededBudget`` lines and 943
``Retrying API call`` lines in one morning).

A budget wall is terminal for the window: waiting does not clear it, and rotating
the credential does not help because the cap is on the end-user, not the key. It
must classify as ``billing`` with ``retryable=False`` and no rotation, so the run
ends on attempt 1 and records the cap in its outcome.

The match is deliberately narrow and vendor-scoped, so a genuine provider rate
limit is untouched — that is what the guard test below pins.
"""

from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError (same shape as the sibling test)."""

    def __init__(self, message, status_code=None, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


# The exact string a live run hit, with the real seat id and cap values.
LIVE_BUDGET_MESSAGE = (
    "ExceededBudget: End User=d40865e0-adcb-46f7-a2a4-f8c6b44a20aa over budget. "
    "Spend=5.570341730439999, Budget=5.57"
)

LITELLM_BUDGET_BODY = {
    "error": {
        "message": LIVE_BUDGET_MESSAGE,
        "type": "budget_exceeded",
        "code": "429",
    }
}


class TestBudgetWallIsTerminal:
    def test_live_budget_wall_is_billing_not_retryable(self):
        """The live body from 2026-10-14 must stop the run on attempt 1."""
        e = MockAPIError(LIVE_BUDGET_MESSAGE, status_code=429, body=LITELLM_BUDGET_BODY)

        result = classify_api_error(
            e,
            provider="custom",
            model="deepinfra/deepseek-v4.1-flash",
            base_url="http://litellm.atek.vpn:4200/v1",
        )

        assert result.reason == FailoverReason.billing
        assert result.retryable is False

    def test_budget_wall_does_not_rotate_the_credential(self):
        """The cap is on the end-user, so rotating the key cannot clear it."""
        e = MockAPIError(LIVE_BUDGET_MESSAGE, status_code=429, body=LITELLM_BUDGET_BODY)

        result = classify_api_error(e, provider="custom")

        assert result.should_rotate_credential is False

    def test_budget_wall_does_not_fall_back_to_another_provider(self):
        """The end-user is over budget everywhere, so a fallback would just 429 again."""
        e = MockAPIError(LIVE_BUDGET_MESSAGE, status_code=429, body=LITELLM_BUDGET_BODY)

        result = classify_api_error(e, provider="custom")

        assert result.should_fallback is False

    @pytest.mark.parametrize(
        "message",
        [
            LIVE_BUDGET_MESSAGE,
            "budget_exceeded",
            "Budget exceeded for key",
            "Key is over budget",
            "you are over budget for this window",
        ],
    )
    def test_vendor_budget_spellings_are_billing(self, message):
        e = MockAPIError(message, status_code=429)

        result = classify_api_error(e, provider="custom")

        assert result.reason == FailoverReason.billing
        assert result.retryable is False

    def test_budget_wall_declared_only_in_the_body_type_is_billing(self):
        """LiteLLM's own ``type`` field is decisive even when the prose is generic.

        The message is vendor-controlled and can be reworded, wrapped or replaced
        by a proxy; ``type: budget_exceeded`` is the structured signal. Reading only
        the prose made the verdict depend on a sentence — the same brittleness that
        produced the original defect.
        """
        body = {
            "error": {
                "message": "Rate limit reached for key",
                "type": "budget_exceeded",
                "code": "429",
            }
        }
        e = MockAPIError("Rate limit reached for key", status_code=429, body=body)

        result = classify_api_error(e, provider="custom")

        assert result.reason == FailoverReason.billing
        assert result.retryable is False
        assert result.should_rotate_credential is False
        assert result.should_fallback is False


class TestRateLimitGuard:
    """A generic provider rate limit must still retry — the fix must not overreach."""

    def test_generic_429_rate_limit_still_retries(self):
        e = MockAPIError("Rate limit exceeded: too many requests", status_code=429)

        result = classify_api_error(e, provider="custom")

        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True

    def test_body_type_rate_limit_error_still_retries(self):
        """A structured ``rate_limit_error`` type must not be read as a budget wall."""
        body = {
            "error": {
                "message": "Rate limit reached for key",
                "type": "rate_limit_error",
                "code": "429",
            }
        }
        e = MockAPIError("Rate limit reached for key", status_code=429, body=body)

        result = classify_api_error(e, provider="custom")

        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True

    def test_org_quota_with_retry_signal_stays_transient(self):
        e = MockAPIError(
            "You exceeded your current quota, please try again in 5 minutes",
            status_code=429,
        )

        result = classify_api_error(e, provider="custom")

        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True
