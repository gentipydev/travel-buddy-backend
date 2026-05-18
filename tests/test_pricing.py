"""Tests for `app.observability.pricing.compute_cost`.

The function is plumbing — its only job is to multiply tokens by the
right per-model rate and never crash on an unknown SKU. So we cover
each of the four token categories independently and assert the unknown-
model and None-token paths return 0.0 (telemetry must never raise).
"""
import pytest

from app.observability.pricing import compute_cost

_HAIKU = "claude-haiku-4-5-20251001"
_SONNET = "claude-sonnet-4-6"


def test_haiku_input_and_output_only():
    # 100 input × $1.00/1M + 200 output × $5.00/1M
    # = 0.0001 + 0.001 = 0.0011
    cost = compute_cost(
        model=_HAIKU,
        input_tokens=100,
        output_tokens=200,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    assert cost == pytest.approx(0.0011)


def test_haiku_cache_tokens_contribute():
    # 1000 cache_creation × $1.25/1M + 5000 cache_read × $0.10/1M
    # = 0.00125 + 0.0005 = 0.00175 (plus 0 for input/output)
    cost = compute_cost(
        model=_HAIKU,
        input_tokens=None,
        output_tokens=None,
        cache_creation_input_tokens=1000,
        cache_read_input_tokens=5000,
    )
    assert cost == pytest.approx(0.00175)


def test_haiku_all_four_categories_sum():
    # 1M of each category at Haiku rates: 1.00 + 5.00 + 1.25 + 0.10 = 7.35
    cost = compute_cost(
        model=_HAIKU,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert cost == pytest.approx(7.35)


def test_sonnet_rates_differ_from_haiku():
    # Sonnet output is $15/1M; Haiku output is $5/1M. Same tokens, 3x cost.
    haiku = compute_cost(
        _HAIKU,
        input_tokens=0,
        output_tokens=1_000_000,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    sonnet = compute_cost(
        _SONNET,
        input_tokens=0,
        output_tokens=1_000_000,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    assert sonnet == pytest.approx(haiku * 3)


def test_unknown_model_returns_zero():
    # Telemetry must never crash on an unfamiliar SKU; an unknown model
    # is a signal to refresh `_PRICING`, not a request-path failure.
    cost = compute_cost(
        model="claude-unreleased-xyz-99",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert cost == 0.0


def test_all_none_tokens_returns_zero():
    cost = compute_cost(
        model=_HAIKU,
        input_tokens=None,
        output_tokens=None,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    assert cost == 0.0


def test_none_in_one_field_treated_as_zero():
    # Only output_tokens populated; the three None fields contribute 0.
    cost = compute_cost(
        model=_HAIKU,
        input_tokens=None,
        output_tokens=500,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    # 500 × $5.00 / 1M = 0.0025
    assert cost == pytest.approx(0.0025)
