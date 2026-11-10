"""Tests for `app.observability.pricing.compute_cost`.

The function is plumbing — its only job is to multiply tokens by the
right per-model rate and never crash on an unknown SKU. So we cover
each of the four token categories independently and assert the unknown-
model and None-token paths return 0.0 (telemetry must never raise).

`tests/test_config.py::test_model_defaults_are_priced` is the companion
guard: it fails if a `settings.model_*` default has no row here at all.
"""
import pytest

from app.observability.pricing import compute_cost

_HAIKU = "claude-haiku-4-5"
_SONNET = "claude-sonnet-5"


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
    # 1M of each category at claude-haiku-4-5 rates:
    # 1.00 + 5.00 + 1.25 + 0.10 = 7.35
    cost = compute_cost(
        model=_HAIKU,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert cost == pytest.approx(7.35)


def test_cache_read_is_the_cheap_path():
    """The whole point of prompt caching: a cached input token costs a tenth
    of an uncached one, and a cache WRITE costs 1.25x. If this inverts, the
    caching layer is a cost regression rather than a win."""
    uncached = compute_cost(
        _HAIKU, 1_000_000, 0, None, None,
    )
    write = compute_cost(
        _HAIKU, 0, 0, 1_000_000, None,
    )
    read = compute_cost(
        _HAIKU, 0, 0, None, 1_000_000,
    )
    assert read < uncached < write
    assert read == pytest.approx(uncached * 0.10)
    assert write == pytest.approx(uncached * 1.25)


def test_haiku_is_cheaper_than_sonnet():
    # Haiku output is $5.00/1M; Sonnet 5 output is $10.00/1M — same tokens,
    # the smaller tier costs less. Sonnet is priced here so a MODEL_CHAT
    # upgrade doesn't silently fall into the unknown-model 0.0 hole.
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
    assert haiku == pytest.approx(5.00)
    assert sonnet == pytest.approx(10.00)
    assert haiku < sonnet


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


def test_dated_model_id_is_not_priced():
    """A dated snapshot id is NOT the public model id. Pinning one in .env
    would silently zero every cost estimate, so make the trap explicit."""
    assert compute_cost(
        "claude-haiku-4-5-20251001", 1_000_000, 0, None, None,
    ) == 0.0


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
