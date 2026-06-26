"""Tests for `app.observability.pricing.compute_cost`.

The function is plumbing — its only job is to multiply tokens by the
right per-model rate and never crash on an unknown SKU. So we cover
each of the four token categories independently and assert the unknown-
model and None-token paths return 0.0 (telemetry must never raise).
"""
import pytest

from app.observability.pricing import compute_cost

_FLASH = "gemini-2.5-flash"
_LITE = "gemini-2.5-flash-lite"


def test_flash_input_and_output_only():
    # 100 input × $0.30/1M + 200 output × $2.50/1M
    # = 0.00003 + 0.0005 = 0.00053
    cost = compute_cost(
        model=_FLASH,
        input_tokens=100,
        output_tokens=200,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    assert cost == pytest.approx(0.00053)


def test_flash_cache_tokens_contribute():
    # 1000 cache_creation × $0.30/1M + 5000 cache_read × $0.075/1M
    # = 0.0003 + 0.000375 = 0.000675 (plus 0 for input/output)
    cost = compute_cost(
        model=_FLASH,
        input_tokens=None,
        output_tokens=None,
        cache_creation_input_tokens=1000,
        cache_read_input_tokens=5000,
    )
    assert cost == pytest.approx(0.000675)


def test_flash_all_four_categories_sum():
    # 1M of each category at gemini-2.5-flash rates:
    # 0.30 + 2.50 + 0.30 + 0.075 = 3.175
    cost = compute_cost(
        model=_FLASH,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert cost == pytest.approx(3.175)


def test_flash_lite_is_cheaper_than_flash():
    # Flash-lite output is $0.40/1M; flash output is $2.50/1M — same tokens,
    # the lite tier costs less.
    flash = compute_cost(
        _FLASH,
        input_tokens=0,
        output_tokens=1_000_000,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    lite = compute_cost(
        _LITE,
        input_tokens=0,
        output_tokens=1_000_000,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    assert flash == pytest.approx(2.50)
    assert lite == pytest.approx(0.40)
    assert lite < flash


def test_unknown_model_returns_zero():
    # Telemetry must never crash on an unfamiliar SKU; an unknown model
    # is a signal to refresh `_PRICING`, not a request-path failure.
    cost = compute_cost(
        model="gemini-unreleased-xyz-99",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert cost == 0.0


def test_deepseek_v4_flash_rates():
    # DeepSeek fallback: 1000 input × $0.14/1M + 2000 output × $0.28/1M
    # = 0.00014 + 0.00056 = 0.0007
    cost = compute_cost(
        model="deepseek-v4-flash",
        input_tokens=1000,
        output_tokens=2000,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    assert cost == pytest.approx(0.0007)


def test_deepseek_v4_flash_cache_read_is_cheap():
    # Cache-hit input bills at the cheap cache_read rate ($0.0028/1M).
    cost = compute_cost(
        model="deepseek-v4-flash",
        input_tokens=None,
        output_tokens=None,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=1_000_000,
    )
    assert cost == pytest.approx(0.0028)


def test_deepseek_flash_cheaper_than_gemini_flash_per_turn():
    # The whole point of the spill: same tokens cost less on the fallback.
    tokens = dict(
        input_tokens=1000,
        output_tokens=2000,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    gemini = compute_cost(model="gemini-3.5-flash", **tokens)
    deepseek = compute_cost(model="deepseek-v4-flash", **tokens)
    assert deepseek < gemini


def test_all_none_tokens_returns_zero():
    cost = compute_cost(
        model=_FLASH,
        input_tokens=None,
        output_tokens=None,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    assert cost == 0.0


def test_none_in_one_field_treated_as_zero():
    # Only output_tokens populated; the three None fields contribute 0.
    cost = compute_cost(
        model=_FLASH,
        input_tokens=None,
        output_tokens=500,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    # 500 × $2.50 / 1M = 0.00125
    assert cost == pytest.approx(0.00125)
