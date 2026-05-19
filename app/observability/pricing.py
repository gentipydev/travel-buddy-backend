"""Per-model token pricing for `$/request` derivation on `RequestTrace`.

USD per 1M tokens. Refresh from https://www.anthropic.com/pricing
whenever Anthropic adjusts rates. `cache_creation` is the write cost
(slightly more than uncached input); `cache_read` is the cheap path
that prompt-caching exists to enable.

Numbers reflect the 5-minute cache write tier (default). If we ever
opt into the 1-hour cache tier, add a second variant per model.
"""

# Pricing as of 2025 model launches. Verify against the Anthropic pricing
# page before relying on these for billing reconciliation.
_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_creation": 1.25,
        "cache_read": 0.10,
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_creation": 3.75,
        "cache_read": 0.30,
    },
    "claude-opus-4-7": {
        "input": 15.00,
        "output": 75.00,
        "cache_creation": 18.75,
        "cache_read": 1.50,
    },
}


def compute_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_creation_input_tokens: int | None,
    cache_read_input_tokens: int | None,
) -> float:
    """Return USD cost for one Anthropic call.

    Unknown model returns 0.0 — telemetry must never crash, and an
    unknown SKU is operational signal that this table needs an update,
    not a request failure. `None` token fields are treated as 0 (the
    Anthropic beta `Usage` shape leaves cache fields unset when caching
    isn't engaged on that turn).
    """
    rates = _PRICING.get(model)
    if not rates:
        return 0.0
    return (
        (input_tokens or 0) * rates["input"] / 1_000_000
        + (output_tokens or 0) * rates["output"] / 1_000_000
        + (cache_creation_input_tokens or 0)
        * rates["cache_creation"]
        / 1_000_000
        + (cache_read_input_tokens or 0) * rates["cache_read"] / 1_000_000
    )
