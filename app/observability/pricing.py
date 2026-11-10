"""Per-model token pricing for `$/request` derivation on `RequestTrace`.

USD per 1M tokens. Refresh from https://www.anthropic.com/pricing whenever
Anthropic adjusts rates. `cache_creation` is the cache-WRITE cost (1.25x
uncached input); `cache_read` is the cheap path (0.1x) that prompt caching
exists to enable. Numbers reflect the default 5-minute cache-write tier; if
we ever opt into the 1-hour tier, add a second variant per model.

NOTE: these are estimates used only for the internal budget guards
(`app/domain/usage_budget.py`) and telemetry — NOT billing reconciliation.

Every model id reachable via `settings.model_*` MUST have a row here.
`compute_cost` returns 0.0 for an unknown id, which would silently disable
both the per-user daily budget and the global kill switch — so a model swap
and a row here belong in the same commit. Sonnet 5 is listed as the obvious
MODEL_CHAT upgrade path so an override doesn't fall through that hole.
"""

_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cache_creation": 1.25,
        "cache_read": 0.10,
    },
    "claude-sonnet-5": {
        "input": 2.00,
        "output": 10.00,
        "cache_creation": 2.50,
        "cache_read": 0.20,
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
    cache fields are unset when caching isn't engaged on that turn).
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
