"""Per-model token pricing for `$/request` derivation on `RequestTrace`.

USD per 1M tokens. Refresh from https://ai.google.dev/gemini-api/docs/pricing
whenever Google adjusts rates. `cache_read` is the cheap cached-input path
(implicit caching on Gemini 2.5+/3). Gemini has no separate cache-write
("cache_creation") billing on implicit caching, so `cache_creation` mirrors
the input rate and is never charged separately by `extract_usage` (which
always reports 0 cache-creation tokens).

NOTE: these are estimates used only for the internal budget guards
(`app/domain/usage_budget.py`) and telemetry — NOT billing reconciliation.
Verify exact Gemini 3.5 Flash numbers against the live pricing page; an
unknown/incorrect model id returns 0.0 (see `compute_cost`).
"""

_PRICING: dict[str, dict[str, float]] = {
    "gemini-3.5-flash": {
        "input": 0.30,
        "output": 2.50,
        "cache_creation": 0.30,
        "cache_read": 0.075,
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 2.50,
        "cache_creation": 0.30,
        "cache_read": 0.075,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.10,
        "output": 0.40,
        "cache_creation": 0.10,
        "cache_read": 0.025,
    },
}


def compute_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_creation_input_tokens: int | None,
    cache_read_input_tokens: int | None,
) -> float:
    """Return USD cost for one Gemini call.

    Unknown model returns 0.0 — telemetry must never crash, and an
    unknown SKU is operational signal that this table needs an update,
    not a request failure. `None` token fields are treated as 0 (cache
    fields are unset when caching isn't engaged on that turn).
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
