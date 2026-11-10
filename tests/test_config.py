"""Settings.validate_required_keys contract tests.

Pydantic already rejects literally-missing required fields at construction.
The added check covers the present-but-empty case (`ANTHROPIC_API_KEY=""`
in `.env`) which would otherwise pass Settings construction and fail at
first LLM call. Surface the misconfiguration at startup instead.
"""
from __future__ import annotations

from app.config import Settings


def _settings_kwargs(**overrides):
    """Construct a Settings with all truthy-required keys filled in; tests
    override one or more keys to simulate empty/missing values."""
    defaults = dict(
        anthropic_api_key="test-key",
        supabase_url="https://x.supabase.co",
        supabase_anon_key="anon",
        supabase_service_key="svc",
        database_url="postgresql+asyncpg://x@y/z",
    )
    defaults.update(overrides)
    return defaults


def test_validate_required_keys_passes_when_all_set():
    s = Settings(**_settings_kwargs())
    assert s.validate_required_keys() == []


def test_validate_required_keys_flags_empty_anthropic_key():
    s = Settings(**_settings_kwargs(anthropic_api_key=""))
    assert s.validate_required_keys() == ["anthropic_api_key"]


def test_validate_required_keys_reports_multiple_at_once():
    s = Settings(**_settings_kwargs(anthropic_api_key="", supabase_url=""))
    missing = s.validate_required_keys()
    assert "anthropic_api_key" in missing
    assert "supabase_url" in missing
    assert len(missing) == 2


def test_model_defaults_are_priced():
    """Every model id reachable via `settings.model_*` must have a row in the
    pricing table. `compute_cost` returns 0.0 for an unknown id, which would
    silently disable the per-user daily budget AND the global kill switch —
    so a model swap that skips `app/observability/pricing.py` must fail here,
    not in production."""
    from app.observability.pricing import compute_cost

    s = Settings(**_settings_kwargs())
    for field in (
        "model_chat",
        "model_persona_gate",
        "model_title_gen",
        "model_summarize",
    ):
        model = getattr(s, field)
        cost = compute_cost(
            model=model,
            input_tokens=1_000_000,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        assert cost > 0, f"{field}={model!r} has no row in pricing._PRICING"
