"""Settings.validate_required_keys contract tests.

Pydantic already rejects literally-missing required fields at construction.
The added check covers the present-but-empty case (`GEMINI_API_KEY=""`
in `.env`) which would otherwise pass Settings construction and fail at
first LLM call. Surface the misconfiguration at startup instead.
"""
from __future__ import annotations

from app.config import Settings


def _settings_kwargs(**overrides):
    """Construct a Settings with all truthy-required keys filled in; tests
    override one or more keys to simulate empty/missing values."""
    defaults = dict(
        gemini_api_key="test-key",
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


def test_validate_required_keys_flags_empty_gemini_key():
    s = Settings(**_settings_kwargs(gemini_api_key=""))
    assert s.validate_required_keys() == ["gemini_api_key"]


def test_validate_required_keys_reports_multiple_at_once():
    s = Settings(**_settings_kwargs(gemini_api_key="", supabase_url=""))
    missing = s.validate_required_keys()
    assert "gemini_api_key" in missing
    assert "supabase_url" in missing
    assert len(missing) == 2
