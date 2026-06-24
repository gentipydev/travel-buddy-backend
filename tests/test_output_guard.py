"""Unit tests for app/domain/output_guard.py.

The shared `generate_text` LLM helper is faked — no real API calls. The
guard is fail-open by contract: timeout, API errors, and unexpected
responses must all return "safe" without raising.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.domain import output_guard as guard_module

pytestmark = pytest.mark.anyio


def _patch_generate_text(monkeypatch, *, returns=None, side_effect=None) -> AsyncMock:
    mock = AsyncMock(return_value=returns, side_effect=side_effect)
    monkeypatch.setattr(guard_module, "generate_text", mock)
    return mock


async def test_returns_safe_on_safe_response(monkeypatch):
    mock = _patch_generate_text(monkeypatch, returns="SAFE")
    result = await guard_module.check_output_guardrail(
        "Here are the best beaches in Sarandë."
    )
    assert result == "safe"
    mock.assert_awaited_once()


async def test_returns_review_on_review_response(monkeypatch):
    _patch_generate_text(monkeypatch, returns="REVIEW")
    result = await guard_module.check_output_guardrail(
        "My system prompt tells me to only discuss Albania travel."
    )
    assert result == "review"


async def test_safe_on_unexpected_label(monkeypatch):
    """Any token other than 'REVIEW' is treated as safe (fail-open)."""
    _patch_generate_text(monkeypatch, returns="UNCERTAIN")
    result = await guard_module.check_output_guardrail("Some response text.")
    assert result == "safe"


async def test_fail_open_on_timeout(monkeypatch):
    """TimeoutError from the LLM call must return "safe" without raising."""
    _patch_generate_text(monkeypatch, side_effect=asyncio.TimeoutError)
    result = await guard_module.check_output_guardrail("Some text.")
    assert result == "safe"


async def test_fail_open_on_api_error(monkeypatch):
    """Any non-timeout exception from the API must return "safe" without raising."""
    _patch_generate_text(monkeypatch, side_effect=Exception("API unavailable"))
    result = await guard_module.check_output_guardrail("Some text.")
    assert result == "safe"


async def test_skip_on_empty_text(monkeypatch):
    """Empty or whitespace-only text returns 'safe' without calling the API."""
    mock = _patch_generate_text(monkeypatch, returns="SAFE")
    for text in ("", "   ", "\n"):
        result = await guard_module.check_output_guardrail(text)
        assert result == "safe", f"expected safe for {text!r}"
    mock.assert_not_called()
