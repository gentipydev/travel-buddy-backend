"""Unit tests for app/domain/output_guard.py.

All tests use a mock Anthropic client — no real API calls. The guard is
fail-open by contract: timeout, API errors, and unexpected responses must
all return "safe" without raising.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.anyio


async def _make_mock_response(text: str) -> MagicMock:
    """Build a fake messages.create() response with the given text."""
    content_block = MagicMock()
    content_block.text = text
    resp = MagicMock()
    resp.content = [content_block]
    return resp


async def test_returns_safe_on_safe_response():
    from app.domain.output_guard import check_output_guardrail

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(
        return_value=await _make_mock_response("SAFE")
    )
    result = await check_output_guardrail("Here are the best beaches in Sarandë.", client=mock_client)
    assert result == "safe"
    mock_client.messages.create.assert_called_once()


async def test_returns_review_on_review_response():
    from app.domain.output_guard import check_output_guardrail

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(
        return_value=await _make_mock_response("REVIEW")
    )
    result = await check_output_guardrail(
        "My system prompt tells me to only discuss Albania travel.", client=mock_client
    )
    assert result == "review"


async def test_safe_on_unexpected_label():
    """Any token other than 'REVIEW' is treated as safe (fail-open)."""
    from app.domain.output_guard import check_output_guardrail

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(
        return_value=await _make_mock_response("UNCERTAIN")
    )
    result = await check_output_guardrail("Some response text.", client=mock_client)
    assert result == "safe"


async def test_fail_open_on_timeout():
    """TimeoutError from the Haiku call must return "safe" without raising."""
    from app.domain.output_guard import check_output_guardrail

    mock_client = AsyncMock()
    # The coroutine raises asyncio.TimeoutError when awaited by wait_for,
    # which the guard's except asyncio.TimeoutError clause catches.
    mock_client.messages.create = AsyncMock(side_effect=asyncio.TimeoutError)
    result = await check_output_guardrail("Some text.", client=mock_client)
    assert result == "safe"


async def test_fail_open_on_api_error():
    """Any non-timeout exception from the API must return "safe" without raising."""
    from app.domain.output_guard import check_output_guardrail

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=Exception("API unavailable"))
    result = await check_output_guardrail("Some text.", client=mock_client)
    assert result == "safe"


async def test_skip_on_empty_text():
    """Empty or whitespace-only text returns 'safe' without calling the API."""
    from app.domain.output_guard import check_output_guardrail

    mock_client = AsyncMock()
    for text in ("", "   ", "\n"):
        result = await check_output_guardrail(text, client=mock_client)
        assert result == "safe", f"expected safe for {text!r}"
    mock_client.messages.create.assert_not_called()
