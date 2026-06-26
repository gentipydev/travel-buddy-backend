"""Unit tests for `app/llm/deepseek_adapter.py`.

The OpenAI/DeepSeek-side mirror of the Gemini adapter: message + tool
translation, the text-only image-drop, and usage mapping to the app's
four-field accounting. Pure functions — no network, no SDK.
"""
from types import SimpleNamespace

import pytest

from app.llm.deepseek_adapter import (
    extract_usage,
    to_openai_messages,
    to_openai_tools,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Tools.
# ---------------------------------------------------------------------------


def test_to_openai_tools_wraps_each_as_function_with_raw_schema():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    tools = to_openai_tools(
        [{"name": "emit_itinerary", "description": "d", "input_schema": schema}]
    )
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "emit_itinerary",
                "description": "d",
                # JSON Schema passes through verbatim as `parameters`.
                "parameters": schema,
            },
        }
    ]


def test_to_openai_tools_empty_and_missing_schema():
    assert to_openai_tools([]) == []
    # A tool with no input_schema still produces a valid (empty-object) params.
    out = to_openai_tools([{"name": "t"}])
    assert out[0]["function"]["parameters"] == {"type": "object"}
    assert out[0]["function"]["description"] == ""


# ---------------------------------------------------------------------------
# Messages + image drop.
# ---------------------------------------------------------------------------


def test_to_openai_messages_prepends_system_and_maps_roles():
    msgs, dropped = to_openai_messages(
        "SYS",
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "content": "x"},  # non-assistant → user
        ],
    )
    assert dropped is False
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert msgs[2] == {"role": "assistant", "content": "hello"}
    assert msgs[3] == {"role": "user", "content": "x"}


def test_to_openai_messages_drops_images_keeps_text_and_flags():
    msgs, dropped = to_openai_messages(
        "",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image", "source": {"url": "http://x/y.jpg"}},
                ],
            }
        ],
    )
    assert dropped is True
    # No system message (empty system), text preserved, image gone.
    assert msgs == [{"role": "user", "content": "look at this"}]


def test_to_openai_messages_skips_empty_after_image_drop():
    # An image-only message flattens to empty text → skipped, but the drop
    # flag still trips so telemetry records it.
    msgs, dropped = to_openai_messages(
        "SYS",
        [{"role": "user", "content": [{"type": "image", "source": {"url": "u"}}]}],
    )
    assert dropped is True
    assert msgs == [{"role": "system", "content": "SYS"}]


# ---------------------------------------------------------------------------
# Usage mapping.
# ---------------------------------------------------------------------------


def test_extract_usage_deepseek_cache_fields():
    u = SimpleNamespace(
        completion_tokens=200,
        prompt_cache_miss_tokens=100,
        prompt_cache_hit_tokens=300,
    )
    # (input, output, cache_creation, cache_read)
    assert extract_usage(u) == (100, 200, 0, 300)


def test_extract_usage_openai_shape_fallback():
    # No DeepSeek fields → fall back to prompt_tokens − cached.
    u = SimpleNamespace(
        completion_tokens=50,
        prompt_tokens=1000,
        prompt_tokens_details=SimpleNamespace(cached_tokens=400),
    )
    assert extract_usage(u) == (600, 50, 0, 400)


def test_extract_usage_none_is_all_zeros():
    assert extract_usage(None) == (0, 0, 0, 0)
