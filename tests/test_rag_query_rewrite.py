"""Unit tests for the Haiku query-rewrite pass.

The Anthropic client is faked, so these exercise the success path and the
fail-soft contract (empty input, timeout, API error, empty response all return
the original query) without a live key or network. The rewrite sits in front of
retrieval, so a failure must never raise — it degrades to the original query.
"""
import asyncio

import pytest

from app.rag import query_rewrite as qr_module
from app.rag.query_rewrite import _build_user_content, rewrite_query

pytestmark = pytest.mark.anyio


class _Block:
    def __init__(self, text: str, type: str = "text"):
        self.text = text
        self.type = type


class _Resp:
    def __init__(self, blocks):
        self.content = blocks


class _Messages:
    def __init__(self, *, blocks=None, raise_exc=None, sleep=None):
        self._blocks = blocks
        self._raise = raise_exc
        self._sleep = sleep
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._sleep is not None:
            await asyncio.sleep(self._sleep)
        if self._raise is not None:
            raise self._raise
        return _Resp(self._blocks or [])


class _Client:
    def __init__(self, messages: _Messages):
        self.messages = messages


def _patch_client(monkeypatch, messages: _Messages) -> None:
    monkeypatch.setattr(qr_module, "client", _Client(messages))


async def test_successful_rewrite_returns_model_text(monkeypatch):
    msgs = _Messages(blocks=[_Block("Sarandë Ksamil Himarë Vlorë Albania")])
    _patch_client(monkeypatch, msgs)
    out = await rewrite_query("What's worth visiting in Vlorë County?")
    assert out == "Sarandë Ksamil Himarë Vlorë Albania"
    # The configured model + the registry prompt were actually used.
    assert msgs.calls and msgs.calls[0]["model"]
    assert "system" in msgs.calls[0]


async def test_empty_query_short_circuits(monkeypatch):
    # Must not touch the client for an empty/whitespace query.
    msgs = _Messages(raise_exc=AssertionError("client should not be called"))
    _patch_client(monkeypatch, msgs)
    assert await rewrite_query("") == ""
    assert await rewrite_query("   ") == "   "
    assert msgs.calls == []


async def test_api_error_returns_original(monkeypatch):
    _patch_client(monkeypatch, _Messages(raise_exc=RuntimeError("anthropic down")))
    original = "What can I see around Korçë County?"
    assert await rewrite_query(original) == original


async def test_empty_response_returns_original(monkeypatch):
    # No text blocks (or whitespace-only) → fall back to the original query.
    _patch_client(monkeypatch, _Messages(blocks=[_Block("   ")]))
    original = "Tell me about Theth"
    assert await rewrite_query(original) == original


async def test_timeout_returns_original(monkeypatch):
    monkeypatch.setattr(qr_module, "_REWRITE_TIMEOUT_SECONDS", 0.05)
    _patch_client(monkeypatch, _Messages(blocks=[_Block("late")], sleep=0.5))
    original = "How do I get to Theth?"
    assert await rewrite_query(original) == original


def test_build_user_content_without_history():
    assert _build_user_content("hi there", None) == "Question to rewrite: hi there"


def test_build_user_content_with_history_resolves_reference():
    history = [
        {"role": "user", "content": "Tell me about Berat"},
        {"role": "assistant", "content": "Berat is a UNESCO old town..."},
    ]
    out = _build_user_content("what about its castle?", history)
    assert "Recent conversation:" in out
    assert "User: Tell me about Berat" in out
    assert "Assistant: Berat is a UNESCO old town..." in out
    assert out.endswith("Question to rewrite: what about its castle?")


def test_build_user_content_skips_nonstring_content():
    # Multimodal/non-string content (e.g. image blocks) is skipped; with no
    # usable string history we fall back to the bare question.
    history = [{"role": "user", "content": [{"type": "image"}]}]
    assert _build_user_content("q", history) == "Question to rewrite: q"
