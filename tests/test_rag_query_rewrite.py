"""Unit tests for the LLM query-rewrite pass.

The shared `generate_text` helper is faked, so these exercise the success
path and the fail-soft contract (empty input, timeout, API error, empty
response all return the original query) without a live key or network. The
rewrite sits in front of retrieval, so a failure must never raise — it
degrades to the original query.
"""
import asyncio

import pytest

from app.rag import query_rewrite as qr_module
from app.rag.query_rewrite import _build_user_content, rewrite_query

pytestmark = pytest.mark.anyio


class _FakeGenerateText:
    """Stands in for `app.llm.generate_text`: records calls, returns a fixed
    string, or raises / sleeps to exercise the fail-soft paths."""

    def __init__(self, *, returns=None, raise_exc=None, sleep=None):
        self._returns = returns
        self._raise = raise_exc
        self._sleep = sleep
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self._sleep is not None:
            await asyncio.sleep(self._sleep)
        if self._raise is not None:
            raise self._raise
        return self._returns or ""


def _patch_generate_text(monkeypatch, fake: _FakeGenerateText) -> None:
    monkeypatch.setattr(qr_module, "generate_text", fake)


async def test_successful_rewrite_returns_model_text(monkeypatch):
    fake = _FakeGenerateText(returns="Sarandë Ksamil Himarë Vlorë Albania")
    _patch_generate_text(monkeypatch, fake)
    out = await rewrite_query("What's worth visiting in Vlorë County?")
    assert out == "Sarandë Ksamil Himarë Vlorë Albania"
    # The configured model + the registry prompt were actually used.
    assert fake.calls and fake.calls[0]["model"]
    assert "system" in fake.calls[0]


async def test_empty_query_short_circuits(monkeypatch):
    # Must not touch the LLM for an empty/whitespace query.
    fake = _FakeGenerateText(raise_exc=AssertionError("LLM should not be called"))
    _patch_generate_text(monkeypatch, fake)
    assert await rewrite_query("") == ""
    assert await rewrite_query("   ") == "   "
    assert fake.calls == []


async def test_api_error_returns_original(monkeypatch):
    _patch_generate_text(
        monkeypatch, _FakeGenerateText(raise_exc=RuntimeError("gemini down"))
    )
    original = "What can I see around Korçë County?"
    assert await rewrite_query(original) == original


async def test_empty_response_returns_original(monkeypatch):
    # Empty (or whitespace-only) text → fall back to the original query.
    _patch_generate_text(monkeypatch, _FakeGenerateText(returns="   "))
    original = "Tell me about Theth"
    assert await rewrite_query(original) == original


async def test_timeout_returns_original(monkeypatch):
    monkeypatch.setattr(qr_module, "_REWRITE_TIMEOUT_SECONDS", 0.05)
    _patch_generate_text(
        monkeypatch, _FakeGenerateText(returns="late", sleep=0.5)
    )
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
