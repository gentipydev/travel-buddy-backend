"""Unit tests for `retrieve_chunks` orchestration.

`search_chunks`, `rerank_chunks`, and `rewrite_query` are faked so these pin the
wiring without a DB or network:
  - routing skips retrieval for greetings,
  - `rerank` is driven by `settings.rerank_top_n` (the #2 knob),
  - the Haiku rewrite is gated by `settings.query_rewrite_enabled` (the #3 flag)
    and, when on, drives both search and rerank.
"""
import pytest

from app.rag import retrieval as retrieval_module
from app.rag.retrieval import needs_retrieval, retrieve_chunks, retrieve_context

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------------- #
# needs_retrieval routing (greetings / thanks / yes-no skip)                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "msg",
    [
        "hi", "hello", "Hey!", "thanks", "thank you", "Thank you.",
        "bye", "ok", "okay", "yes", "no", "sure", "great", "cool",
        "good morning", "Good evening!", "", "   ",
    ],
)
def test_needs_retrieval_skips_non_informational(msg):
    assert needs_retrieval(msg) is False


@pytest.mark.parametrize(
    "msg",
    [
        "Tell me about Theth",
        "Berat or Gjirokastër?",
        "What can I see in Vlorë?",
        "thanks for the tip, where is the castle?",  # not a pure thanks token
        "no buses to Theth?",  # informational despite leading "no"
    ],
)
def test_needs_retrieval_fires_for_informational(msg):
    assert needs_retrieval(msg) is True


def _wire(monkeypatch, *, candidates=None):
    """Install recording fakes for the three collaborators. Returns the dict
    the fakes write their observed call args into."""
    seen: dict = {}

    async def fake_search(query, db, *, conversation_history=None):
        seen["search_query"] = query
        seen["history"] = conversation_history
        return candidates if candidates is not None else ["chunk"]

    async def fake_rerank(query, chunks, *, top_n):
        seen["rerank_query"] = query
        seen["top_n"] = top_n
        return chunks[:top_n], 0.9

    monkeypatch.setattr(retrieval_module, "search_chunks", fake_search)
    monkeypatch.setattr(retrieval_module, "rerank_chunks", fake_rerank)
    return seen


async def test_greeting_skips_retrieval(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("greeting must not reach search/rewrite")

    monkeypatch.setattr(retrieval_module, "search_chunks", boom)
    monkeypatch.setattr(retrieval_module, "rewrite_query", boom)
    chunks, score = await retrieve_chunks("hello", db=None)
    assert chunks == []
    assert score is None


async def test_disabled_rewrite_uses_original_query_and_config_top_n(monkeypatch):
    monkeypatch.setattr(retrieval_module.settings, "query_rewrite_enabled", False)
    monkeypatch.setattr(retrieval_module.settings, "rerank_top_n", 4)
    seen = _wire(monkeypatch, candidates=["a", "b", "c", "d", "e"])

    def no_rewrite(*a, **k):
        raise AssertionError("rewrite_query must not run when disabled")

    monkeypatch.setattr(retrieval_module, "rewrite_query", no_rewrite)

    chunks, score = await retrieve_chunks("What can I see in Berat?", db=None)
    assert seen["search_query"] == "What can I see in Berat?"
    assert seen["rerank_query"] == "What can I see in Berat?"
    assert seen["top_n"] == 4
    assert len(chunks) == 4  # rerank truncates to top_n


async def test_enabled_rewrite_drives_search_and_rerank(monkeypatch):
    monkeypatch.setattr(retrieval_module.settings, "query_rewrite_enabled", True)
    monkeypatch.setattr(retrieval_module.settings, "rerank_top_n", 3)
    seen = _wire(monkeypatch)

    async def fake_rewrite(query, history):
        return f"REWRITTEN::{query}"

    monkeypatch.setattr(retrieval_module, "rewrite_query", fake_rewrite)

    await retrieve_chunks("What's worth visiting in Vlorë County?", db=None)
    assert seen["search_query"] == "REWRITTEN::What's worth visiting in Vlorë County?"
    assert seen["rerank_query"] == "REWRITTEN::What's worth visiting in Vlorë County?"


async def test_no_candidates_returns_empty_without_rerank(monkeypatch):
    monkeypatch.setattr(retrieval_module.settings, "query_rewrite_enabled", False)
    _wire(monkeypatch, candidates=[])

    def no_rerank(*a, **k):
        raise AssertionError("rerank must not run with no candidates")

    monkeypatch.setattr(retrieval_module, "rerank_chunks", no_rerank)
    chunks, score = await retrieve_chunks("Anything in Shkodër?", db=None)
    assert chunks == []
    assert score is None


async def test_retrieve_context_returns_none_when_both_arms_empty(monkeypatch):
    """When search yields nothing (both arms empty / Cohere down + no lexical),
    retrieve_context returns None so the caller proceeds ungrounded — not "" or
    a crash."""
    monkeypatch.setattr(retrieval_module.settings, "query_rewrite_enabled", False)
    _wire(monkeypatch, candidates=[])
    ctx = await retrieve_context("Anything in Pogradec?", db=None)
    assert ctx is None


async def test_retrieve_context_joins_chunk_contents(monkeypatch):
    """When chunks come back, retrieve_context joins their contents with the
    document separator."""
    from types import SimpleNamespace

    monkeypatch.setattr(retrieval_module.settings, "query_rewrite_enabled", False)
    monkeypatch.setattr(retrieval_module.settings, "rerank_top_n", 2)
    c1 = SimpleNamespace(content="Berat is a UNESCO town.")
    c2 = SimpleNamespace(content="Gjirokastër has a stone castle.")
    _wire(monkeypatch, candidates=[c1, c2])
    ctx = await retrieve_context("Tell me about Berat and Gjirokastër", db=None)
    assert ctx == "Berat is a UNESCO town.\n\n---\n\nGjirokastër has a stone castle."
