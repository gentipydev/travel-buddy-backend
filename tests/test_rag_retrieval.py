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
from app.rag.retrieval import retrieve_chunks

pytestmark = pytest.mark.anyio


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
