"""Unit tests for Cohere reranking fail-soft behavior.

The Cohere `.rerank` call (and the `get_cohere()` client factory) are faked,
so these exercise the fallback / ordering logic without a live Cohere key or
network. The key regression guarded here: a missing COHERE_API_KEY (which makes
`get_cohere()` raise) must degrade `rerank_chunks` to embed-only order instead
of crashing the chat turn.
"""
import uuid

import pytest

from app.db.models import Chunk
from app.rag import reranking as reranking_module
from app.rag.reranking import rerank_chunks

pytestmark = pytest.mark.anyio


def _chunk(content: str) -> Chunk:
    return Chunk(id=uuid.uuid4(), content=content)


class _FakeResult:
    def __init__(self, index: int, relevance_score: float):
        self.index = index
        self.relevance_score = relevance_score


class _FakeRerankResponse:
    def __init__(self, results):
        self.results = results


class _FakeCohere:
    """Minimal stand-in for cohere.AsyncClientV2 covering only `.rerank`."""

    def __init__(self, *, raise_rerank: bool = False):
        self._raise = raise_rerank

    async def rerank(self, **kwargs):
        if self._raise:
            raise RuntimeError("cohere rerank down")
        # Reverse the candidate order so we can assert the mapping back to
        # the original Chunk objects via `.index` actually happens.
        docs = kwargs["documents"]
        top_n = kwargs["top_n"]
        results = [
            _FakeResult(index=i, relevance_score=1.0 - 0.1 * rank)
            for rank, i in enumerate(reversed(range(len(docs))))
        ]
        return _FakeRerankResponse(results[:top_n])


async def test_empty_chunks_short_circuits(monkeypatch):
    # get_cohere must never be called when there are no candidates.
    def _boom():
        raise AssertionError("get_cohere should not be called for empty chunks")

    monkeypatch.setattr(reranking_module, "get_cohere", _boom)
    ranked, score = await rerank_chunks("any query", [])
    assert ranked == []
    assert score is None


async def test_missing_api_key_degrades_to_embed_order(monkeypatch):
    # get_cohere() raises RuntimeError when COHERE_API_KEY is unset. The
    # reranker must catch it and fall back to chunks[:top_n] with no score.
    def _no_key():
        raise RuntimeError("COHERE_API_KEY is not set")

    monkeypatch.setattr(reranking_module, "get_cohere", _no_key)
    chunks = [_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d")]
    ranked, score = await rerank_chunks("query", chunks, top_n=3)
    assert ranked == chunks[:3]
    assert score is None


async def test_rerank_call_failure_degrades_to_embed_order(monkeypatch):
    # Regression guard for the pre-existing path: a failure inside `.rerank`
    # also falls back to embed-only order.
    monkeypatch.setattr(
        reranking_module, "get_cohere",
        lambda: _FakeCohere(raise_rerank=True),
    )
    chunks = [_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d")]
    ranked, score = await rerank_chunks("query", chunks, top_n=2)
    assert ranked == chunks[:2]
    assert score is None


async def test_successful_rerank_reorders_and_returns_top_score(monkeypatch):
    monkeypatch.setattr(
        reranking_module, "get_cohere",
        lambda: _FakeCohere(),
    )
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    ranked, score = await rerank_chunks("query", chunks, top_n=2)
    # Fake reverses order, so top result maps back to the last chunk.
    assert ranked == [chunks[2], chunks[1]]
    assert score == pytest.approx(1.0)
