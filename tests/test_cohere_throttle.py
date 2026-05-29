"""Tests for the Cohere rate-limit throttle + 429 telemetry counter.

`app/rag/cohere_throttle.py` paces embed + rerank under the trial-key
10/min ceiling (a per-process token bucket) and counts every 429 the
fail-soft RAG layer would otherwise swallow. These pin:
  - the token bucket admits the burst capacity immediately then paces,
  - capacity<=0 disables limiting (no ceiling),
  - the 429 counter increments via note_rate_limited(),
  - the embed + rerank call sites increment the counter on a Cohere 429
    and still fail soft (lexical-only / passthrough).
"""
from __future__ import annotations

import time
import uuid

import cohere
import pytest

from app.db.models import Chunk
from app.rag import cohere_throttle
from app.rag.cohere_throttle import (
    AsyncTokenBucket,
    note_rate_limited,
    rate_limited_count,
    reset_rate_limited_count,
)

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _reset_counter():
    reset_rate_limited_count()
    yield
    reset_rate_limited_count()


# --------------------------------------------------------------------------- #
# AsyncTokenBucket                                                             #
# --------------------------------------------------------------------------- #


async def test_bucket_admits_burst_up_to_capacity_without_blocking():
    """Capacity tokens are available immediately at construction — a burst up
    to `capacity` must not sleep."""
    bucket = AsyncTokenBucket(capacity=5, window_seconds=60.0)
    started = time.monotonic()
    for _ in range(5):
        await bucket.acquire()
    elapsed = time.monotonic() - started
    # Five immediate acquisitions should be effectively instant.
    assert elapsed < 0.1


async def test_bucket_paces_once_capacity_exhausted():
    """The (capacity+1)th acquire must wait for a refill. With a fast refill
    window the wait is short but measurably > 0."""
    # capacity 1 over 0.1s → ~10 tokens/sec; 2nd acquire waits ~0.1s.
    bucket = AsyncTokenBucket(capacity=1, window_seconds=0.1)
    await bucket.acquire()  # consumes the only token immediately
    started = time.monotonic()
    await bucket.acquire()  # must wait for refill
    elapsed = time.monotonic() - started
    assert elapsed > 0.01


async def test_bucket_capacity_zero_disables_limiting():
    """capacity<=0 means "no ceiling" — acquire is a no-op, never blocks."""
    bucket = AsyncTokenBucket(capacity=0, window_seconds=60.0)
    started = time.monotonic()
    for _ in range(100):
        await bucket.acquire()
    assert time.monotonic() - started < 0.1


# --------------------------------------------------------------------------- #
# 429 counter                                                                  #
# --------------------------------------------------------------------------- #


def test_note_rate_limited_increments_counter():
    assert rate_limited_count() == 0
    note_rate_limited()
    note_rate_limited()
    assert rate_limited_count() == 2


# --------------------------------------------------------------------------- #
# Integration with the embed + rerank call sites                              #
# --------------------------------------------------------------------------- #


def _chunk(content: str) -> Chunk:
    return Chunk(id=uuid.uuid4(), content=content)


def _make_429() -> cohere.TooManyRequestsError:
    # The SDK error constructor signature varies; build via __new__ to avoid
    # coupling to it — we only need an instance of the right type to raise.
    return cohere.TooManyRequestsError.__new__(cohere.TooManyRequestsError)


async def test_embed_429_counts_and_degrades_to_lexical_only(monkeypatch, fake_db):
    from app.rag import embedding as embedding_module
    from app.rag.embedding import search_chunks

    class _RateLimitedCohere:
        async def embed(self, **kwargs):
            raise _make_429()

    monkeypatch.setattr(
        embedding_module, "get_cohere", lambda: _RateLimitedCohere()
    )
    lexical_hits = [_chunk("Theth guesthouse")]
    fake_db.queue(lexical_hits)

    fused = await search_chunks("Theth", fake_db)

    # Fail-soft preserved: lexical-only result, no crash.
    assert [c.id for c in fused] == [lexical_hits[0].id]
    # The 429 was counted — no longer invisible.
    assert rate_limited_count() == 1


async def test_rerank_429_counts_and_passes_through_top_n(monkeypatch):
    from app.rag import reranking as reranking_module
    from app.rag.reranking import rerank_chunks

    class _RateLimitedCohere:
        async def rerank(self, **kwargs):
            raise _make_429()

    monkeypatch.setattr(
        reranking_module, "get_cohere", lambda: _RateLimitedCohere()
    )
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    ranked, score = await rerank_chunks("q", chunks, top_n=2)

    # Fail-soft passthrough preserved.
    assert ranked == chunks[:2]
    assert score is None
    assert rate_limited_count() == 1


async def test_embed_call_site_awaits_the_limiter(monkeypatch, fake_db):
    """The embed path must call `cohere_limiter.acquire()` before the network
    call so a burst is actually paced."""
    from app.rag import embedding as embedding_module
    from app.rag.embedding import search_chunks

    acquired = {"count": 0}

    async def _fake_acquire():
        acquired["count"] += 1

    monkeypatch.setattr(cohere_throttle.cohere_limiter, "acquire", _fake_acquire)

    class _FakeEmbeddings:
        float_ = [[0.0] * 1024]

    class _FakeResp:
        embeddings = _FakeEmbeddings()

    class _OkCohere:
        async def embed(self, **kwargs):
            return _FakeResp()

    monkeypatch.setattr(embedding_module, "get_cohere", lambda: _OkCohere())
    fake_db.queue([_chunk("hit")])  # vector arm result
    fake_db.queue([])  # lexical arm

    await search_chunks("Berat", fake_db)
    assert acquired["count"] == 1
