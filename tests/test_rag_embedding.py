"""Unit tests for hybrid (vector + lexical) chunk search.

The vector arm's Cohere call is faked; the lexical arm's SQL is never executed
(the FakeSession just records the statement and serves a queued result), so
these exercise the routing / fusion / fail-soft logic without Postgres or a
live Cohere key.
"""
import uuid

import pytest

from app.db.models import Chunk
from app.rag import embedding as embedding_module
from app.rag.embedding import (
    _reciprocal_rank_fusion,
    _significant_terms,
    search_chunks,
)

pytestmark = pytest.mark.anyio


def _chunk(content: str) -> Chunk:
    return Chunk(id=uuid.uuid4(), content=content)


class _FakeEmbeddings:
    def __init__(self, vec):
        self.float_ = [vec]


class _FakeEmbedResponse:
    def __init__(self, vec):
        self.embeddings = _FakeEmbeddings(vec)


class _FakeCohere:
    """Minimal stand-in for cohere.AsyncClientV2 covering only `.embed`."""

    def __init__(self, *, raise_embed: bool = False):
        self._raise = raise_embed

    async def embed(self, **kwargs):
        if self._raise:
            raise RuntimeError("cohere down")
        return _FakeEmbedResponse([0.0] * 1024)


def _patch_cohere(monkeypatch, *, raise_embed: bool = False) -> None:
    monkeypatch.setattr(
        embedding_module, "get_cohere",
        lambda: _FakeCohere(raise_embed=raise_embed),
    )


# ---------------------------------------------------------------------- #
# _significant_terms (lexical query builder)                             #
# ---------------------------------------------------------------------- #


def test_significant_terms_keeps_proper_noun_drops_question_words():
    """The whole point of the OR-rewrite: a natural-language question keeps only
    the discriminating token, not the function words that made websearch's AND
    match nothing."""
    assert _significant_terms("How do I get to Theth?") == ["theth"]


def test_significant_terms_keeps_multiple_content_words():
    assert _significant_terms("Berat castle history") == ["berat", "castle", "history"]


def test_significant_terms_lowercases_and_dedups_preserving_order():
    assert _significant_terms("Theth theth THETH guesthouse") == [
        "theth",
        "guesthouse",
    ]


def test_significant_terms_strips_punctuation_and_short_tokens():
    # "what's" → what (stopword) + s (too short); "the"/"over" stopwords.
    assert _significant_terms("What's the drive over Llogara Pass?") == [
        "drive",
        "llogara",
        "pass",
    ]


def test_significant_terms_all_stopwords_returns_empty():
    """No significant terms → `_lexical_search` returns [] without querying, so
    the arm contributes nothing rather than matching every chunk."""
    assert _significant_terms("how do I get to it") == []


def test_significant_terms_preserves_accented_proper_noun():
    assert _significant_terms("Tell me about Gjirokastër") == ["gjirokastër"]


# ---------------------------------------------------------------------- #
# RRF fusion (pure function)                                             #
# ---------------------------------------------------------------------- #


def test_rrf_promotes_chunk_present_in_both_lists():
    """A chunk that appears in both arms must outrank chunks in only one."""
    shared = _chunk("shared")
    vector_only = _chunk("vector only")
    lexical_only = _chunk("lexical only")

    # `shared` is rank 1 in vector and rank 1 in lexical → highest fused score.
    fused = _reciprocal_rank_fusion(
        [[vector_only, shared], [lexical_only, shared]], top_k=10
    )

    assert fused[0] is shared
    assert {c.id for c in fused} == {shared.id, vector_only.id, lexical_only.id}


def test_rrf_dedups_by_chunk_id_and_caps_top_k():
    a, b, c, d = (_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d"))
    fused = _reciprocal_rank_fusion([[a, b], [c, d]], top_k=2)
    assert len(fused) == 2
    # No duplicate ids even though both lists are concatenated internally.
    assert len({c.id for c in fused}) == 2


def test_rrf_empty_lists_return_empty():
    assert _reciprocal_rank_fusion([[], []], top_k=5) == []


# ---------------------------------------------------------------------- #
# search_chunks orchestration                                           #
# ---------------------------------------------------------------------- #


async def test_empty_query_short_circuits_without_cohere(monkeypatch, fake_db):
    # get_cohere must never be called for a blank query.
    def _boom():
        raise AssertionError("get_cohere should not be called for empty query")

    monkeypatch.setattr(embedding_module, "get_cohere", _boom)
    assert await search_chunks("   ", fake_db) == []


async def test_hybrid_fuses_vector_and_lexical(monkeypatch, fake_db):
    _patch_cohere(monkeypatch)
    vector_hits = [_chunk("Berat castle history")]
    lexical_hits = [_chunk("Theth guesthouse furgon schedule")]
    # FIFO: vector arm executes first, lexical second.
    fake_db.queue(vector_hits)
    fake_db.queue(lexical_hits)

    fused = await search_chunks("getting to Theth", fake_db, top_k=10)

    fused_ids = {c.id for c in fused}
    assert vector_hits[0].id in fused_ids
    assert lexical_hits[0].id in fused_ids
    # Two db.execute calls: one per arm.
    assert len(fake_db.executed) == 2


async def test_cohere_failure_degrades_to_lexical_only(monkeypatch, fake_db):
    _patch_cohere(monkeypatch, raise_embed=True)
    lexical_hits = [_chunk("Theth guesthouse")]
    # Vector arm raises before it touches the db, so the only queued result
    # is consumed by the lexical arm.
    fake_db.queue(lexical_hits)

    fused = await search_chunks("Theth", fake_db)

    assert [c.id for c in fused] == [lexical_hits[0].id]
    # Only the lexical SELECT ran; the vector arm short-circuited on the embed.
    assert len(fake_db.executed) == 1


async def test_lexical_failure_degrades_to_vector_only(monkeypatch):
    _patch_cohere(monkeypatch)
    vector_hits = [_chunk("Berat castle history")]

    class _Result:
        def __init__(self, items):
            self._items = items

        def scalars(self):
            return self

        def all(self):
            return list(self._items)

    class _SecondCallFailsSession:
        """Vector SELECT succeeds; the lexical SELECT raises a DB error."""

        def __init__(self, first):
            self._first = first
            self._calls = 0

        async def execute(self, stmt):
            self._calls += 1
            if self._calls == 1:
                return _Result(self._first)
            raise RuntimeError("fts boom")

    fused = await search_chunks("Berat", _SecondCallFailsSession(vector_hits))

    assert [c.id for c in fused] == [vector_hits[0].id]
