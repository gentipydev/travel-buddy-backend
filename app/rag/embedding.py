"""Hybrid (semantic + lexical) chunk search for the RAG pipeline.

`search_chunks` runs two retrievers over the `chunks` table and fuses them:

  - **Vector** — embeds the query with Cohere (optionally prepended with the
    last two assistant turns for conversational continuity) and runs a
    cosine-distance lookup against the HNSW index on `chunks.embedding`.
  - **Lexical** — a Postgres full-text (`tsvector` / `ts_rank`) lookup against
    the generated `chunks.search_text` column (migration 016). Catches exact
    proper-noun matches ("Theth", "Gjirokastër", "furgon") that embedding
    search routinely misses.

The two ranked lists are merged with Reciprocal Rank Fusion (RRF) and the
top_k fused candidates are returned for reranking. Hybrid's job is *recall* of
the candidate pool; the Cohere reranker downstream (`app/rag/reranking.py`)
remains the precision filter.

Fail-soft: the lexical arm needs no external service, so even a total Cohere
outage degrades to lexical-only results instead of `[]` — strictly more
resilient than the previous vector-only path. The lexical arm is wrapped too,
so a DB-side FTS error degrades to vector-only. If both arms come back empty,
returns `[]` and the caller proceeds ungrounded.
"""
import logging
import re
import time

import cohere
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Chunk
from app.integrations.cohere_client import get_cohere
from app.rag.cohere_throttle import cohere_limiter, note_rate_limited

logger = logging.getLogger(__name__)


# Cap each prepended assistant message before concatenating into the embed
# input so a long itinerary reply can't push the actual user query past
# Cohere v3's 512-token-per-input limit.
_MEMORY_CHAR_CAP = 200

# RRF damping constant. 60 is the value from Cormack et al. and the de-facto
# default across search stacks; it flattens the contribution of any single
# list's top rank so neither retriever dominates the fused order.
_RRF_K = 60


def _build_embed_input(
    query: str, conversation_history: list[dict] | None
) -> str:
    """Prepend the last 2 string-content assistant messages to the query, most
    recent first: "{last_assistant} {second_last_assistant} {query}". Each
    prepended message is truncated to _MEMORY_CHAR_CAP. Multimodal/non-string
    assistant content is skipped."""
    if not conversation_history:
        return query
    assistants = [
        m["content"]
        for m in conversation_history
        if m.get("role") == "assistant" and isinstance(m.get("content"), str)
    ]
    if not assistants:
        return query
    recent = list(reversed(assistants[-2:]))  # most recent first
    truncated = [m[:_MEMORY_CHAR_CAP] for m in recent]
    return " ".join([*truncated, query])


async def _vector_search(
    query: str,
    db: AsyncSession,
    *,
    top_k: int,
    conversation_history: list[dict] | None,
) -> list[Chunk]:
    """Cohere-embedding cosine search against the HNSW index.

    Embeds with input_type=search_query (the v3 counterpart to the
    search_document inputs used at indexing time). Chunks without an embedding
    are excluded so a half-finished embed run cannot poison results. Returns
    [] on any Cohere failure so the hybrid caller can fall back to lexical-only.
    """
    embed_input = _build_embed_input(query, conversation_history)

    started = time.perf_counter()
    try:
        # get_cohere() is inside the guard so a missing COHERE_API_KEY (it
        # raises RuntimeError) degrades to lexical-only just like an embed
        # API failure does — the whole point of the hybrid fallback.
        co = get_cohere()
        # Pace under the trial-key 10/min ceiling so a burst can't trip a 429
        # that the fail-soft fallback would otherwise swallow silently.
        await cohere_limiter.acquire()
        response = await co.embed(
            model=settings.embedding_model,
            texts=[embed_input],
            input_type="search_query",
            embedding_types=["float"],
        )
    except cohere.TooManyRequestsError:
        # Count it before failing soft so the spike is visible in telemetry.
        note_rate_limited()
        logger.warning("Cohere embed rate-limited (429) — degrading to lexical-only")
        return []
    except Exception:
        logger.exception("Cohere embed failed")
        return []
    query_vector = response.embeddings.float_[0]
    embed_ms = round((time.perf_counter() - started) * 1000.0, 2)

    result = await db.execute(
        select(Chunk)
        .where(Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(top_k)
    )
    chunks = list(result.scalars().all())
    logger.info(
        "cohere.embed",
        extra={
            "model": settings.embedding_model,
            "input_chars": len(embed_input),
            "duration_ms": embed_ms,
            "top_k": top_k,
            "chunks_returned": len(chunks),
        },
    )
    return chunks


# Function/question words that carry no retrieval signal. The `simple` FTS
# config does no stopword removal, so we strip them here — otherwise every chat
# query would match every chunk on tokens like "to"/"the"/"about". Kept small
# and English-only on purpose: the corpus is English and the discriminating
# tokens we care about (proper nouns, "furgon", "unesco") are never in here.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "from", "by", "as", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "doing", "have", "has", "had", "you", "your",
    "yours", "we", "our", "they", "them", "their", "this", "that", "these",
    "those", "what", "whats", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "can", "could", "should", "would", "will", "shall", "may",
    "might", "must", "about", "into", "over", "out", "off", "get", "got",
    "getting", "going", "want", "need", "like", "tell", "show", "find", "there",
    "here", "any", "some", "all", "more", "most", "very", "just", "also",
    "than", "then", "too", "much", "many", "near", "around", "between", "good",
    "best", "worth", "see", "visit", "visiting", "go", "me", "my", "it", "its",
})

# Drop tokens shorter than this; together with _STOPWORDS it clears the
# low-signal residue ("i", "vs", "of"). Short proper nouns are rare enough in
# the Albania corpus that the recall cost is negligible.
_MIN_TERM_LEN = 3

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _significant_terms(query: str) -> list[str]:
    """Lowercased, de-duplicated content tokens for the lexical OR-query.

    Drops stopwords and very short tokens so the lexical arm fires on the
    *discriminating* words (proper nouns, domain terms) instead of requiring
    every query word the way `websearch_to_tsquery`'s AND-semantics did — under
    which "How do I get to Theth?" became `how & do & i & get & to & theth` and
    matched nothing. Order-preserving dedup keeps the tsquery compact.
    """
    seen: dict[str, None] = {}
    for tok in _WORD_RE.findall(query.lower()):
        if len(tok) >= _MIN_TERM_LEN and tok not in _STOPWORDS:
            seen.setdefault(tok, None)
    return list(seen)


async def _lexical_search(
    query: str,
    db: AsyncSession,
    *,
    top_k: int,
) -> list[Chunk]:
    """Postgres full-text search over the generated `chunks.search_text`.

    Builds an OR tsquery from the query's *significant* terms (see
    `_significant_terms`) rather than `websearch_to_tsquery`, whose AND-of-every-
    term semantics meant a natural-language question never matched a chunk unless
    it contained all of the question's words. OR + `ts_rank` (which rewards
    matching more distinct terms) recovers the proper-noun recall this arm exists
    for; the downstream RRF fuse + Cohere rerank stay the precision filter.

    Terms are clean `\\w+` tokens joined with ` | `, so the `to_tsquery` parse
    can't hit a syntax error or injection. `simple` config matches the indexing
    side (migration 016). Returns [] when the query has no significant terms or
    on any DB error, so a FTS problem still degrades to vector-only.
    """
    terms = _significant_terms(query)
    if not terms:
        return []
    tsquery = func.to_tsquery("simple", " | ".join(terms))
    try:
        result = await db.execute(
            select(Chunk)
            .where(Chunk.search_text.op("@@")(tsquery))
            .order_by(func.ts_rank(Chunk.search_text, tsquery).desc())
            .limit(top_k)
        )
        chunks = list(result.scalars().all())
    except Exception:
        logger.exception("lexical chunk search failed")
        return []
    logger.info(
        "chunks.lexical",
        extra={
            "query_chars": len(query),
            "terms": len(terms),
            "top_k": top_k,
            "chunks_returned": len(chunks),
        },
    )
    return chunks


def reciprocal_rank_fusion_scores(
    ranked_key_lists: list[list], *, k: int = _RRF_K
) -> dict:
    """Pure RRF over opaque, hashable keys: score(key) = Σ 1/(k + rank_i).

    No I/O, no Chunk dependency — takes ranked lists of any hashable key
    (chunk id, doc id, string) and returns `{key: fused_score}`. `rank_i`
    is 0-based per list, so a key ranked first in a list contributes
    `1/(k+1)`. A key appearing in multiple lists accumulates across them,
    which is the recall win hybrid retrieval is after. `k` damps the
    contribution of any single list's top rank (Cormack et al. default 60).

    Extracted so the fusion math is unit-testable in isolation from the
    embedding pipeline; `_reciprocal_rank_fusion` is the Chunk-aware caller.
    """
    scores: dict = {}
    for keys in ranked_key_lists:
        for rank, key in enumerate(keys):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]], *, top_k: int
) -> list[Chunk]:
    """Merge ranked Chunk lists via RRF and return the top_k chunks.

    Delegates the score math to the pure `reciprocal_rank_fusion_scores`,
    then dedups by `Chunk.id` (keeping the first-seen instance) and returns
    the top_k by fused score. A chunk that ranks well in both arms outscores
    one that only appears in a single arm.
    """
    scores = reciprocal_rank_fusion_scores(
        [[c.id for c in chunks] for chunks in ranked_lists], k=_RRF_K
    )
    chunk_by_id: dict = {}
    for chunks in ranked_lists:
        for chunk in chunks:
            chunk_by_id.setdefault(chunk.id, chunk)
    ranked = sorted(
        chunk_by_id.values(), key=lambda c: scores[c.id], reverse=True
    )
    return ranked[:top_k]


async def search_chunks(
    query: str,
    db: AsyncSession,
    *,
    top_k: int = 10,
    conversation_history: list[dict] | None = None,
) -> list[Chunk]:
    """Hybrid retrieval: fuse vector + lexical search into top_k candidates.

    Both arms fetch up to `top_k`; the fused result is also capped at `top_k`,
    so the downstream Cohere reranker sees the same candidate-pool size it did
    under the old vector-only path while gaining lexical recall.

    When `conversation_history` is provided, the last 2 assistant messages are
    prepended to the query before embedding (vector arm only) so follow-ups
    like "what about its castle?" stay anchored to the previously discussed
    place. Returns [] for an empty query or when both arms return nothing.
    """
    if not query or not query.strip():
        return []

    started = time.perf_counter()
    # Sequential, not concurrent: both arms share one AsyncSession, which is
    # not safe for overlapping statements. The extra GIN lookup is cheap next
    # to the Cohere embed + rerank network round-trips that dominate latency.
    vector = await _vector_search(
        query, db, top_k=top_k, conversation_history=conversation_history
    )
    lexical = await _lexical_search(query, db, top_k=top_k)
    fused = _reciprocal_rank_fusion([vector, lexical], top_k=top_k)

    logger.info(
        "rag.hybrid_search",
        extra={
            "query_chars": len(query),
            "vector_hits": len(vector),
            "lexical_hits": len(lexical),
            "fused_returned": len(fused),
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
        },
    )
    return fused
