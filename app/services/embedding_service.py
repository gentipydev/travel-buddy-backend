"""Cohere-backed vector search and reranking for the RAG pipeline.

`search_chunks` embeds the user query (with optional conversational memory
prepended) and runs a cosine-distance lookup against the HNSW index on
`chunks.embedding`. `rerank_chunks` then re-scores the top embedding hits
with Cohere's reranker. Both fail soft — the chat path keeps working with
empty / fallback retrieval so a Cohere outage doesn't take down chat.
"""
import logging

import cohere
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Chunk


logger = logging.getLogger(__name__)

_cohere_client: cohere.AsyncClientV2 | None = None

# Cap each prepended assistant message before concatenating into the embed
# input so a long itinerary reply can't push the actual user query past
# Cohere v3's 512-token-per-input limit.
_MEMORY_CHAR_CAP = 200


def _get_cohere() -> cohere.AsyncClientV2:
    """Lazy singleton — the v2 client wraps a shared httpx.AsyncClient and is
    safe to reuse across requests inside FastAPI's single event loop."""
    global _cohere_client
    if _cohere_client is None:
        if not settings.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY is not set")
        _cohere_client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
    return _cohere_client


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


async def search_chunks(
    query: str,
    db: AsyncSession,
    *,
    top_k: int = 10,
    conversation_history: list[dict] | None = None,
) -> list[Chunk]:
    """Embed `query` with Cohere (input_type=search_query, the v3 counterpart
    to the search_document inputs used at indexing time) and return the
    top_k chunks ranked by cosine distance against the HNSW index.

    When `conversation_history` is provided, the last 2 assistant messages
    are prepended to the query before embedding so that follow-ups like
    "what about its castle?" stay anchored to the previously discussed place.

    Chunks without an embedding are excluded so a half-finished embed run
    cannot poison results. On Cohere API failure, returns [] and logs —
    the caller should treat that as "retrieval unavailable" and continue.
    """
    if not query or not query.strip():
        return []

    embed_input = _build_embed_input(query, conversation_history)

    co = _get_cohere()
    try:
        response = await co.embed(
            model=settings.embedding_model,
            texts=[embed_input],
            input_type="search_query",
            embedding_types=["float"],
        )
    except Exception:
        logger.exception("Cohere embed failed")
        return []
    query_vector = response.embeddings.float_[0]

    result = await db.execute(
        select(Chunk)
        .where(Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(top_k)
    )
    return list(result.scalars().all())


async def rerank_chunks(
    query: str,
    chunks: list[Chunk],
    *,
    top_n: int = 3,
) -> tuple[list[Chunk], float | None]:
    """Re-score `chunks` against `query` with Cohere rerank-multilingual-v3.0
    and return (top_n chunks, top relevance_score). We map response.results
    back to the original Chunk objects via `.index` so the caller still has
    access to place_id / metadata.

    On Cohere failure, falls back to (chunks[:top_n], None) so chat keeps
    working with embed-only ranking.
    """
    if not chunks:
        return [], None

    co = _get_cohere()
    try:
        response = await co.rerank(
            model=settings.rerank_model,
            query=query,
            documents=[c.content for c in chunks],
            top_n=min(top_n, len(chunks)),
        )
    except Exception:
        logger.exception("Cohere rerank failed; falling back to embed-only order")
        return chunks[:top_n], None

    ranked = [chunks[r.index] for r in response.results]
    # getattr guards against future Cohere SDK shape changes — telemetry must
    # never break the chat path.
    top_score = (
        getattr(response.results[0], "relevance_score", None)
        if response.results
        else None
    )
    return ranked, top_score
