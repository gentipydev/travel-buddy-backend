"""Cohere-backed query embedding + vector search.

`search_chunks` embeds the user's query (optionally prepended with the
last two assistant turns for conversational continuity) and runs a
cosine-distance lookup against the HNSW index on `chunks.embedding`.

Fail-soft: any Cohere error returns `[]` so a Cohere outage doesn't take
down /chat/stream — the agent continues with no retrieved context.
"""
import logging
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Chunk
from app.integrations.cohere_client import get_cohere

logger = logging.getLogger(__name__)


# Cap each prepended assistant message before concatenating into the embed
# input so a long itinerary reply can't push the actual user query past
# Cohere v3's 512-token-per-input limit.
_MEMORY_CHAR_CAP = 200


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

    co = get_cohere()
    started = time.perf_counter()
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
