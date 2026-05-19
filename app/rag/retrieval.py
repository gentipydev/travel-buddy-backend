"""Adaptive routing + retrieval orchestration for the chat RAG pipeline.

Flow per turn:
    needs_retrieval(message) → search_chunks(query, history) → rerank_chunks
                                                              → top_n chunks

`retrieve_chunks` returns the chunk list (so callers can read place_id for
the place card); `retrieve_context` is a thin convenience wrapper that joins
the chunk contents into a single string.
"""
import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk
from app.observability import stage
from app.rag.embedding import search_chunks
from app.rag.reranking import rerank_chunks

logger = logging.getLogger(__name__)


# Skip retrieval only for non-informational turns. Word-count thresholds
# were rejected because legitimate Albania queries are routinely short
# ("Tell me about Theth", "Berat or Gjirokastër?").
_GREETING_RE = re.compile(
    r"^\s*("
    r"hi|hello|hey|hiya|yo|"
    r"thanks|thank\s*you|thx|cheers|"
    r"bye|goodbye|see\s*you|"
    r"ok|okay|yep|yes|no|nope|sure|great|cool|nice|awesome|"
    r"good\s*(morning|afternoon|evening|night)"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)


def needs_retrieval(message: str) -> bool:
    """Cheap heuristic: True unless the message is empty or a pure greeting/
    thanks/yes-no token. Errs toward retrieving — false positives just spend
    one Cohere embed call; false negatives leave Claude ungrounded."""
    if not message or not message.strip():
        return False
    return _GREETING_RE.match(message) is None


async def retrieve_chunks(
    query: str,
    db: AsyncSession,
    history: list[dict] | None = None,
) -> tuple[list[Chunk], float | None]:
    """Routing-aware retrieval. Returns ([], None) when retrieval is skipped,
    the embed call fails, or no chunks match — callers should treat empty as
    "no context available" and proceed normally. The float is the top Cohere
    rerank relevance score, surfaced for telemetry."""
    if not needs_retrieval(query):
        return [], None
    with stage("rag_retrieval"):
        candidates = await search_chunks(
            query, db, conversation_history=history
        )
    if not candidates:
        return [], None
    with stage("rag_rerank"):
        return await rerank_chunks(query, candidates)


async def retrieve_context(
    query: str,
    db: AsyncSession,
    history: list[dict] | None = None,
) -> str | None:
    """Convenience wrapper for callers that only need the joined string.
    `chat_stream` uses `retrieve_chunks` directly because it also needs the
    first chunk's place_id to build the place card."""
    chunks, _ = await retrieve_chunks(query, db, history)
    if not chunks:
        return None
    return "\n\n---\n\n".join(c.content for c in chunks)
