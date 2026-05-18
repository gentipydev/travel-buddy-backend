"""Cohere-backed reranking of vector-search hits.

`rerank_chunks` re-scores the top embedding hits from `search_chunks`
with Cohere's rerank-multilingual model and returns the top_n chunks
plus the top relevance score (used for telemetry).

Fail-soft: on Cohere error, falls back to `chunks[:top_n]` with no
score, so chat continues with embed-only ranking.
"""
import logging
import time

from app.config import settings
from app.db.models import Chunk
from app.integrations.cohere_client import get_cohere

logger = logging.getLogger(__name__)


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

    co = get_cohere()
    started = time.perf_counter()
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

    rerank_ms = round((time.perf_counter() - started) * 1000.0, 2)
    ranked = [chunks[r.index] for r in response.results]
    # getattr guards against future Cohere SDK shape changes — telemetry must
    # never break the chat path.
    top_score = (
        getattr(response.results[0], "relevance_score", None)
        if response.results
        else None
    )
    logger.info(
        "cohere.rerank",
        extra={
            "model": settings.rerank_model,
            "candidates": len(chunks),
            "top_n": top_n,
            "duration_ms": rerank_ms,
            "top_score": top_score,
        },
    )
    return ranked, top_score
