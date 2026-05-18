"""Cohere AsyncClient singleton.

One client shared across the RAG pipeline (`app/rag/embedding.py` and
`app/rag/reranking.py`). The v2 client wraps a single `httpx.AsyncClient`
and is safe to reuse across requests inside FastAPI's event loop, so
holding it as a process-level singleton both saves connect time and keeps
the underlying connection pool warm.
"""
import cohere

from app.config import settings

_cohere_client: cohere.AsyncClientV2 | None = None


def get_cohere() -> cohere.AsyncClientV2:
    """Return the shared Cohere v2 client, instantiating it on first use.

    Raises RuntimeError when `COHERE_API_KEY` is not configured — the RAG
    layer catches this and short-circuits to "retrieval unavailable" so
    the chat path keeps working.
    """
    global _cohere_client
    if _cohere_client is None:
        if not settings.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY is not set")
        _cohere_client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
    return _cohere_client
