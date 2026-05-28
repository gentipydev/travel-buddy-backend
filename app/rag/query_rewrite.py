"""LLM query rewrite for the RAG retrieval pipeline.

`rewrite_query` runs a single Haiku pass that turns a raw user question into a
retrieval-optimized query before hybrid search + rerank:

  - resolves follow-up coreference using recent history ("its beaches" →
    "<place> beaches"), and
  - expands broad county/region/category questions into the specific towns and
    landmarks they cover ("Vlorë County" → "Sarandë, Ksamil, Himarë …") — which
    is what pulls a specific pin into the candidate pool when it would otherwise
    be crowded out by its county siblings.

It sits on the critical path in front of retrieval, so it is bounded by a short
timeout and is fail-soft on every path (empty input, timeout, API error, empty
response): any failure returns the original query unchanged, so retrieval
proceeds exactly as it would without the rewrite. Whether it runs at all is
gated by `settings.query_rewrite_enabled` in the caller (`retrieve_chunks`).
"""
from __future__ import annotations

import asyncio
import logging

from app.config import settings
from app.llm import client
from app.prompts import get as get_prompt

logger = logging.getLogger(__name__)

# Bounded because the rewrite blocks retrieval (and therefore time-to-first-
# token): a slow rewrite must degrade to the original query, not stall the turn.
_REWRITE_TIMEOUT_SECONDS = 4.0
_REWRITE_MAX_TOKENS = 128

# Recent turns fed to the rewrite for coreference resolution. Small + truncated
# so a long itinerary reply can't blow up the rewrite prompt.
_HISTORY_TURNS = 4
_HISTORY_CHAR_CAP = 300


def _build_user_content(query: str, history: list[dict] | None) -> str:
    """Render the rewrite input: recent conversation (for coreference) followed
    by the question to rewrite. Falls back to the bare question when there is no
    usable string-content history."""
    if not history:
        return f"Question to rewrite: {query}"
    lines: list[str] = []
    for m in history[-_HISTORY_TURNS:]:
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content[:_HISTORY_CHAR_CAP]}")
    if not lines:
        return f"Question to rewrite: {query}"
    return "Recent conversation:\n" + "\n".join(lines) + f"\n\nQuestion to rewrite: {query}"


async def rewrite_query(query: str, history: list[dict] | None = None) -> str:
    """Return a retrieval-optimized rewrite of `query`, or `query` unchanged on
    any failure. Never raises."""
    if not query or not query.strip():
        return query
    try:
        system_prompt = get_prompt("query_rewrite.system").text
        response = await asyncio.wait_for(
            client.messages.create(
                model=settings.model_chat,
                max_tokens=_REWRITE_MAX_TOKENS,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": _build_user_content(query, history)}
                ],
            ),
            timeout=_REWRITE_TIMEOUT_SECONDS,
        )
        rewritten = "".join(
            b.text for b in response.content if b.type == "text"
        ).strip()
    except TimeoutError:
        logger.warning(
            "query_rewrite: Haiku call timed out after %.1fs — using original",
            _REWRITE_TIMEOUT_SECONDS,
        )
        return query
    except Exception:
        logger.warning("query_rewrite: exception — using original", exc_info=True)
        return query
    if not rewritten:
        return query
    logger.info(
        "rag.query_rewrite",
        extra={"original_chars": len(query), "rewritten_chars": len(rewritten)},
    )
    return rewritten
