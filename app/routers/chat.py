"""POST /chat/stream — thin HTTP shell around `ChatAgent`.

The agent loop (Anthropic streaming, tool-use dispatch, persistence,
telemetry) lives in `app/agents/chat_agent.py`. This router constructs a
`ChatAgent` and returns its `StreamingResponse`.

Keeping the SSE response headers here (X-Accel-Buffering, Cache-Control,
Connection) so they stay co-located with the FastAPI route definition.

The router owns `RequestTrace` construction (so `prepare_turn` already
runs inside the trace's `context_assembly` stage) and binds it to the
`trace_var` ContextVar so deep stages in `chat_pipeline` and `retrieval`
record into the same dict without function-signature plumbing.
"""
import logging
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.chat_agent import ChatAgent
from app.auth.supabase_auth import get_current_user
from app.config import settings
from app.db.engine import get_session
from app.domain.chat_pipeline import prepare_turn
from app.domain.usage_budget import (
    BudgetExceeded,
    acquire_stream_slot,
    check_global_ceiling,
    check_user_budget,
    global_over_ceiling,
    release_stream_slot,
    user_over_budget,
)
from app.llm import deepseek_enabled
from app.observability import (
    RequestTrace,
    new_request_id,
    trace_var,
    utc_now_iso,
)
from app.rate_limit import limiter, user_or_ip_key
from app.schemas import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# Two stacked limits, both fire on every request:
#   - per-IP: anti-curl-loop. Bounds anonymous abuse.
#   - per-user (falls back to IP when no Bearer token): per-day ceiling
#     is the real cost-runaway guard; per-minute keeps interactive UX OK.
# Picked from the engineering health roadmap (docs/07 item 1) — tune in
# rate_limit.py if real traffic shows different shape.
@router.post(
    "/chat/stream",
    responses={
        # Documented here so the generated OpenAPI lists 429 as a real
        # response shape rather than the FastAPI default 200-only schema.
        # The body shape comes from `app/rate_limit.py:rate_limit_exceeded_handler`
        # — `{"error": "rate_limited", "retry_after_seconds": int}`.
        429: {
            "description": (
                "Rate limit exceeded. The response includes a Retry-After "
                "header (seconds) and a JSON body the Flutter client parses "
                "into a `RateLimitedEvent` for inline UI feedback."
            ),
        },
        422: {"description": "Request body failed Pydantic validation"},
        401: {
            "description": (
                "Missing or invalid bearer token. Chat requires a signed-in "
                "user — anonymous calls would make per-user cost limits "
                "unenforceable."
            ),
        },
        503: {
            "description": (
                "Global daily spend ceiling reached "
                "(`{\"detail\": \"service_at_capacity\"}`). Resets at UTC "
                "midnight; Retry-After matches the ceiling cache window. "
                "The 429 shape is also returned for `daily_budget_exceeded` "
                "(per-user cost budget) and `concurrent_stream_limit` in "
                "addition to SlowAPI rate limits — same body contract."
            ),
        },
    },
)
@limiter.limit("30/minute", key_func=get_remote_address)
@limiter.limit("60/minute;200/day", key_func=user_or_ip_key)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    supa_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    user_id = supa_user["id"]
    # Cost gates run BEFORE prepare_turn, which already spends money
    # (Cohere embed/rerank, optional Haiku query rewrite). Order after
    # auth (401) and the SlowAPI decorators (429 rate): global ceiling
    # (503) → per-user budget (429) → concurrency slot (429).
    #
    # Trigger B (DeepSeek overflow): when a fallback is configured, an
    # over-budget turn SPILLS to the cheaper DeepSeek model instead of being
    # hard-blocked — reusing the same limits as the cutover. Global ceiling
    # takes precedence over the per-user budget as the spill reason. With no
    # fallback configured, the raising gates preserve today's hard-block.
    force_fallback_reason: str | None = None
    if deepseek_enabled():
        force_fallback_reason = await global_over_ceiling(db) or (
            await user_over_budget(db, user_id)
        )
    else:
        await check_global_ceiling(db)
        await check_user_budget(db, user_id)
    if not acquire_stream_slot(user_id):
        raise BudgetExceeded(
            detail="concurrent_stream_limit",
            status_code=429,
            retry_after_seconds=5,
        )
    started_at = time.perf_counter()
    # Construct the trace BEFORE prepare_turn so its inner stages
    # (context_assembly, rag_*) can append to stage_latencies. The
    # ctx-derived fields (retrieval_used, chunks_returned, etc.) and
    # the LLM-derived fields (tokens, latency, cost) are filled in by
    # ChatAgent before log_trace fires.
    trace = RequestTrace(
        request_id=new_request_id(),
        timestamp=utc_now_iso(),
        user_id=user_id,
        message_length=len(body.message),
        retrieval_used=False,
        chunks_returned=0,
        top_rerank_score=None,
        input_tokens=None,
        output_tokens=None,
        latency_ms=0.0,
        model=settings.model_chat,
        error=None,
    )
    trace_var.set(trace)
    # The slot is normally released in ChatAgent.stream()'s finally. If
    # prepare_turn raises (404 trip, DB error) the generator never runs,
    # so release here instead of leaking the slot until its TTL prune.
    try:
        ctx = await prepare_turn(body, supa_user, db)
        agent = ChatAgent(
            db=db,
            request=body,
            ctx=ctx,
            user_id=user_id,
            trace=trace,
            started_at=started_at,
            force_fallback_reason=force_fallback_reason,
        )
    except BaseException:
        release_stream_slot(user_id)
        raise
    return StreamingResponse(
        agent.stream(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
