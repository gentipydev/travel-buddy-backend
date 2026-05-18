"""POST /chat/stream — thin HTTP shell around `ChatAgent`.

The agent loop (Anthropic streaming, tool-use dispatch, persistence,
telemetry) lives in `app/agents/chat_agent.py`. This router constructs a
`ChatAgent` and returns its `StreamingResponse`.

Keeping the SSE response headers here (X-Accel-Buffering, Cache-Control,
Connection) so they stay co-located with the FastAPI route definition.
"""
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.chat_agent import ChatAgent
from app.auth.supabase_auth import get_optional_user
from app.db.engine import get_session
from app.domain.chat_pipeline import prepare_turn
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
@router.post("/chat/stream")
@limiter.limit("30/minute", key_func=get_remote_address)
@limiter.limit("60/minute;200/day", key_func=user_or_ip_key)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    supa_user: dict | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_session),
):
    ctx = await prepare_turn(body, supa_user, db)
    agent = ChatAgent(
        db=db,
        request=body,
        ctx=ctx,
        user_id=supa_user["id"] if supa_user else None,
    )
    return StreamingResponse(
        agent.stream(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
