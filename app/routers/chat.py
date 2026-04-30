import logging
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.supabase_auth import get_optional_user
from app.config import client, settings
from app.db.engine import get_session
from app.schemas import ChatRequest
from app.services.chat_pipeline import prepare_turn
from app.services.conversation_service import (
    append_messages,
    generate_title,
    maybe_summarize,
)
from app.services.telemetry import (
    RequestTrace,
    log_trace,
    new_request_id,
    utc_now_iso,
)
from app.sse import encode_sse_text


logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    supa_user: dict | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_session),
):
    request_id = new_request_id()
    started_at = time.perf_counter()
    timestamp = utc_now_iso()
    user_id = supa_user["id"] if supa_user else None
    message_length = len(request.message)

    ctx = await prepare_turn(request, supa_user, db)

    async def stream():
        input_tokens: int | None = None
        output_tokens: int | None = None
        assistant_text: str | None = None
        try:
            if ctx.new_conversation_id is not None:
                yield f"event: conversation_id\ndata: {ctx.new_conversation_id}\n\n"
            if ctx.place_card_payload is not None:
                yield f"event: place_card\ndata: {ctx.place_card_payload}\n\n"
            async with client.messages.stream(
                model=settings.model_chat,
                max_tokens=2048,
                system=ctx.system_prompt,
                messages=ctx.messages,
            ) as response:
                async for text in response.text_stream:
                    yield encode_sse_text(text)
                # Capture final message inside the context manager so the
                # accumulated state is still alive. Wrapped — token-count or
                # SDK-shape failures must not crash the request.
                try:
                    final = await response.get_final_message()
                    input_tokens = final.usage.input_tokens
                    output_tokens = final.usage.output_tokens
                    assistant_text = "".join(
                        b.text for b in final.content if b.type == "text"
                    )
                except Exception:
                    input_tokens = None
                    output_tokens = None
                    assistant_text = None

            # Persist BEFORE yielding [DONE]. The Flutter SSE parser closes the
            # connection on [DONE], and Starlette cancels the still-running
            # generator at its next await point. In Python 3.12 CancelledError
            # is BaseException, so `except Exception` cannot catch it — the
            # commit silently never lands. Doing the write here guarantees
            # durability before the client can disconnect. Tokens still stream
            # live during the loop above, so the user-visible latency added
            # is only the DB write (~50–150ms typical).
            if ctx.conversation is not None and assistant_text:
                try:
                    await append_messages(
                        db,
                        ctx.conversation,
                        request.message,
                        assistant_text,
                        images=request.images,
                    )
                    if ctx.new_conversation_id is not None:
                        await generate_title(
                            db,
                            ctx.conversation,
                            client,
                            settings.model_chat,
                            request.message,
                            assistant_text,
                        )
                    await maybe_summarize(
                        db, ctx.conversation, client, settings.model_chat
                    )
                except Exception:
                    logger.exception("conversation persist failed")

            yield "data: [DONE]\n\n"

            log_trace(RequestTrace(
                request_id=request_id,
                timestamp=timestamp,
                user_id=user_id,
                message_length=message_length,
                retrieval_used=ctx.retrieval_used,
                chunks_returned=ctx.chunks_returned,
                top_rerank_score=ctx.top_rerank_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                model=settings.model_chat,
                error=None,
            ))
        except Exception as e:
            log_trace(RequestTrace(
                request_id=request_id,
                timestamp=timestamp,
                user_id=user_id,
                message_length=message_length,
                retrieval_used=ctx.retrieval_used,
                chunks_returned=ctx.chunks_returned,
                top_rerank_score=ctx.top_rerank_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                model=settings.model_chat,
                error=str(e),
            ))
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
