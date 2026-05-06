import json
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
from app.services.itinerary_tool import ITINERARY_TOOL
from app.services.itinerary_validator import validate_itinerary
from app.services.refine_tools import (
    REFINE_TOOL_NAMES,
    REFINE_TOOLS,
    apply_refine_tool,
)
from app.services.telemetry import (
    RequestTrace,
    log_trace,
    new_request_id,
    utc_now_iso,
)
from app.services.trips_service import _serialize, replace_trip_from_payload
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
        itinerary_emitted: bool = False
        itinerary_days: int | None = None
        itinerary_validation_error: str | None = None
        try:
            if ctx.new_conversation_id is not None:
                yield f"event: conversation_id\ndata: {ctx.new_conversation_id}\n\n"
            if ctx.place_card_payload is not None:
                yield f"event: place_card\ndata: {ctx.place_card_payload}\n\n"
            # Refine turns get the surgical patch tools too (update_stop,
            # delete_stop, …). Initial creation / open chat keeps just
            # emit_itinerary so a casual planner can't accidentally edit a
            # trip they aren't scoped to.
            available_tools = (
                [ITINERARY_TOOL, *REFINE_TOOLS]
                if ctx.refine_trip is not None
                else [ITINERARY_TOOL]
            )
            async with client.messages.stream(
                model=settings.model_chat,
                # 8192 fits ~12-day itineraries with 6 stops/day plus the
                # intro/outro paragraph. The previous 2048 cap was hitting
                # `stop_reason: max_tokens` mid-tool-call, leaving Claude's
                # `emit_itinerary` input with a missing `days` field and
                # triggering an `itinerary_invalid` ("days: Field required")
                # on what should have been a healthy plan. Real itineraries
                # rarely need more than this and we only pay for tokens
                # actually generated, so a higher ceiling is mostly free.
                # Refine patch tools are O(edit-size), so this ceiling no
                # longer scales with trip size for incremental edits.
                max_tokens=8192,
                system=ctx.system_prompt,
                messages=ctx.messages,
                tools=available_tools,
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
                    # Stream emit_itinerary tool_use blocks as structured SSE
                    # events, gated by a Pydantic validator. The Anthropic API
                    # does not enforce the tool's input_schema, so a missing
                    # field or out-of-enum kind would otherwise break the
                    # Flutter parser mid-stream. On validation failure we emit
                    # `event: itinerary_invalid` with a one-line reason instead
                    # of a corrupt itinerary; the natural-language reply
                    # already streamed and stays useful. Lands BEFORE [DONE]
                    # because we're still inside the `async with`; the response
                    # is alive and Starlette has not yet cancelled the
                    # generator. Wrapped in the same try as the token-count
                    # capture so a validator bug cannot crash the request.
                    # TODO: auto-persist the validated payload into the `trips`
                    # table once the schema and ownership model land.
                    tool_use_blocks = [
                        b for b in final.content if b.type == "tool_use"
                    ]
                    emit_block = next(
                        (b for b in tool_use_blocks if b.name == "emit_itinerary"),
                        None,
                    )
                    refine_blocks = [
                        b for b in tool_use_blocks if b.name in REFINE_TOOL_NAMES
                    ]

                    if emit_block is not None:
                        # Whole-trip rewrite path. If Claude also emitted patch
                        # tools in the same turn (rare; the prompt discourages
                        # it), we ignore them — emit_itinerary's full replace
                        # supersedes any narrow op.
                        block = emit_block
                        # Capture raw day count even on rejection — distinguishes
                        # "Claude tried a 5-day plan we rejected" from "Claude
                        # emitted no plan" in telemetry.
                        raw_days = (
                            block.input.get("days")
                            if isinstance(block.input, dict)
                            else None
                        )
                        if isinstance(raw_days, list):
                            itinerary_days = len(raw_days)
                        result = validate_itinerary(block.input)
                        if result.ok:
                            if ctx.refine_trip is not None:
                                # Refine path: apply the validated payload to
                                # the saved trip in-place and ship the canonical
                                # post-replace Trip back to the client. The
                                # inner try is required — the outer except
                                # below silently zeroes tokens, so a refine
                                # failure that fell through there would leave
                                # the client believing the refine succeeded.
                                # Commit BEFORE yielding `trip_refined` so the
                                # client's "this is the new state" trust is
                                # actually durable. The later `append_messages`
                                # path runs a second commit for the chat turn;
                                # if it fails we emit `persist_failed` (the
                                # trip stays refined — known split per plan).
                                try:
                                    updated = await replace_trip_from_payload(
                                        db,
                                        trip_id=ctx.refine_trip.id,
                                        user_id=ctx.refine_trip.user_id,
                                        payload=result.payload,
                                    )
                                    await db.commit()
                                    itinerary_emitted = True
                                    yield (
                                        "event: trip_refined\n"
                                        f"data: {json.dumps(_serialize(updated))}\n\n"
                                    )
                                except Exception:
                                    # Roll back any partial flush so the next
                                    # statement (e.g. append_messages) starts
                                    # in a clean transaction.
                                    await db.rollback()
                                    logger.exception("refine apply failed")
                                    yield (
                                        "event: refine_failed\n"
                                        "data: " + json.dumps(
                                            {"reason": "could not apply refinement"}
                                        ) + "\n\n"
                                    )
                            else:
                                itinerary_emitted = True
                                yield f"event: itinerary\ndata: {json.dumps(result.payload)}\n\n"
                        else:
                            itinerary_validation_error = result.error
                            # Surface raw input keys + stop_reason on the wire
                            # so the Flutter client logs can distinguish "Claude
                            # truncated mid-tool-call" (stop_reason=max_tokens,
                            # missing days/stops) from "Claude emitted shape we
                            # didn't expect" (stop_reason=tool_use, but with
                            # bad enum/format). Cheap diagnostic — always-on.
                            raw_keys = (
                                sorted(block.input.keys())
                                if isinstance(block.input, dict)
                                else None
                            )
                            error_payload = json.dumps({
                                "reason": result.error,
                                "stop_reason": getattr(final, "stop_reason", None),
                                "raw_keys": raw_keys,
                            })
                            yield f"event: itinerary_invalid\ndata: {error_payload}\n\n"
                            logger.warning(
                                "itinerary validation failed: %s | stop_reason=%s | raw=%s",
                                result.error,
                                getattr(final, "stop_reason", None),
                                json.dumps(result.raw)[:1500],
                            )
                    elif refine_blocks and ctx.refine_trip is not None:
                        # Patch-tool refine path. Apply every tool call in
                        # order within a single transaction; emit ONE
                        # `trip_refined` with the canonical post-state once
                        # all ops have landed. On any failure roll the whole
                        # turn back — never half-apply.
                        try:
                            latest = None
                            for block in refine_blocks:
                                raw_input = (
                                    block.input
                                    if isinstance(block.input, dict)
                                    else {}
                                )
                                latest = await apply_refine_tool(
                                    db,
                                    trip_id=ctx.refine_trip.id,
                                    user_id=ctx.refine_trip.user_id,
                                    name=block.name,
                                    raw_input=raw_input,
                                )
                            await db.commit()
                            itinerary_emitted = True
                            yield (
                                "event: trip_refined\n"
                                f"data: {json.dumps(_serialize(latest))}\n\n"
                            )
                        except Exception as e:
                            await db.rollback()
                            logger.exception("refine patch apply failed")
                            itinerary_validation_error = (
                                f"{type(e).__name__}: {e}"[:500]
                            )
                            yield (
                                "event: refine_failed\n"
                                "data: " + json.dumps(
                                    {"reason": "could not apply refinement"}
                                ) + "\n\n"
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
                    # Tokens already streamed live, but persistence failed —
                    # tell the client so it can mark the assistant message as
                    # "not saved" inline (no snackbar). Emit BEFORE [DONE]
                    # so the parser sees it before the stream-done sentinel.
                    yield "event: persist_failed\ndata: server_persist_failed\n\n"
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
                itinerary_emitted=itinerary_emitted,
                itinerary_days=itinerary_days,
                itinerary_validation_error=itinerary_validation_error,
                trip_refine=ctx.refine_trip is not None,
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
                itinerary_emitted=itinerary_emitted,
                itinerary_days=itinerary_days,
                itinerary_validation_error=itinerary_validation_error,
                trip_refine=ctx.refine_trip is not None,
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
