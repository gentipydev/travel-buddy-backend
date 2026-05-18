"""Streaming chat agent for /chat/stream.

`ChatAgent.stream()` is an async generator yielding SSE frames in this
order:

    [event: conversation_id]?
    [event: place_card]?
    data: <text chunks>...
    [event: itinerary | trip_refined | itinerary_invalid | refine_failed]?
    [event: persist_failed]?
    data: [DONE]\\n\\n

Critical invariant: every byte must hit the wire BEFORE `data: [DONE]`.
The Flutter parser closes the connection on `[DONE]`, and Starlette
cancels the still-running generator at the next `await` — in Python 3.12
that raises `CancelledError` (a `BaseException`, NOT an `Exception`), so
`except Exception` cannot catch it and a write that lands after `[DONE]`
silently never happens. The `append_messages` / `generate_title` /
`maybe_summarize` block must therefore complete (and emit
`persist_failed` on its own failure) before the final `[DONE]` yield.
"""
import json
import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.domain.chat_pipeline import ChatTurnContext
from app.domain.conversations import (
    append_messages,
    generate_title,
    maybe_summarize,
)
from app.domain.trips.mutations import replace_trip_from_payload
from app.domain.trips.repository import serialize_trip
from app.llm import (
    PROMPT_CACHING_BETA,
    assemble_chat_system,
    client,
    encode_sse_text,
)
from app.observability import (
    RequestTrace,
    log_trace,
    new_request_id,
    utc_now_iso,
)
from app.schemas import ChatRequest
from app.tools.itinerary_tool import ITINERARY_TOOL
from app.tools.itinerary_validator import validate_itinerary
from app.tools.refine import (
    REFINE_TOOL_NAMES,
    REFINE_TOOLS,
    apply_refine_tool,
)


logger = logging.getLogger(__name__)


class ChatAgent:
    """Streaming agent: one instance per /chat/stream request.

    The router instantiates with the request-scoped `db`, the assembled
    `ChatTurnContext`, the live `request`, and the resolved `user_id`
    (None for anonymous). `stream()` is the only entry point.
    """

    def __init__(
        self,
        *,
        db: AsyncSession,
        request: ChatRequest,
        ctx: ChatTurnContext,
        user_id: str | None,
    ):
        self.db = db
        self.request = request
        self.ctx = ctx
        self.user_id = user_id
        self.request_id = new_request_id()
        self.started_at = time.perf_counter()
        self.timestamp = utc_now_iso()
        self.message_length = len(request.message)

    async def stream(self):
        """Async generator yielding SSE frames. See module docstring for
        the strict frame ordering invariant."""
        db = self.db
        ctx = self.ctx
        request = self.request

        input_tokens: int | None = None
        output_tokens: int | None = None
        cache_creation_input_tokens: int | None = None
        cache_read_input_tokens: int | None = None
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
            # Prompt caching: the system payload is a 2-block list with the
            # ~3-4k-token stable prefix marked as a cache breakpoint and the
            # refine addendum + RAG context appended as an uncached suffix.
            # Beta endpoint is required in anthropic==0.39.0; cache token
            # accounting (`cache_creation_input_tokens` /
            # `cache_read_input_tokens`) ships back on `response.usage`.
            system_payload = assemble_chat_system(
                cached_prefix=ctx.cached_prefix,
                volatile_suffix=ctx.volatile_suffix,
            )
            async with client.beta.messages.stream(
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
                system=system_payload,
                messages=ctx.messages,
                tools=available_tools,
                betas=[PROMPT_CACHING_BETA],
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
                    # Cache tokens only exist on the beta Usage variant; pull
                    # them defensively so an SDK shape change can't crash the
                    # request. Cache hit rate = read / (read + creation + input).
                    cache_creation_input_tokens = getattr(
                        final.usage, "cache_creation_input_tokens", None
                    )
                    cache_read_input_tokens = getattr(
                        final.usage, "cache_read_input_tokens", None
                    )
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
                                        f"data: {json.dumps(await serialize_trip(db, updated))}\n\n"
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
                                f"data: {json.dumps(await serialize_trip(db, latest))}\n\n"
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
                    cache_creation_input_tokens = None
                    cache_read_input_tokens = None
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
                            settings.model_title_gen,
                            request.message,
                            assistant_text,
                        )
                    await maybe_summarize(
                        db, ctx.conversation, client, settings.model_summarize
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
                request_id=self.request_id,
                timestamp=self.timestamp,
                user_id=self.user_id,
                message_length=self.message_length,
                retrieval_used=ctx.retrieval_used,
                chunks_returned=ctx.chunks_returned,
                top_rerank_score=ctx.top_rerank_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(time.perf_counter() - self.started_at) * 1000,
                model=settings.model_chat,
                error=None,
                itinerary_emitted=itinerary_emitted,
                itinerary_days=itinerary_days,
                itinerary_validation_error=itinerary_validation_error,
                trip_refine=ctx.refine_trip is not None,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ))
        except Exception as e:
            log_trace(RequestTrace(
                request_id=self.request_id,
                timestamp=self.timestamp,
                user_id=self.user_id,
                message_length=self.message_length,
                retrieval_used=ctx.retrieval_used,
                chunks_returned=ctx.chunks_returned,
                top_rerank_score=ctx.top_rerank_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(time.perf_counter() - self.started_at) * 1000,
                model=settings.model_chat,
                error=str(e),
                itinerary_emitted=itinerary_emitted,
                itinerary_days=itinerary_days,
                itinerary_validation_error=itinerary_validation_error,
                trip_refine=ctx.refine_trip is not None,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ))
            yield f"event: error\ndata: {str(e)}\n\n"
