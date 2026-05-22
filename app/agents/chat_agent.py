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
    assemble_chat_system,
    client,
    encode_sse_text,
)
from app.observability import (
    RequestTrace,
    chat_span,
    compute_cost,
    log_trace,
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
        trace: RequestTrace,
        started_at: float,
    ):
        self.db = db
        self.request = request
        self.ctx = ctx
        self.user_id = user_id
        # The router constructs the trace BEFORE prepare_turn so that
        # `context_assembly` / `rag_*` stages can record into the same
        # `stage_latencies` dict. ctx-derived fields are filled in here
        # because the router doesn't see the ChatTurnContext.
        self.trace = trace
        self.started_at = started_at
        self.request_id = trace.request_id
        self.timestamp = trace.timestamp
        self.message_length = trace.message_length
        trace.retrieval_used = ctx.retrieval_used
        trace.chunks_returned = ctx.chunks_returned
        trace.top_rerank_score = ctx.top_rerank_score
        trace.trip_refine = ctx.refine_trip is not None

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
        # Tool names Claude fired this turn — empty for plain chat,
        # `["emit_itinerary"]` for whole-trip create, one or more
        # `update_stop` / `insert_stop` / etc. for patch refine. Lands on
        # the Braintrust span so dashboards can filter/group by tool.
        tool_calls_for_span: list[str] = []
        # `chat_span` no-ops to a stand-in span when BRAINTRUST_ENABLED is
        # false — the same `span.log(...)` calls below work unchanged.
        # When enabled, every `client.beta.prompt_caching.messages.stream(...)`
        # inside the block auto-emits a child span via `wrap_anthropic` at the
        # client level, so the parent only logs request-scope context.
        with chat_span(name="chat_stream") as span:
            span.log(
                input=(
                    {
                        "message": request.message,
                        "images_count": len(request.images or []),
                    }
                    if settings.braintrust_log_user_payloads
                    else {
                        "message_length": self.message_length,
                        "images_count": len(request.images or []),
                    }
                ),
                metadata={
                    "request_id": self.request_id,
                    "user_id": self.user_id,
                    "model": settings.model_chat,
                    "trip_refine": ctx.refine_trip is not None,
                    "trip_id": (
                        str(ctx.refine_trip.id)
                        if ctx.refine_trip is not None
                        else None
                    ),
                    "place_card_emitted": ctx.place_card_payload is not None,
                    "retrieval_used": ctx.retrieval_used,
                    "chunks_returned": ctx.chunks_returned,
                    "top_rerank_score": ctx.top_rerank_score,
                    "message_length": self.message_length,
                },
            )
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
                # In anthropic==0.39.0 the streaming entrypoint for prompt
                # caching lives at `client.beta.prompt_caching.messages.stream`;
                # the `anthropic-beta: prompt-caching-2024-07-31` header is
                # injected by the SDK so we don't pass it ourselves. Cache token
                # accounting (`cache_creation_input_tokens` /
                # `cache_read_input_tokens`) ships back on `response.usage`.
                system_payload = assemble_chat_system(
                    cached_prefix=ctx.cached_prefix,
                    volatile_suffix=ctx.volatile_suffix,
                )
                # Manual stage timer (not `with self.trace.stage(...)`) so the
                # SSE-strict body inside the `async with` doesn't need to be
                # re-indented. Recorded on success after `async with` exits;
                # mid-stream exceptions fall through to the outer except and
                # the partial duration is captured there.
                _anthropic_start = time.perf_counter()
                async with client.beta.prompt_caching.messages.stream(
                    model=settings.model_chat,
                    # 8192 sits well above what the schema now allows
                    # (`MAX_STOPS_PER_DAY=3`, so ~12 days × 3 stops + the
                    # intro/outro paragraph). Sizing was originally picked
                    # for ~6 stops/day under the older prompt, but the
                    # headroom is still useful — refine patch tools are
                    # O(edit-size) and don't scale with trip size, and the
                    # previous 2048 cap was hitting `stop_reason: max_tokens`
                    # mid-tool-call, leaving `emit_itinerary` input with a
                    # missing `days` field and triggering `itinerary_invalid`
                    # ("days: Field required") on otherwise healthy plans.
                    # We only pay for tokens actually generated, so the
                    # ceiling is mostly free.
                    max_tokens=8192,
                    system=system_payload,
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
                        tool_calls_for_span = [b.name for b in tool_use_blocks]
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
                            _td_start = time.perf_counter()
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
                            self.trace.stage_latencies["tool_dispatch"] = (
                                self.trace.stage_latencies.get("tool_dispatch", 0.0)
                                + time.perf_counter() - _td_start
                            )
                        elif refine_blocks and ctx.refine_trip is not None:
                            # Patch-tool refine path. Apply every tool call in
                            # order within a single transaction; emit ONE
                            # `trip_refined` with the canonical post-state once
                            # all ops have landed. On any failure roll the whole
                            # turn back — never half-apply.
                            _td_start = time.perf_counter()
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
                            self.trace.stage_latencies["tool_dispatch"] = (
                                self.trace.stage_latencies.get("tool_dispatch", 0.0)
                                + time.perf_counter() - _td_start
                            )
                    except Exception:
                        input_tokens = None
                        output_tokens = None
                        cache_creation_input_tokens = None
                        cache_read_input_tokens = None
                        assistant_text = None
                self.trace.stage_latencies["anthropic_stream"] = (
                    time.perf_counter() - _anthropic_start
                )

                # Persist BEFORE yielding [DONE]. The Flutter SSE parser closes the
                # connection on [DONE], and Starlette cancels the still-running
                # generator at its next await point. In Python 3.12 CancelledError
                # is BaseException, so `except Exception` cannot catch it — the
                # commit silently never lands. Doing the write here guarantees
                # durability before the client can disconnect. Tokens still stream
                # live during the loop above, so the user-visible latency added
                # is only the DB write (~50–150ms typical).
                if ctx.conversation is not None and assistant_text:
                    with self.trace.stage("persistence"):
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

                # Telemetry — JSON-line sink + Braintrust span output. Both
                # are sync / buffered, so they survive the post-[DONE]
                # cancellation window noted above. The Braintrust span
                # output mirrors RequestTrace so dashboards across both
                # sinks line up. The trace is mutated in place (router
                # owns construction) so `stage_latencies` populated by the
                # context_assembly / rag_* / anthropic_stream / tool_dispatch
                # / persistence stages survives to here intact.
                self.trace.input_tokens = input_tokens
                self.trace.output_tokens = output_tokens
                self.trace.cache_creation_input_tokens = cache_creation_input_tokens
                self.trace.cache_read_input_tokens = cache_read_input_tokens
                self.trace.itinerary_emitted = itinerary_emitted
                self.trace.itinerary_days = itinerary_days
                self.trace.itinerary_validation_error = itinerary_validation_error
                self.trace.latency_ms = (
                    time.perf_counter() - self.started_at
                ) * 1000
                self.trace.estimated_cost_usd = compute_cost(
                    model=self.trace.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_creation_input_tokens=cache_creation_input_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                )
                log_trace(self.trace)
                span.log(
                    output=(
                        assistant_text
                        if settings.braintrust_log_user_payloads
                        else None
                    ),
                    metadata={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_creation_input_tokens": cache_creation_input_tokens,
                        "cache_read_input_tokens": cache_read_input_tokens,
                        "tool_calls": tool_calls_for_span,
                        "itinerary_emitted": itinerary_emitted,
                        "itinerary_days": itinerary_days,
                        "itinerary_validation_error": itinerary_validation_error,
                        "latency_ms": self.trace.latency_ms,
                        "stage_latencies": self.trace.stage_latencies,
                        "estimated_cost_usd": self.trace.estimated_cost_usd,
                        "error": None,
                    },
                )
            except Exception as e:
                self.trace.input_tokens = input_tokens
                self.trace.output_tokens = output_tokens
                self.trace.cache_creation_input_tokens = cache_creation_input_tokens
                self.trace.cache_read_input_tokens = cache_read_input_tokens
                self.trace.itinerary_emitted = itinerary_emitted
                self.trace.itinerary_days = itinerary_days
                self.trace.itinerary_validation_error = itinerary_validation_error
                self.trace.latency_ms = (
                    time.perf_counter() - self.started_at
                ) * 1000
                self.trace.error = str(e)
                self.trace.estimated_cost_usd = compute_cost(
                    model=self.trace.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_creation_input_tokens=cache_creation_input_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                )
                log_trace(self.trace)
                span.log(
                    output=(
                        assistant_text
                        if settings.braintrust_log_user_payloads
                        else None
                    ),
                    metadata={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_creation_input_tokens": cache_creation_input_tokens,
                        "cache_read_input_tokens": cache_read_input_tokens,
                        "tool_calls": tool_calls_for_span,
                        "itinerary_emitted": itinerary_emitted,
                        "itinerary_days": itinerary_days,
                        "itinerary_validation_error": itinerary_validation_error,
                        "latency_ms": self.trace.latency_ms,
                        "stage_latencies": self.trace.stage_latencies,
                        "estimated_cost_usd": self.trace.estimated_cost_usd,
                        "error": str(e),
                    },
                )
                yield f"event: error\ndata: {str(e)}\n\n"
