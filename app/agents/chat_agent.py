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
import asyncio
import json
import logging
import time
from types import SimpleNamespace

from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.domain.chat_pipeline import ChatTurnContext
from app.domain.conversations import (
    append_messages,
    generate_title,
    maybe_summarize,
)
from app.domain.output_guard import check_output_guardrail
from app.domain.trips.enrichment import (
    geocode_itinerary_preview,
    geocode_stop_preview,
)
from app.domain.trips.mutations import replace_trip_from_payload
from app.domain.trips.repository import serialize_trip
from app.domain.usage_budget import (
    record_usage,
    record_usage_detached,
    release_stream_slot,
)
from app.llm import (
    assemble_chat_system,
    classify_gemini_error,
    client,
    encode_sse_text,
    extract_usage,
    to_gemini_contents,
    to_gemini_tools,
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


def _flatten_stops(itinerary: dict) -> list[dict]:
    """Flatten an emit_itinerary input (full or partially-parsed snapshot)
    into an ordered list of stop dicts carrying their day_number +
    in-day sort_order. Tolerant of missing/half-built keys so it can run
    against a streaming `input_json` snapshot."""
    out: list[dict] = []
    days = itinerary.get("days")
    if not isinstance(days, list):
        return out
    for day in days:
        if not isinstance(day, dict):
            continue
        day_number = day.get("day_number")
        day_title = day.get("title")
        stops = day.get("stops")
        if not isinstance(stops, list):
            continue
        for i, stop in enumerate(stops):
            if not isinstance(stop, dict):
                continue
            out.append(
                {
                    "day_number": day_number,
                    "day_title": day_title,
                    "sort_order": i,
                    "name": stop.get("name"),
                    "kind": stop.get("kind"),
                    "city": stop.get("city"),
                    # `notes` is the per-stop prose the model writes into the
                    # tool call; it streams to the live chat row as `description`.
                    "description": stop.get("notes"),
                    "lat": stop.get("lat"),
                    "lng": stop.get("lng"),
                    "google_place_id": stop.get("google_place_id"),
                }
            )
    return out


async def _stop_marker_frame(stop: dict, index: int) -> str | None:
    """Geocode one streamed stop and render it as an `itinerary_stop` SSE
    frame for the live map. Returns None when the stop has no name or can't
    be mapped (the client simply doesn't get a marker for it)."""
    name = (stop.get("name") or "").strip()
    if not name:
        return None
    geo = await geocode_stop_preview(
        name=name,
        city=stop.get("city"),
        lat=stop.get("lat"),
        lng=stop.get("lng"),
    )
    if geo is None:
        return None
    lat, lng, place_id = geo
    payload = {
        "id": f"live-{stop.get('day_number')}-{stop.get('sort_order')}",
        "index": index,
        "day_number": stop.get("day_number"),
        "day_title": stop.get("day_title"),
        "sort_order": stop.get("sort_order"),
        "name": name,
        "kind": stop.get("kind"),
        "city": stop.get("city"),
        # Per-stop prose (from the tool's `notes`) — renders as the chat row's
        # description in lockstep with this marker dropping on the map.
        "description": stop.get("description"),
        "lat": lat,
        "lng": lng,
        "google_place_id": stop.get("google_place_id") or place_id,
    }
    return f"event: itinerary_stop\ndata: {json.dumps(payload)}\n\n"


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
        # Usage-budget accounting state (app/domain/usage_budget.py).
        # `_streamed_chars` accumulates streamed text so a client
        # disconnect can still be billed from an estimate.
        self._usage_recorded = False
        self._streamed_chars = 0

    async def stream(self):
        """Public entry: delegates to `_stream_inner` and owns the
        teardown that must survive client disconnect. Starlette closes
        the generator on disconnect, raising CancelledError /
        GeneratorExit (BaseExceptions the inner `except Exception`
        cannot see), so this `finally` is the only reliable place to
        release the concurrency slot and account for tokens that
        streamed but were never recorded. `asyncio.create_task` is
        synchronous, so scheduling the detached write is safe during
        cancellation unwinding (no `await` allowed here); it runs on a
        fresh session because the request-scoped one is being torn down.
        """
        try:
            async for frame in self._stream_inner():
                yield frame
        finally:
            if self.user_id:
                release_stream_slot(self.user_id)
                if not self._usage_recorded:
                    self._usage_recorded = True
                    # Disconnect before usage arrived: estimate output
                    # from streamed chars (~4 chars/token, same heuristic
                    # as app/llm/token_budget). Input is unknown → 0; a
                    # conservative undercount is fine — the request still
                    # lands in `usage_daily` and the per-minute rate
                    # limit bounds disconnect-spam.
                    est_out = self._streamed_chars // 4
                    asyncio.create_task(
                        record_usage_detached(
                            self.user_id,
                            input_tokens=0,
                            output_tokens=est_out,
                            estimated_cost_usd=compute_cost(
                                model=self.trace.model,
                                input_tokens=0,
                                output_tokens=est_out,
                                cache_creation_input_tokens=0,
                                cache_read_input_tokens=0,
                            ),
                        )
                    )

    async def _stream_inner(self):
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
        # Validated emit_itinerary payload, set on the initial-creation path.
        # Used only as a fallback (whole-trip geocode) when per-stop streaming
        # produced no markers — e.g. partial-JSON parsing surfaced nothing.
        itinerary_preview_payload: dict | None = None
        # Per-stop live-map streaming state (initial-creation path only).
        # `preview_started` gates the one-time stream-start frame the client
        # uses to reset its live trip; `preview_emitted` is how many flattened
        # stops we've consumed (mapped or skipped) so the end-of-stream flush
        # knows where to resume.
        preview_started: bool = False
        preview_emitted: int = 0
        # Tool names the model fired this turn — empty for plain chat,
        # `["emit_itinerary"]` for whole-trip create, one or more
        # `update_stop` / `insert_stop` / etc. for patch refine. Lands on
        # the Braintrust span so dashboards can filter/group by tool.
        tool_calls_for_span: list[str] = []
        # `chat_span` no-ops to a stand-in span when BRAINTRUST_ENABLED is
        # false — the same `span.log(...)` calls below work unchanged. Token /
        # latency metadata is logged explicitly on the parent span (Braintrust
        # ships no Gemini auto-tracing wrapper; see `bt_wrap_genai`).
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
                # Gemini takes the system prompt as a single
                # `config.system_instruction` string (no per-block cache
                # breakpoints — caching is implicit on 2.5+/3 models). The
                # stable prefix is joined first so implicit caching can reuse
                # it across turns; cache accounting comes back on
                # `chunk.usage_metadata` and is mapped by `extract_usage`.
                system_instruction = assemble_chat_system(
                    cached_prefix=ctx.cached_prefix,
                    volatile_suffix=ctx.volatile_suffix,
                )
                gemini_tools = to_gemini_tools(available_tools)
                contents = await to_gemini_contents(ctx.messages)
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    # 8192 sits well above what the schema now allows
                    # (`MAX_STOPS_PER_DAY=3`, so ~12 days × 3 stops + the
                    # intro/outro paragraph). The headroom is useful — refine
                    # patch tools are O(edit-size) and don't scale with trip
                    # size, and a too-low cap can truncate mid-tool-call,
                    # leaving `emit_itinerary` with a missing `days` field and
                    # triggering `itinerary_invalid`. We only pay for tokens
                    # actually generated, so the ceiling is mostly free.
                    max_output_tokens=8192,
                    tools=gemini_tools,
                    # We never let the SDK invoke tools — `emit_itinerary` and
                    # the refine ops are terminal, rendered client-side — so
                    # disable automatic function calling and read the
                    # function_call parts off the stream ourselves.
                    automatic_function_calling=(
                        types.AutomaticFunctionCallingConfig(disable=True)
                    ),
                )
                # Manual stage timer (not `with self.trace.stage(...)`) so the
                # SSE-strict streaming body doesn't need to be re-indented.
                # Recorded on success after the loop; mid-stream exceptions
                # fall through to the outer except, capturing the partial
                # duration there.
                _llm_start = time.perf_counter()
                # Gemini streams text deltas progressively but delivers the
                # function call as a complete block (the Python SDK does not
                # reliably stream partial tool args). So we accumulate text +
                # any function_call parts here, then flush every itinerary
                # stop onto the live map after the stream closes (below) using
                # the validated payload — the live map populates in one burst
                # rather than progressively, but the SSE contract is identical.
                _text_parts: list[str] = []
                function_calls: list[SimpleNamespace] = []
                finish_reason = None
                usage_meta = None
                stream = await client.aio.models.generate_content_stream(
                    model=settings.model_chat,
                    contents=contents,
                    config=config,
                )
                async for chunk in stream:
                    for cand in getattr(chunk, "candidates", None) or []:
                        fr = getattr(cand, "finish_reason", None)
                        if fr is not None:
                            finish_reason = fr
                        cand_content = getattr(cand, "content", None)
                        parts = (
                            getattr(cand_content, "parts", None) or []
                            if cand_content is not None
                            else []
                        )
                        for part in parts:
                            txt = getattr(part, "text", None)
                            if txt:
                                self._streamed_chars += len(txt)
                                _text_parts.append(txt)
                                yield encode_sse_text(txt)
                            fc = getattr(part, "function_call", None)
                            if fc is not None:
                                function_calls.append(
                                    SimpleNamespace(
                                        name=fc.name,
                                        input=dict(fc.args or {}),
                                    )
                                )
                    if getattr(chunk, "usage_metadata", None) is not None:
                        usage_meta = chunk.usage_metadata
                # Capture usage + dispatch tool calls. Wrapped — token-count
                # or SDK-shape failures must not crash the request.
                try:
                    (
                        input_tokens,
                        output_tokens,
                        cache_creation_input_tokens,
                        cache_read_input_tokens,
                    ) = extract_usage(usage_meta)
                    assistant_text = "".join(_text_parts)
                    stop_reason = (
                        str(finish_reason) if finish_reason is not None else None
                    )
                    # Validate emit_itinerary tool calls with a Pydantic
                    # validator before emitting `event: itinerary`. The model
                    # is not forced to honor the tool schema, so a missing
                    # field or out-of-enum kind would otherwise break the
                    # Flutter parser mid-stream. On validation failure we emit
                    # `event: itinerary_invalid` with a one-line reason; the
                    # natural-language reply already streamed and stays useful.
                    tool_use_blocks = function_calls
                    tool_calls_for_span = [b.name for b in tool_use_blocks]
                    emit_block = next(
                        (b for b in tool_use_blocks if b.name == "emit_itinerary"),
                        None,
                    )
                    refine_blocks = [
                        b for b in tool_use_blocks if b.name in REFINE_TOOL_NAMES
                    ]

                    if emit_block is not None:
                        # Whole-trip rewrite path. If the model also emitted
                        # patch tools in the same turn (rare; the prompt
                        # discourages it), we ignore them — emit_itinerary's
                        # full replace supersedes any narrow op.
                        _td_start = time.perf_counter()
                        block = emit_block
                        # Capture raw day count even on rejection — distinguishes
                        # "model tried a 5-day plan we rejected" from "model
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
                                # Flush every stop onto the live map using the
                                # validated payload — the canonical, complete
                                # plan. (Gemini delivers the tool call whole, so
                                # this is where the markers stream.)
                                final_flat = _flatten_stops(result.payload)
                                while preview_emitted < len(final_flat):
                                    s = final_flat[preview_emitted]
                                    preview_emitted += 1
                                    frame = await _stop_marker_frame(
                                        s, preview_emitted
                                    )
                                    if frame:
                                        if not preview_started:
                                            preview_started = True
                                            yield (
                                                "event: itinerary_stream_start\n"
                                                "data: {}\n\n"
                                            )
                                        yield frame
                                yield f"event: itinerary\ndata: {json.dumps(result.payload)}\n\n"
                                # Fallback only: if per-stop streaming yielded
                                # nothing mappable, geocode the whole trip
                                # after the stream closes (handled below).
                                itinerary_preview_payload = result.payload
                        else:
                            itinerary_validation_error = result.error
                            # Surface raw input keys + finish_reason on the wire
                            # so the Flutter client logs can distinguish "model
                            # truncated mid-tool-call" (finish_reason=MAX_TOKENS,
                            # missing days/stops) from "model emitted a shape we
                            # didn't expect". Cheap diagnostic — always-on.
                            raw_keys = (
                                sorted(block.input.keys())
                                if isinstance(block.input, dict)
                                else None
                            )
                            error_payload = json.dumps({
                                "reason": result.error,
                                "stop_reason": stop_reason,
                                "raw_keys": raw_keys,
                            })
                            yield f"event: itinerary_invalid\ndata: {error_payload}\n\n"
                            logger.warning(
                                "itinerary validation failed: %s | stop_reason=%s | raw=%s",
                                result.error,
                                stop_reason,
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
                self.trace.stage_latencies["llm_stream"] = (
                    time.perf_counter() - _llm_start
                )

                # Live map fallback: per-stop streaming above normally populates
                # the map as stops generate. Only if it surfaced NOTHING mappable
                # (e.g. partial-JSON parsing yielded no usable stops) do we
                # geocode the whole validated trip in one shot here and push a
                # single preview. Best-effort — a failure just leaves the map
                # empty; the text + optional Save already streamed. The refine
                # path needs nothing extra — it emits `trip_refined`.
                if itinerary_preview_payload is not None and not preview_started:
                    with self.trace.stage("itinerary_preview"):
                        try:
                            preview = await geocode_itinerary_preview(
                                db, itinerary_preview_payload
                            )
                            if preview:
                                yield (
                                    "event: itinerary_preview\n"
                                    f"data: {json.dumps(preview)}\n\n"
                                )
                        except Exception:
                            logger.exception("itinerary preview geocode failed")

                # Account usage into `usage_daily` BEFORE [DONE] — same
                # durability reasoning as the persistence block below:
                # after [DONE] the client disconnects and cancellation
                # would silently drop the write. Fail-soft internally, so
                # a broken accounting write can't kill the stream.
                if self.user_id:
                    await record_usage(
                        db,
                        self.user_id,
                        input_tokens=input_tokens or 0,
                        output_tokens=output_tokens or 0,
                        cache_creation_input_tokens=(
                            cache_creation_input_tokens or 0
                        ),
                        cache_read_input_tokens=cache_read_input_tokens or 0,
                        estimated_cost_usd=compute_cost(
                            model=self.trace.model,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cache_creation_input_tokens=cache_creation_input_tokens,
                            cache_read_input_tokens=cache_read_input_tokens,
                        ),
                    )
                    self._usage_recorded = True

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
                                    settings.model_title_gen,
                                    request.message,
                                    assistant_text,
                                )
                            await maybe_summarize(
                                db, ctx.conversation, settings.model_summarize
                            )
                        except Exception:
                            # Tokens already streamed live, but persistence failed —
                            # tell the client so it can mark the assistant message as
                            # "not saved" inline (no snackbar). Emit BEFORE [DONE]
                            # so the parser sees it before the stream-done sentinel.
                            yield "event: persist_failed\ndata: server_persist_failed\n\n"
                            logger.exception("conversation persist failed")

                # Output guardrail — fail-open, observational only. The reply
                # is already in the client's buffer; we are auditing, not
                # blocking. Adds ~100-200 ms LLM latency to [DONE] (the
                # text is already rendered on the client — comparable to the
                # persistence window above). Result is written to
                # self.trace.output_guard_label before span.log() below.
                if assistant_text:
                    with self.trace.stage("output_guardrail"):
                        try:
                            label = await check_output_guardrail(assistant_text)
                            self.trace.output_guard_label = label
                        except Exception:
                            logger.warning(
                                "output_guardrail stage failed — fail open",
                                exc_info=True,
                            )

                yield "data: [DONE]\n\n"

                # Telemetry — JSON-line sink + Braintrust span output. Both
                # are sync / buffered, so they survive the post-[DONE]
                # cancellation window noted above. The Braintrust span
                # output mirrors RequestTrace so dashboards across both
                # sinks line up. The trace is mutated in place (router
                # owns construction) so `stage_latencies` populated by the
                # context_assembly / rag_* / llm_stream / tool_dispatch
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
                        "output_guard_label": self.trace.output_guard_label,
                        "latency_ms": self.trace.latency_ms,
                        "stage_latencies": self.trace.stage_latencies,
                        "estimated_cost_usd": self.trace.estimated_cost_usd,
                        "error": None,
                    },
                )
            except Exception as e:
                # Collapse Google's HTML 403 edge-block page (origin/free-tier
                # block) to a stable code; passes other errors through as-is.
                error_str = classify_gemini_error(e)
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
                self.trace.error = error_str
                self.trace.estimated_cost_usd = compute_cost(
                    model=self.trace.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_creation_input_tokens=cache_creation_input_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                )
                # Mid-stream failure: usage may never have arrived, so
                # fall back to the streamed-chars estimate — partial
                # output still costs money and must count toward the
                # budget. `record_usage` rolls back internally if this
                # session's transaction is already aborted.
                if self.user_id and not self._usage_recorded:
                    self._usage_recorded = True
                    err_out = (
                        output_tokens
                        if output_tokens is not None
                        else self._streamed_chars // 4
                    )
                    await record_usage(
                        db,
                        self.user_id,
                        input_tokens=input_tokens or 0,
                        output_tokens=err_out,
                        cache_creation_input_tokens=(
                            cache_creation_input_tokens or 0
                        ),
                        cache_read_input_tokens=cache_read_input_tokens or 0,
                        estimated_cost_usd=compute_cost(
                            model=self.trace.model,
                            input_tokens=input_tokens,
                            output_tokens=err_out,
                            cache_creation_input_tokens=cache_creation_input_tokens,
                            cache_read_input_tokens=cache_read_input_tokens,
                        ),
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
                        "output_guard_label": self.trace.output_guard_label,
                        "latency_ms": self.trace.latency_ms,
                        "stage_latencies": self.trace.stage_latencies,
                        "estimated_cost_usd": self.trace.estimated_cost_usd,
                        "error": error_str,
                    },
                )
                yield f"event: error\ndata: {error_str}\n\n"
