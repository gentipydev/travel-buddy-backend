"""Provider-neutral chat streaming with DeepSeek fallback.

`ChatStreamRouter.stream()` is an async generator that yields NORMALIZED
events so the chat agent's SSE-strict loop doesn't care which provider served
the turn:

    ("text", str)                              # a streamed text delta
    ("tool_call", SimpleNamespace(name, input))# a complete tool/function call
    ("usage", (input, output, cache_creation, cache_read))  # token accounting

After iterating, the agent reads the served-by metadata off the router:
`served_by` ("gemini"|"deepseek"), `served_model`, `fallback_reason`,
`images_dropped`, and `stop_reason` (the provider finish reason, for
diagnostics on the itinerary-invalid path).

Two independent fallback triggers (see app/domain/usage_budget.py +
app/routers/chat.py for B):

  (A) Error fallback — Gemini fails at/before stream open with 503 / 429 /
      the 403 origin-block AND no text has streamed yet → retry the SAME turn
      on DeepSeek. The Gemini client already exhausts its own retries
      ([429,500,502,503,504]) on the open call first; we fall back only after
      that, and only when nothing has hit the wire (a mid-stream switch can't
      cleanly resume). Anything else re-raises unchanged → today's behavior.

  (B) Budget overflow — the router is constructed with a non-None
      `force_fallback_reason`, so it serves DeepSeek directly instead of the
      caller hard-blocking the turn.

gemini-3.5-flash always stays PRIMARY: the router only ever falls back to
`settings.model_fallback` (DeepSeek), never to another Gemini tier, and every
fallback is surfaced on the trace.
"""
from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

from google.genai import types

from app.config import settings
from app.llm import deepseek_client, gemini_client
from app.llm.deepseek_adapter import extract_usage as deepseek_extract_usage
from app.llm.deepseek_adapter import to_openai_messages, to_openai_tools
from app.llm.gemini_adapter import extract_usage as gemini_extract_usage
from app.llm.gemini_adapter import to_gemini_contents, to_gemini_tools
from app.llm.gemini_client import classify_gemini_error

logger = logging.getLogger(__name__)

# Shared output ceiling for both providers. Matches the historical chat-agent
# value: it sits well above what the itinerary schema allows, and a too-low
# cap can truncate mid-tool-call, leaving emit_itinerary with a missing field.
# We only pay for tokens actually generated, so the headroom is mostly free.
_MAX_OUTPUT_TOKENS = 8192


def _gemini_fallback_eligible(exc: BaseException) -> bool:
    """True for the Gemini failures we spill to DeepSeek: 503/429 capacity +
    rate limits, and the 403 HTML origin-block (free-tier datacenter-IP
    block). Mirrors the 403 heuristic in `classify_gemini_error`. Other
    errors are NOT eligible — they re-raise so the existing error path runs."""
    text = str(exc)
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status in (429, 503):
        return True
    is_403 = status == 403 or text.lstrip().startswith("403")
    is_html = "<html" in text.lower() or "Error 403 (Forbidden)" in text
    return bool(is_403 and is_html)


class ChatStreamRouter:
    """One per chat turn. Streams from Gemini (primary) with a DeepSeek
    fallback, exposing which provider/model actually served the turn."""

    def __init__(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        force_fallback_reason: str | None = None,
    ):
        self.system = system
        self.messages = messages
        self.tools = tools
        # Set by trigger B (budget overflow) — when present and DeepSeek is
        # enabled, serve DeepSeek from the start instead of trying Gemini.
        self.force_fallback_reason = force_fallback_reason
        # Served-by metadata the agent reads after the stream completes.
        self.served_by = "gemini"
        self.served_model = settings.model_chat
        self.fallback_reason: str | None = None
        self.images_dropped = False
        self.stop_reason: str | None = None

    def _switch_to_deepseek(self, reason: str) -> None:
        self.served_by = "deepseek"
        self.served_model = settings.model_fallback
        self.fallback_reason = reason

    async def stream(self):
        # Trigger B: forced fallback (budget overflow). Serve DeepSeek
        # directly. Guarded on `deepseek_enabled()` for safety — the caller
        # only sets this when enabled, so otherwise fall through to Gemini.
        if self.force_fallback_reason and deepseek_client.deepseek_enabled():
            self._switch_to_deepseek(self.force_fallback_reason)
            async for ev in self._deepseek_events():
                yield ev
            return

        # Trigger A: try Gemini; on a pre-emit eligible error, fall back.
        emitted_text = False
        do_fallback = False
        try:
            async for ev in self._gemini_events():
                if ev[0] == "text":
                    emitted_text = True
                yield ev
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnect / generator close — never a provider error.
            raise
        except BaseException as exc:
            if not (
                deepseek_client.deepseek_enabled()
                and not emitted_text
                and _gemini_fallback_eligible(exc)
            ):
                raise
            classified = classify_gemini_error(exc)
            logger.warning(
                "chat router: Gemini failed before first token; falling back "
                "to DeepSeek (%s)",
                classified,
            )
            self._switch_to_deepseek(f"gemini_error: {classified}")
            do_fallback = True
        if do_fallback:
            async for ev in self._deepseek_events():
                yield ev

    # -- Gemini (primary) ---------------------------------------------------

    async def _gemini_events(self):
        """Stream from Gemini, yielding normalized events. References
        `gemini_client.client` by module attribute so tests/fixtures that
        patch that shared object still take effect."""
        contents = await to_gemini_contents(self.messages)
        gemini_tools = to_gemini_tools(self.tools)
        config = types.GenerateContentConfig(
            system_instruction=self.system,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            tools=gemini_tools,
            # Tools are terminal + rendered client-side; we read function_call
            # parts off the stream ourselves rather than letting the SDK invoke.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        stream = await gemini_client.client.aio.models.generate_content_stream(
            model=settings.model_chat,
            contents=contents,
            config=config,
        )
        usage_meta = None
        async for chunk in stream:
            for cand in getattr(chunk, "candidates", None) or []:
                fr = getattr(cand, "finish_reason", None)
                if fr is not None:
                    self.stop_reason = str(fr)
                cand_content = getattr(cand, "content", None)
                parts = (
                    getattr(cand_content, "parts", None) or []
                    if cand_content is not None
                    else []
                )
                for part in parts:
                    txt = getattr(part, "text", None)
                    if txt:
                        yield ("text", txt)
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        yield (
                            "tool_call",
                            SimpleNamespace(
                                name=fc.name, input=dict(fc.args or {})
                            ),
                        )
            if getattr(chunk, "usage_metadata", None) is not None:
                usage_meta = chunk.usage_metadata
        yield ("usage", gemini_extract_usage(usage_meta))

    # -- DeepSeek (fallback) ------------------------------------------------

    async def _deepseek_events(self):
        """Stream from DeepSeek (OpenAI-compatible), yielding normalized
        events. Text streams as it arrives; tool-call fragments are
        accumulated by index and parsed once the stream closes."""
        oai_messages, images_dropped = to_openai_messages(
            self.system, self.messages
        )
        self.images_dropped = images_dropped
        oai_tools = to_openai_tools(self.tools)
        create_kwargs: dict = {
            "model": settings.model_fallback,
            "messages": oai_messages,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "stream": True,
            # Ask for usage on the final chunk (OpenAI streaming omits it
            # otherwise) so token accounting / cost stay accurate.
            "stream_options": {"include_usage": True},
        }
        if oai_tools:
            create_kwargs["tools"] = oai_tools
            create_kwargs["tool_choice"] = "auto"
        stream = await deepseek_client.client.chat.completions.create(
            **create_kwargs
        )
        # index -> {"name": str|None, "args": str} — OpenAI streams a tool
        # call as fragments: the name arrives once, `arguments` arrives as a
        # JSON string split across many deltas, keyed by `index`.
        tool_frag: dict[int, dict] = {}
        usage = None
        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            for choice in getattr(chunk, "choices", None) or []:
                fr = getattr(choice, "finish_reason", None)
                if fr is not None:
                    self.stop_reason = str(fr)
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if content:
                    yield ("text", content)
                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(tc, "index", 0) or 0
                    frag = tool_frag.setdefault(idx, {"name": None, "args": ""})
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            frag["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            frag["args"] += fn.arguments
        # Flush accumulated tool calls in index order, parsing the JSON args.
        for idx in sorted(tool_frag):
            frag = tool_frag[idx]
            if not frag["name"]:
                continue
            try:
                parsed = json.loads(frag["args"]) if frag["args"] else {}
            except json.JSONDecodeError:
                logger.warning(
                    "deepseek: tool_call arguments not valid JSON (name=%s)",
                    frag["name"],
                )
                parsed = {}
            yield ("tool_call", SimpleNamespace(name=frag["name"], input=parsed))
        yield ("usage", deepseek_extract_usage(usage))
