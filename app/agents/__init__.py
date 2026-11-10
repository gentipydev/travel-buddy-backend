"""Multi-step LLM orchestration.

Today: `ChatAgent` (in `chat_agent.py`) — the streaming turn handler for
`/chat/stream`. It owns:
  - the `client.messages.stream(...)` call with the cache-marked system
    payload assembled by `app/llm/caching.assemble_chat_system`
  - live text-delta forwarding as SSE `data:` frames, plus per-stop live-map
    frames driven off the streamed `input_json` tool-input snapshots
  - `emit_itinerary` validation + SSE emission (or `event: itinerary_invalid`
    on Pydantic failure)
  - refine-tool dispatch via `app.tools.refine.apply_refine_tool`,
    emitting `event: trip_refined` with the canonical post-state
  - persist-then-DONE ordering (append_messages + generate_title +
    maybe_summarize all land BEFORE `data: [DONE]\\n\\n`)
  - per-request `RequestTrace` emission

The chat router (`app/routers/chat.py`) is a ~25-line shell that
constructs a ChatAgent and returns its `StreamingResponse`.
"""
from app.agents.chat_agent import ChatAgent

__all__ = ["ChatAgent"]
