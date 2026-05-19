"""Multi-step LLM orchestration.

Today: `ChatAgent` (in `chat_agent.py`) — the streaming turn handler for
`/chat/stream`. It owns:
  - the `client.beta.prompt_caching.messages.stream(...)` call with the
    cached system payload assembled by `app/llm/caching.assemble_chat_system`
  - live `text_stream` forwarding as SSE `data:` frames
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
