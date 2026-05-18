# AI Engineering Patterns

The Travel Buddy backend showcases six patterns. Each section here
links to the actual code; treat the codebase as the worked example.

## 1. Streaming agent loop with tool use

[`app/agents/chat_agent.py`](../app/agents/chat_agent.py)

Per turn, the agent:

1. Streams Claude's text response live (SSE `data:` frames) so the user
   sees tokens immediately, then captures the full message **inside**
   the `async with` context so accumulated state — content blocks, tool
   uses, token counts — is still alive.
2. Filters the final message for `tool_use` blocks and dispatches based
   on which tool was called: `emit_itinerary` triggers the full-plan
   path; any of the seven refine tools triggers the surgical-patch path.
3. **Persist before [DONE].** The Flutter parser closes the SSE
   connection on `data: [DONE]`, and Starlette cancels the generator at
   the next `await` — in Python 3.12 that raises `CancelledError` (a
   `BaseException`, not an `Exception`), so `except Exception` cannot
   catch it. The `append_messages` / `generate_title` /
   `maybe_summarize` block runs before the final `[DONE]` yield, and
   emits `event: persist_failed` on its own failure.

## 2. Retrieval-augmented generation with reranking

[`app/rag/`](../app/rag/)

Routing first ([`retrieval.py:needs_retrieval`](../app/rag/retrieval.py)):
a small regex skips greetings ("hi", "thanks", "ok"), yes-no answers,
and small-talk. False positives just spend one Cohere embed call; false
negatives leave Claude ungrounded, so the heuristic errs toward
retrieving.

Embedding ([`embedding.py:search_chunks`](../app/rag/embedding.py))
prepends the last two assistant turns (each capped at 200 chars) before
the query so follow-ups like "what about its castle?" stay anchored to
the previously discussed place. Cohere `embed-multilingual-v3.0` with
`input_type=search_query` hits a HNSW index on
`chunks.embedding`.

Reranking ([`reranking.py:rerank_chunks`](../app/rag/reranking.py))
re-scores the top embedding hits with
`rerank-multilingual-v3.0` and returns the top 3 plus the top
relevance score. Both calls are fail-soft — any Cohere outage falls
back to `[]` (chat continues with no retrieved context) or the
embed-only ordering (no rerank).

The retrieved context lands in the system prompt's **volatile suffix**
(see Prompt Caching below) so chunks differ per query without
cache-busting the prefix.

## 3. Dual-mode tool use — full re-emit + surgical refine

[`app/tools/`](../app/tools/)

Claude has two ways to mutate a trip:

- **`emit_itinerary`** ([`itinerary_tool.py`](../app/tools/itinerary_tool.py))
  — the full multi-day plan, emitted alongside the assistant's text
  reply. On the initial-creation chat path this becomes
  `event: itinerary` in the SSE stream and powers the "Save itinerary"
  button. On refine turns it triggers
  [`replace_trip_from_payload`](../app/domain/trips/mutations.py),
  atomically replacing days/stops.
- **Surgical refine tools** ([`tools/refine/`](../app/tools/refine/)) —
  seven small JSON-schema tools (`update_stop`, `insert_stop`,
  `delete_stop`, `reorder_stops`, `add_day`, `delete_day`,
  `update_trip`) offered only on refine turns. Each addresses a
  specific stop or day by its UUID (UUIDs are injected into the system
  prompt via `serialize_trip_for_context`). The dispatcher
  ([`refine/dispatcher.py`](../app/tools/refine/dispatcher.py))
  validates each tool's input against its Pydantic body
  (`TripStopUpdate`, `TripDayCreateBody`, …) and routes to the matching
  `apply_*` in `app.domain.trips.mutations`. Multiple patch tools in
  one turn apply atomically in a single transaction.

Validation matters: the Anthropic API does not enforce a tool's
`input_schema`, so a missing field or out-of-enum kind would otherwise
break the Flutter parser mid-stream. The validators give us schema
guarantees the API doesn't.

## 4. Prompt caching

[`app/llm/caching.py`](../app/llm/caching.py)

The system prompt for `/chat/stream` is ~3-4k tokens of stable
instructions: BASE persona + addenda + (optional) preferences block.
That prefix barely changes within a chat session — so
[`assemble_chat_system`](../app/llm/caching.py) wraps it with
`cache_control: {"type": "ephemeral"}` and the chat agent passes
`betas=["prompt-caching-2024-07-31"]` to
`client.beta.messages.stream(...)`.

**Cached prefix + volatile suffix.** The volatile half (the refine
addendum with its per-turn `trip_json`, plus any retrieved RAG context)
is appended as a second, *uncached* block. That way trip-refine
sessions still get a cache hit on the BASE + addenda prefix while their
`trip_json` blob varies every turn.

Hit-rate visibility: `RequestTrace`
([`observability/telemetry.py`](../app/observability/telemetry.py))
captures `cache_creation_input_tokens` and `cache_read_input_tokens`
from `response.usage`. Cache hit rate =
`cache_read / (cache_read + cache_creation + input_tokens)`. Render's
Logs UI can compute this as a derived metric.

Where caching does **not** pay off: `maybe_summarize`,
`generate_title`, the persona refresh gate, and the manual
`/preferences/persona` endpoint all use systems under ~300 tokens or
are called rarely. The cache-write cost would exceed the savings.

## 5. Background tasks — persona refresh

[`app/domain/conversations/persona.py`](../app/domain/conversations/persona.py)

Every time `maybe_summarize` rolls 15 old turns into a new
`Conversation.summary`, it fires `refresh_travel_persona` via
`asyncio.create_task` — fire-and-forget so the chat stream is never
blocked.

The task opens its own DB session (the request-scoped one is closed by
the time it runs), then:

1. Skips if `user_preferences.updated_at` is younger than 24 h
   (cooldown + defers to a recent manual
   `POST /preferences/persona` write).
2. Runs a Haiku YES/NO gate
   ([`_should_refresh_persona`](../app/domain/conversations/persona.py))
   asking "does this summary capture something the current persona is
   missing?". Cheap (~10 input tokens, ~3 output tokens) — the gate
   protects against a flood of refresh calls on every summarization.
3. On YES, generates a new persona from the five most-recent summaries
   and writes it to `user_preferences.travel_persona`.

The Haiku gate's model ID is `settings.model_persona_gate`, held
separate from `settings.model_chat` so a future model upgrade doesn't
silently make this YES/NO call expensive.

## 6. Multimodal — images in chat

[`app/domain/conversations/repository.py:_render_message`](../app/domain/conversations/repository.py)

User chat messages can carry image URLs (public Supabase Storage links,
uploaded via [`integrations/supabase_storage.py`](../app/integrations/supabase_storage.py)
into the `chat-attachments` bucket). The `append_messages` writer
persists those URLs alongside the text in `Conversation.messages` JSONB,
and `_render_message` rebuilds Anthropic's multimodal content-block
shape on later turns:

```python
[
  {"type": "image", "source": {"type": "url", "url": "<storage_url>"}},
  {"type": "text",  "text": "<the typed message>"},
]
```

Resumed conversations preserve visual context — a user who uploaded a
photo of a beach can ask "what was that beach again?" three turns later
and Claude still sees the image.
