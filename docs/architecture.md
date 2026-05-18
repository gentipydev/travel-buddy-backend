# Architecture

Travel Buddy is a FastAPI backend that powers an Albania-first AI
travel assistant. It speaks SSE to a Flutter client, streams Anthropic
Claude responses with tool use, and persists trips + conversations into
Supabase (Postgres + Auth + Storage). The package layout is organized
around **what each module talks to** — `llm/` talks to Anthropic,
`integrations/` talks to Google + Cohere + Supabase Storage, `agents/`
composes the LLM + the tools + the domain into a streaming turn, and so
on.

## Per-package responsibilities

```
app/
├── routers/         HTTP surface. Thin shells around domain + agents.
├── agents/          Multi-step LLM orchestration. Owns the streaming loop.
├── prompts/         Single source of truth for every system prompt.
├── tools/           Claude tool-use schemas + validators + dispatchers.
├── llm/             Anthropic client, prompt-caching helpers, SSE encoding.
├── rag/             Embedding + reranking + retrieval + place detection.
├── integrations/    Third-party APIs (Google Maps, Cohere, Supabase Storage).
├── domain/          Business logic — trips and conversations.
├── observability/   Structured logging + per-request telemetry traces.
├── auth/            Supabase JWT verification.
├── db/              SQLAlchemy engine + ORM models.
├── schemas.py       Pydantic wire contracts (request/response DTOs).
├── config.py        Settings (env-driven).
├── deps.py          FastAPI dependency utilities.
└── main.py          FastAPI app construction, lifespan, middleware.
```

### `routers/` — HTTP surface

One module per resource: `chat.py`, `trips.py`, `conversations.py`,
`preferences.py`, `users.py`, `images.py`, `health.py`. Each router is a
thin shell — `chat.py` is 49 lines, all the agent loop lives in
[`app/agents/chat_agent.py`](../app/agents/chat_agent.py). Routers
exclusively call into `domain/` / `agents/`; they never reach into
`integrations/` or `llm/` directly.

### `agents/` — multi-step LLM orchestration

`ChatAgent` (in
[`chat_agent.py`](../app/agents/chat_agent.py)) is the one entry point
today. It owns:

- `client.beta.messages.stream(...)` with the cached system payload from
  `app/llm/caching.assemble_chat_system`
- live `text_stream` forwarding as SSE `data:` frames
- `emit_itinerary` validation + SSE emission (or
  `event: itinerary_invalid` on Pydantic failure)
- refine-tool dispatch via `app.tools.refine.apply_refine_tool`,
  emitting `event: trip_refined` with the canonical post-state
- persist-then-DONE ordering (`append_messages` +
  `generate_title` + `maybe_summarize` all land BEFORE `data: [DONE]`)
- per-request `RequestTrace` emission, including cache-hit metrics

### `prompts/` — single source of truth

Every prompt body lives in a `.md` file under one of
`chat/ conversation/ persona/`. The Python
[`registry.py`](../app/prompts/registry.py) maps a `(name, version,
cache_eligible)` tuple per prompt; call sites look up by name
(`get("chat.base")`) instead of importing a global string. Versioning
is what evals pin to.

The `builders.py` module exposes
[`build_system_payload(prefs, trip_context)`](../app/prompts/builders.py)
which returns `(cached_prefix, volatile_suffix)` — the chat agent feeds
those to `assemble_chat_system` for the Anthropic prompt-caching
payload.

### `tools/` — Claude tool-use schemas

Two flavours:

- [`itinerary_tool.py`](../app/tools/itinerary_tool.py): the
  `emit_itinerary` full-plan emission tool, offered on every chat turn.
  Validated by
  [`itinerary_validator.py`](../app/tools/itinerary_validator.py) (the
  Anthropic API does not enforce input_schema).
- [`refine/`](../app/tools/refine/): seven surgical patch tools
  (`update_stop`, `insert_stop`, `delete_stop`, `reorder_stops`,
  `add_day`, `delete_day`, `update_trip`) offered only on refine turns
  (`ctx.refine_trip is not None`). One file per tool for visibility +
  one
  [`dispatcher.py`](../app/tools/refine/dispatcher.py) that validates
  each call against its Pydantic body and routes to the matching
  `apply_*` in `app.domain.trips.mutations`.

### `llm/` — model layer

- [`anthropic_client.py`](../app/llm/anthropic_client.py): the shared
  `AsyncAnthropic` singleton.
- [`caching.py`](../app/llm/caching.py): builds a 2-block system
  payload with a `cache_control: ephemeral` marker on the stable
  prefix. The beta endpoint
  (`client.beta.messages.stream(..., betas=[PROMPT_CACHING_BETA])`) is
  required in `anthropic==0.39.0`.
- [`streaming.py`](../app/llm/streaming.py): `encode_sse_text` —
  serializes streamed text into SSE `data:` frames (escapes embedded
  newlines so multi-line chunks stay one event).
- [`models.py`](../app/llm/models.py): re-exports the four
  `settings.model_*` IDs so call sites depend on the LLM layer instead
  of poking into config.

### `rag/` — retrieval-augmented generation

- [`embedding.py`](../app/rag/embedding.py): Cohere `embed` +
  PostgreSQL HNSW cosine search. Prepends the last two assistant turns
  to the embed input so "what about its castle?" stays anchored.
- [`reranking.py`](../app/rag/reranking.py): Cohere rerank-multilingual.
- [`retrieval.py`](../app/rag/retrieval.py): `needs_retrieval` regex
  router (skip greetings / yes-no / thanks) + the full pipeline
  `retrieve_chunks(query, db, history)` returning `(top_n_chunks,
  top_rerank_score)`.
- [`place_detection.py`](../app/rag/place_detection.py): substring
  match against curated `places` rows for the place-card payload.

Fail-soft throughout: any Cohere outage returns `[]` so the chat agent
continues with no retrieved context.

### `integrations/` — third-party APIs

One module per provider:
[`google_places.py`](../app/integrations/google_places.py) (three-tier:
curated → cache → API; backfills coords, photos, rating),
[`google_directions.py`](../app/integrations/google_directions.py)
(routed polylines with snap-to-roads fallback + 90-day cache),
[`static_maps.py`](../app/integrations/static_maps.py) (signed Static
Maps URL builder, pure URL construction),
[`supabase_storage.py`](../app/integrations/supabase_storage.py)
(bucket helpers), [`cohere_client.py`](../app/integrations/cohere_client.py)
(shared Cohere v2 client). Every integration is fail-soft.

### `domain/` — business logic

Two subdomains plus the chat-turn pipeline:

- [`trips/`](../app/domain/trips/) — `repository.py` (load/serialize),
  `mutations.py` (`replace_trip_from_payload` + all `apply_*`),
  `enrichment.py` (post-save Google Places + Directions + static map),
  `geo.py` (Albania bbox + haversine).
- [`conversations/`](../app/domain/conversations/) — `repository.py`
  (get/create + append + history rendering), `summarization.py`
  (rolling summarization + title generation), `persona.py` (background
  travel-persona refresh task).
- [`chat_pipeline.py`](../app/domain/chat_pipeline.py) — `prepare_turn`
  composes prefs + conversation + trip context + RAG into a
  `ChatTurnContext` that the chat agent streams against.

### `observability/` — logging + telemetry

JSON-line logging with per-request `request_id` / `user_id` ContextVars
in [`request_context.py`](../app/observability/request_context.py),
formatter + middleware in
[`logging_config.py`](../app/observability/logging_config.py), and the
`RequestTrace` per-request structured trace in
[`telemetry.py`](../app/observability/telemetry.py). Trace fields
include `cache_creation_input_tokens` and `cache_read_input_tokens` so
Render's Logs UI can compute hit rate as a derived metric.

### `db/`, `auth/`, `schemas.py`, `config.py`

`db/engine.py` is the async SQLAlchemy session factory + pgbouncer
tuning for the Supabase transaction pooler. `db/models.py` keeps every
ORM model in one declarative_base (the FK web is load-bearing).
`auth/supabase_auth.py` verifies Supabase JWTs via JWKS. `schemas.py`
is the Pydantic wire contract — the **DTOs the Flutter client sees**.
`config.py` is just `Settings` + the lazy Supabase Storage client.

## Chat-turn data flow

The single hottest path. Everything else in the codebase exists to
support this:

```
POST /chat/stream
   │
   ├── routers/chat.py
   │       └── ctx = await prepare_turn(request, supa_user, db)
   │                    │
   │                    ├── load User + UserPreferences (db)
   │                    ├── refine? → _load_trip + get_or_create_refine_conversation
   │                    │             (domain/trips/repository)
   │                    ├── get_or_create_conversation (domain/conversations/repository)
   │                    ├── retrieve_chunks (rag/retrieval)
   │                    │     └── search_chunks → rerank_chunks → Cohere
   │                    ├── detect_place (rag/place_detection)
   │                    └── build_system_payload (prompts/builders)
   │                          → (cached_prefix, volatile_suffix)
   │
   └── ChatAgent(ctx, db, request, user_id).stream()
           │
           ├── yield "event: conversation_id" (if new)
           ├── yield "event: place_card" (if place detected)
           ├── system_payload = assemble_chat_system(cached_prefix, volatile_suffix)
           │       (llm/caching → list[TextBlockParam] with cache_control)
           │
           ├── async with client.beta.messages.stream(
           │     system=system_payload,
           │     tools=[ITINERARY_TOOL, *REFINE_TOOLS?],
           │     betas=[PROMPT_CACHING_BETA],
           │   ):
           │     async for text in response.text_stream:
           │         yield encode_sse_text(text)       (llm/streaming)
           │
           │     final = await response.get_final_message()
           │     cache_*_tokens = final.usage.cache_*_input_tokens
           │
           │     if emit_itinerary block:
           │         validate_itinerary (tools/itinerary_validator)
           │         if refine_trip:
           │             replace_trip_from_payload (domain/trips/mutations)
           │             commit
           │             yield "event: trip_refined" + serialize_trip
           │         else:
           │             yield "event: itinerary"
           │     elif refine patch blocks:
           │         for each block: apply_refine_tool (tools/refine/dispatcher)
           │         commit
           │         yield "event: trip_refined"
           │
           ├── append_messages (domain/conversations/repository)
           ├── generate_title (domain/conversations/summarization) [first-turn only]
           ├── maybe_summarize (domain/conversations/summarization)
           │       └── asyncio.create_task(refresh_travel_persona)
           │             (domain/conversations/persona, fire-and-forget)
           │
           ├── yield "data: [DONE]\n\n"
           │
           └── log_trace(RequestTrace(...))       (observability/telemetry)
```

## Boundary rules (what each layer is allowed to import)

- `routers/` → `domain/`, `agents/`, `schemas.py`, `db/`, `auth/`. **Not**
  `integrations/`, `llm/`, `tools/`, `rag/`.
- `agents/` → everything. The agent is the composition root.
- `domain/` → `db/`, `prompts/`, `rag/` (for `chat_pipeline`),
  `integrations/` (only `domain/trips/enrichment.py`). **Not**
  `agents/`, `routers/`, `tools/`, `llm/`.
- `tools/` → `domain/trips/mutations` (dispatcher only),
  `schemas`. **Not** `agents/`, `routers/`, `llm/`.
- `rag/` → `integrations/cohere_client`, `db/`. **Not**
  `agents/`, `routers/`, `domain/`, `tools/`.
- `integrations/` → `config`, `db/`, possibly each other. **Not**
  upward.
- `llm/` → `config`, `prompts` (none today, but allowed). **Not**
  upward.
- `prompts/` → `db/models` only (for type hints). **Not** anything else.
- `observability/` → stdlib + FastAPI middleware types only.

Any import that crosses these rules backwards is a smell; either the
import is wrong or the rule needs updating.
