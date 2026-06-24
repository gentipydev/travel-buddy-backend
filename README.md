# Travel Buddy — Backend

FastAPI + async SQLAlchemy + Google Gemini + Cohere + Supabase. The
backend for an Albania-first AI travel assistant: streaming chat with
tool use, surgical trip refinement, RAG over curated Albanian content,
and post-save enrichment via Google Maps. Powers a Flutter mobile app.

## Architecture

```
app/
├── routers/         HTTP surface — thin shells around domain + agents
├── agents/          Multi-step LLM orchestration (streaming chat agent)
├── prompts/         Versioned prompt registry + .md prompt bodies
├── tools/           tool-use schemas + validators + refine dispatcher
├── llm/             Gemini client + adapters, system-prompt helpers, SSE encoding
├── rag/             Cohere embed + rerank + adaptive retrieval routing
├── integrations/    Google Maps (Places + Directions + Static Maps),
│                    Cohere client, Supabase Storage
├── domain/          Business logic — trips + conversations + chat pipeline
├── observability/   JSON logging + per-request telemetry traces
├── auth/            Supabase JWT verification (JWKS)
├── db/              Async SQLAlchemy engine + ORM models
├── schemas.py       Pydantic wire contracts (DTOs the Flutter client sees)
├── config.py        env-driven settings
└── main.py          FastAPI app, lifespan, middleware
```

Full per-package responsibilities and the chat-turn data flow live in
[`docs/architecture.md`](docs/architecture.md).

## AI engineering patterns

Each pattern links to working code:

- **[Streaming agent loop with tool use](docs/ai-patterns.md#1-streaming-agent-loop-with-tool-use)**
  — `app/agents/chat_agent.py`. Token-level streaming + dual-mode tool
  dispatch (full re-emit or surgical patch) + persist-before-DONE
  ordering for SSE durability.
- **[RAG with reranking](docs/ai-patterns.md#2-retrieval-augmented-generation-with-reranking)**
  — `app/rag/`. Routing heuristic + Cohere embed + HNSW search + Cohere
  rerank, all fail-soft.
- **[Dual-mode tool use](docs/ai-patterns.md#3-dual-mode-tool-use--full-re-emit--surgical-refine)**
  — `app/tools/`. `emit_itinerary` for whole-trip generation, seven
  surgical patch tools for refine sessions.
- **[Prompt caching](docs/ai-patterns.md#4-prompt-caching)** —
  `app/llm/caching.py`. Cached prefix + volatile suffix split so refine
  turns still hit the cache while `trip_json` varies per turn.
- **[Background persona refresh](docs/ai-patterns.md#5-background-tasks--persona-refresh)**
  — `app/domain/conversations/persona.py`. Fire-and-forget task with a
  cheap Haiku YES/NO gate.
- **[Multimodal](docs/ai-patterns.md#6-multimodal--images-in-chat)** —
  Image URLs persisted in conversation JSONB and rebuilt into
  multimodal content blocks on later turns.

## Quickstart

```bash
# 1. Configure
cp .env.example .env
# Fill in GEMINI_API_KEY, COHERE_API_KEY, SUPABASE_URL,
# SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY, DATABASE_URL,
# GOOGLE_API_KEY, GOOGLE_URL_SIGNING_SECRET.
# Every variable lives in `.env.example` with an inline comment.

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 3. Apply migrations (Supabase SQL editor or `supabase db push`)
ls supabase/migrations/*.sql       # 14 migrations, run in order

# 4. Seed (optional — curated Albanian places + RAG chunks)
python -m scripts.seed_places
python -m scripts.chunk_places
python -m scripts.embed_chunks    # needs COHERE_API_KEY
python -m scripts.seed_artifacts
python -m scripts.setup_storage   # idempotent bucket creation

# 5. Run
uvicorn app.main:app --reload
curl localhost:8000/health        # {"status":"ok"}
```

## Tests

```bash
pytest -q       # 64 service-layer tests, ~0.5s
```

Coverage today: pure functions (geo bbox, static map URL, Pydantic
schemas, itinerary validator) + service-level integration tests for the
Google enrichment paths (mocked HTTP via `httpx.MockTransport`). Router
endpoints + LLM behavior are **not** covered — see
[`evals/`](evals/) for the planned eval framework, and the **Smoke
test** section below for the manual procedure.

## Smoke test

```bash
# Two requests back-to-back in the same session — the second should
# show a non-zero `cache_read_input_tokens` in the RequestTrace JSON
# log line (prompt caching is hitting).
curl -N -X POST http://localhost:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"Plan me a 3-day Tirana trip."}' | head -200

curl -N -X POST http://localhost:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"Add a half-day to the riviera."}' | head -200
```

Expected SSE event ordering: `event: conversation_id` (if new
conversation), `event: place_card` (if a curated place matched the
message), streamed `data:` text chunks, `event: itinerary` (for
multi-day asks) **or** `event: trip_refined` (for refine sessions) **or**
`event: itinerary_invalid` (validator rejected the model's tool input),
then `data: [DONE]`. Persistence-failure mode: `event: persist_failed`
lands before `[DONE]`.

## Operations

- **Logs.** One JSON object per record, every line carries
  `request_id`. `RequestTrace` is emitted as a single JSON line per
  `/chat/stream` request with fields including
  `cache_creation_input_tokens`, `cache_read_input_tokens`,
  `input_tokens`, `output_tokens`, `latency_ms`,
  `itinerary_emitted`, `trip_refine`. Render's Logs UI filters per-field
  on JSON.
- **Cache hit rate.** Compute as
  `cache_read_input_tokens / (cache_read_input_tokens +
  cache_creation_input_tokens + input_tokens)`.
- **Kill switches.** `GOOGLE_PLACES_ENABLED=false` and
  `GOOGLE_DIRECTIONS_ENABLED=false` short-circuit the corresponding
  enrichment paths. Trips save without that data; the client falls
  back to straight-line polylines.
- **Model overrides.** `MODEL_CHAT`, `MODEL_PERSONA_GATE`,
  `MODEL_TITLE_GEN`, `MODEL_SUMMARIZE` can be pinned independently in
  `.env` for A/B tests.

## Deployment

Container in `Dockerfile`, hosted on Render. The Postgres connection
points at the Supabase transaction pooler (port 6543) — see
[`app/db/engine.py`](app/db/engine.py) for the pgbouncer-aware
tuning.

## Repository layout

```
travel-buddy-backend/
├── app/                  Application code (see Architecture section)
├── tests/                Unit + service-level tests (pytest)
├── evals/                Eval-style regression tests (skeleton today)
├── docs/                 architecture.md + ai-patterns.md
├── supabase/migrations/  14 SQL migrations, run in order
├── scripts/              Seed, chunk, embed, smoke, storage setup
├── Dockerfile, docker-compose.yml, requirements*.txt
└── .env.example          14-var template
```

## Contributing notes

- Prompts are the single source of truth in
  [`app/prompts/`](app/prompts/) — `.md` files for bodies,
  [`registry.py`](app/prompts/registry.py) for metadata. Bump
  `version` on behavior changes; future evals will pin fixtures to
  `(name, version)`.
- The SSE event ordering in `app/agents/chat_agent.py` is load-bearing
  for the Flutter parser. If you change the order of yields around
  `data: [DONE]`, also update the smoke-test recipe above and ideally
  add a router-level integration test (see Risks section in the
  restructure plan).
- Import boundaries between packages are listed at the bottom of
  [`docs/architecture.md`](docs/architecture.md#boundary-rules-what-each-layer-is-allowed-to-import).
  Crossing them backwards is a smell.
