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

## 2. Hybrid retrieval-augmented generation with reranking

[`app/rag/`](../app/rag/)

Routing first ([`retrieval.py:needs_retrieval`](../app/rag/retrieval.py)):
a small regex skips greetings ("hi", "thanks", "ok"), yes-no answers,
and small-talk. False positives just spend one Cohere embed call; false
negatives leave Claude ungrounded, so the heuristic errs toward
retrieving.

Hybrid search ([`embedding.py:search_chunks`](../app/rag/embedding.py))
runs two retrievers and fuses them:

- **Vector** — Cohere `embed-multilingual-v3.0` with
  `input_type=search_query` against a HNSW index on `chunks.embedding`.
  The query is prepended with the last two assistant turns (each capped
  at 200 chars) so follow-ups like "what about its castle?" stay
  anchored to the previously discussed place.
- **Lexical** — a Postgres full-text (`tsvector` / `ts_rank`) lookup
  against the generated `chunks.search_text` column (migration 016,
  `simple` config). Catches exact proper-noun matches ("Theth",
  "Gjirokastër", "furgon") that embedding search routinely misses. The
  query is built as an **OR of the message's significant terms**
  (stopwords/short tokens dropped — `embedding.py:_significant_terms`).
  The earlier `websearch_to_tsquery` ANDed *every* word, so "How do I get
  to Theth?" became `how & do & i & get & to & theth` and matched
  nothing — the arm was silently inert for natural-language queries,
  leaving the "hybrid" retriever effectively vector-only. OR + `ts_rank`
  (which rewards matching more distinct terms) restores the recall this
  arm exists for; RRF + the reranker stay the precision filter.

**Contextual retrieval** (migration 017,
[`contextual.py`](../app/rag/contextual.py)): each chunk carries a
`context` header templated from its place metadata — e.g. `"Berat — city
in Berat, Berat County, Albania. Themes: unesco, ottoman, …"`. The header
is prepended to the embedded text ([`embed_chunks.py`](../scripts/embed_chunks.py))
and folded into the generated `search_text`, so structured metadata the
prose omits ("Vlorë County", "beach", "unesco") becomes searchable by
both arms. It is **not** shown to the model —
[`retrieval.py:retrieve_context`](../app/rag/retrieval.py) joins `content`
only. This adapts Anthropic's *Contextual Retrieval* to structured data:
a deterministic template rather than an LLM call, because each chunk
already *is* the whole place doc and the missing context is columns we
own.

The two ranked lists merge with Reciprocal Rank Fusion (RRF, K=60) into
a single top-`k` candidate pool. Hybrid's job is *recall*; the reranker
is still the precision filter.

Reranking ([`reranking.py:rerank_chunks`](../app/rag/reranking.py))
re-scores the fused candidates with `rerank-multilingual-v3.0` and
returns the top 3 plus the top relevance score.

Fail-soft, layered: a Cohere embed outage degrades to lexical-only (embed
error caught → `[]`), a FTS error degrades to vector-only, and a rerank
API error falls back to the fused order. Only an empty fused pool leaves
Claude with no context.

**Measured** ([`evals/measure_retrieval.py`](../evals/measure_retrieval.py)
over `retrieval_quality.jsonl`). Two corpora tell two halves of the story.

*Lexical-arm code fix (12-chunk corpus).* Isolating the lexical arm (NULL
embeddings, reranker live) — the path that carries a freshly-edited-but-
not-yet-re-embedded chunk, or an embed failure:

| lexical arm (12-chunk) | hit-rate | MRR |
|---|---|---|
| `websearch` AND-of-all-terms (before) | 0.00 | 0.00 |
| OR of significant terms | 0.75 | 0.75 |
| + contextual header | 1.00 | 0.94 |

On 12 chunks the full vector + lexical + rerank pipeline sat at 1.00 / 0.94
no matter what: the multilingual embedder already retrieved every chunk, so
the metric was *saturated* and blind to any retrieval change. The lexical
fix still mattered (the arm went from *inert* to *functional*, the fail-soft
path from 0 to full recall) — it just couldn't be seen in the headline
number.

*Corpus growth to un-saturate (30-chunk corpus).* To make the metric able to
move — the prerequisite for measuring any future retrieval change and for
promoting these scorers to fail-on-regression — the eval corpus was grown to
30 chunks built around **confusable clusters** (5 castles, 6 Ionian beaches,
two distinct "Blue Eye" springs, …) with fixtures that pin one chunk among
its lookalikes. Measured 2026-05-28, reranker live:

| 30-chunk corpus | hit-rate | MRR |
|---|---|---|
| full vector + lexical + rerank | 0.89 | 0.895 |
| lexical arm only (NULL embeddings) | 0.95 | 0.921 |

The ceiling is gone. The remaining gap is two header-only county-recall
fixtures, and the failures are instructive rather than cosmetic:
`korce-county-recall` retrieves the right chunk (Lake Ohrid lands in the
fused top-10) but the reranker drops it below `top_n=3`; `vlore-county-recall`'s
pinned chunk is crowded out of the candidate pool by its nine Vlorë-County
siblings under the full pipeline, yet the lexical arm alone still surfaces it
at rank 2 (hit 1.00 / MRR 0.50). That arm-level divergence — vector + rerank
scoring *lower* than lexical alone on one query — is exactly the signal a
saturated corpus hides. Both are left red as headroom targets for query
rewrite (#3) and reranker / `top_n` tuning.

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

## 7. Output safety guardrail

[`app/domain/output_guard.py`](../app/domain/output_guard.py) ·
[`app/agents/chat_agent.py`](../app/agents/chat_agent.py) (hook point) ·
[`app/prompts/output_guard/system.md`](../app/prompts/output_guard/system.md)

### What it is

An observational, fail-open guardrail that runs after the assistant's reply
has been fully streamed to the client. It calls Haiku with a narrow binary
classifier (`SAFE` / `REVIEW`) to flag three violation types:

1. **System-prompt leaks** — the model quotes or paraphrases its operating
   instructions in the reply.
2. **Harmful content** — dangerous advice, hate speech, or content that could
   cause real-world harm.
3. **Severe off-topic** — the reply has abandoned travel assistance entirely
   (e.g. financial fraud instructions, extremist content). A polite redirect
   is explicitly *not* off-topic.

Ambiguous cases default to `"safe"` — the prompt is intentionally narrow.

### Why observational-only

The text is already in the client's buffer before the guardrail runs. It is
**not** a blocking gate; `"review"` is a signal for offline analysis and
Braintrust monitoring, not a runtime rejection. This keeps the design
fail-open: a guardrail outage or timeout never degrades the user experience.

### Hook point and latency budget

```
[text tokens stream live]      ← user sees text immediately
[persist conversation]         +50–150 ms
[output_guardrail stage]       +100–200 ms  ← Haiku call, max_tokens=8
[DONE]
```

The guardrail fits inside `with self.trace.stage("output_guardrail"):`,
between `append_messages` and `yield "data: [DONE]\n\n"`, so its latency
is automatically recorded in `stage_latencies` and the result is available
to `span.log()` before the Braintrust span closes.

### Fail-open contract

`check_output_guardrail()` never raises. Three paths return `"safe"`:

| Condition | Behaviour |
|-----------|-----------|
| Empty / whitespace-only text | `"safe"` immediately, no API call |
| `asyncio.TimeoutError` after 5 s | `"safe"` + WARNING log |
| Any other exception | `"safe"` + WARNING log |

### Telemetry

Every `RequestTrace` JSON line includes `"output_guard_label": "safe" | "review" | null`.
`null` means the guard was skipped (empty text or the agent errored before
`assistant_text` was populated). Every Braintrust span carries the same field
in `metadata.output_guard_label` alongside `stage_latencies.output_guardrail`.

### Eval suite

`evals/fixtures/output_guardrail.jsonl` (10 fixtures: 5 safe, 5 review)
calibrates the judge offline. Run via Braintrust:

```bash
python -m evals.runners.braintrust_run  # triggers "output-guardrail" experiment
```

The CI fixture guard (`tests/test_eval_fixtures.py::test_output_guardrail_fixtures_are_valid`)
validates fixture format without any API calls.

### Prompt versioning

The guard prompt lives in the registry (`"output_guard.system"`, version
`2026-05-28`). The eval scorer loads the same entry, so the offline
calibration suite always reflects the production prompt — bumping the `.md`
file and the version string in `registry.py` keeps both in sync.
