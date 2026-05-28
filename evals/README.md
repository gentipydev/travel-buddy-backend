# Evals

Behavior-pinning tests for the LLM layer. Schema-validity tests already
live in `tests/` (Pydantic shape, geo bbox, static map URL); this
directory is reserved for **eval-style regression tests** — runs that
hit a real Claude model and assert qualitative properties of the output.

Six suites are wired in [`runners/braintrust_run.py`](runners/braintrust_run.py)
with fixtures in [`fixtures/`](fixtures/). The deterministic scorers and the
retrieval / faithfulness-live suites run against a seeded `chunks` corpus
(below); the refine suite's task is still stubbed (needs a test JWT — see its
docstring).

### Seeding the corpus

The retrieval-quality and faithfulness-live suites query a real `chunks`
corpus. Seed it with [`seed_eval_db.py`](seed_eval_db.py):

```bash
# fresh/empty Postgres (local docker pgvector or CI service): create schema too
DATABASE_URL=postgresql+asyncpg://... python -m evals.seed_eval_db --bootstrap
# already-migrated DB: omit --bootstrap; only eval-tagged chunks are touched
python -m evals.seed_eval_db
```

CI ([`backend-evals.yml`](../.github/workflows/backend-evals.yml)) runs a
pgvector service and this seed step automatically. The LLM-as-judge suites
(persona-gate, faithfulness) need the `openai` package (autoevals' proxy
client — see `requirements-dev.txt`) and `BRAINTRUST_API_KEY` to call the
judge.

## Categories

1. **Itinerary regression.** Given a fixed seed of user-message inputs,
   `chat.base + itinerary_tool_addendum` should produce a
   structurally-valid `emit_itinerary` tool call 100% of the time and
   should keep every stop inside the named city (or named region). The
   geographic-boundary rule lives in
   [`app/prompts/chat/itinerary_tool_addendum.md`](../app/prompts/chat/itinerary_tool_addendum.md)
   — these evals are how we catch regressions on that rule.

2. **Refine surgical-edit fidelity.** Given a saved trip JSON + an edit
   instruction ("shorten the museum stop to 45 minutes", "swap stops 2
   and 3", "drop the rooftop bar"), only the targeted fields should
   change. The 7 patch tools in
   [`app/tools/refine/`](../app/tools/refine/) are designed so any
   un-mentioned field stays byte-identical post-application.

3. **Persona refresh signal / noise.** The Haiku YES/NO gate
   ([`app/domain/conversations/persona.py`](../app/domain/conversations/persona.py))
   should fire only when a summary genuinely captures something the
   current persona is missing. Eval inputs: pairs of `(current_persona,
   new_summary)` with ground-truth labels.

4. **Retrieval quality.** The hybrid retriever
   ([`app/rag/embedding.py`](../app/rag/embedding.py)) should surface a
   chunk relevant to the query. `retrieval_hit_rate` asks "is the right
   chunk anywhere in the result set?" (the recall win lexical search adds
   for proper nouns); `retrieval_mrr` asks "how high did the reranker
   rank it?". Fixture inputs: a query + `relevant_keywords` that a correct
   chunk must contain. The `retrieval_task` runs the live retriever against
   the seeded corpus (above).

5. **Answer faithfulness (judge calibration).** The assistant's reply should
   be grounded in the retrieved context — no contradictions, no fabricated
   specifics (prices, hours, names). `answer_faithfulness` is an LLM-as-judge
   (Haiku) that grades `{message, context}` → `answer`. This suite feeds it
   pre-baked faithful/unfaithful pairs to verify the judge discriminates. It
   deliberately tolerates general background knowledge; it flags invented
   verifiable details, the harmful hallucinations for a travel app.

6. **Answer faithfulness (live).** Same judge, but on real /chat/stream
   answers: `faithfulness_live_task` runs a turn, captures the streamed reply,
   and pairs it with the context the retriever surfaces for that query. This
   measures whether the production system hallucinates, not just whether the
   judge works.

## Fixture format

Pin each eval fixture to a specific prompt version (`name`, `version`)
from [`app/prompts/registry.py`](../app/prompts/registry.py). When a
prompt's `version` bumps, re-run the affected fixtures.

```json
{
  "prompts": {"chat.base": "2026-05-18", "chat.itinerary_tool_addendum": "2026-05-18"},
  "input": {"message": "3 days in Tirana"},
  "expected": {
    "tool_called": "emit_itinerary",
    "days": 3,
    "every_stop_city_in": ["Tirana"]
  }
}
```

## Suggested runners

- [Promptfoo](https://promptfoo.dev) — assertion-style, low-code, can
  read this directory layout natively.
- [Anthropic Evals](https://docs.anthropic.com/en/docs/test-and-evaluate)
  — first-party, currently beta; consume the same fixture format.

## Cost-effectiveness

Refine-fidelity and persona-gate evals are small (1–5 Haiku calls per
case). Itinerary regression is heavier (one full 8k-token completion per
case). Budget the suite to run nightly in CI on the prompt-affecting
diffs only, not on every PR.
