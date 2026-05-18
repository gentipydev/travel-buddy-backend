# Evals

Behavior-pinning tests for the LLM layer. Schema-validity tests already
live in `tests/` (Pydantic shape, geo bbox, static map URL); this
directory is reserved for **eval-style regression tests** — runs that
hit a real Claude model and assert qualitative properties of the output.

Empty for now. The skeleton + this README describe the intended shape so
the work has a home when it's prioritised.

## Three planned categories

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
