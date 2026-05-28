"""Braintrust scorers for Travel Buddy chat evals.

Each scorer is a pure function returning 0.0..1.0 (or a dict with `score`
and `metadata`). The same function runs offline in `Eval(...)` experiments
AND online via Braintrust's project-level scoring rules — no duplication
between modes.

Public scorers:

  - `itinerary_schema_valid` — deterministic. Output must pass the same
    Pydantic validator the live chat handler uses (`validate_itinerary`).
  - `stop_count_cap` — deterministic. Every day in the emitted itinerary
    must have at most `MAX_STOPS_PER_DAY` stops. Distinct from
    `itinerary_schema_valid` so Braintrust dashboards can triage cap
    regressions separately from generic schema failures.
  - `geo_boundary_adherence` — deterministic. When the user's message
    names a single Albanian city, every emitted stop's `city` must match.
    This is the regression catcher for the 2026-05-18 incident where
    "3 days in Tirana" leaked Kruja stops — both inside Albania, so a
    country-bbox check missed it.
  - `refine_fidelity` — deterministic. Verifies that a refine turn only
    mutated the stop UUIDs the instruction implies.
  - `retrieval_hit_rate` — deterministic. Did the RAG retriever surface a
    chunk relevant to the query (keyword-grounded) anywhere in its result
    set? The headline recall metric for the hybrid retriever
    (`app/rag/embedding.py`).
  - `retrieval_mrr` — deterministic. Mean reciprocal rank of the first
    relevant chunk — rewards the reranker for putting the right chunk near
    the top, not just somewhere in the set.
  - `persona_gate` — LLM-as-judge (Haiku). Did the system correctly
    decide whether to refresh the persona?
  - `answer_faithfulness` — LLM-as-judge (Haiku). Is the assistant's reply
    grounded in the retrieved context — no contradictions, no fabricated
    specifics (prices, hours, names)? The hallucination guard for a travel
    app where invented details are a real user harm.
  - `output_guardrail` — LLM-as-judge (Haiku). Offline calibration scorer
    for the production output guardrail (`app/domain/output_guard.py`).
    Checks whether the judge correctly classifies replies as "safe" or
    "review" (system-prompt leaks, harmful content, severe off-topic).

Naming: scorers are referenced by these slugs in the Braintrust UI when
configuring online scoring rules, so don't rename without updating the
project settings.
"""
from __future__ import annotations

import unicodedata
from typing import Any

from app.schemas import MAX_STOPS_PER_DAY
from app.tools.itinerary_validator import validate_itinerary

# Canonical Albanian-city set with accepted spelling variants. All forms
# lowercase + diacritic-stripped — comparisons go through `_canon()` so a
# stop labeled "Vlorë" matches a user message saying "vlora".
_ALBANIAN_CITIES: dict[str, set[str]] = {
    "tirana": {"tirana", "tirane"},
    "berat": {"berat", "berati"},
    "vlora": {"vlora", "vlore"},
    "shkodra": {"shkodra", "shkoder", "skadar"},
    "saranda": {"saranda", "sarande"},
    "gjirokastra": {"gjirokastra", "gjirokaster"},
    "korca": {"korca", "korce"},
    "durresi": {"durresi", "durres", "durazzo"},
    "kruja": {"kruja", "kruje"},
}

# Words signaling multi-city scope. When any appear, the scorer abstains
# (returns 1.0) because there's no single target city to enforce.
_REGIONAL_WORDS = {
    "riviera", "south", "north", "coast", "tour", "circuit",
    "country", "loop", "across",
}


def _canon(text: str) -> str:
    """Lowercase + strip combining diacritics for robust string match."""
    nfkd = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


# ---------------------------------------------------------------------- #
# Deterministic scorers                                                  #
# ---------------------------------------------------------------------- #


def itinerary_schema_valid(output: Any, **_: Any) -> float:
    """1.0 iff `output` passes `validate_itinerary`, else 0.0.

    Reuses the exact validator the chat handler runs against
    `emit_itinerary` tool inputs — by construction this scorer agrees
    with production rejection logic, so a regression here is a
    regression in `itinerary_validation_error` rate.
    """
    if not isinstance(output, dict):
        return 0.0
    return 1.0 if validate_itinerary(output).ok else 0.0


def stop_count_cap(output: Any, **_: Any) -> float:
    """1.0 iff every day in `output` has at most `MAX_STOPS_PER_DAY`
    stops, else 0.0.

    Split out from `itinerary_schema_valid` so a cap regression doesn't
    hide inside generic schema failures. Both scorers move together when
    the cap is enforced as expected — if `stop_count_cap` drops while
    `itinerary_schema_valid` stays at 1.0, the validator is silently
    accepting >cap days (which shouldn't be possible while the Pydantic
    `max_length` is in place).
    """
    if not isinstance(output, dict):
        return 0.0
    for day in (output.get("days") or []):
        if not isinstance(day, dict):
            return 0.0
        if len(day.get("stops") or []) > MAX_STOPS_PER_DAY:
            return 0.0
    return 1.0


def geo_boundary_adherence(
    output: Any, input: Any = None, **_: Any
) -> float:
    """Share of stops whose `city` matches the city the user named.

    Heuristic: only fires when the input message mentions exactly one
    Albanian city AND contains no regional words. Otherwise returns 1.0
    — there's no single target to enforce. When it does fire, every
    stop's `city` field must include one of the recognized variants of
    the named city.

    Returns 0.0..1.0 — share of stops that match. Empty itineraries
    return 1.0 (nothing to violate).
    """
    if not isinstance(output, dict):
        return 0.0

    message = (
        input.get("message", "")
        if isinstance(input, dict)
        else str(input or "")
    )
    msg = _canon(message)

    if any(w in msg for w in _REGIONAL_WORDS):
        return 1.0

    mentioned = [
        canon
        for canon, variants in _ALBANIAN_CITIES.items()
        if any(v in msg for v in variants)
    ]
    if len(mentioned) != 1:
        return 1.0  # no single target city → nothing to enforce

    target_variants = _ALBANIAN_CITIES[mentioned[0]]

    stops = [
        s
        for d in (output.get("days") or [])
        for s in (d.get("stops") or [])
    ]
    if not stops:
        return 1.0

    matches = sum(
        1
        for s in stops
        if any(v in _canon(s.get("city", "") or "") for v in target_variants)
    )
    return matches / len(stops)


def refine_fidelity(
    output: Any, expected: Any = None, **_: Any
) -> dict[str, Any] | float:
    """Refine turns must only touch the stop UUIDs the instruction implies.

    `expected` shape (per fixture row):
        {
          "expected_touched_stop_ids": ["uuid1", ...],
          "expected_unchanged_stop_ids": ["uuid2", ...],
          "original_stops": {"uuid1": {<stop>}, "uuid2": {<stop>}, ...},
        }
    `output` is the post-refine trip JSON (same shape as TripCreate).

    Returns a dict with `score` + `metadata` listing which stops were
    wrongly preserved or wrongly touched — Braintrust UI surfaces both
    for fast triage.
    """
    if not isinstance(output, dict) or not isinstance(expected, dict):
        return 0.0

    out_stops = {
        s["id"]: s
        for d in (output.get("days") or [])
        for s in (d.get("stops") or [])
        if isinstance(s, dict) and "id" in s
    }
    orig_stops = expected.get("original_stops") or {}
    expected_touched = set(expected.get("expected_touched_stop_ids") or [])
    expected_unchanged = set(expected.get("expected_unchanged_stop_ids") or [])

    wrong_preserved: list[str] = []  # should have changed, didn't
    wrong_touched: list[str] = []    # should not have changed, did

    for sid in expected_touched:
        # A delete leaves the stop out of out_stops; that counts as
        # touched (correctly). Only a still-present, byte-identical stop
        # is wrong here.
        if sid in out_stops and sid in orig_stops:
            if out_stops[sid] == orig_stops[sid]:
                wrong_preserved.append(sid)

    for sid in expected_unchanged:
        # Must still exist AND be byte-identical to the original.
        if sid not in out_stops or sid not in orig_stops:
            wrong_touched.append(sid)
        elif out_stops[sid] != orig_stops[sid]:
            wrong_touched.append(sid)

    total = len(expected_touched) + len(expected_unchanged)
    if total == 0:
        return 1.0
    score = 1.0 - (len(wrong_preserved) + len(wrong_touched)) / total

    return {
        "score": max(0.0, score),
        "metadata": {
            "wrong_preserved": wrong_preserved,
            "wrong_touched": wrong_touched,
        },
    }


def _chunk_is_relevant(content: str, keywords: list[str]) -> bool:
    """A chunk is relevant to a row iff ALL of its `relevant_keywords` appear
    (canonicalized substring) in the chunk content. AND-semantics let an
    author require a specific chunk ("Theth" + "guesthouse"); a single-keyword
    row degrades to plain presence. Matching goes through `_canon` so
    "Gjirokastër" in the fixture matches "gjirokaster" in a chunk."""
    if not keywords:
        return False
    canon_content = _canon(content)
    return all(_canon(k) in canon_content for k in keywords)


def retrieval_hit_rate(output: Any, expected: Any = None, **_: Any) -> float:
    """1.0 iff at least one retrieved chunk is relevant, else 0.0.

    `output` is the ordered list of retrieved chunk *contents* (the
    `retrieval_task` runs the live hybrid retriever and returns chunk text —
    matching by content, not by UUID, keeps fixtures stable across re-seeds).
    `expected` is `{"relevant_keywords": [...]}`.

    This is the recall metric the hybrid retriever exists to move: a
    proper-noun query ("Theth") whose chunk the embedding arm misses should
    still be surfaced by the lexical arm. Abstains (1.0) when a row carries no
    keywords — same convention as `geo_boundary_adherence`.
    """
    if not isinstance(output, list):
        return 0.0
    keywords = expected.get("relevant_keywords") if isinstance(expected, dict) else None
    if not keywords:
        return 1.0
    for content in output:
        if isinstance(content, str) and _chunk_is_relevant(content, keywords):
            return 1.0
    return 0.0


def retrieval_mrr(output: Any, expected: Any = None, **_: Any) -> float:
    """Reciprocal rank of the first relevant chunk: 1/rank (1-indexed), 0.0 if
    none is relevant.

    Distinct from `retrieval_hit_rate`: hit rate only asks "is the right chunk
    anywhere in the set?", MRR asks "how high did it rank?". A drop in MRR while
    hit rate holds points at the reranker (`app/rag/reranking.py`) ordering a
    relevant chunk below an irrelevant one. Abstains (1.0) when a row carries no
    keywords.
    """
    if not isinstance(output, list):
        return 0.0
    keywords = expected.get("relevant_keywords") if isinstance(expected, dict) else None
    if not keywords:
        return 1.0
    for rank, content in enumerate(output, start=1):
        if isinstance(content, str) and _chunk_is_relevant(content, keywords):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------- #
# LLM-as-judge scorers                                                   #
# ---------------------------------------------------------------------- #

# Lazy-built — autoevals is an optional dep and the prompt registry
# requires the `.md` files to be present on disk.
_persona_gate_scorer: Any = None


def _build_persona_gate() -> Any:
    """Construct the LLMClassifier on first call.

    Reuses the production gate prompt verbatim via the prompt registry,
    so when the prompt is bumped the judge updates automatically. Model
    defaults to Haiku (matches `settings.model_persona_gate` for cost
    parity) — override via Braintrust's per-scorer model config if
    stronger judging is needed.
    """
    from autoevals import LLMClassifier  # type: ignore[import-not-found]

    from app.prompts.registry import get

    gate_system = get("persona.gate_system").text

    return LLMClassifier(
        name="persona_gate",
        prompt_template=(
            f"{gate_system}\n\n"
            "---\n"
            "Inputs: {{input}}\n"
            "Decision the system made: {{output}}\n"
            "Expected (ground truth): {{expected}}\n\n"
            "Was the system's decision correct? "
            "A=YES (system was right) B=NO (system was wrong)."
        ),
        choice_scores={"A": 1.0, "B": 0.0},
        model="claude-haiku-4-5-20251001",
        use_cot=False,
    )


def persona_gate(
    output: Any, input: Any = None, expected: Any = None, **kwargs: Any
) -> Any:
    """LLM judge: was the YES/NO persona-refresh decision correct?

    The live gate (`app/domain/conversations/persona`) decides whether
    a recently-summarized conversation reveals enough new persona signal
    to justify a refresh. This scorer asks Haiku to grade that decision
    against the same prompt the production gate uses.

    `input` should be a dict {"current_persona": ..., "summary": ...}.
    `output` is the system's YES/NO call (or the new persona / None).
    `expected` is the human-labeled ground truth (YES / NO).
    """
    global _persona_gate_scorer
    if _persona_gate_scorer is None:
        _persona_gate_scorer = _build_persona_gate()
    return _persona_gate_scorer(
        output=output, input=input, expected=expected, **kwargs
    )


# Lazy-built for the same reason as the persona gate — autoevals is optional.
_faithfulness_scorer: Any = None

# Eval-only judge prompt (no production equivalent, unlike persona_gate which
# reuses the live gate prompt), so it lives inline here rather than in the
# `app/prompts` registry — that registry is for prompts the backend sends to
# Claude in production. Mustache dotted access (`{{input.context}}`) is
# supported by autoevals' chevron renderer.
_FAITHFULNESS_TEMPLATE = """\
You are a strict fact-checker for an Albania travel assistant.

You are given the REFERENCE CONTEXT the assistant was shown, the USER QUESTION,
and the assistant's ANSWER. Decide whether the answer is faithful to the
reference context.

Faithful means:
- The answer does NOT contradict any fact in the reference context.
- The answer does NOT invent specific, verifiable details — place names,
  prices, opening hours, distances, phone numbers, dates — that are absent
  from the reference context.

General, widely-known statements that are not specific verifiable claims
(e.g. "Albania uses the lek", "the coast is warm in summer") are acceptable
even when they do not appear in the context.

REFERENCE CONTEXT:
{{input.context}}

USER QUESTION:
{{input.message}}

ANSWER:
{{output}}

Is the answer faithful to the reference context?
A = FAITHFUL (no contradictions and no fabricated specifics)
B = UNFAITHFUL (contradicts the context, or fabricates specific verifiable details)
"""


def _build_faithfulness() -> Any:
    """Construct the faithfulness LLMClassifier on first call.

    `use_cot=True` (unlike the persona gate): spotting a fabricated price or a
    subtle contradiction benefits from the judge reasoning before it commits to
    A/B, and this is an offline eval where the extra Haiku tokens are cheap.
    """
    from autoevals import LLMClassifier  # type: ignore[import-not-found]

    return LLMClassifier(
        name="answer_faithfulness",
        prompt_template=_FAITHFULNESS_TEMPLATE,
        choice_scores={"A": 1.0, "B": 0.0},
        model="claude-haiku-4-5-20251001",
        use_cot=True,
    )


def answer_faithfulness(
    output: Any, input: Any = None, **_: Any
) -> Any:
    """LLM judge: is the assistant's answer grounded in the retrieved context?

    Returns 1.0 (faithful) / 0.0 (unfaithful). Accepts two input shapes so the
    same scorer serves both the offline judge-calibration suite and the live
    system-grounding suite:

      - **calibration** — `output` is the answer string; `input` is
        `{"message", "context"}` with the context pre-baked in the fixture.
      - **live** — `output` is `{"answer", "context"}` produced by
        `faithfulness_live_task` (it retrieves the context at run time); `input`
        carries just `{"message"}`.

    Either way the underlying judge always sees the same three fields
    (`message`, `context`, `answer`), so the prompt template stays stable.

    Scope: this catches contradictions and fabricated specifics (prices, hours,
    names) — the harmful hallucinations for a travel app. It deliberately does
    NOT require every sentence to be quoted from context, because Travel Buddy
    blends retrieved chunks with the model's general Albania knowledge; a strict
    extractive-faithfulness judge would false-positive on legitimate
    background. The `expected` label in calibration fixtures is for human
    triage in the Braintrust UI; the judge does not consume it.
    """
    message = input.get("message", "") if isinstance(input, dict) else ""
    if isinstance(output, dict):
        answer = output.get("answer", "")
        context = output.get("context", "")
    else:
        answer = output if isinstance(output, str) else ""
        context = input.get("context", "") if isinstance(input, dict) else ""

    global _faithfulness_scorer
    if _faithfulness_scorer is None:
        _faithfulness_scorer = _build_faithfulness()
    return _faithfulness_scorer(
        output=answer, input={"message": message, "context": context}
    )


# ---------------------------------------------------------------------- #
# output_guardrail — offline calibration for the production guard        #
# ---------------------------------------------------------------------- #

# Lazy-built on first call — autoevals is an optional dev dependency.
_output_guardrail_scorer: Any = None


def _build_output_guardrail() -> Any:
    """Construct the output guardrail LLMClassifier on first call.

    Loads the production prompt from the registry so the eval template stays
    in sync with what the live guard (`app/domain/output_guard.py`) sends.
    `use_cot=False`: the classification is binary and the prompt is already
    unambiguous — no chain-of-thought overhead needed in the eval loop.
    """
    from autoevals import LLMClassifier

    from app.prompts import get as get_prompt

    system_text = get_prompt("output_guard.system").text
    # Mustache template: {{output}} is the assistant reply under review.
    # We append the REPLY block so the judge sees the same structure as the
    # production messages.create() call (system sets the rules; user message
    # is the reply body).
    template = system_text + "\n\nREPLY:\n{{output}}"
    return LLMClassifier(
        name="output_guardrail",
        prompt_template=template,
        choice_scores={"SAFE": 1.0, "REVIEW": 0.0},
        model="claude-haiku-4-5-20251001",
        use_cot=False,
    )


def output_guardrail(
    output: Any, input: Any = None, expected: Any = None, **_: Any
) -> Any:
    """LLM judge: does the classifier correctly label this reply?

    `output` is the assistant reply string. `expected` should be a dict
    with a `"label"` key: `{"label": "safe"}` or `{"label": "review"}`.

    Returns 1.0 when the judge's label matches expected, 0.0 otherwise.

    This is the offline calibration suite — it verifies that the judge
    discriminates correctly across curated fixtures. The production guard
    (`check_output_guardrail` in `app/domain/output_guard.py`) makes the
    same Haiku call on every live turn; keeping the template in the shared
    registry ensures both paths stay in sync.
    """
    global _output_guardrail_scorer
    if _output_guardrail_scorer is None:
        _output_guardrail_scorer = _build_output_guardrail()
    return _output_guardrail_scorer(output=output, input=input, expected=expected)
