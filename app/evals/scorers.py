"""Braintrust scorers for Travel Buddy chat evals.

Each scorer is a pure function returning 0.0..1.0 (or a dict with `score`
and `metadata`). The same function runs offline in `Eval(...)` experiments
AND online via Braintrust's project-level scoring rules — no duplication
between modes.

Public scorers:

  - `itinerary_schema_valid` — deterministic. Output must pass the same
    Pydantic validator the live chat handler uses (`validate_itinerary`).
  - `geo_boundary_adherence` — deterministic. When the user's message
    names a single Albanian city, every emitted stop's `city` must match.
    This is the regression catcher for the 2026-05-18 incident where
    "3 days in Tirana" leaked Kruja stops — both inside Albania, so a
    country-bbox check missed it.
  - `refine_fidelity` — deterministic. Verifies that a refine turn only
    mutated the stop UUIDs the instruction implies.
  - `persona_gate` — LLM-as-judge (Haiku). Did the system correctly
    decide whether to refresh the persona?

Naming: scorers are referenced by these slugs in the Braintrust UI when
configuring online scoring rules, so don't rename without updating the
project settings.
"""
from __future__ import annotations

import unicodedata
from typing import Any

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
