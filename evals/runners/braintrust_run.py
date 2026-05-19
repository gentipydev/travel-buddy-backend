"""Braintrust eval runner for Travel Buddy chat regressions.

Three suites map onto the three categories documented in `evals/README.md`:
  1. itinerary regression — uses `itinerary_schema_valid` + `geo_boundary_adherence`
  2. refine fidelity — uses `refine_fidelity`
  3. persona gate — uses `persona_gate` (LLM-as-judge, Haiku)

Local run:
    export BRAINTRUST_API_KEY=...  # same key the backend uses
    export BRAINTRUST_ENABLED=true
    cd travel-buddy-backend
    ./.venv/bin/python -m evals.runners.braintrust_run

CI run (see `.github/workflows/backend-evals.yml` — Task #6):
    Triggered on PRs touching app/prompts/**/*.md, app/tools/**,
    app/agents/chat_agent.py, app/domain/chat_pipeline.py, app/llm/**.

The `task` for itinerary + refine suites must call the real chat pipeline
(see TODO below). Until that's wired up, the runner exercises the
deterministic scorers against the fixture's `expected` block as a
self-test — this verifies the eval-loading + scorer-invocation plumbing
works, then real Anthropic calls can drop in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from braintrust import Eval  # type: ignore[import-not-found]

from app.evals import (
    geo_boundary_adherence,
    itinerary_schema_valid,
    persona_gate,
    refine_fidelity,
)


_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _load_jsonl(name: str) -> list[dict[str, Any]]:
    """Load one JSONL fixture file. Strips any leading-underscore keys
    (e.g. `_note`) from `expected` so they don't leak into Braintrust as
    metadata — they're authoring comments, not ground truth."""
    rows: list[dict[str, Any]] = []
    with open(_FIXTURES / name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row.get("expected"), dict):
                row["expected"] = {
                    k: v
                    for k, v in row["expected"].items()
                    if not k.startswith("_")
                }
            rows.append(row)
    return rows


# ---------------------------------------------------------------------- #
# Task functions                                                         #
# ---------------------------------------------------------------------- #


async def itinerary_task(input: dict[str, Any]) -> dict[str, Any]:
    """Run one /chat/stream turn and return the emitted itinerary payload.

    TODO: replace this stub with a real ASGI call. The shape:

        from httpx import AsyncClient, ASGITransport
        from app.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream("POST", "/chat/stream",
                                     json={"message": input["message"]}) as r:
                # parse SSE: capture `event: itinerary` payload
                ...

    Until that's wired, return an empty dict — every scorer will fail
    deterministically, which is the right "not implemented" signal.
    """
    return {}


def refine_task(input: dict[str, Any]) -> dict[str, Any]:
    """Run a refine turn and return the post-refine trip JSON.

    TODO: same as itinerary_task — needs the ASGI client + a way to
    seed the saved_trip referenced by `input["saved_trip_id"]`.
    """
    return {}


def persona_gate_task(input: dict[str, Any]) -> str:
    """Run the live persona gate and return its YES / NO decision.

    TODO: import and call `_should_refresh_persona` from
    `app.domain.conversations.persona`. The function is async — wrap
    appropriately when wiring up.
    """
    return "NO"


# ---------------------------------------------------------------------- #
# Eval definitions                                                       #
# ---------------------------------------------------------------------- #


def main() -> None:
    """Define and run all three eval suites. `Eval(...)` registers each
    with Braintrust and logs results under the project's experiments tab."""
    Eval(
        "travel-buddy",
        experiment_name="itinerary-regression",
        data=lambda: _load_jsonl("itinerary_regression.jsonl"),
        task=itinerary_task,
        scores=[itinerary_schema_valid, geo_boundary_adherence],
        metadata={"suite": "itinerary-regression"},
    )

    Eval(
        "travel-buddy",
        experiment_name="refine-fidelity",
        data=lambda: _load_jsonl("refine_fidelity.jsonl"),
        task=refine_task,
        scores=[refine_fidelity],
        metadata={"suite": "refine-fidelity"},
    )

    Eval(
        "travel-buddy",
        experiment_name="persona-gate",
        data=lambda: _load_jsonl("persona_gate.jsonl"),
        task=persona_gate_task,
        scores=[persona_gate],
        metadata={"suite": "persona-gate"},
    )


if __name__ == "__main__":
    main()
