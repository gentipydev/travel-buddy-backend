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
import logging
from pathlib import Path
from typing import Any

from braintrust import Eval  # type: ignore[import-not-found]

from app.evals import (
    geo_boundary_adherence,
    itinerary_schema_valid,
    persona_gate,
    refine_fidelity,
)

logger = logging.getLogger(__name__)


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


async def _parse_sse_itinerary_event(stream_iter) -> dict[str, Any]:
    """Pull the `event: itinerary` payload out of a /chat/stream SSE response.

    The chat agent emits one `event: itinerary` (or `event: itinerary_invalid`)
    per turn, followed by `data: <JSON>`. We accumulate raw bytes, split on
    blank-line frame boundaries, and return the first matching JSON dict.
    Returns an empty dict when the stream ends with no itinerary event
    (which is the correct signal for a non-multi-day prompt).
    """
    buf = b""
    found: dict[str, Any] = {}
    async for chunk in stream_iter:
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            text = frame.decode("utf-8", errors="replace")
            event_line = ""
            data_lines: list[str] = []
            for line in text.split("\n"):
                if line.startswith("event:"):
                    event_line = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if event_line == "itinerary" and data_lines:
                try:
                    found = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    found = {}
                return found
    return found


async def itinerary_task(input: dict[str, Any]) -> dict[str, Any]:
    """Run one /chat/stream turn and return the emitted itinerary payload.

    Hits the live Anthropic API via the in-process ASGI transport — no
    network listener required. Anonymous chat is fine here; the fixture
    inputs never name a real user, and the agent's behavior on anonymous
    turns is what the scorers measure.
    """
    # Lazy import so a runner without httpx/the app available can still
    # import the module for `--help` etc.
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    message = input.get("message", "")
    if not message:
        return {}
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        timeout=120.0,
    ) as client:
        async with client.stream(
            "POST", "/chat/stream", json={"message": message}
        ) as response:
            if response.status_code != 200:
                logger.warning(
                    "itinerary_task: /chat/stream returned %d for message=%r",
                    response.status_code, message,
                )
                return {}
            return await _parse_sse_itinerary_event(response.aiter_bytes())


async def refine_task(input: dict[str, Any]) -> dict[str, Any]:
    """Run a refine turn and return the post-refine trip JSON.

    Stub: refine requires an authenticated user with a saved trip whose
    UUIDs match the fixture's expected_touched_stop_ids. Wiring this
    needs:
      (a) a test-only Supabase JWT issuer or a `Depends` override
      (b) a DB-seeding helper that creates the saved_trip fixture under a
          known user
      (c) capturing `event: trip_refined` from the SSE stream (mirroring
          `_parse_sse_itinerary_event` for the new event name)
    Each piece is small; the combination is enough lift to warrant its
    own PR. Until then, every refine_fidelity score will be 0.0 — the
    correct "not implemented" signal.
    """
    return {}


async def persona_gate_task(input: dict[str, Any]) -> str:
    """Run the live persona gate and return its YES / NO decision.

    Calls `_should_refresh_persona` directly — same code path the
    production `refresh_travel_persona` background task runs through.
    The scorer compares the result against the fixture's ground-truth
    `expected` field ("YES" / "NO").
    """
    from app.domain.conversations.persona import _should_refresh_persona
    from app.llm import client as llm_client

    summary = input.get("summary", "")
    current_persona = input.get("current_persona")
    should = await _should_refresh_persona(
        summary=summary,
        current_persona=current_persona,
        client=llm_client,
    )
    return "YES" if should else "NO"


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
