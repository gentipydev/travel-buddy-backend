"""Travel Buddy chat evaluation primitives.

Exports the scorers from `scorers.py` so Braintrust experiment files
(`evals/runners/`) and inline online-scoring callers can do
`from app.evals import itinerary_schema_valid` rather than reaching into
the submodule.

This is the canonical import path — keep scorer names stable, since the
Braintrust UI's online-scoring rules reference them by these slugs.
"""
from app.evals.scorers import (
    answer_faithfulness,
    geo_boundary_adherence,
    itinerary_schema_valid,
    output_guardrail,
    persona_gate,
    refine_fidelity,
    retrieval_hit_rate,
    retrieval_mrr,
    stop_count_cap,
)

__all__ = [
    "answer_faithfulness",
    "geo_boundary_adherence",
    "itinerary_schema_valid",
    "output_guardrail",
    "persona_gate",
    "refine_fidelity",
    "retrieval_hit_rate",
    "retrieval_mrr",
    "stop_count_cap",
]
