"""Travel Buddy chat evaluation primitives.

Exports the scorers from `scorers.py` so Braintrust experiment files
(`evals/runners/`) and inline online-scoring callers can do
`from app.evals import itinerary_schema_valid` rather than reaching into
the submodule.

This is the canonical import path — keep scorer names stable, since the
Braintrust UI's online-scoring rules reference them by these slugs.
"""
from app.evals.scorers import (
    geo_boundary_adherence,
    itinerary_schema_valid,
    persona_gate,
    refine_fidelity,
    stop_count_cap,
)

__all__ = [
    "geo_boundary_adherence",
    "itinerary_schema_valid",
    "persona_gate",
    "refine_fidelity",
    "stop_count_cap",
]
