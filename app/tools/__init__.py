"""Claude tool-use schemas.

Two kinds of tool today:

  - `itinerary_tool` (the `emit_itinerary` tool) — full multi-day plan
    emission. Offered on every chat turn; Claude is instructed to call it
    only when the user asks for a multi-day plan.
  - `refine/*` — seven surgical patch tools (`update_stop`, `insert_stop`,
    `delete_stop`, `reorder_stops`, `add_day`, `delete_day`, `update_trip`)
    offered only on refine turns (when `ctx.refine_trip is not None`).
    These mutate a saved trip in place; the dispatcher (`refine.dispatcher`)
    validates each tool's input against its Pydantic body and applies via
    `app.domain.trips.mutations.apply_*`.

`itinerary_validator.validate_itinerary` is the gate that runs on Claude's
emit_itinerary output before it ships to the Flutter client.
"""
from app.tools.itinerary_tool import ITINERARY_TOOL
from app.tools.itinerary_validator import (
    ItineraryValidationResult,
    validate_itinerary,
)
from app.tools.refine import (
    REFINE_TOOL_NAMES,
    REFINE_TOOLS,
    apply_refine_tool,
)

__all__ = [
    "ITINERARY_TOOL",
    "ItineraryValidationResult",
    "REFINE_TOOL_NAMES",
    "REFINE_TOOLS",
    "apply_refine_tool",
    "validate_itinerary",
]
