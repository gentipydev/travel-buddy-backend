"""Shared JSON-Schema enums and property dicts used by every refine tool.

Kept here (not pulled from `app.schemas`) so the tool definitions can be
imported into the registry without dragging in the Pydantic body classes —
the Anthropic tool schemas are static JSON Schema strings, not Python
types, and want to stay that way.
"""

# Reused enums from the original itinerary tool. Keep the JSON Schema literal
# and standalone so this module has no runtime dep on schemas.py.
STOP_KIND_ENUM = [
    "sight", "beach", "hike", "meal", "stay", "transport", "activity",
]
TRANSPORTATION_ENUM = ["walk", "taxi", "tram", "bus", "car", "other"]


# Properties for a fresh stop being inserted/added — mirrors
# TripStopCreate fields (no id, no sort_order — sort_order is set per-tool).
STOP_CREATE_PROPS = {
    "name": {"type": "string"},
    "kind": {"type": "string", "enum": STOP_KIND_ENUM},
    "city": {"type": "string"},
    "address": {"type": "string"},
    "duration_minutes": {"type": "integer", "minimum": 0},
    "arrival_time": {
        "type": "string",
        "description": "HH:MM (24h).",
    },
    "end_time": {
        "type": "string",
        "description": "HH:MM (24h). Paired with `arrival_time`.",
    },
    "transportation": {"type": "string", "enum": TRANSPORTATION_ENUM},
    "notes": {"type": "string"},
    "lat": {"type": "number"},
    "lng": {"type": "number"},
}


# Properties for a partial stop update — every field optional, no required.
# Matches TripStopUpdate's `extra="forbid"` semantics.
STOP_UPDATE_PROPS = dict(STOP_CREATE_PROPS)
