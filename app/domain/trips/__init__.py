"""Trips domain: load/serialize, mutate, enrich.

Public surface (most callers want one of these):
  - `serialize_trip(db, trip)` — async chokepoint; enriches + serializes
  - `_load_trip(db, trip_id, user_id)` — auth-checked loader
  - `replace_trip_from_payload` — Claude's `emit_itinerary` full re-emit
  - `apply_update_trip`, `apply_add_day`, `apply_delete_day`,
    `apply_insert_stop`, `apply_update_stop`, `apply_delete_stop`,
    `apply_reorder_stops` — surgical patch tools
  - `serialize_trip_for_context(trip)` — refine-mode system-prompt payload
  - `get_or_create_refine_conversation` — 1:1 Trip↔Conversation
  - `enrich_trip` — best-effort post-save enrichment

The `geo` submodule (Albania bbox + haversine) is also re-exported here.
"""
from app.domain.trips.enrichment import enrich_trip
from app.domain.trips.geo import (
    ALBANIA_LAT_MAX,
    ALBANIA_LAT_MIN,
    ALBANIA_LNG_MAX,
    ALBANIA_LNG_MIN,
    haversine_km,
    is_in_albania,
)
from app.domain.trips.mutations import (
    apply_add_day,
    apply_delete_day,
    apply_delete_stop,
    apply_insert_stop,
    apply_reorder_stops,
    apply_update_stop,
    apply_update_trip,
    replace_trip_from_payload,
)
from app.domain.trips.repository import (
    _load_day,
    _load_stop,
    _load_trip,
    _serialize,
    _touch_trip,
    get_or_create_refine_conversation,
    serialize_trip,
    serialize_trip_for_context,
)

__all__ = [
    "ALBANIA_LAT_MAX",
    "ALBANIA_LAT_MIN",
    "ALBANIA_LNG_MAX",
    "ALBANIA_LNG_MIN",
    "apply_add_day",
    "apply_delete_day",
    "apply_delete_stop",
    "apply_insert_stop",
    "apply_reorder_stops",
    "apply_update_stop",
    "apply_update_trip",
    "enrich_trip",
    "get_or_create_refine_conversation",
    "haversine_km",
    "is_in_albania",
    "replace_trip_from_payload",
    "serialize_trip",
    "serialize_trip_for_context",
]
