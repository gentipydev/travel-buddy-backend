"""Anthropic tool definitions for surgical trip edits during refine.

Each tool mirrors an existing manual-editor endpoint: Claude addresses a stop
or day by its UUID (already injected into the system prompt by
`serialize_trip_for_context`), and the chat dispatcher applies the op via the
matching `app.services.trips_service.apply_*` function. Tools are offered to
Claude only when `ctx.refine_trip is not None`.

Replaces the previous "always re-emit the full TripCreate via emit_itinerary"
flow for small/medium edits — emit_itinerary is still offered alongside these
as the escape hatch for whole-trip rewrites.
"""
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Trip
from app.schemas import (
    StopReorderBody,
    TripDayCreateBody,
    TripStopCreateBody,
    TripStopUpdate,
    TripUpdate,
)
from app.services.trips_service import (
    apply_add_day,
    apply_delete_day,
    apply_delete_stop,
    apply_insert_stop,
    apply_reorder_stops,
    apply_update_stop,
    apply_update_trip,
)

# Reused enums from the original itinerary tool. Keep the JSON Schema literal
# and standalone so this module has no runtime dep on schemas.py.
_STOP_KIND_ENUM = ["sight", "beach", "hike", "meal", "stay", "transport", "activity"]
_TRANSPORTATION_ENUM = ["walk", "taxi", "tram", "bus", "car", "other"]


# Properties for a fresh stop being inserted/added — mirrors
# TripStopCreate fields (no id, no sort_order — sort_order is set per-tool).
_STOP_CREATE_PROPS = {
    "name": {"type": "string"},
    "kind": {"type": "string", "enum": _STOP_KIND_ENUM},
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
    "transportation": {"type": "string", "enum": _TRANSPORTATION_ENUM},
    "notes": {"type": "string"},
    "lat": {"type": "number"},
    "lng": {"type": "number"},
    "google_place_id": {"type": "string"},
}


# Properties for a partial stop update — every field optional, no required.
# Matches TripStopUpdate's `extra="forbid"` semantics.
_STOP_UPDATE_PROPS = dict(_STOP_CREATE_PROPS)


UPDATE_TRIP_TOOL = {
    "name": "update_trip",
    "description": (
        "Patch top-level trip fields (title, summary, start_date). Only "
        "include keys you want to change; everything else stays as-is. Use "
        "this for things like 'rename the trip', 'shift the start date', or "
        "'rewrite the summary' — never for changes to days or stops."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 200},
            "summary": {"type": "string"},
            "start_date": {
                "type": "string",
                "description": "ISO 8601 calendar date (YYYY-MM-DD).",
            },
        },
    },
}


UPDATE_STOP_TOOL = {
    "name": "update_stop",
    "description": (
        "Change one or more fields of a single existing stop. Address the "
        "stop by its `stop_id` (UUID from the trip JSON in the system "
        "prompt). Include only the fields you want to change — `name` and "
        "`kind` cannot be cleared once set. Prefer this over "
        "`emit_itinerary` for any single-stop edit (shorter time, different "
        "name, swap restaurant, attach a `google_place_id`, etc.)."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["stop_id"],
        "properties": {
            "stop_id": {
                "type": "string",
                "description": "UUID of the stop to update.",
            },
            **_STOP_UPDATE_PROPS,
        },
    },
}


DELETE_STOP_TOOL = {
    "name": "delete_stop",
    "description": (
        "Remove a single stop by its `stop_id`. Other stops in the day keep "
        "their existing `sort_order` (gaps are fine). Use when the user "
        "asks to drop a single stop — do not chain this with a re-emit of "
        "the whole trip."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["stop_id"],
        "properties": {
            "stop_id": {
                "type": "string",
                "description": "UUID of the stop to delete.",
            },
        },
    },
}


INSERT_STOP_TOOL = {
    "name": "insert_stop",
    "description": (
        "Insert a new stop into a specific day. Address the day by its "
        "`day_id` (UUID from the trip JSON). Omit `sort_order` to append at "
        "the end of the day; provide it to insert at a position (existing "
        "stops at that index and after shift up by 1). Use for 'add a "
        "lunch stop on day 2', 'squeeze a museum between the two beaches', "
        "etc."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["day_id", "name", "kind"],
        "properties": {
            "day_id": {
                "type": "string",
                "description": "UUID of the target day.",
            },
            "sort_order": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Optional. Position within the day (0-based). Omit to "
                    "append. Inserting at an occupied index shifts later "
                    "stops up by 1."
                ),
            },
            **_STOP_CREATE_PROPS,
        },
    },
}


ADD_DAY_TOOL = {
    "name": "add_day",
    "description": (
        "Add a new day to the trip. `day_number` must be unique within the "
        "trip — pick the next free integer (existing day_numbers are in the "
        "trip JSON). `stops` may be empty if you plan to fill the day with "
        "subsequent `insert_stop` calls. Use for 'add a day 4 in Berat'."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["day_number"],
        "properties": {
            "day_number": {"type": "integer", "minimum": 1},
            "title": {"type": "string"},
            "base_city": {"type": "string"},
            "stops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "kind"],
                    "properties": _STOP_CREATE_PROPS,
                },
            },
        },
    },
}


DELETE_DAY_TOOL = {
    "name": "delete_day",
    "description": (
        "Remove a single day (and all its stops) by its `day_id`. Other "
        "days keep their `day_number` — there is no automatic renumbering. "
        "Use for 'drop day 3', 'we don't need the rest day anymore'."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["day_id"],
        "properties": {
            "day_id": {
                "type": "string",
                "description": "UUID of the day to delete.",
            },
        },
    },
}


REORDER_STOPS_TOOL = {
    "name": "reorder_stops",
    "description": (
        "Reorder the stops within a single day. `order` MUST list every "
        "current stop in the day, each with its desired 0-based "
        "`sort_order`. The set of stop_ids must exactly match what's "
        "currently in the day; sort_order values must be 0..N-1 "
        "contiguous. Use for 'do the museum before lunch', 'swap stops 2 "
        "and 3'."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["day_id", "order"],
        "properties": {
            "day_id": {
                "type": "string",
                "description": "UUID of the day whose stops to reorder.",
            },
            "order": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["stop_id", "sort_order"],
                    "properties": {
                        "stop_id": {"type": "string"},
                        "sort_order": {"type": "integer", "minimum": 0},
                    },
                },
            },
        },
    },
}


# Order matters only for prompt clarity — Anthropic doesn't care.
REFINE_TOOLS = [
    UPDATE_STOP_TOOL,
    INSERT_STOP_TOOL,
    DELETE_STOP_TOOL,
    REORDER_STOPS_TOOL,
    ADD_DAY_TOOL,
    DELETE_DAY_TOOL,
    UPDATE_TRIP_TOOL,
]


REFINE_TOOL_NAMES = {tool["name"] for tool in REFINE_TOOLS}


async def apply_refine_tool(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    name: str,
    raw_input: dict,
) -> Trip:
    """Validate `raw_input` against the matching Pydantic body and apply.

    No-commit: caller owns the transaction so multiple tool calls in one turn
    can be applied atomically. Raises HTTPException (404/409/...) on bad
    addressing and pydantic.ValidationError on bad payload shape — both flow
    up to the chat dispatcher which rolls back and emits `refine_failed`.
    """
    if name == "update_stop":
        stop_id = UUID(str(raw_input["stop_id"]))
        body = TripStopUpdate.model_validate(
            {k: v for k, v in raw_input.items() if k != "stop_id"}
        )
        return await apply_update_stop(
            db,
            trip_id=trip_id,
            user_id=user_id,
            stop_id=stop_id,
            payload=body.model_dump(exclude_unset=True),
        )
    if name == "insert_stop":
        day_id = UUID(str(raw_input["day_id"]))
        body = TripStopCreateBody.model_validate(
            {k: v for k, v in raw_input.items() if k != "day_id"}
        )
        return await apply_insert_stop(
            db,
            trip_id=trip_id,
            user_id=user_id,
            day_id=day_id,
            payload=body.model_dump(),
        )
    if name == "delete_stop":
        stop_id = UUID(str(raw_input["stop_id"]))
        return await apply_delete_stop(
            db, trip_id=trip_id, user_id=user_id, stop_id=stop_id
        )
    if name == "reorder_stops":
        day_id = UUID(str(raw_input["day_id"]))
        body = StopReorderBody.model_validate(
            {"order": raw_input.get("order", [])}
        )
        return await apply_reorder_stops(
            db,
            trip_id=trip_id,
            user_id=user_id,
            day_id=day_id,
            payload=body.model_dump(mode="json", exclude_unset=True),
        )
    if name == "add_day":
        body = TripDayCreateBody.model_validate(raw_input)
        return await apply_add_day(
            db,
            trip_id=trip_id,
            user_id=user_id,
            payload=body.model_dump(),
        )
    if name == "delete_day":
        day_id = UUID(str(raw_input["day_id"]))
        return await apply_delete_day(
            db, trip_id=trip_id, user_id=user_id, day_id=day_id
        )
    if name == "update_trip":
        body = TripUpdate.model_validate(raw_input)
        return await apply_update_trip(
            db,
            trip_id=trip_id,
            user_id=user_id,
            payload=body.model_dump(exclude_unset=True, mode="json"),
        )
    raise ValueError(f"Unknown refine tool name: {name}")
