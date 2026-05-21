"""Validate + apply one refine tool call.

The chat agent (`app/agents/chat_agent.py` and `app/routers/chat.py`) calls
`apply_refine_tool` once per `tool_use` block in the assistant's final
message. Each call:

  1. Picks the matching Pydantic body schema (`TripStopUpdate`,
     `TripDayCreateBody`, etc.) from `app.schemas` and validates the raw
     tool input.
  2. Dispatches to the appropriate `apply_*` mutation in
     `app.domain.trips.mutations`.

No-commit: the caller owns the transaction so multiple tool calls within
one assistant turn can land in a single commit. The router rolls the
whole turn back on any failure and emits `event: refine_failed`.

Raises `ValueError` on unknown tool names and `pydantic.ValidationError`
on bad payload shape; both flow up to the chat dispatcher.
"""
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Trip
from app.domain.trips.mutations import (
    apply_add_day,
    apply_delete_day,
    apply_delete_stop,
    apply_insert_stop,
    apply_move_stop,
    apply_reorder_stops,
    apply_update_day,
    apply_update_stop,
    apply_update_trip,
)
from app.schemas import (
    StopMoveBody,
    StopReorderBody,
    TripDayCreateBody,
    TripDayUpdate,
    TripStopCreateBody,
    TripStopUpdate,
    TripUpdate,
)


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
    if name == "move_stop":
        stop_id = UUID(str(raw_input["stop_id"]))
        body = StopMoveBody.model_validate(
            {k: v for k, v in raw_input.items() if k != "stop_id"}
        )
        return await apply_move_stop(
            db,
            trip_id=trip_id,
            user_id=user_id,
            stop_id=stop_id,
            payload=body.model_dump(mode="json"),
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
    if name == "update_day":
        day_id = UUID(str(raw_input["day_id"]))
        body = TripDayUpdate.model_validate(
            {k: v for k, v in raw_input.items() if k != "day_id"}
        )
        return await apply_update_day(
            db,
            trip_id=trip_id,
            user_id=user_id,
            day_id=day_id,
            payload=body.model_dump(exclude_unset=True),
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
