"""Move a single stop to a different day within the same trip."""
from app.schemas import MAX_STOPS_PER_DAY

MOVE_STOP_TOOL = {
    "name": "move_stop",
    "description": (
        "Move an existing stop to a different day within the same trip, "
        "preserving its id, times, completed_at, and any enrichment. "
        "`sort_order` is the 0-based target position on the new day; omit to "
        "append, or pass a value >= len(target.stops) to append. Source-day "
        "sort_orders are compacted automatically. Same-day moves are "
        "rejected — use `reorder_stops` for that. Use when the user asks "
        "to push a stop to another day (e.g. 'move Berat Castle to Day 3'). "
        f"Soft guideline: target days default to ≤{MAX_STOPS_PER_DAY} stops "
        "for AI-initiated moves; follow an explicit user override."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["stop_id", "target_day_id"],
        "properties": {
            "stop_id": {
                "type": "string",
                "description": "UUID of the stop to move.",
            },
            "target_day_id": {
                "type": "string",
                "description": "UUID of the destination day on the same trip.",
            },
            "sort_order": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Optional 0-based target position. Omit to append."
                ),
            },
        },
    },
}
