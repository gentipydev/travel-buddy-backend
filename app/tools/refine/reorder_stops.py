"""Reorder the stops within a single day."""

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
