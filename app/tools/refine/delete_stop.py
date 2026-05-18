"""Remove a single stop by its UUID."""

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
