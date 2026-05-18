"""Remove a single day (and all its stops) by its UUID."""

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
