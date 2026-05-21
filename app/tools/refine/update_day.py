"""Patch a day's metadata (title / base_city)."""

UPDATE_DAY_TOOL = {
    "name": "update_day",
    "description": (
        "Patch a single day's metadata. Use to rename a day or change its "
        "base city after it's been created. `day_number` is intentionally "
        "NOT patchable — renumber via delete + add if needed. Omit a field "
        "to leave it untouched; pass `null` to clear it."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["day_id"],
        "properties": {
            "day_id": {
                "type": "string",
                "description": "UUID of the day to patch.",
            },
            "title": {
                "type": ["string", "null"],
                "description": "Short label for the day. null clears.",
            },
            "base_city": {
                "type": ["string", "null"],
                "description": "City the day is anchored in. null clears.",
            },
        },
    },
}
