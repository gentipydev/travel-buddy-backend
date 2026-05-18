"""Patch top-level trip fields (title / summary / start_date)."""

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
