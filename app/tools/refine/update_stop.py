"""Change one or more fields of a single existing stop."""
from app.tools.refine._shared import STOP_UPDATE_PROPS

UPDATE_STOP_TOOL = {
    "name": "update_stop",
    "description": (
        "Change one or more fields of a single existing stop. Address the "
        "stop by its `stop_id` (UUID from the trip JSON in the system "
        "prompt). Include only the fields you want to change — `name` and "
        "`kind` cannot be cleared once set. Prefer this over "
        "`emit_itinerary` for any single-stop edit (shorter time, different "
        "name, swap restaurant, correct the coordinates, etc.)."
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
            **STOP_UPDATE_PROPS,
        },
    },
}
