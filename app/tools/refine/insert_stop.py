"""Insert a new stop into a specific day."""
from app.schemas import MAX_STOPS_PER_DAY
from app.tools.refine._shared import STOP_CREATE_PROPS

INSERT_STOP_TOOL = {
    "name": "insert_stop",
    "description": (
        "Insert a new stop into a specific day. Address the day by its "
        "`day_id` (UUID from the trip JSON). Omit `sort_order` to append at "
        "the end of the day; provide it to insert at a position (existing "
        "stops at that index and after shift up by 1). Use for 'add a "
        "lunch stop on day 2', 'squeeze a museum between the two beaches', "
        f"etc. Soft guideline: AI-initiated plans aim for ≤{MAX_STOPS_PER_DAY} "
        "stops per day; follow the user if they explicitly ask to exceed that."
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
            **STOP_CREATE_PROPS,
        },
    },
}
