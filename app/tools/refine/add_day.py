"""Add a new day to the trip."""
from app.tools.refine._shared import STOP_CREATE_PROPS

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
                    "properties": STOP_CREATE_PROPS,
                },
            },
        },
    },
}
