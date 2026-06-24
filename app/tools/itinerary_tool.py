"""Anthropic tool definition for emitting structured itineraries.

Claude calls `emit_itinerary` when the user asks for a day-by-day trip
plan — whether it spans a single day or several. The tool's JSON `input`
is forwarded to the Flutter
client as a parallel `event: itinerary` SSE frame and rendered as an
itinerary card; the conversational text continues to stream live as
ordinary `data:` events that render in the chat bubble. We never execute
this tool server-side — `stop_reason: tool_use` is treated as terminal,
not as a callback request, so no `tool_result` follow-up turn is needed.
"""

ITINERARY_TOOL = {
    "name": "emit_itinerary",
    "description": (
        "Emit a structured itinerary alongside your text reply. "
        "Call this when the user asks for an itinerary, a day-by-day "
        "schedule, or a routed trip — whether it spans a single day or "
        "many. A one-day plan of two or more sequenced stops counts. "
        "Do NOT call it for single recommendations, factual questions, "
        "opinions, greetings, casual chat, or a one-off place suggestion. "
        "When you call this tool, you MUST also write a brief, warm "
        "conversational reply in the same response — never call the tool "
        "in silence."
        "\n\n"
        "CRITICAL — where the prose lives (this is a live, streaming UI):\n\n"
        "The structured stops you emit here are streamed, one at a time, "
        "onto a live map AND into the chat as a per-stop list, the instant "
        "each stop is produced. The map marker and the chat row for a stop "
        "are rendered together from this tool's data. So the descriptive "
        "prose for each stop must live INSIDE that stop — in its `notes` "
        "field — not in a separate free-text block.\n\n"
        "Each stop's `notes` is the warm, vivid, 1–3 sentence description "
        "the traveler reads in the chat: what the place is, what they can "
        "do there, why it's worth their time. Write `notes` for EVERY "
        "stop — this is the conversational experience, not an afterthought. "
        "Do NOT also repeat the day-by-day plan as a long free-text block "
        "after the tool call; that would duplicate what the per-stop list "
        "already shows and break the live sync.\n\n"
        "ORDER — write a SHORT warm intro line as text FIRST (one sentence, "
        "e.g. \"Here's your 3 days exploring Albania ✨\"), THEN call "
        "emit_itinerary with every day and stop, each stop carrying its "
        "`notes` description. You may add one short closing line of text "
        "after the tool call (e.g. an invitation to tweak the plan), but "
        "keep it to a single line — never restate the itinerary. Do NOT "
        "write meta-commentary like \"I've created an itinerary, tap Save "
        "to see it.\""
    ),
    "input_schema": {
        "type": "object",
        "required": ["title", "days"],
        "properties": {
            "title": {
                "type": "string",
                "description": (
                    "Destination-focused title for the trip card "
                    "(e.g. 'Tirana', 'Discovering Tirana', "
                    "'Albanian Riviera Escape'). Do NOT bake in the "
                    "day count or duration — the UI shows it as a "
                    "separate pill, so a prefix like '3 Days in …' "
                    "duplicates information."
                ),
            },
            "summary": {"type": "string"},
            "start_date": {
                "type": "string",
                "description": (
                    "ISO 8601 calendar date (YYYY-MM-DD) for day 1. "
                    "Only include `start_date` if the user gave specific "
                    "dates (e.g., 'from June 12', 'next weekend starting "
                    "Saturday'). Do NOT set it for vague or open-ended "
                    "requests like 'a 3-day trip'."
                ),
            },
            "days": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["day_number", "stops"],
                    "properties": {
                        "day_number": {"type": "integer", "minimum": 1},
                        "title": {"type": "string"},
                        "base_city": {"type": "string"},
                        "stops": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["name", "kind"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "kind": {
                                        "type": "string",
                                        "enum": [
                                            "sight",
                                            "beach",
                                            "hike",
                                            "meal",
                                            "stay",
                                            "transport",
                                            "activity",
                                        ],
                                    },
                                    "city": {"type": "string"},
                                    "address": {
                                        "type": "string",
                                        "description": (
                                            "Full street address. Only "
                                            "include if you are confident "
                                            "in the exact address. If "
                                            "unsure, omit the field — the "
                                            "client will display name + "
                                            "city instead."
                                        ),
                                    },
                                    "duration_minutes": {"type": "integer", "minimum": 0},
                                    "arrival_time": {
                                        "type": "string",
                                        "description": "HH:MM (24h).",
                                    },
                                    "end_time": {
                                        "type": "string",
                                        "description": (
                                            "HH:MM (24h). Paired with "
                                            "`arrival_time` to render time "
                                            "ranges. Omit when the stop "
                                            "duration is open-ended."
                                        ),
                                    },
                                    "transportation": {
                                        "type": "string",
                                        "enum": [
                                            "walk",
                                            "taxi",
                                            "tram",
                                            "bus",
                                            "car",
                                            "other",
                                        ],
                                        "description": (
                                            "How the traveler arrives at "
                                            "this stop from the previous "
                                            "one. Default to `walk` for "
                                            "nearby stops. Omit when "
                                            "unclear."
                                        ),
                                    },
                                    "notes": {
                                        "type": "string",
                                        "description": (
                                            "REQUIRED for every stop. The "
                                            "warm, vivid 1–3 sentence "
                                            "description shown beside this "
                                            "stop in the live chat list: "
                                            "what the place is, what the "
                                            "traveler can do there, and why "
                                            "it's worth visiting. This is "
                                            "the per-stop prose the user "
                                            "reads while the plan streams — "
                                            "write it for each stop, not "
                                            "just logistics."
                                        ),
                                    },
                                    "lat": {"type": "number"},
                                    "lng": {"type": "number"},
                                    "google_place_id": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}
