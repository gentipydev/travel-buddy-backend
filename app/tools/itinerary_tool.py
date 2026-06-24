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
        "CRITICAL — text reply behavior when calling this tool:\n\n"
        "Calling this tool does NOT replace your natural-language reply. "
        "Your text response must still contain the FULL itinerary written "
        "in rich, descriptive prose — every day, every stop, every "
        "recommendation — exactly as you would write it if this tool did "
        "not exist. The tool call is a parallel structured payload that "
        "powers the \"Save itinerary\" button and the persisted trip "
        "card. The text is the conversational experience the user sees "
        "while streaming.\n\n"
        "Do NOT shorten, summarize, or abbreviate your text reply because "
        "you are also calling the tool. Do NOT write meta-commentary like "
        "\"I've created an itinerary for you, tap Save to see it.\" The "
        "user should be able to read the entire plan in the chat without "
        "saving anything.\n\n"
        "ORDER — call this tool FIRST, then write the prose. Emit the "
        "emit_itinerary tool call at the very start of your response "
        "(before any prose), then write the full warm, descriptive, "
        "day-by-day, stop-by-stop plan as text. The structured stops are "
        "streamed onto a live map as you produce them, so they must come "
        "first. Calling the tool first does NOT let you shorten the prose: "
        "the text plan that follows must still be complete — every day, "
        "every stop. Both in the same response."
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
                                    "notes": {"type": "string"},
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
