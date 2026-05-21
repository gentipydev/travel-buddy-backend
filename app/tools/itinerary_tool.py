"""Anthropic tool definition for emitting structured multi-day itineraries.

Claude calls `emit_itinerary` when (and only when) the user asks for a
multi-day trip plan. The tool's JSON `input` is forwarded to the Flutter
client as a parallel `event: itinerary` SSE frame and rendered as an
itinerary card; the conversational text continues to stream live as
ordinary `data:` events that render in the chat bubble. We never execute
this tool server-side — `stop_reason: tool_use` is treated as terminal,
not as a callback request, so no `tool_result` follow-up turn is needed.
"""

ITINERARY_TOOL = {
    "name": "emit_itinerary",
    "description": (
        "Emit a structured multi-day itinerary alongside your text reply. "
        "Call this ONLY when the user asks for a multi-day plan, a "
        "day-by-day schedule, or a routed trip across two or more days. "
        "Do NOT call it for single recommendations, factual questions, "
        "opinions, greetings, casual chat, or trivial single-day asks. "
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
        "Write the text first as you normally would: warm, descriptive, "
        "day by day, stop by stop. Then emit the tool call with the "
        "structured equivalent. Both in the same response."
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
