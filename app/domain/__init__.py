"""Business-logic layer.

Two sub-domains today:
  - `app.domain.trips` — trip CRUD (load/serialize), mutations (apply_*),
    enrichment (Google Places + Directions + static map), and the geo
    helpers (Albania bounding-box + haversine).
  - `app.domain.conversations` — conversation persistence (get_or_create,
    append_messages, build_history), rolling summarization + title
    generation, and the persona refresh background task.

`chat_pipeline.py` lives at this level (not under either subdomain)
because it composes both — loading prefs/conversation/trip context, then
returning a `ChatTurnContext` for the chat agent to stream against.
"""
