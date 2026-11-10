"""Third-party API integrations.

Public API:
  - `cohere_client.get_cohere()` — shared Cohere v2 AsyncClient singleton
  - `mapbox_geocoding` — three-tier place resolution (curated → cache → API)
  - `mapbox_directions` — routed polylines, cached by coords + mode
  - `mapbox_static` — Mapbox Static Images URL builder
  - `supabase_storage` — Supabase Storage helpers (buckets for photos,
    chat attachments, trip exports)

Each integration is fail-soft for the chat path: a Mapbox or Cohere
outage degrades RAG / enrichment but never blocks user-visible streaming.
"""

__all__: list[str] = []
