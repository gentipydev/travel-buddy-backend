"""Third-party API integrations.

Public API:
  - `cohere_client.get_cohere()` — shared Cohere v2 AsyncClient singleton
  - `google_places` — three-tier place enrichment (curated → cache → API)
  - `google_directions` — routed polylines with snap-to-roads fallback
  - `static_maps` — signed Google Static Maps URL builder
  - `supabase_storage` — Supabase Storage helpers (buckets for photos,
    chat attachments, trip exports)

Each integration is fail-soft for the chat path: a Google or Cohere
outage degrades RAG / enrichment but never blocks user-visible streaming.
"""

__all__: list[str] = []
