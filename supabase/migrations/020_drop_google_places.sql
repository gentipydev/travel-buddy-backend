-- 020_drop_google_places.sql
--
-- Retires the Google Maps Platform integration. The Google Cloud project is
-- gone, so every Google-sourced value in the DB is either unusable (photo
-- URLs signed against a dead key, static-map URLs) or was never real to begin
-- with (`google_place_id`, which the model invented — the backend always
-- re-resolved stops by name anyway).
--
-- Replaces `places_cache` (keyed by Google place id) with `geocode_cache`
-- (keyed by the normalized "{name}|{city}" query the Mapbox resolver uses).
-- `directions_cache` and `trip_routes` are deliberately kept: their polylines
-- are precision-5 encoded, which is exactly what Mapbox Directions returns.

DROP TABLE IF EXISTS public.places_cache;

CREATE TABLE public.geocode_cache (
  query_key         TEXT PRIMARY KEY,   -- "{norm_name}|{norm_city}"
  name              TEXT NOT NULL,
  formatted_address TEXT,
  lat  DOUBLE PRECISION NOT NULL,
  lng  DOUBLE PRECISION NOT NULL,
  raw_response JSONB,
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '90 days')
);
CREATE INDEX geocode_cache_expires_at_idx ON public.geocode_cache (expires_at);

ALTER TABLE public.trip_stops DROP COLUMN IF EXISTS google_place_id;
ALTER TABLE public.places     DROP COLUMN IF EXISTS google_place_id;

-- Evict stale enrichment so saved trips re-resolve against Mapbox on their
-- next save / refine instead of serving photo and thumbnail URLs that now
-- 403. Coordinates are kept — they're still correct.
UPDATE public.trip_stops SET photo_url = NULL, photos = '[]'::jsonb;
UPDATE public.trips SET static_map_url = NULL;
