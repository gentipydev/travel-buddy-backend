-- 013_add_places_cache_and_routes.sql
--
-- Server-side trip enrichment groundwork. Three new tables:
--   places_cache    — Google Place Details responses, keyed by place_id
--                     (90-day TTL). Miss path for places not in the curated
--                     `public.places` table.
--   directions_cache — Routed polylines from Google Directions, keyed by
--                     rounded (origin, destination, mode). Reusable across
--                     trips that share endpoints at ~10m precision.
--   trip_routes     — Per-trip routing rows linking specific stop pairs to
--                     encoded polylines. ON DELETE CASCADE from trip and day
--                     so a trip delete or day delete cleans up automatically.
--
-- Plus two ALTER TABLE additions denormalizing enrichment onto trip_stops
-- (photo_url / rating / formatted_address) and trips (static_map_url) so
-- list/detail reads are a single SELECT — no joins to caches at read time.

CREATE TABLE public.places_cache (
  google_place_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  formatted_address TEXT,
  lat DOUBLE PRECISION NOT NULL,
  lng DOUBLE PRECISION NOT NULL,
  rating DOUBLE PRECISION,
  user_ratings_total INTEGER,
  primary_photo_url TEXT,
  additional_photo_urls JSONB DEFAULT '[]'::jsonb,
  types JSONB DEFAULT '[]'::jsonb,
  raw_response JSONB,
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '90 days')
);
CREATE INDEX places_cache_expires_at_idx ON public.places_cache (expires_at);

CREATE TABLE public.directions_cache (
  cache_key TEXT PRIMARY KEY,
  encoded_polyline TEXT NOT NULL,
  distance_meters INTEGER,
  duration_seconds INTEGER,
  travel_mode TEXT NOT NULL,
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '90 days')
);
CREATE INDEX directions_cache_expires_at_idx ON public.directions_cache (expires_at);

CREATE TABLE public.trip_routes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  trip_id UUID NOT NULL REFERENCES public.trips(id) ON DELETE CASCADE,
  trip_day_id UUID NOT NULL REFERENCES public.trip_days(id) ON DELETE CASCADE,
  from_stop_id UUID NOT NULL REFERENCES public.trip_stops(id) ON DELETE CASCADE,
  to_stop_id UUID NOT NULL REFERENCES public.trip_stops(id) ON DELETE CASCADE,
  encoded_polyline TEXT NOT NULL,
  distance_meters INTEGER,
  duration_seconds INTEGER,
  travel_mode TEXT NOT NULL DEFAULT 'driving',
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX trip_routes_trip_idx ON public.trip_routes (trip_id);
CREATE INDEX trip_routes_day_idx ON public.trip_routes (trip_day_id);

ALTER TABLE public.trip_stops
  ADD COLUMN photo_url TEXT,
  ADD COLUMN rating DOUBLE PRECISION,
  ADD COLUMN formatted_address TEXT;

ALTER TABLE public.trips
  ADD COLUMN static_map_url TEXT;
