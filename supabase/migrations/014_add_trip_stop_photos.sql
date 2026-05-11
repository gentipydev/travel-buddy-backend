-- 014_add_trip_stop_photos.sql
--
-- Multi-photo support for trip stops. Adds a `photos` JSONB array column
-- denormalized onto trip_stops so the trip-detail / map-sheet UIs can show
-- a hero + thumb-stack without a join into places_cache. The list mirrors
-- (and includes) the existing single `photo_url` as element 0; the legacy
-- column stays for back-compat with old clients and for code paths that
-- only need the primary photo.
--
-- Backfill is intentionally skipped — every stop will be populated on the
-- next enrichment pass (every save / refine), so existing rows just default
-- to '[]' until then. The Flutter `effectivePhotos` getter falls back to
-- `[photo_url]` when the list is empty, so old clients and un-enriched stops
-- keep rendering the single photo.

ALTER TABLE public.trip_stops
  ADD COLUMN photos JSONB NOT NULL DEFAULT '[]'::jsonb;
