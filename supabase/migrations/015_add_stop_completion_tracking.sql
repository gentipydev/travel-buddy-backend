-- 015_add_stop_completion_tracking.sql
--
-- Persist when an itinerary stop has been visited. NULL = not visited.
-- Written by the mobile auto-detector (foreground location + time-window
-- match) or by the user via the manual override popover on the rail node.
-- Server is single source of truth; the mobile client mirrors via the
-- existing PATCH /trips/{trip_id}/stops/{stop_id} endpoint (handler
-- `apply_update_stop` already accepts any allowlisted field via setattr).
--
-- Index supports the day-progress aggregation ("3 / 5 visited") rendered
-- on the trip-detail day header. selectinload already pulls all stops in
-- one query so this is forward-looking for server-side aggregation if we
-- ever add a /trips/{id}/progress endpoint.

ALTER TABLE public.trip_stops
  ADD COLUMN completed_at TIMESTAMPTZ NULL;

CREATE INDEX IF NOT EXISTS trip_stops_day_completed_idx
  ON public.trip_stops (trip_day_id, completed_at);
