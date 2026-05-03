-- Add place_id to trips so saving the same place twice is idempotent.
-- Existing trips with NULL place_id are preserved and ignored by the
-- partial unique index below.

ALTER TABLE public.trips
    ADD COLUMN IF NOT EXISTS place_id UUID
    REFERENCES public.places(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS trips_user_place_idx
    ON public.trips (user_id, place_id);

-- Partial unique: only one row per (user, place) when place_id is set.
-- NULL place_id rows (legacy / future free-form trips) are unaffected.
CREATE UNIQUE INDEX IF NOT EXISTS trips_user_place_uniq
    ON public.trips (user_id, place_id)
    WHERE place_id IS NOT NULL;
