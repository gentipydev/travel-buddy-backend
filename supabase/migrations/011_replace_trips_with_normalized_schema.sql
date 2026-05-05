-- Pre-launch: no real users. Drop and replace the trips JSONB blob with a
-- normalized 3-table schema (trips → trip_days → trip_stops).
DROP TABLE IF EXISTS public.trips CASCADE;

CREATE TABLE public.trips (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id INTEGER NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  summary TEXT,
  start_date DATE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX trips_user_updated_idx ON public.trips (user_id, updated_at DESC);

CREATE TABLE public.trip_days (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  trip_id UUID NOT NULL REFERENCES public.trips(id) ON DELETE CASCADE,
  day_number INTEGER NOT NULL CHECK (day_number >= 1),
  title TEXT,
  base_city TEXT,
  UNIQUE (trip_id, day_number)
);
CREATE INDEX trip_days_trip_idx ON public.trip_days (trip_id, day_number);

CREATE TABLE public.trip_stops (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  trip_day_id UUID NOT NULL REFERENCES public.trip_days(id) ON DELETE CASCADE,
  sort_order INTEGER NOT NULL,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,
  city TEXT,
  address TEXT,
  duration_minutes INTEGER,
  arrival_time TEXT,
  end_time TEXT,
  transportation TEXT,
  notes TEXT,
  lat DOUBLE PRECISION,
  lng DOUBLE PRECISION,
  google_place_id TEXT,
  UNIQUE (trip_day_id, sort_order)
);
CREATE INDEX trip_stops_day_idx ON public.trip_stops (trip_day_id, sort_order);
