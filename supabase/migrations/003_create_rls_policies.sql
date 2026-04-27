-- Run after 002_create_tables.sql.
-- The FastAPI backend uses the service_role key which bypasses RLS.
-- These policies protect direct anon-key access (e.g., Flutter via supabase_flutter).

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trips ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.places ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chunks ENABLE ROW LEVEL SECURITY;

-- Places & chunks: publicly readable (catalog data).
DROP POLICY IF EXISTS "places_public_read" ON public.places;
CREATE POLICY "places_public_read" ON public.places
    FOR SELECT USING (true);

DROP POLICY IF EXISTS "chunks_public_read" ON public.chunks;
CREATE POLICY "chunks_public_read" ON public.chunks
    FOR SELECT USING (true);

-- Users: self-access only.
DROP POLICY IF EXISTS "users_self_select" ON public.users;
CREATE POLICY "users_self_select" ON public.users
    FOR SELECT USING (supabase_uid = auth.uid()::text);

DROP POLICY IF EXISTS "users_self_update" ON public.users;
CREATE POLICY "users_self_update" ON public.users
    FOR UPDATE USING (supabase_uid = auth.uid()::text);

-- User preferences: owned via users.supabase_uid.
DROP POLICY IF EXISTS "user_preferences_self_all" ON public.user_preferences;
CREATE POLICY "user_preferences_self_all" ON public.user_preferences
    FOR ALL USING (
        user_id IN (
            SELECT id FROM public.users WHERE supabase_uid = auth.uid()::text
        )
    );

-- Trips: owned via users.supabase_uid.
DROP POLICY IF EXISTS "trips_self_all" ON public.trips;
CREATE POLICY "trips_self_all" ON public.trips
    FOR ALL USING (
        user_id IN (
            SELECT id FROM public.users WHERE supabase_uid = auth.uid()::text
        )
    );
