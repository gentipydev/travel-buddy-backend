-- Run in the Supabase SQL Editor after 001_enable_pgvector.sql.
-- pgcrypto (for gen_random_uuid) is enabled by default on Supabase.

-- Users: local mirror of Supabase Auth. supabase_uid links to auth.users.id.
CREATE TABLE IF NOT EXISTS public.users (
    id SERIAL PRIMARY KEY,
    supabase_uid TEXT UNIQUE,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS users_supabase_uid_idx ON public.users (supabase_uid);
CREATE INDEX IF NOT EXISTS users_email_idx ON public.users (email);

-- User preferences (1:1 with users).
CREATE TABLE IF NOT EXISTS public.user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    budget TEXT DEFAULT 'mid-range',
    travel_style JSONB DEFAULT '[]'::jsonb,
    food JSONB DEFAULT '[]'::jsonb,
    pace TEXT DEFAULT 'moderate',
    interests JSONB DEFAULT '[]'::jsonb,
    language TEXT DEFAULT 'en',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS user_preferences_user_id_idx ON public.user_preferences (user_id);

-- Places: Albanian destinations, restaurants, hotels, etc.
CREATE TABLE IF NOT EXISTS public.places (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    google_place_id TEXT UNIQUE,
    name TEXT NOT NULL,
    city TEXT,
    region TEXT,
    category TEXT,
    tags TEXT[],
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    rating DOUBLE PRECISION,
    price_level INTEGER,
    photos JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS places_google_place_id_idx ON public.places (google_place_id);
CREATE INDEX IF NOT EXISTS places_city_idx ON public.places (city);
CREATE INDEX IF NOT EXISTS places_region_idx ON public.places (region);
CREATE INDEX IF NOT EXISTS places_category_idx ON public.places (category);

-- Chunks: embedded text passages for RAG.
CREATE TABLE IF NOT EXISTS public.chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    place_id UUID REFERENCES public.places(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    language TEXT DEFAULT 'en',
    source TEXT,
    embedding vector(1024),
    meta JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS chunks_place_id_idx ON public.chunks (place_id);
CREATE INDEX IF NOT EXISTS chunks_language_idx ON public.chunks (language);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON public.chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Trips: user-saved itineraries.
CREATE TABLE IF NOT EXISTS public.trips (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    itinerary JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS trips_user_id_idx ON public.trips (user_id);
