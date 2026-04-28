-- Run in the Supabase SQL Editor.
-- Adds server-side chat conversation persistence.

CREATE TABLE IF NOT EXISTS public.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES public.users(id) ON DELETE CASCADE,
    title TEXT,
    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS conversations_user_id_idx
    ON public.conversations (user_id);

-- Used by GET /conversations (list ordered by recency).
CREATE INDEX IF NOT EXISTS conversations_user_updated_idx
    ON public.conversations (user_id, updated_at DESC);
