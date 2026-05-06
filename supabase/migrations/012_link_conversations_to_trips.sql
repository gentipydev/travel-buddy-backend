-- 012_link_conversations_to_trips.sql
--
-- Couples a saved trip to its "Refine with Buddy" chat conversation 1:1, so a
-- user can reopen the same chat whenever they want to ask Claude to revise the
-- trip. Regular (non-refine) conversations keep trip_id NULL and are unaffected.
--
-- ON DELETE SET NULL (not CASCADE): if the user deletes a trip, the chat
-- history survives — the conversation is general-purpose, not just refinement.
-- The partial unique index enforces 1:1 only when trip_id is set, so the many
-- pre-existing rows with NULL don't conflict.

ALTER TABLE public.conversations
    ADD COLUMN IF NOT EXISTS trip_id UUID NULL
        REFERENCES public.trips(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS conversations_trip_id_idx
    ON public.conversations (trip_id);

CREATE UNIQUE INDEX IF NOT EXISTS conversations_trip_id_uniq
    ON public.conversations (trip_id)
    WHERE trip_id IS NOT NULL;
