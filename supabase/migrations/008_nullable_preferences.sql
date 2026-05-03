-- Make opinionated preference fields nullable and strip their hardcoded
-- defaults. The application now writes NULL for unset values; Claude
-- infers what's missing from the conversation rather than starting from
-- a wrong assumption (e.g. "mid-range", "solo", "moderate").

ALTER TABLE public.user_preferences
    ALTER COLUMN budget DROP NOT NULL,
    ALTER COLUMN budget DROP DEFAULT,
    ALTER COLUMN pace DROP NOT NULL,
    ALTER COLUMN pace DROP DEFAULT;

-- Idempotent re-add for cohorts that haven't run 007. No-op in production.
ALTER TABLE public.user_preferences ADD COLUMN IF NOT EXISTS group_type TEXT;
ALTER TABLE public.user_preferences ADD COLUMN IF NOT EXISTS accommodation TEXT;
ALTER TABLE public.user_preferences ADD COLUMN IF NOT EXISTS transport TEXT;
ALTER TABLE public.user_preferences ADD COLUMN IF NOT EXISTS travel_persona TEXT;
