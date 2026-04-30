ALTER TABLE public.user_preferences
    ADD COLUMN IF NOT EXISTS group_type TEXT DEFAULT 'solo';

ALTER TABLE public.user_preferences
    ADD COLUMN IF NOT EXISTS accommodation TEXT DEFAULT 'hotel';

ALTER TABLE public.user_preferences
    ADD COLUMN IF NOT EXISTS transport TEXT DEFAULT 'car';

ALTER TABLE public.user_preferences
    ADD COLUMN IF NOT EXISTS travel_persona TEXT;
