-- Run in the Supabase SQL Editor.
-- Curated Albanian artifacts surfaced by the mobile app's destination screens
-- (highlights carousel, collectibles, search). Image bytes live in the
-- `artifact-photos` Storage bucket; `image_path` is the bucket-relative key.

CREATE TABLE IF NOT EXISTS public.artifacts (
    id                TEXT PRIMARY KEY,
    destination_slug  TEXT NOT NULL,
    category          TEXT NOT NULL CHECK (category IN ('highlight', 'collectible')),
    sort_order        INTEGER NOT NULL DEFAULT 0,
    title             TEXT NOT NULL,
    date_text         TEXT,
    period            TEXT,
    country           TEXT,
    culture           TEXT,
    medium            TEXT,
    dimension         TEXT,
    classification    TEXT,
    object_type       TEXT,
    object_begin_year INTEGER,
    object_end_year   INTEGER,
    image_path        TEXT NOT NULL,
    -- Bundled-asset icon category for collectibles UI (jewelry|statue|vase|textile|scroll|picture).
    -- Ignored for highlights. Maps to assets/images/collectibles/{name}.png on mobile.
    icon_name         TEXT,
    search_text       TSVECTOR
);

CREATE INDEX IF NOT EXISTS artifacts_destination_idx
    ON public.artifacts (destination_slug, category, sort_order);

CREATE INDEX IF NOT EXISTS artifacts_search_idx
    ON public.artifacts USING GIN (search_text);

-- Keep search_text in sync with the user-visible text fields. `simple` is
-- intentional — Albanian/English mix means a stemmed config would lose
-- recall on proper nouns (place names, cultural terms).
CREATE OR REPLACE FUNCTION public.artifacts_refresh_search_text()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_text :=
        setweight(to_tsvector('simple', coalesce(NEW.title, '')),          'A') ||
        setweight(to_tsvector('simple', coalesce(NEW.classification, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(NEW.object_type, '')),    'B') ||
        setweight(to_tsvector('simple', coalesce(NEW.culture, '')),        'C') ||
        setweight(to_tsvector('simple', coalesce(NEW.country, '')),        'C') ||
        setweight(to_tsvector('simple', coalesce(NEW.medium, '')),         'D') ||
        setweight(to_tsvector('simple', coalesce(NEW.period, '')),         'D');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS artifacts_search_text_trigger ON public.artifacts;
CREATE TRIGGER artifacts_search_text_trigger
    BEFORE INSERT OR UPDATE ON public.artifacts
    FOR EACH ROW
    EXECUTE FUNCTION public.artifacts_refresh_search_text();
