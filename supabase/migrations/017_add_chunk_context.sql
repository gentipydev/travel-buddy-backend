-- Contextual retrieval: add a metadata header column to chunks and fold it into
-- the lexical index.
--
-- `chunks.context` holds a short, deterministically-templated blurb (place name,
-- city, region/county, category, tags) built by app/rag/contextual.py. It is
-- prepended to the embedded text (scripts/embed_chunks.py) and folded into the
-- generated `search_text` so structured metadata the prose omits — "Vlorë
-- County", "beach", "unesco" — becomes searchable by BOTH retrieval arms. It is
-- NOT shown to the model: app/rag/retrieval.py:retrieve_context joins `content`.
--
-- After applying: re-run scripts/chunk_places.py (populates context, clears
-- stale embeddings) then scripts/embed_chunks.py (re-embeds context + content).
-- The lexical index backfills automatically; until the re-embed runs the vector
-- arm degrades to lexical-only (already fail-soft).

ALTER TABLE public.chunks ADD COLUMN IF NOT EXISTS context TEXT;

-- A generated column's expression can't be altered in place, so drop it (which
-- cascades its GIN index) and recreate over context || content.
ALTER TABLE public.chunks DROP COLUMN IF EXISTS search_text;
ALTER TABLE public.chunks
    ADD COLUMN search_text TSVECTOR
    GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(context, '') || ' ' || coalesce(content, ''))
    ) STORED;

CREATE INDEX IF NOT EXISTS chunks_search_text_idx
    ON public.chunks USING GIN (search_text);
