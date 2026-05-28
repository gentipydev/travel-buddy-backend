-- Run in the Supabase SQL Editor.
-- Hybrid retrieval: add a lexical full-text index to the RAG `chunks` table so
-- the semantic (vector) search in app/rag/embedding.py can be fused with
-- term-based search. Embedding search alone under-retrieves on proper nouns
-- and rare terms — exactly Travel Buddy's domain (Albanian place names like
-- "Theth", "Gjirokastër", terms like "furgon"). The two ranked lists are
-- merged with Reciprocal Rank Fusion before reranking.
--
-- `simple` config matches the artifacts table (migration 010): no stemming,
-- so proper nouns survive intact across the Albanian/English mix. A STORED
-- generated column (vs. the trigger used for artifacts) is enough here because
-- the source is a single column — Postgres backfills every existing row at
-- ADD COLUMN time and keeps it in sync on insert/update with no trigger.

ALTER TABLE public.chunks
    ADD COLUMN IF NOT EXISTS search_text TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED;

CREATE INDEX IF NOT EXISTS chunks_search_text_idx
    ON public.chunks USING GIN (search_text);
