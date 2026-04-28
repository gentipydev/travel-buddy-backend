-- Run in the Supabase SQL Editor after 004_create_storage_rls_policies.sql.
-- Adds the source text column for RAG: chunked + embedded by scripts in later steps.

ALTER TABLE public.places ADD COLUMN IF NOT EXISTS description TEXT;
