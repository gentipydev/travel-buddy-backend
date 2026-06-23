-- 019_nullable_user_email.sql
-- Allow users.email to be NULL for anonymous (guest) accounts.
--
-- The web app signs visitors in anonymously (Supabase anonymous auth), and
-- those JWTs carry no email. get_or_create_user (app/db/engine.py) previously
-- inserted email = '' for them. Because users.email is UNIQUE, the FIRST guest
-- took the '' slot and EVERY subsequent guest hit a duplicate-key error on
-- users_email_key — a 500 on POST /users/me/sync that cascaded to "User not
-- synced" 404s on all authed endpoints.
--
-- Fix: make the column nullable and store NULL for guests. Postgres treats
-- NULLs as distinct under a UNIQUE constraint (NULLS DISTINCT is the default),
-- so any number of guests coexist while real emails stay unique.

ALTER TABLE public.users ALTER COLUMN email DROP NOT NULL;

-- Normalize any pre-existing empty-string emails to NULL so the old guest
-- row(s) no longer occupy the '' unique slot.
UPDATE public.users SET email = NULL WHERE email = '';
