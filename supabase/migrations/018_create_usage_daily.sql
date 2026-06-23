-- 018_create_usage_daily.sql
-- Per-user daily LLM usage accounting for cost-budget enforcement
-- (app/domain/usage_budget.py). Keyed on the VERIFIED Supabase sub
-- (not local users.id) so authenticated-but-unsynced users are still
-- metered, with no FK coupling to the /users/me/sync flow. Dates are
-- UTC. Written exclusively by the backend via the service key; RLS is
-- enabled with NO policies so anon/authenticated client roles can
-- neither read nor write rows.

CREATE TABLE IF NOT EXISTS public.usage_daily (
    supabase_uid TEXT NOT NULL,
    usage_date DATE NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    cache_creation_input_tokens BIGINT NOT NULL DEFAULT 0,
    cache_read_input_tokens BIGINT NOT NULL DEFAULT 0,
    estimated_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (supabase_uid, usage_date)
);

-- The global daily spend kill-switch sums today's rows on every cache
-- refresh (60s); this index keeps that SUM a range scan.
CREATE INDEX IF NOT EXISTS usage_daily_date_idx
    ON public.usage_daily (usage_date);

ALTER TABLE public.usage_daily ENABLE ROW LEVEL SECURITY;
