from pydantic_settings import BaseSettings, SettingsConfigDict
from supabase import acreate_client, AsyncClient


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # Anthropic API key — the LLM provider for chat + all auxiliary calls.
    # DISTINCT from `mapbox_access_token` below, which is the maps credential;
    # they are separate credentials on separate APIs and must not be shared.
    # Get one at https://console.anthropic.com/settings/keys.
    # Optional (default "") so the module imports without secrets in CI/local;
    # `validate_required_keys` fails fast at startup when it is missing.
    anthropic_api_key: str = ""
    cohere_api_key: str | None = None

    # Braintrust observability — LLM trace logging + scorer evals.
    # No-op when `braintrust_api_key` is empty or `braintrust_enabled` is
    # False (the default), so local dev / CI / PRs without the secret run
    # unchanged. `braintrust_log_user_payloads` is a privacy kill switch:
    # when False, request.message + assistant_text are stripped from
    # spans; token counts and structural metadata still ship.
    braintrust_api_key: str | None = None
    braintrust_project: str = "travel-buddy"
    braintrust_enabled: bool = False
    braintrust_log_user_payloads: bool = True

    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/travel_buddy"
    allowed_origins: str = "http://localhost:3000"

    # Claude Haiku 4.5 — the chat model. 200k context, and the cheapest tier
    # that still handles the `emit_itinerary` / refine tool schemas reliably.
    # Use the exact model id with NO date suffix; a dated variant like
    # `claude-haiku-4-5-20251001` is not the public id. Verify any override
    # against `client.models.list()` — the API 404s on a bad id.
    model_chat: str = "claude-haiku-4-5"
    # Auxiliary call sites — the persona YES/NO gate, conversation title
    # generation, and rolling summarization. Held separate so a future
    # model_chat upgrade doesn't silently make these cheap/frequent calls
    # expensive. Defaults match model_chat; override in .env for A/B tests.
    #
    # NOTE: an override must also exist in `app/observability/pricing.py`.
    # `compute_cost` returns 0.0 for an unknown model id, which would silently
    # zero out the per-user daily budget AND the global kill switch below.
    model_persona_gate: str = "claude-haiku-4-5"
    model_title_gen: str = "claude-haiku-4-5"
    model_summarize: str = "claude-haiku-4-5"
    embedding_model: str = "embed-multilingual-v3.0"
    rerank_model: str = "rerank-multilingual-v3.0"

    # Cohere trial keys are capped at 10 requests/minute across embed + rerank
    # combined. Bursting past it returns 429s, and the RAG layer's fail-soft
    # (embed → lexical-only, rerank → passthrough) would then silently corrupt
    # result quality with no error surfaced. A per-process token-bucket throttle
    # around both Cohere calls keeps bursts under the ceiling; raise this for a
    # production (non-trial) key. `cohere_rate_limit_window_seconds` is the
    # refill window the limit applies over (60s = "per minute").
    cohere_rate_limit_per_minute: int = 10
    cohere_rate_limit_window_seconds: float = 60.0

    # Retrieval precision knob: how many reranked chunks reach the prompt.
    # Raising it trades input tokens for recall (each extra chunk appends its
    # content to the RAG block in `chat_pipeline`); kept modest so the added
    # context can't swamp the reply. Tuned against the retrieval eval suite
    # (`evals/measure_retrieval.py`): at 3 the `vlore-county-recall` pin sits at
    # rerank rank 5 and is missed; 5 surfaces it (suite hit 0.89→0.95) for ~2
    # extra chunks/turn. Higher buys nothing on this suite — the remaining miss
    # (`korce-county-recall`) ranks 10th and is the query-rewrite target instead.
    rerank_top_n: int = 5

    # LLM query rewrite: when True, `retrieve_chunks` first runs a Haiku pass
    # that resolves follow-up coreference (using history) and expands broad
    # queries (e.g. a county into its towns) into specific search terms before
    # hybrid search + rerank. Off by default on a measured tradeoff
    # (`evals/measure_retrieval.py`, 2026-05-28): it takes suite hit 0.95→1.00
    # (fixes `korce-county-recall`, whose chunk reranks last against the bare
    # query) but is MRR-neutral — over-expansion regresses two already-perfect
    # precise queries — and adds a serial Haiku call to time-to-first-token on
    # every retrieval turn. Enable where recall matters more than rank-1
    # precision / added latency.
    query_rewrite_enabled: bool = False

    # Mapbox — forward geocoding (coords + address), Directions (routed
    # polylines) and Static Images (the trip thumbnail), all called from
    # domain.trips.enrichment. A public `pk.` token is enough; it ends up
    # embedded in the persisted static-map URL, so restrict it by URL in the
    # Mapbox account rather than treating it as a secret.
    # Both `_enabled` flags are kill switches for cost control: setting either
    # to false (or leaving the token empty) makes the corresponding service
    # short-circuit to None, and trip saves proceed without that data.
    # Curated `places` lookups bypass the kill switch — they're local-only.
    mapbox_access_token: str = ""
    mapbox_geocoding_enabled: bool = True
    mapbox_directions_enabled: bool = True
    # Mapbox only licenses *storing* geocoding results under `permanent=true`,
    # which has to be enabled on the account. We cache results for 90 days and
    # persist coordinates on trip_stops either way, so this must be turned on
    # before any real launch.
    mapbox_permanent_geocoding: bool = False

    # LLM cost-abuse guards (app/domain/usage_budget.py). Costs are the
    # same estimated-USD numbers as RequestTrace.estimated_cost_usd
    # (app/observability/pricing.py). Setting a budget to 0 disables that
    # check. The global ceiling is the kill-switch of last resort: when
    # today's total estimated spend crosses it, LLM endpoints return 503
    # until UTC midnight. Its DB SUM is cached in-process for
    # `global_budget_cache_seconds`, so overshoot is bounded by one cache
    # window of traffic. `max_concurrent_streams_per_user` caps parallel
    # /chat/stream generators per verified user (0 disables).
    daily_cost_budget_usd_per_user: float = 0.50
    daily_cost_ceiling_usd_global: float = 20.0
    global_budget_cache_seconds: float = 60.0
    max_concurrent_streams_per_user: int = 2

    # Logging — drives `setup_logging()` in `app/main.py`. `log_format='json'`
    # is the production default (Render Logs UI filters per-field on JSON);
    # set `LOG_FORMAT=text` locally for human-friendly output.
    log_level: str = "INFO"
    log_format: str = "json"

    # Sentry — empty string disables init (production no-op). When set, the
    # SDK auto-instruments FastAPI request spans. `sentry_traces_sample_rate`
    # is the perf-span sampler (0.1 = 10%); error events are always sent.
    # The `_scrub_pii` hook in `app/main.py` strips chat content / persona
    # / email / stop names before any event leaves the process.
    sentry_dsn: str = ""
    sentry_traces_sample_rate: float = 0.1
    sentry_environment: str = "development"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    def validate_required_keys(self) -> list[str]:
        # Pydantic already rejects literally-missing required fields, but an
        # empty string passes its `str` check and then fails at first use
        # (e.g. anthropic_api_key="" → 401 on the first stream). Listing the
        # truthy-required keys here fails fast in `lifespan` instead, with all
        # missing keys reported at once for a faster ops loop.
        missing = [
            key for key in (
                "anthropic_api_key",
                "supabase_url",
                "supabase_anon_key",
                "supabase_service_key",
                "database_url",
            ) if not getattr(self, key)
        ]
        return missing


settings = Settings()

_supabase_client: AsyncClient | None = None


async def get_supabase() -> AsyncClient:
    """Lazy async singleton. Service key bypasses RLS — backend-only."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = await acreate_client(
            settings.supabase_url,
            settings.supabase_service_key,
        )
    return _supabase_client
