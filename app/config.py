from pydantic_settings import BaseSettings, SettingsConfigDict
from anthropic import AsyncAnthropic
from supabase import acreate_client, AsyncClient


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    anthropic_api_key: str
    cohere_api_key: str | None = None

    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/travel_buddy"
    allowed_origins: str = "http://localhost:3000"

    model_chat: str = "claude-haiku-4-5-20251001"
    embedding_model: str = "embed-multilingual-v3.0"
    rerank_model: str = "rerank-multilingual-v3.0"

    # Logging — drives `setup_logging()` in `app/main.py`. `log_format='json'`
    # is the production default (Render Logs UI filters per-field on JSON);
    # set `LOG_FORMAT=text` locally for human-friendly output.
    log_level: str = "INFO"
    log_format: str = "json"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = Settings()
client = AsyncAnthropic(api_key=settings.anthropic_api_key)

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
