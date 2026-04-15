from pydantic_settings import BaseSettings
from anthropic import Anthropic


class Settings(BaseSettings):
    anthropic_api_key: str
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/travel_buddy"

    class Config:
        env_file = ".env"


settings = Settings()
client = Anthropic(api_key=settings.anthropic_api_key)
