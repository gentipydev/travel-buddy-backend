"""Shared AsyncAnthropic client.

One singleton for every call site (chat agent, persona generator, title
gen, summarization, persona refresh). Keeping it in one module means a
future client-level concern — retries, custom transports, OpenTelemetry
hooks — touches one file instead of seven.
"""
import httpx
from anthropic import AsyncAnthropic

from app.config import settings

# Explicit timeouts and retries instead of SDK defaults:
#   - connect=5s catches network-down fast; the 60s overall read covers
#     the long-tail of a multi-turn streaming response.
#   - max_retries=2 retries on 429 / 5xx with the SDK's built-in
#     exponential backoff. Capped so a sustained outage doesn't pin a
#     request handler indefinitely.
client = AsyncAnthropic(
    api_key=settings.anthropic_api_key,
    max_retries=2,
    timeout=httpx.Timeout(60.0, connect=5.0),
)
