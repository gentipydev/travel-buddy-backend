"""Shared AsyncAnthropic client.

One singleton for every call site (chat agent, persona generator, title
gen, summarization, persona refresh). Keeping it in one module means a
future client-level concern — retries, custom transports, OpenTelemetry
hooks — touches one file instead of seven.
"""
import httpx
from anthropic import AsyncAnthropic

from app.config import settings
from app.observability import bt_wrap_anthropic

# Explicit timeouts and retries instead of SDK defaults:
#   - connect=5s catches network-down fast; the 60s overall read covers
#     the long-tail of a multi-turn streaming response.
#   - max_retries=2 retries on 429 / 5xx with the SDK's built-in
#     exponential backoff. Capped so a sustained outage doesn't pin a
#     request handler indefinitely.
# `bt_wrap_anthropic` is a no-op when BRAINTRUST_ENABLED=false; when
# enabled, every `client.messages.*` and `client.beta.messages.*` call
# emits a child span under whatever parent span is active on the
# request — no per-callsite instrumentation needed.
client = bt_wrap_anthropic(
    AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        max_retries=2,
        timeout=httpx.Timeout(60.0, connect=5.0),
    )
)
