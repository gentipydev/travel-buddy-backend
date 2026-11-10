"""Shared AsyncAnthropic client.

One singleton for every call site (chat-agent streaming, persona endpoint,
title gen, summarization, persona gate/refresh, output guard, query
rewrite). Keeping it in one module means a future client-level concern —
timeouts, custom transports, OpenTelemetry hooks — touches one file
instead of seven.

`generate_message` is the one-shot, non-streaming primitive; `generate_text`
is the thin text-only wrapper the auxiliary call sites share (a system
instruction + a single user string in, the model's plain text out). The
streaming chat agent uses `client` directly (`client.messages.stream`).
"""
import logging

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import Message

from app.config import settings
from app.observability import bt_wrap_anthropic

logger = logging.getLogger(__name__)

# Explicit timeouts and retries instead of SDK defaults:
#   - connect=5s catches network-down fast; the 60s overall read covers the
#     long tail of a multi-turn streaming response.
#   - max_retries=3 retries 408/409/429 and every 5xx (529 `overloaded_error`
#     included) with the SDK's built-in exponential backoff. Capped so a
#     sustained outage can't pin a request handler indefinitely.
#
# `anthropic.Timeout` is an `httpx2.Timeout` — the SDK's HTTP layer is httpx2,
# not httpx. Never pass an object from the `httpx` package here (the rest of
# this app still uses plain httpx for Mapbox); it is rejected at
# construction.
_TIMEOUT = anthropic.Timeout(60.0, connect=5.0)


def _build_client() -> AsyncAnthropic:
    """Construct the singleton.

    `AsyncAnthropic(api_key="")` would defer failure to the first call and
    `api_key=None` reads the ambient env; pass a placeholder so the module
    imports cleanly in environments without a key (CI, local without
    secrets). `settings.validate_required_keys()` fails fast at startup when
    the real key is missing, and tests monkeypatch the client anyway.
    """
    if not settings.anthropic_api_key:
        logger.warning(
            "anthropic: ANTHROPIC_API_KEY empty — client built with a "
            "placeholder; every call will 401 until it is set."
        )
    return AsyncAnthropic(
        api_key=settings.anthropic_api_key or "missing-anthropic-api-key",
        max_retries=3,
        timeout=_TIMEOUT,
    )


# `bt_wrap_anthropic` is a no-op when BRAINTRUST_ENABLED=false; when enabled,
# every `client.messages.*` call emits a child span under whatever parent span
# is active on the request — no per-callsite instrumentation needed.
client = bt_wrap_anthropic(_build_client())


# Stable prefixes for the failure modes worth naming. Everything else falls
# through as `str(exc)` so an unclassified failure is never hidden — see
# `classify_llm_error`.
ANTHROPIC_AUTH_FAILED = "anthropic_auth_failed"
ANTHROPIC_RATE_LIMITED = "anthropic_rate_limited"
ANTHROPIC_OVERLOADED = "anthropic_overloaded"
ANTHROPIC_UNAVAILABLE = "anthropic_unavailable"


def classify_llm_error(exc: BaseException) -> str:
    """Map an Anthropic call failure to a stable, log-friendly error string.

    Collapses the four failure modes an operator actually acts on to a stable
    prefix + a short hint, and returns the raw `str(exc)` for anything else —
    so this never swallows or relabels an error it doesn't recognise. The
    result is emitted on the SSE `event: error` frame, so keep it one line.
    """
    detail = " ".join(str(exc).split())[:300]
    if isinstance(
        exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError
    ):
        return (
            f"{ANTHROPIC_AUTH_FAILED}: the API key is missing, revoked, or lacks "
            f"access to this model — check ANTHROPIC_API_KEY | raw: {detail}"
        )
    if isinstance(exc, anthropic.RateLimitError):
        return (
            f"{ANTHROPIC_RATE_LIMITED}: request rate or token quota exceeded; the "
            f"SDK already retried | raw: {detail}"
        )
    if isinstance(exc, anthropic.APITimeoutError | anthropic.APIConnectionError):
        return f"{ANTHROPIC_UNAVAILABLE}: could not reach the API | raw: {detail}"
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        # 529 is Anthropic's dedicated "overloaded" code; 5xx generally means
        # the far side, not our payload.
        return (
            f"{ANTHROPIC_OVERLOADED}: upstream returned {exc.status_code} — "
            f"transient, retry later | raw: {detail}"
        )
    return str(exc)


def extract_usage(usage) -> tuple[int, int, int, int]:
    """Map an Anthropic `Usage` to the app's four-field accounting:
    `(input_tokens, output_tokens, cache_creation_input_tokens,
    cache_read_input_tokens)`.

    Anthropic's `input_tokens` already EXCLUDES cached tokens, so the three
    counters are additive and no arithmetic is needed. The cache fields are
    unset on turns where caching wasn't engaged, so read them defensively.
    Returns all zeros when usage is missing — telemetry must never crash.
    """
    if usage is None:
        return (0, 0, 0, 0)
    return (
        getattr(usage, "input_tokens", None) or 0,
        getattr(usage, "output_tokens", None) or 0,
        getattr(usage, "cache_creation_input_tokens", None) or 0,
        getattr(usage, "cache_read_input_tokens", None) or 0,
    )


def message_text(message: Message) -> str:
    """Concatenate the text blocks of a `Message` (empty string if none).

    A response may also carry `tool_use` (and, on thinking models, `thinking`)
    blocks; those are skipped rather than stringified.
    """
    return "".join(b.text for b in message.content if b.type == "text")


async def generate_message(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
) -> Message:
    """One-shot, non-streaming generation returning the raw `Message`.

    Used directly by call sites that need `usage` back for cost accounting
    (the persona endpoint). Callers own their own timeout (`asyncio.wait_for`)
    and error handling — this helper deliberately does neither so each site
    keeps its fail-soft contract.
    """
    return await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )


async def generate_text(
    *,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
) -> str:
    """One-shot, non-streaming generation: a system instruction + a single
    user string in, the model's plain text out (empty string if none).

    Shared by the auxiliary call sites (query rewrite, output guard, title
    gen, summarization, persona gate/refresh). `max_output_tokens` keeps the
    name the call sites already pass; it maps to the API's `max_tokens`.
    """
    message = await generate_message(
        model=model, system=system, user=user, max_tokens=max_output_tokens
    )
    return message_text(message)
