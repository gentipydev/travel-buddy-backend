"""Shared Google Gemini client.

One singleton `genai.Client` for every call site (chat-agent streaming,
persona generator, title gen, summarization, persona refresh, output
guard, query rewrite). Keeping it in one module means a future
client-level concern — timeouts, custom transports, OpenTelemetry hooks —
touches one file instead of seven.

`generate_text` is the thin one-shot helper the auxiliary (non-streaming)
call sites share: a system instruction + a single user string in, the
model's plain text out. The streaming chat agent uses `client` directly
(`client.aio.models.generate_content_stream`).
"""
from google import genai
from google.genai import types

from app.config import settings
from app.observability import bt_wrap_genai

# Overall request timeout (milliseconds). The auxiliary calls also wrap their
# own asyncio.wait_for, but this ceiling keeps a hung connection from pinning
# a request handler.
_TIMEOUT_MS = 60_000

# Transient-error retry. Gemini (especially the newest flash tier) intermittently
# returns 503 UNAVAILABLE "high demand" at request open — observed in production
# breaking whole chat turns. These retries fire BEFORE any tokens stream (the
# 503 surfaces on the open call), so retrying is safe for the streaming path too.
# ~0.5 + 1 + 2 = ~3.5s worst-case added latency before the first token on a spike,
# which beats a hard failure. Also covers the 429 rate-limit and 5xx gateway codes.
_RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=4,
    initial_delay=0.5,
    max_delay=4.0,
    exp_base=2.0,
    jitter=0.3,
    http_status_codes=[429, 500, 502, 503, 504],
)

# `genai.Client(api_key="")` raises immediately, which would break import in
# environments without a key (CI, local without secrets). Fall back to a
# placeholder so the module imports; `settings.validate_required_keys()` fails
# fast at startup when the real key is missing, and tests monkeypatch the
# client anyway.
_api_key = settings.gemini_api_key or "missing-gemini-api-key"

# `bt_wrap_genai` is a passthrough today (Braintrust ships no genai wrapper);
# kept as the single hook point if that changes.
client = bt_wrap_genai(
    genai.Client(
        api_key=_api_key,
        http_options=types.HttpOptions(
            timeout=_TIMEOUT_MS,
            retry_options=_RETRY_OPTIONS,
        ),
    )
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
    gen, summarization, persona gate/refresh, persona endpoint). Callers own
    their own timeout (`asyncio.wait_for`) and error handling — this helper
    deliberately does neither so each site keeps its fail-soft contract.
    """
    response = await client.aio.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
        ),
    )
    return response.text or ""
