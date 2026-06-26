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


# Sentinel prefix for an origin-blocked Gemini call. When the request's source
# IP/region is refused, Google's edge (GFE) load-shedder responds with a full
# HTML "Error 403 (Forbidden)" robot page rather than a JSON API error — most
# often the free tier blocking a datacenter IP/region (observed in production
# from the Render-hosted backend, while the identical key/model succeeds from a
# residential IP). `str(exc)` on that is a ~1.5KB HTML blob that buries the
# signal in telemetry and over the SSE error channel. `classify_gemini_error`
# collapses the known case to this stable code; everything else passes through
# unchanged so an unclassified failure is never hidden.
GEMINI_ACCESS_DENIED = "gemini_access_denied"


def classify_gemini_error(exc: BaseException) -> str:
    """Map a Gemini call failure to a stable, log-friendly error string.

    Returns `"gemini_access_denied: ..."` for the origin-blocked 403-with-HTML
    case (see `GEMINI_ACCESS_DENIED`), and the raw `str(exc)` for anything else
    — so this never swallows or relabels an error it doesn't recognise.
    """
    text = str(exc)
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    is_403 = status == 403 or text.lstrip().startswith("403")
    is_html = "<html" in text.lower() or "Error 403 (Forbidden)" in text
    if is_403 and is_html:
        return (
            f"{GEMINI_ACCESS_DENIED}: Gemini returned 403 Forbidden with an "
            "HTML edge response — the request origin is blocked (free-tier "
            "datacenter-IP/region block or key restriction), not a model error"
        )
    return text


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
