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
import json

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

_HTTP_OPTIONS = types.HttpOptions(
    timeout=_TIMEOUT_MS,
    retry_options=_RETRY_OPTIONS,
)


def _vertex_credentials():
    """Service-account credentials for Vertex mode, or None to use ADC.

    When `GOOGLE_SERVICE_ACCOUNT_JSON` holds a raw key-file JSON (the simplest
    setup on Render — paste it as an env var), build explicit credentials from
    it. Otherwise return None so the SDK falls back to Application Default
    Credentials (a `GOOGLE_APPLICATION_CREDENTIALS` file path, or the GCP
    metadata server when running on Google infrastructure).
    """
    raw = settings.google_service_account_json.strip()
    if not raw:
        return None
    from google.oauth2 import service_account

    info = json.loads(raw)
    return service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def _build_client() -> genai.Client:
    """Construct the singleton, picking the LLM surface from settings.

    Vertex AI (aiplatform.googleapis.com) is the server-side surface — it does
    NOT edge-block datacenter egress IPs, which is why the Render-hosted backend
    must use it (the Gemini Developer API HTML-403s Render's IP regardless of
    billing). Locally, leave `GEMINI_USE_VERTEX=false` so the Developer-API key
    path keeps working from a residential IP. The call sites are unchanged: the
    google-genai SDK exposes the same `generate_content` / streaming API and the
    same `gemini-3.5-flash` model id on both surfaces.
    """
    if settings.gemini_use_vertex:
        return genai.Client(
            vertexai=True,
            project=settings.google_cloud_project or None,
            location=settings.google_cloud_location or None,
            credentials=_vertex_credentials(),
            http_options=_HTTP_OPTIONS,
        )
    # `genai.Client(api_key="")` raises immediately, which would break import in
    # environments without a key (CI, local without secrets). Fall back to a
    # placeholder so the module imports; `settings.validate_required_keys()`
    # fails fast at startup when the real key is missing, and tests monkeypatch
    # the client anyway.
    return genai.Client(
        api_key=settings.gemini_api_key or "missing-gemini-api-key",
        http_options=_HTTP_OPTIONS,
    )


# `bt_wrap_genai` is a passthrough today (Braintrust ships no genai wrapper);
# kept as the single hook point if that changes.
client = bt_wrap_genai(_build_client())


# Sentinel prefix for an origin-blocked Gemini call. When the request's source
# IP is refused, Google's edge (GFE) load-shedder responds with a full HTML
# "Error 403 (Forbidden)" robot page rather than a JSON API error. Confirmed
# cause on this project: the Gemini *Developer* API edge-blocks the Render
# backend's datacenter egress IP — the byte-identical key/model succeeds from a
# residential IP and paid billing does NOT lift it. The fix is Vertex AI mode
# (`GEMINI_USE_VERTEX=true`), not a key or billing change. `str(exc)` on that
# 403 is a ~1.5KB HTML blob that buries the signal in telemetry and over the SSE
# error channel; `classify_gemini_error` collapses the known case to this stable
# code, and everything else passes through unchanged so an unclassified failure
# is never hidden.
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
            "HTML edge response — the request origin (datacenter/IP) is blocked "
            "by the Developer API, not a model/key/billing error. Use Vertex AI "
            "mode (GEMINI_USE_VERTEX=true) from server hosts"
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
