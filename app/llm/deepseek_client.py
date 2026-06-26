"""Shared DeepSeek client — the OPTIONAL chat fallback provider.

DeepSeek exposes an OpenAI-compatible API, so we reach it through the
`openai` SDK pointed at `https://api.deepseek.com`. This is the ONLY use of
the `openai` package; no OpenAI models are ever called.

Mirrors `gemini_client`'s one-singleton-for-every-call-site shape so a
future client-level concern (timeouts, transports, tracing) touches one
file. The key is OPTIONAL: when `DEEPSEEK_API_KEY` is unset the fallback is
fully disabled (`deepseek_enabled()` is False and nothing here is called),
but the module must still import — so we fall back to a placeholder key the
same way `gemini_client` does, since `AsyncOpenAI(api_key="")` raises.
"""
from openai import AsyncOpenAI

from app.config import settings

# DeepSeek's OpenAI-compatible base URL. The `openai` SDK appends the
# standard `/chat/completions` path under this root.
_BASE_URL = "https://api.deepseek.com"

# Overall request timeout (seconds). Matches the spirit of gemini_client's
# 60s ceiling so a hung fallback connection can't pin a request handler.
_TIMEOUT_SECONDS = 60.0

# `AsyncOpenAI(api_key="")` raises at construction, which would break import
# in environments without the (optional) key. Use a placeholder so the module
# imports; `deepseek_enabled()` gates every real call on the genuine key.
_api_key = settings.deepseek_api_key or "missing-deepseek-api-key"

client = AsyncOpenAI(
    api_key=_api_key,
    base_url=_BASE_URL,
    timeout=_TIMEOUT_SECONDS,
)


def deepseek_enabled() -> bool:
    """True only when a real DEEPSEEK_API_KEY is configured.

    Every fallback entry point checks this first, so an unset key means the
    chat path behaves exactly as Gemini-only — no DeepSeek call is ever made.
    """
    return bool(settings.deepseek_api_key)
