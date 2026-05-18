"""Re-exports of the model IDs the call sites use.

`settings.model_chat` etc. live in `app/config.py` so the .env can override
them. Importing them from `app/llm/models` lets a caller depend on the
LLM layer instead of poking into the settings module.

`PROMPT_CACHING_BETA` is the Anthropic beta feature flag name passed via
the `betas=[...]` parameter on `client.beta.messages.stream`. As of
anthropic-python 0.39.0 prompt caching is still gated behind this flag;
when the SDK is upgraded to a GA version this constant can go away.
"""
from app.config import settings

# Convenience accessors. These read the live setting on each access — safe
# because Settings is a singleton at module import time.
MODEL_CHAT = settings.model_chat
MODEL_PERSONA_GATE = settings.model_persona_gate
MODEL_TITLE_GEN = settings.model_title_gen
MODEL_SUMMARIZE = settings.model_summarize

# Beta flag enabling cache_control on system / message blocks. Set on the
# `betas` list when calling `client.beta.messages.stream(...)`.
PROMPT_CACHING_BETA = "prompt-caching-2024-07-31"
