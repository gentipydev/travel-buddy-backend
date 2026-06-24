"""Output safety guardrail for the chat pipeline.

Classifies the assembled assistant reply as "safe" or "review".
Fail-open: any exception (timeout, API error, unexpected shape) returns
"safe" so a guardrail outage never blocks or degrades the user experience.

The guard is intentionally narrow — it fires only on clear violations:
system-prompt leaks, harmful content, and severe off-topic responses.
Ambiguous cases default to "safe".
"""
from __future__ import annotations

import asyncio
import logging

from app.config import settings
from app.llm import generate_text
from app.observability.request_context import trace_var
from app.prompts import get as get_prompt

logger = logging.getLogger(__name__)

_GUARD_TIMEOUT_SECONDS = 5.0


def _mark_failed_open() -> None:
    """Flag the ambient RequestTrace so a fail-open is no longer invisible.

    No-op when no trace is bound (e.g. unit tests that call the guard
    directly) — mirrors the `app.observability.stage()` helper's contract.
    Never raises: telemetry must not break the fail-open path.
    """
    try:
        trace = trace_var.get()
        if trace is not None:
            trace.output_guard_failed_open = True
    except Exception:
        pass


async def check_output_guardrail(text: str) -> str:
    """Return "safe" or "review". Never raises.

    Calls the model with the `output_guard.system` prompt. Response is a
    single token: "SAFE" or "REVIEW". Any failure path (timeout, API error,
    empty response, unexpected token) returns "safe" — fail-open is the
    contract.

    The result is telemetry only; the reply has already streamed to the
    client before this function is called.
    """
    if not text or not text.strip():
        return "safe"
    try:
        system_prompt = get_prompt("output_guard.system").text
        label_raw = (
            await asyncio.wait_for(
                generate_text(
                    model=settings.model_chat,
                    system=system_prompt,
                    user=text,
                    max_output_tokens=8,
                ),
                timeout=_GUARD_TIMEOUT_SECONDS,
            )
        ).strip().upper()
        return "review" if label_raw == "REVIEW" else "safe"
    except TimeoutError:
        _mark_failed_open()
        logger.warning(
            "output_guardrail: LLM call timed out after %.1fs — fail open",
            _GUARD_TIMEOUT_SECONDS,
        )
        return "safe"
    except Exception:
        _mark_failed_open()
        logger.warning("output_guardrail: exception — fail open", exc_info=True)
        return "safe"
