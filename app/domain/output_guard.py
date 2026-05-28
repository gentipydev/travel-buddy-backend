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
from typing import TYPE_CHECKING

from app.config import settings
from app.prompts import get as get_prompt

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

_GUARD_TIMEOUT_SECONDS = 5.0


async def check_output_guardrail(text: str, *, client: AsyncAnthropic) -> str:
    """Return "safe" or "review". Never raises.

    Calls Haiku with the `output_guard.system` prompt. Response is a single
    token: "SAFE" or "REVIEW". Any failure path (timeout, API error, empty
    response, unexpected token) returns "safe" — fail-open is the contract.

    The result is telemetry only; the reply has already streamed to the
    client before this function is called.
    """
    if not text or not text.strip():
        return "safe"
    try:
        system_prompt = get_prompt("output_guard.system").text
        response = await asyncio.wait_for(
            client.messages.create(
                model=settings.model_chat,
                max_tokens=8,
                system=system_prompt,
                messages=[{"role": "user", "content": text}],
            ),
            timeout=_GUARD_TIMEOUT_SECONDS,
        )
        label_raw = (
            getattr(response.content[0], "text", "") if response.content else ""
        ).strip().upper()
        return "review" if label_raw == "REVIEW" else "safe"
    except TimeoutError:
        logger.warning(
            "output_guardrail: Haiku call timed out after %.1fs — fail open",
            _GUARD_TIMEOUT_SECONDS,
        )
        return "safe"
    except Exception:
        logger.warning("output_guardrail: exception — fail open", exc_info=True)
        return "safe"
