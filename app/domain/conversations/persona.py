"""Background travel-persona refresh.

`maybe_summarize` (in `summarization.py`) fires `refresh_travel_persona`
as an `asyncio.create_task` whenever it compacts a new summary. The task
opens its own DB session, runs a Haiku YES/NO gate against the user's
current persona, and rewrites the persona from recent summaries when
warranted.

Fire-and-forget — never blocks the chat stream. The outer try/except
catches anything unexpected so a bug here cannot bubble into the event
loop.
"""
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update

from app.config import settings
from app.db.engine import async_session
from app.db.models import Conversation, UserPreferences
from app.prompts import get as _get_prompt


logger = logging.getLogger(__name__)


# Persona-refresh tunables. Gate model lives in settings.model_persona_gate
# so a future model_chat upgrade doesn't silently make this YES/NO check
# expensive — override MODEL_PERSONA_GATE in .env to pin it independently.
_PERSONA_REFRESH_COOLDOWN = timedelta(hours=24)
_PERSONA_RECENT_SUMMARIES = 5
_PERSONA_MAX_TOKENS = 150

_PERSONA_GATE_SYSTEM = _get_prompt("persona.gate_system").text
_PERSONA_GATE_USER = _get_prompt("persona.gate_user").text
_PERSONA_REFRESH_SYSTEM = _get_prompt("persona.refresh_system").text


async def _should_refresh_persona(
    summary: str,
    current_persona: str | None,
    client,  # AsyncAnthropic — typed loosely to avoid an import cycle
) -> bool:
    """Cheap Haiku gate: does `summary` reveal something new about the user
    that the current persona doesn't already capture?

    Pure helper — caller owns the DB. Uses settings.model_persona_gate so
    a future model_chat upgrade can't make the gate pricey. Any API
    failure returns False so the refresh is silently skipped.
    """
    if not summary:
        return False
    if not current_persona:
        return True

    try:
        response = await client.messages.create(
            model=settings.model_persona_gate,
            max_tokens=10,
            system=_PERSONA_GATE_SYSTEM,
            messages=[{
                "role": "user",
                "content": _PERSONA_GATE_USER.format(
                    persona=current_persona, summary=summary
                ),
            }],
        )
        answer = "".join(
            b.text for b in response.content if b.type == "text"
        ).strip().upper()
        return answer.startswith("YES")
    except Exception:
        logger.exception("persona refresh gate failed")
        return False


async def refresh_travel_persona(
    user_id: int,
    latest_summary: str,
    client,  # AsyncAnthropic — typed loosely to avoid an import cycle
    model: str,
) -> None:
    """Fire-and-forget background task: enrich travel_persona from recent
    conversation summaries.

    Opens its own AsyncSession via async_session() — the request-scoped
    session that triggered this task is closed by the time we run. All
    failures log and return; the outer try/except catches anything
    unexpected so a bug here cannot bubble into the event loop.

    Guardrail: skips when user_preferences.updated_at is younger than
    _PERSONA_REFRESH_COOLDOWN, which both prevents rapid-fire refreshes
    and defers to a recent manual POST /preferences/persona write.
    """
    try:
        async with async_session() as db:
            prefs_row = await db.execute(
                select(UserPreferences).where(
                    UserPreferences.user_id == user_id
                )
            )
            prefs = prefs_row.scalar_one_or_none()
            if prefs is None:
                return

            if prefs.updated_at is not None:
                age = datetime.now(timezone.utc) - prefs.updated_at
                if age < _PERSONA_REFRESH_COOLDOWN:
                    logger.info(
                        "persona refresh skipped: prefs updated %.1fh ago "
                        "(user_id=%s)",
                        age.total_seconds() / 3600,
                        user_id,
                    )
                    return

            current_persona = prefs.travel_persona

            if not await _should_refresh_persona(
                latest_summary, current_persona, client
            ):
                return

            summaries_row = await db.execute(
                select(Conversation.summary)
                .where(
                    Conversation.user_id == user_id,
                    Conversation.summary.is_not(None),
                )
                .order_by(Conversation.updated_at.desc())
                .limit(_PERSONA_RECENT_SUMMARIES)
            )
            summaries = [s for s in summaries_row.scalars().all() if s]
            if not summaries:
                return
            transcript = "\n\n---\n\n".join(summaries)

            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=_PERSONA_MAX_TOKENS,
                    system=_PERSONA_REFRESH_SYSTEM,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Existing persona: {current_persona or 'None set yet'}"
                            f"\n\nRecent conversation summaries:\n{transcript}"
                        ),
                    }],
                )
                new_persona = "".join(
                    b.text for b in response.content if b.type == "text"
                ).strip()
            except Exception:
                logger.exception(
                    "persona refresh generation failed (user_id=%s)", user_id
                )
                return

            if not new_persona:
                return

            await db.execute(
                update(UserPreferences)
                .where(UserPreferences.user_id == user_id)
                .values(travel_persona=new_persona)
            )
            await db.commit()
            logger.info("travel_persona refreshed for user_id=%s", user_id)
    except Exception:
        logger.exception(
            "refresh_travel_persona task crashed (user_id=%s)", user_id
        )
