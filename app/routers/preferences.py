import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import client, settings
from app.db.engine import get_session
from app.db.models import User, UserPreferences
from app.deps import require_local_user
from app.schemas import UpdatePreferencesRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/preferences", tags=["preferences"])

PERSONA_SYSTEM_PROMPT = (
    "You generate concise travel persona summaries for a travel assistant app. "
    "Output exactly 2-3 sentences in second person (\"You are...\"). "
    "Be specific and useful for personalizing travel recommendations. "
    "Describe default tendencies, not rigid rules — the user may deviate "
    "per trip. No preamble, no bullet points, no lists."
)


@router.post("")
async def update_preferences(
    request: UpdatePreferencesRequest,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    result = await db.execute(
        select(UserPreferences).where(UserPreferences.user_id == user.id)
    )
    prefs = result.scalar_one_or_none()

    created = prefs is None
    if prefs:
        for field in [
            "budget",
            "travel_style",
            "food",
            "pace",
            "interests",
            "language",
            "group_type",
            "accommodation",
            "transport",
            "travel_persona",
        ]:
            setattr(prefs, field, getattr(request, field))
    else:
        prefs = UserPreferences(user_id=user.id, **request.model_dump())
        db.add(prefs)

    await db.commit()
    logger.info(
        "preferences.saved",
        extra={"local_user_id": user.id, "created": created},
    )
    return {"status": "updated"}


@router.get("/me")
async def get_my_preferences(
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    result = await db.execute(
        select(UserPreferences).where(UserPreferences.user_id == user.id)
    )
    prefs = result.scalar_one_or_none()
    if not prefs:
        raise HTTPException(status_code=404, detail="Preferences not found")

    return {
        "budget": prefs.budget,
        "travel_style": prefs.travel_style,
        "food": prefs.food,
        "pace": prefs.pace,
        "interests": prefs.interests,
        "language": prefs.language,
        "group_type": prefs.group_type,
        "accommodation": prefs.accommodation,
        "transport": prefs.transport,
        "travel_persona": prefs.travel_persona,
    }


@router.post("/persona")
async def generate_persona(
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    result = await db.execute(
        select(UserPreferences).where(UserPreferences.user_id == user.id)
    )
    prefs = result.scalar_one_or_none()
    if prefs is None or not prefs.has_preferences():
        raise HTTPException(
            status_code=400,
            detail="Not enough signal to generate a persona — set some preferences first",
        )

    saved_persona = prefs.travel_persona
    prefs.travel_persona = None
    prefs_text = prefs.to_prompt()
    prefs.travel_persona = saved_persona

    started = time.perf_counter()
    response = await client.messages.create(
        model=settings.model_chat,
        max_tokens=512,
        system=PERSONA_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prefs_text}],
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    new_persona = "".join(
        b.text for b in response.content if b.type == "text"
    ).strip()
    logger.info(
        "anthropic.persona_generated",
        extra={
            "model": settings.model_chat,
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
            "duration_ms": elapsed_ms,
            "persona_chars": len(new_persona),
        },
    )

    await db.execute(
        update(UserPreferences)
        .where(UserPreferences.user_id == user.id)
        .values(travel_persona=new_persona)
    )
    await db.commit()

    return {"travel_persona": new_persona}
