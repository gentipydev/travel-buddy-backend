import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from google.genai import types
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.engine import get_session
from app.db.models import User, UserPreferences
from app.deps import require_local_user
from app.domain.usage_budget import (
    check_global_ceiling,
    check_user_budget,
    record_usage,
)
from app.llm import client, extract_usage
from app.observability import compute_cost
from app.prompts import get as _get_prompt
from app.rate_limit import limiter, user_or_ip_key
from app.schemas import UpdatePreferencesRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/preferences", tags=["preferences"])

# Sourced from app/prompts/persona/system.md via the registry.
PERSONA_SYSTEM_PROMPT = _get_prompt("persona.system").text


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
@limiter.limit("10/hour", key_func=user_or_ip_key)
async def generate_persona(
    request: Request,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    # Per-user 10/hour ceiling. The endpoint makes a non-cached LLM
    # call (~300 input + 512 output tokens) and is normally hit
    # 0-2× per session, so the ceiling is hard to reach legitimately;
    # the limit catches a misbehaving client or a credentials leak before
    # it racks up serious cost. `Request` is in the signature because
    # SlowAPI's decorator inspects it for the rate-limit key.
    #
    # Cost gates: the same per-user daily budget and global kill-switch
    # as /chat/stream (app/domain/usage_budget.py), so a leaked credential
    # can't sidestep the chat budget by hammering this endpoint instead.
    await check_global_ceiling(db)
    await check_user_budget(db, str(user.supabase_uid))
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
    response = await client.aio.models.generate_content(
        model=settings.model_chat,
        contents=prefs_text,
        config=types.GenerateContentConfig(
            system_instruction=PERSONA_SYSTEM_PROMPT,
            max_output_tokens=512,
        ),
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    new_persona = (response.text or "").strip()
    in_tok, out_tok, _cache_creation, cache_read = extract_usage(
        getattr(response, "usage_metadata", None)
    )
    logger.info(
        "gemini.persona_generated",
        extra={
            "model": settings.model_chat,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
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

    # Fail-soft usage accounting into `usage_daily` (same ledger the
    # budget gates above read).
    await record_usage(
        db,
        str(user.supabase_uid),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read_input_tokens=cache_read,
        estimated_cost_usd=compute_cost(
            model=settings.model_chat,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cache_read,
        ),
    )

    return {"travel_persona": new_persona}
