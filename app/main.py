import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import client, settings, get_supabase
from app.database import (
    User,
    UserPreferences,
    init_db,
    get_session,
    get_or_create_user,
)
from app.auth.supabase_auth import get_current_user, get_optional_user
from app.services.place_service import detect_place, to_card_dict

MAX_MESSAGE_LENGTH = 4000
MAX_HISTORY_MESSAGES = 20

BASE_SYSTEM_PROMPT = """You are Travel Buddy, an AI travel assistant specializing
exclusively in Albania. You have deep knowledge of every city, village, beach,
mountain, restaurant, hotel, and hidden gem across the country.

You help travelers:
- Plan detailed itineraries across Albania
- Discover local restaurants, cafes, and nightlife
- Find beaches, hiking trails, and cultural sites
- Navigate transport (furgon schedules, car rentals, ferries)
- Understand local customs, tipping, and safety
- Get off-the-beaten-path recommendations locals would give

Personality: Warm, enthusiastic about Albania, like a knowledgeable local friend.
Not generic or corporate.

Rules:
- Only recommend real, verified places
- Always include practical details (hours, prices in LEK, location, how to get there)
- Flag seasonal closures or conditions when relevant
- If unsure about something, say so rather than guessing
- Respond in the language the user writes in

Response length — match the user's message, do not over-deliver:
- Greetings or small talk ("hi", "thanks", "how are you"): reply in 1–2 short
  sentences. Do NOT dump itineraries, tips, or menus unprompted. A friendly hello
  and one short question to learn what they need is enough.
- Simple factual questions ("what's the currency?", "is tap water safe?"):
  1–3 sentences. Answer, done.
- Single recommendation ("best beach near Vlorë?"): one clear pick + 2–3 lines
  of practical info. Offer to expand only if useful.
- Multi-part planning ("3 days in the south"): go long, use structure
  (headers, bullets, day-by-day).

Default to brevity. Expand only when the question genuinely requires it, or
when the user explicitly asks for more detail. Never pad, never list options
the user didn't ask for, never end with multiple follow-up questions — one at
most, and only if it actually helps."""

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await get_supabase()  # warm the async Storage client
    yield


app = FastAPI(title="Travel Buddy API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def build_system_prompt(preferences: UserPreferences | None) -> str:
    if not preferences:
        return BASE_SYSTEM_PROMPT
    return f"""{BASE_SYSTEM_PROMPT}

User preferences — tailor every recommendation to these:
{preferences.to_prompt()}"""


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=MAX_MESSAGE_LENGTH)


class ChatRequest(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_LENGTH)
    conversation_history: list[ChatMessage] = Field(
        default_factory=list, max_length=MAX_HISTORY_MESSAGES
    )
    images: list[str] | None = Field(default=None, max_length=5)


class UpdatePreferencesRequest(BaseModel):
    budget: str = "mid-range"
    travel_style: list[str] = []
    food: list[str] = []
    pace: str = "moderate"
    interests: list[str] = []
    language: str = "en"


async def _require_local_user(supa_user: dict, db: AsyncSession) -> User:
    result = await db.execute(
        select(User).where(User.supabase_uid == supa_user["id"])
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not synced — call POST /users/me/sync first",
        )
    return user


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/users/me")
async def get_me(
    supa_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    user = await _require_local_user(supa_user, db)
    return {
        "id": user.id,
        "supabase_uid": user.supabase_uid,
        "name": user.name,
        "email": user.email,
    }


@app.post("/users/me/sync")
async def sync_me(
    supa_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    user = await get_or_create_user(db, supa_user)
    return {
        "id": user.id,
        "supabase_uid": user.supabase_uid,
        "name": user.name,
        "email": user.email,
    }


@app.post("/preferences")
async def update_preferences(
    request: UpdatePreferencesRequest,
    supa_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    user = await _require_local_user(supa_user, db)

    result = await db.execute(
        select(UserPreferences).where(UserPreferences.user_id == user.id)
    )
    prefs = result.scalar_one_or_none()

    if prefs:
        for field in ["budget", "travel_style", "food", "pace", "interests", "language"]:
            setattr(prefs, field, getattr(request, field))
    else:
        prefs = UserPreferences(user_id=user.id, **request.model_dump())
        db.add(prefs)

    await db.commit()
    return {"status": "updated"}


@app.get("/preferences/me")
async def get_my_preferences(
    supa_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    user = await _require_local_user(supa_user, db)
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
    }


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    supa_user: dict | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_session),
):
    preferences = None
    if supa_user:
        user_result = await db.execute(
            select(User).where(User.supabase_uid == supa_user["id"])
        )
        user = user_result.scalar_one_or_none()
        if user:
            pref_result = await db.execute(
                select(UserPreferences).where(UserPreferences.user_id == user.id)
            )
            preferences = pref_result.scalar_one_or_none()

    system_prompt = build_system_prompt(preferences)

    if request.images:
        user_content: list[dict] = [
            {"type": "image", "source": {"type": "url", "url": u}}
            for u in request.images
        ]
        if request.message:
            user_content.append({"type": "text", "text": request.message})
        user_payload: str | list[dict] = user_content
    else:
        user_payload = request.message

    messages = [m.model_dump() for m in request.conversation_history] + [
        {"role": "user", "content": user_payload}
    ]

    place = await detect_place(request.message, db)
    card_payload = json.dumps(to_card_dict(place)) if place is not None else None

    async def stream():
        try:
            if card_payload is not None:
                yield f"event: place_card\ndata: {card_payload}\n\n"
            async with client.messages.stream(
                model=settings.model_chat,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
            ) as response:
                async for text in response.text_stream:
                    # SSE: escape embedded newlines so multi-line chunks stay one event
                    safe = text.replace("\n", "\ndata: ")
                    yield f"data: {safe}\n\n"
                yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
