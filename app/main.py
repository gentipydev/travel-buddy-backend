import json
import logging
import time
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import client, settings, get_supabase
from app.database import (
    Conversation,
    User,
    UserPreferences,
    init_db,
    get_session,
    get_or_create_user,
)
from app.auth.supabase_auth import get_current_user, get_optional_user
from app.services.conversation_service import (
    append_messages,
    build_history,
    generate_title,
    get_or_create_conversation,
    maybe_summarize,
)
from app.services.place_service import detect_place, to_card_dict
from app.services.rag_service import retrieve_chunks
from app.services.telemetry import (
    RequestTrace,
    log_trace,
    new_request_id,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

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
    allow_methods=["GET", "POST", "DELETE"],
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
    # When provided alongside an authenticated user, server-side history
    # supersedes `conversation_history`. Anonymous requests ignore this field.
    conversation_id: UUID | None = None


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


@app.get("/conversations")
async def list_conversations(
    supa_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    user = await _require_local_user(supa_user, db)
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    )
    conversations = result.scalars().all()
    return [
        {
            "id": str(c.id),
            "title": c.title,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            "message_count": len(c.messages or []),
        }
        for c in conversations
    ]


@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: UUID,
    supa_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    user = await _require_local_user(supa_user, db)
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id": str(conv.id),
        "title": conv.title,
        "summary": conv.summary,
        "messages": list(conv.messages or []),
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    supa_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    user = await _require_local_user(supa_user, db)
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete(conv)
    await db.commit()
    return {"status": "deleted"}


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    supa_user: dict | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_session),
):
    request_id = new_request_id()
    started_at = time.perf_counter()
    timestamp = utc_now_iso()
    user_id = supa_user["id"] if supa_user else None
    message_length = len(request.message)

    preferences = None
    local_user: User | None = None
    if supa_user:
        user_result = await db.execute(
            select(User).where(User.supabase_uid == supa_user["id"])
        )
        local_user = user_result.scalar_one_or_none()
        if local_user:
            pref_result = await db.execute(
                select(UserPreferences).where(UserPreferences.user_id == local_user.id)
            )
            preferences = pref_result.scalar_one_or_none()

    logger.info(
        "chat_stream: supabase_uid=%s conversation_id=%s local_user_resolved=%s",
        supa_user["id"] if supa_user else None,
        request.conversation_id,
        local_user is not None,
    )

    # Server-side conversation persistence: only when auth + synced. Anonymous
    # users (no token, or token but `users` row not synced via /users/me/sync)
    # fall back to client-supplied `conversation_history` and are never written
    # to the conversations table.
    conversation: Conversation | None = None
    new_conversation_id: UUID | None = None
    if local_user is not None:
        conversation = await get_or_create_conversation(
            db, request.conversation_id, local_user.id
        )
        if request.conversation_id is None:
            # Title comes from the first 6 words of the opening message; only
            # set on creation so revisiting the conversation can't rewrite it.
            conversation.title = " ".join(request.message.split()[:6]) or None
            new_conversation_id = conversation.id
        # Commit eagerly so the id we yield in the SSE event is durable even
        # if the stream errors mid-flight.
        await db.commit()
        await db.refresh(conversation)

    system_prompt = build_system_prompt(preferences)

    if conversation is not None:
        history = build_history(conversation)
    else:
        history = [m.model_dump() for m in request.conversation_history]

    retrieved, top_rerank_score = await retrieve_chunks(
        request.message, db, history
    )
    retrieval_used = bool(retrieved)
    chunks_returned = len(retrieved)

    if retrieved:
        context = "\n\n---\n\n".join(c.content for c in retrieved)
        system_prompt = (
            f"{system_prompt}\n\n"
            "Relevant context from our Albania database:\n"
            f"{context}"
        )

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

    messages = history + [{"role": "user", "content": user_payload}]

    place = await detect_place(request.message, db)
    card_payload = json.dumps(to_card_dict(place)) if place else None

    async def stream():
        input_tokens: int | None = None
        output_tokens: int | None = None
        assistant_text: str | None = None
        try:
            if new_conversation_id is not None:
                yield f"event: conversation_id\ndata: {new_conversation_id}\n\n"
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
                # Capture final message inside the context manager so the
                # accumulated state is still alive. Wrapped — token-count or
                # SDK-shape failures must not crash the request.
                try:
                    final = await response.get_final_message()
                    input_tokens = final.usage.input_tokens
                    output_tokens = final.usage.output_tokens
                    assistant_text = "".join(
                        b.text for b in final.content if b.type == "text"
                    )
                except Exception:
                    input_tokens = None
                    output_tokens = None
                    assistant_text = None

            # Persist BEFORE yielding [DONE]. The Flutter SSE parser closes the
            # connection on [DONE], and Starlette cancels the still-running
            # generator at its next await point. In Python 3.12 CancelledError
            # is BaseException, so `except Exception` cannot catch it — the
            # commit silently never lands. Doing the write here guarantees
            # durability before the client can disconnect. Tokens still stream
            # live during the loop above, so the user-visible latency added
            # is only the DB write (~50–150ms typical).
            if conversation is not None and assistant_text:
                try:
                    await append_messages(
                        db, conversation, request.message, assistant_text
                    )
                    if new_conversation_id is not None:
                        await generate_title(
                            db,
                            conversation,
                            client,
                            settings.model_chat,
                            request.message,
                            assistant_text,
                        )
                    await maybe_summarize(
                        db, conversation, client, settings.model_chat
                    )
                except Exception:
                    logger.exception("conversation persist failed")

            yield "data: [DONE]\n\n"

            log_trace(RequestTrace(
                request_id=request_id,
                timestamp=timestamp,
                user_id=user_id,
                message_length=message_length,
                retrieval_used=retrieval_used,
                chunks_returned=chunks_returned,
                top_rerank_score=top_rerank_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                model=settings.model_chat,
                error=None,
            ))
        except Exception as e:
            log_trace(RequestTrace(
                request_id=request_id,
                timestamp=timestamp,
                user_id=user_id,
                message_length=message_length,
                retrieval_used=retrieval_used,
                chunks_returned=chunks_returned,
                top_rerank_score=top_rerank_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                model=settings.model_chat,
                error=str(e),
            ))
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
