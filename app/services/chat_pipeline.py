"""Per-turn orchestration for /chat/stream.

`prepare_turn` loads user state, runs RAG, detects a place, and assembles
the Anthropic `messages` payload. It returns a ChatTurnContext — the router
owns streaming, persistence, and telemetry on top of that, so FastAPI
lifecycle (StreamingResponse, request-scoped db session) stays in one place.
"""
import json
import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, User, UserPreferences
from app.prompts import build_system_prompt
from app.schemas import ChatRequest
from app.services.conversation_service import (
    build_history,
    get_or_create_conversation,
)
from app.services.place_service import detect_place, to_card_dict
from app.services.rag_service import retrieve_chunks


logger = logging.getLogger(__name__)


@dataclass
class ChatTurnContext:
    system_prompt: str
    messages: list[dict]
    conversation: Conversation | None
    new_conversation_id: UUID | None
    place_card_payload: str | None
    retrieval_used: bool
    chunks_returned: int
    top_rerank_score: float | None


async def prepare_turn(
    request: ChatRequest,
    supa_user: dict | None,
    db: AsyncSession,
) -> ChatTurnContext:
    """Build everything the router needs to start streaming a chat reply.

    Server-side conversation persistence: only when auth + synced. Anonymous
    users (no token, or token but `users` row not synced via /users/me/sync)
    fall back to client-supplied `conversation_history` and are never written
    to the conversations table.
    """
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

    return ChatTurnContext(
        system_prompt=system_prompt,
        messages=messages,
        conversation=conversation,
        new_conversation_id=new_conversation_id,
        place_card_payload=card_payload,
        retrieval_used=retrieval_used,
        chunks_returned=chunks_returned,
        top_rerank_score=top_rerank_score,
    )
