"""Conversation persistence — get/create + append + history rendering.

Storage: messages live as a JSONB array of `{role, content, images,
timestamp}` on `Conversation.messages`. Once the array passes 20 entries
the oldest 15 are compacted into `Conversation.summary` (see
`summarization.py`); the LLM context stays bounded, long-term memory
survives in the summary.

The /chat/stream router persists turns AFTER the `[DONE]` SSE sentinel,
so DB writes never block the user-visible stream.
"""
import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation

logger = logging.getLogger(__name__)


# `messages > _SUMMARIZE_TRIGGER` switches both maybe_summarize and
# build_history into compaction mode. Both modules read these values.
_SUMMARIZE_TRIGGER = 20
_SUMMARIZE_BATCH = 15
_HISTORY_TAIL = 10

_MAX_CONVERSATIONS_PER_USER = 50


async def get_or_create_conversation(
    db: AsyncSession,
    conversation_id: UUID | None,
    user_id: int,
) -> Conversation:
    """Fetch the user's conversation by id, or create a new one.

    - `conversation_id` None → fresh conversation (DB picks the UUID).
    - `conversation_id` provided and owned by `user_id` → return it.
    - `conversation_id` provided but missing/not-owned → create a new row
      with that id (treated as a recovery from a rolled-back previous
      attempt; UUIDs are unguessable enough that collisions don't matter,
      and the user_id scope means the only ‘attack’ is creating your own
      conversation).

    Caller must `db.commit()` if it wants the new row durable before later
    work; this function only flushes so the id is populated.
    """
    if conversation_id is not None:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        conv = result.scalar_one_or_none()
        if conv is not None:
            return conv

    # Refine-with-Buddy conversations are coupled 1:1 to a saved trip; aging
    # one out would silently delete chat history the user expected to find
    # next time they tap "Refine with Buddy" on that trip. Exempt them from
    # the cap — both for counting and for selecting the eviction target.
    count_result = await db.execute(
        select(func.count()).where(
            Conversation.user_id == user_id,
            Conversation.trip_id.is_(None),
        )
    )
    count = count_result.scalar_one()
    if count >= _MAX_CONVERSATIONS_PER_USER:
        oldest_result = await db.execute(
            select(Conversation.id)
            .where(
                Conversation.user_id == user_id,
                Conversation.trip_id.is_(None),
            )
            .order_by(Conversation.created_at.asc())
            .limit(1)
        )
        oldest_id = oldest_result.scalar_one_or_none()
        if oldest_id is not None:
            await db.execute(
                delete(Conversation).where(Conversation.id == oldest_id)
            )

    conv = Conversation(
        id=conversation_id,  # None → SQLAlchemy default uuid.uuid4
        user_id=user_id,
        messages=[],
    )
    db.add(conv)
    await db.flush()
    logger.info(
        "conversation.created",
        extra={"conversation_id": str(conv.id), "local_user_id": user_id},
    )
    return conv


async def append_messages(
    db: AsyncSession,
    conversation: Conversation,
    user_msg: str,
    assistant_msg: str,
    images: list[str] | None = None,
) -> None:
    """Append the (user, assistant) turn pair with UTC timestamps and commit.

    Uses a raw `UPDATE` rather than ORM attribute tracking. Runtime
    instrumentation showed that under FastAPI's request-scoped
    AsyncSession + StreamingResponse generator context, the conversation
    object was not being marked dirty after attribute assignment, so
    `db.commit()` issued no UPDATE. Even `flag_modified` did not fix
    it. The raw UPDATE bypasses the ORM tracking layer entirely and
    is read-back into the Python object via a fresh SELECT, so the
    in-memory state stays consistent for the rest of the request.

    `images` carries the public Supabase Storage URLs attached to the
    current-turn user message. Persisting them here lets `build_history`
    rebuild the multimodal Anthropic content blocks on later turns, so
    resumed conversations preserve visual context.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    new_messages = list(conversation.messages or []) + [
        {
            "role": "user",
            "content": user_msg,
            "images": images or [],
            "timestamp": now_iso,
        },
        {
            "role": "assistant",
            "content": assistant_msg,
            "images": [],
            "timestamp": now_iso,
        },
    ]
    await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation.id)
        .values(messages=new_messages)
    )
    await db.commit()
    conversation.messages = new_messages


def _render_message(m: dict) -> dict:
    """Render a persisted message as a Claude `messages` entry.

    If the message has stored image URLs, rebuild a multimodal content
    block list (images first, optional text last) — same shape produced
    for the current-turn payload in /chat/stream. Otherwise pass the
    text content through. Tolerates rows persisted before the `images`
    field existed (`.get("images")` → None → string branch).
    """
    images = m.get("images") or []
    if images:
        content: list[dict] = [
            {"type": "image", "source": {"type": "url", "url": u}}
            for u in images
        ]
        if m.get("content"):
            content.append({"type": "text", "text": m["content"]})
        return {"role": m["role"], "content": content}
    return {"role": m["role"], "content": m["content"]}


def build_history(conversation: Conversation) -> list[dict]:
    """Render conversation as a Claude `messages` array.

    - len ≤ 20: return all turns verbatim as {role, content}.
    - len > 20: return [{assistant: summary}, *last 10 turns].

    Messages with persisted image URLs are reconstructed as multimodal
    content blocks via `_render_message`.
    """
    messages = list(conversation.messages or [])
    if len(messages) <= _SUMMARIZE_TRIGGER:
        return [_render_message(m) for m in messages]

    summary = conversation.summary or ""
    tail = messages[-_HISTORY_TAIL:]
    return [
        {"role": "assistant", "content": summary},
        *(_render_message(m) for m in tail),
    ]
