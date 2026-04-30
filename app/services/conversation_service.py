"""Server-side conversation persistence with rolling summarization.

Storage: messages live as a JSONB array of {role, content, timestamp}.
Once the array passes 20 entries, the oldest 15 are summarized via Claude
into `Conversation.summary` and dropped from the array — the LLM context
stays bounded, long-term memory survives in the summary.

The /chat/stream endpoint persists turns AFTER `[DONE]` is yielded, so DB
writes never block the user-visible stream.
"""
import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation

logger = logging.getLogger(__name__)


_SUMMARIZE_PROMPT = (
    "Summarize this travel conversation history concisely, preserving key "
    "places, preferences and decisions mentioned."
)
# `messages > _SUMMARIZE_TRIGGER` switches both maybe_summarize and
# build_history into compaction mode.
_SUMMARIZE_TRIGGER = 20
_SUMMARIZE_BATCH = 15
_HISTORY_TAIL = 10

_MAX_CONVERSATIONS_PER_USER = 50

_TITLE_PROMPT = (
    "You will be shown one turn of a chat between a user and an Albania travel "
    "assistant. Reply with a 3–5 word English noun phrase describing the user's "
    "travel topic or question. Do NOT quote the user or assistant verbatim — "
    "describe the topic abstractly. If the message is a greeting, complaint, or "
    "meta-feedback with no clear travel topic, reply with: General travel chat.\n\n"
    "Good examples:\n"
    "  Berat castle history\n"
    "  Tirana food recommendations\n"
    "  Three-day Albania itinerary\n"
    "  Saranda nightlife guide\n"
    "  Driving rules in Albania\n\n"
    "Output only the title text — no quotes, no preamble, no trailing punctuation."
)
_TITLE_MAX_TOKENS = 20
_TITLE_MAX_LEN = 60
# Keep the assistant reply short in the title-gen prompt; long replies dominate
# the input and tempt the model to quote them back as the "title".
_TITLE_ASSISTANT_PREVIEW_LEN = 500


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

    count_result = await db.execute(
        select(func.count()).where(Conversation.user_id == user_id)
    )
    count = count_result.scalar_one()
    if count >= _MAX_CONVERSATIONS_PER_USER:
        oldest_result = await db.execute(
            select(Conversation.id)
            .where(Conversation.user_id == user_id)
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


async def maybe_summarize(
    db: AsyncSession,
    conversation: Conversation,
    client,  # AsyncAnthropic — typed loosely to avoid an import cycle
    model: str,
) -> None:
    """Compact the oldest 15 messages into `summary` (via Claude) when the
    array has grown past 20. Combines with the prior summary so memory
    accumulates across compactions.

    Best-effort — failures log and return; the next turn will retry.
    """
    # FUTURE: once a user accumulates ~5 summarized conversations (or
    # roughly that many compactions), kick off POST /preferences/persona
    # automatically so an inferred travel_persona is built from real chat
    # history rather than relying on the explicit preferences UI.
    # Comment only — not wired.
    messages = list(conversation.messages or [])
    if len(messages) <= _SUMMARIZE_TRIGGER:
        return

    oldest = messages[:_SUMMARIZE_BATCH]
    transcript = "\n\n".join(
        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
        for m in oldest
    )
    if conversation.summary:
        transcript = f"PRIOR SUMMARY: {conversation.summary}\n\n{transcript}"

    try:
        result = await client.messages.create(
            model=model,
            max_tokens=512,
            system=_SUMMARIZE_PROMPT,
            messages=[{"role": "user", "content": transcript}],
        )
        new_summary = "".join(
            block.text for block in result.content if block.type == "text"
        ).strip()
    except Exception:
        logger.exception("conversation summarization failed")
        return

    if not new_summary:
        return

    trimmed = messages[_SUMMARIZE_BATCH:]
    await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation.id)
        .values(summary=new_summary, messages=trimmed)
    )
    await db.commit()
    conversation.summary = new_summary
    conversation.messages = trimmed


async def generate_title(
    db: AsyncSession,
    conversation: Conversation,
    client,  # AsyncAnthropic — typed loosely to avoid an import cycle
    model: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """Best-effort: replace the placeholder first-6-words title with a
    Claude-summarized label after the first assistant turn finishes.

    Caller invokes this only on conversation-creation turns (i.e., the row
    was just inserted in this request). Failures log and return; the
    placeholder title set in `chat_stream` stays in place.

    Uses raw UPDATE for the same reason as `append_messages`.
    """
    assistant_preview = assistant_msg[:_TITLE_ASSISTANT_PREVIEW_LEN]
    if len(assistant_msg) > _TITLE_ASSISTANT_PREVIEW_LEN:
        assistant_preview += "…"
    transcript = (
        "<conversation>\n"
        f"<user>{user_msg or '(no text — image-only message)'}</user>\n"
        f"<assistant>{assistant_preview}</assistant>\n"
        "</conversation>"
    )
    try:
        result = await client.messages.create(
            model=model,
            max_tokens=_TITLE_MAX_TOKENS,
            system=_TITLE_PROMPT,
            messages=[{"role": "user", "content": transcript}],
        )
        raw = "".join(b.text for b in result.content if b.type == "text")
    except Exception:
        logger.exception("conversation title generation failed")
        return

    new_title = _clean_title(raw)
    if not new_title:
        return
    if len(new_title) > _TITLE_MAX_LEN:
        new_title = new_title[:_TITLE_MAX_LEN].rstrip()
    await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation.id)
        .values(title=new_title)
    )
    await db.commit()
    conversation.title = new_title


def _clean_title(raw: str) -> str:
    """Strip the common ways Claude prefixes / wraps a one-line answer."""
    s = raw.strip()
    # Take only the first line — model occasionally adds an explanation below.
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()
    # Drop leading "Title:" / "title:" labels.
    for prefix in ("Title:", "title:", "TITLE:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    # Strip surrounding quotes / asterisks / backticks repeatedly.
    while s and s[0] in '"\'`*' and s[-1] in '"\'`*' and len(s) > 1:
        s = s[1:-1].strip()
    # Strip a single trailing period (titles shouldn't end in one).
    if s.endswith("."):
        s = s[:-1].rstrip()
    return s
