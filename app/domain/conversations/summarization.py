"""Rolling conversation summarization + model-generated titles.

`maybe_summarize` runs after every assistant turn. When the messages
array exceeds `_SUMMARIZE_TRIGGER`, the oldest `_SUMMARIZE_BATCH` are
compacted into `Conversation.summary` via the LLM and dropped from the
array. A persona-refresh task is fired-and-forgotten on success.

`generate_title` is best-effort: on conversation-creation turns it
replaces the placeholder first-6-words title with a model-summarized
3-5 word label.

Both LLM calls fail soft — failures log and return; the next turn
retries. Caller passes the model id so /chat/stream can pin Title and
Summarize to their own settings.* fields.
"""
import asyncio
import logging

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation
from app.domain.conversations.persona import refresh_travel_persona
from app.domain.conversations.repository import (
    _SUMMARIZE_BATCH,
    _SUMMARIZE_TRIGGER,
)
from app.llm import generate_text
from app.prompts import get as _get_prompt

logger = logging.getLogger(__name__)


_SUMMARIZE_PROMPT = _get_prompt("conversation.summarize").text
_TITLE_PROMPT = _get_prompt("conversation.title").text

_TITLE_MAX_TOKENS = 20
_TITLE_MAX_LEN = 60
# Keep the assistant reply short in the title-gen prompt; long replies dominate
# the input and tempt the model to quote them back as the "title".
_TITLE_ASSISTANT_PREVIEW_LEN = 500


async def maybe_summarize(
    db: AsyncSession,
    conversation: Conversation,
    model: str,
) -> None:
    """Compact the oldest 15 messages into `summary` (via the LLM) when the
    array has grown past 20. Combines with the prior summary so memory
    accumulates across compactions.

    Best-effort — failures log and return; the next turn will retry.
    """
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
        new_summary = (
            await generate_text(
                model=model,
                system=_SUMMARIZE_PROMPT,
                user=transcript,
                max_output_tokens=512,
            )
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

    # Signal-based travel_persona refresh. The task opens its own session,
    # runs a cheap YES/NO gate against the user's current persona, and
    # rewrites it from recent summaries when warranted. Fire-and-forget so
    # the chat stream is never blocked.
    if conversation.user_id is not None:
        asyncio.create_task(
            refresh_travel_persona(conversation.user_id, new_summary, model)
        )


async def generate_title(
    db: AsyncSession,
    conversation: Conversation,
    model: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """Best-effort: replace the placeholder first-6-words title with a
    model-summarized label after the first assistant turn finishes.

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
        raw = await generate_text(
            model=model,
            system=_TITLE_PROMPT,
            user=transcript,
            max_output_tokens=_TITLE_MAX_TOKENS,
        )
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
    """Strip the common ways the model prefixes / wraps a one-line answer."""
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
