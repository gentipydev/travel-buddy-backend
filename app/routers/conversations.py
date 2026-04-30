from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session
from app.db.models import Conversation, User
from app.deps import require_local_user


router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
async def list_conversations(
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
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


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: UUID,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
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


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
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
