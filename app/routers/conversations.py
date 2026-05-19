from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session
from app.db.models import Conversation, User
from app.deps import require_local_user
from app.routers.trips import _parse_cursor
from app.schemas import PaginatedResponse

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get(
    "",
    responses={
        400: {"description": "Invalid cursor format"},
        422: {"description": "limit out of range (must be 1..100)"},
    },
)
async def list_conversations(
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(
        None,
        description=(
            "Opaque cursor from a previous response's `next_cursor`. "
            "Omit for the first page."
        ),
    ),
) -> PaginatedResponse[dict]:
    """Keyset-paginated list of the caller's conversations, newest first.

    Same `(updated_at DESC, id DESC)` ordering as `GET /trips`; same opaque
    cursor format (`"{updated_at_iso}|{uuid}"`). The item shape is a thin
    dict (no full message hydration) — clients fetch `GET /conversations/{id}`
    for the full transcript."""
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
    )
    if cursor:
        ts, last_id = _parse_cursor(cursor)
        stmt = stmt.where(
            tuple_(Conversation.updated_at, Conversation.id)
            < tuple_(ts, last_id)
        )
    rows = (await db.execute(stmt.limit(limit + 1))).scalars().all()
    next_cursor: str | None = None
    if len(rows) > limit:
        rows = rows[:limit]
        last = rows[-1]
        next_cursor = f"{last.updated_at.isoformat()}|{last.id}"
    items = [
        {
            "id": str(c.id),
            "title": c.title,
            "trip_id": str(c.trip_id) if c.trip_id else None,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            "message_count": len(c.messages or []),
        }
        for c in rows
    ]
    return PaginatedResponse[dict](items=items, next_cursor=next_cursor)


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
        "trip_id": str(conv.trip_id) if conv.trip_id else None,
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
