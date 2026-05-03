from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.supabase_auth import get_current_user
from app.db.engine import get_session
from app.db.models import User


async def require_local_user(
    supa_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> User:
    """Require authenticated user with a synced row in the local `users` table.

    Raises 404 if the Supabase user has not yet been synced via
    POST /users/me/sync. Routes that need the local-user PK should depend on
    this; routes that only need the Supabase identity (e.g. /users/me/sync
    itself) should depend on get_current_user directly.
    """
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
