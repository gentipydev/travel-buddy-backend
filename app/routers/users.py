from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.supabase_auth import get_current_user
from app.db.engine import get_or_create_user, get_session
from app.db.models import User
from app.deps import require_local_user


router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
async def get_me(user: User = Depends(require_local_user)):
    return {
        "id": user.id,
        "supabase_uid": user.supabase_uid,
        "name": user.name,
        "email": user.email,
    }


@router.post("/me/sync")
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
