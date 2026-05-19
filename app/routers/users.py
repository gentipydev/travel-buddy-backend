import logging

from fastapi import APIRouter, Depends, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.supabase_auth import get_current_user
from app.config import get_supabase
from app.db.engine import get_or_create_user, get_session
from app.db.models import User
from app.deps import require_local_user
from app.integrations.supabase_storage import CHAT_ATTACHMENTS_BUCKET


router = APIRouter(prefix="/users", tags=["users"])
logger = logging.getLogger(__name__)


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


@router.delete("/me", status_code=204)
async def delete_me(
    supa_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> Response:
    """Cascading account delete: storage objects → DB row → Supabase auth user.

    Order matters. Storage first because objects have no FK back to the user
    row, so deleting the row first would orphan them. DB second — the FK
    CASCADE on `trips.user_id`, `conversations.user_id`,
    `user_preferences.user_id` removes trips/days/stops/routes,
    conversations/messages, and prefs in a single statement. Auth last —
    once `supabase.auth.admin.delete_user` succeeds, the JWT is invalid and
    the client is signed out on its next request.

    Failure modes:
      - Storage failure → logged; we proceed. The objects become orphaned
        and can be GC'd by a follow-up sweep. Better than leaving the user
        in a half-deleted state.
      - DB failure → uncaught; 500 to the client. They can retry.
      - Auth failure → logged; we return 204 anyway. The user has no
        local data left; a retry of the endpoint (with the same valid JWT)
        will reach the admin call again. If the JWT is already gone, they
        can re-sign-in and the next call will re-create the row only to
        delete it again — annoying but not data-corrupting.
    """
    supabase_uid = supa_user["id"]
    supabase = await get_supabase()

    # 1. Storage cleanup — chat-attachments/<uid>/<uuid>.<ext> per
    # `attachment_service.dart`. The bucket has no DELETE RLS policy, so the
    # service_role key (used by get_supabase) is the only way to remove these.
    try:
        files = await supabase.storage.from_(CHAT_ATTACHMENTS_BUCKET).list(
            path=supabase_uid
        )
        paths = [
            f"{supabase_uid}/{f['name']}"
            for f in (files or [])
            if isinstance(f, dict) and f.get("name")
        ]
        if paths:
            await supabase.storage.from_(CHAT_ATTACHMENTS_BUCKET).remove(paths)
            logger.info(
                "delete_user.storage_removed uid=%s count=%s",
                supabase_uid,
                len(paths),
            )
    except Exception:
        logger.exception("delete_user.storage_failed uid=%s", supabase_uid)

    # 2. Local users row + FK cascade — wipes trips, conversations, prefs.
    user = await db.scalar(select(User).where(User.supabase_uid == supabase_uid))
    if user is not None:
        await db.delete(user)
        await db.commit()
        logger.info("delete_user.db_removed uid=%s user_id=%s", supabase_uid, user.id)

    # 3. Supabase auth user — invalidates the JWT.
    try:
        await supabase.auth.admin.delete_user(supabase_uid)
        logger.info("delete_user.auth_removed uid=%s", supabase_uid)
    except Exception:
        logger.exception("delete_user.auth_failed uid=%s", supabase_uid)

    # 204 No Content — body is intentionally empty.
    return Response(status_code=204)
