"""SlowAPI limiter + key function + 429 handler.

Single source of truth so adding a limit to another expensive endpoint
(e.g. persona refresh) is one import. The limiter is in-memory, which
is correct for a single-instance Render deployment; if we go multi-
instance, switch the storage backend to Redis here and nowhere else.
"""
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)


def user_or_ip_key(request: Request) -> str:
    """Group requests by VERIFIED authenticated user, falling back to IP.

    Keys on `request.state.verified_sub`, set by the auth dependencies in
    `app/auth/supabase_auth.py` only after JWT signature verification.
    Keying on the raw Bearer token (even hashed) would let an attacker
    mint a fresh limiter bucket per request with garbage tokens, and
    legit Supabase tokens rotate hourly — silently resetting daily
    quotas. The sub is a Supabase UUID already logged via `user_id_var`,
    so it is safe in limiter keys and logs. Unverified calls bucket by
    IP, so the per-IP decorator still bounds them.
    """
    sub = getattr(request.state, "verified_sub", None)
    if sub:
        return f"user:{sub}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=get_remote_address)


async def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Return 429 with an explicit `Retry-After` and a stable JSON shape.

    SlowAPI's built-in handler returns a similar response but the body
    shape is not pinned and `Retry-After` is not always set. The chat
    client parses `Retry-After` to size the "slow down" message, so we
    own both fields here.
    """
    # SlowAPI attaches retry_after (seconds) when available.
    retry_after = getattr(exc, "retry_after", None) or 60
    logger.info(
        "rate_limit.exceeded path=%s key=%s detail=%s retry_after=%ss",
        request.url.path,
        user_or_ip_key(request),
        exc.detail,
        retry_after,
    )
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded",
            "limit": str(exc.detail),
            "retry_after_seconds": int(retry_after),
        },
        headers={"Retry-After": str(int(retry_after))},
    )
