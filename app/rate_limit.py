"""SlowAPI limiter + key function + 429 handler.

Single source of truth so adding a limit to another expensive endpoint
(e.g. persona refresh) is one import. The limiter is in-memory, which
is correct for a single-instance Render deployment; if we go multi-
instance, switch the storage backend to Redis here and nowhere else.
"""
import hashlib
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)


def user_or_ip_key(request: Request) -> str:
    """Group requests by authenticated user, falling back to IP.

    Hashes the Bearer token so the limiter key never includes a raw JWT
    in memory or logs. The first 16 hex chars of SHA-256 are enough to
    avoid collisions at this scale. Anonymous calls bucket by IP, so the
    per-IP decorator (which also fires on every request) still bounds
    them.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        return f"user:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
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
