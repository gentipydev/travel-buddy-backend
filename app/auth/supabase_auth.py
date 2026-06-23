import logging
from functools import lru_cache

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient, decode as jwt_decode

from app.config import settings
from app.observability import user_id_var

logger = logging.getLogger(__name__)

ISSUER = f"{settings.supabase_url}/auth/v1"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"

security = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _jwks() -> PyJWKClient:
    return PyJWKClient(JWKS_URL)


def _verify(token: str) -> dict:
    try:
        key = _jwks().get_signing_key_from_jwt(token).key
        return jwt_decode(
            token,
            key,
            algorithms=["ES256", "RS256", "HS256"],
            issuer=ISSUER,
            audience="authenticated",
        )
    except HTTPException:
        raise
    except Exception as e:
        # Don't log the token; one warning per rejection is enough to
        # distinguish "real auth failure" from "no token sent".
        logger.warning("auth.token_rejected: %s", e)
        raise HTTPException(status_code=401, detail=str(e))


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """Require authenticated user. Raises 401 if no valid token."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = _verify(credentials.credentials)
    user_id_var.set(payload["sub"])
    # Verified identity for the rate limiter. SlowAPI's key func only gets
    # the Request, and deps resolve before the limit decorator fires, so
    # request.state is the handoff point (see app/rate_limit.py).
    request.state.verified_sub = payload["sub"]
    return {
        "id": payload["sub"],
        "email": payload.get("email"),
        "role": payload.get("role", "authenticated"),
    }


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict | None:
    """Optional auth. Returns None if no token or invalid token."""
    if not credentials:
        return None
    try:
        payload = _verify(credentials.credentials)
        user_id_var.set(payload["sub"])
        request.state.verified_sub = payload["sub"]
        return {
            "id": payload["sub"],
            "email": payload.get("email"),
            "role": payload.get("role", "authenticated"),
        }
    except Exception:
        return None
