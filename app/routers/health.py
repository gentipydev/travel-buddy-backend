"""Liveness + readiness probes.

`/health` is the liveness probe: it returns 200 as long as the process is
up, with no dependency checks — a load balancer uses it to decide whether
to restart the container.

`/health/ready` is the readiness probe: it verifies the process can actually
serve traffic by checking DB connectivity (a lightweight `SELECT 1`) and
reporting which third-party API keys are configured. It makes NO paid
external calls — key presence is a config check, not a live probe — so it's
cheap enough to poll. Returns 200 when DB is reachable, 503 otherwise, so an
orchestrator can hold traffic off an instance whose DB is down.
"""
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import settings
from app.db.engine import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness():
    """Dependency-aware readiness check. 200 ready / 503 not ready.

    DB connectivity is the only hard gate — without it no request that
    touches Postgres can succeed. API-key presence is reported but does NOT
    fail readiness: a missing COHERE_API_KEY degrades retrieval to
    lexical-only (still serviceable), and Google keys are kill-switchable.
    Only `gemini` is functionally required, so it's surfaced but the
    instance is still "ready" enough to serve health/static endpoints — the
    DB gate is the operational signal an orchestrator should act on.
    """
    db_ok = False
    db_error: str | None = None
    # get_session is an async generator dependency; drive it manually here
    # since we're outside FastAPI's Depends injection.
    gen = get_session()
    try:
        session = await gen.__anext__()
        try:
            await session.execute(text("SELECT 1"))
            db_ok = True
        finally:
            await gen.aclose()
    except Exception as exc:
        db_error = f"{type(exc).__name__}: {exc}"
        logger.warning("readiness: DB check failed: %s", db_error)

    config = {
        "gemini": bool(settings.gemini_api_key),
        "cohere": bool(settings.cohere_api_key),
        "google": bool(settings.google_api_key),
    }

    ready = db_ok
    body = {
        "status": "ready" if ready else "not_ready",
        "checks": {
            "database": {"ok": db_ok, "error": db_error},
            "config": config,
        },
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)
