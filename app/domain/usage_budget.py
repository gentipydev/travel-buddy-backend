"""Per-user daily cost budgets, global spend kill-switch, stream slots.

Request-count rate limits (app/rate_limit.py) bound *how often* a user
can call the LLM endpoints; this module bounds *how much it costs*. The
`usage_daily` table (migration 018) accumulates per-(user, UTC day)
token counts and estimated USD, written via fail-soft upsert after each
metered call and read by two gates that run before any money is spent:

- `check_user_budget` — per-user daily USD ceiling → 429 until UTC midnight.
- `check_global_ceiling` — whole-service daily USD ceiling → 503. The
  last-resort kill-switch for anything per-user limits miss (mass
  signups, leaked credentials, bugs). Cached in-process for
  `global_budget_cache_seconds`, so worst-case overshoot is one cache
  window of traffic.

Metering scope: the chat-stream main call and the persona endpoint.
The auxiliary Haiku calls inside a turn (title gen ~20 output tokens,
rolling summarize, output guard, optional query rewrite, persona gate)
are deliberately NOT metered — each runs only inside an already-gated
turn and together they are well under 10% of a turn's cost. If that
ratio changes, fold a fixed overhead multiplier into the recorded cost
rather than plumbing usage out of every call site.

Concurrency: `acquire_stream_slot`/`release_stream_slot` cap parallel
/chat/stream generators per user. In-process state — correct for the
single-instance deployment, same caveat as the SlowAPI limiter.
"""
import logging
import time
from datetime import UTC, datetime

from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import UsageDaily

logger = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    """Raised by the budget gates; converted to JSON by the handler in
    `app/main.py`. Mirrors the SlowAPI 429 body shape so the Flutter
    client's existing rate-limit parsing applies unchanged.
    """

    def __init__(self, detail: str, status_code: int, retry_after_seconds: int):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


def _today_utc():
    return datetime.now(UTC).date()


def _seconds_until_utc_midnight() -> int:
    now = datetime.now(UTC)
    return max(1, 86400 - (now.hour * 3600 + now.minute * 60 + now.second))


async def user_over_budget(db: AsyncSession, supabase_uid: str) -> str | None:
    """Return a reason string when the user's estimated spend today is at/over
    budget, else None. The non-raising form behind `check_user_budget` — the
    DeepSeek overflow path (app/routers/chat.py) uses this to SPILL the turn to
    the cheaper fallback instead of hard-blocking it."""
    budget = settings.daily_cost_budget_usd_per_user
    if budget <= 0:
        return None
    spent = await db.scalar(
        select(UsageDaily.estimated_cost_usd).where(
            UsageDaily.supabase_uid == supabase_uid,
            UsageDaily.usage_date == _today_utc(),
        )
    )
    if spent is not None and float(spent) >= budget:
        return "budget_user"
    return None


async def check_user_budget(db: AsyncSession, supabase_uid: str) -> None:
    """Raise 429 if the user's estimated spend today is at/over budget."""
    if await user_over_budget(db, supabase_uid):
        raise BudgetExceeded(
            detail="daily_budget_exceeded",
            status_code=429,
            retry_after_seconds=_seconds_until_utc_midnight(),
        )


# (total_usd, fetched_at_monotonic) — refreshed at most once per
# `global_budget_cache_seconds`, so the ceiling check costs one SUM
# query per window per process instead of one per request.
_global_cache: tuple[float, float] | None = None


def _reset_global_cache() -> None:
    global _global_cache
    _global_cache = None


async def global_over_ceiling(db: AsyncSession) -> str | None:
    """Return a reason string when today's total estimated spend is at/over the
    ceiling, else None. The non-raising form behind `check_global_ceiling` —
    the DeepSeek overflow path spills to the fallback instead of 503-blocking.
    Keeps the same in-process SUM cache so the check stays one query per
    window."""
    global _global_cache
    ceiling = settings.daily_cost_ceiling_usd_global
    if ceiling <= 0:
        return None
    now = time.monotonic()
    if (
        _global_cache is None
        or now - _global_cache[1] >= settings.global_budget_cache_seconds
    ):
        total = await db.scalar(
            select(func.coalesce(func.sum(UsageDaily.estimated_cost_usd), 0)).where(
                UsageDaily.usage_date == _today_utc()
            )
        )
        _global_cache = (float(total or 0), now)
    if _global_cache[0] >= ceiling:
        return "budget_global"
    return None


async def check_global_ceiling(db: AsyncSession) -> None:
    """Raise 503 if today's total estimated spend is at/over the ceiling."""
    if await global_over_ceiling(db):
        raise BudgetExceeded(
            detail="service_at_capacity",
            status_code=503,
            retry_after_seconds=int(settings.global_budget_cache_seconds),
        )


async def record_usage(
    db: AsyncSession,
    supabase_uid: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    estimated_cost_usd: float,
) -> None:
    """Accumulate one call's usage into today's row. Fail-soft: a broken
    accounting write must never take down a stream that already cost
    money — log it and move on.
    """
    try:
        stmt = pg_insert(UsageDaily).values(
            supabase_uid=supabase_uid,
            usage_date=_today_utc(),
            requests=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[UsageDaily.supabase_uid, UsageDaily.usage_date],
            set_={
                "requests": UsageDaily.requests + 1,
                "input_tokens": UsageDaily.input_tokens + stmt.excluded.input_tokens,
                "output_tokens": (
                    UsageDaily.output_tokens + stmt.excluded.output_tokens
                ),
                "cache_creation_input_tokens": (
                    UsageDaily.cache_creation_input_tokens
                    + stmt.excluded.cache_creation_input_tokens
                ),
                "cache_read_input_tokens": (
                    UsageDaily.cache_read_input_tokens
                    + stmt.excluded.cache_read_input_tokens
                ),
                "estimated_cost_usd": (
                    UsageDaily.estimated_cost_usd + stmt.excluded.estimated_cost_usd
                ),
                "updated_at": func.now(),
            },
        )
        await db.execute(stmt)
        await db.commit()
    except Exception:
        logger.exception("usage_daily upsert failed (uid=%s)", supabase_uid)
        try:
            await db.rollback()
        except Exception:
            logger.exception("usage_daily rollback failed (uid=%s)", supabase_uid)


async def record_usage_detached(supabase_uid: str, **kwargs) -> None:
    """`record_usage` on a fresh session — for the client-disconnect path,
    where the request-scoped session is being torn down while we record.
    """
    from app.db.engine import async_session

    try:
        async with async_session() as session:
            await record_usage(session, supabase_uid, **kwargs)
    except Exception:
        logger.exception("detached usage record failed (uid=%s)", supabase_uid)


# --- Per-user concurrent stream cap -----------------------------------

# sub -> monotonic start times of in-flight /chat/stream generators.
# Slots are released in ChatAgent.stream()'s finally; the TTL prune
# self-heals the rare leak where Starlette never iterates the generator
# (so finally never runs). Sync on purpose: single event loop, and
# release must be callable during CancelledError unwinding (no await).
_active_streams: dict[str, list[float]] = {}
_STREAM_SLOT_TTL_SECONDS = 600.0


def acquire_stream_slot(supabase_uid: str) -> bool:
    limit = settings.max_concurrent_streams_per_user
    if limit <= 0:
        return True
    now = time.monotonic()
    slots = [
        t
        for t in _active_streams.get(supabase_uid, [])
        if now - t < _STREAM_SLOT_TTL_SECONDS
    ]
    if len(slots) >= limit:
        _active_streams[supabase_uid] = slots
        return False
    slots.append(now)
    _active_streams[supabase_uid] = slots
    return True


def release_stream_slot(supabase_uid: str) -> None:
    slots = _active_streams.get(supabase_uid)
    if slots:
        slots.pop()
    if not slots:
        _active_streams.pop(supabase_uid, None)


async def budget_exceeded_handler(
    request: Request, exc: BudgetExceeded
) -> JSONResponse:
    """Same body contract as `rate_limit_exceeded_handler`: the client
    reads `detail` + `retry_after_seconds` and the `Retry-After` header.
    """
    logger.info(
        "budget.blocked path=%s detail=%s status=%s retry_after=%ss",
        request.url.path,
        exc.detail,
        exc.status_code,
        exc.retry_after_seconds,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "retry_after_seconds": exc.retry_after_seconds,
        },
        headers={"Retry-After": str(exc.retry_after_seconds)},
    )
