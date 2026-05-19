from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from app.config import get_supabase, settings
from app.db.engine import init_db
from app.observability import bt_init, request_logging_middleware, setup_logging
from app.observability.sentry import init_sentry
from app.rate_limit import limiter, rate_limit_exceeded_handler
from app.routers import chat, conversations, health, images, preferences, trips, users

# Configure logging BEFORE FastAPI is built so uvicorn's own loggers attach
# to the same handler/format. JSON in prod (Render filters per-field on JSON);
# `LOG_FORMAT=text` in `.env` switches to a human-readable format locally.
setup_logging(level=settings.log_level, fmt=settings.log_format)

# Sentry MUST init before the FastAPI app is constructed so the integration
# can wrap the framework's startup. No-op when SENTRY_DSN is unset (the
# common case for local dev / CI). The `_scrub_pii` before_send hook
# strips chat content / persona / emails before any event is sent.
init_sentry(
    dsn=settings.sentry_dsn,
    environment=settings.sentry_environment,
    traces_sample_rate=settings.sentry_traces_sample_rate,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bt_init()  # no-op when BRAINTRUST_ENABLED=false; safe to always call
    await init_db()
    await get_supabase()  # warm the async Storage client
    yield


app = FastAPI(title="Travel Buddy API", lifespan=lifespan)

# Attach the rate limiter so the @limiter.limit decorators on individual
# routes can find it via request.app.state.limiter. The 429 response
# shape and Retry-After header are owned by our handler, not SlowAPI's
# default (the chat client parses both).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Order matters: middleware registered later wraps the handler innermost,
# so register CORS first (innermost — closest to the route), then the
# request logger (outermost — runs first on the way in, last on the way
# out). That way the access log captures the response status post-CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    # PATCH is the most-used mutation verb in trips router (apply_*_stop /
    # apply_*_day endpoints at app/routers/trips.py:179,263,296) and PUT is
    # used by the full-trip replace path at line 320; both were silently
    # missing before docs/07 item 6. OPTIONS is the CORS preflight verb
    # browsers send before non-simple requests — without it, any cross-origin
    # PATCH/PUT/DELETE fails the preflight before the real request is sent.
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    # Tighter than `["*"]`: explicit allow-list trims the surface to what
    # the Flutter client actually sends. Add new headers here as needed.
    allow_headers=["Authorization", "Content-Type"],
)
app.middleware("http")(request_logging_middleware)

app.include_router(health.router)
app.include_router(users.router)
app.include_router(preferences.router)
app.include_router(conversations.router)
app.include_router(chat.router)
app.include_router(trips.router)
app.include_router(images.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
