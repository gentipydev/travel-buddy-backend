"""Request-scoped ContextVars.

`request_id_var` is set by `request_logging_middleware` for every HTTP
request; `user_id_var` is populated by the Supabase auth dependency
(`app/auth/supabase_auth.py`) once a JWT is verified.

Both vars are read by `RequestContextFilter` (in logging_config) so every
log line carries them without each call site passing `extra=...`.

`trace_var` is set by the `/chat/stream` router for the duration of one
request so deep stages (`app/domain/chat_pipeline.py`, `app/rag/retrieval.py`)
can append to `stage_latencies` via the `stage()` helper in
`app/observability/__init__.py` without each function signature gaining a
`trace` kwarg.
"""
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.observability.telemetry import RequestTrace

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
trace_var: ContextVar["RequestTrace | None"] = ContextVar(
    "trace", default=None
)
