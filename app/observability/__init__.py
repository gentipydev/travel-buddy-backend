"""Observability primitives: logging setup, request-scoped context,
per-request telemetry traces.

Public API:
  - `setup_logging(level, fmt)` — configures the root logger + uvicorn
    loggers with the JSON or text formatter.
  - `request_logging_middleware` — FastAPI HTTP middleware that binds a
    request_id ContextVar and emits start/end access records.
  - `request_id_var`, `user_id_var` — request-scoped ContextVars; the
    formatter reads them and the auth dependency populates `user_id_var`.
  - `RequestTrace`, `log_trace`, `new_request_id`, `utc_now_iso` —
    per-request structured trace dataclass + emitter (one JSON line per
    /chat/stream call).
"""
from app.observability.logging_config import (
    JsonFormatter,
    RequestContextFilter,
    TextFormatter,
    request_logging_middleware,
    setup_logging,
)
from app.observability.request_context import request_id_var, user_id_var
from app.observability.telemetry import (
    RequestTrace,
    log_trace,
    new_request_id,
    utc_now_iso,
)

__all__ = [
    "JsonFormatter",
    "RequestContextFilter",
    "RequestTrace",
    "TextFormatter",
    "log_trace",
    "new_request_id",
    "request_id_var",
    "request_logging_middleware",
    "setup_logging",
    "user_id_var",
    "utc_now_iso",
]
