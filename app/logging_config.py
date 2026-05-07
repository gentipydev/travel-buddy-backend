"""Centralised logging setup for the Travel Buddy backend.

Goals
-----
- One JSON line per record so Render's Logs UI can filter by field.
- Every record carries a `request_id` (and `user_id` when auth resolved)
  so a single request is traceable across multiple log lines.
- Existing `logger = logging.getLogger(__name__)` call sites work unchanged
  — the format upgrade happens entirely inside the formatter.
- Uvicorn's own loggers (`uvicorn`, `uvicorn.access`, `uvicorn.error`) are
  re-attached to the same handler so app + framework logs share format.

Usage
-----
1. `from app.logging_config import setup_logging, request_logging_middleware`
2. Call `setup_logging(level, fmt)` at module top of `app/main.py` BEFORE
   creating the FastAPI app — uvicorn's loggers are configured during
   `uvicorn.run()` import path, so the earlier we hook in the better.
3. Register `request_logging_middleware` via
   `app.middleware("http")(request_logging_middleware)`.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from fastapi import Request, Response

# ---------------------------------------------------------------------------
# Request-scoped context, propagated across `await` via ContextVar.
# ---------------------------------------------------------------------------

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)


# ---------------------------------------------------------------------------
# Filter: injects request_id / user_id into every LogRecord.
# ---------------------------------------------------------------------------

class RequestContextFilter(logging.Filter):
    """Stamps the active request_id and user_id onto every LogRecord so the
    formatter can include them without each call site needing to pass them
    via `extra=...`."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.user_id = user_id_var.get()
        return True


# ---------------------------------------------------------------------------
# JSON formatter: one valid JSON object per line.
# ---------------------------------------------------------------------------

# Keys the Python logging machinery sets on every LogRecord; `extra={...}`
# attributes on a call (e.g. `logger.info("x", extra={"latency_ms": 42})`)
# are merged into the JSON payload by skipping these reserved names.
_RESERVED_LOGRECORD_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "request_id", "user_id",
    # `message` is set by Formatter.format(); ignore the raw `msg` too.
    "message", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Serialise each LogRecord as a single-line JSON object.

    Layout:
        {"ts": "2026-...", "level": "info", "logger": "app.routers.chat",
         "message": "...", "request_id": "...", "user_id": "...",
         "<extra-key>": <extra-value>, ...,
         "exception": "Traceback..." }     # only when exc_info
    """

    def format(self, record: logging.LogRecord) -> str:
        # `record.created` is the wall-clock float seconds; `time.strftime`
        # (which the base Formatter.formatTime uses) doesn't expand `%f`,
        # so we build the ISO-8601 timestamp via `datetime` for ms precision.
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload: dict[str, Any] = {
            "ts": dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = getattr(record, "request_id", None)
        uid = getattr(record, "user_id", None)
        if rid:
            payload["request_id"] = rid
        if uid:
            payload["user_id"] = uid
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_KEYS:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable single-line format for local dev (LOG_FORMAT=text)."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, datefmt="%H:%M:%S")
        rid = getattr(record, "request_id", None) or "-"
        uid = getattr(record, "user_id", None) or "-"
        head = f"{ts} {record.levelname:<5} [{record.name}] req={rid} usr={uid}"
        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        return f"{head} {msg}"


# ---------------------------------------------------------------------------
# Setup: configures the root logger + uvicorn loggers in one place.
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure the root logger.

    `level` is anything `logging.getLevelName` accepts (`DEBUG`, `INFO`...).
    `fmt` is `"json"` (default; right for Render) or `"text"` (local dev).
    Idempotent — calling twice replaces handlers cleanly.
    """
    formatter: logging.Formatter
    if fmt == "text":
        formatter = TextFormatter()
    else:
        formatter = JsonFormatter()

    handler = logging.StreamHandler()  # stderr; Render captures this.
    handler.setFormatter(formatter)
    handler.addFilter(RequestContextFilter())

    root = logging.getLogger()
    # Drop any existing handlers (basicConfig leaves one behind).
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Re-route uvicorn's loggers through the same handler so framework logs
    # match app logs. `uvicorn.access` is set to WARNING by default here
    # because our request_logging_middleware emits a richer access record;
    # bump to INFO if you want both.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        for existing in list(lg.handlers):
            lg.removeHandler(existing)
        lg.addHandler(handler)
        lg.propagate = False
        if name == "uvicorn.access":
            lg.setLevel(logging.WARNING)
        else:
            lg.setLevel(level.upper())


# ---------------------------------------------------------------------------
# Middleware: per-request access log + request_id binding.
# ---------------------------------------------------------------------------

_access_logger = logging.getLogger("app.access")


async def request_logging_middleware(request: Request, call_next):
    """FastAPI HTTP middleware. Sets request_id contextvar, logs request
    start + end, echoes `X-Request-ID` in the response so clients can
    correlate. Place this *before* any auth-bearing routes are invoked
    (FastAPI middleware order: first registered runs outermost)."""

    inbound = request.headers.get("x-request-id") or str(uuid.uuid4())
    rid_token = request_id_var.set(inbound)
    uid_token = user_id_var.set(None)
    started = time.perf_counter()
    method = request.method
    path = request.url.path

    _access_logger.info(
        "request.start",
        extra={"http_method": method, "path": path},
    )

    status_code = 500
    try:
        response: Response = await call_next(request)
        status_code = response.status_code
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _access_logger.exception(
            "request.error",
            extra={
                "http_method": method,
                "path": path,
                "duration_ms": round(elapsed_ms, 2),
            },
        )
        request_id_var.reset(rid_token)
        user_id_var.reset(uid_token)
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.headers["X-Request-ID"] = inbound

    level = logging.WARNING if status_code >= 400 else logging.INFO
    _access_logger.log(
        level,
        "request.end",
        extra={
            "http_method": method,
            "path": path,
            "status": status_code,
            "duration_ms": round(elapsed_ms, 2),
        },
    )

    request_id_var.reset(rid_token)
    user_id_var.reset(uid_token)
    return response
