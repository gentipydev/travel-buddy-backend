"""Observability primitives: logging setup, request-scoped context,
per-request telemetry traces, Braintrust spans.

Public API:
  - `setup_logging(level, fmt)` — configures the root logger + uvicorn
    loggers with the JSON or text formatter.
  - `request_logging_middleware` — FastAPI HTTP middleware that binds a
    request_id ContextVar and emits start/end access records.
  - `request_id_var`, `user_id_var`, `trace_var` — request-scoped
    ContextVars; the formatter reads the first two, the chat router
    sets `trace_var` so deep stages can record into `stage_latencies`
    via the `stage()` helper below.
  - `RequestTrace`, `log_trace`, `new_request_id`, `utc_now_iso` —
    per-request structured trace dataclass + emitter (one JSON line per
    /chat/stream call).
  - `stage(name)` — context manager that records elapsed seconds onto
    the ambient `RequestTrace`. No-op when `trace_var` is unset, so
    `app/rag/retrieval.py` and `app/domain/chat_pipeline.py` can be
    imported and unit-tested without a trace context.
  - `compute_cost(...)` — per-model `$/1M token` derivation (see
    `app/observability/pricing.py`).
  - `bt_init`, `bt_wrap_genai`, `chat_span` — Braintrust hooks. All
    no-op when BRAINTRUST_ENABLED=false (the default) or the SDK is not
    installed, so local dev / CI run unchanged.
"""
import time
from contextlib import contextmanager

from app.observability.braintrust import (
    bt_init,
    bt_wrap_genai,
    chat_span,
)
from app.observability.logging_config import (
    JsonFormatter,
    RequestContextFilter,
    TextFormatter,
    request_logging_middleware,
    setup_logging,
)
from app.observability.pricing import compute_cost
from app.observability.request_context import (
    request_id_var,
    trace_var,
    user_id_var,
)
from app.observability.telemetry import (
    RequestTrace,
    log_trace,
    new_request_id,
    utc_now_iso,
)


@contextmanager
def stage(name: str):
    """Record elapsed seconds into the ambient `RequestTrace`'s
    `stage_latencies` dict. No-op when `trace_var` is unset — lets
    `app/rag/retrieval.py` and `app/domain/chat_pipeline.py` be
    instrumented without forcing every test that imports them to set up
    a trace.
    """
    trace = trace_var.get()
    if trace is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        trace.stage_latencies[name] = (
            trace.stage_latencies.get(name, 0.0) + elapsed
        )


__all__ = [
    "JsonFormatter",
    "RequestContextFilter",
    "RequestTrace",
    "TextFormatter",
    "bt_init",
    "bt_wrap_genai",
    "chat_span",
    "compute_cost",
    "log_trace",
    "new_request_id",
    "request_id_var",
    "request_logging_middleware",
    "setup_logging",
    "stage",
    "trace_var",
    "user_id_var",
    "utc_now_iso",
]
