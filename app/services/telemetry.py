"""Per-request observability trace for the chat pipeline.

Emits one JSON line at INFO per /chat/stream request capturing retrieval,
token usage, latency, and errors. Logging is fire-and-forget — failures
inside `log_trace` must never propagate to the request path.
"""
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def new_request_id() -> str:
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RequestTrace:
    request_id: str
    timestamp: str
    user_id: str | None
    message_length: int
    retrieval_used: bool
    chunks_returned: int
    top_rerank_score: float | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    model: str
    error: str | None


def log_trace(trace: RequestTrace) -> None:
    """Emit `trace` as a single JSON line at INFO. Never raises."""
    try:
        logger.info(json.dumps(asdict(trace)))
    except Exception:
        try:
            logger.exception("telemetry: failed to emit trace")
        except Exception:
            pass
