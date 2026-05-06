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
    itinerary_emitted: bool = False
    itinerary_days: int | None = None
    # Set when Claude emitted an emit_itinerary tool block but it failed
    # TripCreate validation. `itinerary_emitted` stays False on this path so
    # success-rate dashboards remain accurate; this field is the signal that
    # the model attempted to emit a structured plan and we rejected it.
    itinerary_validation_error: str | None = None
    # True when the request carried a `trip_id` and the pipeline ran in
    # "Refine with Buddy" mode. Lets us measure refinement vs. fresh planning
    # rates independently of itinerary_emitted (the model may chat without
    # committing a tool call on a refine turn).
    trip_refine: bool = False


def log_trace(trace: RequestTrace) -> None:
    """Emit `trace` as a single JSON line at INFO. Never raises."""
    try:
        logger.info(json.dumps(asdict(trace)))
    except Exception:
        try:
            logger.exception("telemetry: failed to emit trace")
        except Exception:
            pass
