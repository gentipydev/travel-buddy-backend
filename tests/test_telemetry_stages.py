"""Tests for `RequestTrace.stage()` and the module-level `stage()` helper.

Two surfaces:
  - `RequestTrace.stage(name)` — the dataclass method, used directly by
    ChatAgent for the stages it owns (anthropic_stream, persistence,
    tool_dispatch via manual timer in this revision).
  - `app.observability.stage(name)` — the standalone helper, used by
    deeper modules (chat_pipeline, retrieval). MUST no-op when the
    `trace_var` ContextVar is unset so those modules stay unit-testable
    without a trace context.
"""
import time

from app.observability import stage as stage_helper
from app.observability import trace_var
from app.observability.telemetry import RequestTrace


def _new_trace() -> RequestTrace:
    """Minimal valid RequestTrace for stage testing — values don't matter."""
    return RequestTrace(
        request_id="test",
        timestamp="2026-05-18T00:00:00+00:00",
        user_id=None,
        message_length=0,
        retrieval_used=False,
        chunks_returned=0,
        top_rerank_score=None,
        input_tokens=None,
        output_tokens=None,
        latency_ms=0.0,
        model="claude-haiku-4-5-20251001",
        error=None,
    )


# ---------------------------------------------------------------------------
# RequestTrace.stage()
# ---------------------------------------------------------------------------


def test_stage_records_elapsed_for_named_block():
    trace = _new_trace()
    with trace.stage("rag_retrieval"):
        time.sleep(0.01)
    assert "rag_retrieval" in trace.stage_latencies
    # > 0 and within a generous upper bound — perf_counter slack on CI.
    assert 0 < trace.stage_latencies["rag_retrieval"] < 1.0


def test_stage_separate_names_recorded_independently():
    trace = _new_trace()
    with trace.stage("foo"):
        time.sleep(0.005)
    with trace.stage("bar"):
        time.sleep(0.005)
    assert set(trace.stage_latencies.keys()) == {"foo", "bar"}


def test_stage_repeated_name_sums():
    """Multiple `stage("rag_retrieval")` in one turn (e.g. two RAG calls)
    must accumulate, not overwrite — dashboards need the full per-turn
    time, not just the last call's slice."""
    trace = _new_trace()
    with trace.stage("rag_retrieval"):
        time.sleep(0.005)
    first = trace.stage_latencies["rag_retrieval"]
    with trace.stage("rag_retrieval"):
        time.sleep(0.005)
    second = trace.stage_latencies["rag_retrieval"]
    # Second total is at least the first + the second block's duration.
    assert second > first


def test_stage_records_even_when_block_raises():
    """The finally branch must fire so a partial-time recording lands;
    otherwise an exception in `rag_retrieval` would silently drop its
    latency from the dashboard."""
    trace = _new_trace()
    try:
        with trace.stage("boom"):
            raise RuntimeError("intentional")
    except RuntimeError:
        pass
    assert "boom" in trace.stage_latencies
    assert trace.stage_latencies["boom"] >= 0


# ---------------------------------------------------------------------------
# Module-level stage() helper (reads trace_var)
# ---------------------------------------------------------------------------


def test_helper_noops_when_trace_var_unset():
    """`chat_pipeline.py` and `retrieval.py` import this helper at module
    scope; they MUST be importable + callable from tests that don't set
    up a request trace (most of the test suite)."""
    # No trace_var.set() — default is None.
    with stage_helper("not_recorded"):
        time.sleep(0.001)
    # Nothing to assert on the trace because there isn't one. The win is
    # the absence of an exception above.


def test_helper_records_into_active_trace():
    trace = _new_trace()
    token = trace_var.set(trace)
    try:
        with stage_helper("rag_retrieval"):
            time.sleep(0.005)
    finally:
        trace_var.reset(token)
    assert "rag_retrieval" in trace.stage_latencies
    assert trace.stage_latencies["rag_retrieval"] > 0


def test_helper_sums_repeated_names_into_active_trace():
    trace = _new_trace()
    token = trace_var.set(trace)
    try:
        with stage_helper("rag_retrieval"):
            time.sleep(0.003)
        with stage_helper("rag_retrieval"):
            time.sleep(0.003)
    finally:
        trace_var.reset(token)
    # Single key, summed value.
    assert list(trace.stage_latencies.keys()) == ["rag_retrieval"]
    assert trace.stage_latencies["rag_retrieval"] > 0
