"""DeepSeek fallback tests — both triggers, end to end through ChatAgent.

Trigger (A): Gemini fails before any text with 503/429/the 403 origin-block →
the SAME turn completes via DeepSeek (incl. emit_itinerary tool-calling).
Trigger (B): a budget gate would block → the turn is served by DeepSeek
instead of being hard-blocked. Plus the negatives (text already streamed,
non-eligible error, DeepSeek disabled) and the text-only image-drop flag.

DeepSeek is exercised through `tests/fakes/fake_deepseek.py` — no network.
The chat agent streams via `ChatStreamRouter`, which reads
`app.llm.gemini_client.client` and `app.llm.deepseek_client.client`, so we
patch those shared objects.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents.chat_agent import ChatAgent
from app.config import settings
from app.llm import deepseek_client as deepseek_client_module
from app.llm import gemini_client as gemini_client_module
from app.schemas import ChatRequest
from tests.fakes.fake_deepseek import fake_deepseek_client, usage
from tests.test_agents_chat_agent import _ctx, _drain, _events, _new_trace

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Gemini failure fakes.
# ---------------------------------------------------------------------------


class _GeminiError(Exception):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


def _gemini_raise_on_open(*, code: int | None = 503, message: str = "503 high demand"):
    """A client whose stream open raises before any chunk — the common
    capacity/rate/origin-block failure mode."""
    err = _GeminiError(message, code)

    async def _factory(**_kwargs):
        raise err

    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=_factory)
        )
    )


def _gemini_text_then_raise(text: str, *, code: int = 503):
    """A client that streams one text chunk and THEN raises — used to prove we
    do NOT fall back once bytes have hit the wire."""
    err = _GeminiError(str(code), code)

    async def _factory(**_kwargs):
        async def _gen():
            yield SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text=text, function_call=None)]
                        ),
                        finish_reason=None,
                    )
                ],
                usage_metadata=None,
            )
            raise err

        return _gen()

    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=_factory)
        )
    )


@pytest.fixture
def deepseek_on(monkeypatch):
    """Enable the fallback for a test (a real key gates every DeepSeek call)."""
    monkeypatch.setattr(settings, "deepseek_api_key", "test-deepseek-key")


def _install_deepseek(monkeypatch, **kwargs):
    monkeypatch.setattr(
        deepseek_client_module, "client", fake_deepseek_client(**kwargs)
    )


# ---------------------------------------------------------------------------
# Trigger A — error fallback.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gemini_client",
    [
        _gemini_raise_on_open(code=503, message="503 high demand"),
        _gemini_raise_on_open(code=429, message="429 rate limited"),
        _gemini_raise_on_open(
            code=None,
            message="403 <html>Error 403 (Forbidden)!!1</html>",
        ),
    ],
    ids=["503", "429", "403-html"],
)
async def test_trigger_a_gemini_error_completes_via_deepseek(
    monkeypatch, fake_db, deepseek_on, gemini_client
):
    monkeypatch.setattr(gemini_client_module, "client", gemini_client)
    _install_deepseek(
        monkeypatch,
        text_chunks=["served by ", "deepseek"],
        usage_obj=usage(input_tokens=80, output_tokens=40),
    )

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hi"),
        ctx=_ctx(),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    frames = await _drain(agent.stream())

    # Text streamed from DeepSeek (each chunk is its own SSE frame); turn
    # completed normally (DONE, no error).
    body = "".join(frames)
    assert "served by " in body and "deepseek" in body
    assert frames[-1] == "data: [DONE]\n\n"
    assert "error" not in _events(frames)
    # Telemetry attributes the served provider/model.
    assert trace.served_by == "deepseek"
    assert trace.model == "deepseek-v4-flash"
    assert trace.fallback_reason is not None
    assert trace.fallback_reason.startswith("gemini_error:")
    # Usage + cost came from the DeepSeek response at DeepSeek rates.
    assert trace.input_tokens == 80
    assert trace.output_tokens == 40
    assert trace.estimated_cost_usd == pytest.approx(
        80 * 0.14 / 1_000_000 + 40 * 0.28 / 1_000_000
    )


async def test_trigger_a_emit_itinerary_tool_call_through_deepseek(
    monkeypatch, fake_db, deepseek_on
):
    """The fallback must still produce a valid `event: itinerary` via DeepSeek
    tool-calling (fragmented tool_call args accumulated + schema-validated)."""
    monkeypatch.setattr(
        gemini_client_module, "client", _gemini_raise_on_open()
    )
    itinerary = {
        "title": "Berat",
        "days": [
            {"day_number": 1, "stops": [{"name": "Berat Castle", "kind": "sight"}]}
        ],
    }
    _install_deepseek(
        monkeypatch,
        text_chunks=["Here's your plan ✨"],
        tool_uses=[{"name": "emit_itinerary", "input": itinerary}],
        stop_reason="tool_calls",
        usage_obj=usage(input_tokens=120, output_tokens=90),
    )

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="2 days in Berat"),
        ctx=_ctx(),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    frames = await _drain(agent.stream())
    events = _events(frames)

    assert "itinerary" in events
    assert "itinerary_invalid" not in events
    assert frames[-1] == "data: [DONE]\n\n"
    assert trace.served_by == "deepseek"
    assert trace.itinerary_emitted is True
    assert trace.itinerary_days == 1


async def test_trigger_a_no_fallback_after_text_emitted(
    monkeypatch, fake_db, deepseek_on
):
    """A mid-stream failure AFTER bytes hit the wire must NOT switch providers
    — it re-raises and surfaces `event: error` (today's behavior)."""
    monkeypatch.setattr(
        gemini_client_module, "client", _gemini_text_then_raise("partial ")
    )
    # Provide a DeepSeek client that would succeed — proving we never call it.
    _install_deepseek(monkeypatch, text_chunks=["SHOULD NOT APPEAR"])

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hi"),
        ctx=_ctx(),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    frames = await _drain(agent.stream())
    body = "".join(frames)

    assert "partial " in body  # the Gemini text that did stream
    assert "SHOULD NOT APPEAR" not in body
    assert "error" in _events(frames)
    assert trace.served_by == "gemini"


async def test_non_eligible_gemini_error_does_not_fall_back(
    monkeypatch, fake_db, deepseek_on
):
    """Only 503/429/403 spill. A different error re-raises → `event: error`."""
    monkeypatch.setattr(
        gemini_client_module,
        "client",
        _gemini_raise_on_open(code=500, message="500 internal"),
    )
    _install_deepseek(monkeypatch, text_chunks=["SHOULD NOT APPEAR"])

    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hi"),
        ctx=_ctx(),
        user_id=None,
        trace=_new_trace(),
        started_at=0.0,
    )
    frames = await _drain(agent.stream())
    assert "SHOULD NOT APPEAR" not in "".join(frames)
    assert "error" in _events(frames)


async def test_fallback_disabled_when_key_unset(monkeypatch, fake_db):
    """No DEEPSEEK_API_KEY → Gemini error surfaces unchanged (no fallback)."""
    monkeypatch.setattr(settings, "deepseek_api_key", None)
    monkeypatch.setattr(
        gemini_client_module, "client", _gemini_raise_on_open()
    )
    _install_deepseek(monkeypatch, text_chunks=["SHOULD NOT APPEAR"])

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hi"),
        ctx=_ctx(),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    frames = await _drain(agent.stream())
    assert "SHOULD NOT APPEAR" not in "".join(frames)
    assert "error" in _events(frames)
    assert trace.served_by == "gemini"


# ---------------------------------------------------------------------------
# Trigger B — budget overflow.
# ---------------------------------------------------------------------------


async def test_trigger_b_budget_overflow_routes_to_deepseek(
    monkeypatch, fake_db, deepseek_on
):
    """A forced fallback (budget gate would block) serves DeepSeek directly,
    never touching Gemini."""
    # Gemini would raise if called — proving the forced path skips it entirely.
    monkeypatch.setattr(
        gemini_client_module,
        "client",
        _gemini_raise_on_open(code=500, message="must not be called"),
    )
    _install_deepseek(
        monkeypatch,
        text_chunks=["budget spill"],
        usage_obj=usage(input_tokens=10, output_tokens=5),
    )

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hi"),
        ctx=_ctx(),
        user_id=None,
        trace=trace,
        started_at=0.0,
        force_fallback_reason="budget_user",
    )
    frames = await _drain(agent.stream())

    assert "budget spill" in "".join(frames)
    assert frames[-1] == "data: [DONE]\n\n"
    assert trace.served_by == "deepseek"
    assert trace.fallback_reason == "budget_user"


# ---------------------------------------------------------------------------
# Image-drop flag (text-only model).
# ---------------------------------------------------------------------------


async def test_images_dropped_flag_set_on_fallback_turn(
    monkeypatch, fake_db, deepseek_on
):
    monkeypatch.setattr(
        gemini_client_module, "client", _gemini_raise_on_open()
    )
    _install_deepseek(monkeypatch, text_chunks=["ok"])

    ctx = _ctx()
    ctx.messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "source": {"url": "http://x/y.jpg"}},
            ],
        }
    ]
    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="what is this?"),
        ctx=ctx,
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    await _drain(agent.stream())
    assert trace.images_dropped is True
    assert trace.served_by == "deepseek"
