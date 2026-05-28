"""Tests for `app.domain.chat_pipeline.prepare_turn`.

`prepare_turn` is the per-turn context assembler the chat router calls
before instantiating ChatAgent. It loads the user, resolves / creates
the conversation, runs RAG, detects a place, and packages everything
into a `ChatTurnContext`. The chat agent owns streaming and telemetry on
top of that.

Test seam: monkey-patch every helper imported by chat_pipeline (`_load_trip`,
`get_or_create_conversation`, `retrieve_chunks`, `detect_place`,
`build_system_payload`, `build_history`) with canned values, then call
`prepare_turn` with a `ChatRequest` and a `FakeSession` and assert on
the returned ChatTurnContext.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from app.db.models import Conversation, Trip, User
from app.domain import chat_pipeline as pipeline_module
from app.domain.chat_pipeline import prepare_turn
from app.schemas import ChatRequest

pytestmark = pytest.mark.anyio


_USER_LOCAL_ID = 5
_USER_SUPA_ID = "supa-uuid-abc"


def _make_user() -> User:
    now = datetime.now(UTC)
    return User(
        id=_USER_LOCAL_ID,
        supabase_uid=_USER_SUPA_ID,
        email="t@example.com",
        created_at=now,
    )


def _make_trip() -> Trip:
    now = datetime.now(UTC)
    trip = Trip(
        id=uuid.uuid4(),
        user_id=_USER_LOCAL_ID,
        title="Saved trip",
        summary=None,
        start_date=None,
        created_at=now,
        updated_at=now,
    )
    trip.days = []
    return trip


def _make_conversation(*, conv_id: uuid.UUID | None = None) -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        id=conv_id or uuid.uuid4(),
        user_id=_USER_LOCAL_ID,
        title=None,
        created_at=now,
        updated_at=now,
    )


def _stub_pipeline_helpers(
    monkeypatch,
    *,
    chunks: list | None = None,
    place=None,
    refine_trip: Trip | None = None,
    conversation: Conversation | None = None,
):
    """Replace the heavy helpers with light stubs. Returns nothing — tests
    that need to inspect call args should set their own monkeypatch on top."""
    async def _fake_retrieve(query, db, history):
        return (chunks or [], 0.5 if chunks else None)

    async def _fake_detect(message, db):
        return place

    async def _fake_load_trip(db, trip_id, user_id):
        return refine_trip

    async def _fake_get_conv(db, conv_id, user_id):
        return conversation or _make_conversation(conv_id=conv_id)

    async def _fake_get_refine_conv(db, *, trip, user_id):
        # Reuse the same canned conversation when provided so tests can pin
        # its id.
        return conversation or _make_conversation()

    async def _fake_db_refresh(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline_module, "retrieve_chunks", _fake_retrieve)
    monkeypatch.setattr(pipeline_module, "detect_place", _fake_detect)
    monkeypatch.setattr(pipeline_module, "to_card_dict", lambda p: {"name": "X"})
    monkeypatch.setattr(pipeline_module, "_load_trip", _fake_load_trip)
    monkeypatch.setattr(
        pipeline_module, "get_or_create_conversation", _fake_get_conv
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_or_create_refine_conversation",
        _fake_get_refine_conv,
    )
    monkeypatch.setattr(
        pipeline_module, "build_system_payload",
        lambda prefs, trip_context=None: ("CACHED_PREFIX", ""),
    )
    monkeypatch.setattr(pipeline_module, "build_history", lambda conv: [])
    monkeypatch.setattr(
        pipeline_module, "serialize_trip_for_context", lambda trip: {"title": trip.title}
    )


# ---------------------------------------------------------------------------
# Anonymous path: no supa_user → conversation_history wins, no conv row.
# ---------------------------------------------------------------------------


async def test_anonymous_request_uses_client_history(monkeypatch, fake_db):
    _stub_pipeline_helpers(monkeypatch)
    ctx = await prepare_turn(
        ChatRequest(message="hi", conversation_history=[]),
        supa_user=None,
        db=fake_db,
    )
    assert ctx.conversation is None
    assert ctx.new_conversation_id is None
    assert ctx.refine_trip is None


async def test_rag_chunks_wrap_into_volatile_suffix(monkeypatch, fake_db):
    """When retrieve_chunks returns hits, the joined content lands in
    volatile_suffix under the BEGIN/END REFERENCE MATERIAL delimiters that
    the system prompt instructs the model to treat as data, not instructions."""
    class _C:
        def __init__(self, content):
            self.content = content
    chunks = [_C("Berat is a UNESCO site"), _C("Visit Onufri museum")]
    _stub_pipeline_helpers(monkeypatch, chunks=chunks)
    ctx = await prepare_turn(
        ChatRequest(message="Tell me about Berat"),
        supa_user=None,
        db=fake_db,
    )
    assert ctx.retrieval_used is True
    assert ctx.chunks_returned == 2
    assert ctx.top_rerank_score == 0.5
    assert "BEGIN REFERENCE MATERIAL" in ctx.volatile_suffix
    assert "Berat is a UNESCO site" in ctx.volatile_suffix
    assert "Onufri" in ctx.volatile_suffix


async def test_no_rag_leaves_volatile_suffix_bare(monkeypatch, fake_db):
    _stub_pipeline_helpers(monkeypatch, chunks=[])
    ctx = await prepare_turn(
        ChatRequest(message="hi"), supa_user=None, db=fake_db,
    )
    assert ctx.retrieval_used is False
    assert ctx.chunks_returned == 0
    assert "BEGIN REFERENCE MATERIAL" not in ctx.volatile_suffix


# ---------------------------------------------------------------------------
# Authenticated path: user lookup + conversation persistence.
# ---------------------------------------------------------------------------


async def test_authenticated_new_conversation_id_emitted(monkeypatch, fake_db):
    """When no conversation_id is sent and a User row exists, the pipeline
    creates a conversation and returns its id in new_conversation_id so the
    chat agent ships `event: conversation_id` first on the stream."""
    user = _make_user()
    fake_db.queue(user)              # SELECT User
    fake_db.queue(None)              # SELECT UserPreferences
    conv = _make_conversation()
    _stub_pipeline_helpers(monkeypatch, conversation=conv)
    ctx = await prepare_turn(
        ChatRequest(message="hi"),
        supa_user={"id": _USER_SUPA_ID},
        db=fake_db,
    )
    assert ctx.new_conversation_id == conv.id
    assert ctx.conversation is conv


async def test_authenticated_existing_conversation_id_not_re_emitted(
    monkeypatch, fake_db
):
    """When the client sends a conversation_id, new_conversation_id stays None
    so the chat agent doesn't emit a redundant `event: conversation_id` frame."""
    user = _make_user()
    fake_db.queue(user)
    fake_db.queue(None)
    existing_id = uuid.uuid4()
    conv = _make_conversation(conv_id=existing_id)
    _stub_pipeline_helpers(monkeypatch, conversation=conv)
    ctx = await prepare_turn(
        ChatRequest(message="hi", conversation_id=existing_id),
        supa_user={"id": _USER_SUPA_ID},
        db=fake_db,
    )
    assert ctx.new_conversation_id is None
    assert ctx.conversation is conv


# ---------------------------------------------------------------------------
# Refine mode: trip_id present.
# ---------------------------------------------------------------------------


async def test_anonymous_refine_raises_404(monkeypatch, fake_db):
    """Anonymous + trip_id → nothing to refine. 404 (not 401) so a guessed
    trip UUID reveals nothing about its existence."""
    _stub_pipeline_helpers(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        await prepare_turn(
            ChatRequest(message="refine", trip_id=uuid.uuid4()),
            supa_user=None,
            db=fake_db,
        )
    assert exc.value.status_code == 404


async def test_refine_with_unowned_trip_raises_404(monkeypatch, fake_db):
    """Auth gate: _load_trip returns None (wrong owner or doesn't exist) →
    404 from the trip-loader path, identical detail to "doesn't exist"."""
    fake_db.queue(_make_user())
    fake_db.queue(None)  # no preferences
    _stub_pipeline_helpers(monkeypatch, refine_trip=None)  # _load_trip → None
    with pytest.raises(HTTPException) as exc:
        await prepare_turn(
            ChatRequest(message="refine", trip_id=uuid.uuid4()),
            supa_user={"id": _USER_SUPA_ID},
            db=fake_db,
        )
    assert exc.value.status_code == 404


async def test_refine_mismatched_conversation_id_raises_400(monkeypatch, fake_db):
    """If the client sends a conversation_id that doesn't match the trip's
    1:1 conversation, the pipeline refuses rather than silently rewriting the
    wrong thread — fail loud."""
    fake_db.queue(_make_user())
    fake_db.queue(None)
    trip = _make_trip()
    conv = _make_conversation()  # conv.id != the wrong_id we send
    _stub_pipeline_helpers(monkeypatch, refine_trip=trip, conversation=conv)
    wrong_id = uuid.uuid4()
    with pytest.raises(HTTPException) as exc:
        await prepare_turn(
            ChatRequest(
                message="refine", trip_id=trip.id, conversation_id=wrong_id,
            ),
            supa_user={"id": _USER_SUPA_ID},
            db=fake_db,
        )
    assert exc.value.status_code == 400
    assert "trip_id does not match" in exc.value.detail


async def test_refine_loads_trip_and_sets_refine_mode(monkeypatch, fake_db):
    """Happy refine path: trip loads, conversation auto-resolves to the trip's
    1:1, ctx.refine_trip is set so the agent branches into the refine path."""
    fake_db.queue(_make_user())
    fake_db.queue(None)
    trip = _make_trip()
    _stub_pipeline_helpers(monkeypatch, refine_trip=trip)
    ctx = await prepare_turn(
        ChatRequest(message="refine", trip_id=trip.id),
        supa_user={"id": _USER_SUPA_ID},
        db=fake_db,
    )
    assert ctx.refine_trip is trip


# ---------------------------------------------------------------------------
# Multimodal: images attached.
# ---------------------------------------------------------------------------


async def test_images_produce_list_user_content(monkeypatch, fake_db):
    """Image-bearing turns wrap the message into a list of image+text content
    blocks (Anthropic's vision payload shape) rather than a string."""
    _stub_pipeline_helpers(monkeypatch)
    ctx = await prepare_turn(
        ChatRequest(
            message="What's this?", images=["https://example.com/x.jpg"],
        ),
        supa_user=None,
        db=fake_db,
    )
    last = ctx.messages[-1]
    assert last["role"] == "user"
    assert isinstance(last["content"], list)
    assert last["content"][0]["type"] == "image"
    assert last["content"][1]["type"] == "text"


async def test_no_images_produce_string_user_content(monkeypatch, fake_db):
    _stub_pipeline_helpers(monkeypatch)
    ctx = await prepare_turn(
        ChatRequest(message="Just text"), supa_user=None, db=fake_db,
    )
    last = ctx.messages[-1]
    assert isinstance(last["content"], str)
    assert last["content"] == "Just text"


# ---------------------------------------------------------------------------
# Place detection.
# ---------------------------------------------------------------------------


async def test_place_detection_populates_card_payload(monkeypatch, fake_db):
    """When the place detector finds a referenced place, its serialized card
    JSON is set on ctx.place_card_payload so the chat agent emits the
    `event: place_card` frame BEFORE the text stream."""
    _stub_pipeline_helpers(monkeypatch, place=object())  # truthy sentinel
    ctx = await prepare_turn(
        ChatRequest(message="Tell me about Berat"),
        supa_user=None, db=fake_db,
    )
    assert ctx.place_card_payload is not None
    assert "name" in ctx.place_card_payload  # JSON-serialized dict


async def test_no_place_leaves_card_payload_none(monkeypatch, fake_db):
    _stub_pipeline_helpers(monkeypatch, place=None)
    ctx = await prepare_turn(
        ChatRequest(message="hi"), supa_user=None, db=fake_db,
    )
    assert ctx.place_card_payload is None
