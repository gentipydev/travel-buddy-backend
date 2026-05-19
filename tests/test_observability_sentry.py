"""Tests for the Sentry PII scrubber.

Item 8: ensure no user-generated content leaves the process via Sentry.
Run independently of `init_sentry()` — the scrubber is pure-data on the
event dict and doesn't need an initialized SDK.
"""
from __future__ import annotations

import pytest

from app.observability.sentry import _scrub_pii, init_sentry


pytestmark = pytest.mark.anyio


def test_chat_message_is_redacted():
    event = {
        "request": {
            "url": "http://localhost/chat/stream",
            "data": {"message": "What should I do in Berat?"},
        }
    }
    out = _scrub_pii(event, {})
    assert out is not None
    assert out["request"]["data"]["message"] == "[redacted]"


def test_user_email_is_redacted():
    event = {"user": {"id": "uid-123", "email": "alice@example.com"}}
    out = _scrub_pii(event, {})
    assert out["user"]["email"] == "[redacted]"
    # Non-PII identifiers stay intact for debugging.
    assert out["user"]["id"] == "uid-123"


def test_nested_persona_payload_is_redacted():
    event = {
        "request": {
            "data": {
                "persona": {
                    "tone": "friendly",
                    "personality": "loves history and food",
                }
            }
        }
    }
    out = _scrub_pii(event, {})
    assert out["request"]["data"]["persona"]["tone"] == "[redacted]"
    assert out["request"]["data"]["persona"]["personality"] == "[redacted]"


def test_trip_stop_names_in_message_list_are_redacted():
    event = {
        "request": {
            "data": {
                "messages": [
                    {"role": "user", "content": "Add Castle of Berat"},
                    {"role": "assistant", "content": "Added."},
                ]
            }
        }
    }
    out = _scrub_pii(event, {})
    # `messages` itself is a PII key — every string leaf in the subtree is
    # scrubbed, including `role`. This is the safer choice: we'd rather lose
    # the role label than ever leak content. The structural shape of the
    # event (list of dicts with the same keys) is preserved for debugging.
    for m in out["request"]["data"]["messages"]:
        assert m["content"] == "[redacted]"
        assert m["role"] == "[redacted]"


def test_breadcrumb_data_is_redacted():
    event = {
        "breadcrumbs": {
            "values": [
                {"category": "http", "data": {"message": "POST body: ..."}},
                {"category": "info", "message": "ok"},
            ]
        }
    }
    out = _scrub_pii(event, {})
    assert out["breadcrumbs"]["values"][0]["data"]["message"] == "[redacted]"


def test_unrelated_fields_pass_through():
    event = {
        "request": {
            "data": {
                "trip_id": "abc",
                "limit": 20,
                "cursor": "2026-05-18T00:00:00|abc",
            }
        }
    }
    out = _scrub_pii(event, {})
    # Non-PII keys are untouched.
    assert out["request"]["data"]["trip_id"] == "abc"
    assert out["request"]["data"]["limit"] == 20


def test_raw_string_body_is_dropped():
    # FastAPI may pass through the raw body string when JSON parse fails;
    # we don't know its shape so we redact wholesale.
    event = {"request": {"data": '{"message":"hi"}'}}
    out = _scrub_pii(event, {})
    assert out["request"]["data"] == "[redacted]"


def test_scrubber_never_raises_on_malformed_event():
    # Pathological event — non-dict where we'd expect one. Must not raise.
    event = {"request": "this is not a dict", "user": ["nope"]}
    out = _scrub_pii(event, {})
    # Returns the event (possibly partially mutated) rather than None.
    assert out is not None


def test_exception_vars_are_redacted():
    event = {
        "exception": {
            "values": [
                {
                    "type": "ValueError",
                    "stacktrace": {
                        "frames": [
                            {
                                "function": "send",
                                "vars": {
                                    "message": "secret content",
                                    "request_id": "rid-1",
                                },
                            }
                        ]
                    },
                }
            ]
        }
    }
    out = _scrub_pii(event, {})
    frame = out["exception"]["values"][0]["stacktrace"]["frames"][0]
    assert frame["vars"]["message"] == "[redacted]"
    assert frame["vars"]["request_id"] == "rid-1"


def test_init_sentry_noops_when_dsn_empty(caplog):
    # Should not raise, should not initialize the SDK. Logged at INFO.
    import logging

    with caplog.at_level(logging.INFO, logger="app.observability.sentry"):
        init_sentry(dsn="", environment="test", traces_sample_rate=0.5)
    assert any("sentry=disabled" in rec.getMessage() for rec in caplog.records)
