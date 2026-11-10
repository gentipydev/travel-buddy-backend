"""classify_llm_error contract: named failure modes collapse, rest passes through.

The result is emitted verbatim on the SSE `event: error` frame, so the stable
prefixes are a wire contract — the client may pattern-match them. Anything the
classifier doesn't recognise must survive unchanged, so an unclassified failure
is never hidden.
"""
import anthropic
import httpx2
import pytest

from app.llm import (
    ANTHROPIC_AUTH_FAILED,
    ANTHROPIC_OVERLOADED,
    ANTHROPIC_RATE_LIMITED,
    ANTHROPIC_UNAVAILABLE,
    classify_llm_error,
)


def _status_error(cls, status_code: int):
    """Build a real SDK status error. The SDK's constructors want the
    originating httpx2 request/response, so synthesize a minimal pair."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status_code, request=request, json={})
    return cls("boom", response=response, body=None)


def test_authentication_error_collapses():
    out = classify_llm_error(_status_error(anthropic.AuthenticationError, 401))
    assert out.startswith(ANTHROPIC_AUTH_FAILED)
    assert "ANTHROPIC_API_KEY" in out


def test_permission_denied_collapses_to_auth():
    out = classify_llm_error(_status_error(anthropic.PermissionDeniedError, 403))
    assert out.startswith(ANTHROPIC_AUTH_FAILED)


def test_rate_limit_collapses():
    out = classify_llm_error(_status_error(anthropic.RateLimitError, 429))
    assert out.startswith(ANTHROPIC_RATE_LIMITED)


@pytest.mark.parametrize("status", [500, 503, 529])
def test_server_errors_collapse_to_overloaded(status):
    out = classify_llm_error(_status_error(anthropic.APIStatusError, status))
    assert out.startswith(ANTHROPIC_OVERLOADED)
    assert str(status) in out


def test_connection_error_collapses_to_unavailable():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    out = classify_llm_error(anthropic.APIConnectionError(request=request))
    assert out.startswith(ANTHROPIC_UNAVAILABLE)


def test_bad_request_passes_through_unchanged():
    """A 400 is our payload's fault, not a transient condition — it must keep
    its full message so the actual validation complaint is visible."""
    exc = _status_error(anthropic.BadRequestError, 400)
    out = classify_llm_error(exc)
    assert out == str(exc)
    for code in (
        ANTHROPIC_AUTH_FAILED,
        ANTHROPIC_RATE_LIMITED,
        ANTHROPIC_OVERLOADED,
        ANTHROPIC_UNAVAILABLE,
    ):
        assert code not in out


def test_unknown_exception_passes_through_unchanged():
    out = classify_llm_error(RuntimeError("something else entirely"))
    assert out == "something else entirely"
