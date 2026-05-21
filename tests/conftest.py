"""Shared fixtures for service-level tests.

Two helpers:

`FakeSession` — records every `execute` / `add` / `delete` / `flush` call and
serves canned results in FIFO order. Tests queue the rows they expect each
SELECT to return; the production code's logic is exercised against these
without a real Postgres or any ORM machinery beyond statement construction.

`MockTransportRegistry` — install per-URL response handlers, then patch the
private `_http_client` factories in `app.integrations.google_places` and
`app.integrations.google_directions` to use `httpx.MockTransport`. Each
handler can return a Response with any status / headers / body, so the
photo-redirect path is testable too.

Requires `pytest` and `pytest-asyncio` (configured via pytest.ini's
`asyncio_mode = auto`). Install with: `pip install pytest pytest-asyncio`.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from sqlalchemy.sql.expression import Select


# anyio's pytest plugin (already a transitive dep via httpx/FastAPI) runs
# tests that are tagged with `pytest.mark.anyio`. Auto-tag every async test
# function so individual test files don't need boilerplate.
def pytest_collection_modifyitems(config, items):
    for item in items:
        if asyncio.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.anyio)


@pytest.fixture
def anyio_backend() -> str:
    """Pin tests to the asyncio backend; trio isn't a project dep."""
    return "asyncio"


class FakeResult:
    def __init__(self, value: Any):
        self._value = value

    def scalar_one_or_none(self) -> Any:
        if isinstance(self._value, list):
            return self._value[0] if self._value else None
        return self._value

    def scalar_one(self) -> Any:
        if isinstance(self._value, list):
            return self._value[0]
        if self._value is None:
            raise AssertionError("scalar_one called but no value queued")
        return self._value

    def scalars(self):
        items = self._value if isinstance(self._value, list) else (
            [self._value] if self._value is not None else []
        )
        return _Scalars(items)

    def unique(self):
        return self


class _Scalars:
    def __init__(self, items: list):
        self._items = items

    def all(self) -> list:
        return list(self._items)


class FakeSession:
    """Stand-in for `sqlalchemy.ext.asyncio.AsyncSession`.

    Tests `.queue(value)` the next SELECT result before invoking code under
    test. INSERTs / UPDATEs (via `.add()` and `await execute(stmt)`) are
    recorded in `executed` / `added` for assertions.
    """

    def __init__(self):
        self.executed: list[Any] = []
        self.added: list[Any] = []
        self.deleted: list[Any] = []
        self.flushed: int = 0
        self.committed: int = 0
        self.rolled_back: int = 0
        self._next_results: list[Any] = []

    def queue(self, value: Any) -> None:
        self._next_results.append(value)

    async def execute(self, stmt: Any) -> FakeResult:
        self.executed.append(stmt)
        # Only consume queued results for SELECT statements. DML
        # (UPDATE / INSERT / DELETE) returns an empty result and leaves
        # the queue untouched so a sequence like
        # `_load_trip → _touch_trip → _load_trip` only needs the trip
        # queued twice rather than once + a placeholder for the UPDATE.
        if isinstance(stmt, Select):
            value = self._next_results.pop(0) if self._next_results else None
            return FakeResult(value)
        return FakeResult(None)

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def delete(self, obj: Any) -> None:
        self.deleted.append(obj)

    async def flush(self) -> None:
        self.flushed += 1

    async def commit(self) -> None:
        self.committed += 1

    async def rollback(self) -> None:
        self.rolled_back += 1

    async def refresh(self, obj: Any) -> None:
        # No-op stand-in for AsyncSession.refresh — the in-memory object is
        # already the canonical state in this fake.
        pass

    async def scalar(self, stmt: Any) -> Any:
        # Convenience equivalent to `(await execute(stmt)).scalar_one_or_none()`
        # — what `AsyncSession.scalar` does. Used by handlers that don't
        # need the full Result wrapper. Queue value behaves the same as for
        # other SELECT helpers.
        result = await self.execute(stmt)
        return result.scalar_one_or_none()


@pytest.fixture
def fake_db() -> FakeSession:
    return FakeSession()


@pytest.fixture
def fake_anthropic_stream(monkeypatch):
    """Factory fixture: install a cassette-driven Anthropic stream.

    Tests opt-in by calling `fake_anthropic_stream("chat_text_only")` (or
    any cassette name under `tests/fakes/cassettes/`). After the call,
    every `client.beta.prompt_caching.messages.stream(...)` invocation
    inside the test returns a `FakeAnthropicStream` regardless of args —
    so chat-agent tests can exercise the streaming, validation, and
    SSE-emission paths without hitting Anthropic.

    monkeypatch auto-rolls back after each test so cassettes don't leak.
    """
    from app import llm
    from tests.fakes.fake_anthropic import make_stream

    def _install(cassette_name: str):
        factory = make_stream(cassette_name)
        monkeypatch.setattr(
            llm.client.beta.prompt_caching.messages, "stream", factory
        )

    return _install


class MockTransportRegistry:
    """Record HTTP responses keyed by URL prefix.

    Each registered handler is called with the `httpx.Request` and returns
    an `httpx.Response`. Useful for the photo-redirect path: register the
    `/details/json` URL with a 200+JSON, the `/photo` URL with a 302+Location.
    """

    def __init__(self):
        self.handlers: list[tuple[str, Callable[[httpx.Request], httpx.Response]]] = []
        self.requests: list[httpx.Request] = []

    def on(
        self, url_prefix: str, handler: Callable[[httpx.Request], httpx.Response]
    ) -> None:
        self.handlers.append((url_prefix, handler))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        for prefix, handler in self.handlers:
            if url.startswith(prefix):
                return handler(request)
        return httpx.Response(404, text=f"unhandled URL in test: {url}")


@pytest.fixture
def mock_http(monkeypatch) -> MockTransportRegistry:
    """Patch the `_http_client` factories so any AsyncClient created inside
    the services routes through our handlers instead of the real network.

    Also collapses the shared retry-wait policy (`app.integrations._retry`)
    to `wait_none()` so tests that mock 5xx/429 still exercise the retry
    decision logic (count, predicate) without paying the 1-8s exponential
    backoff. Real attempt count is preserved — tests asserting "function
    returned None on error" still see N attempts in `registry.requests`.
    """
    registry = MockTransportRegistry()

    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(registry))

    from tenacity import wait_none

    from app.integrations import _retry, google_directions, google_places

    monkeypatch.setattr(google_places, "_http_client", factory)
    monkeypatch.setattr(google_directions, "_http_client", factory)
    monkeypatch.setattr(_retry.http_get_with_retry.retry, "wait", wait_none())
    return registry
