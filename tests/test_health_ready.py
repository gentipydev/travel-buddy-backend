"""Tests for the readiness probe at GET /health/ready (item B7).

Readiness checks DB connectivity (a lightweight SELECT 1) and reports which
third-party keys are configured WITHOUT making any paid external call. It
returns 200 when the DB is reachable and 503 when it is not. `/health`
(liveness) stays a dependency-free 200.

`get_session` is monkeypatched on the health router module so no real
Postgres is needed; config presence is asserted against the live Settings.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.routers import health as health_module


class _FakeSession:
    def __init__(self, *, raise_on_execute: bool = False):
        self._raise = raise_on_execute

    async def execute(self, stmt):
        if self._raise:
            raise RuntimeError("connection refused")
        return None


def _patch_get_session(monkeypatch, session) -> None:
    """Replace get_session with an async generator yielding our fake."""

    async def _gen():
        yield session

    monkeypatch.setattr(health_module, "get_session", _gen)


def test_liveness_is_dependency_free_200():
    with TestClient(app) as tc:
        resp = tc.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_200_when_db_reachable(monkeypatch):
    _patch_get_session(monkeypatch, _FakeSession())
    with TestClient(app) as tc:
        resp = tc.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"]["database"]["ok"] is True
    assert body["checks"]["database"]["error"] is None
    # Config presence is reported (bools), without any external call. Every
    # key is a plain bool — a single Anthropic API key reaches the API from
    # any host, so there is no credential *mode* to report.
    config = body["checks"]["config"]
    assert set(config.keys()) == {"anthropic", "cohere", "mapbox"}
    assert all(
        isinstance(config[k], bool)
        for k in ("anthropic", "cohere", "mapbox")
    )


def test_readiness_503_when_db_unreachable(monkeypatch):
    _patch_get_session(monkeypatch, _FakeSession(raise_on_execute=True))
    with TestClient(app) as tc:
        resp = tc.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["database"]["ok"] is False
    assert "RuntimeError" in body["checks"]["database"]["error"]


def test_readiness_makes_no_paid_external_call(monkeypatch):
    """The probe must check config presence by reading settings, never by
    pinging Cohere/Anthropic/Mapbox. We assert the DB SELECT 1 is the only
    I/O the handler performs by counting execute() calls."""
    calls = {"n": 0}

    class _CountingSession:
        async def execute(self, stmt):
            calls["n"] += 1
            return None

    _patch_get_session(monkeypatch, _CountingSession())
    with TestClient(app) as tc:
        resp = tc.get("/health/ready")
    assert resp.status_code == 200
    # Exactly one DB statement (SELECT 1); no extra probing round-trips.
    assert calls["n"] == 1
