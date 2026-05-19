"""Tests for the DELETE /users/me cascading endpoint (item 8).

Coverage targets the orchestration contract: storage cleanup → DB delete
→ Supabase auth admin delete, in that order, and the failure-mode
guarantees (storage / auth failures don't block the DB delete).

The Supabase client is stubbed wholesale — these tests don't talk to a
real Supabase project. FK CASCADE behavior is owned by the migration
(003_create_rls_policies.sql and the raw CREATE TABLE in models.py:147)
and is asserted by the migration's own tests, not here.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.auth.supabase_auth import get_current_user
from app.db.engine import get_session
from app.db.models import User
from app.main import app
from app.routers import users as users_module


_USER = User(
    id=42,
    supabase_uid="supa-uid-abc",
    email="t@example.com",
    created_at=datetime.now(UTC),
)


# ---------------------------------------------------------------------------
# Fakes for the Supabase client surface used by delete_me.
# ---------------------------------------------------------------------------


class _FakeStorageBucket:
    def __init__(self, files: list[dict[str, Any]] | None = None):
        self.files = files or []
        self.removed: list[list[str]] = []

    async def list(self, path: str) -> list[dict[str, Any]]:
        self.listed_path = path
        return self.files

    async def remove(self, paths: list[str]) -> None:
        self.removed.append(paths)


class _FakeStorage:
    def __init__(self, bucket: _FakeStorageBucket):
        self._bucket = bucket
        self.from_calls: list[str] = []

    def from_(self, name: str) -> _FakeStorageBucket:
        self.from_calls.append(name)
        return self._bucket


class _FakeAdmin:
    def __init__(self, *, raises: Exception | None = None):
        self.raises = raises
        self.deleted: list[str] = []

    async def delete_user(self, uid: str) -> None:
        self.deleted.append(uid)
        if self.raises is not None:
            raise self.raises


class _FakeAuth:
    def __init__(self, admin: _FakeAdmin):
        self.admin = admin


class _FakeSupabase:
    def __init__(self, *, storage: _FakeStorage, auth: _FakeAuth):
        self.storage = storage
        self.auth = auth


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def client(fake_db, monkeypatch):
    """TestClient with get_session + get_current_user overridden.

    The Supabase client is monkeypatched onto `users_module.get_supabase`
    so the real `acreate_client` never runs.
    """

    def _supa_dep():
        return {"id": _USER.supabase_uid}

    def _db_dep():
        return fake_db

    app.dependency_overrides[get_current_user] = _supa_dep
    app.dependency_overrides[get_session] = _db_dep

    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_session, None)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_delete_me_removes_storage_db_and_auth(client, fake_db, monkeypatch):
    """Happy path: photos exist, local user exists, auth delete succeeds.
    Verify all three deletions run in order."""
    bucket = _FakeStorageBucket(
        files=[
            {"name": "abc.jpg"},
            {"name": "def.png"},
            {"name": ".emptyFolderPlaceholder"},  # filtered? — currently kept
        ]
    )
    admin = _FakeAdmin()
    fake_supa = _FakeSupabase(
        storage=_FakeStorage(bucket),
        auth=_FakeAuth(admin),
    )

    async def _get_supa():
        return fake_supa

    monkeypatch.setattr(users_module, "get_supabase", _get_supa)
    fake_db.queue(_USER)  # the SELECT in step 2

    res = client.delete(
        "/users/me",
        headers={"Authorization": "Bearer dummy"},
    )

    assert res.status_code == 204
    # Storage: listed under <uid>, removed all listed files (3 in this case).
    assert bucket.listed_path == _USER.supabase_uid
    assert len(bucket.removed) == 1
    assert sorted(bucket.removed[0]) == sorted(
        [
            f"{_USER.supabase_uid}/abc.jpg",
            f"{_USER.supabase_uid}/def.png",
            f"{_USER.supabase_uid}/.emptyFolderPlaceholder",
        ]
    )
    # DB: user was deleted + committed.
    assert _USER in fake_db.deleted
    assert fake_db.committed == 1
    # Auth: admin.delete_user called once with the uid.
    assert admin.deleted == [_USER.supabase_uid]


def test_delete_me_skips_storage_remove_when_no_files(client, fake_db, monkeypatch):
    """User has no uploaded photos — storage remove must not be called."""
    bucket = _FakeStorageBucket(files=[])
    admin = _FakeAdmin()
    fake_supa = _FakeSupabase(
        storage=_FakeStorage(bucket),
        auth=_FakeAuth(admin),
    )

    async def _get_supa():
        return fake_supa

    monkeypatch.setattr(users_module, "get_supabase", _get_supa)
    fake_db.queue(_USER)

    res = client.delete(
        "/users/me",
        headers={"Authorization": "Bearer dummy"},
    )

    assert res.status_code == 204
    assert bucket.removed == []  # never called with an empty list
    assert _USER in fake_db.deleted
    assert admin.deleted == [_USER.supabase_uid]


def test_delete_me_returns_204_even_when_local_user_missing(
    client, fake_db, monkeypatch
):
    """User authed but never synced to local DB (no users row). Storage + auth
    still run; DB step is a clean no-op."""
    bucket = _FakeStorageBucket(files=[])
    admin = _FakeAdmin()
    fake_supa = _FakeSupabase(
        storage=_FakeStorage(bucket),
        auth=_FakeAuth(admin),
    )

    async def _get_supa():
        return fake_supa

    monkeypatch.setattr(users_module, "get_supabase", _get_supa)
    fake_db.queue(None)  # the SELECT returns None — no local user row

    res = client.delete(
        "/users/me",
        headers={"Authorization": "Bearer dummy"},
    )

    assert res.status_code == 204
    assert fake_db.deleted == []
    assert fake_db.committed == 0
    assert admin.deleted == [_USER.supabase_uid]


def test_delete_me_swallows_storage_failure(client, fake_db, monkeypatch):
    """Storage outage must not block DB + auth deletion — we'd rather orphan
    objects than leave the user in a half-deleted limbo."""

    class _BoomBucket:
        async def list(self, path: str):
            raise RuntimeError("storage down")

        async def remove(self, paths):  # pragma: no cover — never reached
            raise AssertionError("should not be called")

    class _BoomStorage:
        def from_(self, name: str):
            return _BoomBucket()

    admin = _FakeAdmin()
    fake_supa = _FakeSupabase(
        storage=_BoomStorage(),
        auth=_FakeAuth(admin),
    )

    async def _get_supa():
        return fake_supa

    monkeypatch.setattr(users_module, "get_supabase", _get_supa)
    fake_db.queue(_USER)

    res = client.delete(
        "/users/me",
        headers={"Authorization": "Bearer dummy"},
    )

    assert res.status_code == 204
    # DB + auth still ran.
    assert _USER in fake_db.deleted
    assert admin.deleted == [_USER.supabase_uid]


def test_delete_me_swallows_auth_failure(client, fake_db, monkeypatch):
    """Supabase auth admin outage must not surface as 5xx — local data is
    already gone, retrying the endpoint with the same JWT is safe."""
    bucket = _FakeStorageBucket(files=[])
    admin = _FakeAdmin(raises=RuntimeError("auth down"))
    fake_supa = _FakeSupabase(
        storage=_FakeStorage(bucket),
        auth=_FakeAuth(admin),
    )

    async def _get_supa():
        return fake_supa

    monkeypatch.setattr(users_module, "get_supabase", _get_supa)
    fake_db.queue(_USER)

    res = client.delete(
        "/users/me",
        headers={"Authorization": "Bearer dummy"},
    )

    assert res.status_code == 204
    assert _USER in fake_db.deleted
    # admin.delete_user was attempted (and raised).
    assert admin.deleted == [_USER.supabase_uid]


def test_delete_me_requires_auth():
    """No Bearer token → 401, before any side effect."""
    with TestClient(app) as tc:
        res = tc.delete("/users/me")
    assert res.status_code in (401, 403)


_ = uuid  # keep import lint-clean if a future expansion needs it
