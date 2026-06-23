"""Pins for `app/rate_limit.py:user_or_ip_key`.

The key function must group requests by the VERIFIED identity
(`request.state.verified_sub`, set by the auth dependencies after JWT
signature verification) — never by the raw Authorization header. Keying
on the unverified token would let an attacker mint a fresh limiter
bucket per request with garbage tokens, and legit Supabase tokens
rotate hourly, silently resetting daily quotas.
"""
from starlette.requests import Request

from app.rate_limit import user_or_ip_key


def _request(
    *,
    headers: dict[str, str] | None = None,
    sub: str | None = None,
    client_ip: str = "203.0.113.7",
) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/chat/stream",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
        "query_string": b"",
        "client": (client_ip, 4242),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    request = Request(scope)
    if sub is not None:
        request.state.verified_sub = sub
    return request


def test_verified_sub_wins_over_spoofed_bearer_tokens():
    """Different (garbage) bearer tokens must NOT mint different buckets:
    the key comes from the verified sub, not the header."""
    key_a = user_or_ip_key(
        _request(headers={"Authorization": "Bearer junk-1"}, sub="uid-123")
    )
    key_b = user_or_ip_key(
        _request(headers={"Authorization": "Bearer junk-2"}, sub="uid-123")
    )
    assert key_a == key_b == "user:uid-123"


def test_unverified_bearer_falls_back_to_ip():
    """A token that never passed verification (no verified_sub on
    request.state) buckets by IP — spoofed tokens share the IP bucket."""
    key = user_or_ip_key(
        _request(headers={"Authorization": "Bearer forged-garbage"})
    )
    assert key == "ip:203.0.113.7"


def test_anonymous_request_buckets_by_ip():
    assert user_or_ip_key(_request()) == "ip:203.0.113.7"


def test_token_rotation_keeps_same_bucket():
    """Same user, rotated (different) real token → same bucket, so the
    per-day quota survives Supabase's hourly token refresh."""
    before = user_or_ip_key(
        _request(headers={"Authorization": "Bearer old-token"}, sub="uid-9")
    )
    after = user_or_ip_key(
        _request(headers={"Authorization": "Bearer new-token"}, sub="uid-9")
    )
    assert before == after == "user:uid-9"
