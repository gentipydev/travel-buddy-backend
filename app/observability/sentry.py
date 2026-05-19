"""Sentry init + PII scrubber.

Production error visibility. The DSN is read from `settings.sentry_dsn`;
when empty the `init_sentry()` call is a no-op so local dev / CI / PRs
without the secret run unchanged.

`_scrub_pii(event, hint)` walks the event dict and redacts user-content
fields before any payload leaves the process — chat messages, persona
text, user emails, and trip stop names. Sentry never sees raw user input.
"""
from __future__ import annotations

import logging
from typing import Any

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration


logger = logging.getLogger(__name__)


# Request-body keys that may carry user-generated content. Matched against
# `event['request']['data']` and request JSON form fields. Add new keys as
# new endpoints introduce them.
_PII_REQUEST_KEYS: tuple[str, ...] = (
    "message",        # /chat/stream user message
    "messages",       # historical chat thread payload
    "content",        # generic message content
    "persona",        # /preferences/* persona blob
    "persona_text",
    "personality",
    "summary",        # trip summary
    "title",          # trip / day / stop title — may include place names
    "name",           # stop name
    "base_city",      # trip day base city
    "notes",          # stop notes
)

_REDACTED = "[redacted]"


def _scrub_value(value: Any) -> Any:
    """Recursively redact every string leaf in a value tree. Used when the
    parent key was already flagged PII — the entire subtree is suspect."""
    if isinstance(value, dict):
        return {k: _scrub_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(v) for v in value]
    if isinstance(value, str):
        return _REDACTED
    return value


def _scrub_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Walk a dict and redact known-PII keys. Sub-dicts/lists with non-PII
    keys recurse so deeply-nested PII keys are still caught."""
    scrubbed: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(k, str) and k in _PII_REQUEST_KEYS:
            scrubbed[k] = _scrub_value(v)
        elif isinstance(v, dict):
            scrubbed[k] = _scrub_dict(v)
        elif isinstance(v, list):
            scrubbed[k] = [_scrub_dict(item) if isinstance(item, dict) else item for item in v]
        else:
            scrubbed[k] = v
    return scrubbed


def _scrub_pii(event: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any] | None:
    """Sentry `before_send` hook. Strips PII from outgoing events.

    Never raise — a crash here means errors stop reaching Sentry, which
    defeats the purpose. Swallow + log.
    """
    try:
        # 1. Request body (chat content, persona, trip fields).
        req = event.get("request")
        if isinstance(req, dict):
            data = req.get("data")
            if isinstance(data, dict):
                req["data"] = _scrub_dict(data)
            elif isinstance(data, str):
                # FastAPI sends the raw body string when JSON parsing failed;
                # safer to drop than to send.
                req["data"] = _REDACTED

        # 2. User email (id + uid are non-PII per our privacy model).
        user = event.get("user")
        if isinstance(user, dict) and "email" in user:
            user["email"] = _REDACTED

        # 3. Breadcrumbs may capture sql/http params — redact known-PII keys.
        for crumb in event.get("breadcrumbs", {}).get("values", []) or []:
            crumb_data = crumb.get("data") if isinstance(crumb, dict) else None
            if isinstance(crumb_data, dict):
                crumb["data"] = _scrub_dict(crumb_data)

        # 4. Exception variables (Sentry's `vars` capture) frequently include
        # the message arg — redact any string value under a PII key.
        for ex in event.get("exception", {}).get("values", []) or []:
            stacktrace = ex.get("stacktrace") if isinstance(ex, dict) else None
            if isinstance(stacktrace, dict):
                for frame in stacktrace.get("frames", []) or []:
                    if isinstance(frame, dict) and isinstance(frame.get("vars"), dict):
                        frame["vars"] = _scrub_dict(frame["vars"])

    except Exception:
        logger.exception("sentry_scrub_pii_failed event=%s", event.get("event_id"))
    return event


def init_sentry(
    *,
    dsn: str,
    environment: str = "development",
    traces_sample_rate: float = 0.1,
) -> None:
    """Initialize Sentry. No-op when `dsn` is empty."""
    if not dsn:
        logger.info("sentry=disabled reason=no_dsn")
        return
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        integrations=[FastApiIntegration()],
        before_send=_scrub_pii,
        # `send_default_pii=False` is the SDK default, but make it explicit so
        # a future flip doesn't silently start leaking auth headers / cookies.
        send_default_pii=False,
    )
    logger.info(
        "sentry=enabled environment=%s traces_sample_rate=%s",
        environment,
        traces_sample_rate,
    )
