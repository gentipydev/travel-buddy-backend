"""Shared retry policy for external API calls.

`http_get_with_retry` wraps `httpx.AsyncClient.get` with the project's
standard retry policy:
    - 3 attempts total
    - exponential jitter, 1 s → 8 s
    - retry on `TimeoutException` / `RequestError` (network blip)
    - retry on HTTP 429 (rate-limited by the vendor)
    - retry on HTTP 5xx (server-side capacity)

Explicitly NOT retried:
    - 4xx other than 429 — the caller's request is malformed; retrying
      will produce the same error and just slow things down.
    - 3xx — the caller has set `follow_redirects` deliberately (e.g. the
      Google Places photo endpoint returns 302 by design) and is
      handling them in-line.

Tests inject `httpx.MockTransport` via the `mock_http` conftest fixture,
which monkeypatches each integration's `_http_client` factory. The retry
decorator still fires in tests, but `MockTransport` returns responses
synchronously so retries on a successful mock add no real latency. Tests
that exercise 5xx / timeout paths will see N attempts in the request log
— assertions on request counts should account for this.
"""
import logging

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)


def _is_retryable_status(resp: httpx.Response) -> bool:
    return resp.status_code == 429 or 500 <= resp.status_code < 600


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=8),
    retry=(
        retry_if_exception_type((httpx.TimeoutException, httpx.RequestError))
        | retry_if_result(_is_retryable_status)
    ),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def http_get_with_retry(
    client: httpx.AsyncClient, url: str, **kwargs
) -> httpx.Response:
    """`client.get(url, **kwargs)` with the shared external-API retry policy."""
    return await client.get(url, **kwargs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=8),
    retry=(
        retry_if_exception_type((httpx.TimeoutException, httpx.RequestError))
        | retry_if_result(_is_retryable_status)
    ),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def http_post_with_retry(
    client: httpx.AsyncClient, url: str, **kwargs
) -> httpx.Response:
    """`client.post(url, **kwargs)` with the shared external-API retry policy.

    Used for the Places API (New) `places:searchText` endpoint, which is a
    POST (the legacy Text Search was a GET). Same policy as the GET helper:
    retry on 429/5xx and network blips, never on other 4xx.
    """
    return await client.post(url, **kwargs)
