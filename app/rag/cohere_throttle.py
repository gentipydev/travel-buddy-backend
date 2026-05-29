"""Per-process rate limiter + 429 counter for Cohere calls.

Cohere trial keys allow only 10 requests/minute across embed + rerank
combined. The RAG layer is fail-soft (embed failure → lexical-only, rerank
failure → passthrough top_n), so an unguarded burst that trips the 429
ceiling does not error — it silently degrades retrieval quality with no
operator-visible signal. This module closes both gaps:

  - `cohere_limiter` is a process-wide async token bucket. `embed` and
    `rerank` both `await cohere_limiter.acquire()` before their network
    call, so a burst is paced under the configured ceiling instead of
    being rejected. The bucket refills continuously
    (`per_minute / window_seconds` tokens/second).

  - `note_rate_limited()` increments a monotonic counter every time a
    Cohere call still comes back 429 (e.g. another process sharing the
    key). `rate_limited_count()` exposes it so the spike is visible in
    logs / a future metrics scrape instead of being swallowed by the
    fail-soft fallback.

The limiter is intentionally per-process (no Redis): it caps THIS worker's
burst, which is the dominant source of trial-key 429s in single-instance
dev / small deploys. Multi-worker coordination is out of scope (tracked
with the slowapi→Redis migration).
"""
from __future__ import annotations

import asyncio
import time

from app.config import settings


class AsyncTokenBucket:
    """Continuous-refill token bucket guarding a shared rate limit.

    `capacity` tokens accrue over `window_seconds` (so the steady-state
    rate is `capacity / window_seconds` per second). `acquire()` blocks
    until a token is available, then consumes it. Thread-safety is not
    needed — a single asyncio event loop serializes access — but an
    `asyncio.Lock` guards the await point so concurrent coroutines don't
    both read a stale token count.
    """

    def __init__(self, *, capacity: int, window_seconds: float):
        # capacity<=0 disables limiting (treated as "no ceiling").
        self.capacity = capacity
        self.window_seconds = window_seconds
        self._refill_rate = (
            capacity / window_seconds if window_seconds > 0 else 0.0
        )
        self._tokens = float(capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(
                float(self.capacity), self._tokens + elapsed * self._refill_rate
            )
            self._updated = now

    async def acquire(self) -> None:
        """Block until one token is available, then consume it.

        No-op when capacity<=0 (limiting disabled). Sleeps the minimal time
        for the next token to refill rather than busy-waiting.
        """
        if self.capacity <= 0 or self._refill_rate <= 0:
            return
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                needed = 1.0 - self._tokens
                await asyncio.sleep(needed / self._refill_rate)


cohere_limiter = AsyncTokenBucket(
    capacity=settings.cohere_rate_limit_per_minute,
    window_seconds=settings.cohere_rate_limit_window_seconds,
)


# Monotonic 429 counter — incremented on every Cohere rate-limit response so
# a spike is visible even though the RAG layer fails soft on the error.
_rate_limited_count = 0


def note_rate_limited() -> None:
    """Record one Cohere 429. Cheap, never raises — telemetry only."""
    global _rate_limited_count
    _rate_limited_count += 1


def rate_limited_count() -> int:
    """Total Cohere 429s observed this process since start."""
    return _rate_limited_count


def reset_rate_limited_count() -> None:
    """Reset the 429 counter. Test-only — production never calls this."""
    global _rate_limited_count
    _rate_limited_count = 0
