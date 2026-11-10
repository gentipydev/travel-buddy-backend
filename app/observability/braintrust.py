"""Braintrust observability hooks.

Three public callables:
  - `bt_init()` — set up the project logger. Call once at app startup.
  - `bt_wrap_anthropic(client)` — wraps the Anthropic client so every
    `client.messages.*` call emits a child span under the active parent
    span. Passthrough when disabled or when the SDK isn't installed.
  - `chat_span(name)` — context manager yielding a Span (or a no-op
    stand-in when Braintrust is disabled) so callers use the same
    `span.log(...)` API in both modes.

All three no-op when `settings.braintrust_api_key` is empty or
`settings.braintrust_enabled` is False — so local dev, CI, and PRs
without the secret run unchanged. The `braintrust` package is also
optional: when missing, `_BT_AVAILABLE` flips to False and everything
no-ops with a one-time WARNING log (so a typo'd `pip install` is
visible, not silent).
"""
import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from app.config import settings

logger = logging.getLogger(__name__)

try:
    from braintrust import init_logger as _bt_init_logger
    from braintrust import start_span as _bt_start_span
    _BT_AVAILABLE = True
except ImportError:
    _bt_init_logger = None
    _bt_start_span = None
    _BT_AVAILABLE = False

# Imported separately: `wrap_anthropic` is a newer export than init_logger /
# start_span, and a version that lacks it must still get spans. Folding it into
# the block above would flip `_BT_AVAILABLE` off and disable observability
# wholesale over an optional wrapper.
_bt_wrap_anthropic: Callable[[Any], Any] | None
try:
    from braintrust import wrap_anthropic as _bt_wrap_anthropic
except ImportError:
    _bt_wrap_anthropic = None


def _enabled() -> bool:
    return bool(settings.braintrust_api_key) and settings.braintrust_enabled


def _warn_missing_once() -> None:
    if not getattr(_warn_missing_once, "_warned", False):
        logger.warning(
            "braintrust_enabled=true but `braintrust` package not installed; "
            "observability disabled. Run `pip install braintrust autoevals`."
        )
        _warn_missing_once._warned = True  # type: ignore[attr-defined]


def bt_init() -> None:
    """Initialize the project logger. Idempotent. No-op when disabled."""
    if not _enabled():
        return
    if not _BT_AVAILABLE:
        _warn_missing_once()
        return
    assert _bt_init_logger is not None
    _bt_init_logger(project=settings.braintrust_project)


def bt_wrap_anthropic(client):
    """Wrap the Anthropic client for auto-tracing.

    Returns the client unchanged when Braintrust is disabled or the SDK isn't
    installed, so this is safe to call unconditionally at import time. When
    enabled, `wrap_anthropic` instruments `client.messages.*` so each call
    emits a child span under whatever parent span is active on the request —
    the `chat_span` parent still logs token/latency metadata explicitly.

    Wrapping is best-effort: a Braintrust version whose wrapper doesn't
    understand this SDK must not take the whole app down at import.
    """
    if not _enabled() or not _BT_AVAILABLE or _bt_wrap_anthropic is None:
        return client
    try:
        return _bt_wrap_anthropic(client)
    except Exception:
        logger.warning(
            "braintrust: wrap_anthropic failed — continuing untraced",
            exc_info=True,
        )
        return client


class _NoopSpan:
    """Stand-in span used when Braintrust is disabled. All methods no-op."""

    def log(self, *args: Any, **kwargs: Any) -> None:
        pass

    def end(self, *args: Any, **kwargs: Any) -> None:
        pass


@contextmanager
def chat_span(name: str = "chat_stream") -> Iterator[Any]:
    """Open a Braintrust span for the chat-stream lifetime.

    Yields either a real `braintrust.Span` or `_NoopSpan` when the
    integration is disabled. The `with`-block's exit is sync (Braintrust
    buffers + flushes via a background daemon) — safe to leave inside a
    Starlette streaming generator that may be cancelled on client
    disconnect.
    """
    if not _enabled() or not _BT_AVAILABLE:
        if _enabled():  # enabled but SDK missing
            _warn_missing_once()
        yield _NoopSpan()
        return
    assert _bt_start_span is not None
    with _bt_start_span(name=name) as span:
        yield span
