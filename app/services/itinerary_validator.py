"""Validate Claude's emit_itinerary tool output before it reaches the client.

Claude's tool calls are best-effort, not schema-enforced. The Anthropic API
will not reject a malformed tool_use block — it just returns whatever the
model produced. We validate against the same Pydantic shape the persistence
layer uses (TripCreate) so the SSE contract and the DB contract cannot drift.
"""

from typing import Any

from pydantic import ValidationError

from app.schemas import TripCreate


class ItineraryValidationResult:
    """Outcome of validating a tool_use block.

    `payload` is the validated dict ready for SSE forwarding. `error` is a
    short human-readable summary suitable for an error SSE event and for
    telemetry. Exactly one of (payload, error) is set.
    """

    __slots__ = ("payload", "error", "raw")

    def __init__(
        self,
        payload: dict | None,
        error: str | None,
        raw: Any,
    ):
        self.payload = payload
        self.error = error
        self.raw = raw

    @property
    def ok(self) -> bool:
        return self.payload is not None


def validate_itinerary(raw_input: Any) -> ItineraryValidationResult:
    """Validate a tool_use block's `.input` against TripCreate.

    Returns a result object — never raises. Callers MUST check `.ok` before
    forwarding.
    """
    # Shape check first. Pydantic would reject non-dict inputs, but the error
    # message is noisy — handle the common "Claude returned a list/None" case
    # explicitly so telemetry stays readable.
    if not isinstance(raw_input, dict):
        return ItineraryValidationResult(
            payload=None,
            error=f"itinerary payload was {type(raw_input).__name__}, expected object",
            raw=None,
        )

    try:
        validated = TripCreate.model_validate(raw_input)
    except ValidationError as e:
        # Pydantic errors are verbose. For the client we want a one-liner; for
        # telemetry we want enough to triage. First error rendered as
        # `loc: msg` is the right balance.
        errors = e.errors()
        first = errors[0] if errors else None
        if first:
            loc = ".".join(str(p) for p in first.get("loc", ()))
            msg = first.get("msg", "validation failed")
            summary = f"{loc}: {msg}" if loc else msg
        else:
            summary = "validation failed"
        return ItineraryValidationResult(
            payload=None,
            error=summary,
            raw=raw_input,
        )

    # Re-serialize through Pydantic to normalize types (date → ISO string,
    # unset optionals dropped). What the client receives is identical to
    # what we log, which is what we serialize for tests — one source of truth.
    return ItineraryValidationResult(
        payload=validated.model_dump(mode="json", exclude_none=True),
        error=None,
        raw=raw_input,
    )
