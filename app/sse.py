def encode_sse_text(text: str) -> str:
    """Encode a text chunk as a single SSE `data:` event. Embedded newlines
    must each be prefixed with `data: ` so multi-line chunks remain one event.
    """
    safe = text.replace("\n", "\ndata: ")
    return f"data: {safe}\n\n"
