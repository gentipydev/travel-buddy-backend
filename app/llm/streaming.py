"""SSE (Server-Sent Events) encoding helpers.

The /chat/stream endpoint emits incremental text chunks from Claude's
stream as individual SSE `data:` frames. Multi-line chunks need every
newline escaped so they stay inside ONE SSE event — the Flutter parser
treats a blank line as the end-of-event sentinel, so an un-escaped `\\n\\n`
in the chunk body would split it into two malformed events.
"""


def encode_sse_text(text: str) -> str:
    """Encode `text` as a single SSE `data:` event.

    Replaces every newline with `\\ndata: ` so multi-line chunks remain a
    single event, then appends the `\\n\\n` event terminator. Empty `text`
    still produces a valid (empty) event.
    """
    return "data: " + text.replace("\n", "\ndata: ") + "\n\n"
