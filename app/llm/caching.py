"""System-prompt assembly for the chat path.

Anthropic took the system prompt as a list of text blocks and let us mark a
cacheable "breakpoint" with `cache_control: ephemeral` on the stable prefix.
Gemini has no per-request breakpoint API — it caches implicitly (automatic
on 2.5+/3 models) and takes the system prompt as a single
`config.system_instruction` string. So assembly collapses to a string join.

The split into `cached_prefix` (stable base prompt + addenda + preferences)
and `volatile_suffix` (per-turn refine addendum + retrieved RAG context) is
still meaningful upstream: `app/llm/token_budget.py` only ever trims the
volatile suffix, and implicit caching keys off the longest stable prefix —
so keeping the stable text first still maximizes cache hits.
"""


def assemble_chat_system(
    *,
    cached_prefix: str,
    volatile_suffix: str = "",
) -> str:
    """Join the stable prefix and the per-turn volatile suffix into the single
    `system_instruction` string Gemini takes.

    The stable prefix is placed first so implicit caching can reuse it across
    turns. Pass an empty `volatile_suffix` to send just the prefix.
    """
    if volatile_suffix:
        return f"{cached_prefix}\n\n{volatile_suffix}"
    return cached_prefix
