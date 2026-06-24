"""Named, versioned registry of every system prompt the backend sends to
Claude.

Single source of truth: each prompt body lives in a `.md` sibling file
(readable diffs, hashable for future eval pinning), and each prompt's
metadata (id, version, cache eligibility) lives in `PROMPTS` below.

Call sites look up prompts by id (`get("chat.base")`) rather than by
importing a global string. That way:

  - bumping a prompt version is a one-line edit in this file
  - scattered prompt definitions can't drift out of sync
  - evals (planned) can pin fixtures to a specific `(name, version)` pair
  - prompt caching (`app/llm/caching.py`) keys decisions off
    `cache_eligible` instead of hard-coding which prompts are cached

`.read_text(encoding="utf-8")` is called WITHOUT stripping — the `.md`
files preserve the leading/trailing whitespace the legacy constants
relied on, so `build_system_prompt` output is byte-identical to the
pre-migration string.
"""
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_PROMPTS_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prompt:
    """A registered prompt body plus the metadata callers need to decide
    how to send it to Claude.

    `cache_eligible` is read by `app/llm/caching.py` when assembling the
    Anthropic system payload: True blocks get `cache_control: ephemeral`,
    False blocks are sent uncached.

    `content_hash` is the first-12 hex of sha256(text) at module load; it
    lets the eval harness pin to a specific prompt body (not just version)
    so a `.md` edit that forgets to bump `version` is still detectable in
    regression diffs.
    """

    name: str
    version: str
    text: str
    cache_eligible: bool
    content_hash: str
    notes: str = ""

    def render(self, **fields: object) -> str:
        """Interpolate `{placeholder}` fields. No-op when the prompt has
        no placeholders."""
        return self.text.format(**fields)


def _load(rel_path: str) -> str:
    return (_PROMPTS_DIR / rel_path).read_text(encoding="utf-8")


def _compute_hash(text: str) -> str:
    """First 12 hex of sha256. Short enough to log, long enough to be
    collision-free across ~10^5 prompts."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _make(
    name: str,
    version: str,
    rel_path: str,
    *,
    cache_eligible: bool,
    notes: str = "",
) -> Prompt:
    text = _load(rel_path)
    return Prompt(
        name=name,
        version=version,
        text=text,
        cache_eligible=cache_eligible,
        notes=notes,
        content_hash=_compute_hash(text),
    )


PROMPTS: dict[str, Prompt] = {
    # --- /chat/stream system payload -----------------------------------
    # The chat base + addenda compose into ~3-4k tokens of stable
    # instructions. They are the largest contributor to cache hit rate;
    # `cache_eligible=True` makes them eligible for the cached prefix in
    # `assemble_chat_system`.
    "chat.base": _make("chat.base", "2026-06-24", "chat/base.md", cache_eligible=True),
    "chat.inference_addendum": _make(
        "chat.inference_addendum", "2026-05-18", "chat/inference_addendum.md",
        cache_eligible=True,
        notes="Appended when the user has no stored preferences.",
    ),
    "chat.override_reminder": _make(
        "chat.override_reminder", "2026-05-18", "chat/override_reminder.md",
        cache_eligible=True,
        notes="Appended when preferences exist; tells Claude to follow conversation over defaults.",
    ),
    "chat.itinerary_tool_addendum": _make(
        "chat.itinerary_tool_addendum", "2026-06-24", "chat/itinerary_tool_addendum.md",
        cache_eligible=True,
        notes="Itinerary delivery rule. 2026-05-28: allow single-day "
        "itineraries. 2026-06-24: restructured for concise, streaming "
        "delivery — one intro line of text + structured stops with "
        "one-sentence `notes`; no full-prose duplicate (was 'write the "
        "full plan in rich prose AND call emit_itinerary').",
    ),
    # Refine carries the trip JSON which varies every turn — never cache.
    "chat.refine_addendum": _make(
        "chat.refine_addendum", "2026-05-18", "chat/refine_addendum_template.md",
        cache_eligible=False,
        notes="Template; render with {trip_json}. Volatile suffix in cached payload.",
    ),
    # --- Conversation summarization & title ----------------------------
    # Both are short systems (~30-280 tokens) called once per N turns;
    # the cache write cost exceeds savings, so we don't mark them eligible.
    "conversation.title": _make(
        "conversation.title", "2026-05-18", "conversation/title.md", cache_eligible=False,
    ),
    "conversation.summarize": _make(
        "conversation.summarize", "2026-05-18", "conversation/summarize.md", cache_eligible=False,
    ),
    # --- Persona ---------------------------------------------------------
    # /preferences/persona endpoint (manual user request) + the async
    # YES/NO gate + the refresh task. All low-frequency; no caching.
    "persona.system": _make(
        "persona.system", "2026-05-18", "persona/system.md",
        cache_eligible=False,
        notes="Used by POST /preferences/persona to generate the initial persona.",
    ),
    "persona.gate_system": _make(
        "persona.gate_system", "2026-05-18", "persona/gate_system.md", cache_eligible=False,
    ),
    "persona.gate_user": _make(
        "persona.gate_user", "2026-05-18", "persona/gate_user.md",
        cache_eligible=False,
        notes="Template; render with {persona} and {summary}.",
    ),
    "persona.refresh_system": _make(
        "persona.refresh_system", "2026-05-18", "persona/refresh_system.md", cache_eligible=False,
    ),
    # --- Output safety ---------------------------------------------------
    # Runs once per turn post-assembly. Per-response variation means the
    # cache write cost exceeds savings; not eligible.
    "output_guard.system": _make(
        "output_guard.system", "2026-05-28", "output_guard/system.md",
        cache_eligible=False,
        notes="Binary SAFE/REVIEW classifier. Checks for system-prompt leaks, "
              "harmful content, and severe off-topic replies.",
    ),
    # --- Retrieval query rewrite ----------------------------------------
    # Optional Haiku pass in front of hybrid search: resolves follow-up
    # coreference and expands broad (county/region) queries into specific
    # search terms. Short system, runs at most once per retrieval turn; not
    # cache-eligible.
    "query_rewrite.system": _make(
        "query_rewrite.system", "2026-05-28", "query_rewrite/system.md",
        cache_eligible=False,
        notes="Rewrites a user question into a retrieval-optimized query "
              "(coreference resolution + broad-query expansion).",
    ),
}


def get(name: str) -> Prompt:
    """Look up a prompt by id. Raises KeyError with the registered set so
    typos fail loudly at startup rather than silently sending an empty
    system."""
    try:
        return PROMPTS[name]
    except KeyError:
        raise KeyError(
            f"Unknown prompt {name!r}; registered: {sorted(PROMPTS)}"
        )


def all_prompts() -> Iterable[Prompt]:
    """Iterate every registered prompt. Used by the startup self-check
    that asserts no `.md` file was lost from the Docker image."""
    return PROMPTS.values()


def validate_prompt_hashes() -> dict[str, str]:
    """Startup self-check: recompute each prompt's hash from its `text` and
    confirm it matches `content_hash`. The hash is computed at module load
    from the same `text`, so a mismatch here means the dataclass was
    constructed inconsistently (or someone mutated `text` after load). The
    check is cheap (10 short strings sha256'd) and the returned map of
    name → hash is logged so the value is visible in production logs for
    eval-harness pinning later.

    Raises `RuntimeError` on mismatch listing the offending prompts.
    """
    mismatched: list[str] = []
    snapshot: dict[str, str] = {}
    for prompt in all_prompts():
        recomputed = _compute_hash(prompt.text)
        snapshot[prompt.name] = prompt.content_hash
        if recomputed != prompt.content_hash:
            mismatched.append(
                f"{prompt.name}@{prompt.version}: "
                f"stored={prompt.content_hash} recomputed={recomputed}"
            )
    if mismatched:
        raise RuntimeError(
            "prompt content_hash mismatch on startup: " + "; ".join(mismatched)
        )
    logger.info(
        "prompts.hash_validated count=%d hashes=%s", len(snapshot), snapshot
    )
    return snapshot
