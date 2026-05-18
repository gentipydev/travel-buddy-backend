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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_PROMPTS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Prompt:
    """A registered prompt body plus the metadata callers need to decide
    how to send it to Claude.

    `cache_eligible` is read by `app/llm/caching.py` when assembling the
    Anthropic system payload: True blocks get `cache_control: ephemeral`,
    False blocks are sent uncached.
    """

    name: str
    version: str
    text: str
    cache_eligible: bool
    notes: str = ""

    def render(self, **fields: object) -> str:
        """Interpolate `{placeholder}` fields. No-op when the prompt has
        no placeholders."""
        return self.text.format(**fields)


def _load(rel_path: str) -> str:
    return (_PROMPTS_DIR / rel_path).read_text(encoding="utf-8")


PROMPTS: dict[str, Prompt] = {
    # --- /chat/stream system payload -----------------------------------
    # The chat base + addenda compose into ~3-4k tokens of stable
    # instructions. They are the largest contributor to cache hit rate;
    # `cache_eligible=True` makes them eligible for the cached prefix in
    # `assemble_chat_system`.
    "chat.base": Prompt(
        name="chat.base",
        version="2026-05-18",
        text=_load("chat/base.md"),
        cache_eligible=True,
    ),
    "chat.inference_addendum": Prompt(
        name="chat.inference_addendum",
        version="2026-05-18",
        text=_load("chat/inference_addendum.md"),
        cache_eligible=True,
        notes="Appended when the user has no stored preferences.",
    ),
    "chat.override_reminder": Prompt(
        name="chat.override_reminder",
        version="2026-05-18",
        text=_load("chat/override_reminder.md"),
        cache_eligible=True,
        notes="Appended when preferences exist; tells Claude to follow conversation over defaults.",
    ),
    "chat.itinerary_tool_addendum": Prompt(
        name="chat.itinerary_tool_addendum",
        version="2026-05-18",
        text=_load("chat/itinerary_tool_addendum.md"),
        cache_eligible=True,
        notes="The 'write the full plan in prose AND call emit_itinerary' rule.",
    ),
    # Refine carries the trip JSON which varies every turn — never cache.
    "chat.refine_addendum": Prompt(
        name="chat.refine_addendum",
        version="2026-05-18",
        text=_load("chat/refine_addendum_template.md"),
        cache_eligible=False,
        notes="Template; render with {trip_json}. Volatile suffix in cached payload.",
    ),
    # --- Conversation summarization & title ----------------------------
    # Both are short systems (~30-280 tokens) called once per N turns;
    # the cache write cost exceeds savings, so we don't mark them eligible.
    "conversation.title": Prompt(
        name="conversation.title",
        version="2026-05-18",
        text=_load("conversation/title.md"),
        cache_eligible=False,
    ),
    "conversation.summarize": Prompt(
        name="conversation.summarize",
        version="2026-05-18",
        text=_load("conversation/summarize.md"),
        cache_eligible=False,
    ),
    # --- Persona ---------------------------------------------------------
    # /preferences/persona endpoint (manual user request) + the async
    # YES/NO gate + the refresh task. All low-frequency; no caching.
    "persona.system": Prompt(
        name="persona.system",
        version="2026-05-18",
        text=_load("persona/system.md"),
        cache_eligible=False,
        notes="Used by POST /preferences/persona to generate the initial persona.",
    ),
    "persona.gate_system": Prompt(
        name="persona.gate_system",
        version="2026-05-18",
        text=_load("persona/gate_system.md"),
        cache_eligible=False,
    ),
    "persona.gate_user": Prompt(
        name="persona.gate_user",
        version="2026-05-18",
        text=_load("persona/gate_user.md"),
        cache_eligible=False,
        notes="Template; render with {persona} and {summary}.",
    ),
    "persona.refresh_system": Prompt(
        name="persona.refresh_system",
        version="2026-05-18",
        text=_load("persona/refresh_system.md"),
        cache_eligible=False,
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
