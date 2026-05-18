"""Travel Buddy prompt registry.

Public API:
  - `build_system_prompt(preferences, trip_context=None)` — the /chat/stream
    system payload assembler (returns the same string the legacy
    `app/prompts.py:build_system_prompt` returned).
  - `get(name)` — look up a registered prompt by id (e.g. "chat.base",
    "conversation.title", "persona.gate_user"). Returns a `Prompt` with
    `.text`, `.version`, `.cache_eligible`, `.render(**fields)`.
  - `Prompt`, `PROMPTS`, `all_prompts` — the registry primitives.

See `registry.py` for the full prompt catalog and `builders.py` for the
system-prompt composer.
"""
from app.prompts.builders import build_system_payload, build_system_prompt
from app.prompts.registry import PROMPTS, Prompt, all_prompts, get

__all__ = [
    "PROMPTS",
    "Prompt",
    "all_prompts",
    "build_system_payload",
    "build_system_prompt",
    "get",
]
