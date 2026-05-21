"""Prompt content-hash invariant tests.

The prompt registry computes `content_hash = sha256(text)[:12]` at module
load (`app/prompts/registry.py:_make`). `validate_prompt_hashes` runs as
part of the FastAPI lifespan and asserts the registry's stored hash matches
the recomputed value — the catch is a `.md` edit that bypasses the hash
column (e.g. a manual mutation in tests).

These tests pin the contract:
1. Clean tree (every prompt's stored hash matches its text) → no raise.
2. Mutated `text` field with stale `content_hash` → `RuntimeError` listing
   the offending prompt.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from app.prompts import registry as prompt_registry


def test_all_prompts_load_with_valid_content_hash():
    """Every prompt in PROMPTS has a non-empty text and a stored hash that
    matches the recomputed sha256 of that text. This is the property the
    eval harness will pin to, so it can't silently break."""
    for prompt in prompt_registry.all_prompts():
        assert prompt.text, f"{prompt.name} has empty text — .md missing?"
        assert len(prompt.content_hash) == 12, (
            f"{prompt.name} hash {prompt.content_hash!r} is not 12 hex chars"
        )
        recomputed = prompt_registry._compute_hash(prompt.text)
        assert recomputed == prompt.content_hash, (
            f"{prompt.name}: stored {prompt.content_hash} != recomputed {recomputed}"
        )


def test_validate_prompt_hashes_returns_snapshot():
    """Happy path: returns a {name: hash} dict covering every registered
    prompt; logged at startup so the values are visible for eval pinning."""
    snapshot = prompt_registry.validate_prompt_hashes()
    assert set(snapshot) == set(prompt_registry.PROMPTS)
    for name, hash_ in snapshot.items():
        assert prompt_registry.PROMPTS[name].content_hash == hash_


def test_validate_prompt_hashes_raises_on_text_drift(monkeypatch):
    """If a `Prompt.text` is mutated without bumping `content_hash`, the
    startup validator raises with the offending prompt name. This is the
    safety net for the eval harness: a `.md` edit that forgets to bump
    `version` will still mismatch the hash, surfacing the drift loudly."""
    # Snapshot the original dict so the monkeypatch can restore it; Prompt
    # is frozen, so we replace the entry with a dataclasses.replace clone
    # carrying mutated `text` but the original `content_hash`.
    name = "chat.base"
    original = prompt_registry.PROMPTS[name]
    drifted = replace(original, text=original.text + "\nDRIFT")
    new_prompts = {**prompt_registry.PROMPTS, name: drifted}
    monkeypatch.setattr(prompt_registry, "PROMPTS", new_prompts)

    with pytest.raises(RuntimeError, match=r"prompt content_hash mismatch.*chat\.base"):
        prompt_registry.validate_prompt_hashes()


def test_make_helper_computes_hash_from_text(tmp_path, monkeypatch):
    """The `_make` factory is what guarantees every prompt has a hash in
    sync with its text. Lock that property: a `_make` call with the same
    text always produces the same hash (deterministic)."""
    p1 = prompt_registry._make("test.a", "v1", "chat/base.md", cache_eligible=True)
    p2 = prompt_registry._make("test.b", "v1", "chat/base.md", cache_eligible=True)
    # Same text → same hash, even with different names/versions.
    assert p1.content_hash == p2.content_hash
    # Hash matches recomputation.
    assert p1.content_hash == prompt_registry._compute_hash(p1.text)
