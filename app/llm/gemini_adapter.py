"""Adapters between the app's Anthropic-shaped data and the Gemini SDK.

The rest of the codebase still speaks the original wire shapes — message
dicts with `{"role": "user"/"assistant", "content": str | [blocks]}`, tool
definitions with a JSON-Schema `input_schema`, and the four-field token
accounting (`input/output/cache_creation/cache_read`). Converting at this
single boundary keeps the chat agent, the conversation history builder, and
the tool definitions provider-neutral.

Three concerns:
  - `to_gemini_contents` — message dicts → `list[types.Content]`. Image
    blocks carry a URL (Anthropic fetched them server-side); Gemini does
    NOT fetch arbitrary URLs, so we download the bytes and inline them.
  - `to_gemini_tools` — `[{name, description, input_schema}]` →
    `[types.Tool(function_declarations=[...])]`, recursively converting the
    JSON Schema to a `types.Schema`.
  - `extract_usage` — `usage_metadata` → `(input, output, cache_creation,
    cache_read)` with Anthropic-compatible semantics (input EXCLUDES cached
    tokens; Gemini has no separate cache-creation billing → always 0).
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from google.genai import types

logger = logging.getLogger(__name__)

# Bounded so a slow/broken image URL can't stall a chat turn. Images that
# fail to fetch are dropped (the text still goes through) rather than raising.
_IMAGE_FETCH_TIMEOUT_SECONDS = 8.0

_TYPE_MAP: dict[str, types.Type] = {
    "object": types.Type.OBJECT,
    "array": types.Type.ARRAY,
    "string": types.Type.STRING,
    "integer": types.Type.INTEGER,
    "number": types.Type.NUMBER,
    "boolean": types.Type.BOOLEAN,
    "null": types.Type.NULL,
}


# ---------------------------------------------------------------------------
# Tool schemas.
# ---------------------------------------------------------------------------


def _convert_schema(node: Any) -> types.Schema | None:
    """Recursively convert a JSON-Schema dict to a `types.Schema`.

    Maps the subset our tool definitions use (type, description, enum,
    properties, required, items, minItems, minimum). JSON Schema keywords
    Gemini doesn't model (e.g. `additionalProperties`) are simply dropped.
    """
    if not isinstance(node, dict):
        return None
    kwargs: dict[str, Any] = {}
    t = node.get("type")
    if isinstance(t, str):
        kwargs["type"] = _TYPE_MAP.get(t, types.Type.TYPE_UNSPECIFIED)
    if "description" in node:
        kwargs["description"] = node["description"]
    if "enum" in node:
        kwargs["enum"] = node["enum"]
    if isinstance(node.get("properties"), dict):
        kwargs["properties"] = {
            k: _convert_schema(v) for k, v in node["properties"].items()
        }
    if "required" in node:
        kwargs["required"] = node["required"]
    if "items" in node:
        kwargs["items"] = _convert_schema(node["items"])
    if "minItems" in node:
        kwargs["min_items"] = node["minItems"]
    if "minimum" in node:
        kwargs["minimum"] = node["minimum"]
    return types.Schema(**kwargs)


def to_gemini_tools(tools: list[dict]) -> list[types.Tool]:
    """Translate Anthropic-style tool dicts into a single Gemini `Tool`
    carrying all function declarations.

    Each input dict is `{"name", "description", "input_schema": {...}}`.
    Gemini groups declarations under one `Tool`; the model picks among them.
    Returns an empty list for no tools (the SDK accepts `tools=None`/`[]`).
    """
    declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters=_convert_schema(t.get("input_schema")),
        )
        for t in tools
    ]
    if not declarations:
        return []
    return [types.Tool(function_declarations=declarations)]


# ---------------------------------------------------------------------------
# Messages (contents).
# ---------------------------------------------------------------------------


async def _image_part(url: str | None) -> types.Part | None:
    """Download an image URL and return an inline-data Part, or None on any
    failure (so a broken image never breaks the turn)."""
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=_IMAGE_FETCH_TIMEOUT_SECONDS) as http:
            resp = await http.get(url)
            resp.raise_for_status()
        mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        return types.Part.from_bytes(data=resp.content, mime_type=mime)
    except Exception:
        logger.warning("gemini_adapter: image fetch failed for %s", url, exc_info=True)
        return None


async def _content_parts(content: Any) -> list[types.Part]:
    """Convert one message's `content` (str or list of Anthropic blocks) into
    Gemini Parts. Image blocks are fetched + inlined; text blocks pass through.
    """
    if isinstance(content, str):
        return [types.Part(text=content)] if content else []
    parts: list[types.Part] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and block.get("text"):
                parts.append(types.Part(text=block["text"]))
            elif btype == "image":
                url = (block.get("source") or {}).get("url")
                part = await _image_part(url)
                if part is not None:
                    parts.append(part)
    return parts


async def to_gemini_contents(messages: list[dict]) -> list[types.Content]:
    """Translate Anthropic-format `messages` into Gemini `contents`.

    Role mapping: `assistant` → `model`, everything else → `user`. Messages
    that yield no parts (e.g. an image-only turn whose image failed to fetch)
    are skipped so we never send an empty `Content`.
    """
    contents: list[types.Content] = []
    for m in messages:
        role = "model" if m.get("role") == "assistant" else "user"
        parts = await _content_parts(m.get("content"))
        if parts:
            contents.append(types.Content(role=role, parts=parts))
    return contents


# ---------------------------------------------------------------------------
# Usage / token accounting.
# ---------------------------------------------------------------------------


def extract_usage(usage_metadata: Any) -> tuple[int, int, int, int]:
    """Map Gemini `usage_metadata` to the app's four-field accounting:
    `(input_tokens, output_tokens, cache_creation_input_tokens,
    cache_read_input_tokens)`.

    Semantics mirror the previous Anthropic shape so `compute_cost` and the
    `usage_daily` ledger keep working unchanged:
      - Gemini's `prompt_token_count` INCLUDES cached tokens, so input =
        prompt - cached (the non-cached, full-rate portion).
      - output = candidate tokens + reasoning ("thoughts") tokens, both
        billed at the output rate on a thinking model like Gemini 3 Flash.
      - cache_read = cached_content_token_count.
      - cache_creation has no separate Gemini billing (implicit caching), so
        it is always 0.
    Returns all zeros when usage is missing — telemetry must never crash.
    """
    if usage_metadata is None:
        return (0, 0, 0, 0)
    prompt = getattr(usage_metadata, "prompt_token_count", None) or 0
    candidates = getattr(usage_metadata, "candidates_token_count", None) or 0
    thoughts = getattr(usage_metadata, "thoughts_token_count", None) or 0
    cached = getattr(usage_metadata, "cached_content_token_count", None) or 0
    input_tokens = max(0, prompt - cached)
    output_tokens = candidates + thoughts
    return (input_tokens, output_tokens, 0, cached)
