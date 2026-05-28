"""Contextual retrieval: a metadata header prepended to each chunk for
indexing only.

This adapts Anthropic's *Contextual Retrieval* to structured place data. Their
recipe uses an LLM to situate a fragment within its parent document; here each
chunk already *is* the whole place description, and the context that's actually
missing from the prose is structured metadata we own — region/county, category,
and tags (e.g. "Vlorë County", "beach", "unesco", "skanderbeg"). A deterministic
template is therefore the right tool: reproducible, free, and no per-chunk model
call.

The header is stored in `Chunk.context` (separate from `content`) so it feeds
both retrieval arms without polluting the model-facing text:

  - **vector** — `scripts/embed_chunks.py` embeds `embedding_input(...)`.
  - **lexical** — the generated `Chunk.search_text` column indexes
    `context || content` (migration 017).
  - **model context** — `app/rag/retrieval.py:retrieve_context` joins `content`
    only, so the header anchors retrieval but never reaches the prompt.
"""
from collections.abc import Sequence


def context_header(
    name: str,
    city: str | None = None,
    region: str | None = None,
    category: str | None = None,
    tags: Sequence[str] | None = None,
) -> str:
    """Build a one-line metadata header for a place chunk, e.g.
    `"Berat — city in Berat, Berat County, Albania. Themes: unesco, ottoman."`

    Empty/whitespace fields are omitted. Returns "" when `name` is empty —
    there is nothing to anchor on, so `embedding_input` falls back to content.
    """
    name = (name or "").strip()
    if not name:
        return ""
    place_bits = [b.strip() for b in (city, region) if b and b.strip()]
    locus = ", ".join([*place_bits, "Albania"])
    category = (category or "").strip()
    lead = f"{category} in {locus}" if category else f"in {locus}"
    header = f"{name} — {lead}."
    cleaned_tags = [t.strip() for t in (tags or []) if t and t.strip()]
    if cleaned_tags:
        header += f" Themes: {', '.join(cleaned_tags)}."
    return header


def embedding_input(context: str | None, content: str) -> str:
    """Text to embed / index for a chunk: the context header followed by the
    content, or content alone when there is no header. Whitespace separator is
    immaterial to both the embedder and the `simple` tsvector tokenizer, so this
    stays interchangeable with the `context || ' ' || content` SQL in the
    generated `search_text` column."""
    context = (context or "").strip()
    content = content or ""
    if not context:
        return content
    return f"{context}\n\n{content}"
