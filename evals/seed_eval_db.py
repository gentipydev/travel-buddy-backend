"""Seed a small, deterministic `chunks` corpus for the retrieval + faithfulness
eval suites.

Runs against any Postgres reachable via `settings.database_url`. It only ever
touches chunks tagged `source='eval'`, so it is safe to run against a shared
dev DB — it won't disturb the production corpus. In CI it runs against an
ephemeral pgvector container (see `.github/workflows/backend-evals.yml`); the
`retrieval_task` / `faithfulness_live_task` in `evals/runners/braintrust_run.py`
then query this corpus to produce real numbers.

Usage (from the project root):
    # fresh/empty DB (e.g. local docker pgvector or CI): create schema first
    DATABASE_URL=postgresql+asyncpg://... python -m evals.seed_eval_db --bootstrap

    # already-migrated DB (dev/staging): skip DDL, just (re)seed eval rows
    python -m evals.seed_eval_db

`--bootstrap` runs `CREATE EXTENSION vector` + SQLAlchemy `create_all` so a
bare Postgres has the schema the ORM expects. It is idempotent (checkfirst) and
omitted against a DB that already has the real migrations applied.

Embeddings: when `COHERE_API_KEY` is set, each chunk's contextual text
(`embedding_input(context, content)`, matching `scripts/embed_chunks.py`) is
embedded with `input_type=search_document` so the vector arm has signal. Without
a key, embeddings stay NULL and only the lexical arm contributes — the generated
`search_text` (context + content) populates regardless, so proper-noun and
metadata retrieval still work.

The context header is built from the corpus rows' metadata (city/region/category/
tags) via app/rag/contextual.py, so this eval corpus exercises the exact same
contextual-retrieval pipeline as the production `chunk_places.py` path.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import cohere
from sqlalchemy import delete, select, text

from app.config import settings
from app.db.engine import async_session, engine
from app.db.models import Base, Chunk
from app.rag.contextual import context_header, embedding_input

_CORPUS = Path(__file__).resolve().parent / "fixtures" / "eval_corpus.jsonl"
_EVAL_SOURCE = "eval"
_BATCH_SIZE = 96  # Cohere embed API ceiling per call


def _load_corpus() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(_CORPUS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def bootstrap_schema() -> None:
    """Create the `vector` extension and every ORM table (checkfirst). Needed
    only on a bare Postgres — the production schema is owned by
    supabase/migrations/. Uses create_all rather than replaying those .sql
    files because several of them reference Supabase-only schemas
    (`auth`, `storage`, RLS) that don't exist in a plain container."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def _embed(texts: list[str]) -> list[list[float]] | None:
    """Embed the given texts via Cohere search_document. Returns None when no
    key is configured (caller leaves embeddings NULL → lexical-only)."""
    if not settings.cohere_api_key:
        return None
    vectors: list[list[float]] = []
    async with cohere.AsyncClientV2(api_key=settings.cohere_api_key) as co:
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = await co.embed(
                model=settings.embedding_model,
                texts=batch,
                input_type="search_document",
                embedding_types=["float"],
            )
            vectors.extend(response.embeddings.float_)
    return vectors


def _context_for(row: dict[str, Any]) -> str:
    """Build the contextual header for a corpus row from its metadata."""
    return context_header(
        row.get("place_name", ""),
        row.get("city"),
        row.get("region"),
        row.get("category"),
        row.get("tags"),
    )


async def seed() -> None:
    corpus = _load_corpus()
    contexts = [_context_for(row) for row in corpus]
    embed_inputs = [
        embedding_input(contexts[i], row["content"]) for i, row in enumerate(corpus)
    ]
    vectors = await _embed(embed_inputs)

    async with async_session() as db:
        # Idempotent: clear only eval-tagged chunks, never the real corpus.
        await db.execute(delete(Chunk).where(Chunk.source == _EVAL_SOURCE))
        for i, row in enumerate(corpus):
            db.add(
                Chunk(
                    content=row["content"],
                    context=contexts[i],
                    source=_EVAL_SOURCE,
                    language="en",
                    embedding=vectors[i] if vectors is not None else None,
                    meta={"place_name": row.get("place_name", "")},
                )
            )
        await db.commit()
        count = (
            await db.execute(
                select(Chunk).where(Chunk.source == _EVAL_SOURCE)
            )
        ).scalars().all()

    print(
        f"seed_eval_db: seeded {len(count)} eval chunks "
        f"(embedded={'yes' if vectors is not None else 'no — lexical only'})"
    )


async def main(bootstrap: bool) -> None:
    if bootstrap:
        await bootstrap_schema()
    await seed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="CREATE EXTENSION vector + create_all before seeding (bare DB only).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.bootstrap))
