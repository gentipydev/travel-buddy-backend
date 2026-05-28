"""Standalone local measurement of retrieval quality — no Braintrust key needed.

Loads `evals/fixtures/retrieval_quality.jsonl`, runs the live hybrid retriever
(`app.rag.retrieval.retrieve_chunks`) against the DB pointed to by
`settings.database_url`, and prints per-row + aggregate `retrieval_hit_rate` and
`retrieval_mrr` using the exact scorers the Braintrust suite uses. This is the
before/after harness for contextual retrieval (backlog #5): seed the corpus,
run it, compare.

Requires a seeded `chunks` corpus (`python -m evals.seed_eval_db`). With a Cohere
key both arms run; without one, the vector arm is empty and only the lexical arm
contributes (the generated `search_text` still works).

Usage (point DATABASE_URL at the SEEDED eval DB, never prod):
    DATABASE_URL=postgresql+asyncpg://... python -m evals.measure_retrieval
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.evals import retrieval_hit_rate, retrieval_mrr

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "retrieval_quality.jsonl"


def _load() -> list[dict[str, Any]]:
    """Load the retrieval fixtures, dropping leading-underscore authoring keys
    (e.g. `_note`) from `expected` so they don't count as ground truth — same
    convention as `evals/runners/braintrust_run.py`."""
    rows: list[dict[str, Any]] = []
    with open(_FIXTURE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row.get("expected"), dict):
                row["expected"] = {
                    k: v for k, v in row["expected"].items() if not k.startswith("_")
                }
            rows.append(row)
    return rows


async def _retrieve(message: str) -> list[str]:
    from app.db.engine import async_session
    from app.rag.retrieval import retrieve_chunks

    async with async_session() as db:
        chunks, _ = await retrieve_chunks(message, db)
    return [c.content for c in chunks]


async def main() -> None:
    rows = _load()
    hits: list[float] = []
    mrrs: list[float] = []
    print(f"{'id':<24} {'hit':>4} {'mrr':>6}  message")
    print("-" * 78)
    for row in rows:
        message = row.get("input", {}).get("message", "")
        expected = row.get("expected", {})
        contents = await _retrieve(message)
        hit = retrieval_hit_rate(contents, expected)
        mrr = retrieval_mrr(contents, expected)
        hits.append(hit)
        mrrs.append(mrr)
        print(f"{row.get('id', ''):<24} {hit:>4.2f} {mrr:>6.3f}  {message}")
    n = len(rows) or 1
    print("-" * 78)
    print(
        f"{'MEAN':<24} {sum(hits) / n:>4.2f} {sum(mrrs) / n:>6.3f}  "
        f"(n={len(rows)})"
    )


if __name__ == "__main__":
    asyncio.run(main())
