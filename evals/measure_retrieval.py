"""Standalone local measurement of retrieval quality — no Braintrust key needed.

Loads `evals/fixtures/retrieval_quality.jsonl`, runs the live hybrid retriever
(`app.rag.retrieval.retrieve_chunks`) against the DB pointed to by
`settings.database_url`, and prints per-row + aggregate `retrieval_hit_rate` and
`retrieval_mrr` using the exact scorers the Braintrust suite uses. This is the
before/after harness for retrieval tuning (backlog #2/#3): seed the corpus,
run it, compare.

Requires a seeded `chunks` corpus (`python -m evals.seed_eval_db`). With a Cohere
key both arms run; without one, the vector arm is empty and only the lexical arm
contributes (the generated `search_text` still works).

Measurement integrity: the Cohere reranker is fail-soft (`app/rag/reranking.py`)
— on a rate-limit (the trial key allows 10 req/min) it silently returns
embed-only order with `rerank_score=None`. A run that hits the limit therefore
measures a DIFFERENT pipeline than the one configured, and the numbers are
quietly wrong. Two guards:
  - `--delay` paces fixtures (each does ~2 Cohere calls: 1 embed + 1 rerank) so
    a full 20-fixture run stays under 10 req/min. Default 16s.
  - any row with retrieved chunks but a `None` rerank score is flagged, and the
    run is declared INVALID at the end — do not record it as a baseline.

Usage (point DATABASE_URL at the SEEDED eval DB, never prod):
    DATABASE_URL=postgresql+asyncpg://... python -m evals.measure_retrieval
    DATABASE_URL=... python -m evals.measure_retrieval --delay 16
    DATABASE_URL=... python -m evals.measure_retrieval --only vlore-county-recall
"""
from __future__ import annotations

import argparse
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


async def _retrieve(message: str) -> tuple[list[str], float | None]:
    """Return (chunk contents, top rerank score). The score is surfaced so the
    caller can detect a silent reranker fallback (score None on a non-empty
    result = Cohere failed / was rate-limited → embed-only order)."""
    from app.db.engine import async_session
    from app.rag.retrieval import retrieve_chunks

    async with async_session() as db:
        chunks, score = await retrieve_chunks(message, db)
    return [c.content for c in chunks], score


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delay",
        type=float,
        default=16.0,
        help="Seconds to sleep between fixtures so the ~2 Cohere calls/fixture "
        "stay under the trial key's 10 req/min limit (default: 16).",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated fixture ids to run (default: all). Use during "
        "tuning to re-measure a single target cheaply.",
    )
    args = parser.parse_args()

    rows = _load()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        rows = [r for r in rows if r.get("id") in wanted]
        if not rows:
            print(f"no fixtures matched --only {args.only!r}")
            return

    hits: list[float] = []
    mrrs: list[float] = []
    degraded: list[str] = []  # rows where the reranker silently fell back
    print(f"{'id':<24} {'hit':>4} {'mrr':>6} {'rrk':>6}  message")
    print("-" * 86)
    for i, row in enumerate(rows):
        message = row.get("input", {}).get("message", "")
        expected = row.get("expected", {})
        contents, score = await _retrieve(message)
        hit = retrieval_hit_rate(contents, expected)
        mrr = retrieval_mrr(contents, expected)
        hits.append(hit)
        mrrs.append(mrr)
        # A non-empty result with no rerank score means Cohere rerank fell back
        # to embed-only order (rate limit / outage) — this row's ranking is NOT
        # the configured pipeline, so the whole run is suspect.
        flag = ""
        if contents and score is None:
            degraded.append(row.get("id", message))
            flag = " !! rerank fell back (embed-only)"
        score_str = f"{score:>6.3f}" if score is not None else f"{'—':>6}"
        print(
            f"{row.get('id', ''):<24} {hit:>4.2f} {mrr:>6.3f} {score_str}  "
            f"{message}{flag}"
        )
        # Pace requests under the rate limit; no sleep after the last fixture.
        if args.delay > 0 and i < len(rows) - 1:
            await asyncio.sleep(args.delay)

    n = len(rows) or 1
    print("-" * 86)
    print(
        f"{'MEAN':<24} {sum(hits) / n:>4.2f} {sum(mrrs) / n:>6.3f} {'':>6}  "
        f"(n={len(rows)})"
    )
    if degraded:
        print()
        print(
            f"INVALID RUN: {len(degraded)} row(s) fell back to embed-only order "
            f"(reranker returned no score): {', '.join(degraded)}.\n"
            f"Increase --delay or check COHERE_API_KEY / rate limit, then "
            f"re-run. Do NOT record these numbers as a baseline."
        )


if __name__ == "__main__":
    asyncio.run(main())
