"""Static consistency guards for the retrieval eval fixtures.

These are pure-file checks (no DB, no Cohere) that run in `backend-ci.yml`
alongside the rest of the suite. They protect two invariants that the live
measurement (`evals/measure_retrieval.py`) and the Braintrust suite both rely
on but can't cheaply assert:

  1. Every `eval_corpus.jsonl` row is well-formed and geographically sane —
     `content` + `place_name` present, and any `region` is a real Albanian
     county. This catches the kind of geo drift the 2026-05-18 itinerary
     incident was about, at authoring time instead of in production.
  2. Every `retrieval_quality.jsonl` fixture is *gradeable*: at least one corpus
     chunk's CONTENT satisfies its `relevant_keywords` under the exact predicate
     the scorer uses (`_chunk_is_relevant`). A fixture whose keyword no chunk
     contains would silently score 0.0 forever and look like a real regression.

Note (2) checks content only — not the context header — because
`retrieval_hit_rate` / `retrieval_mrr` score the retrieved chunk *content*; the
header anchors retrieval but never reaches the scorer. Keeping this test aligned
with that means a fixture can't pass here yet be ungradeable in the suite.
"""
import json
from pathlib import Path

from app.evals.scorers import _canon, _chunk_is_relevant

_FIXTURES = Path(__file__).resolve().parents[1] / "evals" / "fixtures"
_CORPUS = _FIXTURES / "eval_corpus.jsonl"
_RETRIEVAL = _FIXTURES / "retrieval_quality.jsonl"
_GUARD = _FIXTURES / "output_guardrail.jsonl"

# The 12 counties (qarqe) of Albania, canonicalized. The corpus stores region
# as "<County> County"; we strip that suffix and match the canon stem.
_ALBANIAN_COUNTIES = {
    "berat", "diber", "durres", "elbasan", "fier", "gjirokaster", "korce",
    "kukes", "lezhe", "shkoder", "tirana", "vlore",
}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - failure path
            raise AssertionError(f"{path.name}:{i} is not valid JSON: {exc}") from exc
    return rows


def _county_stem(region: str) -> str:
    canon = _canon(region).replace("county", "").replace("qark", "").strip()
    return canon


def test_corpus_rows_are_well_formed_and_in_albania():
    corpus = _load_jsonl(_CORPUS)
    assert corpus, "eval corpus is empty"
    for row in corpus:
        name = row.get("place_name", "")
        assert name and name.strip(), f"row missing place_name: {row}"
        assert row.get("content", "").strip(), f"{name}: empty content"
        region = (row.get("region") or "").strip()
        if region:
            assert _county_stem(region) in _ALBANIAN_COUNTIES, (
                f"{name}: region {region!r} is not a recognized Albanian county"
            )


def test_output_guardrail_fixtures_are_valid():
    """Every output_guardrail.jsonl row is well-formed and has a valid label.

    Guards two invariants:
      1. Each fixture has a non-empty `input.response` string.
      2. Each fixture's `expected.label` is exactly "safe" or "review".
      3. Both classes are represented — the judge can't calibrate if one is absent.
    """
    fixtures = _load_jsonl(_GUARD)
    assert fixtures, "output_guardrail fixtures are empty"
    labels: set[str] = set()
    for fx in fixtures:
        fid = fx.get("id", "<no id>")
        assert fx.get("input", {}).get("response", "").strip(), (
            f"{fid}: input.response is empty"
        )
        label = fx.get("expected", {}).get("label", "")
        assert label in {"safe", "review"}, (
            f"{fid}: expected.label must be 'safe' or 'review', got {label!r}"
        )
        labels.add(label)
    assert "safe" in labels, "no safe fixtures — judge cannot calibrate positive class"
    assert "review" in labels, "no review fixtures — judge cannot calibrate negative class"


def test_every_retrieval_fixture_is_gradeable():
    contents = [r["content"] for r in _load_jsonl(_CORPUS)]
    fixtures = _load_jsonl(_RETRIEVAL)
    assert fixtures, "retrieval_quality fixtures are empty"
    for fx in fixtures:
        fid = fx.get("id", "<no id>")
        assert fx.get("input", {}).get("message", "").strip(), f"{fid}: empty message"
        keywords = fx.get("expected", {}).get("relevant_keywords")
        assert keywords, f"{fid}: missing relevant_keywords"
        satisfiable = any(_chunk_is_relevant(c, keywords) for c in contents)
        assert satisfiable, (
            f"{fid}: no corpus chunk content satisfies keywords {keywords!r} — "
            f"fixture would score 0.0 forever (keyword typo or missing chunk?)"
        )
