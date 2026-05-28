"""Unit tests for the retrieval-quality and faithfulness eval scorers.

The deterministic retrieval scorers are tested directly. `answer_faithfulness`
delegates to an autoevals LLMClassifier (live Haiku calls), so we monkeypatch
the lazily-built scorer with a recorder and assert the delegation contract
rather than judging anything for real.
"""
import pytest

from app.evals import scorers
from app.evals.scorers import (
    answer_faithfulness,
    retrieval_hit_rate,
    retrieval_mrr,
)

# ---------------------------------------------------------------------- #
# retrieval_hit_rate                                                    #
# ---------------------------------------------------------------------- #


def test_hit_rate_finds_relevant_chunk_anywhere():
    out = ["Berat castle overlooks the river", "Theth is a mountain village"]
    assert retrieval_hit_rate(out, {"relevant_keywords": ["Theth"]}) == 1.0


def test_hit_rate_zero_when_keyword_absent():
    out = ["Berat castle", "Saranda beaches"]
    assert retrieval_hit_rate(out, {"relevant_keywords": ["Theth"]}) == 0.0


def test_hit_rate_is_case_and_diacritic_insensitive():
    out = ["a guide to the town of gjirokaster"]
    assert retrieval_hit_rate(out, {"relevant_keywords": ["Gjirokastër"]}) == 1.0


def test_hit_rate_requires_all_keywords_and_semantics():
    both = ["Theth guesthouse booking info"]
    one = ["Theth village in the north"]
    kw = {"relevant_keywords": ["Theth", "guesthouse"]}
    assert retrieval_hit_rate(both, kw) == 1.0
    assert retrieval_hit_rate(one, kw) == 0.0


def test_hit_rate_empty_output_is_miss():
    assert retrieval_hit_rate([], {"relevant_keywords": ["Theth"]}) == 0.0


def test_hit_rate_abstains_without_keywords():
    assert retrieval_hit_rate(["anything"], {}) == 1.0
    assert retrieval_hit_rate(["anything"], {"relevant_keywords": []}) == 1.0


def test_hit_rate_non_list_output_is_zero():
    assert retrieval_hit_rate("not a list", {"relevant_keywords": ["Theth"]}) == 0.0


# ---------------------------------------------------------------------- #
# retrieval_mrr                                                         #
# ---------------------------------------------------------------------- #


def test_mrr_rank_one_is_full_score():
    out = ["Theth is up north", "unrelated"]
    assert retrieval_mrr(out, {"relevant_keywords": ["Theth"]}) == 1.0


def test_mrr_rewards_higher_rank():
    out = ["irrelevant", "still no", "Theth at last"]
    assert retrieval_mrr(out, {"relevant_keywords": ["Theth"]}) == pytest.approx(1 / 3)


def test_mrr_zero_when_no_relevant_chunk():
    out = ["Berat", "Saranda", "Vlora"]
    assert retrieval_mrr(out, {"relevant_keywords": ["Theth"]}) == 0.0


def test_mrr_abstains_without_keywords():
    assert retrieval_mrr(["x", "y"], {"relevant_keywords": []}) == 1.0


def test_mrr_non_list_output_is_zero():
    assert retrieval_mrr(None, {"relevant_keywords": ["Theth"]}) == 0.0


# ---------------------------------------------------------------------- #
# answer_faithfulness (delegation only — judge is mocked)               #
# ---------------------------------------------------------------------- #


def test_faithfulness_delegates_output_and_input(monkeypatch):
    calls = []

    def _recorder(**kwargs):
        calls.append(kwargs)
        return 1.0

    # Pre-seed the lazy global so answer_faithfulness skips _build_faithfulness
    # (which would import autoevals and make live Haiku calls).
    monkeypatch.setattr(scorers, "_faithfulness_scorer", _recorder)

    result = answer_faithfulness(
        "Berat castle is free to enter",
        input={"message": "tell me about Berat castle", "context": "Entry is free."},
    )

    assert result == 1.0
    assert len(calls) == 1
    # Calibration shape: answer from output, context from input.
    assert calls[0]["output"] == "Berat castle is free to enter"
    assert calls[0]["input"]["message"] == "tell me about Berat castle"
    assert calls[0]["input"]["context"] == "Entry is free."


def test_faithfulness_live_shape_reads_context_from_output(monkeypatch):
    """Live suite: the task returns {answer, context}; the scorer must pull the
    answer + context from `output` and the message from `input`, then hand the
    judge the same normalized {message, context} + answer."""
    calls = []

    def _recorder(**kwargs):
        calls.append(kwargs)
        return 0.0

    monkeypatch.setattr(scorers, "_faithfulness_scorer", _recorder)

    result = answer_faithfulness(
        {"answer": "It costs 800 lek", "context": "Entry is free."},
        input={"message": "how much is Berat castle?"},
    )

    assert result == 0.0
    assert calls[0]["output"] == "It costs 800 lek"
    assert calls[0]["input"]["message"] == "how much is Berat castle?"
    assert calls[0]["input"]["context"] == "Entry is free."
