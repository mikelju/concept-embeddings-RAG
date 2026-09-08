"""T12: the evaluation harness.

Two guarantees matter beyond the arithmetic: a result without complete provenance
is refused rather than written, and a new run never overwrites an older one. It is
too easy to lose the measurement that contradicted the hypothesis.
"""

import json

import pytest

from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.harness import (
    ProvenanceError,
    RunResult,
    evaluate_retriever,
)

TOKEN_COUNTS = {"g1": 100, "g2": 100, "x": 100, "y": 100}


class ScriptedRetriever:
    """Returns a fixed ranking, so the harness arithmetic is checkable by hand."""

    name = "scripted"

    def __init__(self, ranking):
        self.ranking = ranking

    def retrieve(self, query: str, top_k: int):
        return [(unit_id, 1.0 / (rank + 1)) for rank, unit_id in enumerate(self.ranking)][:top_k]


def a_question(qid: str = "q1", split: str = "dev") -> Question:
    return Question(
        qid=qid,
        question="a question",
        answer="an answer",
        gold_unit_ids=("g1", "g2"),
        supporting_facts=(("A", 0), ("B", 1)),
        split=split,
    )


def a_config() -> dict:
    return {
        "model": "fake",
        "revision": "v0",
        "unit_set_hash": "abc123",
        "seed": 42,
        "tokenizer": "fake-tokenizer",
        "code_version": "0.1.0",
        "top_k": 10,
    }


def test_metrics_reflect_what_fits_in_each_budget():
    retriever = ScriptedRetriever(["x", "g1", "y", "g2"])
    result = evaluate_retriever(
        retriever,
        questions=[a_question()],
        token_counts=TOKEN_COUNTS,
        budgets=(200, 400),
        ks=(2,),
        top_k=10,
        config=a_config(),
    )

    # At 200 tokens only "x" and "g1" fit: half the gold, no full support.
    assert result.metrics["budget_200"]["gold_recall"] == 0.5
    assert result.metrics["budget_200"]["full_support"] == 0.0
    # At 400 everything fits.
    assert result.metrics["budget_400"]["gold_recall"] == 1.0
    assert result.metrics["budget_400"]["full_support"] == 1.0


def test_recall_at_k_is_reported_over_the_ranking_not_the_budget():
    retriever = ScriptedRetriever(["x", "g1", "y", "g2"])
    result = evaluate_retriever(
        retriever,
        questions=[a_question()],
        token_counts=TOKEN_COUNTS,
        budgets=(200,),
        ks=(2, 5),
        top_k=10,
        config=a_config(),
    )

    assert result.metrics["recall_at_2"] == 0.5
    assert result.metrics["recall_at_5"] == 1.0


def test_cost_is_recorded():
    result = evaluate_retriever(
        ScriptedRetriever(["g1", "g2"]),
        questions=[a_question()],
        token_counts=TOKEN_COUNTS,
        budgets=(400,),
        ks=(2,),
        top_k=10,
        config=a_config(),
    )

    assert result.cost["mean_latency_ms"] >= 0.0
    assert result.cost["mean_units_included"] == 2.0
    assert result.cost["n_questions"] == 1


def test_a_result_without_complete_provenance_is_refused(tmp_path):
    incomplete = dict(a_config())
    del incomplete["seed"]

    with pytest.raises(ProvenanceError):
        RunResult(
            system="scripted",
            split="dev",
            config=incomplete,
            metrics={},
            cost={},
        ).save(tmp_path)


def test_a_new_run_never_overwrites_a_previous_one(tmp_path):
    def make() -> RunResult:
        return RunResult(system="scripted", split="dev", config=a_config(), metrics={}, cost={})

    first = make().save(tmp_path)
    second = make().save(tmp_path)

    assert first != second
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_saved_result_is_readable_json_carrying_its_config(tmp_path):
    path = RunResult(system="scripted", split="dev", config=a_config(), metrics={}, cost={}).save(
        tmp_path
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["config"]["seed"] == 42
    assert payload["system"] == "scripted"
    assert payload["created_at"]


def test_splits_are_evaluated_separately():
    questions = [a_question("q1", "dev"), a_question("q2", "test")]
    retriever = ScriptedRetriever(["g1", "g2"])

    dev = evaluate_retriever(
        retriever,
        questions=[q for q in questions if q.split == "dev"],
        token_counts=TOKEN_COUNTS,
        budgets=(400,),
        ks=(2,),
        top_k=10,
        config=a_config(),
        split="dev",
    )

    assert dev.split == "dev"
    assert dev.cost["n_questions"] == 1


def test_a_split_that_is_not_filename_safe_is_refused(tmp_path):
    """SEC-006: `split` can reach the harness from pool.json, and it lands in a
    path. A traversal sequence there would write the result outside results/."""
    result = RunResult(
        system="dense",
        split="../../escaped",
        config=a_config(),
        metrics={},
        cost={},
    )
    with pytest.raises(ProvenanceError, match="not usable in a filename"):
        result.save(tmp_path)


def test_a_system_name_that_is_not_filename_safe_is_refused(tmp_path):
    result = RunResult(
        system="dense/../evil",
        split="dev",
        config=a_config(),
        metrics={},
        cost={},
    )
    with pytest.raises(ProvenanceError):
        result.save(tmp_path)
