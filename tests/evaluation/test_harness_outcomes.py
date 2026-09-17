"""Phase 6, T2: the harness records each question's outcome when asked, and changes nothing else.

The paired statistics of Phase 6 need per-question outcomes, and the harness only kept
means. D8 adds one optional sink: when a list is handed in, the harness appends one record
per question, in its own loop order, holding the very values it averaged. With no sink the
harness is the one Phase 1 wrote, and the reproduction of every historical figure (HU-2) is
what shows it on real data.

A finding recorded while writing this test (implementation note in D9): on Python 3.12 the
builtin `sum` over floats is compensated (Neumaier), so a mean of fractions such as precision
did not change with the order of its terms in any list searched. The records are still kept
in the harness's order and the artifact still averages in that order, as the spec's contract
asks; the equality below is exact, and the order is checked directly rather than through a
sum that would have to disagree.
"""

import math
from dataclasses import fields

from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import harness
from concept_embeddings_rag.evaluation.harness import (
    BudgetOutcome,
    QuestionOutcome,
    RunResult,
    evaluate_retriever,
)

BUDGETS = (3, 6, 12)
KS = (2, 5, 10)
UNITS = [f"u{index:02d}" for index in range(12)]
TOKENS = dict.fromkeys(UNITS, 1)


class ByQuestionRetriever:
    """A fixed ranking per question text, so every figure is checkable by hand."""

    name = "scripted"

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        ranking = self.rankings[query]
        return [(unit, 1.0 / (rank + 1)) for rank, unit in enumerate(ranking)][:top_k]


def a_question(qid: str, gold: tuple[str, ...]) -> Question:
    return Question(
        qid=qid,
        question=f"question {qid}",
        answer="-",
        gold_unit_ids=gold,
        supporting_facts=(("t", 0),),
        split="dev",
    )


def a_config() -> dict:
    return {
        "model": "fake",
        "revision": "v0",
        "unit_set_hash": "abc123",
        "seed": 42,
        "tokenizer": "fake-tokenizer",
        "code_version": "0.1.0",
        "top_k": 12,
    }


# Question ids deliberately not in ascending order: the harness order is the list order.
QUESTIONS = [
    a_question("q3", ("u00", "u05")),
    a_question("q1", ("u01", "u02")),
    a_question("q2", ("u07", "u11")),
    a_question("q0", ("u04", "u09")),
]
RANKINGS = {
    "question q3": ["u00", "u01", "u02", "u03", "u04", "u05", "u06"],
    "question q1": ["u03", "u01", "u06", "u02", "u08", "u09", "u10", "u11"],
    "question q2": ["u07", "u00", "u01", "u02", "u03", "u04", "u05", "u06", "u08", "u11"],
    "question q0": ["u00", "u01", "u02", "u03", "u05", "u06", "u07", "u08", "u09", "u10", "u04"],
}


def run(outcomes: list[QuestionOutcome] | None = None) -> RunResult:
    return evaluate_retriever(
        ByQuestionRetriever(RANKINGS),
        questions=QUESTIONS,
        token_counts=TOKENS,
        budgets=BUDGETS,
        ks=KS,
        top_k=12,
        config=a_config(),
        split="dev",
        outcomes=outcomes,
    )


def test_with_the_sink_there_is_one_record_per_question_in_the_harness_loop_order():
    outcomes: list[QuestionOutcome] = []
    run(outcomes)

    assert [outcome.qid for outcome in outcomes] == [question.qid for question in QUESTIONS]


def test_each_record_carries_every_budget_and_every_recall_depth():
    outcomes: list[QuestionOutcome] = []
    run(outcomes)

    for outcome in outcomes:
        assert sorted(outcome.budgets) == sorted(BUDGETS)
        assert sorted(outcome.recall_at_k) == sorted(KS)
        for reading in outcome.budgets.values():
            assert isinstance(reading, BudgetOutcome)


def test_the_records_hold_the_values_computed_by_hand():
    outcomes: list[QuestionOutcome] = []
    run(outcomes)
    by_qid = {outcome.qid: outcome for outcome in outcomes}

    # q3 at 3 tokens: u00 u01 u02, one gold of two, one gold of three units.
    q3 = by_qid["q3"].budgets[3]
    assert (q3.gold_recall, q3.full_support, q3.precision, q3.units_included) == (
        0.5,
        0.0,
        1 / 3,
        3,
    )
    # q3 at 6 tokens: u00..u05, both gold of six units.
    q3 = by_qid["q3"].budgets[6]
    assert (q3.gold_recall, q3.full_support, q3.precision, q3.units_included) == (
        1.0,
        1.0,
        2 / 6,
        6,
    )
    # q1 at 12 tokens: all eight units fit, both gold.
    q1 = by_qid["q1"].budgets[12]
    assert (q1.gold_recall, q1.full_support, q1.precision, q1.units_included) == (
        1.0,
        1.0,
        2 / 8,
        8,
    )
    # q0 recall at 10 excludes u04, ranked eleventh.
    assert by_qid["q0"].recall_at_k == {2: 0.0, 5: 0.0, 10: 0.5}


def test_the_harness_mean_over_the_records_in_order_equals_the_metrics_exactly():
    outcomes: list[QuestionOutcome] = []
    result = run(outcomes)

    precisions = [outcome.budgets[12].precision for outcome in outcomes]
    assert any(0.0 < value < 1.0 and value not in (0.5, 0.25) for value in precisions), (
        "the toy must hold fractional precisions, or the equality below is trivial"
    )

    for budget in BUDGETS:
        row = result.metrics[f"budget_{budget}"]
        for name in ("gold_recall", "full_support", "precision"):
            values = [getattr(outcome.budgets[budget], name) for outcome in outcomes]
            assert harness._mean(values) == row[name]
    for k in KS:
        values = [outcome.recall_at_k[k] for outcome in outcomes]
        assert harness._mean(values) == result.metrics[f"recall_at_{k}"]

    largest = max(BUDGETS)
    units = [float(outcome.budgets[largest].units_included) for outcome in outcomes]
    assert harness._mean(units) == result.cost["mean_units_included"]


def test_the_harness_mean_is_a_compensated_sum_on_this_python():
    """The D9 note: a naive left fold of these three depends on order, the builtin does not."""
    forward, backward = [0.1, 0.2, 0.3], [0.3, 0.2, 0.1]
    assert math.fsum(forward) == math.fsum(backward)
    assert (0.1 + 0.2) + 0.3 != (0.3 + 0.2) + 0.1
    assert harness._mean(forward) == harness._mean(backward)


def test_the_sink_changes_no_metric_and_no_cost_but_latency():
    with_sink = run([])
    without = run(None)

    assert with_sink.metrics == without.metrics
    assert with_sink.system == without.system
    assert with_sink.split == without.split
    assert with_sink.config == without.config
    for key in ("mean_units_included", "n_questions"):
        assert with_sink.cost[key] == without.cost[key]
    assert set(with_sink.cost) == set(without.cost)


def test_the_run_result_gains_no_field():
    assert [field.name for field in fields(RunResult)] == [
        "system",
        "split",
        "config",
        "metrics",
        "cost",
        "created_at",
    ]


def test_a_sink_that_already_holds_records_is_appended_to_not_replaced():
    earlier = QuestionOutcome(qid="earlier", budgets={}, recall_at_k={})
    outcomes: list[QuestionOutcome] = [earlier]
    run(outcomes)

    assert outcomes[0] is earlier
    assert len(outcomes) == 1 + len(QUESTIONS)
