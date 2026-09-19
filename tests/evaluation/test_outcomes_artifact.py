"""Phase 6, T3: `PerQuestionOutcomes`, the artifact the paired statistics are computed from.

Three properties make it evidence rather than a dump (D9):

- **The whole population, and nothing else.** A question set missing one question of the
  split, or holding a foreign one, is refused: a paired test over a population that quietly
  lost its hardest question is a different test.
- **Bound to the harness's own figures.** The loader recomputes every mean from the records,
  in the pool's split order and with the harness's arithmetic, and refuses the file unless
  the result equals both its stored aggregates and the named run result exactly.
- **Written once.** A second file for the same system and split takes a suffix and never
  replaces the first; a test outcome must name the freeze it was measured under.
"""

import json
from pathlib import Path

import pytest

from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import outcomes as outcomes_module
from concept_embeddings_rag.evaluation.harness import QuestionOutcome, RunResult, evaluate_retriever
from concept_embeddings_rag.evaluation.outcomes import (
    OutcomesError,
    build_outcomes,
    load_outcomes,
    save_outcomes,
)

BUDGETS = (3, 6, 12)
KS = (2, 5, 10)
UNITS = [f"u{index:02d}" for index in range(12)]
TOKENS = dict.fromkeys(UNITS, 1)
FREEZE = "f" * 64


class ByQuestionRetriever:
    name = "dense"

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        ranking = self.rankings[query]
        return [(unit, 1.0 / (rank + 1)) for rank, unit in enumerate(ranking)][:top_k]


def a_question(qid: str, gold: tuple[str, ...], split: str = "dev") -> Question:
    return Question(
        qid=qid,
        question=f"question {qid}",
        answer="-",
        gold_unit_ids=gold,
        supporting_facts=(("t", 0),),
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
        "top_k": 12,
    }


# Pool order is not ascending qid order, so the stored order and the harness order differ.
POOL_ORDER = ["q3", "q1", "q2", "q0"]
GOLD = {"q3": ("u00", "u05"), "q1": ("u01", "u02"), "q2": ("u07", "u11"), "q0": ("u04", "u09")}
RANKINGS = {
    "question q3": ["u00", "u01", "u02", "u03", "u04", "u05", "u06"],
    "question q1": ["u03", "u01", "u06", "u02", "u08", "u09", "u10", "u11"],
    "question q2": ["u07", "u00", "u01", "u02", "u03", "u04", "u05", "u06", "u08", "u11"],
    "question q0": ["u00", "u01", "u02", "u03", "u05", "u06", "u07", "u08", "u09", "u10", "u04"],
}


def measure(split: str = "dev") -> tuple[RunResult, list[QuestionOutcome]]:
    questions = [a_question(qid, GOLD[qid], split) for qid in POOL_ORDER]
    records: list[QuestionOutcome] = []
    result = evaluate_retriever(
        ByQuestionRetriever(RANKINGS),
        questions=questions,
        token_counts=TOKENS,
        budgets=BUDGETS,
        ks=KS,
        top_k=12,
        config=a_config(),
        split=split,
        outcomes=records,
    )
    return result, records


def written(tmp_path: Path, split: str = "dev", freeze: str | None = None) -> tuple[Path, Path]:
    result, records = measure(split)
    run_path = result.save(tmp_path)
    payload = build_outcomes(
        result, records, run_file=run_path.name, split_order=POOL_ORDER, freeze_digest=freeze
    )
    return save_outcomes(payload, tmp_path), run_path


def rewrite(path: Path, change) -> None:
    """Edit a stored payload and re-seal its digest, so only the targeted check can object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    payload["digest"] = outcomes_module.digest_of_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# --- The round trip -------------------------------------------------------------------


def test_a_written_artifact_loads_and_holds_every_question_sorted_by_qid(tmp_path):
    path, _ = written(tmp_path)

    loaded = load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)

    assert loaded.system == "dense"
    assert loaded.split == "dev"
    assert [entry["qid"] for entry in loaded.questions] == sorted(POOL_ORDER)
    assert loaded.freeze_digest is None
    assert loaded.run_result["file"] == next(tmp_path.glob("run-*.json")).name


def test_each_entry_has_every_budget_figure_and_the_three_recall_depths(tmp_path):
    path, _ = written(tmp_path)
    loaded = load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)

    for entry in loaded.questions:
        for budget in BUDGETS:
            assert set(entry[f"budget_{budget}"]) == {
                "gold_recall",
                "full_support",
                "precision",
                "units_included",
            }
        for k in KS:
            assert f"recall_at_{k}" in entry


def test_the_aggregates_equal_the_run_result_exactly(tmp_path):
    path, run_path = written(tmp_path)
    loaded = load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))

    assert loaded.aggregates["metrics"] == run["metrics"]
    assert loaded.aggregates["mean_units_included"] == run["cost"]["mean_units_included"]


def test_the_loader_averages_in_the_pool_split_order_not_the_stored_order(tmp_path, monkeypatch):
    """D9: the records are stored by qid and reordered by the pool before any mean is taken."""
    path, _ = written(tmp_path)
    seen: list[list[float]] = []
    original = outcomes_module._mean

    def spy(values):
        seen.append(list(values))
        return original(values)

    monkeypatch.setattr(outcomes_module, "_mean", spy)
    load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)

    pool_order = [7.0, 8.0, 10.0, 11.0]  # units at 12 tokens for q3, q1, q2, q0
    stored_order = [11.0, 8.0, 10.0, 7.0]  # the same units for q0, q1, q2, q3
    assert sorted(POOL_ORDER) != POOL_ORDER
    assert pool_order in seen
    assert stored_order not in seen


# --- Population -------------------------------------------------------------------------


def test_a_question_set_missing_one_question_of_the_split_is_refused(tmp_path):
    result, records = measure()
    with pytest.raises(OutcomesError, match="population"):
        build_outcomes(
            result,
            records[:-1],
            run_file="run.json",
            split_order=POOL_ORDER,
            freeze_digest=None,
        )


def test_a_question_set_holding_a_foreign_question_is_refused(tmp_path):
    result, records = measure()
    foreign = QuestionOutcome(
        qid="q9", budgets=records[0].budgets, recall_at_k=records[0].recall_at_k
    )
    with pytest.raises(OutcomesError, match="population"):
        build_outcomes(
            result,
            [*records[:-1], foreign],
            run_file="run.json",
            split_order=POOL_ORDER,
            freeze_digest=None,
        )


def test_the_loader_refuses_a_population_other_than_the_splits(tmp_path):
    path, _ = written(tmp_path)
    with pytest.raises(OutcomesError, match="population"):
        load_outcomes(path, split_order=[*POOL_ORDER, "q4"], run_dir=tmp_path)
    with pytest.raises(OutcomesError, match="population"):
        load_outcomes(path, split_order=POOL_ORDER[:-1], run_dir=tmp_path)


def test_a_duplicated_question_is_refused_on_load(tmp_path):
    path, _ = written(tmp_path)
    rewrite(path, lambda payload: payload["questions"].append(dict(payload["questions"][0])))
    with pytest.raises(OutcomesError):
        load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)


# --- Integrity ----------------------------------------------------------------------------


def test_a_modified_value_is_refused_by_the_digest(tmp_path):
    path, _ = written(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["questions"][0]["budget_12"]["full_support"] = 1.0 - float(
        payload["questions"][0]["budget_12"]["full_support"]
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(OutcomesError, match="digest"):
        load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)


def test_a_modified_value_under_a_resealed_digest_is_refused_by_the_recomputed_means(tmp_path):
    path, _ = written(tmp_path)

    def flip(payload):
        entry = payload["questions"][0]["budget_12"]
        entry["full_support"] = 1.0 - float(entry["full_support"])

    rewrite(path, flip)
    with pytest.raises(OutcomesError, match="aggregates"):
        load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)


def test_stored_aggregates_that_disagree_with_the_records_are_refused(tmp_path):
    path, _ = written(tmp_path)

    def nudge(payload):
        payload["aggregates"]["metrics"]["recall_at_2"] += 0.25

    rewrite(path, nudge)
    with pytest.raises(OutcomesError, match="aggregates"):
        load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)


def test_a_mismatched_run_result_is_refused(tmp_path):
    path, run_path = written(tmp_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["metrics"]["budget_6"]["precision"] += 0.001
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(OutcomesError, match="run result"):
        load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)


def test_a_run_result_of_another_system_is_refused(tmp_path):
    path, run_path = written(tmp_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["system"] = "hybrid-bm25"
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(OutcomesError, match="run result"):
        load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)


def test_a_missing_run_result_is_refused(tmp_path):
    path, run_path = written(tmp_path)
    run_path.unlink()
    with pytest.raises(OutcomesError, match="run result"):
        load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)


def test_a_builder_handed_records_that_disagree_with_the_result_refuses(tmp_path):
    result, records = measure()
    first = records[0]
    altered_budget = dict(first.budgets)
    reading = altered_budget[12]
    altered_budget[12] = type(reading)(
        reading.gold_recall, 1.0 - reading.full_support, reading.precision, reading.units_included
    )
    altered = QuestionOutcome(first.qid, altered_budget, first.recall_at_k)
    with pytest.raises(OutcomesError, match="aggregates"):
        build_outcomes(
            result,
            [altered, *records[1:]],
            run_file="run.json",
            split_order=POOL_ORDER,
            freeze_digest=None,
        )


# --- Freeze binding --------------------------------------------------------------------


def test_a_test_outcome_without_a_freeze_digest_is_refused(tmp_path):
    result, records = measure("test")
    with pytest.raises(OutcomesError, match="freeze"):
        build_outcomes(
            result, records, run_file="run.json", split_order=POOL_ORDER, freeze_digest=None
        )


def test_a_test_outcome_whose_stored_freeze_digest_was_removed_is_refused_on_load(tmp_path):
    path, _ = written(tmp_path, split="test", freeze=FREEZE)
    assert load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path).freeze_digest == FREEZE

    rewrite(path, lambda payload: payload.update({"freeze_digest": None}))
    with pytest.raises(OutcomesError, match="freeze"):
        load_outcomes(path, split_order=POOL_ORDER, run_dir=tmp_path)


def test_a_freeze_digest_that_is_not_a_sha256_is_refused():
    result, records = measure("test")
    with pytest.raises(OutcomesError, match="freeze"):
        build_outcomes(
            result, records, run_file="run.json", split_order=POOL_ORDER, freeze_digest="../x"
        )


def test_a_dev_outcome_carries_no_freeze_digest():
    """Dev outcomes exist before the freeze; the freeze references them, not the other way."""
    result, records = measure("dev")
    with pytest.raises(OutcomesError, match="freeze"):
        build_outcomes(
            result, records, run_file="run.json", split_order=POOL_ORDER, freeze_digest=FREEZE
        )


def test_a_system_outside_the_phase_is_refused():
    result, records = measure()
    other = RunResult(
        system="conceptual",
        split=result.split,
        config=result.config,
        metrics=result.metrics,
        cost=result.cost,
    )
    with pytest.raises(OutcomesError, match="system"):
        build_outcomes(
            other, records, run_file="run.json", split_order=POOL_ORDER, freeze_digest=None
        )


# --- Written once ------------------------------------------------------------------------


def test_writing_twice_never_overwrites(tmp_path):
    first, _ = written(tmp_path)
    before = first.read_bytes()
    second, _ = written(tmp_path)

    assert first.name == "outcomes-dense-dev.json"
    assert second.name == "outcomes-dense-dev-2.json"
    assert first.read_bytes() == before
    assert len(list(tmp_path.glob("outcomes-*.json"))) == 2


def test_the_written_file_is_ascii_json_with_a_digest(tmp_path):
    path, _ = written(tmp_path)
    raw = path.read_bytes()
    assert raw.isascii()
    payload = json.loads(raw)
    assert payload["digest"] == outcomes_module.digest_of_payload(payload)
