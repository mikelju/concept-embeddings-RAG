"""T10 of Phase 5: the gate, computed mechanically from per-question hit@10 (HU-7)."""

import itertools
import json
from pathlib import Path

import pytest
from scipy.stats import binomtest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.gate import (
    GO,
    NAMES_ONLY,
    STOP,
    GateError,
    compute_gate,
    decide,
    load_gate,
    save_gate,
    sign_test,
)
from concept_embeddings_rag.evaluation.navigation import ARM_HOPS

E, C, EC = (ARM_HOPS[arm] for arm in config.NAVIGATION_ARMS)
BM25 = config.GATE_COMPARATOR
DEPTH = str(config.GATE_METRIC_DEPTH)


def a_run(columns: dict[str, list[float]], digest: str = "run-digest") -> dict:
    """A hop run holding only what the gate reads: per-question hit@10 per hop."""
    n = len(next(iter(columns.values())))
    per_question = [
        {"qid": f"q{i:03d}", "scores": {hop: {DEPTH: values[i]} for hop, values in columns.items()}}
        for i in range(n)
    ]
    return {"per_question": per_question, "digest": digest}


def column(wins: int, losses: int, ties: int) -> tuple[list[float], list[float]]:
    """Two aligned columns where the first beats the second `wins` times and loses `losses`."""
    first = [1.0] * wins + [0.0] * losses + [0.5] * ties
    second = [0.0] * wins + [1.0] * losses + [0.5] * ties
    return first, second


def run_with(ec, e, c, bm25) -> dict:
    return a_run({EC: ec, E: e, C: c, BM25: bm25})


ZEROS = [0.0] * 20
ONES = [1.0] * 20


def test_the_p_value_is_the_one_sided_exact_binomial():
    first, second = column(wins=26, losses=14, ties=5)
    test = sign_test(a_run({EC: first, BM25: second})["per_question"], EC, BM25)

    assert (test.wins, test.losses, test.ties) == (26, 14, 5)
    assert test.p_value == pytest.approx(binomtest(26, 40, 0.5, alternative="greater").pvalue)
    assert test.passed is (test.p_value < config.GATE_ALPHA)


def test_no_discordant_question_is_p_one_and_a_failed_test():
    test = sign_test(a_run({EC: [0.5] * 10, BM25: [0.5] * 10})["per_question"], EC, BM25)

    assert (test.wins, test.losses, test.ties) == (0, 0, 10)
    assert test.p_value == 1.0
    assert test.passed is False


def test_go_when_entity_plus_concept_beats_the_comparator_and_entity_only():
    decision = compute_gate(run_with(ec=ONES, e=ZEROS, c=ZEROS, bm25=ZEROS))

    assert decision["outcome"] == GO
    assert decision["anomalies"] == []


def test_names_only_when_entities_alone_beat_the_comparator_and_concepts_add_nothing():
    decision = compute_gate(run_with(ec=ONES, e=ONES, c=ZEROS, bm25=ZEROS))

    assert decision["outcome"] == NAMES_ONLY
    assert decision["tests"]["E beats BM25"]["passed"] is True
    assert decision["tests"]["EC beats E"]["passed"] is False


def test_stop_when_neither_arm_beats_the_comparator():
    decision = compute_gate(run_with(ec=ZEROS, e=ZEROS, c=ZEROS, bm25=ONES))

    assert decision["outcome"] == STOP
    assert decision["anomalies"] == []


def test_the_ambiguous_pattern_is_stop_with_an_anomaly():
    """EC beats BM25, but neither beats E nor lets E beat BM25 on its own.

    Twenty questions. EC finds q0-q5 and q10-q13, E finds q0-q5, BM25 finds q10-q13:
    EC beats BM25 6-0 (p = 0.016), EC beats E only 4-0 (p = 0.0625), E beats BM25 6-4.
    """

    def hits(*ranges: range) -> list[float]:
        found = {q for r in ranges for q in r}
        return [1.0 if q in found else 0.0 for q in range(20)]

    ec = hits(range(0, 6), range(10, 14))
    e = hits(range(0, 6))
    bm25 = hits(range(10, 14))
    decision = compute_gate(run_with(ec=ec, e=e, c=ZEROS, bm25=bm25))

    assert decision["tests"]["EC beats BM25"]["passed"] is True
    assert decision["tests"]["EC beats E"]["passed"] is False
    assert decision["tests"]["E beats BM25"]["passed"] is False
    assert decision["outcome"] == STOP
    assert len(decision["anomalies"]) == 1


def test_concept_only_beating_the_comparator_alone_is_stop_with_an_anomaly():
    decision = compute_gate(run_with(ec=ZEROS, e=ZEROS, c=ONES, bm25=ZEROS))

    assert decision["reported"]["C beats BM25"]["passed"] is True
    assert decision["outcome"] == STOP
    assert len(decision["anomalies"]) == 1


def test_names_only_is_impossible_unless_entity_only_beat_the_comparator():
    for ec_bm25, ec_e, e_bm25, c_bm25 in itertools.product((False, True), repeat=4):
        outcome, anomalies = decide(
            ec_beats_comparator=ec_bm25,
            ec_beats_entity=ec_e,
            entity_beats_comparator=e_bm25,
            concept_beats_comparator=c_bm25,
        )
        if outcome == NAMES_ONLY:
            assert e_bm25 and not (ec_bm25 and ec_e)
        if outcome == GO:
            assert ec_bm25 and ec_e
        if outcome == STOP:
            assert not (ec_bm25 and ec_e) and not e_bm25
        assert outcome in (GO, NAMES_ONLY, STOP)
        assert (anomalies != []) == (outcome == STOP and (ec_bm25 or c_bm25))


def test_the_decision_records_its_metric_and_the_run_it_was_computed_from():
    decision = compute_gate(run_with(ec=ONES, e=ZEROS, c=ZEROS, bm25=ZEROS))

    assert decision["hop_run_digest"] == "run-digest"
    assert decision["metric"]["depth"] == config.GATE_METRIC_DEPTH
    assert decision["metric"]["alpha"] == config.GATE_ALPHA
    assert decision["metric"]["comparator"] == BM25
    assert set(decision["tests"]) == {"EC beats BM25", "EC beats E", "E beats BM25"}


def test_the_decision_is_saved_once_and_verified(tmp_path: Path):
    decision = compute_gate(run_with(ec=ONES, e=ZEROS, c=ZEROS, bm25=ZEROS))

    save_gate(decision, tmp_path)
    loaded = load_gate(tmp_path, hop_run_digest="run-digest")

    assert loaded["outcome"] == GO
    with pytest.raises(GateError, match="already"):
        save_gate(decision, tmp_path)


def test_a_modified_decision_or_one_from_another_run_is_refused(tmp_path: Path):
    save_gate(compute_gate(run_with(ec=ZEROS, e=ZEROS, c=ZEROS, bm25=ONES)), tmp_path)

    with pytest.raises(GateError, match="run"):
        load_gate(tmp_path, hop_run_digest="another-run")

    path = next(tmp_path.glob("gate-decision.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["outcome"] = GO
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GateError, match="digest"):
        load_gate(tmp_path, hop_run_digest="run-digest")
