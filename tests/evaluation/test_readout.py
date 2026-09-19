"""Phase 6, T15: the descriptive readout and the qualitative sample (HU-7, HU-8, D18).

Everything the report needs that decides nothing: Full Support, gold recall and precision per
system and budget; the three paired 2x2 tables at 2,048 and the paired counts at the other
budgets; paired gold-recall differences; the failure analysis of B and of A against dense; the
bridge-like subgroup (a gold paragraph outside `read(q)`), compared with the pilot on dev; cost;
and, on test, the frozen sampling rule. It is built from outcome artifacts read back from disk,
and nothing on the decision path reads it.
"""

import ast
import json
from pathlib import Path

import pytest
from outcome_fixtures import FREEZE_DIGEST, vectors_from_counts, write_outcomes

from concept_embeddings_rag import decision_parameters as dp
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.readout import (
    ReadoutError,
    build_readout,
    load_readout,
    sample_groups,
    save_readout,
)
from concept_embeddings_rag.evaluation.replacement_decision import (
    compute_decision,
    parameters_payload,
)

SOURCE = Path(__file__).resolve().parents[2] / "src" / "concept_embeddings_rag" / "evaluation"
SHAPE = {
    (1, 1, 1): 8,
    (1, 1, 0): 3,
    (1, 0, 1): 1,
    (1, 0, 0): 2,
    (0, 1, 1): 6,
    (0, 1, 0): 2,
    (0, 0, 1): 7,
    (0, 0, 0): 1,
}


def a_freeze() -> dict:
    parameters = json.loads(json.dumps(parameters_payload()))
    return {
        "digest": FREEZE_DIGEST,
        "control": {"mode": "reused"},
        "decision_parameters": parameters,
        "qualitative_sample_rule": json.loads(json.dumps(dp.QUALITATIVE_SAMPLE_RULE)),
    }


def questions_for(qids: list[str], split: str) -> list[Question]:
    return [
        Question(qid, f"question {qid}", "-", (f"{qid}-a", f"{qid}-b"), (("t", 0),), split)
        for qid in qids
    ]


def reads_for(qids: list[str], bridge: set[str]) -> dict[str, list[str]]:
    """A read list holding both gold paragraphs, except for the bridge-like questions."""
    reads = {}
    for qid in qids:
        golds = [f"{qid}-a"] if qid in bridge else [f"{qid}-a", f"{qid}-b"]
        reads[qid] = [*golds, *(f"filler-{i}" for i in range(10 - len(golds)))]
    return reads


def costs() -> dict:
    return {
        system: {"mean_latency_ms": 1.5, "mean_units_included": 7.0, "n_questions": 30}
        for system in ("dense", "hybrid-bm25", "hybrid-entity-hop")
    }


@pytest.fixture(scope="module")
def test_world(tmp_path_factory):
    directory = tmp_path_factory.mktemp("readout-test")
    outcomes, qids = write_outcomes(directory, vectors_from_counts(SHAPE))
    return outcomes, qids


def readout_of(world, *, split="test", bridge=None, pilot=None):
    outcomes, qids = world
    bridge = set(qids[:4]) if bridge is None else bridge
    return build_readout(
        split=split,
        outcomes=outcomes,
        questions=questions_for(qids, split),
        read_lists=reads_for(qids, bridge),
        costs=costs(),
        freeze=a_freeze(),
        pilot_qids=pilot,
    )


def vector(counts, position):
    return vectors_from_counts(counts)[("dense", "hybrid-bm25", "hybrid-entity-hop")[position]]


# --- Tables -----------------------------------------------------------------------------------


def test_the_readout_is_labelled_descriptive_and_bound_to_the_freeze_and_outcomes(test_world):
    readout = readout_of(test_world)
    assert readout["descriptive"] is True
    assert readout["freeze_digest"] == FREEZE_DIGEST
    assert set(readout["outcome_digests"]) == {"dense", "hybrid-bm25", "hybrid-entity-hop"}


def test_the_per_system_table_reads_the_outcome_aggregates(test_world):
    outcomes, _ = test_world
    table = readout_of(test_world)["systems"]
    for system, artifact in outcomes.items():
        for budget in dp.QUALITATIVE_SAMPLE_RULE["budget"], 512, 1024, 4096:
            row = table[system][f"budget_{budget}"]
            assert row == artifact.aggregates["metrics"][f"budget_{budget}"]
    assert table["dense"]["budget_2048"]["full_support"] == 14 / 30


def test_the_three_2x2_tables_sum_to_n(test_world):
    tables = readout_of(test_world)["paired"]["2048"]
    assert set(tables) == {"b_vs_a", "b_vs_dense", "a_vs_dense"}
    for cells in tables.values():
        assert sum(cells.values()) == 30
    assert tables["b_vs_a"] == {"both": 14, "first_only": 8, "second_only": 5, "neither": 3}


def test_paired_counts_are_given_at_every_budget(test_world):
    paired = readout_of(test_world)["paired"]
    assert set(paired) == {"512", "1024", "2048", "4096"}


def test_the_tables_agree_with_the_decisions_counts(tmp_path):
    shape = {
        (1, 1, 1): 1100,
        (1, 1, 0): 30,
        (1, 0, 1): 10,
        (1, 0, 0): 15,
        (0, 1, 1): 60,
        (0, 1, 0): 20,
        (0, 0, 1): 20,
        (0, 0, 0): 145,
    }
    outcomes, qids = write_outcomes(tmp_path, vectors_from_counts(shape))
    decision = compute_decision(a_freeze(), outcomes, {"mode": "reused", "passed": True})
    readout = build_readout(
        split="test",
        outcomes=outcomes,
        questions=questions_for(qids, "test"),
        read_lists=reads_for(qids, set()),
        costs=costs(),
        freeze=a_freeze(),
    )
    assert readout["paired"]["2048"] == decision["tables"]


def test_paired_gold_recall_differences_at_2048(test_world):
    differences = readout_of(test_world)["gold_recall_differences"]["b_vs_a"]
    assert differences["positive"] == 8
    assert differences["negative"] == 5
    assert differences["zero"] == 17
    assert differences["mean"] == pytest.approx((8 - 5) * 0.5 / 30)


# --- Failure analysis -------------------------------------------------------------------------


def test_the_failure_lists_are_the_hand_computed_qids(test_world):
    outcomes, qids = test_world
    dense, control, entity = (vector(SHAPE, position) for position in range(3))
    expected_b_recovers = sorted(
        q for q, d, b in zip(qids, dense, entity, strict=True) if b and not d
    )
    expected_b_loses = sorted(q for q, d, b in zip(qids, dense, entity, strict=True) if d and not b)
    expected_a_recovers = sorted(
        q for q, d, a in zip(qids, dense, control, strict=True) if a and not d
    )
    expected_a_loses = sorted(
        q for q, d, a in zip(qids, dense, control, strict=True) if d and not a
    )

    failures = readout_of(test_world)["failure_analysis"]
    assert failures["hybrid-entity-hop"]["recovered"] == expected_b_recovers
    assert failures["hybrid-entity-hop"]["lost"] == expected_b_loses
    assert failures["hybrid-bm25"]["recovered"] == expected_a_recovers
    assert failures["hybrid-bm25"]["lost"] == expected_a_loses
    assert len(expected_b_recovers) == 13 and len(expected_b_loses) == 5


# --- The bridge-like subgroup -----------------------------------------------------------------


def test_the_subgroup_is_exactly_the_questions_with_a_gold_outside_the_read_list(test_world):
    _, qids = test_world
    bridge = {qids[1], qids[7], qids[20]}
    subgroup = readout_of(test_world, bridge=bridge)["subgroup"]
    assert subgroup["qids"] == sorted(bridge)
    assert subgroup["n_questions"] == 3
    assert set(subgroup["full_support"]) == {"dense", "hybrid-bm25", "hybrid-entity-hop"}


def test_on_dev_the_subgroup_is_compared_with_the_pilot_equal_and_different(tmp_path):
    outcomes, qids = write_outcomes(
        tmp_path, vectors_from_counts({(1, 1, 1): 6}), split="dev", freeze_digest=None
    )
    world = (outcomes, qids)
    bridge = {qids[0], qids[3]}

    equal = readout_of(world, split="dev", bridge=bridge, pilot=set(bridge))["subgroup"]
    assert equal["pilot_comparison"] == {"equal": True, "only_subgroup": [], "only_pilot": []}

    other = readout_of(world, split="dev", bridge=bridge, pilot={qids[0], qids[5]})["subgroup"]
    assert other["pilot_comparison"] == {
        "equal": False,
        "only_subgroup": [qids[3]],
        "only_pilot": [qids[5]],
    }


def test_on_test_there_is_no_pilot_comparison(test_world):
    assert readout_of(test_world)["subgroup"]["pilot_comparison"] is None


def test_a_read_list_missing_for_a_question_is_refused(test_world):
    outcomes, qids = test_world
    reads = reads_for(qids, set())
    del reads[qids[0]]
    with pytest.raises(ReadoutError, match="read"):
        build_readout(
            split="test",
            outcomes=outcomes,
            questions=questions_for(qids, "test"),
            read_lists=reads,
            costs=costs(),
            freeze=a_freeze(),
        )


# --- The qualitative sample -------------------------------------------------------------------


def test_the_sample_takes_the_first_five_by_ascending_qid_per_group(test_world):
    outcomes, qids = test_world
    dense, control, entity = (vector(SHAPE, position) for position in range(3))
    rows = list(zip(qids, dense, control, entity, strict=True))
    groups = {
        "B succeeds where A fails": sorted(q for q, d, a, b in rows if b and not a),
        "A succeeds where B fails": sorted(q for q, d, a, b in rows if a and not b),
        "B succeeds where dense fails": sorted(q for q, d, a, b in rows if b and not d),
        "dense succeeds where B fails": sorted(q for q, d, a, b in rows if d and not b),
    }
    sample = readout_of(test_world)["sample"]
    assert sample["rule"] == json.loads(json.dumps(dp.QUALITATIVE_SAMPLE_RULE))
    for group, members in groups.items():
        assert sample["groups"][group]["qids"] == members[:5]
        assert sample["groups"][group]["size"] == len(members)


def test_a_group_smaller_than_five_gives_fewer():
    successes = {
        "dense": {"q1": True, "q2": False, "q3": False},
        "hybrid-bm25": {"q1": True, "q2": True, "q3": False},
        "hybrid-entity-hop": {"q1": False, "q2": False, "q3": True},
    }
    groups = sample_groups(successes, per_group=5)
    assert groups["B succeeds where A fails"] == {"size": 1, "qids": ["q3"]}
    assert groups["A succeeds where B fails"] == {"size": 2, "qids": ["q1", "q2"]}


def test_the_sample_is_applied_on_test_only(tmp_path):
    outcomes, qids = write_outcomes(
        tmp_path, vectors_from_counts({(1, 1, 1): 4}), split="dev", freeze_digest=None
    )
    assert readout_of((outcomes, qids), split="dev", pilot=set())["sample"] is None


def test_cost_is_reported_per_system(test_world):
    cost = readout_of(test_world)["cost"]
    assert cost["hybrid-entity-hop"] == {"mean_latency_ms": 1.5, "mean_units_included": 7.0}


# --- Written once, and read by nothing on the decision path -----------------------------------


def test_the_readout_is_written_once_and_verified_on_load(test_world, tmp_path):
    readout = readout_of(test_world)
    path = save_readout(readout, tmp_path)
    assert path.name == "readout-test.json"
    assert (
        load_readout(tmp_path, "test", freeze_digest=FREEZE_DIGEST)["paired"] == readout["paired"]
    )
    with pytest.raises(ReadoutError, match="already"):
        save_readout(readout, tmp_path)
    with pytest.raises(ReadoutError, match="freeze"):
        load_readout(tmp_path, "test", freeze_digest="0" * 64)


def test_nothing_in_the_readout_is_read_by_the_decision():
    tree = ast.parse((SOURCE / "replacement_decision.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any("readout" in name for name in imported)
