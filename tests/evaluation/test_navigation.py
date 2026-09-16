"""T9 of Phase 5: the navigation run over the pilot (HU-5, HU-6).

A twelve-paragraph pool on twelve axes, two pilot questions, and a fake BM25, so every
figure below can be checked by hand. Read depth 2 and depths (1, 3) stand in for the real
10 and (10, 100): the arithmetic is the same.

- qA reads u00, u01 and misses two gold paragraphs, u05 and u07. From p1 = u00, the entity
  `Ada` leads to u07 and the concept `loom` leads to u05.
- qB reads u01, u00 and misses one, u09, which shares `Bob` and `sail` with p1 = u01. Its
  other gold paragraph, u01, was read: it is the subgroup with exactly one gold read.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.navigation import (
    ARM_HOPS,
    REFERENCE_HOP,
    NavigationError,
    load_hop_run,
    load_traces,
    run_navigation,
    save_navigation,
)
from concept_embeddings_rag.evaluation.pilot import PilotError, build_pilot
from concept_embeddings_rag.evaluation.second_hop import BM25_QUESTION
from concept_embeddings_rag.nodes.extraction import OK, PROMPT_DIGEST, ExtractionRecord
from concept_embeddings_rag.nodes.index import build_node_index

UNIT_IDS = [f"u{index:02d}" for index in range(12)]
TEXTS = [f"text of {unit_id}" for unit_id in UNIT_IDS]
READ_DEPTH = 2
DEPTHS = (1, 3)
E, C, EC = (ARM_HOPS[arm] for arm in config.NAVIGATION_ARMS)


def weights(*ranked: int) -> np.ndarray:
    vector = np.zeros(12, dtype=np.float32)
    for position, row in enumerate(ranked):
        vector[row] = 10.0 - position
    return vector


class Encoder:
    vectors = {"question A": weights(0, 1), "question B": weights(1, 0)}

    def encode(self, texts):
        return np.stack([self.vectors[text] for text in texts])


class FakeBM25:
    rankings = {"question A": [("u02", 1.0)], "question B": [("u09", 2.0)]}

    def retrieve(self, query, top_k):
        return self.rankings.get(query, [])[:top_k]


def a_question(qid, text, gold, split="dev"):
    return Question(
        qid=qid,
        question=text,
        answer="-",
        gold_unit_ids=tuple(gold),
        supporting_facts=tuple((unit, 0) for unit in gold),
        split=split,
    )


QUESTIONS = [
    a_question("qA", "question A", ["u05", "u07"]),
    a_question("qB", "question B", ["u01", "u09"]),
]


def record(unit_id, entities=(), concepts=()):
    return ExtractionRecord(
        unit_id,
        config.EXTRACTION_MODEL,
        PROMPT_DIGEST,
        OK,
        tuple(entities),
        tuple(concepts),
        None,
        1,
        1,
    )


def the_index():
    records = {unit_id: record(unit_id) for unit_id in UNIT_IDS}
    records["u00"] = record("u00", ["Ada"], ["loom"])
    records["u07"] = record("u07", ["Ada"], [])
    records["u05"] = record("u05", [], ["loom"])
    records["u01"] = record("u01", ["Bob"], ["sail"])
    records["u09"] = record("u09", ["Bob"], ["sail"])
    return build_node_index(records, UNIT_IDS, extraction_digest="toy-extraction")


def the_pilot(questions=QUESTIONS):
    return build_pilot(
        [q for q in questions if q.split == "dev"],
        unit_ids=UNIT_IDS,
        vectors=np.eye(12, dtype=np.float32),
        query_backend=Encoder(),
        model="toy-model",
        revision="toy-revision",
        unit_set_hash="toy-pool",
        read_depth=READ_DEPTH,
    )


def navigate(pilot=None, questions=QUESTIONS):
    return run_navigation(
        pilot if pilot is not None else the_pilot(),
        {question.qid: question for question in questions},
        unit_ids=UNIT_IDS,
        texts=TEXTS,
        vectors=np.eye(12, dtype=np.float32),
        query_backend=Encoder(),
        bm25=FakeBM25(),
        index=the_index(),
        concept_matrix=sparse.csr_matrix((12, 3)),
        concept_weights=np.ones(3),
        provenance={"pilot_digest": "toy-pilot"},
        depths=DEPTHS,
    )


def per_question(run, qid):
    return next(entry for entry in run.payload["per_question"] if entry["qid"] == qid)


def test_a_questions_score_is_the_share_of_its_missing_paragraphs_found():
    run = navigate()
    qa = per_question(run, "qA")

    assert qa["n_missing"] == 2
    assert qa["scores"][E] == {"1": 0.5, "3": 0.5}
    assert qa["scores"][C] == {"1": 0.5, "3": 0.5}
    assert qa["scores"][EC] == {"1": 0.5, "3": 1.0}
    assert per_question(run, "qB")["scores"][E] == {"1": 1.0, "3": 1.0}


def test_rates_are_reported_per_question_and_per_paragraph():
    aggregates = navigate().payload["aggregates"][E]

    assert aggregates["per_question"] == {"1": 0.75, "3": 0.75}
    assert aggregates["per_paragraph"] == {"1": pytest.approx(2 / 3), "3": pytest.approx(2 / 3)}


def test_every_hop_of_the_protocol_is_measured():
    hops = set(navigate().payload["aggregates"])

    assert {BM25_QUESTION, REFERENCE_HOP, E, C, EC} <= hops
    assert len(hops) == 4 + 1 + 3


def test_new_relevant_is_found_by_the_arm_and_by_neither_dense_nor_bm25_on_the_question():
    run = navigate()

    # qA: u07 and u05 lie outside dense's read + continued list and BM25's answer.
    assert per_question(run, "qA")["new_relevant"][E] == ["u07"]
    assert per_question(run, "qA")["new_relevant"][C] == ["u05"]
    # qB: u09 is found by the arm at rank 1, but BM25 on the question already had it.
    assert per_question(run, "qB")["new_relevant"][E] == []
    assert run.payload["aggregates"][E]["new_relevant"] == 1


def test_concentration_counts_positive_candidates_per_arm():
    run = navigate()

    assert per_question(run, "qA")["positives"][EC] == 2
    assert run.payload["aggregates"][EC]["positive_candidates"]["mean"] == pytest.approx(1.5)


def test_the_subgroup_with_exactly_one_gold_read_is_reported_and_marked():
    run = navigate()

    assert per_question(run, "qA")["exactly_one_gold_read"] is False
    assert per_question(run, "qB")["exactly_one_gold_read"] is True
    assert run.payload["aggregates"][E]["subgroup_per_question"] == {"1": 1.0, "3": 1.0}


def test_every_found_paragraph_within_the_deepest_depth_has_a_trace():
    run = navigate()

    found = {(t["qid"], t["hop"], t["unit_id"]) for t in run.traces}
    assert found == {
        ("qA", E, "u07"),
        ("qA", C, "u05"),
        ("qA", EC, "u05"),
        ("qA", EC, "u07"),
        ("qB", E, "u09"),
        ("qB", C, "u09"),
        ("qB", EC, "u09"),
    }
    ec_u07 = next(t for t in run.traces if (t["qid"], t["hop"], t["unit_id"]) == ("qA", EC, "u07"))
    assert ec_u07["rank"] == 2
    assert [node["form"] for node in ec_u07["nodes"]] == ["ada"]
    assert sum(node["weight"] for node in ec_u07["nodes"]) == pytest.approx(ec_u07["score"])


def test_the_run_and_its_traces_are_saved_bound_and_verified(tmp_path: Path):
    run = navigate()

    save_navigation(run, tmp_path)
    loaded = load_hop_run(tmp_path)
    traces = load_traces(tmp_path, hop_run_digest=loaded["digest"])

    assert loaded["per_question"] == run.payload["per_question"]
    assert len(traces) == 7


def test_a_modified_hop_run_is_refused(tmp_path: Path):
    save_navigation(navigate(), tmp_path)
    path = tmp_path / "hop-run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["per_question"][0]["scores"][E]["1"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NavigationError, match="digest"):
        load_hop_run(tmp_path)


def test_traces_bound_to_another_run_are_refused(tmp_path: Path):
    save_navigation(navigate(), tmp_path)

    with pytest.raises(NavigationError, match="run"):
        load_traces(tmp_path, hop_run_digest="another-run")


def test_a_second_run_refuses_rather_than_overwriting(tmp_path: Path):
    save_navigation(navigate(), tmp_path)

    with pytest.raises(NavigationError, match="already"):
        save_navigation(navigate(), tmp_path)


def test_a_test_split_question_in_the_pilot_is_refused():
    pilot = the_pilot()
    relabelled = [
        a_question("qA", "question A", ["u05", "u07"], split="test"),
        QUESTIONS[1],
    ]

    with pytest.raises(PilotError):
        navigate(pilot=pilot, questions=relabelled)


def test_a_pilot_whose_read_list_no_longer_matches_the_embeddings_is_refused():
    pilot = the_pilot()
    shifted = [a_question("qA", "question B", ["u05", "u07"]), QUESTIONS[1]]

    with pytest.raises(NavigationError, match="read"):
        navigate(pilot=pilot, questions=shifted)


def test_the_continuity_check_names_every_rate_that_departs_from_the_diagnostic():
    from concept_embeddings_rag.evaluation.navigation import diagnostic_mismatches
    from concept_embeddings_rag.evaluation.second_hop import BASELINE_HOPS

    payload = navigate().payload
    diagnostic = {"hit_rates": {}}
    for hop in BASELINE_HOPS:
        rates = payload["aggregates"][hop]["per_paragraph"]
        diagnostic["hit_rates"][hop] = {f"all@{d}": rates[str(d)] for d in DEPTHS}
    reference = payload["aggregates"][REFERENCE_HOP]["per_paragraph"]
    diagnostic["hit_rates"][f"{REFERENCE_HOP}/m=all"] = {
        f"all@{d}": reference[str(d)] for d in DEPTHS
    }

    assert diagnostic_mismatches(payload, diagnostic) == []

    diagnostic["hit_rates"][BM25_QUESTION]["all@1"] += 0.01
    mismatches = diagnostic_mismatches(payload, diagnostic)
    assert len(mismatches) == 1 and BM25_QUESTION in mismatches[0]


def test_navigate_without_a_pilot_names_the_pilot_stage(tmp_path: Path):
    from concept_embeddings_rag.cli import cmd_navigate
    from concept_embeddings_rag.corpus.pool import IndexingUnit, save_pool, unit_id_for

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    unit = IndexingUnit(unit_id_for("T", ("S.",)), "T", ("S.",))
    question = Question("q0", "Unused?", "-", (unit.unit_id,), (("T", 0),), "dev")
    save_pool([unit], [question], data_dir / "pool.json")

    with pytest.raises(SystemExit) as excinfo:
        cmd_navigate(
            data_dir=data_dir, pilot_dir=tmp_path / "pilot", navigation_dir=tmp_path / "nav"
        )

    assert "pilot" in str(excinfo.value)


def test_navigate_refuses_to_measure_twice_before_doing_any_work(tmp_path: Path):
    from concept_embeddings_rag.cli import cmd_navigate

    navigation_dir = tmp_path / "nav"
    navigation_dir.mkdir()
    (navigation_dir / "hop-run.json").write_text("{}", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        cmd_navigate(data_dir=data_dir, navigation_dir=navigation_dir)

    assert "already" in str(excinfo.value)
