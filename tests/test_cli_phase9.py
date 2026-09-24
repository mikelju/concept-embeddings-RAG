"""Phase 9 CLI: the outcome of a stop, the single pass's start marker, and the probe's one load.

Three regressions a code review found in `cli.py`, each held here against a toy world whose
components are built in memory and handed to the stages through the loader they call:

1. a stop recorded by the questions or the probe is written as the outcome without reading
   run files that the stop means never existed;
2. the pass is marked started only after every input loaded, so a load failure is retryable,
   and once marked it is never started again;
3. the probe loads the components once, and its column-wise Entity Hop is built from them,
   ranking exactly as the reference does.
"""

import json

import pytest

from concept_embeddings_rag import cli, config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation.second_hop import node_weights
from concept_embeddings_rag.nodes.index import build_node_index
from concept_embeddings_rag.nodes.local_extraction import record_from_forms

IDS = [f"u{index:03d}" for index in range(120)]


class DenseFixture:
    """Dense returns the pool in id order, rotated by the question's number."""

    name = "dense"

    def retrieve(self, query, top_k):
        shift = int(query.split()[-1]) % 7
        ids = IDS[shift:] + IDS[:shift]
        return [(uid, 1.0 - position / 120) for position, uid in enumerate(ids[:top_k])]


class StubBM25:
    name = "bm25"

    def retrieve(self, query, top_k):
        return [(uid, float(120 - position)) for position, uid in enumerate(IDS[:top_k])]


def components() -> dict:
    forms = {
        "u000": ["Bridge", "Common"],
        "u003": ["Common", "Third"],
        "u118": ["Common"],
        "u119": ["Bridge", "Common", "Third"],
    }
    records = {
        uid: record_from_forms(uid, forms.get(uid, []), model="fixture", configuration_digest="f")
        for uid in IDS
    }
    index = build_node_index(records, IDS, extraction_digest="e" * 64)
    questions = [
        Question(
            f"q{number:03d}",
            f"question {number}",
            "",
            ("u000", "u119"),
            (),
            config.PHASE_9_HISTORICAL_OVERLAP if number % 2 else config.PHASE_9_RETRIEVAL_UNSEEN,
        )
        for number in range(100)
    ]
    return {
        "dense": DenseFixture(),
        "bm25": StubBM25(),
        "index": index,
        "node_weights": node_weights(index),
        "weights": phase9.read_frozen_weights(),
        "questions": questions,
        "token_counts": dict.fromkeys(IDS, 100),
        "provenance": {"corpus_unit_set_hash": "h", "question_digest": "q"},
    }


# --- 1. A stop is the whole outcome -----------------------------------------------------


def test_an_operational_stop_in_the_probe_is_written_without_any_run_file(tmp_path):
    (tmp_path / phase9.QUESTIONS_FILENAME).write_text(
        json.dumps({"terminal_state": None}), encoding="utf-8"
    )
    (tmp_path / phase9.PROBE_FILENAME).write_text(
        json.dumps({"terminal_state": phase9.OPERATIONAL_STOP, "over_ceiling": ["hybrid-bm25"]}),
        encoding="utf-8",
    )

    path = cli.cmd_fullwiki_outcome(target_dir=tmp_path)

    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["terminal_state"] == phase9.OPERATIONAL_STOP
    assert "hybrid-bm25" in body["stop_reasons"][0]
    assert "wins" not in body
    assert not list(tmp_path.glob("run-*.json"))


def test_a_data_stop_in_the_questions_is_written_without_a_probe_or_runs(tmp_path):
    (tmp_path / phase9.QUESTIONS_FILENAME).write_text(
        json.dumps(
            {
                "terminal_state": phase9.DATA_STOP,
                "questions_with_unresolved_gold": 80,
                "unresolved_ceiling": 74,
            }
        ),
        encoding="utf-8",
    )

    path = cli.cmd_fullwiki_outcome(target_dir=tmp_path)

    assert json.loads(path.read_text(encoding="utf-8"))["terminal_state"] == phase9.DATA_STOP


# --- 2. The pass is marked started only once its inputs are loaded -----------------------


def a_probe(tmp_path) -> None:
    (tmp_path / phase9.PROBE_FILENAME).write_text(
        json.dumps(
            {"terminal_state": None, "digest": "p", "hop_implementation": phase9.REFERENCE_HOP}
        ),
        encoding="utf-8",
    )


def test_a_load_failure_leaves_the_pass_unmarked_and_retryable(tmp_path, monkeypatch):
    a_probe(tmp_path)

    def failing(*_args, **_kwargs):
        cli._die("the FullWiki vectors do not match the digest embedding.json records")

    monkeypatch.setattr(cli, "_phase_9_inputs", failing)
    with pytest.raises(SystemExit):
        cli.cmd_fullwiki_eval(authorized=True, target_dir=tmp_path)
    assert not (tmp_path / phase9.EVALUATION_MARKER).exists()

    monkeypatch.setattr(cli, "_phase_9_inputs", lambda *_args, **_kwargs: components())
    written = cli.cmd_fullwiki_eval(authorized=True, target_dir=tmp_path)

    assert [path.name for path in written] == [
        f"run-{name}.json" for name in config.PHASE_9_SYSTEMS
    ]
    assert (tmp_path / phase9.EVALUATION_MARKER).exists()


def test_once_marked_the_pass_is_never_started_again(tmp_path, monkeypatch):
    a_probe(tmp_path)
    loads: list[int] = []

    def loader(*_args, **_kwargs):
        loads.append(1)
        return components()

    monkeypatch.setattr(cli, "_phase_9_inputs", loader)
    cli.cmd_fullwiki_eval(authorized=True, target_dir=tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.glob("run-*.json")}

    with pytest.raises(SystemExit):
        cli.cmd_fullwiki_eval(authorized=True, target_dir=tmp_path)

    assert loads == [1]
    assert {path.name: path.read_bytes() for path in tmp_path.glob("run-*.json")} == before


# --- 3. The probe loads once, and its column-wise hop ranks as the reference does ---------


def test_the_columnwise_probe_loads_the_components_once(tmp_path, monkeypatch):
    world = components()
    loads: list[int] = []

    def loader(*_args, **_kwargs):
        loads.append(1)
        return world

    monkeypatch.setattr(cli, "_phase_9_inputs", loader)
    monkeypatch.setattr(
        cli, "_historical_dev_qids", lambda _data_dir: {q.qid for q in world["questions"]}
    )
    monkeypatch.setattr(phase9, "hop_implementation", lambda _mean: phase9.COLUMNWISE_HOP)

    path = cli.cmd_fullwiki_probe(data_dir=tmp_path, target_dir=tmp_path)

    body = json.loads(path.read_text(encoding="utf-8"))
    assert loads == [1]
    assert body["hop_implementation"] == phase9.COLUMNWISE_HOP
    assert body["provenance"]["hop_implementation"] == phase9.COLUMNWISE_HOP


def test_the_columnwise_system_built_from_loaded_components_ranks_as_the_reference():
    world = components()
    reference = cli._phase_9_systems(world, phase9.REFERENCE_HOP)["hybrid-entity-hop"]
    columnwise = cli._phase_9_systems(world, phase9.COLUMNWISE_HOP)["hybrid-entity-hop"]

    bridged = 0
    for question in world["questions"]:
        expected = reference.retrieve(question.question, config.PHASE_9_RANKING_DEPTH)
        assert columnwise.retrieve(question.question, config.PHASE_9_RANKING_DEPTH) == expected
        bridged += any(unit_id == "u119" for unit_id, _score in expected)
    # The hop reached the bridge outside Dense's top 100 for the questions whose p1 is u000.
    assert bridged > 0
