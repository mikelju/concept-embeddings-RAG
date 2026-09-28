"""Phase 14 CLI: `p14-lists` refuses a second run and stops on a bad input before any list.

Mirrors `tests/test_cli_phase13.py`. The input checks run over a small fixture: three units,
their vectors saved in a real `EmbeddingCache`, an `embedding.json` naming them, one dev
question, and the loaders of the real inputs replaced by that fixture.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from concept_embeddings_rag import cli, config
from concept_embeddings_rag.embeddings.cache import CachedQueryBackend, EmbeddingCache
from concept_embeddings_rag.evaluation import fullwiki as phase9

UNITS = ["u1", "u2", "u3"]


def write(directory, name, body):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(body), encoding="utf-8")


def fixture(tmp_path, monkeypatch, *, digest_ok=True, order_ok=True, p1_shift=0.0):
    """The inputs of one dev question; returns `(source_dir, target_dir)`."""
    rng = np.random.default_rng(14)
    vectors = rng.normal(size=(3, 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    question = rng.normal(size=4).astype(np.float32)
    question /= np.linalg.norm(question)
    source_dir, target_dir = tmp_path / "phase9", tmp_path / "phase14"
    saved_ids = UNITS if order_ok else UNITS[::-1]
    EmbeddingCache(source_dir / "cache").save("passages", vectors, saved_ids, {})
    write(
        source_dir,
        phase9.EMBEDDING_FILENAME,
        {
            "corpus_cache_key": "passages",
            "vectors_digest": phase9.vectors_digest(vectors) if digest_ok else "not-it",
            "question_cache_key": config.PHASE_14_DEV_QUESTION_VECTORS_KEY,
        },
    )
    p1_score = float(vectors[0] @ question) + p1_shift
    rows10 = [
        {
            "qid": "q1",
            "question": "which one?",
            "dense": [("u1", p1_score), ("u2", 0.1)],
            "bm25": [("u2", 3.0)],
            config.ENTITY_HOP_NAME: [],
        }
    ]
    monkeypatch.setattr(cli, "_p13_phase10_dev_lists", lambda *a: (rows10, "p10-lists"))
    monkeypatch.setattr(cli, "_p11_dev_inputs", lambda *a: ([SimpleNamespace(qid="q1")], {}))
    monkeypatch.setattr(
        cli,
        "_p11_entity_index",
        lambda *a: (SimpleNamespace(unit_ids=list(UNITS)), {"entity_index_digest": "e"}),
    )
    monkeypatch.setattr(
        cli,
        "_p12_dev_question_vector",
        lambda *a: CachedQueryBackend(["which one?"], question[None, :], "bge", "rev"),
    )
    return source_dir, target_dir


def no_hop(*args, **kwargs):
    raise AssertionError("the hop must not start")


def written(target_dir):
    return sorted(p.name for p in target_dir.iterdir()) if target_dir.exists() else []


def test_lists_refuse_a_second_run(tmp_path, monkeypatch):
    write(tmp_path, cli.P14_REPRODUCTION_NAME, {"passed": True})
    monkeypatch.setattr(
        cli, "_p10_json", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p14_lists(target_dir=tmp_path)


def test_lists_refuse_when_the_lists_already_exist(tmp_path, monkeypatch):
    write(tmp_path, cli.P14_DEV_LISTS_MANIFEST, {"digest": "recorded"})
    monkeypatch.setattr(
        cli, "_p10_json", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p14_lists(target_dir=tmp_path)


def test_a_vector_digest_mismatch_stops_the_stage(tmp_path, monkeypatch):
    source_dir, target_dir = fixture(tmp_path, monkeypatch, digest_ok=False)
    monkeypatch.setattr(cli, "node_weights", no_hop)
    with pytest.raises(SystemExit, match="vectors_digest"):
        cli.cmd_p14_lists(source_dir=source_dir, target_dir=target_dir)
    assert written(target_dir) == []


def test_a_unit_order_mismatch_stops_the_stage(tmp_path, monkeypatch):
    source_dir, target_dir = fixture(tmp_path, monkeypatch, order_ok=False)
    monkeypatch.setattr(cli, "node_weights", no_hop)
    with pytest.raises(SystemExit, match="order"):
        cli.cmd_p14_lists(source_dir=source_dir, target_dir=target_dir)
    assert written(target_dir) == []


def test_a_d1_miss_stops_the_stage_before_any_list(tmp_path, monkeypatch):
    source_dir, target_dir = fixture(tmp_path, monkeypatch, p1_shift=2e-5)
    monkeypatch.setattr(cli, "node_weights", no_hop)
    with pytest.raises(SystemExit, match=r"D1 .*q1"):
        cli.cmd_p14_lists(source_dir=source_dir, target_dir=target_dir)
    assert written(target_dir) == []


def test_d1_passes_on_the_recorded_score_and_the_hop_starts(tmp_path, monkeypatch):
    source_dir, target_dir = fixture(tmp_path, monkeypatch)

    class HopStarted(Exception):
        pass

    def started(*args, **kwargs):
        raise HopStarted

    monkeypatch.setattr(cli, "node_weights", started)
    with pytest.raises(HopStarted):
        cli.cmd_p14_lists(source_dir=source_dir, target_dir=target_dir)
    assert written(target_dir) == []


def test_the_p14_lists_stage_is_registered():
    assert cli.build_parser().parse_args(["p14-lists"]).command == "p14-lists"
