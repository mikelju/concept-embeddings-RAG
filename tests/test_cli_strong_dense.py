"""T13, Phase 8: the `strong-embed` and `strong-dense` stages.

Wiring and refusals only. No model is downloaded and no snapshot is fetched: the
backends and the snapshot resolver are injected, exactly as the Phase 7 stage injects
its extractor builder. What is asserted is the order the plan fixes - encode the corpus
and dev, run the gate, and only then encode and measure the held-out split - and the
two things this phase must never do: rewrite `token_counts.json`, or record a timing
for an encoding that was already on disk.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.cli import build_parser, cmd_strong_dense, cmd_strong_embed
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question, save_pool, unit_id_for
from concept_embeddings_rag.embeddings.cache import (
    question_cache_key,
    question_set_hash,
    unit_set_hash,
)
from concept_embeddings_rag.evaluation.strong_dense import Phase8Pins
from concept_embeddings_rag.nodes.index import build_node_index, save_node_index
from concept_embeddings_rag.nodes.local_extraction import record_from_forms

EXTRACTION_DIGEST = "d" * 64
INDEX_STEM = f"nodes-{EXTRACTION_DIGEST[:16]}"


class PhaseEightBackend:
    """Deterministic vectors, no model and no network, carrying the Phase 8 identity.

    It answers to the pinned model name and revision because every cache in the
    workspace is keyed by them: a backend that renamed itself would look like a
    different model to every artifact the stage writes.
    """

    name = config.PHASE_8_DENSE_MODEL
    revision = config.PHASE_8_DENSE_REVISION
    normalize = True

    def __init__(self, dim: int = 4, prompt: str = "") -> None:
        self.dim = dim
        self.resolved_query_prompt = prompt

    def encode(self, texts):
        rows = []
        for text in texts:
            digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            vector = rng.normal(size=self.dim)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype="float32")


def a_strong_workspace(tmp_path: Path):
    """A pool, token counts, a GLiNER-shaped node index over it, and the Phase 8 pins."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    units = []
    for index in range(6):
        title, sentences = f"T{index}", (f"text {index}.",)
        units.append(
            IndexingUnit(unit_id=unit_id_for(title, sentences), title=title, sentences=sentences)
        )
    # Unit ids are a content hash, so the pool decides them and the fixture reads them back.
    unit_ids = sorted(unit.unit_id for unit in units)
    units = sorted(units, key=lambda unit: unit.unit_id)
    questions = [
        Question("dev1", "a dev question", "", (unit_ids[0], unit_ids[5]), (), "dev"),
        Question("dev2", "another dev question", "", (unit_ids[1],), (), "dev"),
        Question("test1", "a held out question", "", (unit_ids[2], unit_ids[5]), (), "test"),
        Question("test2", "another held out question", "", (unit_ids[3],), (), "test"),
    ]
    save_pool(units, questions, data_dir / "pool.json")
    (data_dir / "token_counts.json").write_text(
        json.dumps(dict.fromkeys(unit_ids, 100), sort_keys=True), encoding="utf-8"
    )
    forms = {unit_ids[0]: ["Bridge"], unit_ids[5]: ["Bridge"]}
    records = {
        uid: record_from_forms(
            uid, forms.get(uid, []), model="fixture", configuration_digest="f" * 16
        )
        for uid in unit_ids
    }
    index = build_node_index(records, unit_ids, extraction_digest=EXTRACTION_DIGEST)
    gliner_dir = tmp_path / "phase7" / "gliner"
    save_node_index(index, gliner_dir)
    pins = replace(
        Phase8Pins.from_config(),
        dim=4,
        unit_set_hash=unit_set_hash(unit_ids),
        index_digest=index.digest,
        extraction_digest=EXTRACTION_DIGEST,
        # A toy pool of two dev questions cannot land on 487 of 600; the real bar is
        # exercised where it belongs, against the stop rule itself.
        dense_dev_share=0.0,
    )
    arguments = {
        "data_dir": data_dir,
        "cache_dir": tmp_path / "cache",
        "question_cache_dir": tmp_path / "questions",
        "phase_8_dir": tmp_path / "phase8",
        "pins": pins,
    }
    return data_dir, gliner_dir, pins, arguments


def a_snapshot(tmp_path: Path, revision: str):
    """A resolver returning a local directory named after the commit it claims to serve."""
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / config.PHASE_8_WEIGHTS_FILE).write_bytes(b"weights")

    def resolve(name: str, wanted: str) -> Path:
        return snapshot

    return resolve


def strong_embed(tmp_path: Path, arguments: dict, **overrides):
    call = {
        **arguments,
        "corpus_backend": PhaseEightBackend(),
        "query_backend": PhaseEightBackend(prompt=arguments["pins"].query_prompt),
        "resolve_snapshot": a_snapshot(tmp_path, arguments["pins"].revision),
    }
    call.update(overrides)
    return cmd_strong_embed(**call)


def strong_dense(gliner_dir: Path, arguments: dict, **overrides):
    call = {
        **arguments,
        "gliner_dir": gliner_dir,
        "backend": PhaseEightBackend(prompt=arguments["pins"].query_prompt),
    }
    call.update(overrides)
    return cmd_strong_dense(**call)


def test_parser_exposes_both_phase_8_stages_and_their_gated_modes():
    parser = build_parser()

    assert parser.parse_args(["strong-embed"]).test is False
    assert parser.parse_args(["strong-embed", "--test"]).test is True
    assert parser.parse_args(["strong-embed"]).hourly_rate_usd == 0.0
    assert parser.parse_args(["strong-embed", "--hourly-rate-usd", "0.74"]).hourly_rate_usd == 0.74
    assert parser.parse_args(["strong-dense"]).test is False
    assert parser.parse_args(["strong-dense", "--test"]).test is True
    with pytest.raises(SystemExit):
        parser.parse_args(["strong-dense", "--hourly-rate-usd", "1.0"])


def test_strong_embed_without_a_pool_names_the_build_stage(tmp_path):
    with pytest.raises(SystemExit, match="build"):
        cmd_strong_embed(data_dir=tmp_path / "data", phase_8_dir=tmp_path / "phase8")


@pytest.mark.parametrize("rate", [-1.0, float("nan"), float("inf")])
def test_strong_embed_rejects_an_impossible_machine_rate(tmp_path, rate):
    with pytest.raises(SystemExit, match="finite and non-negative"):
        cmd_strong_embed(data_dir=tmp_path / "data", hourly_rate_usd=rate)


def test_strong_embed_writes_both_caches_and_one_artifact_and_prints_ascii(tmp_path, capsys):
    data_dir, _, pins, arguments = a_strong_workspace(tmp_path)
    before = (data_dir / "token_counts.json").read_bytes()

    path = strong_embed(tmp_path, arguments, hourly_rate_usd=0.74)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "embedding.json"
    assert payload["dim"] == 4
    assert payload["query_prompt"] == pins.query_prompt
    assert payload["weights_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert payload["n_questions"] == 2
    assert payload["embedding_usd"] > 0.0
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
    assert len(list((tmp_path / "questions").glob("*.npz"))) == 1
    # R1: the ruler does not move when the Dense model does.
    assert (data_dir / "token_counts.json").read_bytes() == before
    out = capsys.readouterr().out
    assert out.isascii()
    assert "provenance verified" in out


def test_strong_embed_encodes_documents_without_the_prompt_and_queries_with_it(tmp_path):
    _, _, pins, arguments = a_strong_workspace(tmp_path)

    payload = json.loads(strong_embed(tmp_path, arguments).read_text(encoding="utf-8"))

    question_set = question_set_hash(["dev1", "dev2"])
    prompted = question_cache_key(
        pins.model, pins.revision, question_set, "dev", True, pins.query_prompt
    )
    unprompted = question_cache_key(pins.model, pins.revision, question_set, "dev", True)
    assert payload["question_cache_key"] == prompted
    assert prompted != unprompted
    sidecar = json.loads(
        (tmp_path / "questions" / f"embeddings-{prompted}.json").read_text(encoding="utf-8")
    )
    assert sidecar["query_prompt"] == pins.query_prompt
    corpus_sidecar = json.loads(
        (tmp_path / "cache" / f"embeddings-{payload['corpus_cache_key']}.json").read_text("utf-8")
    )
    assert "query_prompt" not in corpus_sidecar


def test_strong_embed_refuses_a_revision_the_resolver_did_not_serve(tmp_path):
    _, _, _, arguments = a_strong_workspace(tmp_path)

    with pytest.raises(SystemExit, match="pinned to"):
        strong_embed(tmp_path, arguments, resolve_snapshot=a_snapshot(tmp_path, "f" * 40))

    assert not (tmp_path / "phase8" / "embedding.json").exists()
    assert not list((tmp_path / "cache").glob("*.npz")) if (tmp_path / "cache").exists() else True


def test_strong_embed_refuses_a_snapshot_without_the_declared_weight_file(tmp_path):
    _, _, pins, arguments = a_strong_workspace(tmp_path)
    empty = tmp_path / "empty" / pins.revision
    empty.mkdir(parents=True)

    with pytest.raises(SystemExit, match="weight digest"):
        strong_embed(tmp_path, arguments, resolve_snapshot=lambda name, wanted: empty)


def test_strong_embed_refuses_to_time_an_encoding_that_is_already_cached(tmp_path):
    _, _, _, arguments = a_strong_workspace(tmp_path)
    strong_embed(tmp_path, arguments)
    (tmp_path / "phase8" / "embedding.json").unlink()

    with pytest.raises(SystemExit, match="already exists"):
        strong_embed(tmp_path, arguments)


def test_a_second_strong_embed_refuses_rather_than_rewriting_the_artifact(tmp_path):
    _, _, _, arguments = a_strong_workspace(tmp_path)
    strong_embed(tmp_path, arguments)

    with pytest.raises(SystemExit, match="written once"):
        strong_embed(tmp_path, arguments)


def test_strong_embed_test_refuses_before_the_dev_gate_has_run(tmp_path):
    _, _, _, arguments = a_strong_workspace(tmp_path)
    strong_embed(tmp_path, arguments)

    with pytest.raises(SystemExit, match="strong-dense"):
        strong_embed(tmp_path, arguments, test=True)

    assert not (tmp_path / "phase8" / "embedding-test.json").exists()
    assert len(list((tmp_path / "questions").glob("*.npz"))) == 1


def test_strong_dense_without_the_phase_8_caches_names_the_embedding_stage(tmp_path):
    _, gliner_dir, _, arguments = a_strong_workspace(tmp_path)

    with pytest.raises(SystemExit, match="strong-embed"):
        strong_dense(gliner_dir, arguments)


def test_the_phase_8_stages_run_dev_then_the_gate_then_the_held_out_split(tmp_path, capsys):
    data_dir, gliner_dir, _, arguments = a_strong_workspace(tmp_path)
    strong_embed(tmp_path, arguments)
    before = (data_dir / "token_counts.json").read_bytes()

    dev_path = strong_dense(gliner_dir, arguments)

    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    assert dev["split"] == "dev"
    assert dev["stop_rule"]["verdict"] == "pass"
    assert sorted(dev["systems"]) == sorted(config.PHASE_8_SYSTEMS)
    assert dev["systems"]["dense"]["config"]["tokenizer"] == config.BUDGET_TOKENIZER_ID

    encoding = strong_embed(tmp_path, arguments, test=True)
    assert encoding.name == "embedding-test.json"
    assert len(list((tmp_path / "questions").glob("*.npz"))) == 2

    test_path = strong_dense(gliner_dir, arguments, test=True)

    held_out = json.loads(test_path.read_text(encoding="utf-8"))
    assert held_out["split"] == "test"
    assert held_out["dev_digest"] == dev["digest"]
    assert held_out["systems"]["hybrid-entity-hop"]["fit"]["fitted_on"] == "dev"
    assert "surviving_gain" in held_out
    # Phase 7's directory is read and never written to, and the ruler never moved.
    assert sorted(path.name for path in gliner_dir.iterdir()) == [
        f"{INDEX_STEM}.json",
        f"{INDEX_STEM}.npz",
    ]
    assert (data_dir / "token_counts.json").read_bytes() == before
    assert capsys.readouterr().out.isascii()


def test_strong_dense_test_refuses_while_the_held_out_questions_are_unencoded(tmp_path):
    _, gliner_dir, _, arguments = a_strong_workspace(tmp_path)
    strong_embed(tmp_path, arguments)
    strong_dense(gliner_dir, arguments)

    with pytest.raises(SystemExit, match="strong-embed --test"):
        strong_dense(gliner_dir, arguments, test=True)


def test_strong_dense_refuses_a_node_index_that_is_not_the_pinned_one(tmp_path):
    _, gliner_dir, pins, arguments = a_strong_workspace(tmp_path)
    strong_embed(tmp_path, arguments)
    arguments["pins"] = replace(pins, index_digest="0" * 64)

    with pytest.raises(SystemExit, match="inherited, never rebuilt"):
        strong_dense(gliner_dir, arguments)
