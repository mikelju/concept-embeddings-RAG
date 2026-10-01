"""Phase 17, S2: `cer p17-pool` on synthetic sets. The two loaders that read the real Phase
9-16 artifacts are replaced by fixtures; everything after them (D3, the replays, the pools,
the pairs join over a corpus read by `fullwiki.load_corpus`, the manifest) is the stage's
own code."""

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from concept_embeddings_rag import cli, config
from concept_embeddings_rag.artifacts import digest_of
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question, unit_id_for
from concept_embeddings_rag.evaluation import phase17

P10A, P10B, P10C, P14 = config.PHASE_17_SYSTEMS
SETS = tuple(config.PHASE_17_SETS)


def write_corpus(directory: Path, units: list[IndexingUnit]) -> None:
    """A corpus in the format `fullwiki.load_corpus` reads (and proves) for all three sets."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        json.dumps({"unit_id": u.unit_id, "title": u.title, "sentences": list(u.sentences)}) + "\n"
        for u in units
    )
    (directory / "corpus.jsonl.gz").write_bytes(gzip.compress(lines.encode("utf-8")))
    manifest = {
        "corpus_file": "corpus.jsonl.gz",
        "ordered_unit_digest": digest_of(*(u.unit_id for u in units)),
    }
    (directory / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")


def a_unit(set_name: str, number: int) -> IndexingUnit:
    title, sentences = f"{set_name} title {number}", (f"Sentence {number} of {set_name}.",)
    return IndexingUnit(unit_id_for(title, sentences), title, sentences)


def fake_set(set_name: str, units: list[IndexingUnit], *, miss: bool = False) -> dict[str, Any]:
    """Questions, lists and counts shaped as the two real loaders return them."""
    n = config.PHASE_17_SETS[set_name]
    ids = [u.unit_id for u in units]
    questions = [
        Question(f"{set_name}-q{i}", f"question {i} of {set_name}?", "", (ids[i % 5],), (), "x")
        for i in range(n)
    ]
    lists = {
        P10A: [ids[0:3]] * n,
        P10B: [ids[2:5]] * n,
        P10C: [[ids[5], ids[0]]] * n,
        P14: [[ids[6]]] * n,
    }
    observed = {
        **config.PHASE_17_D3_SUPPORTED[set_name],
        **config.PHASE_17_REPLAYS_SUPPORTED[set_name],
    }
    if miss:
        observed[P14] -= 1
    return {
        "questions": questions,
        "token_counts": {u: 10 + k for k, u in enumerate(ids)},
        "lists": lists,
        "observed": observed,
        "extra_reasons": [],
        "evidence": {"fixture": set_name},
        "sources": {"fixture": {"path": set_name, "sha256": "0" * 64, "recorded_digest": None}},
    }


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """Synthetic corpora for the three sets and the loaders replaced; returns a runner."""
    units = {s: [a_unit(s, k) for k in range(8)] for s in SETS}
    dirs = {s: tmp_path / f"corpus-{s}" for s in SETS}
    for s in SETS:
        write_corpus(dirs[s], units[s])
    misses: set[str] = set()
    monkeypatch.setattr(
        cli,
        "_p17_hotpotqa",
        lambda *_a: fake_set(config.PHASE_17_HOTPOTQA, units[config.PHASE_17_HOTPOTQA]),
    )
    monkeypatch.setattr(
        cli,
        "_p17_recorded",
        lambda set_name, _d, _s: fake_set(set_name, units[set_name], miss=set_name in misses),
    )
    target = tmp_path / "phase17"

    def run() -> Path:
        return cli.cmd_p17_pool(
            target,
            phase9_dir=dirs[config.PHASE_17_HOTPOTQA],
            phase15_dir=dirs[config.PHASE_17_MUSIQUE],
            phase16_dir=dirs[config.PHASE_17_MULTIHOP_RAG],
        )

    return {"run": run, "target": target, "units": units, "misses": misses}


def test_the_stage_writes_integrity_pools_pairs_and_the_manifest_once(stage):
    path = stage["run"]()
    target: Path = stage["target"]

    integrity = json.loads((target / "integrity.json").read_text(encoding="utf-8"))
    assert integrity["terminal_state"] is None
    assert integrity["recorded_replays"]["passed"] is True
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["integrity_digest"] == digest_of(
        (target / "integrity.json").read_text(encoding="utf-8")
    )
    for set_name in SETS:
        block = manifest["sets"][set_name]
        # Pool per question: ids 0-6, seven units, every question the same.
        assert block["size"]["max"] == block["size"]["min"] == 7
        assert block["pairs"] == 7 * config.PHASE_17_SETS[set_name]
        assert block["longest_unit"]["budget_tokens"] == 16
        pool = phase17.read_pool(target, set_name, digest=block["pool_file"]["digest"])
        assert len(pool) == config.PHASE_17_SETS[set_name]
        assert (
            phase17.sha256_file(target / phase17.pairs_name(set_name))
            == (block["pairs_file"]["sha256"])
        )
    assert manifest["judges"] == config.PHASE_17_JUDGES

    with pytest.raises(SystemExit, match="written once"):
        stage["run"]()


def test_the_pairs_join_puts_each_units_indexable_text_beside_its_own_id(stage):
    stage["run"]()
    target: Path = stage["target"]
    manifest = json.loads((target / "pool.json").read_text(encoding="utf-8"))
    for set_name in SETS:
        by_id = {u.unit_id: u.indexable_text for u in stage["units"][set_name]}
        rows = phase17.read_pairs(
            target, set_name, digest=manifest["sets"][set_name]["pairs_file"]["digest"]
        )
        assert rows[0]["question"] == f"question 0 of {set_name}?"
        for row in rows:
            assert len(row["unit_ids"]) == len(row["texts"]) == 7
            for unit_id, text in zip(row["unit_ids"], row["texts"], strict=True):
                assert text == by_id[unit_id]
        # The pool order: best rank, then id.
        ids = [u.unit_id for u in stage["units"][set_name]]
        firsts = sorted([ids[0], ids[2], ids[5], ids[6]])
        assert rows[0]["unit_ids"][:4] == firsts


def test_a_d3_miss_writes_data_stop_exits_non_zero_and_writes_no_pool(stage):
    stage["misses"].add(config.PHASE_17_MULTIHOP_RAG)

    with pytest.raises(SystemExit, match="DATA_STOP"):
        stage["run"]()

    target: Path = stage["target"]
    integrity = json.loads((target / "integrity.json").read_text(encoding="utf-8"))
    assert integrity["terminal_state"] == "DATA_STOP"
    assert integrity["stop_reasons"] == ["multihop-rag P14: 512 of 2255, recorded 513"]
    assert sorted(p.name for p in target.iterdir()) == ["integrity.json"]


def test_a_replay_miss_outside_d3_stops_with_nothing_written(stage, monkeypatch):
    original = cli._p17_recorded

    def missing_replay(set_name, directory, source):
        data = original(set_name, directory, source)
        if set_name == config.PHASE_17_MUSIQUE:
            data["observed"][P10A] = 435
        return data

    monkeypatch.setattr(cli, "_p17_recorded", missing_replay)

    with pytest.raises(SystemExit, match="for the author"):
        stage["run"]()

    assert list(stage["target"].iterdir()) == []
