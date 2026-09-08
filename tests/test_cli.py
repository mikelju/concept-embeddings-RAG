"""T13: the command line interface.

Each stage runs standalone and is idempotent, and a stage whose input is missing
says so plainly instead of failing deep inside numpy.
"""

import json
import shutil
from pathlib import Path

import pytest

from concept_embeddings_rag.cli import build_parser, cmd_build, cmd_embed, cmd_evaluate
from concept_embeddings_rag.corpus.download import sha256_of_file
from concept_embeddings_rag.corpus.manifest import CorpusManifest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hotpot_sample.json"


def a_workspace(tmp_path: Path) -> Path:
    """A data directory holding a raw corpus and its manifest, as `fetch` leaves it."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = data_dir / "hotpot_raw.json"
    shutil.copy(FIXTURE, raw)

    # The real digest, as `fetch` records it: a placeholder would make every
    # verification downstream vacuous, which is the bug SEC-001 was about.
    CorpusManifest(
        dataset="fixture",
        source_url="http://example.invalid/x.json",
        sha256=sha256_of_file(raw),
        downloaded_at="2026-09-07T10:00:00",
        seed=42,
        n_questions=4,
        split_sizes={"dev": 2, "test": 2},
    ).save(data_dir / "manifest.json")
    return data_dir


def test_parser_exposes_the_four_stages():
    parser = build_parser()
    for stage in ("fetch", "build", "embed", "evaluate"):
        assert parser.parse_args([stage]).command == stage


def test_build_creates_a_pool_from_the_frozen_corpus(tmp_path):
    data_dir = a_workspace(tmp_path)
    cmd_build(data_dir=data_dir, n_questions=4, n_dev=2, seed=42)

    pool = json.loads((data_dir / "pool.json").read_text(encoding="utf-8"))
    assert len(pool["questions"]) == 4
    assert pool["units"]
    assert {q["split"] for q in pool["questions"]} == {"dev", "test"}


def test_build_is_idempotent(tmp_path):
    data_dir = a_workspace(tmp_path)
    cmd_build(data_dir=data_dir, n_questions=4, n_dev=2, seed=42)
    first = (data_dir / "pool.json").read_text(encoding="utf-8")
    cmd_build(data_dir=data_dir, n_questions=4, n_dev=2, seed=42)
    second = (data_dir / "pool.json").read_text(encoding="utf-8")

    assert first == second


def test_build_records_pool_size_in_the_manifest(tmp_path):
    data_dir = a_workspace(tmp_path)
    cmd_build(data_dir=data_dir, n_questions=4, n_dev=2, seed=42)

    manifest = CorpusManifest.load(data_dir / "manifest.json")
    assert manifest.n_units and manifest.n_units > 0
    assert manifest.unit_set_hash


def test_embed_without_a_pool_fails_with_a_clear_message(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        cmd_embed(data_dir=data_dir)

    assert "build" in str(excinfo.value)


def test_evaluate_without_embeddings_fails_with_a_clear_message(tmp_path):
    data_dir = a_workspace(tmp_path)
    cmd_build(data_dir=data_dir, n_questions=4, n_dev=2, seed=42)

    with pytest.raises(SystemExit) as excinfo:
        cmd_evaluate(data_dir=data_dir, results_dir=tmp_path / "results")

    assert "embed" in str(excinfo.value)


def test_build_refuses_a_corpus_that_does_not_match_the_manifest_hash(tmp_path):
    """SEC-001: the hash was recorded but never compared, so a tampered corpus was
    absorbed silently and the manifest was rewritten to certify it."""
    data_dir = a_workspace(tmp_path)
    raw = data_dir / "hotpot_raw.json"
    raw.write_text(raw.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        cmd_build(data_dir=data_dir, n_questions=4, n_dev=2, seed=42)

    assert "manifest" in str(excinfo.value)


def test_fetch_refuses_a_corpus_that_does_not_match_an_existing_manifest(tmp_path):
    """SEC-001: a second run must verify, not re-describe whatever is on disk."""
    from concept_embeddings_rag.cli import cmd_fetch
    from concept_embeddings_rag.corpus.download import CorpusIntegrityError

    data_dir = a_workspace(tmp_path)
    raw = data_dir / "hotpot_raw.json"
    raw.write_text(raw.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(CorpusIntegrityError, match="manifest expects"):
        cmd_fetch(data_dir=data_dir)


def test_fetch_accepts_a_corpus_that_matches_its_manifest(tmp_path):
    """The check must not reject the corpus the pipeline itself froze."""
    from concept_embeddings_rag.cli import cmd_fetch

    data_dir = a_workspace(tmp_path)
    assert cmd_fetch(data_dir=data_dir) == data_dir / "hotpot_raw.json"


def test_embed_refuses_a_pool_the_manifest_does_not_recognise(tmp_path):
    """SEC-003: `unit_set_hash` has been recorded since the corpus was frozen and
    was never read back, so a pool from a different corpus loaded without complaint."""
    data_dir = a_workspace(tmp_path)
    cmd_build(data_dir=data_dir, n_questions=4, n_dev=2, seed=42)

    manifest = CorpusManifest.load(data_dir / "manifest.json")
    CorpusManifest(
        dataset=manifest.dataset,
        source_url=manifest.source_url,
        sha256=manifest.sha256,
        downloaded_at=manifest.downloaded_at,
        seed=manifest.seed,
        n_questions=manifest.n_questions,
        split_sizes=manifest.split_sizes,
        n_units=manifest.n_units,
        unit_set_hash="0" * 16,
    ).save(data_dir / "manifest.json")

    with pytest.raises(SystemExit) as excinfo:
        cmd_embed(data_dir=data_dir)

    assert "build" in str(excinfo.value)


def test_fetch_refuses_a_reassembled_corpus_that_contradicts_the_manifest(tmp_path, monkeypatch):
    """SEC-009: deleting the corpus while keeping its manifest sent the run down the
    assemble branch, which accepted whatever came back. Both branches verify now."""
    from concept_embeddings_rag import cli as cli_module
    from concept_embeddings_rag.corpus.download import CorpusIntegrityError

    data_dir = a_workspace(tmp_path)
    (data_dir / "hotpot_raw.json").unlink()

    monkeypatch.setattr(
        cli_module.hf_source, "fetch_split", lambda **kwargs: [{"_id": "different"}]
    )

    with pytest.raises(CorpusIntegrityError, match="manifest expects"):
        cli_module.cmd_fetch(data_dir=data_dir)
