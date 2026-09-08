"""T13: the command line interface.

Each stage runs standalone and is idempotent, and a stage whose input is missing
says so plainly instead of failing deep inside numpy.
"""

import json
import shutil
from pathlib import Path

import pytest

from concept_embeddings_rag.cli import build_parser, cmd_build, cmd_embed, cmd_evaluate
from concept_embeddings_rag.corpus.manifest import CorpusManifest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hotpot_sample.json"


def a_workspace(tmp_path: Path) -> Path:
    """A data directory holding a raw corpus and its manifest, as `fetch` leaves it."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = data_dir / "hotpot_raw.json"
    shutil.copy(FIXTURE, raw)

    CorpusManifest(
        dataset="fixture",
        source_url="http://example.invalid/x.json",
        sha256="a" * 64,
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
