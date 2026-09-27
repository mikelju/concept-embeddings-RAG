"""Phase 13 CLI: each stage writes its artifact once and refuses to run again over it.

Mirrors `tests/test_cli_phase12.py`: the refusals are checked against a toy target directory
holding only the artifact that refusal reads, before any real input (the corpus, the entity
index, the window cache) would be loaded.
"""

import json

import pytest

from concept_embeddings_rag import cli
from concept_embeddings_rag.evaluation import phase13


def write(directory, name, body):
    (directory / name).write_text(json.dumps(body), encoding="utf-8")


def test_windows_refuses_a_second_run(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache" / cli.P13_WINDOW_CACHE_DIRNAME
    cache_dir.mkdir(parents=True)
    write(cache_dir, phase13.window_manifest_filename("dev"), {"key": "k"})
    monkeypatch.setattr(
        cli, "_p10_json", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p13_windows(target_dir=tmp_path)


def test_the_p13_windows_stage_is_registered():
    parser = cli.build_parser()
    assert parser.parse_args(["p13-windows"]).command == "p13-windows"
