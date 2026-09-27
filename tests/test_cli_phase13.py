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


def test_screen_refuses_a_second_run(tmp_path, monkeypatch):
    write(tmp_path, cli.P13_SCREEN_NAME, {"terminal_state": None})
    monkeypatch.setattr(
        cli, "_p10_json", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p13_screen(target_dir=tmp_path)


def test_screen_refuses_without_a_window_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "_p13_phase10_dev_lists", lambda *a: ([{"qid": "q1", "dense": [["u1", 1.0]]}], "d")
    )
    monkeypatch.setattr(
        cli, "_p13_dev_questions", lambda *a: ([type("Q", (), {"qid": "q1"})()], {})
    )
    monkeypatch.setattr(
        cli, "_p11_entity_index", lambda *a: pytest.fail("no index read without a cache")
    )
    with pytest.raises(SystemExit, match="does not exist"):
        cli.cmd_p13_screen(target_dir=tmp_path)


def test_the_p13_screen_stage_is_registered():
    assert cli.build_parser().parse_args(["p13-screen"]).command == "p13-screen"
