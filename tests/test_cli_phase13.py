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


def test_lists_refuse_a_second_run(tmp_path, monkeypatch):
    write(tmp_path, cli.P13_REPRODUCTION_NAME, {"passed": True})
    monkeypatch.setattr(
        cli, "_p10_json", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p13_lists(target_dir=tmp_path)


def test_lists_refuse_when_the_lists_already_exist(tmp_path, monkeypatch):
    # A crash between the lists and the reproduction must not let a rerun rewrite the lists.
    write(tmp_path, cli.P13_DEV_LISTS_MANIFEST, {"digest": "recorded"})
    monkeypatch.setattr(
        cli, "_p10_json", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p13_lists(target_dir=tmp_path)


def test_lists_refuse_without_a_screen(tmp_path):
    with pytest.raises(SystemExit, match="screen.json does not exist"):
        cli.cmd_p13_lists(target_dir=tmp_path)


def test_lists_refuse_a_screen_stop(tmp_path, monkeypatch):
    write(tmp_path, cli.P13_SCREEN_NAME, {"passed": False, "terminal_state": "SCREEN_STOP"})
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: True)
    with pytest.raises(SystemExit, match="did not pass"):
        cli.cmd_p13_lists(target_dir=tmp_path)


def test_lists_refuse_an_uncommitted_screen(tmp_path, monkeypatch):
    write(tmp_path, cli.P13_SCREEN_NAME, {"passed": True, "terminal_state": None})
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: False)
    with pytest.raises(SystemExit, match="not committed"):
        cli.cmd_p13_lists(target_dir=tmp_path)


def test_fit_refuses_without_a_reproduction(tmp_path):
    with pytest.raises(SystemExit, match="reproduction.json does not exist"):
        cli.cmd_p13_fit(target_dir=tmp_path)


def test_fit_refuses_after_a_failed_reproduction(tmp_path):
    write(tmp_path, cli.P13_REPRODUCTION_NAME, {"passed": False})
    with pytest.raises(SystemExit, match="did not pass"):
        cli.cmd_p13_fit(target_dir=tmp_path)


def test_a_refit_refuses_to_overwrite(tmp_path, monkeypatch):
    write(tmp_path, cli.P13_REPRODUCTION_NAME, {"passed": True, "dev_lists_digest": "d"})
    write(tmp_path, cli.P13_FIT_NAME, {"terminal_state": None})
    monkeypatch.setattr(
        cli, "_p11_dev_inputs", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p13_fit(target_dir=tmp_path)
