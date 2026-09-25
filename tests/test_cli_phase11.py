"""Phase 11 CLI: the held-out pass opens test-11 once, only on a committed fit, never test-10.

The refusals come before any input is loaded, so each is checked against a toy target
directory holding only the artifacts that refusal reads.
"""

import json

import pytest

from concept_embeddings_rag import cli, config
from concept_embeddings_rag.evaluation import phase10


def write(directory, name, body):
    (directory / name).write_text(json.dumps(body), encoding="utf-8")


def test_the_pass_refuses_without_the_authorization_flag(tmp_path):
    with pytest.raises(SystemExit, match="authorized-pass"):
        cli.cmd_p11_eval(authorized=False, target_dir=tmp_path)


def test_the_pass_refuses_without_a_fit(tmp_path):
    with pytest.raises(SystemExit, match="fit.json does not exist"):
        cli.cmd_p11_eval(authorized=True, target_dir=tmp_path)


def test_the_pass_refuses_after_a_dev_stop(tmp_path):
    write(tmp_path, cli.P11_FIT_NAME, {"terminal_state": "DEV_STOP"})
    with pytest.raises(SystemExit, match="DEV_STOP"):
        cli.cmd_p11_eval(authorized=True, target_dir=tmp_path)


def test_the_pass_refuses_an_uncommitted_fit(tmp_path, monkeypatch):
    write(tmp_path, cli.P11_FIT_NAME, {"terminal_state": None})
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: False)
    with pytest.raises(SystemExit, match="not committed"):
        cli.cmd_p11_eval(authorized=True, target_dir=tmp_path)


def test_the_pass_refuses_a_second_start(tmp_path, monkeypatch):
    write(tmp_path, cli.P11_FIT_NAME, {"terminal_state": None})
    write(tmp_path, cli.P11_MARKER_NAME, {"started_at": "earlier"})
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: True)
    monkeypatch.setattr(
        cli, "_phase_9_inputs", lambda *a, **k: pytest.fail("inputs loaded after a start")
    )
    with pytest.raises(SystemExit, match="already started once"):
        cli.cmd_p11_eval(authorized=True, target_dir=tmp_path)


def test_no_phase_11_stage_reads_test_10(monkeypatch):
    """Every Phase 11 set goes through `_read_set` by name; test-10 is not one of them."""
    assert config.PHASE_10_TEST not in config.PHASE_11_QUESTION_SETS
    monkeypatch.setattr(phase10, "load_set", lambda *a, **k: pytest.fail("phase10.load_set"))
    with pytest.raises(SystemExit, match="reads only"):
        cli._p11_questions(config.PHASE_10_TEST, None, None, "x")  # type: ignore[arg-type]


def test_the_pass_provenance_describes_test_11(tmp_path):
    phase10_dir, target_dir = tmp_path / "p10", tmp_path / "p11"
    phase10_dir.mkdir()
    target_dir.mkdir()
    write(
        phase10_dir,
        phase10.QUESTIONS_FILENAME,
        {"sets": {config.PHASE_11_TEST: {"question_digest": "q11", "mapping_digest": "m11"}}},
    )
    write(target_dir, cli.P11_EMBED_NAME, {"sets": {config.PHASE_11_TEST: {"key": "k11"}}})
    components = {
        "provenance": {
            "question_digest": "q9",
            "mapping_digest": "m9",
            "embedding": {"model": "bge", "question_cache_key": "k9"},
            "weights": {},
        }
    }
    weights = {"hybrid-bm25": {"dense": 0.5, "bm25": 0.5}}
    provenance = cli._p11_pass_provenance(components, weights, phase10_dir, target_dir)
    assert (provenance["question_digest"], provenance["mapping_digest"]) == ("q11", "m11")
    assert provenance["embedding"] == {"model": "bge", "question_cache_key": "k11"}
    assert provenance["weights"] == weights


def test_the_dev_grid_reads_the_four_lists_in_the_fusions_order():
    row = {
        "dense": ["d"],
        "bm25": ["b"],
        "entity-hop@all": ["all"],
        "entity-hop@3000": ["c"],
        "question-hop": ["q"],
    }
    assert cli._p11_quad(row, None) == (["d"], ["b"], ["all"], ["q"])
    assert cli._p11_quad(row, 3000) == (["d"], ["b"], ["c"], ["q"])
