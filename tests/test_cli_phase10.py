"""Phase 10 CLI: the held-out pass opens test-10 once, only on a committed fit, never test-11.

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
        cli.cmd_p10_eval(authorized=False, target_dir=tmp_path)


def test_the_pass_refuses_without_a_fit(tmp_path):
    with pytest.raises(SystemExit, match="fit.json does not exist"):
        cli.cmd_p10_eval(authorized=True, target_dir=tmp_path)


def test_the_pass_refuses_after_a_dev_stop(tmp_path):
    write(tmp_path, cli.P10_FIT_NAME, {"terminal_state": "DEV_STOP"})
    with pytest.raises(SystemExit, match="DEV_STOP"):
        cli.cmd_p10_eval(authorized=True, target_dir=tmp_path)


def test_the_pass_refuses_an_uncommitted_fit(tmp_path, monkeypatch):
    write(tmp_path, cli.P10_FIT_NAME, {"terminal_state": None})
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: False)
    with pytest.raises(SystemExit, match="not committed"):
        cli.cmd_p10_eval(authorized=True, target_dir=tmp_path)


def test_the_pass_refuses_a_second_start(tmp_path, monkeypatch):
    write(tmp_path, cli.P10_FIT_NAME, {"terminal_state": None})
    write(tmp_path, cli.P10_MARKER_NAME, {"started_at": "earlier"})
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: True)
    monkeypatch.setattr(
        cli, "_phase_9_inputs", lambda *a, **k: pytest.fail("inputs loaded after a start")
    )
    with pytest.raises(SystemExit, match="already started once"):
        cli.cmd_p10_eval(authorized=True, target_dir=tmp_path)


def test_the_pass_provenance_describes_test_10_not_the_dev_chain(tmp_path):
    write(
        tmp_path,
        phase10.QUESTIONS_FILENAME,
        {"sets": {config.PHASE_10_TEST: {"question_digest": "q10", "mapping_digest": "m10"}}},
    )
    write(tmp_path, cli.P10_EMBED_NAME, {"sets": {config.PHASE_10_TEST: {"key": "k10"}}})
    components = {
        "provenance": {
            "question_digest": "q9",
            "mapping_digest": "m9",
            "embedding": {"model": "bge", "question_cache_key": "k9"},
            "weights": {"hybrid-bm25": {}, "hybrid-entity-hop": {}},
            "bm25_index_digest": "b",
        },
        "weights": {"hybrid-bm25": {"dense": 0.5, "bm25": 0.5}},
    }
    fit = {"weights": {"dense": 0.5, "bm25": 0.3, "entity-hop": 0.2}}
    provenance = cli._p10_pass_provenance(components, fit, tmp_path)
    assert (provenance["question_digest"], provenance["mapping_digest"]) == ("q10", "m10")
    assert provenance["embedding"] == {"model": "bge", "question_cache_key": "k10"}
    assert provenance["weights"] == {
        config.PHASE_10_SYSTEMS[1]: {"dense": 0.5, "bm25": 0.5},
        config.PHASE_10_SYSTEMS[2]: {"dense": 0.5, "bm25": 0.3, "entity-hop": 0.2},
    }
    assert provenance["bm25_index_digest"] == "b"


def test_the_reserved_set_is_never_loaded(tmp_path):
    with pytest.raises(phase10.Phase10Error, match="reserved"):
        phase10.load_set(tmp_path, config.PHASE_10_RESERVED, corpus_unit_set_hash="h")


def test_the_fit_refuses_after_a_contaminated_probe(tmp_path):
    write(tmp_path, cli.P10_REPRODUCTION_NAME, {"passed": True})
    write(tmp_path, cli.P10_PROBE_NAME, {"terminal_state": "DATA_STOP"})
    with pytest.raises(SystemExit, match="nothing is fitted"):
        cli.cmd_p10_fit(target_dir=tmp_path)


def test_the_probe_refuses_after_a_failed_reproduction(tmp_path):
    write(tmp_path, cli.P10_REPRODUCTION_NAME, {"passed": False})
    with pytest.raises(SystemExit, match="reproduction did not pass"):
        cli.cmd_p10_probe(target_dir=tmp_path)
