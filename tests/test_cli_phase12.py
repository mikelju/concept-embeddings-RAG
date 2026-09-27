"""Phase 12 CLI: each stage writes its artifact once and refuses to run again over it.

Mirrors `tests/test_cli_phase10.py` and `tests/test_cli_phase11.py`: the refusals are checked
against a toy target directory holding only the artifact that refusal reads, before any real
input (the corpus, the entity index, the sentence cache) would be loaded.
"""

import gzip
import json

import pytest

from concept_embeddings_rag import cli, config
from concept_embeddings_rag.artifacts import digest_of


def write(directory, name, body):
    (directory / name).write_text(json.dumps(body), encoding="utf-8")


def test_sentences_refuses_a_second_run(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache" / cli.P12_SENTENCE_CACHE_DIRNAME
    cache_dir.mkdir(parents=True)
    write(cache_dir, "sentences-dev.json", {"key": "k"})
    monkeypatch.setattr(
        cli, "_p10_json", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p12_sentences(target_dir=tmp_path)


def test_lists_refuses_a_second_run(tmp_path, monkeypatch):
    write(tmp_path, cli.P12_REPRODUCTION_NAME, {"passed": True})
    monkeypatch.setattr(
        cli, "_p10_json", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p12_lists(target_dir=tmp_path)


def test_fit_refuses_without_a_reproduction(tmp_path):
    with pytest.raises(SystemExit, match="reproduction.json does not exist"):
        cli.cmd_p12_fit(target_dir=tmp_path)


def test_fit_refuses_after_a_failed_reproduction(tmp_path):
    write(tmp_path, cli.P12_REPRODUCTION_NAME, {"passed": False})
    with pytest.raises(SystemExit, match="did not pass"):
        cli.cmd_p12_fit(target_dir=tmp_path)


def test_fit_refuses_a_second_run(tmp_path, monkeypatch):
    write(tmp_path, cli.P12_REPRODUCTION_NAME, {"passed": True, "dev_lists_digest": "d"})
    write(tmp_path, cli.P12_FIT_NAME, {"terminal_state": None})
    monkeypatch.setattr(
        cli, "_p11_dev_inputs", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p12_fit(target_dir=tmp_path)


def test_load_dev_lists_checks_the_digest(tmp_path):
    rows = [{"qid": "q1", "dense": [["u1", 1.0]]}]
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    with (
        (tmp_path / cli.P12_DEV_LISTS).open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed,
    ):
        packed.write(text.encode("utf-8"))

    write(
        tmp_path,
        cli.P12_DEV_LISTS_MANIFEST,
        {"digest": digest_of(text), "lists": ["dense"]},
    )
    rows_out = cli._p12_load_dev_lists(tmp_path)
    assert rows_out[0]["dense"] == [("u1", 1.0)]

    write(tmp_path, cli.P12_DEV_LISTS_MANIFEST, {"digest": "not-the-digest", "lists": ["dense"]})
    with pytest.raises(SystemExit, match="does not match its recorded digest"):
        cli._p12_load_dev_lists(tmp_path)


def test_p12_systems_and_component_names_agree_in_order():
    assert config.PHASE_12_SYSTEMS[3] == "hybrid-bm25-seeded-hop"
    assert config.PHASE_12_COMPONENT_NAMES == ("dense", "bm25", "seeded-hop")
