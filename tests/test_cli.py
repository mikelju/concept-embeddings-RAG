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


# --- T12: the `induce` stage -------------------------------------------------
#
# The whole K sweep, end to end: induce, calibrate, code, deduplicate, diagnose.
# It never embeds anything - it refuses to run unless `embed` has already cached
# the vectors - and it never recomputes what is already on disk, because that is
# what makes iterating on this phase affordable at all.


def an_induction_workspace(tmp_path: Path, n_units: int = 150, dim: int = 32):
    """A pool with cached embeddings, as `build` and `embed` leave it.

    Built directly rather than through the four-question fixture, which yields
    only eight paragraphs - too few to induce a dictionary from, and far too few
    for a coding to reach the sparsity floor this project refuses to ship below.
    """
    import numpy as np

    from concept_embeddings_rag.corpus.pool import (
        IndexingUnit,
        Question,
        save_pool,
        unit_id_for,
    )
    from concept_embeddings_rag.embeddings.cache import EmbeddingCache, cache_key, unit_set_hash

    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir(parents=True, exist_ok=True)

    def a_unit(index: int) -> IndexingUnit:
        title = f"Article {index}"
        sentences = (f"Sentence one of paragraph {index}.", f"Sentence two of {index}.")
        return IndexingUnit(unit_id=unit_id_for(title, sentences), title=title, sentences=sentences)

    units = [a_unit(index) for index in range(n_units)]
    questions = [
        Question(
            qid="q0",
            question="Does the induction read this?",
            answer="No.",
            gold_unit_ids=(units[0].unit_id,),
            supporting_facts=((units[0].title, 0),),
            split="dev",
        )
    ]
    save_pool(units, questions, data_dir / "pool.json")

    unit_ids = [unit.unit_id for unit in units]
    CorpusManifest(
        dataset="synthetic",
        source_url="http://example.invalid/x.json",
        sha256="0" * 64,
        downloaded_at="2026-09-08T10:00:00",
        seed=42,
        n_questions=len(questions),
        split_sizes={"dev": 1, "test": 0},
        n_units=len(units),
        unit_set_hash=unit_set_hash(unit_ids),
    ).save(data_dir / "manifest.json")

    # Positive mixtures of a handful of signed parts: dense enough in the concept
    # space that a coding can actually reach the target band, like real text is.
    rng = np.random.default_rng(0)
    parts = rng.normal(size=(12, dim))
    weights = rng.random(size=(n_units, 12))
    vectors = (weights @ parts + 0.05 * rng.normal(size=(n_units, dim))).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    from concept_embeddings_rag import config

    EmbeddingCache(cache_dir).save(
        cache_key(
            config.EMBEDDING_MODEL,
            config.EMBEDDING_REVISION,
            unit_set_hash(unit_ids),
            normalized=config.NORMALIZE_EMBEDDINGS,
        ),
        vectors,
        unit_ids,
        metadata={"model": config.EMBEDDING_MODEL, "dim": dim, "normalized": True},
    )
    return data_dir, cache_dir, units


def induce(tmp_path: Path, data_dir: Path, cache_dir: Path, **overrides):
    from concept_embeddings_rag.cli import cmd_induce

    arguments = {
        "data_dir": data_dir,
        "cache_dir": cache_dir,
        "concepts_dir": tmp_path / "concepts",
        "k_sweep": (8, 12),
        "target_band": (4, 8),
    }
    arguments.update(overrides)
    return cmd_induce(**arguments)


def test_parser_exposes_the_induce_stage():
    assert build_parser().parse_args(["induce"]).command == "induce"


def test_induce_without_a_pool_fails_with_a_clear_message(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        induce(tmp_path, data_dir, tmp_path / "cache")

    assert "build" in str(excinfo.value)


def test_induce_without_cached_embeddings_names_the_embed_stage(tmp_path):
    """Not a numpy traceback: the stage says which stage to run first."""
    data_dir, _, _ = an_induction_workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        induce(tmp_path, data_dir, tmp_path / "empty-cache")

    message = str(excinfo.value)
    assert "embed" in message
    assert message.startswith("[ERROR]")


def test_induce_sweeps_every_k_into_its_own_artifacts(tmp_path):
    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)

    keys = induce(tmp_path, data_dir, cache_dir)

    concepts_dir = tmp_path / "concepts"
    assert len(keys) == 2
    assert len(set(keys)) == 2, "one K must never overwrite another"
    # Two dictionaries per K: the induced one and the deduplicated one.
    assert len(list(concepts_dir.glob("dictionary-*.npz"))) == 4
    assert len(list(concepts_dir.glob("matrix-*-raw.npz"))) == 4


def test_induce_writes_a_merge_log_and_diagnostics_for_every_k(tmp_path):
    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)

    keys = induce(tmp_path, data_dir, cache_dir)

    concepts_dir = tmp_path / "concepts"
    for key in keys:
        assert (concepts_dir / f"merge-log-{key}.json").exists()
        assert (concepts_dir / f"diagnostics-{key}.json").exists()


def test_the_diagnostics_describe_the_pool_they_were_built_from(tmp_path):
    from concept_embeddings_rag.concepts.diagnostics import load_diagnostics

    data_dir, cache_dir, units = an_induction_workspace(tmp_path)

    keys = induce(tmp_path, data_dir, cache_dir)

    diagnostics = load_diagnostics(keys[0], tmp_path / "concepts")
    assert diagnostics.n_units == len(units)
    assert diagnostics.active_per_unit["mean"] >= 3


def test_every_space_is_scored_against_a_null_baseline(tmp_path):
    """T10b: the sweep writes the quality block, not only the shape of the space."""
    from concept_embeddings_rag.concepts.diagnostics import load_diagnostics

    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)

    keys = induce(tmp_path, data_dir, cache_dir)

    for key in keys:
        quality = load_diagnostics(key, tmp_path / "concepts").quality
        assert quality is not None
        assert quality.null_mean > 0.0
        assert quality.coherence, "a space with no scored concept is not a scored space"


def test_a_diagnostics_file_that_predates_the_quality_block_is_recomputed(tmp_path):
    """Seconds to recompute, and an unscored artifact would read as a scored one."""
    import json as json_module

    from concept_embeddings_rag.concepts.diagnostics import load_diagnostics

    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)
    key = induce(tmp_path, data_dir, cache_dir)[0]
    path = tmp_path / "concepts" / f"diagnostics-{key}.json"
    stale = json_module.loads(path.read_text(encoding="utf-8"))
    stale["quality"] = None
    path.write_text(json_module.dumps(stale), encoding="utf-8")

    induce(tmp_path, data_dir, cache_dir)

    assert load_diagnostics(key, tmp_path / "concepts").quality is not None


def test_a_second_run_over_unchanged_inputs_recomputes_nothing(tmp_path, monkeypatch):
    """Rebuilding these caches costs hours on the real corpus."""
    from concept_embeddings_rag import cli as cli_module

    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)
    first = induce(tmp_path, data_dir, cache_dir)

    def refuse(*args, **kwargs):
        raise AssertionError("the second run recomputed something already on disk")

    for name in (
        "induce_dictionary",
        "calibrate_coding_alpha",
        "code_corpus",
        "recode_after_merge",
        "deduplicate_dictionary",
        "compute_diagnostics",
    ):
        monkeypatch.setattr(cli_module, name, refuse)

    assert induce(tmp_path, data_dir, cache_dir) == first


def test_every_message_the_stage_prints_is_ascii(tmp_path, capsys):
    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)

    induce(tmp_path, data_dir, cache_dir)

    output = capsys.readouterr().out
    assert output.isascii(), "non-ASCII output dies under charmap on the deployment path"
    assert "[OK]" in output


def test_the_sweep_stops_rather_than_running_past_its_budget(tmp_path, capsys):
    """A fit that costs a night is a design error, not something to wait out."""
    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)

    keys = induce(tmp_path, data_dir, cache_dir, budget_seconds=0)

    assert len(keys) == 1
    assert "[WARN]" in capsys.readouterr().out


def test_the_sweep_reads_no_question_gold_or_split():
    """HU-1: no question, no gold annotation and no split label takes part."""
    import inspect

    from concept_embeddings_rag.cli import cmd_induce

    parameters = set(inspect.signature(cmd_induce).parameters)

    assert not parameters & {"split", "questions", "gold", "gold_unit_ids", "queries"}


# --- T15: the `label` stage --------------------------------------------------
#
# The only stage of this project that spends money, and the only one that needs a
# secret. It labels one dictionary of the sweep, for the report; nothing in
# retrieval, scoring or expansion ever reads what it writes.


class FakeLabelingClient:
    """Answers without a network, a key, or a cent."""

    def __init__(self) -> None:
        from concept_embeddings_rag import config

        self.model = config.LABELING_MODEL
        self.prompts: list[str] = []

    def label(self, prompt: str):
        from concept_embeddings_rag.concepts.labeling import LabelResponse

        self.prompts.append(prompt)
        return LabelResponse(
            name=f"Concept {len(self.prompts)}",
            gloss="What these paragraphs share.",
            input_tokens=1200,
            output_tokens=60,
        )


def label(tmp_path: Path, data_dir: Path, **overrides):
    from concept_embeddings_rag.cli import cmd_label

    arguments = {
        "data_dir": data_dir,
        "concepts_dir": tmp_path / "concepts",
        "k": 8,
        "n_units_shown": 2,
    }
    arguments.update(overrides)
    return cmd_label(**arguments)


def test_parser_exposes_the_label_stage():
    assert build_parser().parse_args(["label"]).command == "label"


def test_label_without_a_concept_space_names_the_induce_stage(tmp_path):
    data_dir, _, _ = an_induction_workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        label(tmp_path, data_dir, client=FakeLabelingClient())

    assert "induce" in str(excinfo.value)


def test_label_without_a_key_exits_naming_the_variable(tmp_path, monkeypatch):
    """A one-line message, before any HTTP client is constructed."""
    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)
    induce(tmp_path, data_dir, cache_dir)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("concept_embeddings_rag.cli.read_api_key", _raise_missing_key)

    with pytest.raises(SystemExit) as excinfo:
        label(tmp_path, data_dir)

    message = str(excinfo.value)
    assert message.startswith("[ERROR]")
    assert "ANTHROPIC_API_KEY" in message
    assert "Traceback" not in message


def _raise_missing_key(*args, **kwargs):
    """Stand in for a machine with no key, whatever this one happens to have set."""
    from concept_embeddings_rag.concepts.labeling import API_KEY_VARIABLE, MissingAPIKey

    raise MissingAPIKey(f"{API_KEY_VARIABLE} is not set. Copy .env.example to .env")


def test_label_never_builds_a_client_when_the_key_is_missing(tmp_path, monkeypatch):
    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)
    induce(tmp_path, data_dir, cache_dir)
    monkeypatch.setattr("concept_embeddings_rag.cli.read_api_key", _raise_missing_key)

    def refuse(*args, **kwargs):
        raise AssertionError("a client was constructed before the key was checked")

    monkeypatch.setattr("concept_embeddings_rag.cli.AnthropicLabelingClient", refuse)

    with pytest.raises(SystemExit):
        label(tmp_path, data_dir)


def test_label_writes_the_artifact_and_reports_the_actual_cost(tmp_path, capsys):
    from concept_embeddings_rag.concepts.labeling import load_labels

    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)
    keys = induce(tmp_path, data_dir, cache_dir)
    client = FakeLabelingClient()

    path = label(tmp_path, data_dir, client=client)

    labels = load_labels(keys[0], tmp_path / "concepts")
    assert path.exists()
    assert labels.labels
    assert labels.cost["calls"] == len(client.prompts)
    assert labels.cost["actual_usd"] > 0
    output = capsys.readouterr().out
    assert "estimated cost" in output
    assert "USD spent" in output
    assert output.isascii()


def test_a_second_labelling_run_makes_no_call_at_all(tmp_path):
    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)
    induce(tmp_path, data_dir, cache_dir)
    label(tmp_path, data_dir, client=FakeLabelingClient())

    second = FakeLabelingClient()
    label(tmp_path, data_dir, client=second)

    assert second.prompts == []


def test_the_labels_are_built_from_the_units_that_activate_each_concept(tmp_path):
    data_dir, cache_dir, units = an_induction_workspace(tmp_path)
    induce(tmp_path, data_dir, cache_dir)
    client = FakeLabelingClient()

    label(tmp_path, data_dir, client=client)

    known = {unit.indexable_text for unit in units}
    assert client.prompts
    for prompt in client.prompts:
        assert any(text in prompt for text in known)
