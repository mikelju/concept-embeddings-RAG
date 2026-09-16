"""T13: the command line interface.

Each stage runs standalone and is idempotent, and a stage whose input is missing
says so plainly instead of failing deep inside numpy.
"""

import json
import shutil
from pathlib import Path

import numpy as np
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


def test_induce_writes_both_views_of_every_matrix_to_disk(tmp_path):
    """Phase 3, decision D3: the sweep loads the artifact for the view it measures.

    `row_normalized` is a pure function of `raw`, but deriving it at load time
    would leave half of Phase 3's cells reading a matrix no hash ever verified.
    It is written like every other artifact of this pipeline, and read back.
    """
    from concept_embeddings_rag.concepts.coding import load_matrix

    data_dir, cache_dir, units = an_induction_workspace(tmp_path)

    keys = induce(tmp_path, data_dir, cache_dir)

    concepts_dir = tmp_path / "concepts"
    assert len(list(concepts_dir.glob("matrix-*-row_normalized.npz"))) == 4
    for key in keys:
        matrix = load_matrix(
            key, concepts_dir, view="row_normalized", expected_unit_ids=[u.unit_id for u in units]
        )
        assert matrix.view == "row_normalized"
        sums = np.asarray(matrix.X.sum(axis=1)).ravel()
        assert np.allclose(sums[sums > 0.0], 1.0, atol=1e-5)


def test_a_missing_derived_view_is_written_without_recoding_the_corpus(tmp_path, monkeypatch):
    """The state the real `data/concepts/` was in: raw on disk, the derived view absent."""
    from concept_embeddings_rag import cli as cli_module

    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)
    induce(tmp_path, data_dir, cache_dir)
    concepts_dir = tmp_path / "concepts"
    for path in concepts_dir.glob("matrix-*-row_normalized.*"):
        path.unlink()

    def refuse(*args, **kwargs):
        raise AssertionError("the missing view was rebuilt by coding the corpus again")

    for name in ("calibrate_coding_alpha", "code_corpus", "recode_after_merge"):
        monkeypatch.setattr(cli_module, name, refuse)

    induce(tmp_path, data_dir, cache_dir)

    assert len(list(concepts_dir.glob("matrix-*-row_normalized.npz"))) == 4


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


def test_diagnostics_that_do_not_say_which_pool_they_describe_are_recomputed(tmp_path):
    """SEC-012: an artifact written before the field existed cannot be checked against

    the pool, and something that cannot be checked is not reused as if it had been.
    Recomputing costs seconds; citing evidence from another pool costs the report.
    """
    from concept_embeddings_rag.concepts.diagnostics import load_diagnostics

    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)
    key = induce(tmp_path, data_dir, cache_dir)[0]
    path = tmp_path / "concepts" / f"diagnostics-{key}.json"
    legacy = json.loads(path.read_text(encoding="utf-8"))
    for field in ("unit_set_hash", "digest"):
        legacy.pop(field, None)
    path.write_text(json.dumps(legacy, indent=2, sort_keys=True), encoding="utf-8")

    induce(tmp_path, data_dir, cache_dir)

    assert load_diagnostics(key, tmp_path / "concepts").unit_set_hash


def test_stale_diagnostics_name_the_artifact_instead_of_raising_a_bare_key_error():
    """SEC-012: the evidence ids come from an artifact and the texts from the pool.

    When the two disagree the raw lookup raises `KeyError('unit9999')`, which reads
    like a bug in the labelling stage rather than what it is - an artifact
    describing a pool that is no longer on disk.
    """
    from concept_embeddings_rag.cli import _text_of

    text_by_id = {"unit0001": "The Douro flows through northern Portugal."}
    assert _text_of("unit0001", text_by_id, "abc123") == text_by_id["unit0001"]

    with pytest.raises(SystemExit) as excinfo:
        _text_of("unit9999", text_by_id, "abc123")

    message = str(excinfo.value)
    assert "unit9999" in message
    assert "induce" in message


# --- T17: the selection stage -------------------------------------------------
#
# `select` is the stage that spends the dev split: it sweeps the four spaces, picks
# one, measures the damping variant on it, fits the fusion twice - once for System B
# and once for the HU-5 control - and freezes the result. Everything it decides is
# in one artifact, and it writes that artifact once.
#
# The workspace below is the induction one with questions in both splits and token
# counts added. The spaces are induced at k = 20 and 24 because the declared query
# truncation keeps the top 16 concepts: a dictionary smaller than that would make
# `projection_top16` and `projection_full` the same arm under two names.


class HashingBackend:
    """Embeds a question deterministically, with no model and no network.

    It carries the real model's name and revision because the corpus cache is keyed
    by them: a backend that renamed itself would look like a different model to
    every artifact in the workspace.
    """

    name = "BAAI/bge-small-en-v1.5"
    revision = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    normalize = True

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode(self, texts):
        import hashlib

        rows = []
        for text in texts:
            digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            vector = rng.normal(size=self.dim)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype="float32")


def a_selection_workspace(tmp_path: Path, n_dev: int = 4, n_test: int = 3):
    """A pool with two concept spaces, questions in both splits, and token counts."""
    from concept_embeddings_rag.corpus.pool import Question, save_pool

    data_dir, cache_dir, units = an_induction_workspace(tmp_path)
    questions = [
        Question(
            qid=f"q{index}",
            question=f"Which paragraph follows paragraph {index}?",
            answer="-",
            gold_unit_ids=(units[index].unit_id, units[index + 1].unit_id),
            supporting_facts=((units[index].title, 0), (units[index + 1].title, 0)),
            split="dev" if index < n_dev else "test",
        )
        for index in range(n_dev + n_test)
    ]
    save_pool(units, questions, data_dir / "pool.json")
    (data_dir / "token_counts.json").write_text(
        json.dumps({unit.unit_id: 10 for unit in units}, sort_keys=True), encoding="utf-8"
    )
    induce(tmp_path, data_dir, cache_dir, k_sweep=(20, 24), merge_threshold=0.99)
    return data_dir, cache_dir, units, questions


def select(tmp_path: Path, data_dir: Path, cache_dir: Path, **overrides):
    from concept_embeddings_rag.cli import cmd_select

    arguments = {
        "data_dir": data_dir,
        "cache_dir": cache_dir,
        "concepts_dir": tmp_path / "concepts",
        "selection_dir": tmp_path / "selection",
        "question_cache_dir": tmp_path / "questions",
        "k_sweep": (20, 24),
        "merge_threshold": 0.99,
        "top_k": 5,
        "backend": HashingBackend(),
    }
    arguments.update(overrides)
    return cmd_select(**arguments)


def test_parser_exposes_the_select_stage():
    assert build_parser().parse_args(["select"]).command == "select"


def test_select_without_a_pool_names_the_build_stage(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        select(tmp_path, data_dir, tmp_path / "cache")

    assert "build" in str(excinfo.value)


def test_select_without_token_counts_names_the_embed_stage(tmp_path):
    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        select(tmp_path, data_dir, cache_dir)

    assert "embed" in str(excinfo.value)


def test_select_without_the_concept_spaces_names_the_induce_stage(tmp_path):
    """A clean error naming the stage that produces them, never a traceback."""
    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        select(tmp_path, data_dir, cache_dir, concepts_dir=tmp_path / "empty-concepts")

    message = str(excinfo.value)
    assert message.startswith("[ERROR]")
    assert "induce" in message


def test_select_writes_exactly_one_artifact_and_prints_its_path(tmp_path, capsys):
    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)

    path = select(tmp_path, data_dir, cache_dir)

    assert [entry.name for entry in (tmp_path / "selection").iterdir()] == ["selection.json"]
    assert str(path) in capsys.readouterr().out


def test_the_stage_prints_ascii_only(tmp_path, capsys):
    """Gunicorn and Cloud Run read stdout as charmap; an arrow there is a crash."""
    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)
    capsys.readouterr()

    select(tmp_path, data_dir, cache_dir)

    assert capsys.readouterr().out.isascii()


def test_the_frozen_selection_names_the_space_and_the_decisions_that_chose_it(tmp_path):
    from concept_embeddings_rag.evaluation.selection import load_selection

    data_dir, cache_dir, _, questions = a_selection_workspace(tmp_path)

    select(tmp_path, data_dir, cache_dir)

    report = load_selection(tmp_path / "selection")
    assert report.selected_k in (20, 24)
    assert report.selected_dictionary_key in report.dictionary_keys
    assert [decision.name for decision in report.decisions][:3] == [
        "space",
        "damping",
        "fusion_scheme",
    ]
    assert report.n_dev_decisions == len(report.decisions)
    assert report.fusion is not None and report.control is not None
    assert report.fusion.components[1] == "conceptual"
    assert report.control.components[1] == "bm25"


def test_the_sweep_behind_the_artifact_measured_the_dev_split_alone(tmp_path):
    from concept_embeddings_rag.evaluation.selection import load_selection

    data_dir, cache_dir, _, questions = a_selection_workspace(tmp_path)
    n_dev = len([q for q in questions if q.split == "dev"])

    select(tmp_path, data_dir, cache_dir)

    report = load_selection(tmp_path / "selection")
    for entry in report.per_k.values():
        assert entry.n_questions == n_dev
    assert report.fusion is not None and report.fusion.n_questions == n_dev


def test_both_splits_are_embedded_once_so_evaluate_never_loads_the_model(tmp_path):
    """Embedding a question is a vector, not a measurement: nothing scores test here."""
    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)

    select(tmp_path, data_dir, cache_dir)

    cached = sorted((tmp_path / "questions").glob("embeddings-*.npz"))
    splits = {
        json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["split"]
        for path in cached
    }
    assert splits == {"dev", "test"}


def test_a_second_selection_refuses_rather_than_moving_the_freeze(tmp_path):
    """A re-freeze is a deviation to be written down, not a file to be replaced."""
    from concept_embeddings_rag.evaluation.selection import SelectionError, load_selection

    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)
    select(tmp_path, data_dir, cache_dir)
    frozen_at = load_selection(tmp_path / "selection").frozen_at

    with pytest.raises(SelectionError, match="already"):
        select(tmp_path, data_dir, cache_dir)

    assert load_selection(tmp_path / "selection").frozen_at == frozen_at


# --- T18: evaluate, extended with the three systems of this phase -------------


def evaluate(tmp_path: Path, data_dir: Path, cache_dir: Path, **overrides):
    from concept_embeddings_rag.cli import cmd_evaluate

    arguments = {
        "data_dir": data_dir,
        "cache_dir": cache_dir,
        "results_dir": tmp_path / "results",
        "concepts_dir": tmp_path / "concepts",
        "selection_dir": tmp_path / "selection",
        "question_cache_dir": tmp_path / "questions",
        "top_k": 5,
        "backend": HashingBackend(),
    }
    arguments.update(overrides)
    return cmd_evaluate(**arguments)


def results_of(paths) -> list[dict]:
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def test_evaluate_refuses_the_test_split_until_a_configuration_is_frozen(tmp_path):
    """HU-7 reads test once, on a frozen configuration; the stage enforces it."""
    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        evaluate(tmp_path, data_dir, cache_dir)

    message = str(excinfo.value)
    assert message.startswith("[ERROR]")
    assert "select" in message


def test_evaluate_on_dev_alone_measures_the_baselines_when_nothing_is_frozen(tmp_path):
    """Dev is the split this phase is allowed to look at, so a quick check stays possible."""
    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)

    written = evaluate(tmp_path, data_dir, cache_dir, splits=("dev",))

    assert {result["system"] for result in results_of(written)} == {"dense", "bm25"}


def test_evaluate_measures_the_three_phase_3_systems_from_the_frozen_selection(tmp_path):
    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)
    select(tmp_path, data_dir, cache_dir)

    written = evaluate(tmp_path, data_dir, cache_dir)

    results = results_of(written)
    assert {result["system"] for result in results} == {
        "dense",
        "bm25",
        "conceptual",
        "hybrid-conceptual",
        "hybrid-bm25",
    }
    assert {result["split"] for result in results} == {"dev", "test"}


def test_every_phase_3_result_carries_the_six_keys_and_the_freeze_it_ran_under(tmp_path):
    from concept_embeddings_rag.evaluation.selection import load_selection

    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)
    select(tmp_path, data_dir, cache_dir)
    frozen_at = load_selection(tmp_path / "selection").frozen_at

    written = evaluate(tmp_path, data_dir, cache_dir)

    phase_3 = [
        result
        for result in results_of(written)
        if result["system"] in {"conceptual", "hybrid-conceptual", "hybrid-bm25"}
    ]
    assert len(phase_3) == 6
    for result in phase_3:
        for key in ("dictionary_key", "k", "view", "query_operator", "damping", "fusion_scheme"):
            assert key in result["config"], f"{result['system']} does not record {key}"
        assert result["config"]["frozen_at"] == frozen_at


def test_the_baselines_do_not_claim_a_space_they_never_used(tmp_path):
    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)
    select(tmp_path, data_dir, cache_dir)

    written = evaluate(tmp_path, data_dir, cache_dir)

    for result in results_of(written):
        if result["system"] in {"dense", "bm25"}:
            assert "dictionary_key" not in result["config"]
            assert "frozen_at" not in result["config"]


def test_the_conceptual_system_records_no_fusion_because_nothing_was_fused_into_it(tmp_path):
    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)
    select(tmp_path, data_dir, cache_dir)

    written = evaluate(tmp_path, data_dir, cache_dir, splits=("dev",))

    conceptual = next(r for r in results_of(written) if r["system"] == "conceptual")
    assert conceptual["config"]["fusion_scheme"] is None


def test_system_b_and_the_control_record_the_same_scheme_they_were_fitted_under(tmp_path):
    from concept_embeddings_rag.evaluation.selection import load_selection

    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)
    select(tmp_path, data_dir, cache_dir)
    report = load_selection(tmp_path / "selection")

    written = evaluate(tmp_path, data_dir, cache_dir, splits=("dev",))

    by_system = {r["system"]: r["config"] for r in results_of(written)}
    assert report.fusion is not None and report.control is not None
    assert by_system["hybrid-conceptual"]["fusion_scheme"] == report.fusion.winning_scheme
    assert by_system["hybrid-bm25"]["fusion_scheme"] == report.control.winning_scheme


def test_the_label_stage_takes_the_dictionary_size_it_is_asked_for():
    """Phase 3 selects its own K, and when it is not Phase 2's the report needs that one."""
    from concept_embeddings_rag import config

    assert build_parser().parse_args(["label"]).k == config.LABELED_K
    assert build_parser().parse_args(["label", "--k", "512"]).k == 512


# --- T15: the `expand` stage ---------------------------------------------------
#
# `expand` is Phase 4's equivalent of `select`: it spends the dev split on the two
# free parameters it is allowed to fit, and freezes the cell it chose. Everything
# else it runs under is **inherited** - the space, the view, the damping and the
# query operator come out of the Phase 3 artifact, read and verified rather than
# retyped - which is why the stage refuses to start when that artifact is missing
# and names the stage that writes it.


def expand(tmp_path: Path, data_dir: Path, cache_dir: Path, **overrides):
    from concept_embeddings_rag.cli import cmd_expand

    arguments = {
        "data_dir": data_dir,
        "cache_dir": cache_dir,
        "concepts_dir": tmp_path / "concepts",
        "selection_dir": tmp_path / "selection",
        "expansion_dir": tmp_path / "expansion",
        "question_cache_dir": tmp_path / "questions",
        "top_k": 5,
        "backend": HashingBackend(),
    }
    arguments.update(overrides)
    return cmd_expand(**arguments)


def an_expansion_workspace(tmp_path: Path):
    """A workspace with the Phase 3 configuration already frozen."""
    data_dir, cache_dir, units, questions = a_selection_workspace(tmp_path)
    select(tmp_path, data_dir, cache_dir)
    return data_dir, cache_dir, units, questions


def test_parser_exposes_the_expand_stage():
    assert build_parser().parse_args(["expand"]).command == "expand"


def test_expand_without_a_pool_names_the_build_stage(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        expand(tmp_path, data_dir, tmp_path / "cache")

    assert "build" in str(excinfo.value)


def test_expand_without_token_counts_names_the_embed_stage(tmp_path):
    data_dir, cache_dir, _ = an_induction_workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        expand(tmp_path, data_dir, cache_dir)

    assert "embed" in str(excinfo.value)


def test_expand_without_the_phase_3_freeze_names_the_select_stage(tmp_path):
    """The four inherited decisions are read from an artifact or the phase does not run."""
    data_dir, cache_dir, _, _ = a_selection_workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        expand(tmp_path, data_dir, cache_dir)

    message = str(excinfo.value)
    assert message.startswith("[ERROR]")
    assert "select" in message


def test_expand_without_cached_embeddings_names_the_embed_stage(tmp_path):
    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    for path in Path(cache_dir).glob("embeddings-*.npz"):
        path.unlink()

    with pytest.raises(SystemExit) as excinfo:
        expand(tmp_path, data_dir, cache_dir)

    assert "embed" in str(excinfo.value)


def test_expand_writes_exactly_one_artifact_and_prints_its_path(tmp_path, capsys):
    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)

    path = expand(tmp_path, data_dir, cache_dir)

    assert path.exists()
    assert sorted(p.name for p in (tmp_path / "expansion").iterdir()) == ["expansion.json"]
    assert str(path) in capsys.readouterr().out


def test_a_second_expansion_refuses_rather_than_moving_the_freeze(tmp_path):
    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    with pytest.raises(SystemExit) as excinfo:
        expand(tmp_path, data_dir, cache_dir)

    assert "already frozen" in str(excinfo.value)


def test_the_stage_prints_the_cell_the_margin_and_what_it_spent(tmp_path, capsys):
    from concept_embeddings_rag import config

    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)

    expand(tmp_path, data_dir, cache_dir)

    out = capsys.readouterr().out
    assert "restart=" in out
    assert "margin" in out
    assert f"of {config.DEV_EVALUATION_CAP} dev evaluations" in out
    assert out.isascii()


def test_the_frozen_cell_names_the_grid_it_was_chosen_from(tmp_path):
    from concept_embeddings_rag import config
    from concept_embeddings_rag.evaluation.expansion_selection import load_expansion_selection

    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    report = load_expansion_selection(tmp_path / "expansion")

    assert len(report.cells) == 24
    assert report.chosen_restart in config.RESTART_GRID
    assert report.chosen_normalization in config.NORMALIZATION_ARMS
    assert report.dev_evaluations_spent == 24
    assert report.n_dev_decisions == 1


def test_the_freeze_inherits_the_phase_3_configuration_it_was_fitted_under(tmp_path):
    from concept_embeddings_rag.evaluation.expansion_selection import load_expansion_selection
    from concept_embeddings_rag.evaluation.selection import load_selection, selection_digest

    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    inherited = load_selection(tmp_path / "selection")
    report = load_expansion_selection(tmp_path / "expansion")

    assert report.inherits_selection_digest == selection_digest(inherited)
    assert report.inherits["dictionary_key"] == inherited.config["dictionary_key"]
    assert report.inherits["view"] == inherited.config["view"]
    assert report.inherits["damping"] == inherited.config["damping"]
    assert report.inherits["query_operator"] == inherited.config["query_operator"]


def test_the_expansion_grid_behind_the_artifact_measured_the_dev_split_alone(tmp_path):
    from concept_embeddings_rag.evaluation.expansion_selection import load_expansion_selection

    data_dir, cache_dir, _, questions = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    report = load_expansion_selection(tmp_path / "expansion")
    n_dev = len([question for question in questions if question.split == "dev"])
    assert {cell.n_questions for cell in report.cells} == {n_dev}


def test_the_artifact_verifies_against_the_pool_it_was_measured_over(tmp_path):
    from concept_embeddings_rag.evaluation.expansion_selection import (
        ExpansionSelectionError,
        load_expansion_selection,
    )

    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    with pytest.raises(ExpansionSelectionError, match="pool"):
        load_expansion_selection(tmp_path / "expansion", expected_unit_set_hash="another-pool")


# --- T16: evaluate, extended with the two expansion systems --------------------


def test_the_selector_measures_the_two_expansion_systems_and_no_phase_3_system(tmp_path):
    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    written = evaluate(
        tmp_path,
        data_dir,
        cache_dir,
        expansion_dir=tmp_path / "expansion",
        systems=["expansion", "expansion-conceptual"],
    )

    results = results_of(written)
    assert {result["system"] for result in results} == {"expansion", "expansion-conceptual"}
    assert {result["split"] for result in results} == {"dev", "test"}


def test_a_bare_evaluate_measures_no_expansion_system_at_all(tmp_path):
    """Decision D11: building them by default would duplicate every Phase 3 result."""
    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    written = evaluate(tmp_path, data_dir, cache_dir, expansion_dir=tmp_path / "expansion")

    assert not {"expansion", "expansion-conceptual"} & {r["system"] for r in results_of(written)}


def test_the_selector_can_also_name_a_phase_3_system(tmp_path):
    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)

    written = evaluate(tmp_path, data_dir, cache_dir, systems=["dense"], splits=("dev",))

    assert {result["system"] for result in results_of(written)} == {"dense"}


def test_an_unknown_system_is_refused_by_name(tmp_path):
    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        evaluate(tmp_path, data_dir, cache_dir, systems=["diffusion"], splits=("dev",))

    assert "diffusion" in str(excinfo.value)


def test_the_expansion_systems_are_refused_without_the_phase_4_freeze(tmp_path):
    """Test is read once, on a frozen configuration - this phase's as well as Phase 3's."""
    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        evaluate(
            tmp_path,
            data_dir,
            cache_dir,
            expansion_dir=tmp_path / "expansion",
            systems=["expansion"],
        )

    message = str(excinfo.value)
    assert message.startswith("[ERROR]")
    assert "expand" in message


def test_every_expansion_result_carries_the_diffusion_configuration(tmp_path):
    from concept_embeddings_rag.evaluation.expansion_selection import load_expansion_selection

    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)
    report = load_expansion_selection(tmp_path / "expansion")

    written = evaluate(
        tmp_path,
        data_dir,
        cache_dir,
        expansion_dir=tmp_path / "expansion",
        systems=["expansion", "expansion-conceptual"],
        splits=("dev",),
    )

    for result in results_of(written):
        recorded = result["config"]
        assert recorded["restart"] == report.chosen_restart
        assert recorded["normalization"] == report.chosen_normalization
        assert recorded["stop_threshold"] == report.stop_threshold
        assert recorded["max_iterations"] == report.max_iterations
        assert recorded["seed_top_k"] == report.config["seed_top_k"]


def test_every_expansion_result_names_its_arm_and_the_seed_that_produced_it(tmp_path):
    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    written = evaluate(
        tmp_path,
        data_dir,
        cache_dir,
        expansion_dir=tmp_path / "expansion",
        systems=["expansion", "expansion-conceptual"],
        splits=("dev",),
    )

    by_system = {result["system"]: result["config"] for result in results_of(written)}
    assert by_system["expansion"]["seed_arm"] == "dense"
    assert by_system["expansion"]["seed_system"] == "dense"
    assert by_system["expansion-conceptual"]["seed_arm"] == "conceptual"
    assert by_system["expansion-conceptual"]["seed_system"] == "conceptual"


def test_every_expansion_result_carries_the_four_inherited_decisions(tmp_path):
    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    written = evaluate(
        tmp_path,
        data_dir,
        cache_dir,
        expansion_dir=tmp_path / "expansion",
        systems=["expansion"],
        splits=("dev",),
    )

    config_of = results_of(written)[0]["config"]
    for key in ("k", "dictionary_key", "view", "damping", "query_operator"):
        assert key in config_of, f"the result does not record {key}"


def test_every_expansion_result_carries_both_selection_digests(tmp_path):
    from concept_embeddings_rag.evaluation.expansion_selection import (
        expansion_selection_digest,
        load_expansion_selection,
    )
    from concept_embeddings_rag.evaluation.selection import load_selection, selection_digest

    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    written = evaluate(
        tmp_path,
        data_dir,
        cache_dir,
        expansion_dir=tmp_path / "expansion",
        systems=["expansion"],
        splits=("dev",),
    )

    config_of = results_of(written)[0]["config"]
    assert config_of["selection_digest"] == selection_digest(load_selection(tmp_path / "selection"))
    assert config_of["expansion_selection_digest"] == expansion_selection_digest(
        load_expansion_selection(tmp_path / "expansion")
    )


def test_the_counters_in_the_result_are_the_ones_the_retriever_recorded(tmp_path):
    from concept_embeddings_rag import config

    data_dir, cache_dir, _, questions = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    written = evaluate(
        tmp_path,
        data_dir,
        cache_dir,
        expansion_dir=tmp_path / "expansion",
        systems=["expansion"],
        splits=("dev",),
    )

    config_of = results_of(written)[0]["config"]
    n_dev = len([question for question in questions if question.split == "dev"])
    assert config_of["n_questions"] == n_dev
    assert sum(config_of["iterations_histogram"].values()) == n_dev
    assert 0 < config_of["mean_iterations"] <= config.MAX_ITERATIONS
    assert config_of["mean_sparse_products"] == pytest.approx(2 * config_of["mean_iterations"])
    assert config_of["clipped_seed_hits"] >= 0


def test_the_counters_are_reset_between_the_two_splits(tmp_path):
    """A retriever that kept counting would report two runs as one."""
    data_dir, cache_dir, _, questions = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    written = evaluate(
        tmp_path,
        data_dir,
        cache_dir,
        expansion_dir=tmp_path / "expansion",
        systems=["expansion"],
    )

    by_split = {result["split"]: result["config"] for result in results_of(written)}
    for split, recorded in by_split.items():
        expected = len([question for question in questions if question.split == split])
        assert recorded["n_questions"] == expected


def test_the_two_arms_differ_in_their_numbers_but_not_in_their_configuration(tmp_path):
    data_dir, cache_dir, _, _ = an_expansion_workspace(tmp_path)
    expand(tmp_path, data_dir, cache_dir)

    written = evaluate(
        tmp_path,
        data_dir,
        cache_dir,
        expansion_dir=tmp_path / "expansion",
        systems=["expansion", "expansion-conceptual"],
        splits=("dev",),
    )

    configs = {result["system"]: result["config"] for result in results_of(written)}
    shared = set(configs["expansion"]) - {
        "seed_arm",
        "seed_system",
        "config_digest",
        "mean_iterations",
        "iterations_histogram",
        "stop_reasons",
        "cap_hits",
        "mean_sparse_products",
        "clipped_seed_hits",
        "n_questions",
    }
    for key in shared:
        assert configs["expansion"][key] == configs["expansion-conceptual"][key], key


# --- Phase 5, T3: the pilot stage ----------------------------------------------
#
# `pilot` measures nothing: it records, once, which dev questions leave a gold
# paragraph outside dense's top-10, where every hop will start, and what it must find.


def a_pilot_workspace(tmp_path: Path, n_dev: int = 6, n_test: int = 3):
    """A pool with cached embeddings and questions in both splits, gold chosen blindly."""
    from concept_embeddings_rag.corpus.pool import Question, save_pool

    data_dir, cache_dir, units = an_induction_workspace(tmp_path)
    questions = [
        Question(
            qid=f"q{index}",
            question=f"Which paragraph follows paragraph {index}?",
            answer="-",
            gold_unit_ids=(units[index * 7].unit_id, units[index * 7 + 1].unit_id),
            supporting_facts=((units[index * 7].title, 0), (units[index * 7 + 1].title, 0)),
            split="dev" if index < n_dev else "test",
        )
        for index in range(n_dev + n_test)
    ]
    save_pool(units, questions, data_dir / "pool.json")
    return data_dir, cache_dir, units, questions


def pilot(tmp_path: Path, data_dir: Path, cache_dir: Path, **overrides):
    from concept_embeddings_rag.cli import cmd_pilot

    arguments = {
        "data_dir": data_dir,
        "cache_dir": cache_dir,
        "question_cache_dir": tmp_path / "questions",
        "pilot_dir": tmp_path / "pilot",
        "backend": HashingBackend(),
    }
    arguments.update(overrides)
    return cmd_pilot(**arguments)


def test_parser_exposes_the_pilot_stage():
    assert build_parser().parse_args(["pilot"]).command == "pilot"


def test_pilot_without_a_pool_names_the_build_stage(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        pilot(tmp_path, data_dir, tmp_path / "cache")

    assert "build" in str(excinfo.value)


def test_pilot_without_cached_embeddings_names_the_embed_stage(tmp_path):
    data_dir, _, _, _ = a_pilot_workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        pilot(tmp_path, data_dir, tmp_path / "empty-cache")

    assert "embed" in str(excinfo.value)


def test_pilot_writes_exactly_one_artifact_of_dev_questions_and_prints_its_path(tmp_path, capsys):
    from concept_embeddings_rag.evaluation.pilot import load_pilot

    data_dir, cache_dir, _, questions = a_pilot_workspace(tmp_path)

    path = pilot(tmp_path, data_dir, cache_dir)

    assert sorted(p.name for p in (tmp_path / "pilot").iterdir()) == ["pilot.json"]
    out = capsys.readouterr().out
    assert str(path) in out
    assert out.isascii()
    frozen = load_pilot(tmp_path / "pilot")
    dev_ids = {question.qid for question in questions if question.split == "dev"}
    assert frozen.questions
    assert {question.qid for question in frozen.questions} <= dev_ids


def test_a_second_pilot_refuses_rather_than_moving_the_freeze(tmp_path):
    data_dir, cache_dir, _, _ = a_pilot_workspace(tmp_path)
    pilot(tmp_path, data_dir, cache_dir)

    with pytest.raises(SystemExit) as excinfo:
        pilot(tmp_path, data_dir, cache_dir)

    assert "already frozen" in str(excinfo.value)
