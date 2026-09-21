"""Deviation 8.1 CLI wiring: the laptop preflight before any GPU measurement."""

import json
from pathlib import Path

import numpy as np
import pytest

from concept_embeddings_rag import cli as cli_module
from concept_embeddings_rag import config
from concept_embeddings_rag.corpus import scale_corpus
from concept_embeddings_rag.corpus.fullwiki import FullWikiRecord
from concept_embeddings_rag.corpus.manifest import CorpusManifest
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question, save_pool, unit_id_for
from concept_embeddings_rag.embeddings.cache import unit_set_hash
from concept_embeddings_rag.evaluation import scale_sensitivity


def _unit(title: str, text: str) -> IndexingUnit:
    sentences = (text,)
    return IndexingUnit(unit_id=unit_id_for(title, sentences), title=title, sentences=sentences)


def _record(title: str, text: str, line: int) -> FullWikiRecord:
    return FullWikiRecord(
        title=title,
        sentences=(text,),
        member="AA/wiki_00",
        line=line,
        page_id=None,
    )


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "EXPECTED_N_UNITS", 2)
    monkeypatch.setattr(config, "N_QUESTIONS", 3)
    monkeypatch.setattr(config, "N_DEV", 2)
    monkeypatch.setattr(config, "N_TEST", 1)
    monkeypatch.setattr(config, "PHASE_8_1_CORPUS_SIZES", (2, 3, 4, 6))
    monkeypatch.setattr(config, "PHASE_8_1_DISTRACTOR_PREFIXES", (1, 2, 4))
    monkeypatch.setattr(config, "PHASE_8_1_C19_DEV_SUPPORTED", {"bge": 2, "qwen": 1})

    data_dir = tmp_path / "data"
    target_dir = tmp_path / "phase8_1"
    cache_dir = tmp_path / "cache"
    phase_cache_dir = target_dir / "cache"
    phase_question_cache_dir = phase_cache_dir / "questions"
    data_dir.mkdir()
    target_dir.mkdir()

    c19 = [_unit("C19 A", "Frozen A."), _unit("C19 B", "Frozen B.")]
    questions = [
        Question("d1", "dev one?", "", (c19[0].unit_id,), (), "dev"),
        Question("d2", "dev two?", "", (c19[1].unit_id,), (), "dev"),
        Question("t1", "held out?", "", (c19[0].unit_id,), (), "test"),
    ]
    save_pool(c19, questions, data_dir / "pool.json")
    ids = [unit.unit_id for unit in c19]
    CorpusManifest(
        dataset="synthetic",
        source_url="https://example.invalid/source",
        sha256="0" * 64,
        downloaded_at="2026-09-21T00:00:00+00:00",
        seed=42,
        n_questions=3,
        split_sizes={"dev": 2, "test": 1},
        n_units=2,
        unit_set_hash=unit_set_hash(ids),
    ).save(data_dir / "manifest.json")
    (data_dir / "token_counts.json").write_text(
        json.dumps({unit_id: 10 for unit_id in ids}), encoding="utf-8"
    )

    records = [_record(unit.title, unit.text, index + 1) for index, unit in enumerate(c19)]
    reconciliation = scale_corpus.reconcile(c19, records, dev_gold_ids=frozenset(ids))
    scale_corpus.write_reconciliation(target_dir, reconciliation)

    distractor_records = [
        _record(f"Distractor {index}", f"Added {index}.", 10 + index) for index in range(4)
    ]
    selected = scale_corpus.select_distractors(
        lambda: iter(distractor_records),
        mapped_ids=reconciliation.mapped_official_ids,
        excluded_titles=frozenset(),
        limit=4,
    )
    source_sha = "a" * 64
    scale_corpus.freeze_selection(
        target_dir,
        c19_unit_ids=ids,
        selected=selected,
        source_sha256=source_sha,
        reconciliation_digest=reconciliation.digest,
        expected_total=4,
    )
    scale_corpus.write_token_counts(
        target_dir,
        {item.unit_id: 10 for item in selected},
        selection_digest=scale_corpus.selection_digest_of([item.unit_id for item in selected]),
    )
    (target_dir / "source.json").write_text(
        json.dumps(
            {
                "bytes_agree": True,
                "md5_agree": True,
                "measured_sha256": source_sha,
                "page_checked": False,
            }
        ),
        encoding="utf-8",
    )
    (target_dir / scale_sensitivity.REPRODUCTION_FILENAME).write_text(
        json.dumps(
            {
                "level": 1,
                "corpus": "c19",
                "corpus_units": 2,
                "questions": 2,
                "required": {"bge": 2, "qwen": 1},
                "models": {
                    "bge": {"supported": 2, "required": 2, "matches": True},
                    "qwen": {"supported": 1, "required": 1, "matches": True},
                },
                "matches": True,
                "terminal_state": None,
            }
        ),
        encoding="utf-8",
    )
    return data_dir, target_dir, cache_dir, phase_cache_dir, phase_question_cache_dir


def test_scale_run_parser_requires_one_frozen_model_and_exposes_cpu_preflight() -> None:
    parser = cli_module.build_parser()
    parsed = parser.parse_args(["scale-run", "--model", "bge", "--preflight-only"])

    assert parsed.command == "scale-run"
    assert parsed.model == "bge"
    assert parsed.preflight_only is True
    with pytest.raises(SystemExit):
        parser.parse_args(["scale-run", "--model", "other", "--preflight-only"])


def test_scale_outcome_parser_is_exposed() -> None:
    parsed = cli_module.build_parser().parse_args(["scale-outcome"])
    assert parsed.command == "scale-outcome"


def test_scale_outcome_loads_both_complete_runs_and_writes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _data_dir, target_dir, _cache_dir, _phase_cache_dir, _question_cache_dir = _workspace(
        tmp_path, monkeypatch
    )
    same_questions = "f" * 64
    for model, supported in (("bge", (2, 2, 2, 2)), ("qwen", (1, 1, 1, 1))):
        body = {
            "model": model,
            "questions": 2,
            "sizes": [2, 3, 4, 6],
            "scales": {
                str(size): {"supported": count, "units": size}
                for size, count in zip((2, 3, 4, 6), supported, strict=True)
            },
            "own_drop_c19_to_c500": supported[0] - supported[-1],
            "level_2": {"difference": 0, "within_tolerance": True, "caveat": None},
            "provenance": {"question_set_hash": same_questions},
            "terminal_state": None,
        }
        (target_dir / scale_sensitivity.scale_filename(model)).write_text(
            json.dumps(body), encoding="utf-8"
        )

    path = cli_module.cmd_scale_outcome(target_dir)
    assert path == target_dir / scale_sensitivity.OUTCOME_FILENAME
    assert path.exists()
    with pytest.raises(SystemExit, match="already records an outcome"):
        cli_module.cmd_scale_outcome(target_dir)


def test_local_preflight_validates_both_models_without_loading_or_writing_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, target_dir, cache_dir, phase_cache_dir, phase_question_cache_dir = _workspace(
        tmp_path, monkeypatch
    )

    def forbidden_model(*args, **kwargs):
        raise AssertionError("the laptop preflight must not construct an embedding model")

    monkeypatch.setattr(cli_module, "SentenceTransformerBackend", forbidden_model)

    for model in ("bge", "qwen"):
        summary = cli_module.cmd_scale_run(
            model,
            preflight_only=True,
            data_dir=data_dir,
            target_dir=target_dir,
            cache_dir=cache_dir,
            phase_cache_dir=phase_cache_dir,
            phase_question_cache_dir=phase_question_cache_dir,
        )
        assert summary["model"] == model
        assert summary["c19_units"] == 2
        assert summary["c500_units"] == 6
        assert summary["dev_questions"] == 2
        assert summary["reproduction_matches"] is True
        assert not (target_dir / scale_sensitivity.scale_filename(model)).exists()

    assert not list(phase_cache_dir.glob("embeddings-*.npz"))
    assert not list(phase_question_cache_dir.glob("embeddings-*.npz"))



def _allow_measurement_host(
    monkeypatch: pytest.MonkeyPatch, *, supported: int = 2
) -> None:
    monkeypatch.setattr(cli_module, "_scale_cuda_available", lambda: True)
    monkeypatch.setattr(cli_module, "_scale_free_bytes", lambda path: 100 * 1024**3)
    monkeypatch.setattr(
        cli_module.local_extraction,
        "hardware_block",
        lambda **kwargs: {"device": kwargs["device"], "test": True},
    )
    monkeypatch.setattr(
        cli_module,
        "_historical_dense",
        lambda *args, **kwargs: (object(), {"top_k": config.EVALUATION_TOP_K}),
    )
    monkeypatch.setattr(
        scale_sensitivity,
        "measure_level_one_on_measurement_host",
        lambda **kwargs: {
            "supported": supported,
            "n_questions": config.N_DEV,
            "matches": True,
            "host": kwargs["host"],
            "run_digest": "d" * 64,
        },
    )


def test_measurement_host_preflight_refuses_without_cuda_before_d_l1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def historical(*args, **kwargs):
        nonlocal called
        called = True
        return object(), {}

    monkeypatch.setattr(cli_module, "_scale_cuda_available", lambda: False)
    monkeypatch.setattr(cli_module, "_historical_dense", historical)
    with pytest.raises(SystemExit, match="requires CUDA"):
        cli_module._scale_measurement_host_preflight(
            "bge",
            units=[],
            questions=[],
            historical_token_counts={},
            cache_dir=tmp_path / "cache",
            question_cache_dir=tmp_path / "questions",
            target_dir=tmp_path,
        )
    assert called is False


class _CountingBackend:
    def __init__(self, *, name: str, revision: str, dim: int, query_prompt: str = "") -> None:
        self.name = name
        self.revision = revision
        self.dim = dim
        self.normalize = True
        self._query_prompt = query_prompt
        self.calls: list[list[str]] = []

    @property
    def resolved_query_prompt(self) -> str:
        return self._query_prompt

    def resolved_revision(self) -> str:
        return self.revision

    def encode(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            if text in {"C19 A. Frozen A.", "dev one?"}:
                vectors[row, 0] = 1.0
            elif text in {"C19 B. Frozen B.", "dev two?"}:
                vectors[row, 1] = 1.0
            else:
                vectors[row, 2] = 1.0
        return vectors


def test_real_scale_run_encodes_c500_and_dev_once_then_measures_four_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, target_dir, cache_dir, phase_cache_dir, phase_question_cache_dir = _workspace(
        tmp_path, monkeypatch
    )
    pins = scale_sensitivity.ModelPins.for_key("bge")
    corpus_backend = _CountingBackend(
        name=pins.model, revision=pins.revision, dim=pins.dim
    )
    query_backend = _CountingBackend(
        name=pins.model,
        revision=pins.revision,
        dim=pins.dim,
        query_prompt=pins.query_prompt,
    )
    snapshot = tmp_path / pins.revision
    snapshot.mkdir()
    monkeypatch.setattr(cli_module, "weights_sha256", lambda path: "f" * 64)
    monkeypatch.setattr(cli_module, "_scale_peak_rss_mb", lambda: 123.5)
    _allow_measurement_host(monkeypatch, supported=2)

    path = cli_module.cmd_scale_run(
        "bge",
        data_dir=data_dir,
        target_dir=target_dir,
        cache_dir=cache_dir,
        phase_cache_dir=phase_cache_dir,
        phase_question_cache_dir=phase_question_cache_dir,
        corpus_backend=corpus_backend,
        query_backend=query_backend,
        resolve_snapshot=lambda model, revision: snapshot,
    )

    assert path == target_dir / scale_sensitivity.scale_filename("bge")
    assert [len(call) for call in corpus_backend.calls] == [6]
    assert [len(call) for call in query_backend.calls] == [2]

    embedding = scale_sensitivity.load_embedding(target_dir, "bge")
    assert embedding["corpus_units"] == 6
    assert embedding["n_questions"] == 2
    assert embedding["selection_digest"]
    assert embedding["corpus_cache_key"] != embedding["question_cache_key"]

    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["sizes"] == [2, 3, 4, 6]
    assert result["level_2"]["within_tolerance"] is True
    assert result["terminal_state"] is None
    assert result["retrieval_cost"]["probe_corpus"] == 6
    assert result["retrieval_cost"]["mean_seconds_per_query"] >= 0.0
    assert result["retrieval_cost"]["peak_rss_mb"] == 123.5
    assert result["retrieval_cost"]["exceeds_query_seconds_ceiling"] is False
    run_files = [result["scales"][str(size)]["run_file"] for size in (2, 3, 4, 6)]
    assert len(set(run_files)) == 4
    assert all(result["scales"][str(size)]["run_digest"] for size in (2, 3, 4, 6))

    units, _ = cli_module.load_pool(data_dir / "pool.json")
    corpus = scale_corpus.load_selection(target_dir, c19_units=units)
    c500_ids = [unit.unit_id for unit in (*corpus.c19, *corpus.distractors)]
    for size, run_file in zip((2, 3, 4, 6), run_files, strict=True):
        run = json.loads((target_dir / run_file).read_text(encoding="utf-8"))
        assert run["config"]["unit_set_hash"] == cli_module.unit_set_hash(c500_ids[:size])
        assert run["config"]["seed"] == config.DEFAULT_SEED
        assert run["config"]["tokenizer"] == config.BUDGET_TOKENIZER_ID
        assert run["config"]["code_version"] == cli_module.__version__


def test_real_scale_run_refuses_an_orphan_8_1_cache_before_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, target_dir, cache_dir, phase_cache_dir, phase_question_cache_dir = _workspace(
        tmp_path, monkeypatch
    )
    pins = scale_sensitivity.ModelPins.for_key("bge")
    snapshot = tmp_path / pins.revision
    snapshot.mkdir()
    monkeypatch.setattr(cli_module, "weights_sha256", lambda path: "f" * 64)
    _allow_measurement_host(monkeypatch, supported=2)

    units, _ = cli_module.load_pool(data_dir / "pool.json")
    corpus = scale_corpus.load_selection(target_dir, c19_units=units)
    c500 = [*corpus.c19, *corpus.distractors]
    key_backend = scale_sensitivity.MetadataBackend(name=pins.model, revision=pins.revision)
    key = cli_module._cache_key_for(key_backend, c500)
    orphan = cli_module.EmbeddingCache(phase_cache_dir)
    orphan.path_for(key).write_bytes(b"partial")

    corpus_backend = _CountingBackend(
        name=pins.model, revision=pins.revision, dim=pins.dim
    )
    query_backend = _CountingBackend(
        name=pins.model,
        revision=pins.revision,
        dim=pins.dim,
        query_prompt=pins.query_prompt,
    )
    with pytest.raises(SystemExit, match="cache exists without its embedding artifact"):
        cli_module.cmd_scale_run(
            "bge",
            data_dir=data_dir,
            target_dir=target_dir,
            cache_dir=cache_dir,
            phase_cache_dir=phase_cache_dir,
            phase_question_cache_dir=phase_question_cache_dir,
            corpus_backend=corpus_backend,
            query_backend=query_backend,
            resolve_snapshot=lambda model, revision: snapshot,
        )
    assert corpus_backend.calls == []
    assert query_backend.calls == []
