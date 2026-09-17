"""Phase 6, T16-T18: the three stages on a toy pipeline, in `tmp_path` (D1, D11, D12, D14).

The toy pipeline is a whole Phase 1-5 pipeline written by the project's own writers
(`tests/evaluation/replacement_fixtures.py`), so the stages verify it exactly as they verify the
real one. Nothing here reads or writes the real `data/`.
"""

import ast
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "evaluation"))

from replacement_fixtures import (  # noqa: E402
    ToyPipeline,
    build_toy_pipeline,
    relocated,
    rewrite_json,
    toy_counter,
)

from concept_embeddings_rag import cli  # noqa: E402
from concept_embeddings_rag.embeddings import cache as cache_module  # noqa: E402
from concept_embeddings_rag.embeddings.cache import (  # noqa: E402
    EmbeddingCache,
    question_cache_key,
    question_set_hash,
)
from concept_embeddings_rag.evaluation import (  # noqa: E402
    replacement_inputs,
    replacement_run,
    selection,
)
from concept_embeddings_rag.evaluation.replacement_run import (  # noqa: E402
    CHECKS_FILENAME,
    ReplacementRunError,
    StageEnvironment,
    load_checks,
    run_check,
)

CLI = Path(cli.__file__)


class RecordingMapping(Mapping):
    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = dict(data)
        self.accesses = 0

    def __getitem__(self, key: str) -> Any:
        self.accesses += 1
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        self.accesses += 1
        return iter(self._data)

    def __len__(self) -> int:
        self.accesses += 1
        return len(self._data)


class RecordingParser:
    def __init__(self) -> None:
        self.recorded: dict[str, tuple[RecordingMapping, RecordingMapping]] = {}

    def __call__(self, text: str) -> dict[str, Any]:
        document = json.loads(text)
        metrics, cost = RecordingMapping(document["metrics"]), RecordingMapping(document["cost"])
        document["metrics"], document["cost"] = metrics, cost
        self.recorded[f"{document['system']}/{document['split']}"] = (metrics, cost)
        return document


@pytest.fixture(scope="module")
def pristine(tmp_path_factory) -> ToyPipeline:
    return build_toy_pipeline(tmp_path_factory.mktemp("pristine") / "toy")


@pytest.fixture
def toy(pristine, tmp_path) -> ToyPipeline:
    return relocated(pristine, tmp_path / "toy")


def environment(pipeline: ToyPipeline, **overrides: Any) -> StageEnvironment:
    arguments: dict[str, Any] = {
        "paths": pipeline.paths,
        "pins": pipeline.pins,
        "replacement_dir": pipeline.replacement_dir,
        "backend": pipeline.backend,
        "token_counter": toy_counter,
    }
    arguments.update(overrides)
    return StageEnvironment(**arguments)


def files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern)) if directory.exists() else []


# --- The CLI ---------------------------------------------------------------------------------


def test_the_parser_exposes_the_three_replacement_stages():
    parser = cli.build_parser()
    for stage in ("replace-check", "replace-freeze", "replace-test"):
        assert parser.parse_args([stage]).command == stage


def calls_named(source: str, name: str) -> int:
    return sum(
        1
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    )


def test_cli_gains_no_fitting_call_and_no_hybrid_construction():
    source = CLI.read_text(encoding="utf-8")
    assert calls_named(source, "fit_fusion_weight") == 2
    assert calls_named(source, "FusedRetriever") == 1


def test_the_cli_command_returns_non_zero_when_the_stage_fails(toy, monkeypatch):
    def failing(environment):
        raise ReplacementRunError("an inherited input does not verify: toy")

    monkeypatch.setattr(cli, "run_check", failing)
    assert cli.cmd_replace_check(environment(toy)) == 1


# --- replace-check ----------------------------------------------------------------------------


def test_passing_inputs_write_checks_dev_with_mode_reused(toy):
    result = run_check(environment(toy))

    assert result.passed, result.message
    checks = load_checks(toy.replacement_dir)
    assert checks["passed"] is True
    assert checks["control"]["mode"] == "reused"
    assert checks["continuity"]["passed"] is True
    assert set(checks["runs"]) == {"dense", "hybrid-bm25"}
    assert files(toy.replacement_dir, "run-dense-dev-*.json")
    assert files(toy.replacement_dir, "outcomes-hybrid-bm25-dev.json")
    assert not files(toy.replacement_dir, "*hybrid-entity-hop*")
    assert not files(toy.paths.results_dir, "*phase*")


def test_the_checks_record_every_reproduction_check_and_the_identities(toy):
    run_check(environment(toy))
    checks = load_checks(toy.replacement_dir)
    names = [check["name"] for check in checks["control"]["reproduction"]["checks"]]
    assert "control rrf_score" in names
    assert "hybrid-bm25/dev budget_2048 full_support" in names
    assert "dense/test config seed" in names
    historical = checks["identities"]["historical"]
    assert set(historical) == {"dense/dev", "dense/test", "hybrid-bm25/dev", "hybrid-bm25/test"}
    assert all("metrics" not in identity for identity in historical.values())


def test_the_results_carry_the_phase_6_provenance(toy):
    run_check(environment(toy))
    result = json.loads(files(toy.replacement_dir, "run-dense-dev-*.json")[0].read_text("utf-8"))
    config = result["config"]
    assert config["phase"] == 6
    assert config["unit_set_hash"] == toy.pins.unit_set_hash
    assert set(config["libraries"]) == {"numpy", "scipy", "bm25s", "transformers", "tokenizers"}
    assert config["dense_identity"]["corpus_cache_key"]


def swap_two_pilot_question_vectors(pipeline: ToyPipeline) -> None:
    """Change D(q) for pilot questions without touching any pinned artifact."""
    dev = [question for question in pipeline.questions if question.split == "dev"]
    qids = [question.qid for question in dev]
    key = question_cache_key(
        pipeline.pins.model, pipeline.pins.revision, question_set_hash(qids), "dev", True
    )
    cache = EmbeddingCache(pipeline.paths.question_cache_dir)
    loaded = cache.load(key, expected_unit_ids=qids)
    assert loaded is not None
    vectors, ids = loaded
    shuffled = np.roll(vectors, 1, axis=0)
    sidecar = json.loads(cache.sidecar_for(key).read_text(encoding="utf-8"))
    metadata = {
        name: sidecar[name]
        for name in ("model", "revision", "resolved_revision", "dim", "normalized", "split")
    }
    cache.save(key, shuffled, ids, metadata)


def test_a_continuity_failure_exits_before_any_fit(toy, monkeypatch):
    swap_two_pilot_question_vectors(toy)
    fits: list[Any] = []
    monkeypatch.setattr(replacement_run, "fit_fusion_weight", lambda *a, **k: fits.append(a))

    result = run_check(environment(toy))

    assert result.passed is False
    assert "continuity" in result.message
    assert fits == []
    checks = load_checks(toy.replacement_dir)
    assert checks["continuity"]["passed"] is False
    assert checks["control"] is None
    assert not files(toy.replacement_dir, "run-*.json")


def test_a_control_mismatch_writes_the_failed_checks_and_leaves_no_b_file(toy):
    name = toy.pins.historical_dev_files["hybrid-bm25"]

    def nudge(payload):
        payload["metrics"]["budget_2048"]["full_support"] += 1e-9

    rewrite_json(toy.paths.results_dir / name, nudge)
    result = run_check(environment(toy))

    assert result.passed is False
    checks = load_checks(toy.replacement_dir)
    assert checks["control"]["mode"] == "re-measured"
    failed = [c["name"] for c in checks["control"]["reproduction"]["checks"] if not c["passed"]]
    assert failed == ["hybrid-bm25/dev budget_2048 full_support"]
    assert not files(toy.replacement_dir, "*hybrid-entity-hop*")


def test_the_cli_exit_status_follows_the_checks(toy, monkeypatch):
    name = toy.pins.historical_dev_files["dense"]
    rewrite_json(toy.paths.results_dir / name, lambda p: p["metrics"].update({"recall_at_2": 0.0}))
    assert cli.cmd_replace_check(environment(toy)) == 1


def test_only_dev_questions_reach_the_harness_and_no_test_backend_is_built(toy, monkeypatch):
    splits: list[str] = []
    backends: list[set[str]] = []
    original_evaluate = selection.evaluate_retriever
    original_run_evaluate = replacement_run.evaluate_retriever
    original_backend = cache_module.build_query_backend

    def spy_evaluate(original):
        def wrapped(retriever, questions, *args, **kwargs):
            splits.extend(question.split for question in questions)
            return original(retriever, questions, *args, **kwargs)

        return wrapped

    def spy_backend(questions, *args, **kwargs):
        backends.append({question.split for question in questions})
        return original_backend(questions, *args, **kwargs)

    monkeypatch.setattr(selection, "evaluate_retriever", spy_evaluate(original_evaluate))
    monkeypatch.setattr(replacement_run, "evaluate_retriever", spy_evaluate(original_run_evaluate))
    monkeypatch.setattr(replacement_inputs, "build_query_backend", spy_backend)

    assert run_check(environment(toy)).passed
    assert splits and set(splits) == {"dev"}
    assert backends == [{"dev"}]


def test_the_historical_test_files_are_identified_and_their_figures_never_accessed(toy):
    parser = RecordingParser()
    assert run_check(environment(toy, parse=parser)).passed
    for system in ("dense", "hybrid-bm25"):
        metrics, cost = parser.recorded[f"{system}/test"]
        assert metrics.accesses == 0 and cost.accesses == 0


def test_a_second_check_is_refused_before_anything_is_measured(toy):
    run_check(environment(toy))
    before = sorted(path.name for path in toy.replacement_dir.iterdir())
    with pytest.raises(ReplacementRunError, match="already"):
        run_check(environment(toy))
    assert sorted(path.name for path in toy.replacement_dir.iterdir()) == before
    assert (toy.replacement_dir / CHECKS_FILENAME).exists()


def test_a_modified_checks_file_is_refused_on_load(toy):
    run_check(environment(toy))
    rewrite_json(
        toy.replacement_dir / CHECKS_FILENAME, lambda p: p.update({"passed": True, "x": 1})
    )
    with pytest.raises(ReplacementRunError, match="digest"):
        load_checks(toy.replacement_dir)
