"""Phase 6, T16-T18: the three stages on a toy pipeline, in `tmp_path` (D1, D11, D12, D14).

The toy pipeline is a whole Phase 1-5 pipeline written by the project's own writers
(`tests/evaluation/replacement_fixtures.py`), so the stages verify it exactly as they verify the
real one. Nothing here reads or writes the real `data/`.
"""

import ast
import importlib.util
import json
import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType, ModuleType
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
    tango,
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


# --- replace-freeze ---------------------------------------------------------------------------

from concept_embeddings_rag.evaluation.replacement_freeze import (  # noqa: E402
    FreezeError,
    load_freeze,
)
from concept_embeddings_rag.evaluation.replacement_run import run_freeze  # noqa: E402
from concept_embeddings_rag.evaluation.selection import FusionFit  # noqa: E402


@pytest.fixture(scope="module")
def checked(pristine, tmp_path_factory) -> ToyPipeline:
    """The toy pipeline after a passing `replace-check`."""
    pipeline = relocated(pristine, tmp_path_factory.mktemp("checked") / "toy")
    assert run_check(environment(pipeline)).passed
    return pipeline


@pytest.fixture
def after_check(checked, tmp_path) -> ToyPipeline:
    return relocated(checked, tmp_path / "toy")


def nudge_historical_dev_precision(pipeline: ToyPipeline) -> None:
    name = pipeline.pins.historical_dev_files["hybrid-bm25"]
    rewrite_json(
        pipeline.paths.results_dir / name,
        lambda p: p["metrics"]["budget_2048"].update({"precision": 0.0}),
    )


def test_freeze_refuses_without_checks(toy):
    with pytest.raises(ReplacementRunError, match="replace-check"):
        run_freeze(environment(toy))


def test_freeze_writes_b_the_reproducibility_block_the_freeze_and_the_dev_readings(after_check):
    result = run_freeze(environment(after_check))
    directory = after_check.replacement_dir

    assert result.passed, result.message
    freeze = load_freeze(directory)
    assert result.path == freeze.path
    assert freeze.control_mode == "reused"
    assert files(directory, "run-hybrid-entity-hop-dev-*.json")
    assert files(directory, "outcomes-hybrid-entity-hop-dev.json")
    assert freeze.payload["dev_results"]["reproducibility"]["passed"] is True
    for name in ("diagnostics-dev.json", "traces-dev.json", "readout-dev.json"):
        payload = json.loads((directory / name).read_text(encoding="utf-8"))
        assert payload["freeze_digest"] == freeze.digest
    assert not files(directory, "*-test*")


def test_bs_selected_point_aggregates_equal_the_fits_reading(after_check):
    run_freeze(environment(after_check))
    freeze = load_freeze(after_check.replacement_dir).payload
    fit = freeze["entity_fit"]
    label = "rrf" if fit["winning_scheme"] == "rrf" else "w={:.1f}".format(fit["best_weight"])
    b = freeze["dev_results"]["systems"]["hybrid-entity-hop"]
    assert b["metrics"] == fit["points"][label]["metrics"]


def landing(monkeypatch, *, weight: float | None) -> None:
    """Keep the real fit and its real points; replace the curve so the fit lands where asked."""
    original = replacement_run.fit_fusion_weight

    def fitted(components, **kwargs):
        real = original(components, **kwargs)
        if real.components[1] != "entity-hop":
            return real
        if weight is None:
            curve = dict.fromkeys(real.grid, 0.0)
            rrf = 1.0
        else:
            curve = {w: (1.0 if w == weight else 0.0) for w in real.grid}
            rrf = 0.5
        return FusionFit(**{**real.__dict__, "curve": curve, "rrf_score": rrf})

    monkeypatch.setattr(replacement_run, "fit_fusion_weight", fitted)


def test_a_fit_landing_on_dense_alone_freezes_normally(after_check, monkeypatch):
    landing(monkeypatch, weight=1.0)
    result = run_freeze(environment(after_check))
    assert result.passed, result.message
    fit = load_freeze(after_check.replacement_dir).payload["entity_fit"]
    assert (fit["winning_scheme"], fit["best_weight"]) == ("weighted", 1.0)


def test_a_fit_where_rrf_wins_freezes_normally(after_check, monkeypatch):
    landing(monkeypatch, weight=None)
    result = run_freeze(environment(after_check))
    assert result.passed, result.message
    freeze = load_freeze(after_check.replacement_dir).payload
    assert freeze["entity_fit"]["winning_scheme"] == "rrf"
    assert freeze["n_dev_decisions"] == 1


def test_a_second_freeze_invocation_refuses(after_check):
    run_freeze(environment(after_check))
    with pytest.raises(ReplacementRunError, match="freeze"):
        run_freeze(environment(after_check))


def test_freeze_refuses_a_failed_check_without_the_re_measured_mode(toy):
    nudge_historical_dev_precision(toy)
    assert run_check(environment(toy)).passed is False
    with pytest.raises(ReplacementRunError, match="re-measured"):
        run_freeze(environment(toy))
    with pytest.raises(ReplacementRunError, match="deviation"):
        run_freeze(environment(toy), control_mode="re-measured")
    with pytest.raises(ReplacementRunError, match="deviation"):
        run_freeze(
            environment(toy), control_mode="re-measured", deviation=str(toy.root / "missing.md")
        )


def test_re_measured_with_an_existing_deviation_freezes_a_remeasured_control(toy):
    nudge_historical_dev_precision(toy)
    assert run_check(environment(toy)).passed is False
    deviation = toy.root / "6.1_control_mismatch.md"
    deviation.write_text("# 6.1 toy deviation", encoding="utf-8")

    result = run_freeze(environment(toy), control_mode="re-measured", deviation=str(deviation))

    assert result.passed, result.message
    freeze = load_freeze(toy.replacement_dir).payload
    assert freeze["control"]["mode"] == "re-measured"
    assert freeze["control"]["deviation"] == str(deviation)
    assert files(toy.replacement_dir, "outcomes-hybrid-bm25-dev-2.json")
    entity_decisions = [d for d in freeze["decisions"] if d["hybrid"] == "hybrid-entity-hop"]
    assert freeze["n_dev_decisions"] > len(entity_decisions)


def test_re_measured_is_refused_when_the_checks_passed(after_check, tmp_path):
    deviation = tmp_path / "6.1.md"
    deviation.write_text("#", encoding="utf-8")
    with pytest.raises(ReplacementRunError, match="passed"):
        run_freeze(environment(after_check), control_mode="re-measured", deviation=str(deviation))


def test_freeze_puts_only_dev_questions_through_the_harness_and_reads_no_test_metric(
    after_check, monkeypatch
):
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
    parser = RecordingParser()

    assert run_freeze(environment(after_check, parse=parser)).passed
    assert splits and set(splits) == {"dev"}
    assert all(split_set == {"dev"} for split_set in backends)
    for system in ("dense", "hybrid-bm25"):
        metrics, cost = parser.recorded[f"{system}/test"]
        assert metrics.accesses == 0 and cost.accesses == 0


def test_the_cli_freeze_command_reports_a_refusal(toy):
    assert cli.cmd_replace_freeze(environment(toy)) == 1


def test_the_loaded_freeze_refuses_tampering(after_check):
    run_freeze(environment(after_check))
    path = after_check.replacement_dir / "freeze.json"
    rewrite_json(path, lambda p: p.update({"seed": 7}))
    with pytest.raises(FreezeError, match="digest"):
        load_freeze(after_check.replacement_dir)


# --- replace-test -----------------------------------------------------------------------------

from concept_embeddings_rag.evaluation.outcomes import PerQuestionOutcomes  # noqa: E402
from concept_embeddings_rag.evaluation.replacement_freeze import Phase6Freeze  # noqa: E402
from concept_embeddings_rag.evaluation.replacement_run import (  # noqa: E402
    TEST_REPRODUCTION_FILENAME,
    confirm_delta_source,
    read_historical_test_figures,
    run_test,
)
from concept_embeddings_rag.evaluation.selection import digest_of_payload  # noqa: E402

N_TEST = 1400


def frozen_copy(source: ToyPipeline, root: Path) -> ToyPipeline:
    pipeline = relocated(source, root)
    return pipeline


@pytest.fixture(scope="module")
def frozen_small(checked, tmp_path_factory) -> ToyPipeline:
    pipeline = relocated(checked, tmp_path_factory.mktemp("frozen-small") / "toy")
    assert run_freeze(environment(pipeline)).passed
    return pipeline


@pytest.fixture(scope="module")
def frozen_full(tmp_path_factory) -> ToyPipeline:
    """A toy whose test split holds the decision's whole population of 1,400 questions."""
    pipeline = build_toy_pipeline(tmp_path_factory.mktemp("full") / "toy", n_test=N_TEST)
    assert run_check(environment(pipeline)).passed
    assert run_freeze(environment(pipeline)).passed
    return pipeline


@pytest.fixture
def small(frozen_small, tmp_path) -> ToyPipeline:
    return relocated(frozen_small, tmp_path / "toy")


@pytest.fixture
def full(frozen_full, tmp_path) -> ToyPipeline:
    return relocated(frozen_full, tmp_path / "toy")


def publish_example(monkeypatch) -> None:
    """Stub the example check for the toy; the real `tango.PUBLISHED_EXAMPLE` is not touched."""
    monkeypatch.setattr(
        replacement_run, "require_published_example", lambda: {"source": "toy injection"}
    )


def delta_confirmed(monkeypatch) -> None:
    """The toy's historical counts are not 1,210 and 1,155; the check itself is tested apart."""
    monkeypatch.setattr(
        replacement_run,
        "confirm_delta_source",
        lambda figures, parameters: {"passed": True, "note": "toy injection"},
    )


def b_files(directory: Path) -> list[Path]:
    return [
        *files(directory, "run-hybrid-entity-hop-test-*.json"),
        *files(directory, "outcomes-hybrid-entity-hop-test*.json"),
        *files(directory, "diagnostics-test*.json"),
        *files(directory, "traces-test*.json"),
        *files(directory, "decision.json"),
    ]


def a_deviation(pipeline: ToyPipeline, name: str = "6.2_test_mismatch.md") -> str:
    path = pipeline.root / name
    path.write_text("# toy deviation", encoding="utf-8")
    return str(path)


def nudge_test_reproduction(patch: pytest.MonkeyPatch) -> None:
    """One historical test figure off by 1e-9: the reused reproduction fails as a real one would."""
    original = replacement_run.read_historical_test_figures

    def nudged(frozen, results_dir):
        figures = original(frozen, results_dir)
        figures["dense"]["metrics"]["recall_at_2"] += 1e-9
        return figures

    patch.setattr(replacement_run, "read_historical_test_figures", nudged)


@pytest.fixture(scope="module")
def full_on_the_branch(frozen_full, tmp_path_factory) -> ToyPipeline:
    """The full toy on D14's deviation branch (OI-4): a reused test run stopped by a failed
    reproduction before B, a deviation document, and a superseding re-measured freeze."""
    pipeline = relocated(frozen_full, tmp_path_factory.mktemp("branch") / "toy")
    stop_on_a_failed_reproduction(pipeline)
    write_superseding_freeze(pipeline)
    return pipeline


@pytest.fixture
def on_the_branch(full_on_the_branch, tmp_path) -> ToyPipeline:
    return relocated(full_on_the_branch, tmp_path / "toy")


def branch_deviation(pipeline: ToyPipeline) -> str:
    """The deviation document the superseding freeze names."""
    return str(load_freeze(pipeline.replacement_dir).payload["deviation"])


# --- Step 1 refusals --------------------------------------------------------------------------


def test_replace_test_refuses_without_a_verified_freeze(after_check, monkeypatch):
    publish_example(monkeypatch)
    with pytest.raises(ReplacementRunError, match="freeze"):
        run_test(environment(after_check))


def test_replace_test_refuses_an_altered_decision_parameter_copy(small, monkeypatch):
    publish_example(monkeypatch)
    path = small.replacement_dir / "freeze.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decision_parameters"]["alpha"] = 0.025
    payload["digest"] = digest_of_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ReplacementRunError, match="decision parameters"):
        run_test(environment(small))


def test_replace_test_refuses_while_the_published_example_slot_is_empty(small, monkeypatch):
    # T11 filled the real slot; the refusal is still the stage's, so the slot is emptied here.
    monkeypatch.setattr(tango, "PUBLISHED_EXAMPLE", MappingProxyType({}))
    read: list[Any] = []
    monkeypatch.setattr(
        replacement_run, "read_historical_test_figures", lambda *a, **k: read.append(a)
    )
    with pytest.raises(ReplacementRunError, match="published"):
        run_test(environment(small))
    assert read == []
    assert not files(small.replacement_dir, "*-test-*.json")


@pytest.mark.parametrize(
    "planted", ["decision.json", "run-hybrid-entity-hop-test-20260917T000000+0000.json"]
)
def test_replace_test_refuses_when_a_decision_or_a_b_test_file_exists(small, monkeypatch, planted):
    publish_example(monkeypatch)
    (small.replacement_dir / planted).write_text("{}", encoding="utf-8")
    with pytest.raises(ReplacementRunError, match="exists"):
        run_test(environment(small))
    assert not files(small.replacement_dir, "run-dense-test-*.json")


# --- The only reader of historical test figures -----------------------------------------------


def test_the_reader_refuses_without_a_freeze(small):
    with pytest.raises(ReplacementRunError, match="freeze"):
        read_historical_test_figures({"control": {}}, small.paths.results_dir)  # type: ignore[arg-type]


def test_the_reader_refuses_bytes_that_differ_from_the_freeze_record(small):
    frozen = load_freeze(small.replacement_dir)
    name = small.pins.historical_test_files["dense"]
    path = small.paths.results_dir / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ReplacementRunError, match="sha256"):
        read_historical_test_figures(frozen, small.paths.results_dir)


def test_the_reader_returns_the_two_test_files_figures_under_the_freeze(small):
    frozen = load_freeze(small.replacement_dir)
    assert isinstance(frozen, Phase6Freeze)
    figures = read_historical_test_figures(frozen, small.paths.results_dir)
    assert set(figures) == {"dense", "hybrid-bm25"}
    assert "budget_2048" in figures["dense"]["metrics"]


# --- The delta confirmation -------------------------------------------------------------------


def historical_pair(control: int, dense: int, n: int = N_TEST) -> dict[str, dict]:
    def one(count: int) -> dict:
        return {
            "metrics": {"budget_2048": {"full_support": count / n}},
            "cost": {"n_questions": n},
        }

    return {"hybrid-bm25": one(control), "dense": one(dense)}


def parameters() -> dict:
    from concept_embeddings_rag.evaluation.replacement_decision import parameters_payload

    return json.loads(json.dumps(parameters_payload()))


def test_the_delta_source_is_confirmed_when_the_pair_differs_by_m1():
    record = confirm_delta_source(historical_pair(1210, 1155), parameters())
    assert record["passed"] is True
    assert (record["control_count"], record["dense_count"], record["difference"]) == (
        1210,
        1155,
        55,
    )


@pytest.mark.parametrize(("control", "dense"), [(1209, 1155), (1210, 1154), (1000, 1000)])
def test_a_pair_whose_difference_is_not_m1_is_not_confirmed(control, dense):
    assert confirm_delta_source(historical_pair(control, dense), parameters())["passed"] is False


def test_a_pair_whose_mean_is_not_a_count_of_the_population_is_not_confirmed():
    pair = historical_pair(1210, 1155)
    pair["dense"]["metrics"]["budget_2048"]["full_support"] += 1e-6
    assert confirm_delta_source(pair, parameters())["passed"] is False


def test_an_unconfirmed_delta_source_stops_before_b_in_reused_mode(small, monkeypatch):
    publish_example(monkeypatch)
    result = run_test(environment(small))

    assert result.passed is False
    assert "delta" in result.message
    assert b_files(small.replacement_dir) == []
    record = json.loads((small.replacement_dir / TEST_REPRODUCTION_FILENAME).read_text("utf-8"))
    assert record["delta_confirmation"]["passed"] is False


def test_an_unconfirmed_delta_source_stops_before_b_on_the_deviation_branch_too(
    on_the_branch, monkeypatch
):
    publish_example(monkeypatch)
    result = run_test(
        environment(on_the_branch),
        control_mode="re-measured",
        deviation=branch_deviation(on_the_branch),
    )

    assert result.passed is False
    assert "delta" in result.message
    assert b_files(on_the_branch.replacement_dir) == []
    record = json.loads(
        (on_the_branch.replacement_dir / "test-reproduction-2.json").read_text("utf-8")
    )
    assert record["mode"] == "re-measured"
    assert record["delta_confirmation"]["passed"] is False


# --- The ordered protocol ---------------------------------------------------------------------


def test_a_failed_reproduction_in_reused_mode_stops_with_no_b_figure(small, monkeypatch):
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)
    nudge_test_reproduction(monkeypatch)
    result = run_test(environment(small))

    assert result.passed is False
    record = json.loads((small.replacement_dir / TEST_REPRODUCTION_FILENAME).read_text("utf-8"))
    assert record["stops_run"] is True
    assert b_files(small.replacement_dir) == []
    assert cli.cmd_replace_test(environment(small)) == 1


def test_the_call_order_is_dense_a_reader_reproduction_then_b(on_the_branch, monkeypatch):
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)
    order: list[str] = []
    original_measure = replacement_run.measure
    original_reader = replacement_run.read_historical_test_figures
    original_compare = replacement_run.compare_test_reproduction

    def measure(retriever, **kwargs):
        measured = original_measure(retriever, **kwargs)
        assert measured.run_path.exists() and measured.outcomes.path.exists()
        order.append(retriever.name)
        return measured

    def reader(frozen, results_dir):
        assert order == ["dense", "hybrid-bm25"], "the reader ran before dense and A were written"
        order.append("historical test figures")
        return original_reader(frozen, results_dir)

    def compare(*args, **kwargs):
        order.append("reproduction check")
        return original_compare(*args, **kwargs)

    monkeypatch.setattr(replacement_run, "measure", measure)
    monkeypatch.setattr(replacement_run, "read_historical_test_figures", reader)
    monkeypatch.setattr(replacement_run, "compare_test_reproduction", compare)

    # On the deviation branch, because the toy's A - dense on test is not 55 questions: in reused
    # mode the decision would rightly refuse it (m1_check, D13).
    result = run_test(
        environment(on_the_branch),
        control_mode="re-measured",
        deviation=branch_deviation(on_the_branch),
    )

    assert result.passed, result.message
    assert order == [
        "dense",
        "hybrid-bm25",
        "historical test figures",
        "reproduction check",
        "hybrid-entity-hop",
    ]
    directory = on_the_branch.replacement_dir
    for name in ("decision.json", "diagnostics-test.json", "traces-test.json", "readout-test.json"):
        assert (directory / name).exists(), name
    assert len(files(directory, "run-hybrid-entity-hop-test-*.json")) == 1


def test_every_test_result_carries_the_freeze_and_the_identities(on_the_branch, monkeypatch):
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)
    deviation = branch_deviation(on_the_branch)
    assert run_test(
        environment(on_the_branch), control_mode="re-measured", deviation=deviation
    ).passed
    frozen = load_freeze(on_the_branch.replacement_dir)
    superseded = frozen.payload["supersedes"]
    everything = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in files(on_the_branch.replacement_dir, "run-*-test-*.json")
    ]
    # The stopped run's dense and A were read under the superseded freeze; nothing else was.
    assert {result["config"]["freeze_digest"] for result in everything} == {
        superseded,
        frozen.digest,
    }
    results = [r for r in everything if r["config"]["freeze_digest"] == frozen.digest]
    assert sorted(result["system"] for result in results) == [
        "dense",
        "hybrid-bm25",
        "hybrid-entity-hop",
    ]
    for result in results:
        config = result["config"]
        assert config["frozen_at"] == frozen.frozen_at
        assert config["unit_set_hash"] == on_the_branch.pins.unit_set_hash
        assert config["dense_identity"]["question_cache_key"]
        assert config["bm25_identity"]["stopwords"] == "en"
        assert config["code_version"] and config["phase"] == 6
        assert result["created_at"] >= frozen.frozen_at
        if result["system"] == "hybrid-entity-hop":
            pins = on_the_branch.pins
            assert config["entity_component"]["node_index_digest"] == pins.node_index_digest
        if result["system"] == "hybrid-bm25":
            assert config["control_mode"] == "re-measured"


def test_the_decision_is_computed_from_outcomes_read_back_from_disk(on_the_branch, monkeypatch):
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)
    seen: dict[str, Any] = {}
    original = replacement_run.compute_decision

    def spy(freeze, outcomes, test_reproduction):
        seen.update(outcomes)
        return original(freeze, outcomes, test_reproduction)

    monkeypatch.setattr(replacement_run, "compute_decision", spy)
    deviation = branch_deviation(on_the_branch)
    assert run_test(
        environment(on_the_branch), control_mode="re-measured", deviation=deviation
    ).passed

    assert set(seen) == {"dense", "hybrid-bm25", "hybrid-entity-hop"}
    for artifact in seen.values():
        assert isinstance(artifact, PerQuestionOutcomes)
        assert artifact.path.exists()
        on_disk = json.loads(artifact.path.read_text(encoding="utf-8"))
        assert on_disk["digest"] == artifact.digest
    directory = on_the_branch.replacement_dir
    decision = json.loads((directory / "decision.json").read_text(encoding="utf-8"))
    assert decision["outcome_digests"] == {system: a.digest for system, a in seen.items()}
    assert decision["primary"]["n"] == N_TEST


def test_a_second_run_refuses_on_the_existing_dense_test_result(small, monkeypatch):
    publish_example(monkeypatch)
    assert run_test(environment(small)).passed is False  # stops at the delta confirmation
    assert files(small.replacement_dir, "run-dense-test-*.json")
    with pytest.raises(ReplacementRunError, match="exists"):
        run_test(environment(small))


def test_re_measured_on_the_deviation_branch_re_reads_dense_and_a_and_evaluates_b_once(
    on_the_branch, monkeypatch
):
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)
    deviation = branch_deviation(on_the_branch)
    result = run_test(environment(on_the_branch), control_mode="re-measured", deviation=deviation)

    assert result.passed, result.message
    directory = on_the_branch.replacement_dir
    assert files(directory, "outcomes-dense-test-2.json")
    assert files(directory, "outcomes-hybrid-bm25-test-2.json")
    assert len(files(directory, "outcomes-hybrid-entity-hop-test*.json")) == 1
    decision = json.loads((directory / "decision.json").read_text(encoding="utf-8"))
    assert decision["control_mode"] == "re-measured"
    second = json.loads((directory / "test-reproduction-2.json").read_text(encoding="utf-8"))
    assert second["mode"] == "re-measured"
    assert second["freeze_digest"] == load_freeze(directory).digest


def test_re_measured_without_an_existing_deviation_is_refused(on_the_branch, monkeypatch):
    publish_example(monkeypatch)
    with pytest.raises(ReplacementRunError, match="deviation document"):
        run_test(environment(on_the_branch), control_mode="re-measured", deviation="missing.md")
    with pytest.raises(ReplacementRunError, match="deviation document"):
        run_test(environment(on_the_branch), control_mode="re-measured")
    assert not files(on_the_branch.replacement_dir, "*-test-2.json")


# --- The superseding freeze (OI-4) ------------------------------------------------------------


def stop_on_a_failed_reproduction(pipeline: ToyPipeline) -> None:
    """A reused test run that confirms delta's source, then fails its reproduction before B."""
    with pytest.MonkeyPatch.context() as patch:
        publish_example(patch)
        delta_confirmed(patch)
        nudge_test_reproduction(patch)
        stopped = run_test(environment(pipeline))
    assert stopped.passed is False and "reproduction" in stopped.message


def write_superseding_freeze(pipeline: ToyPipeline) -> str:
    deviation = a_deviation(pipeline)
    superseding = run_freeze(
        environment(pipeline), supersede=True, control_mode="re-measured", deviation=deviation
    )
    assert superseding.passed, superseding.message
    return deviation


def test_the_superseding_freeze_branch(small, monkeypatch):
    stop_on_a_failed_reproduction(small)
    first = load_freeze(small.replacement_dir)
    deviation = a_deviation(small)

    with pytest.raises(ReplacementRunError, match="deviation"):
        run_freeze(environment(small), supersede=True, control_mode="re-measured")
    superseding = run_freeze(
        environment(small), supersede=True, control_mode="re-measured", deviation=deviation
    )

    assert superseding.passed, superseding.message
    assert superseding.path.name == "freeze-2.json"
    valid = load_freeze(small.replacement_dir)
    assert valid.payload["supersedes"] == first.digest
    assert valid.payload["deviation"] == deviation
    assert valid.control_mode == "re-measured"

    publish_example(monkeypatch)
    again = run_test(environment(small), control_mode="re-measured", deviation=deviation)
    assert again.passed is False  # the toy's delta source this time, before B
    re_read = json.loads(
        (small.replacement_dir / "outcomes-dense-test-2.json").read_text(encoding="utf-8")
    )
    assert re_read["freeze_digest"] == valid.digest
    assert b_files(small.replacement_dir) == []

    # The superseding freeze has now been read on test: a further re-read needs a new deviation
    # and a new superseding freeze, not the same flag again.
    with pytest.raises(ReplacementRunError, match="already been read"):
        run_test(environment(small), control_mode="re-measured", deviation=deviation)
    assert not files(small.replacement_dir, "*-test-3.json")


# --- re-measured on test: only on the deviation branch of D14 (OI-4) ---------------------------


def held_out_files(directory: Path) -> list[Path]:
    return files(directory, "*-test-*.json")


def test_a_first_replace_test_with_re_measured_is_refused(small, monkeypatch):
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)
    deviation = a_deviation(small)

    with pytest.raises(ReplacementRunError, match="deviation branch"):
        run_test(environment(small), control_mode="re-measured", deviation=deviation)
    assert held_out_files(small.replacement_dir) == []
    assert not (small.replacement_dir / TEST_REPRODUCTION_FILENAME).exists()


def test_re_measured_is_refused_after_a_reproduction_failure_without_a_superseding_freeze(
    small, monkeypatch
):
    stop_on_a_failed_reproduction(small)
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)
    before = held_out_files(small.replacement_dir)

    with pytest.raises(ReplacementRunError, match="supersedes no earlier freeze"):
        run_test(environment(small), control_mode="re-measured", deviation=a_deviation(small))
    assert held_out_files(small.replacement_dir) == before


def test_re_measured_is_refused_when_the_stop_was_the_delta_source_not_the_reproduction(
    small, monkeypatch
):
    publish_example(monkeypatch)
    stopped = run_test(environment(small))
    assert stopped.passed is False and "delta" in stopped.message
    deviation = write_superseding_freeze(small)
    delta_confirmed(monkeypatch)
    before = held_out_files(small.replacement_dir)

    with pytest.raises(ReplacementRunError, match="failed reproduction"):
        run_test(environment(small), control_mode="re-measured", deviation=deviation)
    assert held_out_files(small.replacement_dir) == before


def test_a_superseding_freeze_with_no_stopped_test_run_is_never_read_on_test(small, monkeypatch):
    deviation = write_superseding_freeze(small)
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)

    with pytest.raises(ReplacementRunError, match="failed reproduction"):
        run_test(environment(small), control_mode="re-measured", deviation=deviation)
    with pytest.raises(ReplacementRunError, match="supersedes an earlier"):
        run_test(environment(small))
    assert held_out_files(small.replacement_dir) == []


@pytest.mark.parametrize("mode", [None, "reused"])
def test_a_superseding_freeze_is_read_on_test_only_with_re_measured(
    on_the_branch, monkeypatch, mode
):
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)
    before = held_out_files(on_the_branch.replacement_dir)

    with pytest.raises(ReplacementRunError, match="supersedes an earlier"):
        run_test(
            environment(on_the_branch),
            control_mode=mode,
            deviation=branch_deviation(on_the_branch),
        )
    assert held_out_files(on_the_branch.replacement_dir) == before


def test_re_measured_is_refused_when_the_stopped_runs_results_are_missing(
    on_the_branch, monkeypatch
):
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)
    for path in files(on_the_branch.replacement_dir, "run-dense-test-*.json"):
        path.unlink()

    with pytest.raises(ReplacementRunError, match="stopped run"):
        run_test(
            environment(on_the_branch),
            control_mode="re-measured",
            deviation=branch_deviation(on_the_branch),
        )


def test_a_modified_record_of_the_stopped_run_is_refused(on_the_branch, monkeypatch):
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)
    record = on_the_branch.replacement_dir / TEST_REPRODUCTION_FILENAME
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["observed_runs"]["dense"] = "run-dense-test-elsewhere.json"
    record.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ReplacementRunError, match="digest"):
        run_test(
            environment(on_the_branch),
            control_mode="re-measured",
            deviation=branch_deviation(on_the_branch),
        )


def test_a_dev_branch_freeze_is_read_on_test_without_the_flag_and_refuses_it(toy, monkeypatch):
    """The other re-measured control of the spec (HU-2, incompatible at dev): the freeze itself is
    re-measured and supersedes nothing. Its first test read takes the mode from the freeze; the
    explicit flag stays reserved to the deviation branch of D14."""
    nudge_historical_dev_precision(toy)
    assert run_check(environment(toy)).passed is False
    deviation = toy.root / "6.1_control_mismatch.md"
    deviation.write_text("# 6.1 toy deviation", encoding="utf-8")
    assert run_freeze(environment(toy), control_mode="re-measured", deviation=str(deviation)).passed
    publish_example(monkeypatch)

    with pytest.raises(ReplacementRunError, match="deviation branch"):
        run_test(environment(toy), control_mode="re-measured", deviation=str(deviation))
    assert held_out_files(toy.replacement_dir) == []

    result = run_test(environment(toy))
    assert result.passed is False and "delta" in result.message  # past the gate, the toy's delta
    record = json.loads((toy.replacement_dir / TEST_REPRODUCTION_FILENAME).read_text("utf-8"))
    assert record["mode"] == "re-measured"


def test_the_cli_test_command_reports_a_refusal(small):
    assert cli.cmd_replace_test(environment(small)) == 1


def test_the_cli_test_stage_takes_the_re_measured_branch_options():
    args = cli.build_parser().parse_args(
        ["replace-test", "--control-mode", "re-measured", "--deviation", "docs/x.md"]
    )
    assert (args.control_mode, args.deviation) == ("re-measured", "docs/x.md")
    args = cli.build_parser().parse_args(["replace-freeze", "--supersede"])
    assert args.supersede is True


# --- The secrets gate's allowlist, reconciled with these writers (T20, D20) --------------------

SECRETS_FILTER = Path(__file__).resolve().parents[1] / ".github" / "detect_secrets_filters.py"
HEX_KEYED_LINE = re.compile(r'^\s*"(?P<key>[^"]+)": "(?P<hex>[0-9a-f]{16,})",?\s*$')
HEX_BARE_LINE = re.compile(r'^\s*"(?P<hex>[0-9a-f]{16,})",?\s*$')


def secrets_filter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detect_secrets_filters", SECRETS_FILTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_secrets_allowlist_is_exactly_the_hex_keys_these_writers_emit(
    on_the_branch, monkeypatch
):
    """Every artifact the three stages write on the deviation branch - the first freeze, the
    stopped test run, the superseding freeze, the re-measured run and the decision - is read.
    The (key, length) pairs holding a lowercase hex value must be the allowlist, no more and no
    less; the one exception is a key detect-secrets' own id heuristic already skips (`unit_id`)."""
    heuristic = pytest.importorskip("detect_secrets.filters.heuristic")
    publish_example(monkeypatch)
    delta_confirmed(monkeypatch)
    result = run_test(
        environment(on_the_branch),
        control_mode="re-measured",
        deviation=branch_deviation(on_the_branch),
    )
    assert result.passed, result.message

    emitted: set[tuple[str, int]] = set()
    left_to_the_heuristic: set[tuple[str, int]] = set()
    bare_lengths: set[int] = set()
    for path in files(on_the_branch.replacement_dir, "*.json"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if (keyed := HEX_KEYED_LINE.match(line)) is not None:
                pair = (keyed["key"], len(keyed["hex"]))
                if heuristic.is_likely_id_string(keyed["hex"], line):
                    left_to_the_heuristic.add(pair)
                else:
                    emitted.add(pair)
            elif (bare := HEX_BARE_LINE.match(line)) is not None:
                bare_lengths.add(len(bare["hex"]))

    module = secrets_filter()
    allowlisted = {(key, length) for key, lengths in module.ALLOWLIST.items() for length in lengths}
    assert emitted - allowlisted == set(), "hex keys the writers emit that the gate would report"
    assert allowlisted - emitted == set(), "allowlisted keys no writer emits"
    assert left_to_the_heuristic == {("unit_id", 16)}
    assert bare_lengths <= module.BARE_LENGTHS
