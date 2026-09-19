"""Phase 6, T9: the control is reused only if it reproduces exactly (HU-2, D12).

The checks are pure functions over figures handed in, each recorded as
`{name, expected, observed, passed}`:

- the re-executed fit against `selection.json`'s control: the 11 curve points, the RRF score,
  the grid and question count, and the selection they derive (scheme and weights);
- the HU-2 configuration keys of the four historical files against Phase 6's own runs;
- the dev aggregates of `dense` and `hybrid-bm25` against the historical dev files, by the
  spec's list: per budget gold recall, Full Support and precision; recall at 2, 5 and 10; mean
  units included. Latency is never compared.

The mode is `reused` if and only if every dev check passes. The test-side comparison uses the
same figure list; it is exercised here on toy mappings only, and in `re-measured` mode it is
recorded and returns without raising.

The toy's "historical" figures are the toy's own harness runs and fits, built in memory: no
test of this task opens any historical file.
"""

import ast
import copy
from pathlib import Path

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.control_reproduction import (
    FIGURE_BUDGET_METRICS,
    REMEASURED,
    REUSED,
    Check,
    compare_test_reproduction,
    config_checks,
    control_mode,
    dev_reproduction,
    figure_checks,
    fit_checks,
)
from concept_embeddings_rag.evaluation.harness import RunResult, evaluate_retriever
from concept_embeddings_rag.evaluation.selection import FusionFit, fit_fusion_weight
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.fusion import FusedRetriever

MODULE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "concept_embeddings_rag"
    / "evaluation"
    / "control_reproduction.py"
)
UNITS = [f"u{index}" for index in range(8)]
TOKENS = dict.fromkeys(UNITS, 700)


class Scripted:
    def __init__(self, name: str, rankings: dict[str, list[Hit]]) -> None:
        self.name = name
        self.rankings = rankings

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return self.rankings[query][:top_k]


def questions() -> list[Question]:
    golds = [("u0", "u5"), ("u1", "u6"), ("u2", "u7")]
    return [
        Question(f"q{i}", f"question {i}", "-", gold, (("t", 0),), "dev")
        for i, gold in enumerate(golds)
    ]


DENSE = {
    "question 0": [("u0", 0.9), ("u1", 0.8), ("u2", 0.7), ("u5", 0.1)],
    "question 1": [("u3", 0.9), ("u1", 0.6), ("u6", 0.5), ("u7", 0.4)],
    "question 2": [("u2", 0.95), ("u0", 0.4), ("u4", 0.3), ("u7", 0.2)],
}
BM25 = {
    "question 0": [("u5", 12.0), ("u3", 4.0), ("u0", 1.0)],
    "question 1": [("u6", 7.0), ("u1", 6.5), ("u4", 0.5)],
    "question 2": [("u4", 9.0), ("u7", 3.0), ("u2", 2.0)],
}


def base_config() -> dict:
    return {
        "model": "toy-model",
        "revision": "rev",
        "unit_set_hash": "pool",
        "seed": 42,
        "tokenizer": "toy-tokenizer",
        "code_version": "0.1.0",
        "top_k": 4,
    }


def a_fit() -> FusionFit:
    return fit_fusion_weight(
        [Scripted("dense", DENSE), Scripted("bm25", BM25)],
        questions=questions(),
        token_counts=TOKENS,
        base_config=base_config(),
    )


def a_run(system: str, fit: FusionFit) -> RunResult:
    dense = Scripted("dense", DENSE)
    if system == "dense":
        retriever, extra = dense, {}
    else:
        hybrid = FusedRetriever(
            [dense, Scripted("bm25", BM25)], scheme=fit.winning_scheme, weights=fit.weights
        )
        retriever = hybrid
        extra = {"fusion_scheme": hybrid.scheme, "fusion_weights": hybrid.weights}
    return evaluate_retriever(
        retriever,
        questions=questions(),
        token_counts=TOKENS,
        budgets=config.CONTEXT_BUDGETS,
        ks=config.RECALL_AT_K,
        top_k=4,
        config={**base_config(), **extra},
        split="dev",
    )


@pytest.fixture(scope="module")
def world():
    """A 'historical' control and a Phase 6 re-execution: the same code, run twice."""
    frozen, refit = a_fit(), a_fit()
    historical = {system: a_run(system, frozen) for system in ("dense", "hybrid-bm25")}
    observed = {system: a_run(system, refit) for system in ("dense", "hybrid-bm25")}
    return frozen, refit, historical, observed


def figures(runs: dict[str, RunResult]) -> dict[str, dict]:
    return {system: {"metrics": run.metrics, "cost": run.cost} for system, run in runs.items()}


def configs(runs: dict[str, RunResult], splits=("dev", "test")) -> dict[str, dict]:
    return {
        f"{system}/{split}": copy.deepcopy(run.config)
        for system, run in runs.items()
        for split in splits
    }


def reproduce(world, *, frozen=None, historical=None, historical_configs=None):
    base_frozen, refit, base_historical, observed = world
    return dev_reproduction(
        frozen=base_frozen if frozen is None else frozen,
        refit=refit,
        historical_dev=figures(base_historical) if historical is None else historical,
        observed_dev=figures(observed),
        historical_configs=(
            configs(base_historical) if historical_configs is None else historical_configs
        ),
        observed_configs={system: run.config for system, run in observed.items()},
    )


def failed_names(report) -> list[str]:
    return [check.name for check in report.checks if not check.passed]


# --- Everything reproduces ---------------------------------------------------------------------


def test_all_checks_pass_when_the_history_is_the_toys_own_runs_and_the_mode_is_reused(world):
    report = reproduce(world)

    assert failed_names(report) == []
    assert report.mode == REUSED
    assert report.passed


def test_every_check_is_recorded_as_name_expected_observed_passed(world):
    report = reproduce(world)
    for entry in report.as_payload()["checks"]:
        assert set(entry) == {"name", "expected", "observed", "passed"}


def test_the_dev_checks_cover_the_curve_rrf_selection_configs_and_the_figure_list(world):
    names = [check.name for check in reproduce(world).checks]

    for weight in config.FUSION_WEIGHT_GRID:
        assert f"control curve w={weight:.1f}" in names
    for expected in (
        "control rrf_score",
        "control grid",
        "control n_questions",
        "control winning_scheme",
        "control weights",
    ):
        assert expected in names
    for system in ("dense", "hybrid-bm25"):
        for budget in config.CONTEXT_BUDGETS:
            for metric in FIGURE_BUDGET_METRICS:
                assert f"{system}/dev budget_{budget} {metric}" in names
        for k in config.RECALL_AT_K:
            assert f"{system}/dev recall_at_{k}" in names
        assert f"{system}/dev mean_units_included" in names
    for label in ("dense/dev", "dense/test", "hybrid-bm25/dev", "hybrid-bm25/test"):
        for key in config.HISTORICAL_CONFIG_KEYS:
            assert f"{label} config {key}" in names
    for label in ("hybrid-bm25/dev", "hybrid-bm25/test"):
        for key in config.HISTORICAL_HYBRID_CONFIG_KEYS:
            assert f"{label} config {key}" in names


def test_the_toy_fit_is_not_trivial(world):
    frozen = world[0]
    assert len(set(frozen.curve.values())) > 1


# --- Any single alteration makes the mode re-measured and names the check -----------------------


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_an_altered_curve_point_names_its_check(world, weight):
    frozen = world[0]
    curve = dict(frozen.curve)
    curve[weight] = curve[weight] + 1e-12
    altered = FusionFit(**{**frozen.__dict__, "curve": curve})

    report = reproduce(world, frozen=altered)
    assert report.mode == REMEASURED
    assert f"control curve w={weight:.1f}" in failed_names(report)


def test_an_altered_rrf_score_names_its_check(world):
    frozen = world[0]
    altered = FusionFit(**{**frozen.__dict__, "rrf_score": frozen.rrf_score - 1e-12})
    report = reproduce(world, frozen=altered)
    assert report.mode == REMEASURED
    assert "control rrf_score" in failed_names(report)


def test_a_curve_that_derives_another_selection_fails_the_selection_checks(world):
    frozen = world[0]
    curve = dict(frozen.curve)
    curve[0.0] = 2.0
    altered = FusionFit(**{**frozen.__dict__, "curve": curve})
    failed = failed_names(reproduce(world, frozen=altered))
    assert "control weights" in failed


@pytest.mark.parametrize(
    "path",
    [
        ("dense", "metrics", "budget_2048", "full_support"),
        ("dense", "metrics", "budget_512", "precision"),
        ("hybrid-bm25", "metrics", "budget_4096", "gold_recall"),
        ("hybrid-bm25", "metrics", "recall_at_5"),
        ("dense", "cost", "mean_units_included"),
    ],
)
def test_any_single_altered_figure_names_its_check(world, path):
    historical = copy.deepcopy(figures(world[2]))
    node = historical[path[0]][path[1]]
    for key in path[2:-1]:
        node = node[key]
    node[path[-1]] = node[path[-1]] + 1e-12

    report = reproduce(world, historical=historical)
    assert report.mode == REMEASURED
    split_name = f"{path[0]}/dev"
    expected_name = (
        f"{split_name} {' '.join(path[2:])}" if path[1] == "metrics" else f"{split_name} {path[2]}"
    )
    assert failed_names(report) == [expected_name]


@pytest.mark.parametrize(
    ("label", "key", "value"),
    [
        ("dense/dev", "seed", 7),
        ("dense/test", "top_k", 50),
        ("hybrid-bm25/dev", "tokenizer", "other"),
        ("hybrid-bm25/test", "fusion_weights", {"dense": 0.9, "bm25": 0.1}),
    ],
)
def test_any_single_altered_config_key_names_its_check(world, label, key, value):
    historical_configs = copy.deepcopy(configs(world[2]))
    historical_configs[label][key] = value
    report = reproduce(world, historical_configs=historical_configs)
    assert report.mode == REMEASURED
    assert failed_names(report) == [f"{label} config {key}"]


def test_a_missing_figure_fails_rather_than_passing_by_absence(world):
    historical = copy.deepcopy(figures(world[2]))
    del historical["dense"]["metrics"]["recall_at_10"]
    assert "dense/dev recall_at_10" in failed_names(reproduce(world, historical=historical))


def test_latency_is_never_compared(world):
    historical = copy.deepcopy(figures(world[2]))
    for system in historical:
        historical[system]["cost"]["mean_latency_ms"] = 1e9
    report = reproduce(world, historical=historical)

    assert report.mode == REUSED
    assert not any("latency" in check.name for check in report.checks)


def test_the_mode_rule_is_all_checks_or_re_measured():
    passing = Check("a", 1, 1)
    failing = Check("b", 1, 2)
    assert control_mode([passing, passing]) == REUSED
    assert control_mode([passing, failing]) == REMEASURED
    with pytest.raises(ValueError):
        control_mode([])


def test_a_check_compares_exactly():
    assert Check("x", 0.1 + 0.2, 0.3).passed is False
    assert Check("x", {"dense": 0.5}, {"dense": 0.5}).passed is True


# --- The test-side comparison, on toy mappings only ----------------------------------------------


def toy_figures(full_support: float) -> dict[str, dict]:
    metrics = {
        f"budget_{budget}": {"gold_recall": 0.5, "full_support": full_support, "precision": 0.1}
        for budget in config.CONTEXT_BUDGETS
    }
    metrics.update({f"recall_at_{k}": 0.25 for k in config.RECALL_AT_K})
    cost = {"mean_units_included": 3.0, "mean_latency_ms": 1.0, "n_questions": 4}
    return {system: {"metrics": metrics, "cost": cost} for system in ("dense", "hybrid-bm25")}


def test_the_test_side_comparison_passes_on_equal_figures_in_reused_mode():
    result = compare_test_reproduction(toy_figures(0.75), toy_figures(0.75), mode=REUSED)
    assert result.passed
    assert not result.stops_run
    assert all(check.name.split(" ")[0].endswith("/test") for check in result.checks)


def test_a_failed_test_side_comparison_stops_the_run_in_reused_mode():
    result = compare_test_reproduction(toy_figures(0.75), toy_figures(0.5), mode=REUSED)
    assert not result.passed
    assert result.stops_run
    assert result.descriptive is False


def test_in_re_measured_mode_the_comparison_is_recorded_and_returns_without_raising():
    result = compare_test_reproduction(toy_figures(0.75), toy_figures(0.5), mode=REMEASURED)
    assert not result.passed
    assert not result.stops_run
    assert result.descriptive is True
    assert result.as_payload()["mode"] == REMEASURED
    assert any(not entry["passed"] for entry in result.as_payload()["checks"])


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        compare_test_reproduction(toy_figures(0.75), toy_figures(0.75), mode="partly")


def test_figure_checks_and_config_checks_and_fit_checks_are_public_pure_functions(world):
    frozen, refit, historical, observed = world
    assert all(check.passed for check in fit_checks(frozen, refit))
    run = observed["dense"]
    assert all(
        check.passed
        for check in figure_checks(
            "dense/dev", {"metrics": run.metrics, "cost": run.cost}, figures(historical)["dense"]
        )
    )
    assert all(
        check.passed
        for check in config_checks(
            configs(historical), {system: r.config for system, r in observed.items()}
        )
    )


# --- The module opens no file ---------------------------------------------------------------------


def test_the_module_has_no_file_reading_call():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    for forbidden in ("open", "read_text", "read_bytes", "loads", "load", "glob", "exists"):
        assert forbidden not in called, forbidden
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert not imported & {"json", "pathlib", "Path", "os"}
