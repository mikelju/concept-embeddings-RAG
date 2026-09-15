"""T14: the rivals' numbers are read, never recomputed.

HU-7 asks for System C to be measured against dense, BM25, System B and the control,
and for **their** figures to come from the Phase 3 result files rather than from a
fresh run. The reason is not thrift: re-measuring a rival is a second instrument, and
a difference between two instruments is indistinguishable from a difference between
two systems.

What makes reading them safe is the check that they were measured under the same
conditions. A result file says which pool it walked, which tokenizer counted its
budget, which seed and which `top_k`. If any of the four disagrees with this phase's
run, the table is refused rather than built - because a table like that would still
have five rows and every one of them would be wrong by an amount nobody could see.
"""

import json

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.comparison import (
    PHASE_3_SYSTEMS,
    ComparisonError,
    build_comparison,
    comparison_table,
    cost_table,
    load_rivals,
)
from concept_embeddings_rag.evaluation.harness import RunResult

POOL = "101f564fdcca620c"


def a_config(**overrides) -> dict:
    base = {
        "model": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c",
        "unit_set_hash": POOL,
        "seed": 42,
        "tokenizer": "BAAI/bge-small-en-v1.5",
        "code_version": "0.1.0",
        "top_k": 100,
    }
    base.update(overrides)
    return base


def a_result(system: str, split: str = "test", *, figure: float = 0.8, **config_overrides):
    return RunResult(
        system=system,
        split=split,
        config=a_config(**config_overrides),
        metrics={
            f"budget_{budget}": {
                "gold_recall": figure + 0.05,
                "full_support": figure,
                "precision": 0.1,
            }
            for budget in config.CONTEXT_BUDGETS
        },
        cost={"mean_latency_ms": 1.5, "mean_units_included": 16.4, "n_questions": 1400},
    )


def a_results_dir(tmp_path, split: str = "test", **overrides):
    for index, system in enumerate(PHASE_3_SYSTEMS):
        a_result(system, split, figure=0.70 + index / 100, **overrides).save(tmp_path)
    return tmp_path


# --- The five rivals, found by name -------------------------------------------


def test_the_five_phase_3_systems_are_the_ones_this_phase_compares_against():
    assert PHASE_3_SYSTEMS == ("dense", "bm25", "conceptual", "hybrid-conceptual", "hybrid-bm25")


def test_every_rival_is_found_by_name(tmp_path):
    rivals = load_rivals(a_results_dir(tmp_path), split="test")

    assert sorted(rivals) == sorted(PHASE_3_SYSTEMS)


def test_a_missing_rival_is_named_in_the_error_rather_than_skipped(tmp_path):
    a_results_dir(tmp_path)
    for path in tmp_path.glob("run-hybrid-bm25-*.json"):
        path.unlink()

    with pytest.raises(ComparisonError, match="hybrid-bm25"):
        load_rivals(tmp_path, split="test")


def test_a_rival_measured_on_another_split_does_not_count_as_present(tmp_path):
    a_results_dir(tmp_path, split="dev")

    with pytest.raises(ComparisonError, match="test"):
        load_rivals(tmp_path, split="test")


def test_the_newest_reading_of_a_system_is_the_one_read(tmp_path):
    """Phase 1 measured dense and BM25 too; the Phase 3 batch is the later one."""
    a_results_dir(tmp_path)
    newer = RunResult(
        system="dense",
        split="test",
        config=a_config(),
        metrics={
            f"budget_{budget}": {"gold_recall": 0.99, "full_support": 0.98, "precision": 0.1}
            for budget in config.CONTEXT_BUDGETS
        },
        cost={"n_questions": 1400},
        created_at="2099-01-01T00:00:00+00:00",
    )
    newer.save(tmp_path)

    rivals = load_rivals(tmp_path, split="test")

    assert rivals["dense"].metrics["budget_2048"]["full_support"] == 0.98


def test_reading_from_a_directory_with_no_result_at_all_names_what_is_missing(tmp_path):
    with pytest.raises(ComparisonError, match="dense"):
        load_rivals(tmp_path, split="test")


# --- Measured under the same conditions, or not tabulated ---------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("unit_set_hash", "another-pool"),
        ("tokenizer", "cl100k_base"),
        ("seed", 7),
        ("top_k", 50),
    ],
)
def test_a_rival_measured_under_another_configuration_refuses_to_tabulate(tmp_path, key, value):
    a_results_dir(tmp_path)
    a_result("dense", "test", **{key: value}).save(tmp_path)

    with pytest.raises(ComparisonError, match=key):
        build_comparison(tmp_path, split="test", reference=a_result("expansion", "test"))


def test_a_rival_measured_under_the_same_configuration_tabulates(tmp_path):
    a_results_dir(tmp_path)

    table = build_comparison(tmp_path, split="test", reference=a_result("expansion", "test"))

    assert table


def test_the_comparison_refuses_a_reference_measured_on_another_split(tmp_path):
    a_results_dir(tmp_path)

    with pytest.raises(ComparisonError, match="split"):
        build_comparison(tmp_path, split="test", reference=a_result("expansion", "dev"))


def test_the_four_keys_checked_are_the_ones_that_make_two_numbers_comparable(tmp_path):
    """Named rather than inferred: the model itself may differ only if nothing else does."""
    a_results_dir(tmp_path)

    table = build_comparison(
        tmp_path, split="test", reference=a_result("expansion", "test", revision="other")
    )

    assert table


# --- The table itself ---------------------------------------------------------


def test_the_table_holds_one_row_per_system_per_budget(tmp_path):
    rivals = load_rivals(a_results_dir(tmp_path), split="test")

    table = comparison_table({**rivals, "expansion": a_result("expansion", "test")})

    assert len(table) == 6 * len(config.CONTEXT_BUDGETS)
    assert {row.budget for row in table} == set(config.CONTEXT_BUDGETS)


def test_every_row_carries_the_three_figures_the_report_prints(tmp_path):
    rivals = load_rivals(a_results_dir(tmp_path), split="test")

    for row in comparison_table(rivals):
        assert 0.0 <= row.gold_recall <= 1.0
        assert 0.0 <= row.full_support <= 1.0
        assert 0.0 <= row.precision <= 1.0


def test_nothing_is_recomputed_and_the_figures_are_the_ones_on_disk(tmp_path):
    a_results_dir(tmp_path)
    rivals = load_rivals(tmp_path, split="test")
    written = json.loads(sorted(tmp_path.glob("run-dense-test-*.json"))[0].read_text("utf-8"))

    row = next(
        row for row in comparison_table(rivals) if row.system == "dense" and row.budget == 2048
    )

    assert row.full_support == written["metrics"]["budget_2048"]["full_support"]


def test_a_system_never_measured_at_a_budget_is_refused_rather_than_defaulted(tmp_path):
    a_results_dir(tmp_path)
    thin = RunResult(
        system="dense",
        split="test",
        config=a_config(),
        metrics={"budget_512": {"gold_recall": 0.8, "full_support": 0.7, "precision": 0.1}},
        cost={"n_questions": 1400},
        created_at="2099-01-01T00:00:00+00:00",
    )
    thin.save(tmp_path)

    with pytest.raises(ComparisonError, match="never measured at budget"):
        comparison_table(load_rivals(tmp_path, split="test"))


def test_the_cost_table_reports_latency_and_units_per_system(tmp_path):
    rivals = load_rivals(a_results_dir(tmp_path), split="test")

    costs = cost_table(rivals)

    assert sorted(costs) == sorted(PHASE_3_SYSTEMS)
    assert costs["dense"]["mean_latency_ms"] == 1.5
    assert costs["dense"]["mean_units_included"] == 16.4


def test_the_cost_of_the_expansion_carries_its_iteration_statistics(tmp_path):
    expansion = a_result("expansion", "test")
    expansion = RunResult(
        system=expansion.system,
        split=expansion.split,
        config={**expansion.config, "mean_iterations": 2.4, "cap_hits": 12},
        metrics=expansion.metrics,
        cost=expansion.cost,
    )

    costs = cost_table({"expansion": expansion})

    assert costs["expansion"]["mean_iterations"] == 2.4
    assert costs["expansion"]["cap_hits"] == 12


# --- Against the numbers this project actually published ----------------------


def test_the_reader_reproduces_the_published_phase_3_table():
    """The strongest form of "nothing is recomputed": the figures come back unchanged.

    These are the Full Support figures of `3.results.md` at 2,048 tokens on test, as
    published. `data/` is git-ignored, so a checkout without the result files skips.
    """
    published = {
        "conceptual": 0.3550,
        "bm25": 0.7593,
        "dense": 0.8250,
        "hybrid-conceptual": 0.8250,
        "hybrid-bm25": 0.8643,
    }
    if not sorted(config.RESULTS_DIR.glob("run-hybrid-bm25-test-*.json")):
        pytest.skip("no Phase 3 result files on disk to read the published table back from")

    rivals = load_rivals(config.RESULTS_DIR, split="test")
    table = {row.system: row.full_support for row in comparison_table(rivals) if row.budget == 2048}

    for system, figure in published.items():
        assert round(table[system], 4) == figure, system
