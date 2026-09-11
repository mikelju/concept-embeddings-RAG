"""T10: the selection artifact, and what it refuses to be.

This file is the record of the only decision Phase 2 deferred - which K, which view
and which query operator this project retrieves through - plus every other choice
fitted on the 600 dev questions. HU-3 asks for it because "a K chosen in
conversation and remembered is not a decision this project can reproduce", and
HU-7 asks it to say how many dev-fitted choices there were rather than leaving
them to be counted afterwards.

Three properties are what make it evidence rather than a note:

- it is **verified on load**, against the pool it was computed over and the
  dictionaries it names, the rule Phase 1's audit left behind;
- it is **never overwritten**, so a second freeze has to be a visible act;
- **no field of it can be populated from a test-split number**, and that is
  asserted by construction of the API rather than by discipline: a cell is built
  from a `RunResult`, and a `RunResult` on any split but dev is refused.
"""

import json
from dataclasses import fields
from datetime import UTC, datetime

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.harness import RunResult
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    SELECTION_FILENAME,
    Decision,
    PerKEntry,
    SelectionError,
    SelectionReport,
    SweepCell,
    load_selection,
    save_selection,
    selection_path,
)

POOL = "poolhash0001"
KEYS = {512: "key0512", 1024: "key1024", 2048: "key2048", 4096: "key4096"}
FROZEN_AT = "2026-09-10T09:41:07+00:00"


def a_config(**overrides) -> dict:
    """The frozen configuration of System B: enough to rebuild it from the file alone."""
    frozen = {
        "model": "fake-model",
        "revision": "v0",
        "unit_set_hash": POOL,
        "seed": 42,
        "tokenizer": "fake-tokenizer",
        "code_version": "0.1.0",
        "top_k": 100,
    }
    frozen.update(overrides)
    return frozen


def a_result(
    *,
    k: int = 2048,
    view: str = "raw",
    query_operator: str = "projection_top16",
    damping: str = "none",
    split: str = DEV_SPLIT,
    primary: float = 0.62,
    dictionary_key: str | None = None,
) -> RunResult:
    """One conceptual-only evaluation, exactly as the unchanged harness returns it."""
    budgets = {
        budget: {
            "gold_recall": primary if budget == config.SELECTION_BUDGET else primary - 0.05,
            "full_support": primary - 0.10,
            "precision": 0.03,
        }
        for budget in config.CONTEXT_BUDGETS
    }
    metrics: dict = {f"budget_{budget}": values for budget, values in budgets.items()}
    metrics["recall_at_100"] = primary + 0.05
    return RunResult(
        system="conceptual",
        split=split,
        config=a_config(
            dictionary_key=dictionary_key or KEYS[k],
            k=k,
            view=view,
            query_operator=query_operator,
            damping=damping,
        ),
        metrics=metrics,
        cost={"mean_latency_ms": 4.2, "mean_units_included": 5.0, "n_questions": 600},
    )


def an_entry(k: int = 2048, primary: float = 0.62, **columns) -> PerKEntry:
    cells = [
        SweepCell.from_result(a_result(k=k, view=view, query_operator=arm, primary=primary))
        for arm in ("projection_top16", "projection_full")
        for view in config.CONCEPT_VIEWS
    ]
    return PerKEntry.from_cells(cells, **columns)


def a_decision(name: str = "space") -> Decision:
    return Decision(
        name=name,
        chosen="k=2048/projection_top16/raw",
        alternatives=["k=2048/projection_top16/raw", "k=1024/projection_top16/raw"],
        runner_up="k=1024/projection_top16/raw",
        margin=0.021,
    )


def a_report(**overrides) -> SelectionReport:
    fields_ = {
        "selected_k": 2048,
        "selected_dictionary_key": KEYS[2048],
        "selection_metric": config.SELECTION_METRIC,
        "selection_budget": config.SELECTION_BUDGET,
        "per_k": {2048: an_entry(2048, 0.62), 1024: an_entry(1024, 0.59)},
        "decisions": [a_decision("space")],
        "n_dev_decisions": 1,
        "tie_break": None,
        "config": a_config(dictionary_key=KEYS[2048], k=2048, view="raw"),
        "seed": 42,
        "code_version": "0.1.0",
        "frozen_at": FROZEN_AT,
    }
    fields_.update(overrides)
    return SelectionReport(**fields_)


# --- The round trip -----------------------------------------------------------


def test_the_report_survives_the_round_trip_with_every_field_intact(tmp_path):
    report = a_report()

    save_selection(report, tmp_path)

    assert load_selection(tmp_path) == report


def test_the_freeze_timestamp_survives_to_the_second(tmp_path):
    """`frozen_at` is what proves the configuration closed before test was read."""
    save_selection(a_report(frozen_at=FROZEN_AT), tmp_path)

    loaded = load_selection(tmp_path)

    assert loaded.frozen_at == FROZEN_AT
    assert datetime.fromisoformat(loaded.frozen_at) == datetime(2026, 9, 10, 9, 41, 7, tzinfo=UTC)


def test_the_freeze_timestamp_is_stamped_when_the_report_is_built_if_it_is_not_given():
    stamped = SelectionReport(
        selected_k=2048,
        selected_dictionary_key=KEYS[2048],
        selection_metric=config.SELECTION_METRIC,
        selection_budget=config.SELECTION_BUDGET,
        per_k={2048: an_entry()},
        decisions=[],
        n_dev_decisions=0,
        tie_break=None,
        config=a_config(),
        seed=42,
        code_version="0.1.0",
    ).frozen_at

    assert datetime.fromisoformat(stamped).tzinfo is not None
    assert "." not in stamped, "a freeze is recorded to the second, not to the microsecond"


def test_the_artifact_is_plain_json_that_a_human_can_read(tmp_path):
    path = save_selection(a_report(), tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == selection_path(tmp_path)
    assert path.name == SELECTION_FILENAME
    assert payload["selected_k"] == 2048
    assert payload["selected_dictionary_key"] == KEYS[2048]
    assert payload["n_dev_decisions"] == 1
    assert payload["tie_break"] is None


def test_the_four_budget_table_is_kept_whole_and_not_reduced_to_the_primary_budget(tmp_path):
    """HU-3 asks for the full table, so a ranking that only holds at one budget is visible."""
    save_selection(a_report(), tmp_path)

    cell = load_selection(tmp_path).per_k[2048].cells[0]

    assert sorted(cell.metrics) == sorted(
        [f"budget_{budget}" for budget in config.CONTEXT_BUDGETS] + ["recall_at_100"]
    )


def test_every_cell_of_the_table_carries_the_dictionary_key_that_produced_it(tmp_path):
    save_selection(a_report(), tmp_path)

    entry = load_selection(tmp_path).per_k[2048]

    assert entry.dictionary_key == KEYS[2048]
    assert [cell.dictionary_key for cell in entry.cells] == [KEYS[2048]] * len(entry.cells)


# --- Verified on load ---------------------------------------------------------


def test_a_report_computed_over_another_pool_is_refused_on_load(tmp_path):
    save_selection(a_report(), tmp_path)

    with pytest.raises(SelectionError, match="pool"):
        load_selection(tmp_path, expected_unit_set_hash="a-different-pool")


def test_a_report_computed_over_this_pool_loads(tmp_path):
    save_selection(a_report(), tmp_path)

    assert load_selection(tmp_path, expected_unit_set_hash=POOL).selected_k == 2048


def test_a_report_naming_a_dictionary_that_is_not_on_disk_is_refused(tmp_path):
    save_selection(a_report(), tmp_path)

    with pytest.raises(SelectionError, match="key2048"):
        load_selection(tmp_path, known_dictionary_keys=[KEYS[512], KEYS[1024]])


def test_a_report_whose_dictionaries_are_all_on_disk_loads(tmp_path):
    save_selection(a_report(), tmp_path)

    loaded = load_selection(tmp_path, known_dictionary_keys=list(KEYS.values()))

    assert loaded.selected_dictionary_key == KEYS[2048]


def test_an_edited_artifact_is_caught_by_its_own_digest(tmp_path):
    path = save_selection(a_report(), tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_k"] = 4096
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(SelectionError, match="modified"):
        load_selection(tmp_path)


def test_loading_where_no_selection_was_ever_written_names_the_stage_that_writes_it(tmp_path):
    with pytest.raises(SelectionError, match="select"):
        load_selection(tmp_path)


# --- Never overwritten --------------------------------------------------------


def test_a_second_write_does_not_overwrite_the_first(tmp_path):
    """A re-freeze is a deviation to be written down, not a file to be replaced."""
    save_selection(a_report(frozen_at=FROZEN_AT), tmp_path)

    with pytest.raises(SelectionError, match="already"):
        save_selection(a_report(frozen_at="2027-01-01T00:00:00+00:00"), tmp_path)

    assert load_selection(tmp_path).frozen_at == FROZEN_AT


def test_the_written_artifact_lands_whole_leaving_no_temporary_behind(tmp_path):
    save_selection(a_report(), tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == [SELECTION_FILENAME]


# --- No field can come from the test split ------------------------------------


def test_a_cell_cannot_be_built_from_a_result_measured_on_test(tmp_path):
    """The guard is the API, not the caller's care: this is the whole of HU-7's first half."""
    with pytest.raises(SelectionError, match="test"):
        SweepCell.from_result(a_result(split="test"))


def test_a_cell_cannot_be_built_from_a_result_on_any_split_but_dev():
    with pytest.raises(SelectionError, match="train"):
        SweepCell.from_result(a_result(split="train"))


def test_the_report_records_the_split_it_was_fitted_on_as_a_constant_not_a_parameter():
    assert DEV_SPLIT == "dev"
    assert SelectionReport.evaluated_on == DEV_SPLIT
    assert "evaluated_on" not in {field.name for field in fields(SelectionReport)}


def test_the_split_the_report_was_fitted_on_is_written_into_the_artifact(tmp_path):
    path = save_selection(a_report(), tmp_path)

    assert json.loads(path.read_text(encoding="utf-8"))["evaluated_on"] == DEV_SPLIT


def test_an_artifact_claiming_to_have_been_fitted_on_test_is_refused_on_load(tmp_path):
    path = save_selection(a_report(), tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluated_on"] = "test"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(SelectionError, match="dev"):
        load_selection(tmp_path)


# --- What the report refuses to be --------------------------------------------


def test_a_selected_k_that_is_not_in_the_table_is_refused():
    with pytest.raises(SelectionError, match="512"):
        a_report(selected_k=512)


def test_a_selected_dictionary_key_the_selected_row_does_not_name_is_refused():
    with pytest.raises(SelectionError, match="key4096"):
        a_report(selected_dictionary_key=KEYS[4096])


def test_a_table_row_filed_under_the_wrong_k_is_refused():
    with pytest.raises(SelectionError, match="1024"):
        a_report(per_k={1024: an_entry(2048)})


def test_the_reported_count_of_dev_decisions_must_match_the_ledger():
    """HU-7 wants the count reported, and a reported count that disagrees is worse than none."""
    with pytest.raises(SelectionError, match="n_dev_decisions"):
        a_report(n_dev_decisions=4)


def test_a_decision_whose_choice_is_not_among_its_alternatives_is_refused():
    with pytest.raises(SelectionError, match="alternatives"):
        Decision(
            name="damping",
            chosen="idf",
            alternatives=["none"],
            runner_up="none",
            margin=0.01,
        )


def test_a_decision_with_a_negative_margin_is_refused():
    """A margin is how far the winner won by; a negative one means the loser was recorded."""
    with pytest.raises(SelectionError, match="margin"):
        Decision(
            name="damping",
            chosen="idf",
            alternatives=["idf", "none"],
            runner_up="none",
            margin=-0.01,
        )


def test_a_configuration_that_could_not_reproduce_the_system_is_refused():
    incomplete = a_config()
    del incomplete["seed"]

    with pytest.raises(SelectionError, match="seed"):
        a_report(config=incomplete)


def test_a_result_without_the_phase_3_keys_cannot_become_a_cell():
    result = a_result()
    stripped = dict(result.config)
    del stripped["dictionary_key"]

    with pytest.raises(SelectionError, match="dictionary_key"):
        SweepCell.from_result(
            RunResult(
                system=result.system,
                split=result.split,
                config=stripped,
                metrics=result.metrics,
                cost=result.cost,
            )
        )


def test_a_k_row_built_from_cells_of_two_different_dictionaries_is_refused():
    mixed = [
        SweepCell.from_result(a_result(k=2048)),
        SweepCell.from_result(a_result(k=1024)),
    ]

    with pytest.raises(SelectionError, match="one dictionary"):
        PerKEntry.from_cells(mixed)


def test_a_k_row_without_a_single_measurement_is_refused():
    with pytest.raises(SelectionError, match="no cell"):
        PerKEntry.from_cells([])


# --- The best arm of each K ---------------------------------------------------


def test_the_best_arm_of_a_k_is_the_one_that_wins_the_selection_figure():
    cells = [
        SweepCell.from_result(a_result(view="raw", query_operator="projection_full", primary=0.55)),
        SweepCell.from_result(
            a_result(view="row_normalized", query_operator="sparse_coding", primary=0.71)
        ),
    ]

    entry = PerKEntry.from_cells(cells)

    assert entry.primary == 0.71
    assert entry.best_arm == {
        "query_operator": "sparse_coding",
        "view": "row_normalized",
        "damping": "none",
    }


def test_a_tie_between_two_arms_is_broken_the_same_way_every_time():
    """Two identical figures must not make the selection depend on iteration order."""
    tied = [
        SweepCell.from_result(a_result(view="row_normalized", primary=0.6)),
        SweepCell.from_result(a_result(view="raw", primary=0.6)),
    ]

    assert PerKEntry.from_cells(tied).best_arm == PerKEntry.from_cells(tied[::-1]).best_arm
    assert PerKEntry.from_cells(tied).best_arm["view"] == "raw"


def test_the_selection_figure_is_read_at_the_declared_metric_and_budget():
    cell = SweepCell.from_result(a_result(primary=0.62))

    assert cell.primary(config.SELECTION_METRIC, config.SELECTION_BUDGET) == 0.62
    assert cell.primary(config.SELECTION_METRIC, 512) == pytest.approx(0.57)


def test_a_selection_figure_the_cell_never_measured_is_refused_rather_than_defaulted():
    cell = SweepCell.from_result(a_result())

    with pytest.raises(SelectionError, match="8192"):
        cell.primary(config.SELECTION_METRIC, 8192)


# --- The structural columns Phase 2 asked for ---------------------------------


def test_the_structural_columns_are_absent_until_they_are_read_from_phase_2(tmp_path):
    """T12 fills them from the diagnostics artifact; nothing here invents a number for them."""
    entry = an_entry()

    assert entry.hub_share is None
    assert entry.hub_mass_share is None
    assert entry.thin_concepts is None


def test_the_structural_columns_survive_the_round_trip_beside_recall(tmp_path):
    report = a_report(
        per_k={
            2048: an_entry(2048, 0.62, hub_share=0.994, hub_mass_share=0.31, thin_concepts=118),
            1024: an_entry(1024, 0.59),
        }
    )

    save_selection(report, tmp_path)
    entry = load_selection(tmp_path).per_k[2048]

    assert (entry.hub_share, entry.hub_mass_share, entry.thin_concepts) == (0.994, 0.31, 118)


def test_the_tie_break_records_the_structural_criterion_when_it_was_applied(tmp_path):
    save_selection(a_report(tie_break="smaller hub: 0.61 vs 0.99 hub share"), tmp_path)

    assert load_selection(tmp_path).tie_break == "smaller hub: 0.61 vs 0.99 hub share"
