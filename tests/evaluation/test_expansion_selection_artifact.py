"""T10: the artifact that closes Phase 4's configuration, and what it refuses to be.

The pattern is Phase 3's, deliberately: the same shape, the same three properties,
the same primitives (decision D13). Inventing a second pattern for the second frozen
artifact of this project would mean two formats that agree until the day they do not.

What Phase 4 adds is what Phase 4 inherits. This artifact names the digest of the
frozen Phase 3 selection it was built on, and refuses to load against a different
one - because the four decisions K, view, damping and query operator are not
re-opened here, and "not re-opened" is a property that has to be checkable rather
than remembered. It also counts what the phase spent on dev, so the declared cap of
40 evaluations is auditable from the file rather than from the narrative.
"""

import json
from dataclasses import fields
from datetime import UTC, datetime

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.expansion_selection import (
    EXPANSION_FILENAME,
    ExpansionCell,
    ExpansionSelection,
    ExpansionSelectionError,
    expansion_path,
    load_expansion_selection,
    save_expansion_selection,
)
from concept_embeddings_rag.evaluation.harness import RunResult
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    Decision,
    SelectionError,
    check_freeze_precedes,
)

POOL = "101f564fdcca620c"
KEY = "d84c327aa8af0cdc"
PHASE_3_DIGEST = "91daa10ef0a75b6e" + "0" * 48
FROZEN_AT = "2026-09-13T09:41:07+00:00"

INHERITED = {
    "k": 512,
    "dictionary_key": KEY,
    "view": "raw",
    "damping": "idf",
    "query_operator": "projection_full",
}


def a_config(**overrides) -> dict:
    frozen = {
        "model": "fake-model",
        "revision": "v0",
        "unit_set_hash": POOL,
        "seed": 42,
        "tokenizer": "fake-tokenizer",
        "code_version": "0.1.0",
        "top_k": 100,
        "dictionary_key": KEY,
    }
    frozen.update(overrides)
    return frozen


def metrics(full_support: float = 0.83, gold_recall: float = 0.91) -> dict:
    return {
        f"budget_{budget}": {
            "full_support": full_support,
            "gold_recall": gold_recall,
            "precision": 0.1,
        }
        for budget in config.CONTEXT_BUDGETS
    }


def a_result(
    *,
    split: str = DEV_SPLIT,
    seed_arm: str = "dense",
    restart: float = 0.4,
    normalization: str = "symmetric",
    mean_iterations: float = 2.3,
    full_support: float = 0.83,
    created_at: str | None = None,
) -> RunResult:
    payload = {
        "system": "expansion",
        "split": split,
        "config": a_config(
            seed_arm=seed_arm,
            restart=restart,
            normalization=normalization,
            mean_iterations=mean_iterations,
        ),
        "metrics": metrics(full_support),
        "cost": {"n_questions": 600},
    }
    if created_at is not None:
        return RunResult(**payload, created_at=created_at)
    return RunResult(**payload)


def a_cell(
    seed_arm: str = "dense",
    restart: float = 0.4,
    normalization: str = "symmetric",
    full_support: float = 0.83,
    mean_iterations: float = 2.3,
) -> ExpansionCell:
    return ExpansionCell.from_result(
        a_result(
            seed_arm=seed_arm,
            restart=restart,
            normalization=normalization,
            full_support=full_support,
            mean_iterations=mean_iterations,
        )
    )


def a_decision(chosen: str = "restart=0.4/symmetric") -> Decision:
    return Decision(
        name="expansion_cell",
        chosen=chosen,
        alternatives=[chosen, "restart=0.2/none"],
        runner_up="restart=0.2/none",
        margin=0.012,
    )


def a_report(**overrides) -> ExpansionSelection:
    payload = {
        "selection_metric": config.EXPANSION_SELECTION_METRIC,
        "selection_budget": config.EXPANSION_SELECTION_BUDGET,
        "cells": [a_cell(), a_cell(restart=0.2, normalization="none", full_support=0.818)],
        "chosen": "restart=0.4/symmetric",
        "chosen_restart": 0.4,
        "chosen_normalization": "symmetric",
        "runner_up": "restart=0.2/none",
        "margin": 0.012,
        "resolution": 0.0091,
        "resolvable": True,
        "tie_break": None,
        "decisions": [a_decision()],
        "n_dev_decisions": 1,
        "config": a_config(),
        "seed": 42,
        "code_version": "0.1.0",
        "inherits_selection_digest": PHASE_3_DIGEST,
        "inherits": dict(INHERITED),
        "dev_evaluations_spent": 24,
        "stop_threshold": config.STOP_THRESHOLD,
        "max_iterations": config.MAX_ITERATIONS,
        "frozen_at": FROZEN_AT,
    }
    payload.update(overrides)
    return ExpansionSelection(**payload)


# --- The round trip -----------------------------------------------------------


def test_the_report_survives_the_round_trip_with_every_field_intact(tmp_path):
    report = a_report()
    save_expansion_selection(report, tmp_path)

    read = load_expansion_selection(tmp_path)

    for field in fields(ExpansionSelection):
        assert getattr(read, field.name) == getattr(report, field.name), field.name


def test_the_artifact_is_plain_json_a_human_can_read(tmp_path):
    save_expansion_selection(a_report(), tmp_path)
    payload = json.loads(expansion_path(tmp_path).read_text(encoding="utf-8"))

    assert payload["evaluated_on"] == DEV_SPLIT
    assert payload["chosen"] == "restart=0.4/symmetric"
    assert payload["dev_evaluations_spent"] == 24
    assert payload["inherits"]["damping"] == "idf"
    assert expansion_path(tmp_path).name == EXPANSION_FILENAME


def test_the_declared_constants_travel_so_a_later_run_can_prove_it_used_them(tmp_path):
    save_expansion_selection(a_report(), tmp_path)
    read = load_expansion_selection(tmp_path)

    assert read.stop_threshold == config.STOP_THRESHOLD
    assert read.max_iterations == config.MAX_ITERATIONS


def test_the_four_budget_table_is_kept_whole_and_not_reduced_to_the_one_that_chose(tmp_path):
    save_expansion_selection(a_report(), tmp_path)
    read = load_expansion_selection(tmp_path)

    for cell in read.cells:
        assert sorted(cell.metrics) == sorted(f"budget_{b}" for b in config.CONTEXT_BUDGETS)


def test_the_written_artifact_lands_whole_leaving_no_temporary_behind(tmp_path):
    save_expansion_selection(a_report(), tmp_path)

    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".json"]


def test_a_second_write_does_not_overwrite_the_first(tmp_path):
    save_expansion_selection(a_report(), tmp_path)

    with pytest.raises(ExpansionSelectionError, match="already frozen"):
        save_expansion_selection(a_report(), tmp_path)


# --- No field can come from the test split ------------------------------------


def test_a_cell_cannot_be_built_from_a_result_measured_on_another_split():
    """The door of HU-6: a finished measurement on test is refused entry to the table."""
    with pytest.raises(SelectionError, match="test"):
        ExpansionCell.from_result(a_result(split="test"))


def test_the_report_records_the_split_it_was_fitted_on_as_a_constant_not_a_parameter():
    assert ExpansionSelection.evaluated_on == DEV_SPLIT
    assert "evaluated_on" not in {field.name for field in fields(ExpansionSelection)}


def test_an_artifact_claiming_to_have_been_fitted_on_anything_else_is_refused_on_load(tmp_path):
    """Checked before the digest: no hash makes a configuration fitted on test usable."""
    save_expansion_selection(a_report(), tmp_path)
    path = expansion_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluated_on"] = "test"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ExpansionSelectionError, match="fitted on"):
        load_expansion_selection(tmp_path)


# --- Verified on load, in the declared order ----------------------------------


def test_an_edited_artifact_is_caught_by_its_own_digest(tmp_path):
    save_expansion_selection(a_report(), tmp_path)
    path = expansion_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["chosen_restart"] = 0.8
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ExpansionSelectionError, match="modified"):
        load_expansion_selection(tmp_path)


def test_a_report_computed_over_another_pool_is_refused_on_load(tmp_path):
    save_expansion_selection(a_report(), tmp_path)

    with pytest.raises(ExpansionSelectionError, match="pool"):
        load_expansion_selection(tmp_path, expected_unit_set_hash="another-pool")


def test_a_report_computed_over_this_pool_loads(tmp_path):
    save_expansion_selection(a_report(), tmp_path)

    assert load_expansion_selection(tmp_path, expected_unit_set_hash=POOL).chosen_restart == 0.4


def test_a_report_naming_a_dictionary_that_is_not_on_disk_is_refused(tmp_path):
    save_expansion_selection(a_report(), tmp_path)

    with pytest.raises(ExpansionSelectionError, match="not on disk"):
        load_expansion_selection(tmp_path, known_dictionary_keys=["someone-elses-key"])


def test_a_phase_3_digest_that_does_not_match_the_artifact_on_disk_is_refused(tmp_path):
    """The inheritance made checkable: this is the freeze Phase 4 was built on, or none."""
    save_expansion_selection(a_report(), tmp_path)

    with pytest.raises(ExpansionSelectionError, match="inherit"):
        load_expansion_selection(tmp_path, inherits_selection_digest="f" * 64)


def test_the_phase_3_digest_it_was_built_on_loads(tmp_path):
    save_expansion_selection(a_report(), tmp_path)

    read = load_expansion_selection(tmp_path, inherits_selection_digest=PHASE_3_DIGEST)

    assert read.inherits_selection_digest == PHASE_3_DIGEST


def test_loading_where_nothing_was_ever_written_names_the_stage_that_writes_it(tmp_path):
    with pytest.raises(ExpansionSelectionError, match="expand"):
        load_expansion_selection(tmp_path)


# --- The invariants of the schema ---------------------------------------------


def test_a_chosen_cell_that_was_never_measured_is_refused():
    with pytest.raises(ExpansionSelectionError, match="never measured"):
        a_report(
            chosen="restart=0.8/stochastic",
            chosen_restart=0.8,
            chosen_normalization="stochastic",
        )


def test_the_chosen_label_has_to_agree_with_the_restart_and_normalization_it_names():
    with pytest.raises(ExpansionSelectionError, match="chosen"):
        a_report(chosen_restart=0.2)


def test_a_report_without_a_single_measured_cell_is_refused():
    with pytest.raises(ExpansionSelectionError, match="cell"):
        a_report(cells=[])


def test_cells_measured_over_different_numbers_of_questions_are_refused():
    wrong = a_cell(restart=0.2, normalization="none")
    other = ExpansionCell(
        seed_arm="dense",
        restart=0.6,
        normalization="none",
        mean_iterations=1.0,
        n_questions=599,
        metrics=metrics(),
    )
    with pytest.raises(ExpansionSelectionError, match="questions"):
        a_report(cells=[a_cell(), wrong, other])


def test_the_reported_count_of_dev_decisions_must_match_the_ledger():
    with pytest.raises(ExpansionSelectionError, match="n_dev_decisions"):
        a_report(n_dev_decisions=3)


def test_a_ledger_that_does_not_read_in_the_declared_order_is_refused():
    with pytest.raises(ExpansionSelectionError, match="D8"):
        a_report(
            decisions=[
                Decision(
                    name="normalization",
                    chosen="none",
                    alternatives=["none"],
                    runner_up=None,
                    margin=0.0,
                )
            ]
        )


def test_a_configuration_that_could_not_reproduce_the_system_is_refused():
    with pytest.raises(ExpansionSelectionError, match="seed"):
        a_report(config={key: value for key, value in a_config().items() if key != "seed"})


def test_a_restart_outside_the_declared_grid_is_refused():
    with pytest.raises(ExpansionSelectionError, match="grid"):
        a_report(chosen="restart=0.5/symmetric", chosen_restart=0.5)


def test_a_normalization_arm_this_phase_does_not_have_is_refused():
    with pytest.raises(ExpansionSelectionError, match="normalization"):
        a_report(chosen="restart=0.4/laplacian", chosen_normalization="laplacian")


def test_more_dev_evaluations_than_the_declared_cap_is_refused():
    """HU-6 caps the phase at 40, and a report that spent more is a deviation."""
    with pytest.raises(ExpansionSelectionError, match="cap"):
        a_report(dev_evaluations_spent=config.DEV_EVALUATION_CAP + 1)


def test_a_report_that_spent_nothing_on_dev_is_refused():
    with pytest.raises(ExpansionSelectionError, match="spent"):
        a_report(dev_evaluations_spent=0)


def test_the_four_inherited_decisions_have_to_be_named():
    with pytest.raises(ExpansionSelectionError, match="inherit"):
        a_report(inherits={"k": 512})


def test_a_report_inheriting_no_phase_3_freeze_at_all_is_refused():
    with pytest.raises(ExpansionSelectionError, match="inherit"):
        a_report(inherits_selection_digest="")


def test_an_unresolvable_margin_without_a_tie_break_is_refused():
    """When the split cannot tell two cells apart, D8 says which criterion decided."""
    with pytest.raises(ExpansionSelectionError, match="tie"):
        a_report(resolvable=False, tie_break=None, margin=0.0001)


def test_a_report_that_disagrees_with_itself_about_what_the_split_resolves_is_refused():
    """`resolvable` is a reading of the margin against the resolution, not a third opinion."""
    with pytest.raises(ExpansionSelectionError, match="resolv"):
        a_report(resolvable=False, margin=0.5, tie_break="fewer iterations: 1.4 against 3.2")


def test_a_tie_break_recorded_where_the_margin_was_resolvable_is_refused():
    with pytest.raises(ExpansionSelectionError, match="tie"):
        a_report(resolvable=True, tie_break="fewer iterations: 1.4 against 3.2")


def test_a_frozen_at_that_is_not_a_timestamp_is_refused():
    with pytest.raises(ExpansionSelectionError, match="timestamp"):
        a_report(frozen_at="last tuesday")


def test_the_freeze_timestamp_is_stamped_when_the_report_is_built_if_it_is_not_given():
    before = datetime.now(UTC)
    report = ExpansionSelection(
        **{key: value for key, value in a_report().__dict__.items() if key != "frozen_at"}
    )

    assert datetime.fromisoformat(report.frozen_at) >= before.replace(microsecond=0)


# --- The cell reads itself off the run that produced it -----------------------


def test_a_cell_describes_itself_out_of_the_result_config_rather_than_out_of_labels():
    cell = ExpansionCell.from_result(
        a_result(
            seed_arm="conceptual", restart=0.8, normalization="stochastic", mean_iterations=1.2
        )
    )

    assert (cell.seed_arm, cell.restart, cell.normalization) == ("conceptual", 0.8, "stochastic")
    assert cell.mean_iterations == 1.2
    assert cell.n_questions == 600
    assert cell.label == "restart=0.8/stochastic"


def test_a_result_that_cannot_say_which_cell_it_measured_is_refused():
    broken = RunResult(
        system="expansion",
        split=DEV_SPLIT,
        config=a_config(restart=0.4),
        metrics=metrics(),
        cost={"n_questions": 600},
    )

    with pytest.raises(SelectionError, match="normalization"):
        ExpansionCell.from_result(broken)


def test_the_mean_iterations_of_a_cell_is_what_the_run_counted_not_a_constant():
    assert a_cell(mean_iterations=4.87).mean_iterations == 4.87


def test_the_selection_figure_is_read_at_the_declared_metric_and_budget():
    cell = a_cell(full_support=0.77)

    assert (
        cell.primary(config.EXPANSION_SELECTION_METRIC, config.EXPANSION_SELECTION_BUDGET) == 0.77
    )


def test_a_selection_figure_the_cell_never_measured_is_refused_rather_than_defaulted():
    with pytest.raises(SelectionError, match="8192"):
        a_cell().primary(config.EXPANSION_SELECTION_METRIC, 8192)


# --- T9: the generalized freeze check serves this report too ------------------


def test_a_result_created_before_this_phase_froze_is_refused():
    report = a_report(frozen_at="2026-09-13T09:41:07+00:00")

    with pytest.raises(SelectionError, match="before this configuration froze"):
        check_freeze_precedes(report, a_result(created_at="2026-09-13T09:00:00+00:00"))


def test_a_result_created_after_this_phase_froze_is_accepted():
    report = a_report(frozen_at="2026-09-13T09:41:07+00:00")

    check_freeze_precedes(report, a_result(created_at="2026-09-13T10:00:00+00:00"))
