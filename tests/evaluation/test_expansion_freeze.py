"""T12: the freeze, and the ordering it has to make observable.

HU-6 asks for the configuration to be closed **before the test split is read**, and
the observable form of that is a stamp every test-split result of this phase was
created after. `frozen_at` is therefore stamped in exactly one place - the report's
own default - and `check_freeze_precedes` is applied per result rather than once per
run, because a check that runs once proves a property about one result.
"""

from datetime import UTC, datetime, timedelta

import pytest
from expansion_fixtures import PHASE_3_DIGEST, a_config, a_dictionary, inherited_for

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.expansion_selection import (
    DENSE_ARM,
    ExpansionCell,
    ExpansionSelectionError,
    choose_cell,
    freeze_expansion_selection,
    load_expansion_selection,
    save_expansion_selection,
)
from concept_embeddings_rag.evaluation.harness import RunResult
from concept_embeddings_rag.evaluation.selection import SelectionError, check_freeze_precedes

METRIC = config.EXPANSION_SELECTION_METRIC
BUDGET = config.EXPANSION_SELECTION_BUDGET


def a_cell(restart, normalization, primary, mean_iterations=2.0, seed_arm=DENSE_ARM):
    return ExpansionCell(
        seed_arm=seed_arm,
        restart=restart,
        normalization=normalization,
        mean_iterations=mean_iterations,
        n_questions=600,
        metrics={f"budget_{BUDGET}": {METRIC: primary, "gold_recall": 0.9, "precision": 0.1}},
    )


def a_grid() -> list[ExpansionCell]:
    cells = []
    for restart in config.RESTART_GRID:
        for normalization in config.NORMALIZATION_ARMS:
            figure = 0.86 if (restart, normalization) == (0.4, "symmetric") else 0.50
            cells.append(a_cell(restart, normalization, figure))
            cells.append(a_cell(restart, normalization, figure - 0.2, seed_arm="conceptual"))
    return cells


def a_freeze(**overrides):
    dictionary = a_dictionary()
    cells = overrides.pop("cells", a_grid())
    arguments = {
        "cells": cells,
        "choice": choose_cell(cells),
        "base_config": a_config(),
        "inherits": inherited_for(dictionary),
        "inherits_selection_digest": PHASE_3_DIGEST,
        "dev_evaluations_spent": 24,
    }
    arguments.update(overrides)
    return freeze_expansion_selection(**arguments)


def a_result(created_at: str, split: str = "test") -> RunResult:
    return RunResult(
        system="expansion",
        split=split,
        config=a_config(seed_arm="dense", restart=0.4, normalization="symmetric"),
        metrics={f"budget_{BUDGET}": {METRIC: 0.86}},
        cost={"n_questions": 1400},
        created_at=created_at,
    )


# --- What the freeze writes down ----------------------------------------------


def test_the_freeze_records_the_chosen_cell_and_the_grid_it_was_chosen_from():
    report = a_freeze()

    assert report.chosen == "restart=0.4/symmetric"
    assert report.chosen_restart == 0.4
    assert report.chosen_normalization == "symmetric"
    assert len(report.cells) == 24


def test_the_frozen_configuration_carries_what_it_takes_to_rebuild_the_system():
    report = a_freeze()

    assert report.config["restart"] == 0.4
    assert report.config["normalization"] == "symmetric"
    assert report.config["seed_top_k"] == config.SEED_TOP_K
    assert report.config["dictionary_key"] == a_dictionary().key


def test_the_four_inherited_decisions_travel_with_the_freeze():
    report = a_freeze()

    assert report.inherits["view"] == "raw"
    assert report.inherits["damping"] == "idf"
    assert report.inherits["query_operator"] == "projection_full"
    assert report.inherits_selection_digest == PHASE_3_DIGEST


def test_the_ledger_holds_the_one_decision_and_the_count_agrees():
    report = a_freeze()

    assert report.n_dev_decisions == 1
    assert report.decisions[0].name == "expansion_cell"
    assert report.decisions[0].chosen == "restart=0.4/symmetric"


def test_the_declared_constants_are_recorded_rather_than_assumed():
    report = a_freeze()

    assert report.stop_threshold == config.STOP_THRESHOLD
    assert report.max_iterations == config.MAX_ITERATIONS
    assert report.dev_evaluations_spent == 24


def test_the_seed_and_the_code_version_come_out_of_the_configuration():
    report = a_freeze()

    assert report.seed == config.DEFAULT_SEED
    assert report.code_version == "0.1.0"


def test_a_freeze_that_spent_more_than_the_cap_is_refused():
    with pytest.raises(ExpansionSelectionError, match="cap"):
        a_freeze(dev_evaluations_spent=config.DEV_EVALUATION_CAP + 1)


def test_a_freeze_whose_configuration_could_not_reproduce_it_is_refused():
    with pytest.raises(SelectionError, match="seed"):
        a_freeze(base_config={k: v for k, v in a_config().items() if k != "seed"})


# --- Stamped in one place only ------------------------------------------------


def test_the_freeze_stamps_the_moment_it_closed():
    before = datetime.now(UTC).replace(microsecond=0)

    report = a_freeze()

    assert datetime.fromisoformat(report.frozen_at) >= before


def test_the_stamp_is_not_a_parameter_anyone_can_hand_the_freeze():
    """One place stamps it, so a second freeze cannot be backdated to look like the first."""
    import inspect

    assert "frozen_at" not in inspect.signature(freeze_expansion_selection).parameters


def test_the_stamp_survives_the_round_trip_to_the_second(tmp_path):
    report = a_freeze()
    save_expansion_selection(report, tmp_path)

    assert load_expansion_selection(tmp_path).frozen_at == report.frozen_at


# --- The ordering, checked per result -----------------------------------------


def test_a_test_result_created_before_the_freeze_is_refused():
    report = a_freeze()
    earlier = datetime.fromisoformat(report.frozen_at) - timedelta(seconds=1)

    with pytest.raises(SelectionError, match="before this configuration froze"):
        check_freeze_precedes(report, a_result(earlier.isoformat(timespec="seconds")))


def test_a_test_result_created_after_the_freeze_is_accepted():
    report = a_freeze()
    later = datetime.fromisoformat(report.frozen_at) + timedelta(seconds=1)

    check_freeze_precedes(report, a_result(later.isoformat(timespec="seconds")))


def test_the_check_is_the_one_phase_3_wrote_rather_than_a_second_copy():
    """D13: one comparison, serving both frozen artifacts."""
    from concept_embeddings_rag.evaluation import selection

    assert check_freeze_precedes is selection.check_freeze_precedes
