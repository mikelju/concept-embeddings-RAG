"""T12: the one decision this phase takes, and the two ties declared before it.

D8 fixes all of it in advance, which is the point: the cell is chosen on **Full
Support at 2,048 tokens on dev, on the dense-seeded arm**, and the conceptual arm
gets the same cell applied unchanged. Choosing separately per arm would make the two
differ in more than the seed, and the isolating variant would stop isolating
anything.

The two ties, resolved in the conservative direction before any number existed:

- **Margin below what 600 questions resolve** -> the cell with fewer mean iterations
  wins. §73 is the reason, and it hands a close call to a declared criterion instead
  of to a few thousandths.
- **Equal figure and equal cost** -> the larger restart wins. More restart is less
  expansion, so an undecided measurement cannot arrive as evidence for the system
  under test.
"""

import pytest
from expansion_fixtures import (
    a_dictionary,
    a_matrix,
    inherited_for,
    seeds,
)

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.expansion_selection import (
    DENSE_ARM,
    EXPANSION_DECISION_ORDER,
    ExpansionCell,
    ExpansionSelectionError,
    build_expansion_retrievers,
    choose_cell,
)

METRIC = config.EXPANSION_SELECTION_METRIC
BUDGET = config.EXPANSION_SELECTION_BUDGET


def a_cell(
    restart: float,
    normalization: str,
    primary: float,
    mean_iterations: float = 2.0,
    seed_arm: str = DENSE_ARM,
    n_questions: int = 600,
) -> ExpansionCell:
    return ExpansionCell(
        seed_arm=seed_arm,
        restart=restart,
        normalization=normalization,
        mean_iterations=mean_iterations,
        n_questions=n_questions,
        metrics={f"budget_{BUDGET}": {METRIC: primary, "gold_recall": 0.9, "precision": 0.1}},
    )


def a_grid(**primaries: float) -> list[ExpansionCell]:
    """Twelve dense cells with a flat figure, overridden where a test cares."""
    cells = []
    for restart in config.RESTART_GRID:
        for normalization in config.NORMALIZATION_ARMS:
            label = f"{restart}_{normalization}"
            cells.append(a_cell(restart, normalization, primaries.get(label, 0.50)))
    return cells


# --- The choice is on the dense arm, at the declared figure -------------------


def test_the_cell_with_the_best_full_support_at_2048_wins():
    cells = a_grid(**{"0.6_symmetric": 0.86})

    choice = choose_cell(cells)

    assert (choice.restart, choice.normalization) == (0.6, "symmetric")
    assert choice.label == "restart=0.6/symmetric"
    assert choice.primary == 0.86


def test_the_choice_reads_the_dense_arm_and_ignores_the_conceptual_one():
    """D8: the conceptual arm is measured and reported, never consulted for the choice."""
    cells = [
        *a_grid(**{"0.2_none": 0.70}),
        a_cell(0.8, "stochastic", 0.99, seed_arm="conceptual"),
    ]

    choice = choose_cell(cells)

    assert (choice.restart, choice.normalization) == (0.2, "none")


def test_a_grid_with_no_dense_cell_is_refused():
    with pytest.raises(ExpansionSelectionError, match=DENSE_ARM):
        choose_cell([a_cell(0.4, "none", 0.8, seed_arm="conceptual")])


def test_an_empty_grid_is_refused():
    with pytest.raises(ExpansionSelectionError, match="no cell"):
        choose_cell([])


def test_cells_measured_over_different_question_counts_cannot_be_compared():
    cells = [*a_grid(), a_cell(0.2, "none", 0.9, n_questions=599)]

    with pytest.raises(ExpansionSelectionError, match="questions"):
        choose_cell(cells)


def test_the_selection_figure_is_the_one_config_declares():
    assert METRIC == "full_support"
    assert BUDGET == 2048


# --- The ledger: one decision, twelve alternatives ----------------------------


def test_the_ledger_holds_one_decision_over_the_twelve_cells_of_the_grid():
    choice = choose_cell(a_grid(**{"0.4_none": 0.9}))
    decision = choice.as_decision()

    assert decision.name == EXPANSION_DECISION_ORDER[0]
    assert len(decision.alternatives) == 12
    assert decision.chosen == "restart=0.4/none"


def test_the_runner_up_and_the_margin_are_recorded_whether_or_not_a_tie_fired():
    choice = choose_cell(a_grid(**{"0.4_none": 0.90, "0.2_symmetric": 0.80}))

    assert choice.runner_up == "restart=0.2/symmetric"
    assert choice.margin == pytest.approx(0.10)
    assert choice.tie_break is None


def test_the_alternatives_are_the_cells_in_the_order_the_grid_declares_them():
    choice = choose_cell(a_grid(**{"0.2_none": 0.9}))

    assert choice.alternatives[0] == "restart=0.2/none"
    assert choice.alternatives[-1] == "restart=0.8/stochastic"


# --- The first tie: cost decides what the split cannot ------------------------


def test_a_margin_below_what_the_split_resolves_triggers_the_cost_tie_break():
    """0.8300 against 0.8305 on 600 questions is noise, and §73 says which wins."""
    cells = a_grid()
    cells = [
        a_cell(0.2, "none", 0.8305, mean_iterations=4.4),
        a_cell(0.4, "symmetric", 0.8300, mean_iterations=1.4),
        *[
            cell
            for cell in cells
            if cell.label not in {"restart=0.2/none", "restart=0.4/symmetric"}
        ],
    ]

    choice = choose_cell(cells)

    assert (choice.restart, choice.normalization) == (0.4, "symmetric")
    assert choice.tie_break is not None
    assert "iterations" in choice.tie_break
    assert choice.resolvable is False


def test_the_tie_break_names_both_costs_so_the_artifact_says_how_it_was_decided():
    cells = [
        a_cell(0.2, "none", 0.8305, mean_iterations=4.4),
        a_cell(0.4, "symmetric", 0.8300, mean_iterations=1.4),
    ]

    choice = choose_cell(cells)

    assert "1.4" in choice.tie_break
    assert "4.4" in choice.tie_break


def test_a_resolvable_margin_leaves_the_leader_alone_however_expensive_it_is():
    cells = [
        a_cell(0.2, "none", 0.90, mean_iterations=5.0),
        a_cell(0.4, "symmetric", 0.70, mean_iterations=1.0),
    ]

    choice = choose_cell(cells)

    assert (choice.restart, choice.normalization) == (0.2, "none")
    assert choice.tie_break is None
    assert choice.resolvable is True


def test_the_resolution_recorded_is_the_phase_3_function_unchanged():
    from concept_embeddings_rag.evaluation.selection import resolvable_margin

    cells = a_grid(**{"0.4_none": 0.83})
    choice = choose_cell(cells)

    assert choice.resolution == pytest.approx(resolvable_margin(0.83, 600))


# --- The second tie: the larger restart claims less ---------------------------


def test_equal_figure_and_equal_cost_resolve_toward_the_larger_restart():
    cells = [
        a_cell(0.2, "none", 0.83, mean_iterations=2.0),
        a_cell(0.8, "none", 0.83, mean_iterations=2.0),
    ]

    choice = choose_cell(cells)

    assert choice.restart == 0.8


def test_a_cheaper_cell_still_beats_a_larger_restart_that_costs_more():
    """Cost is the first tie-break; the restart only decides what cost cannot."""
    cells = [
        a_cell(0.2, "none", 0.83, mean_iterations=1.2),
        a_cell(0.8, "none", 0.83, mean_iterations=3.0),
    ]

    assert choose_cell(cells).restart == 0.2


# --- The chosen cell is applied unchanged to the other arm --------------------


def test_both_arms_are_built_from_the_same_cell_and_differ_only_in_the_seed():
    dictionary = a_dictionary()
    matrix = a_matrix(dictionary.key)
    retrievers = build_expansion_retrievers(
        matrix=matrix,
        dictionary=dictionary,
        seed_retrievers=seeds(),
        restart=0.6,
        normalization="symmetric",
        inherits=inherited_for(dictionary),
    )

    assert sorted(retriever.seed_arm for retriever in retrievers) == sorted(config.SEED_ARMS)
    described = [retriever.describe() for retriever in retrievers]
    for key in described[0]:
        if key in {"seed_arm", "seed_system", "name"}:
            continue
        assert described[0][key] == described[1][key], key


def test_the_two_arms_are_named_by_their_seed_and_cannot_claim_the_other():
    dictionary = a_dictionary()
    retrievers = build_expansion_retrievers(
        matrix=a_matrix(dictionary.key),
        dictionary=dictionary,
        seed_retrievers=seeds(),
        restart=0.4,
        normalization="none",
        inherits=inherited_for(dictionary),
    )
    by_name = {retriever.name: retriever for retriever in retrievers}

    assert by_name["expansion"].seed_arm == "dense"
    assert by_name["expansion-conceptual"].seed_arm == "conceptual"


def test_building_the_arms_without_both_seeds_is_refused():
    dictionary = a_dictionary()

    with pytest.raises(ExpansionSelectionError, match="arm"):
        build_expansion_retrievers(
            matrix=a_matrix(dictionary.key),
            dictionary=dictionary,
            seed_retrievers={"dense": seeds()["dense"]},
            restart=0.4,
            normalization="none",
            inherits=inherited_for(dictionary),
        )
