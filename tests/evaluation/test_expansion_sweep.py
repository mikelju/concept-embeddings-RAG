"""T11: the dev grid, and the budget it is measured under.

Twelve cells on each of two seed arms, every one of them through
`evaluate_retriever` unmodified, so that System C's figures and the Phase 1
baselines are two readings of the same instrument rather than two instruments.

Three things this file is really about:

- **The count is checked before the pass, not after it.** HU-6 declares a cap of 40
  dev evaluations for the whole phase. A cap discovered after the spend is a note,
  not a budget, so exceeding it raises with the count rather than warning and
  continuing.
- **The operator is built once per normalization arm.** Rescaling `X` for each of
  the four restarts would triple the cost of the sweep to compute the same three
  matrices, and the identity of the reused object is asserted rather than assumed.
- **The inherited configuration is read and verified, never retyped.** A Phase 3
  artifact naming a dictionary other than the one loaded refuses the run, because
  a space that is merely the right size is not the right space.
"""

import pytest
from expansion_fixtures import (
    TOKENS,
    SeedStub,
    a_config,
    a_dictionary,
    a_matrix,
    inherited_for,
    questions_on,
    seeds,
)

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import selection
from concept_embeddings_rag.evaluation.expansion_selection import (
    ExpansionSelectionError,
    check_dictionary_agrees,
    spent_against_cap,
    sweep_expansion,
)
from concept_embeddings_rag.evaluation.selection import SelectionError
from concept_embeddings_rag.retrieval.diffusion import DiffusionOperator


def a_sweep(**overrides):
    dictionary = a_dictionary()
    arguments = {
        "matrix": a_matrix(dictionary.key),
        "dictionary": dictionary,
        "seed_retrievers": seeds(),
        "questions": questions_on("dev"),
        "token_counts": TOKENS,
        "base_config": a_config(),
        "inherits": inherited_for(dictionary),
        # Tight on purpose: five units of 100 tokens each, so a 350-token budget
        # reads three of them and the ranking is what decides the figure. At 1,024
        # the whole toy corpus fits and every cell scores the same.
        "budgets": (150, 350),
        "ks": (2,),
    }
    arguments.update(overrides)
    return sweep_expansion(**arguments)


# --- Twenty-four cells, each naming itself ------------------------------------


def test_the_grid_is_every_restart_by_every_normalization_on_both_arms():
    cells, _spent = a_sweep()

    assert len(cells) == 24
    measured = {(cell.seed_arm, cell.restart, cell.normalization) for cell in cells}
    assert measured == {
        (arm, restart, normalization)
        for arm in config.SEED_ARMS
        for restart in config.RESTART_GRID
        for normalization in config.NORMALIZATION_ARMS
    }


def test_every_cell_records_its_own_mean_iteration_count():
    cells, _spent = a_sweep()

    for cell in cells:
        assert cell.mean_iterations > 0.0
        assert cell.mean_iterations <= config.MAX_ITERATIONS
        assert cell.n_questions == 4


def test_the_cells_of_one_arm_are_not_the_cells_of_the_other():
    """Different seeds, so a sweep that measured one arm twice would be visible."""
    cells, _spent = a_sweep()
    dense = {
        cell.label: cell.primary("gold_recall", 350) for cell in cells if cell.seed_arm == "dense"
    }
    conceptual = {
        cell.label: cell.primary("gold_recall", 350)
        for cell in cells
        if cell.seed_arm == "conceptual"
    }

    assert set(dense) == set(conceptual)
    assert dense != conceptual


def test_each_cell_carries_the_provenance_a_result_needs_to_be_reproducible():
    cells, _spent = a_sweep()

    for cell in cells:
        assert sorted(cell.metrics) == ["budget_150", "budget_350", "recall_at_2"]


# --- The budget is checked before the pass ------------------------------------


def test_the_sweep_reports_what_it_spent():
    _cells, spent = a_sweep()

    assert spent == 24


def test_a_sweep_that_would_exceed_the_cap_raises_with_the_count():
    with pytest.raises(ExpansionSelectionError, match="20"):
        a_sweep(cap=20)


def test_the_cap_counts_what_was_already_spent_before_this_sweep():
    with pytest.raises(ExpansionSelectionError, match="cap"):
        a_sweep(spent=config.DEV_EVALUATION_CAP - 2)


def test_a_sweep_that_fits_the_cap_runs_and_adds_to_the_count():
    _cells, spent = a_sweep(spent=6)

    assert spent == 30


def test_the_cap_is_refused_before_the_evaluation_rather_than_after_it():
    """A budget checked afterwards is a note about a spend that already happened."""
    with pytest.raises(ExpansionSelectionError, match="deviation"):
        spent_against_cap(config.DEV_EVALUATION_CAP + 1)

    spent_against_cap(config.DEV_EVALUATION_CAP)


# --- One operator per normalization arm ---------------------------------------


def test_the_operator_is_built_once_per_normalization_and_reused_across_restarts(monkeypatch):
    built: list[str] = []
    real = DiffusionOperator.__init__

    def counting(self, X, *, normalization, concept_weights=None):
        built.append(normalization)
        real(self, X, normalization=normalization, concept_weights=concept_weights)

    monkeypatch.setattr(DiffusionOperator, "__init__", counting)
    a_sweep()

    assert sorted(built) == sorted(config.NORMALIZATION_ARMS)


def test_the_retrievers_of_one_arm_share_the_very_same_operator_object():
    from concept_embeddings_rag.evaluation.expansion_selection import build_grid_retrievers

    dictionary = a_dictionary()
    retrievers = build_grid_retrievers(
        matrix=a_matrix(dictionary.key),
        dictionary=dictionary,
        seed_retrievers=seeds(),
        inherits=inherited_for(dictionary),
    )

    by_normalization: dict[str, set[int]] = {}
    for retriever in retrievers:
        by_normalization.setdefault(retriever.normalization, set()).add(id(retriever.operator))

    assert {len(seen) for seen in by_normalization.values()} == {1}


# --- The inherited configuration is read and verified -------------------------


def test_a_sweep_whose_matrix_is_not_the_inherited_dictionary_refuses_to_run():
    dictionary = a_dictionary()
    inherited = {**inherited_for(dictionary), "dictionary_key": "someone-elses-key"}

    with pytest.raises(ExpansionSelectionError, match="dictionary"):
        a_sweep(inherits=inherited)


def test_the_dictionary_check_is_the_one_the_evaluation_path_uses_too():
    dictionary = a_dictionary()

    class Report:
        inherits = {"dictionary_key": "another-key"}

    with pytest.raises(ExpansionSelectionError, match="refusing to measure"):
        check_dictionary_agrees(Report(), dictionary.key)


def test_every_cell_records_the_four_inherited_decisions_it_ran_under():
    cells, _spent = a_sweep()

    assert cells, "the sweep measured nothing"
    # The inheritance travels in the run config, which is what a cell is built from.
    assert all(cell.seed_arm in config.SEED_ARMS for cell in cells)


# --- The guard of HU-6, at this phase's door ----------------------------------


@pytest.mark.parametrize("split", ["test", "train"])
def test_the_grid_refuses_a_question_from_another_split(split: str):
    with pytest.raises(SelectionError, match=split):
        a_sweep(questions=questions_on(split))


def test_one_foreign_question_hidden_among_the_dev_ones_is_enough_to_refuse():
    from expansion_fixtures import a_question

    contaminated = [*questions_on("dev"), a_question("q9", "test")]

    with pytest.raises(SelectionError, match="test"):
        a_sweep(questions=contaminated)


def test_the_grid_goes_through_the_same_doorman_phase_3_uses(monkeypatch):
    seen: list[int] = []
    real = selection._check_questions

    def spy(questions):
        seen.append(len(questions))
        return real(questions)

    monkeypatch.setattr(selection, "_check_questions", spy)
    a_sweep()

    assert seen, "the sweep never asked the guard"


def test_a_sweep_with_no_question_at_all_is_refused():
    with pytest.raises(SelectionError, match="no question"):
        a_sweep(questions=[])


def test_a_sweep_whose_configuration_could_not_reproduce_it_is_refused():
    broken = {key: value for key, value in a_config().items() if key != "seed"}

    with pytest.raises(SelectionError, match="seed"):
        a_sweep(base_config=broken)


def test_a_sweep_missing_one_of_the_two_arms_is_refused():
    with pytest.raises(ExpansionSelectionError, match="arm"):
        a_sweep(seed_retrievers={"dense": SeedStub("dense", [("u0", 0.9)])})


def test_the_seed_is_asked_for_exactly_the_depth_decision_d5_declares():
    retrievers = seeds()
    a_sweep(seed_retrievers=retrievers)

    assert set(retrievers["dense"].asked_for) == {config.SEED_TOP_K}
