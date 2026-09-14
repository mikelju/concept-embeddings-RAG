"""T11: the dev sweep that chooses the space, and the four things it may not do.

Phase 2 induced four spaces and chose none. This is the measurement that chooses:
every K crossed with every query operator and every view, conceptual-only, on the
600 dev questions, judged by gold recall at 2,048 tokens.

What the tests here pin down is not the ranking - that is a result, not a contract -
but the four properties that make the ranking readable afterwards:

- it touches dev and nothing else, and says so by refusing anything else;
- every cell names the dictionary that produced it, so a row cannot be read as
  belonging to a space it was not measured over;
- the four-budget table survives whole, so a winner that only wins at 2,048 is
  visible as such;
- re-running it reproduces the table exactly, because a sweep that drifts cannot
  be the evidence for a freeze.

Nothing here measures anything real: the backend reads the query string as a vector
and the matrices are three units wide, so the arithmetic of every cell is checkable
by hand and the whole file runs in under a second.
"""

from dataclasses import fields
from itertools import product

import numpy as np
import pytest
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.concepts.coding import ConceptMatrix, load_matrix, row_normalized
from concept_embeddings_rag.concepts.diagnostics import compute_diagnostics, load_diagnostics
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary, load_dictionary
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import selection
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    PerKEntry,
    SelectionError,
    SweepCell,
    SweepSpace,
    choose_space,
    resolvable_margin,
    run_dev_sweep,
    structure_of,
    sweep_space,
)


class DirectionBackend:
    """Encodes a query as the literal vector written in the query string."""

    name = "fake"
    revision = "v0"
    dim = 4
    normalize = True

    def encode(self, texts):
        return np.array([[float(x) for x in t.split(",")] for t in texts], dtype=np.float32)


class CountingDictionary(ConceptDictionary):
    """A dictionary that counts how many times the coder was actually run."""

    def encode(self, vectors, alpha=None):
        type(self).calls += 1
        return super().encode(vectors, alpha=alpha)

    calls = 0


UNITS = ["focused", "broad", "absent"]
TOKEN_COUNTS = {"focused": 100, "broad": 100, "absent": 100}

# Two spaces, both wider than `config.QUERY_TOP_M`, so that `projection_top16`
# actually truncates: under 16 concepts it would silently become `projection_full`
# and the grid would measure one arm twice under two names.
SMALL_K = 32
LARGE_K = 64


def a_dictionary(k: int, kind=ConceptDictionary) -> ConceptDictionary:
    """`k` atoms over four dimensions, so a projection is readable off the query."""
    return kind(
        atoms=np.eye(k, 4, dtype=np.float32),
        k=k,
        seed=42,
        sparsity_param=0.05,
        max_iter=3,
        unit_set_hash="101f564fdcca620c",
        model="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c",
    )


def a_matrix(dictionary: ConceptDictionary, view: str) -> ConceptMatrix:
    """Three units over `k` concepts: focused, broad, and one about something else."""
    k = dictionary.k
    rows = np.zeros((3, k), dtype=np.float32)
    rows[0, 0] = 1.0
    rows[1, 0] = 2.0
    rows[1, min(1, k - 1)] = 6.0
    rows[2, k - 1] = 3.0
    if view == "row_normalized":
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        rows = rows / np.where(norms == 0.0, 1.0, norms)

    X = sparse.csr_matrix(rows)
    return ConceptMatrix(
        X=X,
        unit_ids=list(UNITS),
        dictionary_key=dictionary.key,
        view=view,
        coding_alpha=0.07,
        mean_active_per_unit=float(X.nnz) / X.shape[0],
        reconstruction_error=0.0,
    )


def a_space(k: int, kind=ConceptDictionary) -> SweepSpace:
    dictionary = a_dictionary(k, kind=kind)
    return SweepSpace(
        k=k,
        dictionary=dictionary,
        matrices={view: a_matrix(dictionary, view) for view in config.CONCEPT_VIEWS},
    )


def the_spaces() -> list[SweepSpace]:
    return [a_space(SMALL_K), a_space(LARGE_K)]


def a_question(qid: str, vector: str, gold: tuple[str, ...], split: str = DEV_SPLIT) -> Question:
    return Question(
        qid=qid,
        question=vector,
        answer="-",
        gold_unit_ids=gold,
        supporting_facts=(("focused", 0),),
        split=split,
    )


def the_questions(split: str = DEV_SPLIT) -> list[Question]:
    return [
        a_question("q1", "1,0,0,0", ("focused", "broad"), split),
        a_question("q2", "0,1,0,0", ("broad",), split),
        a_question("q3", "0,0,0,1", ("absent",), split),
    ]


def a_config(**overrides) -> dict:
    base = {
        "model": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c",
        "unit_set_hash": "101f564fdcca620c",
        "seed": 42,
        "tokenizer": "cl100k_base",
        "code_version": "0.1.0",
        "top_k": 3,
    }
    base.update(overrides)
    return base


def sweep(spaces=None, questions=None, **kwargs) -> dict[int, PerKEntry]:
    return run_dev_sweep(
        spaces if spaces is not None else the_spaces(),
        questions=questions if questions is not None else the_questions(),
        token_counts=TOKEN_COUNTS,
        backend=DirectionBackend(),
        base_config=a_config(),
        **kwargs,
    )


# --- The grid it covers -------------------------------------------------------


def test_the_sweep_covers_every_arm_of_every_view_of_every_k():
    """Decision D5's grid, entire: K x query operator x view, nothing pruned."""
    table = sweep()

    assert sorted(table) == [SMALL_K, LARGE_K]
    for k, entry in table.items():
        measured = {(cell.query_operator, cell.view) for cell in entry.cells}
        assert measured == set(product(config.QUERY_OPERATORS, config.CONCEPT_VIEWS))
        assert len(entry.cells) == len(config.QUERY_OPERATORS) * len(config.CONCEPT_VIEWS)
        assert entry.k == k


def test_the_arm_is_recorded_not_the_primitive_operator():
    """`projection_top16` and `projection_full` are two alternatives, not one.

    D4 makes truncation part of the operator, so a cell that recorded only
    `projection` would collapse two of the three arms into one unreadable row.
    """
    table = sweep()

    arms = {cell.query_operator for cell in table[LARGE_K].cells}
    assert arms == set(config.QUERY_OPERATORS)


def test_the_table_is_keyed_by_the_k_that_was_measured():
    table = sweep()

    for k, entry in table.items():
        assert {cell.k for cell in entry.cells} == {k}


# --- Every cell names the space that produced it ------------------------------


def test_every_cell_carries_the_dictionary_key_that_produced_it():
    spaces = the_spaces()
    keys = {space.k: space.dictionary.key for space in spaces}
    assert keys[SMALL_K] != keys[LARGE_K]

    table = sweep(spaces=spaces)

    for k, entry in table.items():
        assert entry.dictionary_key == keys[k]
        assert {cell.dictionary_key for cell in entry.cells} == {keys[k]}


def test_a_matrix_coded_against_another_dictionary_cannot_enter_a_space():
    other = a_dictionary(LARGE_K)
    matrices = {view: a_matrix(other, view) for view in config.CONCEPT_VIEWS}

    with pytest.raises(SelectionError, match="dictionary"):
        SweepSpace(k=SMALL_K, dictionary=a_dictionary(SMALL_K), matrices=matrices)


def test_a_matrix_filed_under_another_view_cannot_enter_a_space():
    dictionary = a_dictionary(LARGE_K)
    matrices = {"raw": a_matrix(dictionary, "row_normalized")}

    with pytest.raises(SelectionError, match="row_normalized"):
        SweepSpace(k=LARGE_K, dictionary=dictionary, matrices=matrices)


def test_a_space_whose_k_disagrees_with_its_dictionary_is_refused():
    dictionary = a_dictionary(LARGE_K)

    with pytest.raises(SelectionError, match="K=8"):
        SweepSpace(
            k=8,
            dictionary=dictionary,
            matrices={view: a_matrix(dictionary, view) for view in config.CONCEPT_VIEWS},
        )


def test_a_space_missing_a_view_the_sweep_asks_for_is_refused():
    dictionary = a_dictionary(LARGE_K)
    space = SweepSpace(
        k=LARGE_K, dictionary=dictionary, matrices={"raw": a_matrix(dictionary, "raw")}
    )

    with pytest.raises(SelectionError, match="row_normalized"):
        sweep(spaces=[space])


def test_two_spaces_at_the_same_k_are_refused():
    with pytest.raises(SelectionError, match="twice"):
        sweep(spaces=[a_space(LARGE_K), a_space(LARGE_K)])


# --- Dev, and only dev --------------------------------------------------------


def test_a_test_split_question_is_refused_by_the_sweep():
    """The guard HU-7 rests on: the selection cannot see the split it is judged on."""
    with pytest.raises(SelectionError, match="test"):
        sweep(questions=the_questions(split="test"))


def test_one_test_question_among_the_dev_ones_is_enough_to_refuse():
    contaminated = [*the_questions(), a_question("q9", "1,0,0,0", ("focused",), split="test")]

    with pytest.raises(SelectionError, match="test"):
        sweep(questions=contaminated)


def test_a_train_split_question_is_refused_too():
    with pytest.raises(SelectionError, match="train"):
        sweep(questions=the_questions(split="train"))


def test_the_sweep_needs_questions_at_all():
    with pytest.raises(SelectionError, match="no question"):
        sweep(questions=[])


def test_the_cells_declare_the_split_they_were_measured_on():
    """`SweepCell.from_result` is the door, and the sweep goes through it."""
    table = sweep()

    for entry in table.values():
        for cell in entry.cells:
            assert isinstance(cell, SweepCell)
    assert "split" not in {f.name for f in fields(SweepCell)}


# --- Measured by the unchanged harness ----------------------------------------


def test_the_sweep_measures_through_the_harness_at_every_budget(monkeypatch):
    """One call to `evaluate_retriever` per cell, each asking for all four budgets.

    The harness is what Phase 1 measured the baselines with. If this phase measured
    the conceptual system any other way, System B's numbers and the dense ones would
    not be two readings of the same instrument.
    """
    real = selection.evaluate_retriever
    calls = []

    def spy(retriever, **kwargs):
        calls.append((retriever, kwargs))
        return real(retriever, **kwargs)

    monkeypatch.setattr(selection, "evaluate_retriever", spy)
    sweep()

    assert len(calls) == 2 * len(config.QUERY_OPERATORS) * len(config.CONCEPT_VIEWS)
    for retriever, kwargs in calls:
        assert retriever.name == "conceptual"
        assert tuple(kwargs["budgets"]) == config.CONTEXT_BUDGETS
        assert kwargs["split"] == DEV_SPLIT
        assert kwargs["top_k"] == 3


def test_the_four_budget_table_is_kept_whole():
    """Not reduced to the primary budget: the other three are what qualify it."""
    expected = sorted(
        [f"budget_{budget}" for budget in config.CONTEXT_BUDGETS]
        + [f"recall_at_{k}" for k in config.RECALL_AT_K]
    )
    table = sweep()

    for entry in table.values():
        for cell in entry.cells:
            assert sorted(cell.metrics) == expected
            for budget in config.CONTEXT_BUDGETS:
                row = cell.metrics[f"budget_{budget}"]
                assert sorted(row) == ["full_support", "gold_recall", "precision"]


def test_the_primary_is_gold_recall_at_the_selection_budget():
    table = sweep()

    for entry in table.values():
        best = max(
            cell.primary(config.SELECTION_METRIC, config.SELECTION_BUDGET) for cell in entry.cells
        )
        assert entry.primary == pytest.approx(best)


def test_the_winning_arm_of_a_row_is_one_of_its_cells():
    table = sweep()

    for entry in table.values():
        assert entry.best_arm in [cell.arm for cell in entry.cells]


def test_the_top_k_comes_from_the_configuration_that_is_recorded():
    """A recorded `top_k` that the sweep did not use would be a false provenance."""
    with pytest.raises(SelectionError, match="top_k"):
        run_dev_sweep(
            the_spaces(),
            questions=the_questions(),
            token_counts=TOKEN_COUNTS,
            backend=DirectionBackend(),
            base_config={key: value for key, value in a_config().items() if key != "top_k"},
        )


def test_a_configuration_that_could_not_reproduce_the_run_is_refused():
    with pytest.raises(SelectionError, match="seed"):
        run_dev_sweep(
            the_spaces(),
            questions=the_questions(),
            token_counts=TOKEN_COUNTS,
            backend=DirectionBackend(),
            base_config={key: value for key, value in a_config().items() if key != "seed"},
        )


# --- Reproducible -------------------------------------------------------------


def test_rerunning_the_sweep_reproduces_the_table_exactly():
    """Cell for cell, metric for metric. A drifting sweep cannot justify a freeze."""
    first = sweep()
    second = sweep()

    assert sorted(first) == sorted(second)
    for k in first:
        assert first[k].cells == second[k].cells
        assert first[k].best_arm == second[k].best_arm
        assert first[k].primary == second[k].primary


def test_the_order_the_spaces_arrive_in_does_not_change_the_table():
    ascending = sweep(spaces=[a_space(SMALL_K), a_space(LARGE_K)])
    descending = sweep(spaces=[a_space(LARGE_K), a_space(SMALL_K)])

    for k in ascending:
        assert ascending[k].cells == descending[k].cells


# --- The query vector is computed once per (K, arm) ---------------------------


def test_the_two_views_of_one_k_share_the_query_coding():
    """The coder runs once per question per K, not once per question per view.

    Sparse coding is the expensive arm: a lasso solve per question. The two views
    differ in `X` alone, so the concept vector of a question is the same for both,
    and computing it twice would double the cost of the arm for nothing.
    """
    CountingDictionary.calls = 0
    space = a_space(LARGE_K, kind=CountingDictionary)
    questions = the_questions()

    sweep_space(
        space,
        questions=questions,
        token_counts=TOKEN_COUNTS,
        backend=DirectionBackend(),
        base_config=a_config(),
        arms=("sparse_coding",),
    )

    assert CountingDictionary.calls == len(questions)


def test_the_shared_coding_returns_what_the_dictionary_itself_returns():
    """The memo is an optimisation, so it must not be able to change a number."""
    plain = sweep(spaces=[a_space(LARGE_K)])
    counted = sweep(spaces=[a_space(LARGE_K, kind=CountingDictionary)])

    assert plain[LARGE_K].cells == counted[LARGE_K].cells


# --- One function, both decisions ---------------------------------------------


def test_damping_is_measured_by_the_same_function_on_the_chosen_space():
    """D8's second decision reuses D8's first code path, with the weights supplied."""
    space = a_space(LARGE_K)
    weights = np.full(LARGE_K, 2.0, dtype=np.float32)

    undamped = sweep_space(
        space,
        questions=the_questions(),
        token_counts=TOKEN_COUNTS,
        backend=DirectionBackend(),
        base_config=a_config(),
        arms=("projection_top16",),
        views=("raw",),
    )
    damped = sweep_space(
        space,
        questions=the_questions(),
        token_counts=TOKEN_COUNTS,
        backend=DirectionBackend(),
        base_config=a_config(),
        arms=("projection_top16",),
        views=("raw",),
        damping="idf",
        concept_weights=weights,
    )

    assert [cell.damping for cell in undamped] == ["none"]
    assert [cell.damping for cell in damped] == ["idf"]
    # A positive rescaling of every concept cannot reorder a ranking, and the two
    # readings agreeing is what says the damping arm changed nothing but the label.
    assert undamped[0].metrics == damped[0].metrics


def test_an_unknown_arm_is_refused_before_anything_is_measured():
    with pytest.raises(ValueError, match="not_an_arm"):
        sweep_space(
            a_space(LARGE_K),
            questions=the_questions(),
            token_counts=TOKEN_COUNTS,
            backend=DirectionBackend(),
            base_config=a_config(),
            arms=("not_an_arm",),
        )


def test_an_unknown_view_is_refused_before_anything_is_measured():
    with pytest.raises(SelectionError, match="sideways"):
        sweep_space(
            a_space(LARGE_K),
            questions=the_questions(),
            token_counts=TOKEN_COUNTS,
            backend=DirectionBackend(),
            base_config=a_config(),
            views=("sideways",),
        )


# --- The structural columns Phase 2 asked for ---------------------------------

# `2.results.md`, by K: the hub concept, the units it activates, the concepts under
# 30 units, the hub's share of the pool and its share of row-normalized mass.
#
# The mass figure at K = 1,024 is 0.4% here and 0.7% in the Phase 2 table. Both are
# right about different concepts: that table's last column is the largest mass share
# held by any atom, and at K = 1,024 alone that atom (`c587`) is not the one with the
# most units (`c28`). HU-3 asks for the hub's own share -- "units activated by the
# largest concept ... and *its* share of row-normalized mass" -- so the column here
# follows the hub, which is what makes the pair readable: 66.7% of units against 5.2%
# of mass says something about one concept, not about two.
PHASE_2_STRUCTURE = {
    512: (450, 1_403, 0, 0.072, 0.008),
    1024: (28, 1_356, 40, 0.070, 0.004),
    2048: (0, 12_914, 261, 0.667, 0.052),
    4096: (0, 19_253, 2_610, 0.994, 0.235),
}
PHASE_2_POOL = 19_366


def a_structured_space(k: int, rows) -> SweepSpace:
    """A space whose structure is written out by hand, both views derived as Phase 2 does."""
    dictionary = a_dictionary(k)
    dense = np.asarray(rows, dtype=np.float32)
    raw = ConceptMatrix(
        X=sparse.csr_matrix(dense),
        unit_ids=[f"u{index}" for index in range(dense.shape[0])],
        dictionary_key=dictionary.key,
        view="raw",
        coding_alpha=0.07,
        mean_active_per_unit=float(np.count_nonzero(dense)) / dense.shape[0],
        reconstruction_error=0.0,
    )
    return SweepSpace(
        k=k,
        dictionary=dictionary,
        matrices={"raw": raw, "row_normalized": row_normalized(raw)},
    )


def a_space_supporting(k: int, n_units: int, supports: dict[int, int]) -> SweepSpace:
    """`supports[c]` units activate concept `c` with weight 1; every other cell is zero."""
    rows = np.zeros((n_units, k), dtype=np.float32)
    for concept, units in supports.items():
        rows[:units, concept] = 1.0
    return a_structured_space(k, rows)


def test_the_hub_is_the_concept_that_activates_the_most_units():
    space = a_space_supporting(SMALL_K, n_units=40, supports={0: 12, 5: 30, 9: 1})

    structure = structure_of(space)

    assert structure.hub_concept == 5
    assert structure.hub_share == pytest.approx(30 / 40)


def test_a_concept_is_thin_below_the_declared_threshold_and_not_at_it():
    """`THIN_CONCEPT_MAX_UNITS` is 30 and the count is `< 30`, so 30 units is not thin."""
    space = a_space_supporting(SMALL_K, n_units=40, supports={0: 40, 1: 30, 2: 29})

    # Concept 1 sits exactly at the threshold and is not counted; concept 2 is one unit
    # below it and is, along with the concepts no unit activates at all.
    assert structure_of(space).thin_concepts == SMALL_K - 2


def test_the_hub_mass_share_is_read_on_the_row_normalized_view_not_on_raw():
    """D5's column gives each unit one unit of attention first: raw weights are not comparable."""
    rows = np.zeros((2, SMALL_K), dtype=np.float32)
    rows[0, 0], rows[0, 1] = 3.0, 1.0
    rows[1, 0], rows[1, 2] = 1.0, 9.0
    space = a_structured_space(SMALL_K, rows)

    structure = structure_of(space)

    assert structure.hub_concept == 0
    # 3/4 of the first unit's attention and 1/10 of the second's, over two units.
    assert structure.hub_mass_share == pytest.approx((0.75 + 0.1) / 2)
    # On raw weights the same concept would read as 4/14, which is a different claim.
    assert structure.hub_mass_share != pytest.approx(4 / 14)


def test_the_hub_share_is_the_figure_the_phase_2_diagnostics_artifact_records():
    """`units_per_concept.max / n_units` is the one of the three the artifact carries."""
    space = a_space_supporting(SMALL_K, n_units=40, supports={0: 12, 5: 30, 9: 1})

    recorded = compute_diagnostics(space.matrix_for("raw"))

    assert structure_of(space).hub_share == pytest.approx(
        recorded.units_per_concept["max"] / recorded.n_units
    )


def test_the_structural_columns_reproduce_the_phase_2_results_table():
    """The four induced spaces, read off disk, against the table `2.results.md` prints.

    Nothing here is re-embedded, re-induced or recoded: the matrices are the artifacts
    Phase 2 wrote, loaded hash-verified, and the three figures are read off them. What
    the test pins is the agreement, in both directions - a change to this derivation
    that moved a number, or a re-induction that moved the space, fails here rather than
    reaching a report as a footnote.
    """
    directory = config.CONCEPTS_DIR
    recorded = sorted(directory.glob("diagnostics-*.json")) if directory.exists() else []
    if len(recorded) != len(PHASE_2_STRUCTURE):
        pytest.skip("the Phase 2 spaces are not on this machine; `cer induce` rebuilds them")

    measured = {}
    for path in recorded:
        key = path.stem.split("-", 1)[1]
        diagnostics = load_diagnostics(key, directory)
        space = SweepSpace(
            k=diagnostics.n_concepts,
            dictionary=load_dictionary(key, directory),
            matrices={
                view: load_matrix(key, directory, view=view) for view in config.CONCEPT_VIEWS
            },
        )
        structure = structure_of(space)
        assert space.matrix_for("raw").X.shape[0] == PHASE_2_POOL
        measured[diagnostics.n_concepts] = (
            structure.hub_concept,
            round(structure.hub_share * PHASE_2_POOL),
            structure.thin_concepts,
            round(structure.hub_share, 3),
            round(structure.hub_mass_share, 3),
        )

    assert measured == PHASE_2_STRUCTURE


def test_the_structural_columns_reach_the_table_the_sweep_returns():
    table = sweep()

    for k, entry in table.items():
        structure = structure_of(a_space(k))
        assert entry.hub_share == pytest.approx(structure.hub_share)
        assert entry.hub_mass_share == pytest.approx(structure.hub_mass_share)
        assert entry.thin_concepts == structure.thin_concepts


def test_a_space_that_cannot_report_a_mass_share_is_refused_rather_than_guessed():
    dictionary = a_dictionary(LARGE_K)
    space = SweepSpace(
        k=LARGE_K, dictionary=dictionary, matrices={"raw": a_matrix(dictionary, "raw")}
    )

    with pytest.raises(SelectionError, match="row_normalized"):
        structure_of(space)


# --- The structural tie-break -------------------------------------------------


def a_row(k: int, primary: float, n_questions: int = 600, **structure) -> PerKEntry:
    """One row of the table with its figure set by hand, so the margins are checkable.

    The cell is built directly rather than through `SweepCell.from_result`: what these
    tests are about is which row wins, not where a cell's provenance comes from, and a
    hand-written figure is what makes a margin of exactly 0.01 readable as such.
    """
    cell = SweepCell(
        k=k,
        dictionary_key=f"key{k}",
        view="raw",
        query_operator="projection_top16",
        damping="none",
        n_questions=n_questions,
        metrics={f"budget_{config.SELECTION_BUDGET}": {config.SELECTION_METRIC: primary}},
    )
    return PerKEntry.from_cells([cell], **structure)


# A hub share, a mass share and a thin count per K, in the shape Phase 2 measured them.
SMALL_HUB = {"hub_share": 0.070, "hub_mass_share": 0.004, "thin_concepts": 40}
LARGE_HUB = {"hub_share": 0.667, "hub_mass_share": 0.052, "thin_concepts": 261}


def test_the_resolution_is_one_standard_error_of_the_figure_over_the_questions_measured():
    """Not a declared constant: the dev split's resolving power is a property of the split."""
    assert resolvable_margin(0.60, 600) == pytest.approx(0.02)
    assert resolvable_margin(0.60, 60_000) == pytest.approx(0.002)


def test_two_spaces_further_apart_than_the_dev_split_resolves_are_decided_by_recall():
    """0.05 over 600 questions is outside the noise, so the larger hub wins on its recall."""
    choice = choose_space(
        {2048: a_row(2048, 0.60, **LARGE_HUB), 1024: a_row(1024, 0.55, **SMALL_HUB)}
    )

    assert choice.k == 2048
    assert choice.tie_break is None


def test_a_margin_inside_what_the_dev_split_resolves_is_broken_by_the_smaller_hub():
    """0.01 over 600 questions is inside it, and then the declared structural criterion decides."""
    choice = choose_space(
        {2048: a_row(2048, 0.60, **LARGE_HUB), 1024: a_row(1024, 0.59, **SMALL_HUB)}
    )

    assert choice.k == 1024
    assert choice.tie_break is not None


def test_the_recorded_tie_break_names_the_criterion_and_both_hub_shares():
    choice = choose_space(
        {2048: a_row(2048, 0.60, **LARGE_HUB), 1024: a_row(1024, 0.59, **SMALL_HUB)}
    )

    assert choice.tie_break is not None
    assert "hub" in choice.tie_break
    assert "7.0%" in choice.tie_break and "66.7%" in choice.tie_break


def test_the_tie_break_is_stated_even_when_structure_confirms_the_leading_recall():
    """The spec asks for the tie to be stated, not only for the winner to be swapped."""
    choice = choose_space(
        {2048: a_row(2048, 0.59, **LARGE_HUB), 1024: a_row(1024, 0.60, **SMALL_HUB)}
    )

    assert choice.k == 1024
    assert choice.tie_break is not None


def test_the_runner_up_and_its_margin_are_recorded_when_the_tie_break_fired():
    choice = choose_space(
        {2048: a_row(2048, 0.60, **LARGE_HUB), 1024: a_row(1024, 0.59, **SMALL_HUB)}
    )

    assert choice.runner_up == "k=2048/projection_top16/raw"
    assert choice.margin == pytest.approx(0.01)


def test_the_runner_up_and_its_margin_are_recorded_when_it_did_not():
    choice = choose_space(
        {2048: a_row(2048, 0.60, **LARGE_HUB), 1024: a_row(1024, 0.55, **SMALL_HUB)}
    )

    assert choice.runner_up == "k=1024/projection_top16/raw"
    assert choice.margin == pytest.approx(0.05)


def test_a_space_the_dev_split_can_tell_apart_is_not_rescued_by_its_smaller_hub():
    """The criterion breaks a tie; it does not overturn a difference the split can resolve."""
    choice = choose_space(
        {
            2048: a_row(2048, 0.60, **LARGE_HUB),
            1024: a_row(1024, 0.59, **SMALL_HUB),
            512: a_row(512, 0.40, hub_share=0.001, hub_mass_share=0.001, thin_concepts=0),
        }
    )

    assert choice.k == 1024


def test_a_tie_whose_structural_columns_were_never_read_is_refused_rather_than_guessed():
    with pytest.raises(SelectionError, match="hub"):
        choose_space({2048: a_row(2048, 0.60), 1024: a_row(1024, 0.59)})


def test_two_spaces_with_the_same_hub_are_separated_by_the_thinner_one():
    choice = choose_space(
        {
            2048: a_row(2048, 0.60, hub_share=0.070, hub_mass_share=0.004, thin_concepts=261),
            1024: a_row(1024, 0.59, hub_share=0.070, hub_mass_share=0.004, thin_concepts=40),
        }
    )

    assert choice.k == 1024


def test_a_single_space_is_chosen_with_no_runner_up_and_a_margin_of_zero():
    choice = choose_space({2048: a_row(2048, 0.60, **LARGE_HUB)})

    assert (choice.k, choice.runner_up, choice.margin, choice.tie_break) == (2048, None, 0.0, None)


def test_the_choice_is_a_ledger_entry_naming_every_space_it_was_decided_over():
    """D8's first decision, in the shape the ledger HU-7 counts stores it."""
    decision = choose_space(
        {2048: a_row(2048, 0.60, **LARGE_HUB), 1024: a_row(1024, 0.59, **SMALL_HUB)}
    ).as_decision()

    assert decision.name == "space"
    assert decision.chosen == "k=1024/projection_top16/raw"
    assert decision.alternatives == [
        "k=1024/projection_top16/raw",
        "k=2048/projection_top16/raw",
    ]
    assert decision.runner_up == "k=2048/projection_top16/raw"
    assert decision.margin == pytest.approx(0.01)


def test_rows_measured_over_different_numbers_of_questions_cannot_be_compared():
    with pytest.raises(SelectionError, match="questions"):
        choose_space(
            {
                2048: a_row(2048, 0.60, n_questions=600, **LARGE_HUB),
                1024: a_row(1024, 0.59, n_questions=300, **SMALL_HUB),
            }
        )
