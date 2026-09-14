"""T14: the ledger of what dev decided, and the moment the configuration closed.

This phase makes more dev-fitted choices than any other in the project: the space
(K, query operator, view), the damping, the fusion scheme and its weight. Each is
defensible alone; together they are a multiple-comparisons problem on 600 questions,
and HU-7's answer is not to make fewer choices but to write down how many there were
and what each one cost.

So the ledger is the deliverable, and three properties make it one. It reads in the
order decision D8 fixed **before** any of it was measured, because an order that can
be rearranged afterwards is an order that will be rearranged to flatter a result.
Every entry carries the alternatives it beat and the margin it won by, so a choice
taken by a thousandth is visible as such. And the count is reported rather than left
to whoever is reading to tally up.

The freeze is the other half. `frozen_at` is stamped once, it is what a test result
has to postdate, and the artifact refuses to be written twice - so "the configuration
was frozen before test was read" is a thing the files can be checked for rather than
a thing the author remembers doing.

Nothing here measures anything: every figure is written by hand, which is what makes
a margin of exactly 0.05 readable as one.
"""

from datetime import UTC, datetime, timedelta

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.harness import RunResult
from concept_embeddings_rag.evaluation.selection import (
    CONTROL_SIGNAL,
    DECISION_ORDER,
    SYSTEM_B_SIGNAL,
    Decision,
    FusionFit,
    PerKEntry,
    SelectionError,
    SelectionReport,
    SpaceChoice,
    SweepCell,
    build_ledger,
    check_freeze_precedes,
    choose_damping,
    freeze_selection,
    fusion_decisions,
    load_selection,
    save_selection,
)
from concept_embeddings_rag.retrieval.fusion import RRF, WEIGHTED

KEYS = {1024: "key1024", 2048: "key2048"}
GRID = config.FUSION_WEIGHT_GRID


def a_cell(
    *,
    k: int = 2048,
    view: str = "raw",
    arm: str = "projection_top16",
    damping: str = "none",
    primary: float = 0.62,
    n_questions: int = 600,
) -> SweepCell:
    return SweepCell(
        k=k,
        dictionary_key=KEYS[k],
        view=view,
        query_operator=arm,
        damping=damping,
        n_questions=n_questions,
        metrics={f"budget_{config.SELECTION_BUDGET}": {config.SELECTION_METRIC: primary}},
    )


def a_row(k: int = 2048, primary: float = 0.62) -> PerKEntry:
    return PerKEntry.from_cells([a_cell(k=k, primary=primary)])


def a_table() -> dict[int, PerKEntry]:
    return {2048: a_row(2048, 0.62), 1024: a_row(1024, 0.59)}


def a_space_choice(tie_break: str | None = None) -> SpaceChoice:
    return SpaceChoice(
        k=2048,
        dictionary_key=KEYS[2048],
        best_arm={"query_operator": "projection_top16", "view": "raw", "damping": "none"},
        primary=0.62,
        label="k=2048/projection_top16/raw",
        runner_up="k=1024/projection_top16/raw",
        margin=0.03,
        alternatives=["k=1024/projection_top16/raw", "k=2048/projection_top16/raw"],
        tie_break=tie_break,
    )


def a_damping_decision(damped_primary: float = 0.60) -> Decision:
    """The undamped arm reaches 0.62; what the damped one reaches is the argument."""
    return choose_damping(a_cell(damping="none"), a_cell(damping="idf", primary=damped_primary))


def a_curve(best: float = 0.5, top: float = 0.70, floor: float = 0.60) -> dict[float, float]:
    """One point above a flat floor, so the optimum and the runner-up are both known."""
    return {weight: (top if weight == best else floor) for weight in GRID}


def a_fit(
    *,
    second_signal: str = SYSTEM_B_SIGNAL,
    curve: dict[float, float] | None = None,
    rrf_score: float = 0.65,
) -> FusionFit:
    return FusionFit(
        components=("dense", second_signal),
        metric=config.SELECTION_METRIC,
        budget=config.SELECTION_BUDGET,
        n_questions=600,
        grid=GRID,
        curve=a_curve() if curve is None else curve,
        rrf_score=rrf_score,
    )


def an_rrf_fit(**overrides) -> FusionFit:
    """The same curve against a reference the weighted scheme does not beat."""
    return a_fit(rrf_score=0.75, **overrides)


def a_config(**overrides) -> dict:
    frozen = {
        "model": "fake-model",
        "revision": "v0",
        "unit_set_hash": "poolhash0001",
        "seed": 42,
        "tokenizer": "fake-tokenizer",
        "code_version": "0.1.0",
        "top_k": 100,
    }
    frozen.update(overrides)
    return frozen


def a_freeze(**overrides) -> SelectionReport:
    arguments = {
        "space": a_space_choice(),
        "damping": a_damping_decision(),
        "fusion": a_fit(),
        "base_config": a_config(),
    }
    arguments.update(overrides)
    return freeze_selection(a_table(), **arguments)


def names_of(ledger: list[Decision]) -> list[str]:
    return [decision.name for decision in ledger]


# --- The ledger, in the order D8 fixed ----------------------------------------


def test_the_ledger_reads_in_the_order_d8_declared():
    ledger = build_ledger(space=a_space_choice(), damping=a_damping_decision(), fusion=a_fit())

    assert names_of(ledger) == list(DECISION_ORDER)


def test_the_ledger_stops_at_three_when_no_weight_was_ever_chosen():
    """D8's fourth entry is conditional, and its absence is the record of that.

    The eleven points were still measured; what was not spent is the degree of
    freedom, because RRF took the scheme decision and no weight was selected.
    """
    ledger = build_ledger(space=a_space_choice(), damping=a_damping_decision(), fusion=an_rrf_fit())

    assert names_of(ledger) == list(DECISION_ORDER[:3])


def test_every_entry_names_the_alternatives_it_beat_and_the_margin_it_won_by():
    for decision in build_ledger(
        space=a_space_choice(), damping=a_damping_decision(), fusion=a_fit()
    ):
        assert len(decision.alternatives) >= 2
        assert decision.chosen in decision.alternatives
        assert decision.margin >= 0.0


def test_the_arm_counts_are_visible_inside_the_entries():
    """HU-7 counts the decisions; the alternatives inside them say how wide each was."""
    ledger = build_ledger(space=a_space_choice(), damping=a_damping_decision(), fusion=a_fit())
    widths = {decision.name: len(decision.alternatives) for decision in ledger}

    assert widths == {
        "space": 2,
        "damping": len(config.DAMPING_MODES),
        "fusion_scheme": len(config.FUSION_SCHEMES),
        "fusion_weight": len(config.FUSION_WEIGHT_GRID),
    }


def test_a_ledger_whose_second_decision_is_not_the_damping_is_refused():
    with pytest.raises(SelectionError, match="damping"):
        build_ledger(
            space=a_space_choice(),
            damping=Decision(
                name="view", chosen="raw", alternatives=["raw"], runner_up=None, margin=0.0
            ),
            fusion=a_fit(),
        )


def test_a_reordered_ledger_is_refused_rather_than_accepted_and_sorted():
    """Reordering D8 is a deviation to be written down, not an edit to be made."""
    ledger = build_ledger(space=a_space_choice(), damping=a_damping_decision(), fusion=a_fit())

    with pytest.raises(SelectionError, match="deviation"):
        SelectionReport(
            selected_k=2048,
            selected_dictionary_key=KEYS[2048],
            selection_metric=config.SELECTION_METRIC,
            selection_budget=config.SELECTION_BUDGET,
            per_k=a_table(),
            decisions=[ledger[1], ledger[0], *ledger[2:]],
            n_dev_decisions=4,
            tie_break=None,
            config=a_config(),
            seed=42,
            code_version="0.1.0",
            fusion=a_fit(),
        )


# --- D8's second decision: the damping ----------------------------------------


def test_the_damping_takes_its_place_when_it_helps():
    decision = a_damping_decision(damped_primary=0.65)

    assert decision.chosen == "idf"
    assert decision.runner_up == "none"
    assert decision.margin == pytest.approx(0.03)


def test_a_damping_that_only_matches_the_undamped_system_does_not_take_its_place():
    """A declared variant has to show it helps; matching is not showing it."""
    decision = a_damping_decision(damped_primary=0.62)

    assert decision.chosen == "none"
    assert decision.margin == pytest.approx(0.0)


def test_the_alternatives_are_the_two_declared_modes_in_the_declared_order():
    assert a_damping_decision().alternatives == list(config.DAMPING_MODES)


def test_two_readings_that_differ_in_more_than_the_damping_are_refused():
    """Otherwise the decision would be taken between two things, not about one."""
    with pytest.raises(SelectionError, match="view"):
        choose_damping(
            a_cell(damping="none", view="raw"),
            a_cell(damping="idf", view="row_normalized"),
        )


def test_a_damped_reading_from_another_space_is_refused():
    with pytest.raises(SelectionError, match="k"):
        choose_damping(a_cell(k=2048, damping="none"), a_cell(k=1024, damping="idf"))


def test_two_undamped_readings_are_not_a_decision():
    with pytest.raises(SelectionError, match="undamped"):
        choose_damping(a_cell(damping="none"), a_cell(damping="none"))


def test_the_reference_reading_has_to_be_the_undamped_one():
    with pytest.raises(SelectionError, match="undamped"):
        choose_damping(a_cell(damping="idf"), a_cell(damping="idf"))


def test_readings_over_different_numbers_of_questions_cannot_be_compared():
    with pytest.raises(SelectionError, match="questions"):
        choose_damping(
            a_cell(damping="none", n_questions=600),
            a_cell(damping="idf", n_questions=300),
        )


# --- D8's third and fourth decisions: the fusion ------------------------------


def test_the_scheme_decision_names_both_schemes_and_the_gap_between_them():
    scheme = fusion_decisions(a_fit())[0]

    assert scheme.chosen == WEIGHTED
    assert scheme.runner_up == RRF
    assert scheme.alternatives == list(config.FUSION_SCHEMES)
    assert scheme.margin == pytest.approx(0.05)


def test_the_scheme_decision_records_rrf_winning_as_readily_as_losing():
    scheme = fusion_decisions(an_rrf_fit())[0]

    assert scheme.chosen == RRF
    assert scheme.runner_up == WEIGHTED
    assert scheme.margin == pytest.approx(0.05)


def test_the_weight_decision_names_every_point_of_the_declared_grid():
    weight = fusion_decisions(a_fit())[1]

    assert weight.alternatives == [f"w={point:.1f}" for point in GRID]
    assert weight.chosen == "w=0.5"


def test_the_weight_margin_is_the_gap_to_the_second_best_point():
    """A flat curve and a sharp one say different things about surviving the test split."""
    weight = fusion_decisions(a_fit())[1]

    assert weight.runner_up == "w=1.0"
    assert weight.margin == pytest.approx(0.10)


def test_there_is_no_weight_decision_when_the_weighted_scheme_lost():
    assert len(fusion_decisions(an_rrf_fit())) == 1


# --- The freeze ---------------------------------------------------------------


def test_the_frozen_configuration_carries_the_six_keys_that_say_what_was_measured():
    frozen = a_freeze().config

    assert frozen["dictionary_key"] == KEYS[2048]
    assert frozen["k"] == 2048
    assert frozen["view"] == "raw"
    assert frozen["query_operator"] == "projection_top16"
    assert frozen["damping"] == "none"
    assert frozen["fusion_scheme"] == WEIGHTED


def test_the_frozen_weights_are_the_ones_the_curve_selected():
    assert a_freeze().config["fusion_weights"] == {"dense": 0.5, SYSTEM_B_SIGNAL: 0.5}


def test_a_parameter_free_fusion_freezes_no_weights_rather_than_inventing_one():
    assert a_freeze(fusion=an_rrf_fit()).config["fusion_weights"] is None


def test_the_damping_that_won_is_the_one_frozen():
    frozen = a_freeze(damping=a_damping_decision(damped_primary=0.65)).config

    assert frozen["damping"] == "idf"


def test_the_count_of_dev_decisions_is_reported_rather_than_left_to_be_counted():
    report = a_freeze()

    assert report.n_dev_decisions == 4
    assert report.n_dev_decisions == len(report.decisions)
    assert a_freeze(fusion=an_rrf_fit()).n_dev_decisions == 3


def test_the_seed_and_the_code_version_come_out_of_the_configuration():
    """Passed twice is passed to disagree, so they are read from the config that travels."""
    report = a_freeze(base_config=a_config(seed=7, code_version="9.9.9"))

    assert (report.seed, report.code_version) == (7, "9.9.9")


def test_the_tie_break_travels_from_the_space_choice_into_the_report():
    report = a_freeze(space=a_space_choice(tie_break="smaller hub: 7.0% vs 66.7% hub share"))

    assert report.tie_break == "smaller hub: 7.0% vs 66.7% hub share"


def test_a_configuration_that_could_not_reproduce_the_run_is_refused():
    incomplete = a_config()
    del incomplete["seed"]

    with pytest.raises(SelectionError, match="seed"):
        a_freeze(base_config=incomplete)


# --- The stamp, and what has to postdate it -----------------------------------


def test_the_freeze_is_stamped_to_the_second_when_the_configuration_closes():
    stamped = a_freeze().frozen_at

    assert datetime.fromisoformat(stamped).tzinfo is not None
    assert "." not in stamped, "a freeze is recorded to the second, not to the microsecond"


def test_a_second_write_does_not_move_the_stamp(tmp_path):
    first = a_freeze()
    save_selection(first, tmp_path)

    with pytest.raises(SelectionError, match="already"):
        save_selection(a_freeze(), tmp_path)

    assert load_selection(tmp_path).frozen_at == first.frozen_at


def a_result(created_at: str, split: str = "test") -> RunResult:
    return RunResult(
        system="hybrid-conceptual",
        split=split,
        config=a_config(),
        metrics={},
        cost={},
        created_at=created_at,
    )


def test_a_result_measured_after_the_freeze_is_accepted():
    report = a_freeze()
    later = datetime.fromisoformat(report.frozen_at) + timedelta(seconds=30)

    check_freeze_precedes(report, a_result(later.isoformat(timespec="seconds")))


def test_a_result_measured_before_the_freeze_is_refused():
    """The observable form of "frozen before test was read", rather than the intended one."""
    report = a_freeze()
    earlier = datetime.fromisoformat(report.frozen_at) - timedelta(seconds=1)

    with pytest.raises(SelectionError, match="before this configuration froze"):
        check_freeze_precedes(report, a_result(earlier.isoformat(timespec="seconds")))


def test_the_stamp_is_read_as_a_moment_and_not_as_a_string():
    report = a_freeze(
        space=a_space_choice(),
    )
    frozen = datetime.fromisoformat(report.frozen_at)

    assert frozen <= datetime.now(UTC)


# --- The two curves the report carries ----------------------------------------


def test_both_fitted_curves_survive_the_round_trip(tmp_path):
    report = a_freeze(control=a_fit(second_signal=CONTROL_SIGNAL, rrf_score=0.61))

    save_selection(report, tmp_path)
    loaded = load_selection(tmp_path)

    assert loaded == report
    assert loaded.fusion is not None and loaded.control is not None
    assert loaded.fusion.curve == report.fusion.curve
    assert loaded.control.rrf_score == pytest.approx(0.61)


def test_the_whole_curve_is_recorded_and_not_only_the_point_that_won():
    """HU-4 asks for the grid, the selected value and the curve around it, not the optimum."""
    fusion = a_freeze().fusion

    assert fusion is not None
    assert sorted(fusion.curve) == sorted(GRID)
    assert fusion.best_weight == 0.5


def test_the_control_is_recorded_beside_the_ledger_and_never_inside_it():
    """Its fitted weight is a measurement of this phase, not one of System B's decisions."""
    without = a_freeze()
    with_control = a_freeze(control=a_fit(second_signal=CONTROL_SIGNAL))

    assert with_control.decisions == without.decisions
    assert with_control.n_dev_decisions == without.n_dev_decisions
    assert with_control.control is not None


def test_a_control_fit_of_the_wrong_hybrid_is_refused():
    with pytest.raises(SelectionError, match="control"):
        a_freeze(control=a_fit(second_signal=SYSTEM_B_SIGNAL))


def test_system_b_s_fit_has_to_be_the_conceptual_hybrid():
    with pytest.raises(SelectionError, match="fusion"):
        a_freeze(fusion=a_fit(second_signal=CONTROL_SIGNAL))


def test_a_fusion_decision_without_the_curve_behind_it_is_refused():
    """A ledger entry whose evidence is absent is a claim, not a measurement."""
    ledger = build_ledger(space=a_space_choice(), damping=a_damping_decision(), fusion=a_fit())

    with pytest.raises(SelectionError, match="no fusion fit"):
        SelectionReport(
            selected_k=2048,
            selected_dictionary_key=KEYS[2048],
            selection_metric=config.SELECTION_METRIC,
            selection_budget=config.SELECTION_BUDGET,
            per_k=a_table(),
            decisions=ledger,
            n_dev_decisions=len(ledger),
            tie_break=None,
            config=a_config(),
            seed=42,
            code_version="0.1.0",
        )
