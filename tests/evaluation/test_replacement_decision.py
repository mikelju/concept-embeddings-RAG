"""Phase 6, T12: the sign tests, the state procedure and `Phase6Decision` (HU-10, D16).

The state is computed by code from per-question Full Support at 2,048 on test, from the three
outcome artifacts as read back from disk and from the freeze's copy of the decision parameters,
and from nothing else: no parameter of the decision function names a metric, a budget, a margin,
an alpha, a method or a population. The toy populations below hold exactly `N_TEST` questions,
because the decision refuses any other population.
"""

import inspect
import json
from fractions import Fraction

import pytest
from outcome_fixtures import FREEZE_DIGEST, vectors_from_counts, write_outcomes
from scipy.stats import binomtest, norm

from concept_embeddings_rag import decision_parameters as dp
from concept_embeddings_rag.evaluation import tango
from concept_embeddings_rag.evaluation.replacement_decision import (
    DecisionError,
    check_parameters,
    compute_decision,
    decide,
    load_decision,
    paired_table,
    parameters_payload,
    save_decision,
    scenarios,
    sign_test_of,
)

N = dp.N_TEST
S_A, S_B = "hybrid-bm25", "hybrid-entity-hop"


def a_freeze(mode: str = "reused", **parameter_changes) -> dict:
    parameters = json.loads(json.dumps(parameters_payload()))
    parameters.update(parameter_changes)
    return {"digest": FREEZE_DIGEST, "control": {"mode": mode}, "decision_parameters": parameters}


def reproduction(mode: str = "reused", passed: bool = True) -> dict:
    return {"mode": mode, "passed": passed, "checks": []}


# (dense, A, B) -> questions. A 1,210, dense 1,155, B 1,190: the reused-mode shape.
REUSED_SHAPE = {
    (1, 1, 1): 1100,
    (1, 1, 0): 30,
    (1, 0, 1): 10,
    (1, 0, 0): 15,
    (0, 1, 1): 60,
    (0, 1, 0): 20,
    (0, 0, 1): 20,
    (0, 0, 0): 145,
}


@pytest.fixture(scope="module")
def reused_world(tmp_path_factory):
    directory = tmp_path_factory.mktemp("decision")
    assert sum(REUSED_SHAPE.values()) == N
    outcomes, qids = write_outcomes(directory, vectors_from_counts(REUSED_SHAPE))
    return directory, outcomes, qids


def counts_of(outcomes, system) -> int:
    return sum(int(entry["budget_2048"]["full_support"]) for entry in outcomes[system].questions)


# --- The parameters come from the freeze and must equal the spec's ----------------------------


def test_the_parameters_payload_is_the_decision_parameters_module(reused_world):
    payload = parameters_payload()
    assert payload["metric"] == dp.DECISION_METRIC
    assert payload["budget"] == dp.DECISION_BUDGET
    assert payload["split"] == dp.DECISION_SPLIT
    assert payload["n"] == N
    assert payload["delta"]["numerator"] == 55 and payload["delta"]["denominator"] == 2800
    assert payload["alpha"] == dp.DECISION_ALPHA
    assert len(payload["state_table"]) == 8


@pytest.mark.parametrize(
    ("field", "value"),
    [("budget", 4096), ("metric", "gold_recall"), ("alpha", 0.025), ("n", 1399)],
)
def test_a_freeze_copy_that_differs_from_the_spec_is_refused(reused_world, field, value):
    _, outcomes, _ = reused_world
    with pytest.raises(DecisionError, match="decision parameters"):
        compute_decision(a_freeze(**{field: value}), outcomes, reproduction())


def test_a_changed_open_question_text_in_the_freeze_is_refused():
    freeze = a_freeze()
    freeze["decision_parameters"]["state_table"][1]["open_question"] += " "
    with pytest.raises(DecisionError, match="decision parameters"):
        check_parameters(freeze["decision_parameters"])


def test_the_decision_takes_no_metric_budget_margin_alpha_method_or_population():
    parameters = set(inspect.signature(compute_decision).parameters)
    assert parameters == {"freeze", "outcomes", "test_reproduction"}


# --- The state table, row by row --------------------------------------------------------------


def table_copy() -> list[dict]:
    return a_freeze()["decision_parameters"]["state_table"]


@pytest.mark.parametrize("row", range(8))
def test_every_row_gives_the_tables_state_route_anomalies_and_question(row):
    table = table_copy()
    expected = table[row]
    got = decide(expected["s"], expected["v"], expected["n"], table)
    assert got == expected


def test_the_texts_come_from_the_table_copy_handed_in():
    table = table_copy()
    table[4]["open_question"] = "a text only this copy holds"
    assert decide(False, True, True, table)["open_question"] == "a text only this copy holds"


def test_row_5_non_inferior_without_a_gain_over_dense_is_partial_p2():
    row = decide(False, True, True, table_copy())
    assert (row["state"], row["route"]) == (dp.STATE_PARTIAL_INVESTIGATE, dp.ROUTE_P2)


def test_row_7_non_inferior_to_a_control_that_shows_nothing_is_stop():
    row = decide(False, False, True, table_copy())
    assert (row["state"], row["route"]) == (dp.STATE_STOP, None)
    assert row["open_question"] is None


def test_a_table_that_disagrees_with_the_ordered_procedure_is_refused():
    table = table_copy()
    table[4]["state"] = dp.STATE_STOP
    table[4]["route"] = None
    with pytest.raises(DecisionError, match="procedure"):
        decide(False, True, True, table)


# --- Sign tests -------------------------------------------------------------------------------


def test_the_sign_test_p_values_are_the_exact_one_sided_binomial(reused_world):
    _, outcomes, _ = reused_world
    decision = compute_decision(a_freeze(), outcomes, reproduction())

    for block in (decision["secondary"], decision["assay_sensitivity"]):
        wins, losses = block["wins"], block["losses"]
        expected = binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue
        assert block["p_value"] == pytest.approx(expected, rel=1e-12)
        assert block["passed"] is (block["p_value"] < dp.DECISION_ALPHA)


def test_no_discordant_question_gives_p_one_and_a_failed_test():
    same = dict.fromkeys(["q1", "q2", "q3"], True)
    test = sign_test_of(S_B, same, "dense", same, budget=2048, alpha=0.05)
    assert (test.wins, test.losses, test.ties) == (0, 0, 3)
    assert test.p_value == 1.0
    assert test.passed is False


def test_the_reused_mode_argument_every_split_with_a_55_excess_passes():
    """Spec: A-only exceeds dense-only by 55, and dense-only is at most 190: p < 0.05 always."""
    for dense_only in range(0, 191):
        control_only = dense_only + 55
        p = binomtest(control_only, control_only + dense_only, 0.5, alternative="greater").pvalue
        assert p < 0.05, dense_only


def test_ties_stay_in_n_for_n_and_leave_the_sign_test_denominators(reused_world):
    _, outcomes, _ = reused_world
    decision = compute_decision(a_freeze(), outcomes, reproduction())
    primary, secondary = decision["primary"], decision["secondary"]

    assert primary["n"] == N
    assert primary["both"] + primary["neither"] + primary["b"] + primary["c"] == N
    assert secondary["wins"] + secondary["losses"] + secondary["ties"] == N
    discordant = secondary["wins"] + secondary["losses"]
    assert discordant < N


# --- The whole decision on a reused-shaped outcome --------------------------------------------


def test_the_decision_records_every_statistic_and_the_state_of_its_row(reused_world):
    _, outcomes, _ = reused_world
    decision = compute_decision(a_freeze(), outcomes, reproduction())
    primary = decision["primary"]

    assert counts_of(outcomes, "dense") == 1155
    assert counts_of(outcomes, S_A) == 1210
    assert counts_of(outcomes, S_B) == 1190
    assert (primary["b"], primary["c"]) == (30, 50)
    assert primary["delta_hat"] == (30 - 50) / N
    assert primary["delta"] == {"fraction": "55/2800", "value": float(dp.DELTA)}
    reference = tango.non_inferiority(30, 50, N, delta=dp.DELTA, alpha=dp.DECISION_ALPHA)
    assert primary["lower_limit"] == reference.lower_limit
    assert primary["statistic"] == reference.statistic
    assert primary["passed"] is reference.passed

    row = decide(
        decision["secondary"]["passed"],
        decision["assay_sensitivity"]["passed"],
        primary["passed"],
        table_copy(),
    )
    for field in ("state", "route", "anomalies", "open_question"):
        assert decision[field] == row[field]
    assert decision["control_mode"] == "reused"
    assert decision["freeze_digest"] == FREEZE_DIGEST
    assert set(decision["outcome_digests"]) == {"dense", S_A, S_B}
    assert decision["m1_check"] == {"expected": 55, "observed": 55, "passed": True}


def test_the_sign_test_counts_are_the_discordant_pairs(reused_world):
    _, outcomes, _ = reused_world
    decision = compute_decision(a_freeze(), outcomes, reproduction())
    # B vs dense: B-only = (0,*,1) = 60 + 20 = 80; dense-only = (1,*,0) = 30 + 15 = 45.
    assert (decision["secondary"]["wins"], decision["secondary"]["losses"]) == (80, 45)
    # A vs dense: A-only = (0,1,*) = 80; dense-only = (1,0,*) = 25.
    assert (decision["assay_sensitivity"]["wins"], decision["assay_sensitivity"]["losses"]) == (
        80,
        25,
    )


def test_the_descriptive_block_is_labelled_and_computed_by_the_same_functions(reused_world):
    _, outcomes, _ = reused_world
    descriptive = compute_decision(a_freeze(), outcomes, reproduction())["descriptive"]

    assert descriptive["decides_nothing"] is True
    z90 = float(norm.ppf(0.95))
    low, high = tango.score_interval(30, 50, N, z90)
    assert descriptive["superiority"]["interval_90"] == [low, high]
    expected_position = (
        "above zero" if low > 0.0 else "below zero" if high < 0.0 else "contains zero"
    )
    assert descriptive["superiority"]["position"] == expected_position
    assert descriptive["superiority"]["mcnemar_two_sided_p"] == pytest.approx(
        binomtest(30, 80, 0.5, alternative="two-sided").pvalue
    )
    assert descriptive["lower_bound_97_5"] == tango.lower_limit(30, 50, N, float(norm.ppf(0.975)))
    assert descriptive["m1_interval_90"] == list(tango.score_interval(80, 25, N, z90))
    assert descriptive["b_vs_dense_two_sided_p"] == pytest.approx(
        binomtest(80, 125, 0.5, alternative="two-sided").pvalue
    )


def test_delta_is_the_same_in_both_modes(reused_world):
    _, outcomes, _ = reused_world
    reused = compute_decision(a_freeze("reused"), outcomes, reproduction("reused"))
    remeasured = compute_decision(
        a_freeze("re-measured"), outcomes, reproduction("re-measured", passed=False)
    )
    assert reused["primary"]["delta"] == remeasured["primary"]["delta"]
    assert reused["primary"]["n"] == remeasured["primary"]["n"] == N
    assert remeasured["control_mode"] == "re-measured"
    assert remeasured["test_reproduction"]["passed"] is False


def test_a_failed_test_reproduction_in_reused_mode_is_refused(reused_world):
    _, outcomes, _ = reused_world
    with pytest.raises(DecisionError, match="reproduction"):
        compute_decision(a_freeze(), outcomes, reproduction("reused", passed=False))


def test_a_reproduction_recorded_under_another_mode_is_refused(reused_world):
    _, outcomes, _ = reused_world
    with pytest.raises(DecisionError, match="mode"):
        compute_decision(a_freeze("reused"), outcomes, reproduction("re-measured"))


# --- Refusals on the outcomes -----------------------------------------------------------------


def test_outcomes_bound_to_another_freeze_are_refused(reused_world):
    _, outcomes, _ = reused_world
    freeze = a_freeze()
    freeze["digest"] = "e" * 64
    with pytest.raises(DecisionError, match="freeze"):
        compute_decision(freeze, outcomes, reproduction())


def test_a_missing_system_is_refused(reused_world):
    _, outcomes, _ = reused_world
    partial = {system: outcomes[system] for system in ("dense", S_B)}
    with pytest.raises(DecisionError, match="systems"):
        compute_decision(a_freeze(), partial, reproduction())


def test_outcomes_not_read_back_from_disk_are_refused(reused_world):
    _, outcomes, _ = reused_world
    forged = dict(outcomes)
    forged[S_B] = {"questions": outcomes[S_B].questions}
    with pytest.raises(DecisionError, match="disk"):
        compute_decision(a_freeze(), forged, reproduction())


def test_a_population_other_than_n_is_refused(tmp_path):
    small = vectors_from_counts({(1, 1, 1): 10})
    outcomes, _ = write_outcomes(tmp_path, small)
    with pytest.raises(DecisionError, match="population"):
        compute_decision(a_freeze(), outcomes, reproduction())


def test_dev_outcomes_are_refused(tmp_path):
    outcomes, _ = write_outcomes(
        tmp_path, vectors_from_counts({(1, 1, 1): N}), split="dev", freeze_digest=None
    )
    with pytest.raises(DecisionError, match="split"):
        compute_decision(a_freeze(), outcomes, reproduction())


# --- Scenarios --------------------------------------------------------------------------------


DELTA = dp.DELTA


def scenario(delta_hat, lower, upper, s=False, b_count=1200, dense_count=1155):
    return scenarios(
        delta_hat=Fraction(delta_hat),
        lower=lower,
        upper=upper,
        s_passed=s,
        b_count=b_count,
        dense_count=dense_count,
        delta=DELTA,
    )


def test_scenario_a_tied_but_inconclusive():
    assert "a" in scenario(Fraction(0), -0.03, 0.03)


def test_scenario_b_slightly_worse():
    assert "b" in scenario(Fraction(-10, 1400), -0.015, 0.001)


def test_scenario_c_better_than_dense_worse_than_bm25():
    assert "c" in scenario(Fraction(-40, 1400), -0.04, -0.017, s=True)


def test_scenario_d_better_but_not_conclusive():
    assert "d" in scenario(Fraction(10, 1400), -0.001, 0.02)


def test_scenario_e_worse_than_bm25_and_dense():
    assert "e" in scenario(Fraction(-70, 1400), -0.07, -0.03, b_count=1140)


def test_more_than_one_scenario_may_hold():
    assert scenario(Fraction(-10, 1400), -0.03, 0.01) == ["a", "b"]


def test_a_case_where_no_scenario_holds_gives_an_empty_list():
    assert scenario(Fraction(20, 1400), 0.001, 0.03, s=True) == []


# --- Written once, verified on load -----------------------------------------------------------


def test_the_decision_is_saved_once_and_loads_verified(reused_world, tmp_path):
    _, outcomes, _ = reused_world
    decision = compute_decision(a_freeze(), outcomes, reproduction())
    path = save_decision(decision, tmp_path)

    loaded = load_decision(tmp_path, freeze=a_freeze())
    assert loaded["state"] == decision["state"]
    assert path.read_bytes().isascii()
    with pytest.raises(DecisionError, match="already"):
        save_decision(decision, tmp_path)


def test_a_decision_bound_to_another_freeze_is_refused_on_load(reused_world, tmp_path):
    _, outcomes, _ = reused_world
    save_decision(compute_decision(a_freeze(), outcomes, reproduction()), tmp_path)
    other = a_freeze()
    other["digest"] = "e" * 64
    with pytest.raises(DecisionError, match="freeze"):
        load_decision(tmp_path, freeze=other)


def test_a_decision_whose_state_was_rewritten_is_refused_on_load(reused_world, tmp_path):
    _, outcomes, _ = reused_world
    path = save_decision(compute_decision(a_freeze(), outcomes, reproduction()), tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"] = dp.STATE_REPLACEMENT_SUPPORTED
    payload["route"] = None
    payload["open_question"] = None
    from concept_embeddings_rag.evaluation.selection import digest_of_payload

    payload["digest"] = digest_of_payload(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    passed = (
        payload["secondary"]["passed"],
        payload["assay_sensitivity"]["passed"],
        payload["primary"]["passed"],
    )
    assert passed != (True, True, True), "the toy must not already land on the rewritten row"
    with pytest.raises(DecisionError, match="row"):
        load_decision(tmp_path, freeze=a_freeze())


def test_a_tampered_decision_is_refused_by_its_digest(reused_world, tmp_path):
    _, outcomes, _ = reused_world
    path = save_decision(compute_decision(a_freeze(), outcomes, reproduction()), tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["primary"]["b"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DecisionError, match="digest"):
        load_decision(tmp_path, freeze=a_freeze())


def test_paired_tables_sum_to_n():
    first = {"q1": True, "q2": True, "q3": False, "q4": False}
    second = {"q1": True, "q2": False, "q3": True, "q4": False}
    table = paired_table(first, second)
    assert (table.both, table.first_only, table.second_only, table.neither) == (1, 1, 1, 1)
    assert table.n == 4
