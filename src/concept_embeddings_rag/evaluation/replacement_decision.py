"""The decision of Phase 6: S, V and N, the state procedure, and `Phase6Decision` (D16).

Everything here reads per-question Full Support at 2,048 on test, from the three outcome
artifacts as read back from disk, and the decision parameters from the freeze's copy of them,
which must equal the spec's (`decision_parameters`). The decision function takes the freeze,
the outcomes and the test reproduction record, and nothing else: no metric, budget, margin,
alpha, method or population can be passed to it.

- **S** (B beats dense) and **V** (A beats dense) are `gate.sign_test`, unchanged, reached
  through an adapter that builds its input shape from two success vectors. Ties leave the
  denominator; no discordant question gives p = 1.
- **N** (B non-inferior to A) is Tango's score test at `-delta`, over all `n` questions, ties
  kept in `n` (`tango.non_inferiority`).
- **The state** follows the ordered procedure, and the row of the freeze's state table for
  `(S, V, N)` must agree with it; its route, anomalies and open question are copied from that
  row verbatim.
- **Scenarios** (a)-(e) are descriptive, by the definitions decided in OI-3.
- **Descriptive readings**, labelled as deciding nothing: the superiority reading of B against A,
  the one-sided 97.5% bound, the 90% interval of A - dense (the M1 interval) and the two-sided
  exact McNemar p-value of B against dense.

Written once, with a digest, and refused on load if it is modified, bound to another freeze,
or records a state other than its row's.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from scipy.stats import binomtest, norm

from concept_embeddings_rag import decision_parameters as dp
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.evaluation import tango
from concept_embeddings_rag.evaluation.gate import SignTest, sign_test
from concept_embeddings_rag.evaluation.outcomes import PerQuestionOutcomes
from concept_embeddings_rag.evaluation.selection import digest_of_payload, serialize_payload

DECISION_FILENAME = "decision.json"
DENSE, SYSTEM_A, SYSTEM_B = "dense", "hybrid-bm25", "hybrid-entity-hop"
SYSTEMS: tuple[str, ...] = (DENSE, SYSTEM_A, SYSTEM_B)
MODES: tuple[str, ...] = ("reused", "re-measured")
REUSED = "reused"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
DESCRIPTIVE_Z_90 = 0.95
DESCRIPTIVE_Z_97_5 = 0.975


class DecisionError(Exception):
    """The decision cannot be computed from what it was handed, or its artifact does not verify."""


# --- The decision parameters, as the freeze copies them ---------------------------------------


def _delta_as_written() -> tuple[int, int]:
    """delta as the spec writes it, 55 / 2,800: M1's terms times (1 - f), not reduced."""
    kept = 1 - dp.PRESERVED_FRACTION
    numerator, denominator = dp.M1_NUMERATOR * kept.numerator, dp.M1_DENOMINATOR * kept.denominator
    if Fraction(numerator, denominator) != dp.DELTA:
        raise DecisionError(f"{numerator} / {denominator} is not the declared delta {dp.DELTA}")
    return numerator, denominator


def parameters_payload() -> dict[str, Any]:
    """The spec's decision parameters as JSON-ready values: what the freeze copies."""
    numerator, denominator = _delta_as_written()
    return {
        "metric": dp.DECISION_METRIC,
        "budget": dp.DECISION_BUDGET,
        "split": dp.DECISION_SPLIT,
        "n": dp.N_TEST,
        "delta": {
            "numerator": numerator,
            "denominator": denominator,
            "m1_numerator": dp.M1_NUMERATOR,
            "m1_denominator": dp.M1_DENOMINATOR,
            "preserved_fraction": str(dp.PRESERVED_FRACTION),
            "m1_source_files": list(dp.M1_SOURCE_FILES),
            "rule": "delta = (1 - f) * M1",
        },
        "alpha": dp.DECISION_ALPHA,
        "non_inferiority": {
            "method": dp.NON_INFERIORITY_METHOD,
            "sidedness": "one-sided",
            "reference": dp.NON_INFERIORITY_REFERENCE,
        },
        "sign_test": {"method": dp.SIGN_TEST_METHOD},
        "multiplicity_reference": dp.MULTIPLICITY_REFERENCE,
        "state_procedure": list(dp.STATE_PROCEDURE),
        "state_table": [
            {**row._asdict(), "anomalies": list(row.anomalies)} for row in dp.STATE_TABLE
        ],
        "scenario_definitions": dict(dp.SCENARIO_DEFINITIONS),
        "qualitative_sample_rule": _json_ready(dp.QUALITATIVE_SAMPLE_RULE),
    }


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value))


def check_parameters(copy: Mapping[str, Any]) -> None:
    """Refuse a copy of the decision parameters that differs from the spec's in any field."""
    expected = _json_ready(parameters_payload())
    observed = _json_ready(copy)
    if observed != expected:
        differing = sorted(
            key for key in set(expected) | set(observed) if expected.get(key) != observed.get(key)
        )
        raise DecisionError(
            f"the decision parameters copied into the freeze differ from the spec's in {differing}"
        )


def delta_of(parameters: Mapping[str, Any]) -> Fraction:
    block = parameters["delta"]
    delta = Fraction(int(block["numerator"]), int(block["denominator"]))
    derived = Fraction(int(block["m1_numerator"]), int(block["m1_denominator"])) * (
        1 - Fraction(str(block["preserved_fraction"]))
    )
    if delta != derived:
        raise DecisionError(f"delta {delta} is not (1 - f) * M1 = {derived}")
    return delta


# --- Paired counts and sign tests -------------------------------------------------------------


@dataclass(frozen=True)
class PairedTable:
    both: int
    first_only: int
    second_only: int
    neither: int

    @property
    def n(self) -> int:
        return self.both + self.first_only + self.second_only + self.neither

    def as_payload(self) -> dict[str, int]:
        return {
            "both": self.both,
            "first_only": self.first_only,
            "second_only": self.second_only,
            "neither": self.neither,
        }


def paired_table(first: Mapping[str, bool], second: Mapping[str, bool]) -> PairedTable:
    if set(first) != set(second):
        raise DecisionError("a paired table needs the same questions on both sides")
    both = first_only = second_only = neither = 0
    for qid in first:
        if first[qid] and second[qid]:
            both += 1
        elif first[qid]:
            first_only += 1
        elif second[qid]:
            second_only += 1
        else:
            neither += 1
    return PairedTable(both, first_only, second_only, neither)


def sign_test_of(
    system: str,
    successes: Mapping[str, bool],
    against: str,
    against_successes: Mapping[str, bool],
    *,
    budget: int,
    alpha: float,
) -> SignTest:
    """`gate.sign_test`, unchanged, over two success vectors at one budget."""
    if set(successes) != set(against_successes):
        raise DecisionError("a sign test needs the same questions on both sides")
    key = str(budget)
    per_question = [
        {
            "scores": {
                system: {key: float(successes[qid])},
                against: {key: float(against_successes[qid])},
            }
        }
        for qid in sorted(successes)
    ]
    return sign_test(per_question, system, against, depth=budget, alpha=alpha)


def _sign_payload(test: SignTest) -> dict[str, Any]:
    return {
        "system": test.hop,
        "against": test.against,
        "wins": test.wins,
        "losses": test.losses,
        "ties": test.ties,
        "p_value": test.p_value,
        "passed": test.passed,
    }


# --- The state procedure ----------------------------------------------------------------------


def decide(s: bool, v: bool, n: bool, table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The ordered procedure, and the table row it must agree with; texts from the row."""
    if s and v and n:
        state, route = dp.STATE_REPLACEMENT_SUPPORTED, None
    elif s:
        state, route = dp.STATE_PARTIAL_INVESTIGATE, dp.ROUTE_P1
    elif v and n:
        state, route = dp.STATE_PARTIAL_INVESTIGATE, dp.ROUTE_P2
    else:
        state, route = dp.STATE_STOP, None
    rows = [row for row in table if (row["s"], row["v"], row["n"]) == (s, v, n)]
    if len(rows) != 1:
        raise DecisionError(f"the state table holds {len(rows)} rows for (S, V, N) = {(s, v, n)}")
    row = dict(rows[0])
    if (row["state"], row["route"]) != (state, route):
        raise DecisionError(
            f"the table row for {(s, v, n)} records {(row['state'], row['route'])}, and the "
            f"ordered procedure gives {(state, route)}"
        )
    return row


def scenarios(
    *,
    delta_hat: Fraction,
    lower: float,
    upper: float,
    s_passed: bool,
    b_count: int,
    dense_count: int,
    delta: Fraction,
) -> list[str]:
    """Which of (a)-(e) describe the outcome, by the OI-3 definitions. Descriptive."""
    low, high = Fraction(lower), Fraction(upper)
    held = {
        "a": low <= 0 <= high and low <= -delta,
        "b": -delta < delta_hat < 0,
        "c": s_passed and low <= -delta,
        "d": delta_hat > 0 and low <= 0,
        "e": delta_hat < 0 and b_count < dense_count and not s_passed,
    }
    return [name for name in sorted(held) if held[name]]


def _two_sided_mcnemar(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if not discordant:
        return 1.0
    return float(binomtest(first_only, discordant, 0.5, alternative="two-sided").pvalue)


# --- The decision -----------------------------------------------------------------------------


def _successes(outcomes: PerQuestionOutcomes, metric: str, budget: int) -> dict[str, bool]:
    values: dict[str, bool] = {}
    for entry in outcomes.questions:
        value = entry[f"budget_{budget}"][metric]
        if value not in (0.0, 1.0):
            raise DecisionError(f"{outcomes.system}: {metric} of {entry['qid']} is not 0 or 1")
        values[str(entry["qid"])] = value == 1.0
    return values


def _check_outcomes(
    outcomes: Mapping[str, Any], parameters: Mapping[str, Any], freeze_digest: str
) -> dict[str, PerQuestionOutcomes]:
    if set(outcomes) != set(SYSTEMS):
        raise DecisionError(f"the decision needs the outcomes of the systems {SYSTEMS}")
    checked: dict[str, PerQuestionOutcomes] = {}
    population: set[str] | None = None
    for system in SYSTEMS:
        artifact = outcomes[system]
        if not isinstance(artifact, PerQuestionOutcomes) or not Path(artifact.path).exists():
            raise DecisionError(f"the {system} outcomes were not read back from disk")
        if artifact.system != system:
            raise DecisionError(f"the outcomes filed under {system} belong to {artifact.system}")
        if artifact.split != parameters["split"]:
            raise DecisionError(f"the {system} outcomes are on split {artifact.split!r}")
        if artifact.freeze_digest != freeze_digest:
            raise DecisionError(f"the {system} outcomes were measured under another freeze")
        qids = {str(entry["qid"]) for entry in artifact.questions}
        if len(qids) != int(parameters["n"]):
            raise DecisionError(
                f"the {system} population holds {len(qids)} questions, not n = {parameters['n']}"
            )
        if population is not None and qids != population:
            raise DecisionError("the three systems were measured on different populations")
        population = qids
        checked[system] = artifact
    return checked


def compute_decision(
    freeze: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    test_reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    """The decision payload, from the freeze, the three test outcomes and step 5's record."""
    parameters = freeze["decision_parameters"]
    check_parameters(parameters)
    freeze_digest = str(freeze.get("digest", ""))
    if not SHA256.match(freeze_digest):
        raise DecisionError("the freeze carries no sha256 digest")
    mode = str(freeze["control"]["mode"])
    if mode not in MODES:
        raise DecisionError(f"unknown control mode {mode!r}")
    if test_reproduction.get("mode") != mode:
        raise DecisionError(
            f"the test reproduction was recorded under mode {test_reproduction.get('mode')!r}, "
            f"and the freeze says {mode!r}"
        )
    if mode == REUSED and test_reproduction.get("passed") is not True:
        raise DecisionError(
            "in reused mode B is decided only after a passing test reproduction; this one failed"
        )

    artifacts = _check_outcomes(outcomes, parameters, freeze_digest)
    metric, budget = str(parameters["metric"]), int(parameters["budget"])
    alpha, n = float(parameters["alpha"]), int(parameters["n"])
    delta = delta_of(parameters)
    written_delta = f"{parameters['delta']['numerator']}/{parameters['delta']['denominator']}"
    dense = _successes(artifacts[DENSE], metric, budget)
    control = _successes(artifacts[SYSTEM_A], metric, budget)
    entity = _successes(artifacts[SYSTEM_B], metric, budget)

    b_vs_a = paired_table(entity, control)
    b_vs_dense = paired_table(entity, dense)
    a_vs_dense = paired_table(control, dense)
    primary_test = tango.non_inferiority(
        b_vs_a.first_only, b_vs_a.second_only, n, delta=delta, alpha=alpha
    )
    secondary = sign_test_of(SYSTEM_B, entity, DENSE, dense, budget=budget, alpha=alpha)
    assay = sign_test_of(SYSTEM_A, control, DENSE, dense, budget=budget, alpha=alpha)

    row = decide(secondary.passed, assay.passed, primary_test.passed, parameters["state_table"])

    control_count, dense_count = sum(control.values()), sum(dense.values())
    m1_numerator = int(parameters["delta"]["m1_numerator"])
    m1_check = {
        "expected": m1_numerator,
        "observed": control_count - dense_count,
        "passed": control_count - dense_count == m1_numerator,
    }
    if mode == REUSED and not m1_check["passed"]:
        raise DecisionError(
            f"in reused mode A - dense must be {m1_numerator} questions on test, and the outcomes "
            f"give {control_count - dense_count}"
        )

    z90 = float(norm.ppf(DESCRIPTIVE_Z_90))
    low, high = tango.score_interval(b_vs_a.first_only, b_vs_a.second_only, n, z90)
    delta_hat = Fraction(b_vs_a.first_only - b_vs_a.second_only, n)
    position = "above zero" if low > 0.0 else "below zero" if high < 0.0 else "contains zero"

    return {
        "freeze_digest": freeze_digest,
        "outcome_digests": {system: artifacts[system].digest for system in SYSTEMS},
        "control_mode": mode,
        "test_reproduction": dict(test_reproduction),
        "counts": {
            DENSE: dense_count,
            SYSTEM_A: control_count,
            SYSTEM_B: sum(entity.values()),
        },
        "tables": {
            "b_vs_a": b_vs_a.as_payload(),
            "b_vs_dense": b_vs_dense.as_payload(),
            "a_vs_dense": a_vs_dense.as_payload(),
        },
        "primary": {
            "n": n,
            "b": b_vs_a.first_only,
            "c": b_vs_a.second_only,
            "both": b_vs_a.both,
            "neither": b_vs_a.neither,
            "delta_hat": float(delta_hat),
            "delta": {"fraction": written_delta, "value": float(delta)},
            "alpha": alpha,
            "method": parameters["non_inferiority"]["method"],
            "statistic": primary_test.statistic,
            "z": primary_test.z,
            "p_value": primary_test.p_value,
            "lower_limit": primary_test.lower_limit,
            "passed": primary_test.passed,
        },
        "secondary": _sign_payload(secondary),
        "assay_sensitivity": _sign_payload(assay),
        "m1_check": m1_check,
        "state": row["state"],
        "route": row["route"],
        "anomalies": list(row["anomalies"]),
        "open_question": row["open_question"],
        "scenarios": scenarios(
            delta_hat=delta_hat,
            lower=low,
            upper=high,
            s_passed=secondary.passed,
            b_count=sum(entity.values()),
            dense_count=dense_count,
            delta=delta,
        ),
        "descriptive": {
            "decides_nothing": True,
            "superiority": {
                "delta_hat": float(delta_hat),
                "interval_90": [low, high],
                "position": position,
                "mcnemar_two_sided_p": _two_sided_mcnemar(b_vs_a.first_only, b_vs_a.second_only),
            },
            "lower_bound_97_5": tango.lower_limit(
                b_vs_a.first_only, b_vs_a.second_only, n, float(norm.ppf(DESCRIPTIVE_Z_97_5))
            ),
            "m1_interval_90": list(
                tango.score_interval(a_vs_dense.first_only, a_vs_dense.second_only, n, z90)
            ),
            "b_vs_dense_two_sided_p": _two_sided_mcnemar(
                b_vs_dense.first_only, b_vs_dense.second_only
            ),
        },
    }


def save_decision(decision: Mapping[str, Any], directory: Path | str) -> Path:
    """Write the decision once. It is never recomputed with other parameters."""
    path = Path(directory) / DECISION_FILENAME
    if path.exists():
        raise DecisionError(
            f"a decision is already written at {path}; the state is computed once and never "
            "re-run with another metric, budget, margin, alpha, method, comparator or population"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in decision.items() if key != "digest"}
    payload["digest"] = digest_of_payload(payload)
    write_text_atomic(path, serialize_payload(payload))
    return path


def load_decision(directory: Path | str, *, freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Read the decision back: its digest, its freeze, and the row its tests select."""
    path = Path(directory) / DECISION_FILENAME
    if not path.exists():
        raise DecisionError(f"no decision in {directory}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("digest") != digest_of_payload(payload):
        raise DecisionError(f"{path.name} does not match its digest; it has been modified")
    if payload.get("freeze_digest") != freeze.get("digest"):
        raise DecisionError(f"{path.name} was computed under another freeze")
    check_parameters(freeze["decision_parameters"])
    row = decide(
        bool(payload["secondary"]["passed"]),
        bool(payload["assay_sensitivity"]["passed"]),
        bool(payload["primary"]["passed"]),
        freeze["decision_parameters"]["state_table"],
    )
    for field in ("state", "route", "anomalies", "open_question"):
        if payload.get(field) != row[field]:
            raise DecisionError(f"{path.name} records a {field} other than its row's")
    return payload
