"""The decision parameters of Phase 6, exactly as the spec's statistical convention fixes them.

Everything under "Decision rules and statistical convention" in `6.spec.md` that a program
has to read: the metric, budget, split and population; the margin with its derivation; alpha;
the methods and their references; the state procedure, its eight-row table with the anomaly
texts and open questions; the scenario definitions (OI-3); and the qualitative sample rule of
HU-8. The freeze copies them from here and the test stage refuses a copy that differs.

**Why this is not `config.py`.** Plan D13 placed these constants in `config.py`. They name
the test split, and `config.py` is imported by the Phase 3 selection, whose guard
(`test_no_test_contamination.py`) forbids any module on that path from holding a string that
names a split other than dev. Keeping that closed guard intact is worth more than one file
for all constants, so the convention lives here, beside `config`, and nothing on the Phase 3
or Phase 4 selection paths imports it. The values are the spec's either way; only their
module changed (recorded in the plan's D13).

Nothing here is computed from data. The texts are stored as the table cells write them,
emphasis included; the two deltas are escapes, so the source stays ASCII (D21).
"""

from fractions import Fraction
from typing import Final, NamedTuple

from concept_embeddings_rag import config

DECISION_METRIC: Final[str] = "full_support"
DECISION_BUDGET: Final[int] = 2048
DECISION_SPLIT: Final[str] = "test"
N_TEST: Final[int] = config.N_TEST

# delta = (1 - f) * M1: M1 is the historical test effect of A over dense on Full Support
# at 2,048 (1,210 - 1,155 of 1,400), f the preserved fraction. An exact fraction, never a
# decimal: the boundary case must behave as `L > -delta` strictly.
M1_NUMERATOR: Final[int] = 55
M1_DENOMINATOR: Final[int] = 1400
PRESERVED_FRACTION: Final[Fraction] = Fraction(1, 2)
DELTA: Final[Fraction] = Fraction(M1_NUMERATOR, M1_DENOMINATOR) * (1 - PRESERVED_FRACTION)
M1_SOURCE_FILES: Final[tuple[str, str]] = (
    config.HISTORICAL_TEST_RESULT_FILES["hybrid-bm25"],
    config.HISTORICAL_TEST_RESULT_FILES["dense"],
)
DECISION_ALPHA: Final[float] = config.GATE_ALPHA

NON_INFERIORITY_METHOD: Final[str] = (
    "Tango score test for the difference of paired proportions at Delta0 = -delta, one-sided "
    "at alpha; equivalently the lower limit L of Tango's two-sided 90% score interval, over "
    "all n questions, ties kept in n; N passes if and only if L > -delta, strictly"
)
NON_INFERIORITY_REFERENCE: Final[str] = (
    "Tango, T. (1998). Equivalence test and confidence interval for the difference in "
    "proportions for the paired-sample design. Statistics in Medicine, 17(8), 891-908."
)
SIGN_TEST_METHOD: Final[str] = (
    "paired exact sign test (exact McNemar) on Full Support at 2,048 over the test questions; "
    "ties excluded from the denominator; one-sided exact binomial (p = 0.5, "
    "alternative = 'greater'); no discordant question gives p = 1; passes if and only if "
    "p < alpha"
)
MULTIPLICITY_REFERENCE: Final[str] = (
    "Berger, R. L. (1982). Multiparameter hypothesis testing and acceptance sampling. "
    "Technometrics, 24(4), 295-300."
)

STATE_REPLACEMENT_SUPPORTED: Final[str] = "ENTITY_REPLACEMENT_SUPPORTED"
STATE_PARTIAL_INVESTIGATE: Final[str] = "ENTITY_PARTIAL_INVESTIGATE"
STATE_STOP: Final[str] = "STOP"
ROUTE_P1: Final[str] = "P1"
ROUTE_P2: Final[str] = "P2"

STATE_PROCEDURE: Final[tuple[str, ...]] = (
    "1. S, V and N pass -> ENTITY_REPLACEMENT_SUPPORTED",
    "2. otherwise S passes -> ENTITY_PARTIAL_INVESTIGATE, route P1",
    "3. otherwise V and N pass -> ENTITY_PARTIAL_INVESTIGATE, route P2",
    "4. otherwise -> STOP",
)


class StateRow(NamedTuple):
    """One row of the spec's state table: the three test outcomes and what they record."""

    s: bool
    v: bool
    n: bool
    state: str
    route: str | None
    anomalies: tuple[str, ...]
    open_question: str | None


_NO_CONTROL_EFFECT = "control shows no effect over dense in this measurement"
_NOT_JUDGEABLE = (
    "*Can B replace A under this protocol?* Not judgeable: B adds to dense, but the control "
    "showed no effect over dense, so non-inferiority to it carries no evidence."
)
STATE_TABLE: Final[tuple[StateRow, ...]] = (
    StateRow(True, True, True, STATE_REPLACEMENT_SUPPORTED, None, (), None),
    StateRow(
        True,
        True,
        False,
        STATE_PARTIAL_INVESTIGATE,
        ROUTE_P1,
        (),
        "*Does the raw entity hop keep more than half of BM25's contribution to the hybrid?* "
        "Not established: B adds to dense, but its deficit against A was not shown to be "
        "smaller than \u03b4.",
    ),
    StateRow(
        True,
        False,
        True,
        STATE_PARTIAL_INVESTIGATE,
        ROUTE_P1,
        (f"{_NO_CONTROL_EFFECT}; the non-inferiority pass is uninterpretable",),
        _NOT_JUDGEABLE,
    ),
    StateRow(
        True,
        False,
        False,
        STATE_PARTIAL_INVESTIGATE,
        ROUTE_P1,
        (_NO_CONTROL_EFFECT,),
        _NOT_JUDGEABLE,
    ),
    StateRow(
        False,
        True,
        True,
        STATE_PARTIAL_INVESTIGATE,
        ROUTE_P2,
        ("non-inferiority shown without evidence of a gain over dense",),
        "*Does B add anything over dense?* Not established: B lies within \u03b4 of a "
        "control shown to beat dense, but its own gain over dense was not shown.",
    ),
    StateRow(False, True, False, STATE_STOP, None, (), None),
    StateRow(
        False,
        False,
        True,
        STATE_STOP,
        None,
        (
            f"{_NO_CONTROL_EFFECT}; the non-inferiority pass is uninterpretable and grounds no "
            "state",
        ),
        None,
    ),
    StateRow(False, False, False, STATE_STOP, None, (_NO_CONTROL_EFFECT,), None),
)

# OI-3, decided: operational definitions of the five descriptive scenarios. They decide
# nothing; more than one may hold, and none may.
SCENARIO_DEFINITIONS: Final[dict[str, str]] = {
    "a": "the 90% interval contains 0 and L <= -delta",
    "b": "-delta < delta_hat < 0",
    "c": "S passes and L <= -delta",
    "d": "delta_hat > 0 and L <= 0",
    "e": "delta_hat < 0, B's Full Support count below dense's, and S fails",
}

# HU-8: the qualitative sample, fixed before test, for reporting only.
QUALITATIVE_SAMPLE_GROUPS: Final[tuple[str, ...]] = (
    "B succeeds where A fails",
    "A succeeds where B fails",
    "B succeeds where dense fails",
    "dense succeeds where B fails",
)
QUALITATIVE_SAMPLE_RULE: Final[dict[str, object]] = {
    "split": DECISION_SPLIT,
    "budget": DECISION_BUDGET,
    "groups": QUALITATIVE_SAMPLE_GROUPS,
    "per_group": config.TRACES_READ_PER_ARM,
    "order": "ascending question id",
}
