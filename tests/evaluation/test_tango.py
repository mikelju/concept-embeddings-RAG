"""Phase 6, T10: Tango's score statistic and interval for paired proportions (D15).

The implementation is checked by properties that can be derived and verified independently of
any reference value:

- at `delta0 = 0` the statistic is McNemar's, `(b - c) / sqrt(b + c)`;
- the nuisance estimate `q` is the restricted maximum-likelihood estimate: it maximizes the
  multinomial likelihood under `p_b - p_c = delta0`, checked with a bounded numerical optimizer;
- `L <= delta_hat <= U`; swapping `b` and `c` negates the interval; the limits lie in [-1, 1];
- the interval is defined at `b = c = 0` and at extreme splits, and undefined arithmetic raises;
- the limits agree with an independent bisection written here from the likelihood, not copied;
- the decision uses the exact fraction delta and `L > -delta` strictly, and a disagreement
  between the statistic rule and the limit rule is refused.

The published worked example (T11) is not here: its values come from the paper and nowhere
else, and `PUBLISHED_EXAMPLE` stays empty until then.
"""

import math
from fractions import Fraction

import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from concept_embeddings_rag import decision_parameters as dp
from concept_embeddings_rag.evaluation import tango
from concept_embeddings_rag.evaluation.tango import (
    PUBLISHED_EXAMPLE,
    TangoError,
    limit_passes,
    lower_limit,
    non_inferiority,
    require_published_example,
    restricted_estimate,
    score_interval,
    score_statistic,
    upper_limit,
)

Z95 = float(norm.ppf(0.95))
GRID = [
    (b, c, n)
    for n in (1, 7, 60, 1400)
    for b in sorted({0, 1, n // 7, n // 3, n})
    for c in sorted({0, 1, n // 5, n // 2, n})
    if b + c <= n
]


# --- The statistic --------------------------------------------------------------------------


@pytest.mark.parametrize(("b", "c", "n"), [(40, 20, 160), (1, 0, 5), (3, 9, 1400), (55, 0, 60)])
def test_at_delta0_zero_the_statistic_is_mcnemars(b, c, n):
    assert score_statistic(b, c, n, 0.0) == pytest.approx((b - c) / math.sqrt(b + c), rel=1e-12)


def log_likelihood(q: float, b: int, c: int, n: int, delta0: float) -> float:
    p_b, p_c, rest = q + delta0, q, 1.0 - 2.0 * q - delta0
    total = 0.0
    for count, probability in ((b, p_b), (c, p_c), (n - b - c, rest)):
        if count:
            if probability <= 0.0:
                return -math.inf
            total += count * math.log(probability)
    return total


@pytest.mark.parametrize(
    ("b", "c", "n", "delta0"),
    [(40, 20, 160, -0.05), (40, 20, 160, 0.2), (3, 9, 1400, -0.0196), (120, 80, 1400, 0.01)],
)
def test_the_nuisance_estimate_is_the_restricted_maximum_likelihood(b, c, n, delta0):
    q = restricted_estimate(b, c, n, delta0)
    low, high = max(0.0, -delta0), (1.0 - delta0) / 2.0
    best = minimize_scalar(
        lambda x: -log_likelihood(x, b, c, n, delta0),
        bounds=(low + 1e-12, high - 1e-12),
        method="bounded",
        options={"xatol": 1e-13},
    )
    assert q == pytest.approx(best.x, abs=1e-7)


def test_the_statistic_decreases_in_delta0():
    values = [score_statistic(55, 30, 1400, x / 1000) for x in range(-100, 101, 5)]
    assert all(later < earlier for earlier, later in zip(values, values[1:], strict=False))


@pytest.mark.parametrize(
    ("b", "c", "n", "delta0"), [(0, 0, 10, 0.0), (5, 5, 10, 1.0), (5, 5, 10, -1.0)]
)
def test_undefined_arithmetic_raises_and_nothing_is_clipped(b, c, n, delta0):
    with pytest.raises(TangoError):
        score_statistic(b, c, n, delta0)


def test_a_negative_discriminant_raises_rather_than_being_clipped():
    with pytest.raises(TangoError, match="discriminant"):
        tango._root_of(n=10, pb=1.0, pc=1.0)


@pytest.mark.parametrize(("b", "c", "n"), [(-1, 0, 5), (3, 3, 5), (0, 0, 0), (1.5, 0, 5)])
def test_counts_that_are_not_a_paired_table_are_refused(b, c, n):
    with pytest.raises(TangoError):
        score_statistic(b, c, n, 0.1)


# --- The interval ---------------------------------------------------------------------------


@pytest.mark.parametrize(("b", "c", "n"), GRID)
def test_the_interval_contains_the_observed_difference_and_lies_in_the_parameter_space(b, c, n):
    low, high = score_interval(b, c, n, Z95)
    delta_hat = (b - c) / n

    assert -1.0 <= low <= delta_hat + 1e-12
    assert delta_hat - 1e-12 <= high <= 1.0


@pytest.mark.parametrize(("b", "c", "n"), GRID)
def test_swapping_b_and_c_negates_the_interval(b, c, n):
    low, high = score_interval(b, c, n, Z95)
    swapped_low, swapped_high = score_interval(c, b, n, Z95)
    assert swapped_low == pytest.approx(-high, abs=1e-9)
    assert swapped_high == pytest.approx(-low, abs=1e-9)


def test_the_interval_is_defined_with_no_discordant_pair():
    low, high = score_interval(0, 0, 1400, Z95)
    assert -1.0 < low < 0.0 < high < 1.0
    assert low == pytest.approx(-high, abs=1e-12)


@pytest.mark.parametrize(("b", "c", "n"), [(1400, 0, 1400), (0, 1400, 1400), (1, 0, 1), (0, 1, 1)])
def test_the_interval_is_defined_at_extreme_splits(b, c, n):
    low, high = score_interval(b, c, n, Z95)
    assert -1.0 <= low <= high <= 1.0
    if b == n:
        assert high == 1.0
    if c == n:
        assert low == -1.0


def test_the_limits_are_the_roots_of_the_statistic_at_plus_and_minus_z():
    low, high = score_interval(55, 30, 1400, Z95)
    assert score_statistic(55, 30, 1400, low) == pytest.approx(Z95, abs=1e-8)
    assert score_statistic(55, 30, 1400, high) == pytest.approx(-Z95, abs=1e-8)
    assert lower_limit(55, 30, 1400, Z95) == low
    assert upper_limit(55, 30, 1400, Z95) == high


def independent_statistic(b: int, c: int, n: int, delta0: float) -> float:
    """From the likelihood: q by bisection on the score equation, then the Wald-type ratio."""
    low, high = max(0.0, -delta0), (1.0 - delta0) / 2.0

    def score(q: float) -> float:
        m = n - b - c
        terms = 0.0
        if b:
            terms += b / (q + delta0)
        if c:
            terms += c / q
        if m:
            terms -= 2.0 * m / (1.0 - 2.0 * q - delta0)
        return terms

    lo, hi = low + 1e-15, high - 1e-15
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    q = (lo + hi) / 2.0
    return (b - c - n * delta0) / math.sqrt(n * (2.0 * q + delta0 * (1.0 - delta0)))


def independent_lower(b: int, c: int, n: int, z: float) -> float:
    lo, hi = -1.0 + 1e-12, (b - c) / n
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if independent_statistic(b, c, n, mid) > z:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@pytest.mark.parametrize(
    ("b", "c", "n"),
    [(b, c, n) for n in (60, 1400) for b in (1, 13, 55) for c in (2, 21, 40) if b + c < n],
)
def test_the_lower_limit_agrees_with_an_independent_bisection(b, c, n):
    assert lower_limit(b, c, n, Z95) == pytest.approx(independent_lower(b, c, n, Z95), abs=1e-7)


# --- The decision rule ---------------------------------------------------------------------


def test_the_decision_reads_the_statistic_at_minus_delta_and_records_the_limit():
    result = non_inferiority(40, 45, 1400, delta=dp.DELTA, alpha=dp.DECISION_ALPHA)

    assert result.delta == dp.DELTA
    assert result.statistic == score_statistic(40, 45, 1400, float(-dp.DELTA))
    assert result.z == pytest.approx(Z95, rel=1e-15)
    assert result.p_value == pytest.approx(float(norm.sf(result.statistic)), rel=1e-12)
    assert result.passed is (result.statistic > result.z)
    assert result.passed is limit_passes(result.lower_limit, dp.DELTA)


def test_a_clearly_non_inferior_and_a_clearly_inferior_case():
    assert non_inferiority(10, 10, 1400, delta=dp.DELTA, alpha=0.05).passed is True
    assert non_inferiority(0, 90, 1400, delta=dp.DELTA, alpha=0.05).passed is False


def test_the_limit_rule_uses_the_exact_fraction_and_is_strict_on_the_boundary():
    delta = Fraction(1, 64)
    boundary = -0.015625
    assert Fraction(boundary) == -delta
    assert limit_passes(boundary, delta) is False
    assert limit_passes(math.nextafter(boundary, 1.0), delta) is True
    assert limit_passes(math.nextafter(boundary, -1.0), delta) is False


def test_the_decision_delta_must_be_a_fraction():
    with pytest.raises(TangoError, match="Fraction"):
        non_inferiority(10, 10, 1400, delta=0.019643, alpha=0.05)  # type: ignore[arg-type]


def test_a_forced_disagreement_between_the_two_rules_raises(monkeypatch):
    honest = non_inferiority(10, 10, 1400, delta=dp.DELTA, alpha=0.05)
    assert honest.passed
    monkeypatch.setattr(tango, "lower_limit", lambda b, c, n, z: -0.5)
    with pytest.raises(TangoError, match="disagree"):
        non_inferiority(10, 10, 1400, delta=dp.DELTA, alpha=0.05)


def test_ties_stay_in_n_and_enter_only_through_it():
    """No parameter names 'both' or 'neither': moving a tie between them cannot change N."""
    import inspect

    assert list(inspect.signature(non_inferiority).parameters) == ["b", "c", "n", "delta", "alpha"]
    fewer_ties = non_inferiority(12, 20, 1000, delta=dp.DELTA, alpha=0.05)
    more_ties = non_inferiority(12, 20, 1400, delta=dp.DELTA, alpha=0.05)
    assert fewer_ties.statistic != more_ties.statistic


# --- The declared gap ----------------------------------------------------------------------


def test_the_published_example_slot_exists_and_is_empty():
    assert len(PUBLISHED_EXAMPLE) == 0
    with pytest.raises(TypeError):
        PUBLISHED_EXAMPLE["b"] = 1  # type: ignore[index]


def test_requiring_the_published_example_refuses_while_the_slot_is_empty():
    with pytest.raises(TangoError, match="published"):
        require_published_example()


def test_the_module_says_the_paper_equation_was_not_checked_in_this_session():
    assert "not verified against the paper" in (tango.__doc__ or "")
