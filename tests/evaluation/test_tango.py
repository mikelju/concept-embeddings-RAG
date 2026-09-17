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

T11 adds the paper itself: the statistic is compared term by term with equations (24)-(26) of
Tango (1998), and the worked example of its section 6.1 is reproduced at the printed precision.
The example's values come from the paper and nowhere else.
"""

import math
from fractions import Fraction
from types import MappingProxyType

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


# --- The paper: equations (24)-(26) and the worked example (T11) ---------------------------


def tango_1998_statistic(b: int, c: int, n: int, margin: float) -> float:
    """Equations (24)-(26) of Tango (1998), p. 895, transcribed as printed.

    `Z(b, c; n, Delta) = (b - c + n Delta) / sqrt(n (2 q21 - Delta (Delta + 1)))`, where `q21` is
    the larger root of `A x^2 + B x + C = 0`, `A = 2n`, `B = -b - c - (2n - b + c) Delta` and
    `C = c Delta (Delta + 1)`. The paper's `Delta > 0` is the margin of its hypothesis (17).
    """
    a_coef = 2 * n
    b_coef = -b - c - (2 * n - b + c) * margin
    c_coef = c * margin * (margin + 1)
    q21 = (math.sqrt(b_coef**2 - 4 * a_coef * c_coef) - b_coef) / (2 * a_coef)
    return (b - c + n * margin) / math.sqrt(n * (2 * q21 - margin * (margin + 1)))


@pytest.mark.parametrize("margin", [0.05, 0.1, 0.2, float(dp.DELTA)])
@pytest.mark.parametrize(
    ("b", "c", "n"),
    [(b, c, n) for n in (44, 60, 1400) for b in (0, 1, 13) for c in (0, 1, 21) if b + c <= n],
)
def test_the_statistic_is_tangos_equations_24_to_26_at_minus_the_margin(b, c, n, margin):
    """D15's formula at `D0 = -Delta` is the paper's `Z(b, c; n, Delta)`, term by term."""
    assert score_statistic(b, c, n, -margin) == pytest.approx(
        tango_1998_statistic(b, c, n, margin), rel=1e-12
    )


def test_the_published_example_names_its_source_location():
    example = require_published_example()

    assert "Tango" in str(example["source"]) and "891-908" in str(example["source"])
    assert example["section"] == "6.1 Cross-over Clinical Trials on Soft Contact Lenses"
    assert example["table"] == "Table V"
    assert (example["equations"], example["equations_page"]) == ("(24)-(26)", 895)
    assert (example["values_page"], example["table_page"]) == (902, 903)


def test_the_published_example_is_table_v_as_printed():
    """Table V, p. 903: rows hydrogen peroxide (new), columns thermal (standard), n = 44.

    Table (15), p. 894, puts `b` at row 1, column 2 (new-only successes) and `c` at row 2,
    column 1 (standard-only successes): the spec's `b` and `c`.
    """
    example = PUBLISHED_EXAMPLE
    cells = (example["a"], example["b"], example["c"], example["d"])

    assert cells == (43, 0, 1, 0)
    assert example["n"] == sum(cells) == 44


def printed(value: float, example) -> str:
    """A computed value at the precision the paper prints."""
    return f"{value:.{example['decimals']}f}"


def test_the_implementation_reproduces_the_published_contact_lens_example():
    """Section 6.1, p. 902: `Z = 1.709 > Z_0.05 = 1.645` (one-tailed p = 0.044), and the 90 per
    cent lower limit -0.096 > -Delta = -0.1, so the two methods are concluded equivalent.

    Compared at the printed precision: our value rounded to the printed decimals must be the
    printed string. The formula is never adjusted to make this pass (T11).
    """
    example = PUBLISHED_EXAMPLE
    b, c, n = example["b"], example["c"], example["n"]
    margin = Fraction(str(example["delta"]))
    z = float(norm.ppf(1.0 - float(example["alpha"])))

    result = non_inferiority(b, c, n, delta=margin, alpha=float(example["alpha"]))

    assert printed(z, example) == example["z_alpha"]
    assert printed(result.statistic, example) == example["statistic"]
    assert printed(result.p_value, example) == example["one_sided_p"]
    assert printed(result.lower_limit, example) == example["lower_limit_90"]
    assert result.passed is True


def test_the_published_lower_limit_fixes_which_cell_is_b():
    """Swapping the off-diagonal cells moves the published limit to the other side of zero."""
    swapped = lower_limit(1, 0, 44, float(norm.ppf(0.95)))

    assert f"{swapped:.3f}" != PUBLISHED_EXAMPLE["lower_limit_90"]
    assert f"{upper_limit(1, 0, 44, float(norm.ppf(0.95))):.3f}" == "0.096"


def test_the_published_example_slot_is_read_only():
    with pytest.raises(TypeError):
        PUBLISHED_EXAMPLE["b"] = 1  # type: ignore[index]


def test_requiring_the_published_example_still_refuses_an_empty_slot(monkeypatch):
    monkeypatch.setattr(tango, "PUBLISHED_EXAMPLE", MappingProxyType({}))
    with pytest.raises(TangoError, match="published"):
        require_published_example()


def test_the_module_records_the_paper_check_and_its_equation_numbers():
    doc = tango.__doc__ or ""
    assert "not verified against the paper" not in doc
    assert "(24)-(26)" in doc and "p. 895" in doc
