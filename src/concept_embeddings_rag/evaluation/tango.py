"""Tango's score test and interval for a difference of paired proportions (Phase 6, D15).

Notation of the spec: over `n` paired questions, `b` counts B-only successes and `c` A-only
successes; the difference is `Delta = p_b - p_c`, observed as `delta_hat = (b - c) / n`. Ties -
questions both systems support, or neither does - stay in `n` and enter only through it.

**The formula.** Transcribed in plan D15 from the R source of `PropCIs::scoreci.mp`, whose
documentation cites Tango (1998), and rewritten in this notation:

    pb   = -(b + c) + (2n - b + c) * D0
    pc   = -c * D0 * (1 - D0)
    q    = (sqrt(pb^2 - 8n * pc) - pb) / (4n)
    T(D0) = (b - c - n * D0) / sqrt(n * (2q + D0 * (1 - D0)))

**Source: checked against the paper (T11).** Tango, T. (1998), "Equivalence test and confidence
interval for the difference in proportions for the paired-sample design", Statistics in Medicine
17(8), 891-908, equations (24)-(26), p. 895:

    Z(b, c; n, Delta) = (b - c + n Delta) / sqrt(n (2 q21 - Delta (Delta + 1)))      (24)
    q21 = (sqrt(B^2 - 4AC) - B) / (2A)                                              (25)
    A = 2n,  B = -b - c - (2n - b + c) Delta,  C = c Delta (Delta + 1)                (26)

The paper's `Delta > 0` is the margin of its hypothesis (17), p. 894, `H0: pi_N = pi_S - Delta`
against `H1: pi_N > pi_S - Delta`. At `D0 = -Delta` the formula above is (24)-(26) term by term:
`pb = B`, `pc = C`, `q = q21`, and the variance and numerator coincide. Table (15), p. 894, puts
`b` at row 1, column 2 (new-only successes) and `c` at row 2, column 1 (standard-only), which
are the spec's `b` and `c`. The check was made on 2026-09-17 against a copy of the article; the
tests compare the two forms over a grid of counts and margins.

What T10 checked independently of the paper still holds: `q` is the restricted
maximum-likelihood estimate of `p_c` under `p_b - p_c = D0` (the score equation of `b log(q + D0)
+ c log q + (n - b - c) log(1 - 2q - D0)` is `2n q^2 + pb q + pc = 0`), and at `D0 = 0` the
statistic is McNemar's `(b - c) / sqrt(b + c)`, as the paper's section 6.2 also notes.

**The worked example** is section 6.1 of the paper, reproduced in `PUBLISHED_EXAMPLE` with the
values the paper prints. The author read them from the paper, and this session checked them
against its copy of the article.

**The decision reads the statistic.** N passes if and only if `T(-delta) > z`, with `delta` the
exact fraction and `z = norm.ppf(1 - alpha)`. The lower limit `L` of the two-sided
`1 - 2 alpha` interval is also computed, by bracketed root finding, and the decision refuses to
return if `(L > -delta) != (T(-delta) > z)`: the equivalence rests on `T` decreasing in `D0`,
and a numerical failure of it is a deviation, not a tie to break.

Undefined arithmetic - a negative discriminant, a non-positive variance, counts that are not a
paired table, `D0` outside `(-1, 1)` - raises. Nothing is clipped. At `D0 = delta_hat` the
numerator is zero and the statistic is 0; with no discordant pair that point is `0 / 0`, whose
limit from both sides is 0, and the root finding uses that value there.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Final

from scipy.optimize import brentq
from scipy.stats import norm

METHOD: Final[str] = "Tango (1998) score test for paired proportions"

# T11: the worked example of Tango (1998), section 6.1, as the paper prints it. Cells of Table V
# (rows hydrogen peroxide, the new system; columns thermal, the standard): a = both effective,
# b = new-only, c = standard-only, d = neither. Printed figures are kept as the printed strings,
# so the precision they were published at travels with them. Read-only.
PUBLISHED_EXAMPLE: Final[Mapping[str, object]] = MappingProxyType(
    {
        "source": (
            "Tango, T. (1998). Equivalence test and confidence interval for the difference in "
            "proportions for the paired-sample design. Statistics in Medicine 17(8), 891-908"
        ),
        "section": "6.1 Cross-over Clinical Trials on Soft Contact Lenses",
        "values_page": 902,
        "table": "Table V",
        "table_page": 903,
        "equations": "(24)-(26)",
        "equations_page": 895,
        "a": 43,
        "b": 0,
        "c": 1,
        "d": 0,
        "n": 44,
        "delta": "0.1",
        "alpha": "0.05",
        "z_alpha": "1.645",
        "statistic": "1.709",
        "one_sided_p": "0.044",
        "lower_limit_90": "-0.096",
        "decimals": 3,
    }
)

# How close to the parameter-space boundary the bracket starts. The statistic diverges there
# whenever the observed difference is inside the space; a bracket that does not straddle the
# root is refused rather than widened silently.
BOUNDARY_OFFSET: Final[float] = 1e-10
ROOT_XTOL: Final[float] = 1e-15
ROOT_MAXITER: Final[int] = 500


class TangoError(Exception):
    """Arithmetic outside the statistic's domain, or two rules that should agree and do not."""


def _check_table(b: int, c: int, n: int) -> None:
    for name, value in (("b", b), ("c", c), ("n", n)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TangoError(f"{name} must be an integer count, not {value!r}")
    if n < 1 or b < 0 or c < 0 or b + c > n:
        raise TangoError(f"b = {b}, c = {c}, n = {n} is not a paired table")


def _root_of(*, n: int, pb: float, pc: float) -> float:
    discriminant = pb * pb - 8.0 * n * pc
    if discriminant < 0.0:
        raise TangoError(f"the discriminant {discriminant!r} is negative; T is undefined here")
    return (math.sqrt(discriminant) - pb) / (4.0 * n)


def restricted_estimate(b: int, c: int, n: int, delta0: float) -> float:
    """`q`: the maximum-likelihood estimate of `p_c` under `p_b - p_c = delta0`."""
    _check_table(b, c, n)
    if not -1.0 < delta0 < 1.0:
        raise TangoError(f"delta0 = {delta0!r} is outside (-1, 1)")
    pb = -(b + c) + (2 * n - b + c) * delta0
    pc = -c * delta0 * (1.0 - delta0)
    return _root_of(n=n, pb=pb, pc=pc)


def score_statistic(b: int, c: int, n: int, delta0: float) -> float:
    """`T(delta0)`; raises where it is undefined."""
    q = restricted_estimate(b, c, n, delta0)
    variance = n * (2.0 * q + delta0 * (1.0 - delta0))
    if not variance > 0.0:
        raise TangoError(f"the variance {variance!r} is not positive at delta0 = {delta0!r}")
    return (b - c - n * delta0) / math.sqrt(variance)


def _statistic_or_centre(b: int, c: int, n: int, delta0: float, centre: float) -> float:
    """`T`, taking its value 0 at the observed difference, where the numerator vanishes."""
    if delta0 == centre:
        return 0.0
    return score_statistic(b, c, n, delta0)


def lower_limit(b: int, c: int, n: int, z: float) -> float:
    """The `L` with `T(L) = z` below the observed difference; -1 when that is the difference."""
    _check_table(b, c, n)
    centre = (b - c) / n
    if centre == -1.0:
        return -1.0
    start = -1.0 + BOUNDARY_OFFSET

    def f(x: float) -> float:
        return _statistic_or_centre(b, c, n, x, centre) - z

    if not f(start) > 0.0:
        raise TangoError(f"T does not exceed z near -1 for b={b}, c={c}, n={n}; no bracket")
    return float(brentq(f, start, centre, xtol=ROOT_XTOL, maxiter=ROOT_MAXITER))


def upper_limit(b: int, c: int, n: int, z: float) -> float:
    """The `U` with `T(U) = -z` above the observed difference; 1 when that is the difference."""
    _check_table(b, c, n)
    centre = (b - c) / n
    if centre == 1.0:
        return 1.0
    end = 1.0 - BOUNDARY_OFFSET

    def f(x: float) -> float:
        return _statistic_or_centre(b, c, n, x, centre) + z

    if not f(end) < 0.0:
        raise TangoError(f"T does not fall below -z near 1 for b={b}, c={c}, n={n}; no bracket")
    return float(brentq(f, centre, end, xtol=ROOT_XTOL, maxiter=ROOT_MAXITER))


def score_interval(b: int, c: int, n: int, z: float) -> tuple[float, float]:
    """The two limits at `z`: `norm.ppf(0.95)` gives the two-sided 90% interval."""
    return lower_limit(b, c, n, z), upper_limit(b, c, n, z)


def limit_passes(lower: float, delta: Fraction) -> bool:
    """`L > -delta`, strictly, with the float `L` compared exactly against the fraction."""
    return Fraction(lower) > -delta


@dataclass(frozen=True)
class NonInferiority:
    """Everything the decision records about N."""

    b: int
    c: int
    n: int
    delta: Fraction
    alpha: float
    z: float
    statistic: float
    p_value: float
    lower_limit: float
    passed: bool

    @property
    def delta_hat(self) -> float:
        return (self.b - self.c) / self.n


def non_inferiority(b: int, c: int, n: int, *, delta: Fraction, alpha: float) -> NonInferiority:
    """N: `T(-delta) > z`, with `L > -delta` computed beside it and required to agree."""
    _check_table(b, c, n)
    if not isinstance(delta, Fraction):
        raise TangoError(f"delta must be an exact Fraction, not {delta!r}")
    if not 0.0 < alpha < 0.5:
        raise TangoError(f"alpha = {alpha!r} is not a one-sided level")
    z = float(norm.ppf(1.0 - alpha))
    statistic = score_statistic(b, c, n, float(-delta))
    by_statistic = statistic > z
    lower = lower_limit(b, c, n, z)
    by_limit = limit_passes(lower, delta)
    if by_statistic != by_limit:
        raise TangoError(
            f"the statistic rule (T(-delta) = {statistic!r} > z = {z!r}: {by_statistic}) and the "
            f"limit rule (L = {lower!r} > -delta: {by_limit}) disagree; this is a deviation"
        )
    return NonInferiority(
        b=b,
        c=c,
        n=n,
        delta=delta,
        alpha=alpha,
        z=z,
        statistic=statistic,
        p_value=float(norm.sf(statistic)),
        lower_limit=lower,
        passed=by_statistic,
    )


def require_published_example() -> Mapping[str, object]:
    """The worked example of Tango (1998), or a refusal while the T11 slot is empty (D14)."""
    if not PUBLISHED_EXAMPLE:
        raise TangoError(
            "the published worked example of Tango (1998) is not in PUBLISHED_EXAMPLE; the "
            "implementation has not been validated against it (T11), so the test stage refuses"
        )
    return PUBLISHED_EXAMPLE
