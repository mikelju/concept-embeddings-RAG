"""Whether the Phase 3 control is reused or re-measured, decided by exact reproduction (D12).

HU-2: the historical Dense + BM25 configuration becomes Phase 6's control only if Phase 6
reproduces it, value for value. Everything here is a pure function over figures the caller
hands in; this module opens no file. Every check is recorded as `{name, expected, observed,
passed}`, with `expected` the historical value and `observed` Phase 6's, compared with `==`:

- **The fit.** The re-executed `fit_fusion_weight([dense, bm25])` against `selection.json`'s
  control: components, metric, budget, grid, question count, the curve at every grid point,
  the RRF score, and the selection they derive (scheme and weights by the Phase 3 rules).
- **The configuration.** For each of the four historical files, the HU-2 keys (`unit_set_hash`,
  `tokenizer`, `seed`, `top_k`, `model`, `revision`, and for `hybrid-bm25` also `fusion_scheme`
  and `fusion_weights`) against the configuration Phase 6's own run of that system records.
- **The figures.** For `dense` and `hybrid-bm25`, per budget gold recall, Full Support and
  precision; recall at every depth; mean units included. Latency is not deterministic and is
  never compared.

The mode is `reused` if and only if every dev check passes, and `re-measured` otherwise. The
test-side comparison applies the same figure list to the test aggregates; its outcome is not
part of the mode. In `reused` mode a failure stops the run before B exists; in `re-measured`
mode the comparison is descriptive and returns without raising.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from concept_embeddings_rag import config, decision_parameters
from concept_embeddings_rag.evaluation.selection import DEV_SPLIT, FusionFit

REUSED = "reused"
REMEASURED = "re-measured"
MODES: tuple[str, ...] = (REUSED, REMEASURED)
CONTROL_SYSTEMS: tuple[str, ...] = ("dense", "hybrid-bm25")
HYBRID = "hybrid-bm25"
FIGURE_BUDGET_METRICS: tuple[str, ...] = ("gold_recall", "full_support", "precision")
MISSING = None


@dataclass(frozen=True)
class Check:
    """One reproduction check: the historical value, Phase 6's value, and whether they are equal."""

    name: str
    expected: Any
    observed: Any

    @property
    def passed(self) -> bool:
        return bool(self.expected == self.observed)

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected": self.expected,
            "observed": self.observed,
            "passed": self.passed,
        }


def _get(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    node: Any = mapping
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return MISSING
        node = node[key]
    return node


def fit_checks(frozen: FusionFit, refit: FusionFit, label: str = "control") -> list[Check]:
    """The re-executed fit against the frozen one, point by point, and the selection derived."""
    checks = [
        Check(f"{label} components", list(frozen.components), list(refit.components)),
        Check(f"{label} metric", frozen.metric, refit.metric),
        Check(f"{label} budget", frozen.budget, refit.budget),
        Check(f"{label} grid", list(frozen.grid), list(refit.grid)),
        Check(f"{label} n_questions", frozen.n_questions, refit.n_questions),
    ]
    for weight in frozen.grid:
        checks.append(
            Check(f"{label} curve w={weight:.1f}", frozen.curve[weight], refit.curve.get(weight))
        )
    checks.extend(
        [
            Check(f"{label} rrf_score", frozen.rrf_score, refit.rrf_score),
            Check(f"{label} winning_scheme", frozen.winning_scheme, refit.winning_scheme),
            Check(f"{label} weights", frozen.weights, refit.weights),
        ]
    )
    return checks


def config_checks(
    historical_configs: Mapping[str, Mapping[str, Any]],
    observed_configs: Mapping[str, Mapping[str, Any]],
) -> list[Check]:
    """HU-2's keys of each historical file (`system/split`) against Phase 6's run of that system."""
    checks: list[Check] = []
    for label in sorted(historical_configs):
        system = label.split("/", maxsplit=1)[0]
        keys = list(config.HISTORICAL_CONFIG_KEYS)
        if system == HYBRID:
            keys.extend(config.HISTORICAL_HYBRID_CONFIG_KEYS)
        for key in keys:
            checks.append(
                Check(
                    f"{label} config {key}",
                    _get(historical_configs[label], key),
                    _get(observed_configs.get(system), key),
                )
            )
    return checks


def figure_checks(
    label: str, observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[Check]:
    """The spec's figure list for one system and split. `observed` and `expected` hold
    `metrics` and `cost` as the harness writes them; latency is not read."""
    checks: list[Check] = []
    for budget in config.CONTEXT_BUDGETS:
        for metric in FIGURE_BUDGET_METRICS:
            key = f"budget_{budget}"
            checks.append(
                Check(
                    f"{label} {key} {metric}",
                    _get(expected, "metrics", key, metric),
                    _get(observed, "metrics", key, metric),
                )
            )
    for k in config.RECALL_AT_K:
        key = f"recall_at_{k}"
        checks.append(
            Check(f"{label} {key}", _get(expected, "metrics", key), _get(observed, "metrics", key))
        )
    checks.append(
        Check(
            f"{label} mean_units_included",
            _get(expected, "cost", "mean_units_included"),
            _get(observed, "cost", "mean_units_included"),
        )
    )
    return checks


def control_mode(checks: Sequence[Check]) -> str:
    """`reused` if and only if every check passed; a mode decided on no check is refused."""
    if not checks:
        raise ValueError("the control mode cannot be decided on no check")
    return REUSED if all(check.passed for check in checks) else REMEASURED


@dataclass(frozen=True)
class ReproductionReport:
    """The dev reproduction: every check, and the mode they decide."""

    checks: tuple[Check, ...]

    @property
    def mode(self) -> str:
        return control_mode(self.checks)

    @property
    def passed(self) -> bool:
        return self.mode == REUSED

    def failed(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]

    def as_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failed": len(self.failed()),
            "checks": [check.as_payload() for check in self.checks],
        }


def dev_reproduction(
    *,
    frozen: FusionFit,
    refit: FusionFit,
    historical_dev: Mapping[str, Mapping[str, Any]],
    observed_dev: Mapping[str, Mapping[str, Any]],
    historical_configs: Mapping[str, Mapping[str, Any]],
    observed_configs: Mapping[str, Mapping[str, Any]],
) -> ReproductionReport:
    """Every dev check of D12, in a fixed order: fit, configurations, figures."""
    checks = fit_checks(frozen, refit)
    checks.extend(config_checks(historical_configs, observed_configs))
    for system in CONTROL_SYSTEMS:
        checks.extend(
            figure_checks(
                f"{system}/{DEV_SPLIT}",
                observed_dev.get(system, {}),
                historical_dev.get(system, {}),
            )
        )
    return ReproductionReport(checks=tuple(checks))


@dataclass(frozen=True)
class HeldOutReproduction:
    """The test-side comparison of `dense` and A against their historical test figures."""

    mode: str
    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def descriptive(self) -> bool:
        """In `re-measured` mode the incompatibility is already recorded; this decides nothing."""
        return self.mode == REMEASURED

    @property
    def stops_run(self) -> bool:
        """In `reused` mode a failure stops the run before any B test figure exists."""
        return self.mode == REUSED and not self.passed

    def as_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "passed": self.passed,
            "descriptive": self.descriptive,
            "stops_run": self.stops_run,
            "checks": [check.as_payload() for check in self.checks],
        }


def compare_test_reproduction(
    observed_test: Mapping[str, Mapping[str, Any]],
    historical_test: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
) -> HeldOutReproduction:
    """The same figure list on test. Called only by the test stage, at step 5 of D14."""
    if mode not in MODES:
        raise ValueError(f"unknown control mode {mode!r}, expected one of {MODES}")
    split = decision_parameters.DECISION_SPLIT
    checks: list[Check] = []
    for system in CONTROL_SYSTEMS:
        checks.extend(
            figure_checks(
                f"{system}/{split}",
                observed_test.get(system, {}),
                historical_test.get(system, {}),
            )
        )
    return HeldOutReproduction(mode=mode, checks=tuple(checks))
