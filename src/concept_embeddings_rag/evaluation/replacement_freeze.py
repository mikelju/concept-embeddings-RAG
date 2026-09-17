"""`Phase6Freeze`: everything decided on dev, written once before test is read (D13).

The freeze is the observable form of "the test configuration is frozen before the first test
question is evaluated". It records:

- `protocol`: pool hash, manifest identity, split sizes and question-set hashes, embedding model
  and revision, tokenizer and token-count hash, budgets, `top_k`, recall depths, metrics, seed,
  code version;
- `control`: the mode (`reused` or `re-measured`, with its deviation document), every dev
  reproduction check, the identity of `selection.json` and of the four historical result files
  (name, sha256, configuration - never a figure), the historical selection as `history`, and the
  Phase 6 fit of `[dense, bm25]` with its derived selection and every point's dev metrics;
- `entity_component`: the node index, extraction and prompt digests, normalization version,
  types, weight rule, read depth, maximum depth and the order `D(q)` is taken in;
- `continuity`: the Phase 5 digests and checks (a)-(d) with their counts and outcomes;
- `entity_fit`: the fit of `[dense, entity-hop]`, derived selection and per-point dev metrics;
- `dev_results`: the dev tables of dense, A and B's selected configuration, their outcome digests,
  and the reproducibility block (D19);
- `decisions` / `n_dev_decisions`: the ledger, by Phase 3's rule;
- `decision_parameters` and `qualitative_sample_rule`, copied from the spec's constants;
- `seed`, `code_version`, `evaluated_on = "dev"`, `frozen_at`; and, by OI-4, `supersedes` and
  `deviation`, both null in the first freeze.

**No field can hold a test figure by construction**: the builder accepts results and outcomes
whose split is dev and refuses anything else, and refuses a historical identity that carries a
`metrics` or `cost` key anywhere. The loader checks the split claim before the digest, then the
decision parameters against the spec's, the selections against their curves, the ledger
against the rule, the reused control against its history, and the supersession chain.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag import decision_parameters as dp
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.evaluation.continuity import ContinuityReport
from concept_embeddings_rag.evaluation.control_reproduction import (
    MODES,
    REMEASURED,
    REUSED,
    ReproductionReport,
)
from concept_embeddings_rag.evaluation.harness import RunResult
from concept_embeddings_rag.evaluation.outcomes import PerQuestionOutcomes
from concept_embeddings_rag.evaluation.replacement_decision import (
    DecisionError,
    check_parameters,
    parameters_payload,
)
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    FusionFit,
    digest_of_payload,
    fusion_decisions,
    serialize_payload,
)
from concept_embeddings_rag.retrieval.fusion import RRF, FusedRetriever

FREEZE_FILENAME = "freeze.json"
FREEZE_PATTERN = re.compile(r"^freeze(?:-(\d+))?\.json$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CONTROL, ENTITY = "hybrid-bm25", "hybrid-entity-hop"
SYSTEMS: tuple[str, ...] = ("dense", CONTROL, ENTITY)
HISTORICAL_LABELS: tuple[str, ...] = (
    "dense/dev",
    "dense/test",
    "hybrid-bm25/dev",
    "hybrid-bm25/test",
)
IDENTITY_KEYS = frozenset({"name", "sha256", "system", "split", "created_at", "config"})
FIGURE_KEYS: tuple[str, ...] = ("metrics", "cost")
REQUIRED_PROTOCOL: tuple[str, ...] = (
    "unit_set_hash",
    "manifest",
    "split_sizes",
    "question_set_hashes",
    "model",
    "revision",
    "tokenizer",
    "token_counts_sha256",
    "budgets",
    "top_k",
    "recall_depths",
    "metrics",
    "seed",
    "code_version",
)
ENTITY_COMPONENT_KEYS: tuple[str, ...] = (
    "node_index_digest",
    "extraction_digest",
    "prompt_digest",
    "normalization_version",
    "types",
    "weight_rule",
    "read_depth",
    "max_depth",
    "dense_order",
)
CONTINUITY_DIGESTS: tuple[str, ...] = ("pilot_digest", "hop_run_digest", "traces_digest")


class FreezeError(Exception):
    """The freeze cannot be built from what it was handed, or does not verify on load."""


@dataclass(frozen=True)
class MeasuredFit:
    """A fit and the dev result of every point it measured, keyed by `point_label`."""

    fit: FusionFit
    points: Mapping[str, RunResult]


@dataclass(frozen=True)
class Phase6Freeze:
    """A built or loaded freeze. Satisfies `FrozenReport`, so `check_freeze_precedes` serves it."""

    payload: dict[str, Any]
    digest: str
    path: Path | None = None

    @property
    def frozen_at(self) -> str:
        return str(self.payload["frozen_at"])

    @property
    def control_mode(self) -> str:
        return str(self.payload["control"]["mode"])


def point_label(hybrid: FusedRetriever) -> str:
    """`rrf`, or `w=<dense weight>` to one decimal: how a fit's measurements are keyed."""
    if hybrid.scheme == RRF or hybrid.weights is None:
        return "rrf"
    return f"w={hybrid.weights[hybrid.components[0]]:.1f}"


def expected_labels(grid: Sequence[float]) -> set[str]:
    return {"rrf", *(f"w={float(weight):.1f}" for weight in grid)}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else config.PROJECT_ROOT / candidate


def _deviation_exists(path: str | None) -> bool:
    return path is not None and _resolve(path).is_file()


def _keys_named(value: Any, names: Sequence[str]) -> list[str]:
    """Every key in a nested JSON value that is one of `names`."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, inner in value.items():
            if key in names:
                found.append(str(key))
            found.extend(_keys_named(inner, names))
    elif isinstance(value, list | tuple):
        for inner in value:
            found.extend(_keys_named(inner, names))
    return found


def _require_dev(what: str, split: str) -> None:
    if split != DEV_SPLIT:
        raise FreezeError(f"{what} was measured on {split!r}; the freeze accepts dev results only")


# --- Building ---------------------------------------------------------------------------------


def _fit_fields(fit: FusionFit) -> dict[str, Any]:
    return {
        "components": list(fit.components),
        "metric": fit.metric,
        "budget": fit.budget,
        "n_questions": fit.n_questions,
        "grid": [float(weight) for weight in fit.grid],
        "curve": {repr(float(weight)): value for weight, value in fit.curve.items()},
        "rrf_score": fit.rrf_score,
        "winning_scheme": fit.winning_scheme,
        "best_weight": fit.best_weight,
        "weights": fit.weights,
    }


def _measured_payload(measured: MeasuredFit, components: tuple[str, str], system: str) -> dict:
    fit = measured.fit
    if tuple(fit.components) != components:
        raise FreezeError(f"the fit of {system} is over {fit.components}, not {components}")
    if set(measured.points) != expected_labels(fit.grid):
        raise FreezeError(
            f"the fit of {system} keeps points {sorted(measured.points)}, not RRF and every weight"
        )
    points: dict[str, Any] = {}
    for label, result in sorted(measured.points.items()):
        _require_dev(f"point {label} of {system}", result.split)
        if result.system != system:
            raise FreezeError(f"point {label} of {system} is a result of {result.system}")
        points[label] = {
            "split": result.split,
            "system": result.system,
            "fusion_scheme": result.config.get("fusion_scheme"),
            "fusion_weights": result.config.get("fusion_weights"),
            "metrics": result.metrics,
            "cost": result.cost,
            "created_at": result.created_at,
        }
    return {**_fit_fields(fit), "points": points}


def _ledger(fit: FusionFit, hybrid: str) -> list[dict[str, Any]]:
    return [
        {
            "hybrid": hybrid,
            "name": decision.name,
            "chosen": decision.chosen,
            "alternatives": list(decision.alternatives),
            "runner_up": decision.runner_up,
            "margin": decision.margin,
        }
        for decision in fusion_decisions(fit)
    ]


def reproducibility_block(
    first: Mapping[str, PerQuestionOutcomes], second: Mapping[str, PerQuestionOutcomes]
) -> dict[str, Any]:
    """D19: a second dev run over unchanged inputs, compared question by question."""
    if set(first) != set(second):
        raise FreezeError("the two runs cover different systems")
    systems: dict[str, Any] = {}
    for system in sorted(first):
        ours, theirs = first[system].by_qid(), second[system].by_qid()
        differing = sorted(
            qid for qid in set(ours) | set(theirs) if ours.get(qid) != theirs.get(qid)
        )
        identical = not differing and first[system].aggregates == second[system].aggregates
        systems[system] = {
            "identical": identical,
            "n_questions": len(ours),
            "differing_questions": len(differing),
            "examples": differing[:5],
            "first_digest": first[system].digest,
            "second_digest": second[system].digest,
        }
    return {
        "compared": "every per-question outcome and aggregate, latency excepted",
        "systems": systems,
        "passed": all(entry["identical"] for entry in systems.values()),
    }


def build_freeze(
    *,
    protocol: Mapping[str, Any],
    control_mode: str,
    reproduction: ReproductionReport,
    selection: Mapping[str, Any],
    historical: Mapping[str, Mapping[str, Any]],
    history: FusionFit,
    control: MeasuredFit,
    entity_component: Mapping[str, Any],
    continuity: ContinuityReport,
    continuity_digests: Mapping[str, str],
    entity: MeasuredFit,
    dev_results: Mapping[str, tuple[RunResult, PerQuestionOutcomes]],
    reproducibility: Mapping[str, Any],
    seed: int,
    code_version: str,
    control_deviation: str | None = None,
    supersedes: str | None = None,
    deviation: str | None = None,
    frozen_at: str | None = None,
) -> Phase6Freeze:
    """Assemble the freeze from dev evidence only, refusing anything the contract forbids."""
    missing = [key for key in REQUIRED_PROTOCOL if key not in protocol]
    if missing:
        raise FreezeError(f"the protocol block lacks {missing}")
    if protocol["top_k"] != config.EVALUATION_TOP_K:
        raise FreezeError(f"top_k is {protocol['top_k']}, not {config.EVALUATION_TOP_K}")
    if list(protocol["budgets"]) != list(config.CONTEXT_BUDGETS):
        raise FreezeError(f"the budgets are {protocol['budgets']}, not {config.CONTEXT_BUDGETS}")
    if protocol["seed"] != seed:
        raise FreezeError("the protocol's seed is not the freeze's seed")

    if control_mode not in MODES:
        raise FreezeError(f"unknown control mode {control_mode!r}")
    if control_mode != reproduction.mode:
        raise FreezeError(
            f"the control mode {control_mode!r} disagrees with the dev checks, which give "
            f"{reproduction.mode!r}"
        )
    if control_mode == REMEASURED and not _deviation_exists(control_deviation):
        raise FreezeError(
            "a re-measured control needs an existing deviation document recording the mismatch"
        )

    figures = _keys_named(historical, FIGURE_KEYS)
    if figures:
        raise FreezeError(f"the historical identities carry figures ({sorted(set(figures))})")
    if sorted(historical) != sorted(HISTORICAL_LABELS):
        raise FreezeError(f"the historical identities are {sorted(historical)}")
    for label, identity in historical.items():
        if set(identity) != IDENTITY_KEYS:
            raise FreezeError(f"historical identity {label} holds {sorted(identity)}")
    if set(selection) != {"digest", "frozen_at"}:
        raise FreezeError("the selection identity is its digest and frozen_at")

    control_block = _measured_payload(control, ("dense", "bm25"), CONTROL)
    history_block = _fit_fields(history)
    if control_mode == REUSED:
        compared = ("components", "grid", "curve", "rrf_score", "winning_scheme", "weights")
        if any(control_block[key] != history_block[key] for key in compared):
            raise FreezeError("a reused control's fit must equal the history it reproduces")

    missing_component = [key for key in ENTITY_COMPONENT_KEYS if key not in entity_component]
    if missing_component:
        raise FreezeError(f"the entity component lacks {missing_component}")
    if list(entity_component["types"]) != list(config.ENTITY_HOP_TYPES):
        raise FreezeError(f"the entity component's types are {entity_component['types']}")
    if entity_component["read_depth"] != config.PILOT_READ_DEPTH:
        raise FreezeError("the entity component's read depth is not the declared one")
    if entity_component["max_depth"] != config.ENTITY_HOP_MAX_DEPTH:
        raise FreezeError("the entity component's maximum depth is not the declared one")

    if not continuity.passed:
        raise FreezeError("continuity with Phase 5 failed; no freeze is written after that stop")
    if set(continuity_digests) != set(CONTINUITY_DIGESTS):
        raise FreezeError(f"the continuity digests are {sorted(continuity_digests)}")

    entity_block = _measured_payload(entity, ("dense", config.ENTITY_HOP_NAME), ENTITY)

    if set(dev_results) != set(SYSTEMS):
        raise FreezeError(f"the dev results cover {sorted(dev_results)}, not {SYSTEMS}")
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        result, outcomes = dev_results[system]
        _require_dev(f"the {system} result", result.split)
        _require_dev(f"the {system} outcomes", outcomes.split)
        if result.system != system or outcomes.system != system:
            raise FreezeError(f"the dev result filed under {system} is not {system}'s")
        systems[system] = {
            "split": result.split,
            "system": system,
            "created_at": result.created_at,
            "config": result.config,
            "metrics": result.metrics,
            "cost": result.cost,
            "outcomes_digest": outcomes.digest,
            "outcomes_file": outcomes.path.name,
        }
    if reproducibility.get("passed") is not True:
        raise FreezeError(
            "the dev reproducibility check did not pass; the source of nondeterminism is recorded "
            "and the phase stops"
        )

    decisions = _ledger(entity.fit, ENTITY)
    if control_mode == REMEASURED:
        decisions.extend(_ledger(control.fit, CONTROL))

    payload: dict[str, Any] = {
        "protocol": dict(protocol),
        "control": {
            "mode": control_mode,
            "deviation": control_deviation if control_mode == REMEASURED else None,
            "checks": [check.as_payload() for check in reproduction.checks],
            "n_failed_checks": len(reproduction.failed()),
            "selection": dict(selection),
            "historical_files": {label: dict(identity) for label, identity in historical.items()},
            "history": history_block,
            "fit": control_block,
        },
        "entity_component": dict(entity_component),
        "continuity": {
            **dict(continuity_digests),
            "checks": continuity.as_payload()["checks"],
            "passed": continuity.passed,
        },
        "entity_fit": entity_block,
        "dev_results": {"systems": systems, "reproducibility": dict(reproducibility)},
        "decisions": decisions,
        "n_dev_decisions": len(decisions),
        "decision_parameters": parameters_payload(),
        "qualitative_sample_rule": dict(dp.QUALITATIVE_SAMPLE_RULE),
        "seed": seed,
        "code_version": code_version,
        "evaluated_on": DEV_SPLIT,
        "frozen_at": frozen_at or _now(),
        "supersedes": supersedes,
        "deviation": deviation,
    }
    payload = json.loads(serialize_payload(payload))
    return Phase6Freeze(payload=payload, digest=digest_of_payload(payload))


# --- Writing and loading ----------------------------------------------------------------------


def _freeze_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    indexed: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = FREEZE_PATTERN.match(path.name)
        if match:
            indexed.append((int(match.group(1) or 1), path))
    indexed.sort()
    if [index for index, _ in indexed] != list(range(1, len(indexed) + 1)):
        raise FreezeError(f"the freezes in {directory} are not numbered 1, 2, ...: {indexed}")
    return [path for _, path in indexed]


def _freeze_path(directory: Path, index: int) -> Path:
    return directory / (FREEZE_FILENAME if index == 1 else f"freeze-{index}.json")


def save_freeze(freeze: Phase6Freeze, directory: Path | str) -> Path:
    """Write the first freeze once; a later one only as a recorded supersession (OI-4)."""
    directory = Path(directory)
    existing = _freeze_files(directory)
    supersedes, deviation = freeze.payload.get("supersedes"), freeze.payload.get("deviation")
    if not existing:
        if supersedes is not None or deviation is not None:
            raise FreezeError("the first freeze supersedes nothing and names no deviation")
        path = _freeze_path(directory, 1)
    else:
        latest = load_freeze(directory)
        if supersedes != latest.digest:
            raise FreezeError(
                f"a freeze is already written at {latest.path}; a second one must supersede its "
                "digest and name an existing deviation document"
            )
        if not _deviation_exists(deviation):
            raise FreezeError("a superseding freeze must name an existing deviation document")
        path = _freeze_path(directory, len(existing) + 1)
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FreezeError(f"{path} already exists; a freeze is never overwritten")
    sealed = {**freeze.payload, "digest": freeze.digest}
    write_text_atomic(path, serialize_payload(sealed))
    return path


def _refit(block: Mapping[str, Any]) -> FusionFit:
    dense_name, second_name = (str(name) for name in block["components"])
    return FusionFit(
        components=(dense_name, second_name),
        metric=str(block["metric"]),
        budget=int(block["budget"]),
        n_questions=int(block["n_questions"]),
        grid=tuple(float(weight) for weight in block["grid"]),
        curve={float(weight): float(value) for weight, value in block["curve"].items()},
        rrf_score=float(block["rrf_score"]),
    )


def _verify_file(path: Path) -> Phase6Freeze:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluated_on") != DEV_SPLIT:
        raise FreezeError(
            f"{path.name} claims {payload.get('evaluated_on')!r}; a freeze is dev only"
        )
    recorded = payload.pop("digest", None)
    if recorded is None or digest_of_payload(payload) != recorded:
        raise FreezeError(f"{path.name} does not match its digest; it has been modified")
    try:
        check_parameters(payload["decision_parameters"])
    except DecisionError as error:
        raise FreezeError(f"{path.name}: {error}") from error
    if payload["qualitative_sample_rule"] != json.loads(json.dumps(dp.QUALITATIVE_SAMPLE_RULE)):
        raise FreezeError(f"{path.name}: the qualitative sample rule differs from the spec's")

    fits = {CONTROL: payload["control"]["fit"], ENTITY: payload["entity_fit"]}
    for system, block in fits.items():
        fit = _refit(block)
        derived = {
            "winning_scheme": fit.winning_scheme,
            "best_weight": fit.best_weight,
            "weights": fit.weights,
        }
        if any(block[key] != value for key, value in derived.items()):
            raise FreezeError(
                f"{path.name}: the {system} selection does not re-derive from its curve"
            )
        for label, point in block["points"].items():
            _require_dev(f"{path.name}: point {label} of {system}", str(point["split"]))
    for system, entry in payload["dev_results"]["systems"].items():
        _require_dev(f"{path.name}: the {system} dev result", str(entry["split"]))

    mode = payload["control"]["mode"]
    expected = _ledger(_refit(fits[ENTITY]), ENTITY)
    if mode == REMEASURED:
        expected.extend(_ledger(_refit(fits[CONTROL]), CONTROL))
    if payload["decisions"] != expected or payload["n_dev_decisions"] != len(expected):
        raise FreezeError(f"{path.name}: the decisions do not follow the Phase 3 counting rule")
    if mode == REUSED:
        compared = ("components", "grid", "curve", "rrf_score", "winning_scheme", "weights")
        history = payload["control"]["history"]
        if any(fits[CONTROL][key] != history[key] for key in compared):
            raise FreezeError(f"{path.name}: a reused control's fit differs from its history")
    elif not _deviation_exists(payload["control"]["deviation"]):
        raise FreezeError(f"{path.name}: the re-measured control's deviation document is missing")
    if payload["continuity"]["passed"] is not True:
        raise FreezeError(f"{path.name}: continuity did not pass")
    if payload["dev_results"]["reproducibility"]["passed"] is not True:
        raise FreezeError(f"{path.name}: the dev reproducibility check did not pass")
    return Phase6Freeze(payload=payload, digest=str(recorded), path=path)


def load_freeze(directory: Path | str) -> Phase6Freeze:
    """The valid freeze: the last of the chain, each link verified and superseding the last."""
    directory = Path(directory)
    files = _freeze_files(directory)
    if not files:
        raise FreezeError(f"no freeze in {directory}: run `cer replace-freeze` first")
    previous: Phase6Freeze | None = None
    for path in files:
        freeze = _verify_file(path)
        supersedes, deviation = freeze.payload.get("supersedes"), freeze.payload.get("deviation")
        if previous is None:
            if supersedes is not None or deviation is not None:
                raise FreezeError(f"{path.name} opens the chain but claims to supersede a freeze")
        else:
            if supersedes != previous.digest:
                raise FreezeError(f"{path.name} breaks the supersession chain")
            if not _deviation_exists(deviation):
                raise FreezeError(f"{path.name} supersedes without an existing deviation document")
        previous = freeze
    if previous is None:  # pragma: no cover - files is not empty
        raise FreezeError(f"no freeze in {directory}")
    return previous
