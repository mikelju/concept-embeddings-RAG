"""`PerQuestionOutcomes`: one system's per-question readings on one split (Phase 6, D9).

The paired statistics of Phase 6 are computed from these files and from nothing else, so each
one is bound to the harness run that produced it:

- `questions` holds one entry per question of the split, sorted by question id: per budget the
  gold recall, Full Support, precision and units included, and recall at every depth.
- `aggregates` holds the means recomputed from those entries. They are taken in the **pool's
  split order**, the order the harness measured in (D7), with the harness's own `_mean`, and
  they must equal the named run result's `metrics` and `cost.mean_units_included` exactly.
- the qid set must be the split's whole population: nothing sampled, nothing foreign.
- a test outcome names the freeze it was measured under; a dev outcome exists before any
  freeze and names none.

Written once, atomically, with a digest over the serialization; a second file for the same
system and split takes a numbered suffix and never replaces the first. Nothing on the
retrieval path reads this artifact.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config, decision_parameters
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.evaluation.harness import (
    SAFE_NAME,
    QuestionOutcome,
    RunResult,
    _mean,
)
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    digest_of_payload,
    serialize_payload,
)

OUTCOMES_PREFIX = "outcomes"
BUDGET_FIELDS: tuple[str, ...] = ("gold_recall", "full_support", "precision", "units_included")
METRIC_FIELDS: tuple[str, ...] = ("gold_recall", "full_support", "precision")
SPLITS: tuple[str, ...] = (DEV_SPLIT, decision_parameters.DECISION_SPLIT)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_RUN_FILE = re.compile(r"^run-[A-Za-z0-9_+-]+\.json$")


class OutcomesError(Exception):
    """A per-question outcomes artifact is incomplete, modified, or bound to another run."""


@dataclass(frozen=True)
class PerQuestionOutcomes:
    """A verified outcomes artifact, as the loader hands it on."""

    system: str
    split: str
    questions: tuple[dict[str, Any], ...]
    aggregates: dict[str, Any]
    run_result: dict[str, Any]
    freeze_digest: str | None
    digest: str
    path: Path

    def by_qid(self) -> dict[str, dict[str, Any]]:
        return {str(entry["qid"]): entry for entry in self.questions}


def _budget_key(budget: int) -> str:
    return f"budget_{budget}"


def _recall_key(k: int) -> str:
    return f"recall_at_{k}"


def _shape_of(metrics: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """The budgets and recall depths a harness result was measured at, in ascending order."""
    budgets = sorted(
        int(key.removeprefix("budget_")) for key in metrics if key.startswith("budget_")
    )
    ks = sorted(
        int(key.removeprefix("recall_at_")) for key in metrics if key.startswith("recall_at_")
    )
    if not budgets:
        raise OutcomesError("the run result carries no budget; there is nothing to record")
    return budgets, ks


def _check_identity(system: str, split: str, freeze_digest: str | None) -> None:
    if system not in config.PHASE_6_SYSTEMS:
        raise OutcomesError(
            f"system {system!r} is not one of this phase's systems {config.PHASE_6_SYSTEMS}"
        )
    if split not in SPLITS:
        raise OutcomesError(f"split {split!r} is not one of {SPLITS}")
    if split == DEV_SPLIT:
        if freeze_digest is not None:
            raise OutcomesError(
                "a dev outcome exists before the freeze and names no freeze digest; the freeze "
                "references the dev outcomes instead"
            )
    elif freeze_digest is None or not SHA256.match(str(freeze_digest)):
        raise OutcomesError(
            f"a {split} outcome must name the freeze it was measured under as a sha256 digest, "
            f"not {freeze_digest!r}"
        )


def _check_population(qids: Sequence[str], split_order: Sequence[str]) -> None:
    if len(set(split_order)) != len(split_order):
        raise OutcomesError("the split's population repeats a question id")
    if len(set(qids)) != len(qids):
        raise OutcomesError("the outcomes repeat a question id")
    missing = sorted(set(split_order) - set(qids))
    foreign = sorted(set(qids) - set(split_order))
    if missing or foreign:
        raise OutcomesError(
            f"the outcomes are not the split's full population: {len(missing)} question(s) "
            f"missing ({missing[:3]}), {len(foreign)} foreign ({foreign[:3]})"
        )


def _entry_of(record: QuestionOutcome, budgets: Sequence[int], ks: Sequence[int]) -> dict:
    entry: dict[str, Any] = {"qid": record.qid}
    for budget in budgets:
        reading = record.budgets.get(budget)
        if reading is None:
            raise OutcomesError(f"question {record.qid} has no reading at budget {budget}")
        entry[_budget_key(budget)] = {
            "gold_recall": reading.gold_recall,
            "full_support": reading.full_support,
            "precision": reading.precision,
            "units_included": reading.units_included,
        }
    for k in ks:
        if k not in record.recall_at_k:
            raise OutcomesError(f"question {record.qid} has no recall at {k}")
        entry[_recall_key(k)] = record.recall_at_k[k]
    return entry


def recompute_aggregates(
    entries: Sequence[Mapping[str, Any]],
    split_order: Sequence[str],
    budgets: Sequence[int],
    ks: Sequence[int],
) -> dict[str, Any]:
    """Every mean, in the pool's split order, with the harness's own arithmetic (D9)."""
    by_qid = {str(entry["qid"]): entry for entry in entries}
    ordered = [by_qid[qid] for qid in split_order]
    metrics: dict[str, Any] = {}
    for budget in budgets:
        key = _budget_key(budget)
        metrics[key] = {
            name: _mean([float(entry[key][name]) for entry in ordered]) for name in METRIC_FIELDS
        }
    for k in ks:
        metrics[_recall_key(k)] = _mean([float(entry[_recall_key(k)]) for entry in ordered])
    largest = _budget_key(max(budgets))
    units = _mean([float(entry[largest]["units_included"]) for entry in ordered])
    return {"metrics": metrics, "mean_units_included": units}


def _check_against_run(aggregates: Mapping[str, Any], metrics: Any, units: Any, label: str) -> None:
    if aggregates["metrics"] != metrics or aggregates["mean_units_included"] != units:
        raise OutcomesError(
            f"the aggregates recomputed from the records differ from the {label}; the records do "
            "not describe that measurement"
        )


def build_outcomes(
    result: RunResult,
    records: Sequence[QuestionOutcome],
    *,
    run_file: str,
    split_order: Sequence[str],
    freeze_digest: str | None,
) -> dict[str, Any]:
    """The payload of one outcomes artifact, refused unless it reproduces `result` exactly."""
    _check_identity(result.system, result.split, freeze_digest)
    budgets, ks = _shape_of(result.metrics)
    qids = [record.qid for record in records]
    _check_population(qids, split_order)

    entries = sorted((_entry_of(record, budgets, ks) for record in records), key=lambda e: e["qid"])
    aggregates = recompute_aggregates(entries, split_order, budgets, ks)
    _check_against_run(
        aggregates,
        result.metrics,
        result.cost.get("mean_units_included"),
        "harness's aggregates of the run result",
    )
    return {
        "system": result.system,
        "split": result.split,
        "questions": entries,
        "aggregates": aggregates,
        "run_result": {
            "file": run_file,
            "system": result.system,
            "split": result.split,
            "created_at": result.created_at,
        },
        "freeze_digest": freeze_digest,
    }


def outcomes_path(directory: Path | str, system: str, split: str, index: int = 1) -> Path:
    """`outcomes-{system}-{split}.json`, or `-{index}` for a later file of the same pair."""
    for label, value in (("system", system), ("split", split)):
        if not SAFE_NAME.match(value):
            raise OutcomesError(f"{label} {value!r} is not usable in a filename")
    suffix = "" if index == 1 else f"-{index}"
    return Path(directory) / f"{OUTCOMES_PREFIX}-{system}-{split}{suffix}.json"


def save_outcomes(payload: Mapping[str, Any], directory: Path | str) -> Path:
    """Write the artifact atomically, never over an existing one: a second file takes `-2`."""
    _check_identity(str(payload["system"]), str(payload["split"]), payload.get("freeze_digest"))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index = 1
    path = outcomes_path(directory, str(payload["system"]), str(payload["split"]), index)
    while path.exists():
        index += 1
        path = outcomes_path(directory, str(payload["system"]), str(payload["split"]), index)
    sealed = {key: value for key, value in payload.items() if key != "digest"}
    sealed["digest"] = digest_of_payload(sealed)
    write_text_atomic(path, serialize_payload(sealed))
    return path


def _read_run_result(run_dir: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    name = str(identity.get("file", ""))
    if not SAFE_RUN_FILE.match(name):
        raise OutcomesError(f"the named run result {name!r} is not a run result file name")
    path = run_dir / name
    if not path.exists():
        raise OutcomesError(f"the named run result {name} is not in {run_dir}")
    run: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for key in ("system", "split", "created_at"):
        if run.get(key) != identity.get(key):
            raise OutcomesError(
                f"the named run result {name} records {key} {run.get(key)!r}, and the outcomes "
                f"were bound to {identity.get(key)!r}"
            )
    return run


def load_outcomes(
    path: Path | str, *, split_order: Sequence[str], run_dir: Path | str
) -> PerQuestionOutcomes:
    """Read one artifact back, proving it is whole, unmodified and bound to its run."""
    path = Path(path)
    if not path.exists():
        raise OutcomesError(f"no outcomes artifact at {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("digest")
    if recorded is None or digest_of_payload(payload) != recorded:
        raise OutcomesError(f"{path.name} does not match its digest; it has been modified")

    system, split = str(payload["system"]), str(payload["split"])
    freeze_digest = payload.get("freeze_digest")
    _check_identity(system, split, freeze_digest)

    entries = list(payload["questions"])
    qids = [str(entry["qid"]) for entry in entries]
    if qids != sorted(qids):
        raise OutcomesError(f"{path.name} does not hold its questions sorted by question id")
    _check_population(qids, split_order)

    run = _read_run_result(Path(run_dir), payload["run_result"])
    if run.get("system") != system or run.get("split") != split:
        raise OutcomesError(f"{path.name} is bound to a run result of another system or split")
    budgets, ks = _shape_of(run["metrics"])
    for entry in entries:
        for budget in budgets:
            if set(entry.get(_budget_key(budget), {})) != set(BUDGET_FIELDS):
                raise OutcomesError(f"question {entry['qid']} is incomplete at budget {budget}")

    aggregates = recompute_aggregates(entries, split_order, budgets, ks)
    _check_against_run(
        aggregates,
        payload["aggregates"]["metrics"],
        payload["aggregates"]["mean_units_included"],
        "aggregates the file stores",
    )
    _check_against_run(
        aggregates,
        run["metrics"],
        run["cost"].get("mean_units_included"),
        f"named run result {run['system']}/{run['split']}",
    )
    return PerQuestionOutcomes(
        system=system,
        split=split,
        questions=tuple(entries),
        aggregates=dict(payload["aggregates"]),
        run_result=dict(payload["run_result"]),
        freeze_digest=None if freeze_digest is None else str(freeze_digest),
        digest=str(recorded),
        path=path,
    )
