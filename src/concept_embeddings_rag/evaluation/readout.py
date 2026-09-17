"""The descriptive readout of one split, and the qualitative sample on test (Phase 6, D18).

Everything the report needs that decides nothing, labelled as such and built from the outcome
artifacts read back from disk:

- Full Support, gold recall and precision per system and budget, and recall at every depth;
- the three paired 2x2 tables (B-A, B-dense, A-dense: both, each only, neither) at every budget;
- the paired gold-recall differences at the decision budget;
- the failure analysis of B and of A against dense at the decision budget, by `CrossTab`: the
  question ids each recovers and loses;
- the bridge-like subgroup - questions with at least one gold paragraph outside `read(q)` - with
  its Full Support per system, compared on dev with the pilot's question ids;
- cost per system: mean latency (descriptive, not deterministic) and mean units at the largest
  budget;
- on the decision split only, the qualitative sample fixed in the freeze: four groups at the
  decision budget, the first `per_group` question ids of each in ascending order.

Gold is read here, in `evaluation/`, and never on the retrieval path. Nothing on the decision
path imports this module.
"""

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag import decision_parameters as dp
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.failure_analysis import cross_tabulate
from concept_embeddings_rag.evaluation.harness import SAFE_NAME
from concept_embeddings_rag.evaluation.outcomes import PerQuestionOutcomes
from concept_embeddings_rag.evaluation.selection import digest_of_payload, serialize_payload

DENSE, SYSTEM_A, SYSTEM_B = "dense", "hybrid-bm25", "hybrid-entity-hop"
SYSTEMS: tuple[str, ...] = (DENSE, SYSTEM_A, SYSTEM_B)
PAIRS: dict[str, tuple[str, str]] = {
    "b_vs_a": (SYSTEM_B, SYSTEM_A),
    "b_vs_dense": (SYSTEM_B, DENSE),
    "a_vs_dense": (SYSTEM_A, DENSE),
}
GROUPS: dict[str, tuple[str, str]] = dict(
    zip(
        dp.QUALITATIVE_SAMPLE_GROUPS,
        ((SYSTEM_B, SYSTEM_A), (SYSTEM_A, SYSTEM_B), (SYSTEM_B, DENSE), (DENSE, SYSTEM_B)),
        strict=True,
    )
)


class ReadoutError(Exception):
    """The readout cannot be built from what it was handed, or does not verify on load."""


def _successes(artifact: PerQuestionOutcomes, budget: int) -> dict[str, bool]:
    return {
        str(entry["qid"]): float(entry[f"budget_{budget}"]["full_support"]) == 1.0
        for entry in artifact.questions
    }


def _gold_recall(artifact: PerQuestionOutcomes, budget: int) -> dict[str, float]:
    return {
        str(entry["qid"]): float(entry[f"budget_{budget}"]["gold_recall"])
        for entry in artifact.questions
    }


def _budgets(artifact: PerQuestionOutcomes) -> list[int]:
    metrics = artifact.aggregates["metrics"]
    return sorted(int(key.removeprefix("budget_")) for key in metrics if key.startswith("budget_"))


def sample_groups(
    successes: Mapping[str, Mapping[str, bool]], *, per_group: int
) -> dict[str, dict[str, Any]]:
    """The four groups of HU-8, each with its size and its first `per_group` qids, ascending."""
    groups: dict[str, dict[str, Any]] = {}
    for label, (winner, loser) in GROUPS.items():
        members = sorted(
            qid for qid in successes[winner] if successes[winner][qid] and not successes[loser][qid]
        )
        groups[label] = {"size": len(members), "qids": members[:per_group]}
    return groups


def _differences(first: Mapping[str, float], second: Mapping[str, float]) -> dict[str, Any]:
    values = [first[qid] - second[qid] for qid in sorted(first)]
    distribution: dict[str, int] = {}
    for value in values:
        distribution[repr(value)] = distribution.get(repr(value), 0) + 1
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "positive": sum(1 for value in values if value > 0),
        "negative": sum(1 for value in values if value < 0),
        "zero": sum(1 for value in values if value == 0),
        "distribution": dict(sorted(distribution.items())),
    }


def build_readout(
    *,
    split: str,
    outcomes: Mapping[str, PerQuestionOutcomes],
    questions: Sequence[Question],
    read_lists: Mapping[str, Sequence[str]],
    costs: Mapping[str, Mapping[str, Any]],
    freeze: Mapping[str, Any],
    pilot_qids: Collection[str] | None = None,
) -> dict[str, Any]:
    """The readout payload for one split. Descriptive only."""
    if not SAFE_NAME.match(split):
        raise ReadoutError(f"split {split!r} is not usable in a filename")
    if set(outcomes) != set(SYSTEMS):
        raise ReadoutError(f"the readout needs the outcomes of {SYSTEMS}")
    for system, artifact in outcomes.items():
        if artifact.system != system or artifact.split != split:
            raise ReadoutError(f"the outcomes filed under {system} are not {system}'s on {split}")
    population = {str(entry["qid"]) for entry in outcomes[DENSE].questions}
    by_qid = {question.qid: question for question in questions}
    if set(by_qid) != population:
        raise ReadoutError("the questions handed in are not the outcomes' population")
    missing_reads = sorted(population - set(read_lists))
    if missing_reads:
        raise ReadoutError(
            f"no read list for {len(missing_reads)} question(s), e.g. {missing_reads[:3]}"
        )

    rule = freeze["qualitative_sample_rule"]
    budget = int(rule["budget"])
    budgets = _budgets(outcomes[DENSE])

    systems = {system: dict(outcomes[system].aggregates["metrics"]) for system in SYSTEMS}

    paired: dict[str, dict[str, dict[str, int]]] = {}
    for each in budgets:
        successes = {system: _successes(outcomes[system], each) for system in SYSTEMS}
        paired[str(each)] = {}
        for name, (first, second) in PAIRS.items():
            table = cross_tabulate(
                successes[first], successes[second], names=(first, second), budget=each
            )
            paired[str(each)][name] = {
                "both": len(table.both),
                "first_only": len(table.only_a),
                "second_only": len(table.only_b),
                "neither": len(table.neither),
            }

    at_budget = {system: _successes(outcomes[system], budget) for system in SYSTEMS}
    recall = {system: _gold_recall(outcomes[system], budget) for system in SYSTEMS}
    differences = {
        name: _differences(recall[first], recall[second]) for name, (first, second) in PAIRS.items()
    }

    failure_analysis: dict[str, dict[str, Any]] = {}
    for system in (SYSTEM_B, SYSTEM_A):
        table = cross_tabulate(
            at_budget[system], at_budget[DENSE], names=(system, DENSE), budget=budget
        )
        failure_analysis[system] = {
            "against": DENSE,
            "budget": budget,
            "recovered": list(table.only_a),
            "lost": list(table.only_b),
            "counts": table.counts,
        }

    subgroup_qids = sorted(
        qid
        for qid in population
        if any(gold not in set(read_lists[qid]) for gold in by_qid[qid].gold_unit_ids)
    )
    comparison = None
    if pilot_qids is not None:
        pilot = set(pilot_qids)
        comparison = {
            "equal": set(subgroup_qids) == pilot,
            "only_subgroup": sorted(set(subgroup_qids) - pilot),
            "only_pilot": sorted(pilot - set(subgroup_qids)),
        }
    subgroup = {
        "rule": "at least one gold paragraph outside read(q), the first 10 units of D(q)",
        "n_questions": len(subgroup_qids),
        "qids": subgroup_qids,
        "full_support": {
            system: sum(1 for qid in subgroup_qids if at_budget[system][qid]) for system in SYSTEMS
        },
        "pilot_comparison": comparison,
    }

    sample = None
    if split == rule["split"]:
        sample = {
            "rule": json.loads(json.dumps(rule)),
            "groups": sample_groups(at_budget, per_group=int(rule["per_group"])),
        }

    return {
        "split": split,
        "descriptive": True,
        "freeze_digest": str(freeze["digest"]),
        "outcome_digests": {system: outcomes[system].digest for system in SYSTEMS},
        "n_questions": len(population),
        "systems": systems,
        "paired": paired,
        "gold_recall_differences": differences,
        "failure_analysis": failure_analysis,
        "subgroup": subgroup,
        "cost": {
            system: {
                "mean_latency_ms": costs[system]["mean_latency_ms"],
                "mean_units_included": costs[system]["mean_units_included"],
            }
            for system in SYSTEMS
        },
        "largest_budget": max(config.CONTEXT_BUDGETS),
        "sample": sample,
    }


def readout_path(directory: Path | str, split: str) -> Path:
    if not SAFE_NAME.match(split):
        raise ReadoutError(f"split {split!r} is not usable in a filename")
    return Path(directory) / f"readout-{split}.json"


def save_readout(readout: Mapping[str, Any], directory: Path | str) -> Path:
    path = readout_path(directory, str(readout["split"]))
    if path.exists():
        raise ReadoutError(f"{path} already exists; the readout is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = {key: value for key, value in readout.items() if key != "digest"}
    sealed["digest"] = digest_of_payload(sealed)
    write_text_atomic(path, serialize_payload(sealed))
    return path


def load_readout(directory: Path | str, split: str, *, freeze_digest: str) -> dict[str, Any]:
    path = readout_path(directory, split)
    if not path.exists():
        raise ReadoutError(f"no readout at {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("digest") != digest_of_payload(payload):
        raise ReadoutError(f"{path.name} does not match its digest; it has been modified")
    if payload.get("freeze_digest") != freeze_digest:
        raise ReadoutError(f"{path.name} is bound to another freeze")
    return payload
