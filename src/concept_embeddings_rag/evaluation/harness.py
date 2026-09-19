"""The evaluation loop and its results.

The harness treats every retriever identically: it only knows the interface, so
it cannot favour one system over another. Results carry the full configuration
that produced them and are written append-only, because a measurement you cannot
reproduce - or that a later run silently replaced - is not evidence.
"""

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.budget import fill_context
from concept_embeddings_rag.evaluation.metrics import (
    context_precision,
    full_support,
    gold_recall,
    recall_at_k,
)
from concept_embeddings_rag.retrieval.base import Retriever

REQUIRED_CONFIG_KEYS: frozenset[str] = frozenset(
    {"model", "revision", "unit_set_hash", "seed", "tokenizer", "code_version", "top_k"}
)


# Both halves of a result's identity end up in a filename, and `split` can reach here
# from pool.json, which is not the pipeline's own output to trust blindly.
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class ProvenanceError(Exception):
    """A result is missing the configuration needed to reproduce it."""


@dataclass(frozen=True)
class BudgetOutcome:
    """One question's reading at one budget: the values the harness averages, unreduced."""

    gold_recall: float
    full_support: float
    precision: float
    units_included: int


@dataclass(frozen=True)
class QuestionOutcome:
    """One question's outcome, recorded only when a caller asks for it.

    Phase 6's paired statistics need what a mean throws away. The values are the very
    float objects the harness appends to its own lists, so a mean over the records, taken
    in the order they were appended, is the harness's mean.
    """

    qid: str
    budgets: dict[int, BudgetOutcome]
    recall_at_k: dict[int, float]


@dataclass(frozen=True)
class RunResult:
    system: str
    split: str
    config: dict
    metrics: dict
    cost: dict
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def validate(self) -> None:
        missing = REQUIRED_CONFIG_KEYS - self.config.keys()
        if missing:
            raise ProvenanceError(
                f"result for {self.system}/{self.split} lacks config keys: {sorted(missing)}"
            )
        for label, value in (("system", self.system), ("split", self.split)):
            if not SAFE_NAME.match(value):
                raise ProvenanceError(f"{label} {value!r} is not usable in a filename")

    def save(self, directory: Path) -> Path:
        """Write the result without ever overwriting an existing one."""
        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        stamp = self.created_at.replace(":", "").replace("-", "")
        base = f"run-{self.system}-{self.split}-{stamp}"
        path = directory / f"{base}.json"
        suffix = 2
        while path.exists():
            path = directory / f"{base}-{suffix}.json"
            suffix += 1

        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return path


def evaluate_retriever(
    retriever: Retriever,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    budgets: Sequence[int],
    ks: Sequence[int],
    top_k: int,
    config: dict,
    split: str | None = None,
    outcomes: list[QuestionOutcome] | None = None,
) -> RunResult:
    """Measure one retriever over one split, at every requested budget.

    `outcomes`, when given, receives one `QuestionOutcome` per question in this loop's
    order (Phase 6, decision D8). It records; it changes no figure the harness returns.
    """
    per_budget: dict[int, dict[str, list[float]]] = {
        budget: {"gold_recall": [], "full_support": [], "precision": []} for budget in budgets
    }
    per_k: dict[int, list[float]] = {k: [] for k in ks}
    latencies: list[float] = []
    units_included: list[int] = []

    for question in questions:
        started = time.perf_counter()
        hits = retriever.retrieve(question.question, top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000.0)

        ranked_ids = [unit_id for unit_id, _score in hits]

        recalls: dict[int, float] = {}
        for k in ks:
            recall = recall_at_k(ranked_ids, question.gold_unit_ids, k)
            per_k[k].append(recall)
            recalls[k] = recall

        readings: dict[int, BudgetOutcome] = {}
        for budget in budgets:
            context = fill_context(ranked_ids, token_counts, budget)
            covered = gold_recall(context, question.gold_unit_ids)
            supported = float(full_support(context, question.gold_unit_ids))
            precision = context_precision(context, question.gold_unit_ids)
            per_budget[budget]["gold_recall"].append(covered)
            per_budget[budget]["full_support"].append(supported)
            per_budget[budget]["precision"].append(precision)
            if budget == budgets[-1]:
                units_included.append(len(context))
            readings[budget] = BudgetOutcome(covered, supported, precision, len(context))

        if outcomes is not None:
            outcomes.append(QuestionOutcome(question.qid, readings, recalls))

    metrics: dict = {}
    for budget in budgets:
        metrics[f"budget_{budget}"] = {
            name: _mean(values) for name, values in per_budget[budget].items()
        }
    for k in ks:
        metrics[f"recall_at_{k}"] = _mean(per_k[k])

    cost = {
        "mean_latency_ms": _mean(latencies),
        "mean_units_included": _mean([float(n) for n in units_included]),
        "n_questions": len(questions),
    }

    resolved_split = split or (questions[0].split if questions else "unknown")
    return RunResult(
        system=retriever.name,
        split=resolved_split,
        config=dict(config),
        metrics=metrics,
        cost=cost,
    )


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
