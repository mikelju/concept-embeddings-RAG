"""The rivals' numbers, read from the Phase 3 result files rather than recomputed.

HU-7 asks for System C to join the same table as everything else, and for the four
rivals in it to be the measurements Phase 3 already made. Re-measuring them here
would be a second instrument, and a difference between two instruments cannot be
told apart from a difference between two systems.

What makes reading them safe is the check this module exists for. A result file says
which pool it walked, which tokenizer counted its budget, which seed and which
`top_k`. If any of the four disagrees with this phase's run, the table is refused
rather than built: a table like that would still have its rows, and every one of them
would be wrong by an amount nobody could see.

Nothing here computes a metric. It reads files, checks that they are comparable, and
arranges what they already say.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.harness import RunResult

# The five systems Phase 3 measured, by the names their result files carry. System C
# is compared against four of them; `conceptual` is in the list because the report
# prints it too, and because a missing file is a finding either way.
PHASE_3_SYSTEMS: tuple[str, ...] = (
    "dense",
    "bm25",
    "conceptual",
    "hybrid-conceptual",
    "hybrid-bm25",
)

# What has to agree before two measurements can share a table. Not the model and not
# the code version: a rival may have been measured by an earlier commit, and that is
# what reading it back rather than re-running it means. These four are the ones that
# change what a number *is* - the corpus it walked, the ruler that measured its
# budget, the seed behind it and the depth it was read to.
COMPARABLE_KEYS: tuple[str, ...] = ("unit_set_hash", "tokenizer", "seed", "top_k")

# What the cost columns of HU-7 are called, wherever they come from. The first two are
# written by the harness; the rest reach a result through decision D12 and exist only
# for the expansion systems.
COST_KEYS: tuple[str, ...] = ("mean_latency_ms", "mean_units_included", "n_questions")
EXPANSION_COST_KEYS: tuple[str, ...] = (
    "mean_iterations",
    "cap_hits",
    "mean_sparse_products",
    "clipped_seed_hits",
)


class ComparisonError(Exception):
    """Two measurements were about to be tabulated that cannot be compared."""


@dataclass(frozen=True)
class Reading:
    """One system at one budget, exactly as its result file records it."""

    system: str
    split: str
    budget: int
    gold_recall: float
    full_support: float
    precision: float


def load_rivals(
    directory: Path | str,
    *,
    split: str,
    systems: Sequence[str] = PHASE_3_SYSTEMS,
) -> dict[str, RunResult]:
    """The newest result of each named system on one split, or an error naming the gap.

    The newest rather than the only one: dense and BM25 were measured in Phase 1 as
    well, and the Phase 3 batch is the later reading of the same instrument. A system
    with no file at all is named in the error - skipping it would produce a table
    missing a row that the reader has no way to notice.
    """
    directory = Path(directory)
    found: dict[str, RunResult] = {}

    for system in systems:
        candidates = sorted(directory.glob(f"run-{system}-{split}-*.json"))
        readings = [_read(path) for path in candidates]
        readings = [result for result in readings if result.split == split]
        if not readings:
            continue
        found[system] = max(readings, key=lambda result: result.created_at)

    missing = [system for system in systems if system not in found]
    if missing:
        raise ComparisonError(
            f"no {split!r} result in {directory} for {missing}; this phase reads its rivals' "
            "numbers rather than recomputing them, so a missing file is a gap to be closed "
            "by running the stage that writes it, not a row to be left out"
        )
    return found


def check_comparable(rivals: Mapping[str, RunResult], reference: RunResult) -> None:
    """Refuse a table whose rows were not measured under the same conditions."""
    for system, result in rivals.items():
        if result.split != reference.split:
            raise ComparisonError(
                f"the {system} result was measured on the {result.split!r} split and this run "
                f"is on {reference.split!r}; the two do not belong in one table"
            )
        for key in COMPARABLE_KEYS:
            theirs = result.config.get(key)
            mine = reference.config.get(key)
            if theirs != mine:
                raise ComparisonError(
                    f"the {system} result records {key}={theirs!r} and this run records "
                    f"{mine!r}; comparing them would put two different measurements in one "
                    "column and call the difference a result"
                )


def comparison_table(
    results: Mapping[str, RunResult],
    budgets: Sequence[int] = config.CONTEXT_BUDGETS,
) -> list[Reading]:
    """Everything the result files already say, arranged by budget and by system.

    Ordered by budget and then by the order the systems were handed in, so two runs of
    the same report read line by line.
    """
    table: list[Reading] = []
    for budget in budgets:
        for system, result in results.items():
            row = result.metrics.get(f"budget_{budget}")
            if not isinstance(row, Mapping):
                raise ComparisonError(
                    f"the {system} result was never measured at budget {budget}; it carries "
                    f"{sorted(name for name in result.metrics if name.startswith('budget_'))}"
                )
            table.append(
                Reading(
                    system=system,
                    split=result.split,
                    budget=budget,
                    gold_recall=float(row["gold_recall"]),
                    full_support=float(row["full_support"]),
                    precision=float(row["precision"]),
                )
            )
    return table


def cost_table(results: Mapping[str, RunResult]) -> dict[str, dict[str, Any]]:
    """What each system cost, as §73 asks it to be reported beside what it bought.

    The expansion systems carry more than the others - the iteration distribution and
    the sparse-product count reach a result through decision D12 - and those keys are
    simply absent for a system that never walked, rather than filled with a zero that
    would read as a measurement.
    """
    costs: dict[str, dict[str, Any]] = {}
    for system, result in results.items():
        entry = {key: result.cost[key] for key in COST_KEYS if key in result.cost}
        entry.update(
            {key: result.config[key] for key in EXPANSION_COST_KEYS if key in result.config}
        )
        costs[system] = entry
    return costs


def build_comparison(
    directory: Path | str,
    *,
    split: str,
    reference: RunResult,
    systems: Sequence[str] = PHASE_3_SYSTEMS,
    budgets: Sequence[int] = config.CONTEXT_BUDGETS,
) -> list[Reading]:
    """Load the rivals, prove they are comparable, and tabulate them with this run."""
    rivals = load_rivals(directory, split=split, systems=systems)
    check_comparable(rivals, reference)
    return comparison_table({**rivals, reference.system: reference}, budgets)


def _read(path: Path) -> RunResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return RunResult(
            system=str(payload["system"]),
            split=str(payload["split"]),
            config=dict(payload["config"]),
            metrics=dict(payload["metrics"]),
            cost=dict(payload["cost"]),
            created_at=str(payload["created_at"]),
        )
    except KeyError as error:
        raise ComparisonError(f"{path.name} is missing {error}; it is not a result file") from error
