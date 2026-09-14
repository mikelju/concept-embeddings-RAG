"""What Phase 4 decided against the dev split, written down so it can be checked.

Two free parameters, twelve cells, two seed arms, one decision. The restart fraction
and the operator normalization are fitted here and nothing else is: the stop
threshold and the iteration cap are declared constants, `seed_top_k` is fixed by
decision D5, and K, the view of `X`, the damping and the query operator are
**inherited frozen from Phase 3 and not re-opened**.

The artifact follows `selection.py` rather than inventing a second pattern (decision
D13), so it carries the same three properties:

- **Verified on load**, in a declared order: the split it claims, then its own
  digest, then the pool, then the dictionary, then the Phase 3 freeze it inherits.
  That last one is what makes "the four decisions were not re-opened" checkable
  instead of remembered.
- **Never overwritten.** A second freeze is a deviation to be written down.
- **No field can come from the test split**, and that is a property of the API: a
  cell is built from a `RunResult`, and a result measured on any split but dev is
  refused at the door by the same guard Phase 3 uses.

The file also counts what the phase spent on dev, so the declared cap of 40
evaluations is auditable from the artifact rather than from the narrative.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.concepts.coding import ConceptMatrix
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.harness import (
    REQUIRED_CONFIG_KEYS,
    RunResult,
    evaluate_retriever,
)
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    Decision,
    SelectionError,
    check_dev_only,
    digest_of_payload,
    resolvable_margin,
    serialize_payload,
)
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.diffusion import DiffusionOperator, DiffusionRetriever

EXPANSION_FILENAME: str = "expansion.json"

# What a measured cell has to name about itself before it can enter the table. The
# mean iteration count is in there because D8's tie-break is cost: a cell that cannot
# say what it cost cannot be compared on it.
EXPANSION_CELL_KEYS: tuple[str, ...] = ("seed_arm", "restart", "normalization", "mean_iterations")

# The four decisions this phase reads from the Phase 3 freeze and does not re-take.
INHERITED_KEYS: tuple[str, ...] = ("k", "dictionary_key", "view", "damping", "query_operator")

# One decision, so one entry. D8 says so before any number was measured, and a
# ledger that grew a second entry would be a second degree of freedom nobody declared.
EXPANSION_DECISION_ORDER: tuple[str, ...] = ("expansion_cell",)

# The arm the choice is made on, and the arm it is applied to unchanged (D8).
DENSE_ARM, CONCEPTUAL_ARM = config.SEED_ARMS

# Where each normalization sits in the order the phase declared it. Used to order the
# grid and to break an exact tie, so that both read as `config` writes them rather
# than alphabetically - `none, symmetric, stochastic` is a sequence with a meaning,
# and `none, stochastic, symmetric` is the same three words sorted.
_ARM_ORDER: dict[str, int] = {
    normalization: index for index, normalization in enumerate(config.NORMALIZATION_ARMS)
}


class ExpansionSelectionError(SelectionError):
    """The expansion selection is missing, inconsistent, or not the one asked for.

    A subclass rather than a new exception: every guard this phase inherits from
    Phase 3 raises `SelectionError`, and a caller that wants to catch "the selection
    refused" should not have to know which phase refused.
    """


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def cell_label(restart: float, normalization: str) -> str:
    """How one cell of the grid is named, in the artifact and in the ledger alike."""
    return f"restart={restart:.1f}/{normalization}"


@dataclass(frozen=True)
class ExpansionCell:
    """One evaluation of one (restart, normalization) on one seed arm, kept whole.

    `metrics` is the harness's own output, unreduced. HU-6 asks for the full
    four-budget table beside the one figure that chose, so that a cell winning at
    2,048 and losing everywhere else is visible as exactly that.
    """

    seed_arm: str
    restart: float
    normalization: str
    mean_iterations: float
    n_questions: int
    metrics: dict[str, Any]

    @classmethod
    def from_result(cls, result: RunResult) -> "ExpansionCell":
        """Turn one dev measurement into a table cell, refusing anything else.

        The cell describes itself out of the result's own config rather than out of
        labels handed alongside it, so it cannot claim a configuration that is not
        the one the harness measured.
        """
        if result.split != DEV_SPLIT:
            raise SelectionError(
                f"the selection is fitted on the {DEV_SPLIT!r} split only; this result was "
                f"measured on {result.split!r} and cannot enter the table"
            )

        missing = [key for key in EXPANSION_CELL_KEYS if key not in result.config]
        if missing:
            raise SelectionError(
                f"a table cell must name the configuration that produced it; the result "
                f"lacks {missing}"
            )

        return cls(
            seed_arm=str(result.config["seed_arm"]),
            restart=float(result.config["restart"]),
            normalization=str(result.config["normalization"]),
            mean_iterations=float(result.config["mean_iterations"]),
            n_questions=int(result.cost.get("n_questions", 0)),
            metrics=dict(result.metrics),
        )

    @property
    def label(self) -> str:
        return cell_label(self.restart, self.normalization)

    def primary(self, metric: str, budget: int) -> float:
        """The figure this cell is judged by, refused rather than defaulted if absent."""
        row = self.metrics.get(f"budget_{budget}")
        if not isinstance(row, Mapping):
            raise SelectionError(
                f"this cell was never measured at budget {budget}; it carries "
                f"{sorted(name for name in self.metrics if name.startswith('budget_'))}"
            )
        if metric not in row:
            raise SelectionError(f"there is no {metric!r} at budget {budget}; it has {sorted(row)}")
        return float(row[metric])


@dataclass(frozen=True)
class ExpansionSelection:
    """The frozen configuration of Phase 4, and the dev evidence that chose it.

    `evaluated_on` is a class constant for the reason it is one in Phase 3: a
    selection fitted on test is not a thing this project wants to be able to express,
    let alone write to disk.

    `resolvable` is not a third opinion about the margin - it is the reading of
    `margin` against `resolution`, and a report where the two disagree is refused.
    When the margin is below what the split resolves, D8's cost criterion decided and
    the artifact has to say so.
    """

    selection_metric: str
    selection_budget: int
    cells: list[ExpansionCell]
    chosen: str
    chosen_restart: float
    chosen_normalization: str
    runner_up: str | None
    margin: float
    resolution: float
    resolvable: bool
    tie_break: str | None
    decisions: list[Decision]
    n_dev_decisions: int
    config: dict
    seed: int
    code_version: str
    inherits_selection_digest: str
    inherits: dict
    dev_evaluations_spent: int
    stop_threshold: float
    max_iterations: int
    frozen_at: str = field(default_factory=_now)

    evaluated_on: ClassVar[str] = DEV_SPLIT

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Every invariant the spec states, enforced when built and again when loaded."""
        if self.selection_budget <= 0:
            raise ExpansionSelectionError(
                f"a selection budget is a token count, not {self.selection_budget}"
            )
        if not self.cells:
            raise ExpansionSelectionError(
                "the report holds no measured cell; nothing was swept and nothing can be chosen"
            )

        counts = {cell.n_questions for cell in self.cells}
        if len(counts) != 1:
            raise ExpansionSelectionError(
                f"the cells were measured over different numbers of questions ({sorted(counts)}); "
                "a margin across them would not mean what it says"
            )

        if self.chosen_restart not in config.RESTART_GRID:
            raise ExpansionSelectionError(
                f"the chosen restart {self.chosen_restart} is not in the declared grid "
                f"{list(config.RESTART_GRID)}; a value outside it was never measured"
            )
        if self.chosen_normalization not in config.NORMALIZATION_ARMS:
            raise ExpansionSelectionError(
                f"unknown normalization arm {self.chosen_normalization!r}; expected one of "
                f"{sorted(config.NORMALIZATION_ARMS)}"
            )

        expected = cell_label(self.chosen_restart, self.chosen_normalization)
        if self.chosen != expected:
            raise ExpansionSelectionError(
                f"the report says the chosen cell is {self.chosen!r} while naming restart "
                f"{self.chosen_restart} and normalization {self.chosen_normalization!r}, "
                f"which is {expected!r}"
            )

        measured = {cell.label for cell in self.cells if cell.seed_arm == DENSE_ARM}
        if self.chosen not in measured:
            raise ExpansionSelectionError(
                f"the report chose {self.chosen!r}, which was never measured on the "
                f"{DENSE_ARM!r} arm the choice is made on: {sorted(measured)}"
            )

        if self.n_dev_decisions != len(self.decisions):
            raise ExpansionSelectionError(
                f"n_dev_decisions says {self.n_dev_decisions} but the ledger holds "
                f"{len(self.decisions)}; the count is reported, not estimated"
            )
        names = tuple(decision.name for decision in self.decisions)
        if names != EXPANSION_DECISION_ORDER[: len(names)]:
            raise ExpansionSelectionError(
                f"the decisions are recorded as {list(names)}, and D8 fixes them as "
                f"{list(EXPANSION_DECISION_ORDER)}; a reordering is a deviation, not an edit"
            )

        missing = REQUIRED_CONFIG_KEYS - self.config.keys()
        if missing:
            raise ExpansionSelectionError(
                "the frozen configuration could not reproduce the system it describes; "
                f"it lacks {sorted(missing)}"
            )

        if not 0 < self.dev_evaluations_spent <= config.DEV_EVALUATION_CAP:
            raise ExpansionSelectionError(
                f"this phase spent {self.dev_evaluations_spent} dev evaluations against a "
                f"declared cap of {config.DEV_EVALUATION_CAP}; a report that spent none is "
                "not a measurement, and one that spent more is a deviation to be written down"
            )

        if not self.inherits_selection_digest:
            raise ExpansionSelectionError(
                "the report names no Phase 3 freeze to inherit from; this phase reads its "
                "space, view, damping and query operator from one and never retypes them"
            )
        absent = [key for key in INHERITED_KEYS if key not in self.inherits]
        if absent:
            raise ExpansionSelectionError(
                f"the report inherits a configuration that does not name {absent}; the four "
                "decisions Phase 3 froze travel with every result of this phase"
            )

        if self.stop_threshold <= 0.0:
            raise ExpansionSelectionError(
                f"the stop threshold is a fraction of the mass: expected it above 0, "
                f"got {self.stop_threshold}"
            )
        if self.max_iterations < 1:
            raise ExpansionSelectionError(
                f"the walk runs at least one round, not {self.max_iterations}"
            )

        if self.tie_break is not None and not self.tie_break.strip():
            raise ExpansionSelectionError(
                "a tie-break is either a stated criterion or absent, not empty"
            )
        if self.runner_up is not None:
            # Strictly greater, so a margin sitting exactly on the resolution counts as
            # unresolved. That is the conservative direction and the one `choose_cell`
            # takes, and the two have to agree or a legitimate choice would be refused
            # by the artifact that records it.
            if self.resolvable != (self.margin > self.resolution):
                raise ExpansionSelectionError(
                    f"the report says the margin of {self.margin} is "
                    f"{'resolvable' if self.resolvable else 'unresolvable'} against a "
                    f"resolution of {self.resolution}; `resolvable` is that comparison, "
                    "not a separate opinion about it"
                )
            if not self.resolvable and self.tie_break is None:
                raise ExpansionSelectionError(
                    "the margin is below what the split resolves and no tie-break is "
                    "recorded; D8 declares one, so the artifact has to say it fired"
                )
            if self.resolvable and self.tie_break is not None:
                raise ExpansionSelectionError(
                    "a tie-break is recorded although the margin was resolvable; the cell "
                    "won on the figure, and saying otherwise misreports how it was chosen"
                )

        try:
            datetime.fromisoformat(self.frozen_at)
        except ValueError as error:
            raise ExpansionSelectionError(
                f"frozen_at {self.frozen_at!r} is not a timestamp"
            ) from error

    @property
    def dictionary_keys(self) -> list[str]:
        """Every dictionary this report names. Inherited, so there is exactly one."""
        named = {str(self.inherits["dictionary_key"])}
        recorded = self.config.get("dictionary_key")
        if recorded is not None:
            named.add(str(recorded))
        return sorted(named)

    def cells_of(self, seed_arm: str) -> list[ExpansionCell]:
        """The grid measured on one arm, in the order it was swept."""
        return [cell for cell in self.cells if cell.seed_arm == seed_arm]


def expansion_path(directory: Path | str) -> Path:
    """Where this phase's one selection artifact lives."""
    return Path(directory) / EXPANSION_FILENAME


def _cell_payload(cell: ExpansionCell) -> dict[str, Any]:
    return {
        "seed_arm": cell.seed_arm,
        "restart": cell.restart,
        "normalization": cell.normalization,
        "mean_iterations": cell.mean_iterations,
        "n_questions": cell.n_questions,
        "metrics": cell.metrics,
    }


def _cell_from_payload(payload: Mapping[str, Any]) -> ExpansionCell:
    return ExpansionCell(
        seed_arm=str(payload["seed_arm"]),
        restart=float(payload["restart"]),
        normalization=str(payload["normalization"]),
        mean_iterations=float(payload["mean_iterations"]),
        n_questions=int(payload["n_questions"]),
        metrics=dict(payload["metrics"]),
    )


def _decision_payload(decision: Decision) -> dict[str, Any]:
    return {
        "name": decision.name,
        "chosen": decision.chosen,
        "alternatives": list(decision.alternatives),
        "runner_up": decision.runner_up,
        "margin": decision.margin,
    }


def _decision_from_payload(payload: Mapping[str, Any]) -> Decision:
    runner_up = payload.get("runner_up")
    return Decision(
        name=str(payload["name"]),
        chosen=str(payload["chosen"]),
        alternatives=[str(alternative) for alternative in payload["alternatives"]],
        runner_up=None if runner_up is None else str(runner_up),
        margin=float(payload["margin"]),
    )


def _payload_of(report: ExpansionSelection) -> dict[str, Any]:
    return {
        "evaluated_on": ExpansionSelection.evaluated_on,
        "selection_metric": report.selection_metric,
        "selection_budget": report.selection_budget,
        "cells": [_cell_payload(cell) for cell in report.cells],
        "chosen": report.chosen,
        "chosen_restart": report.chosen_restart,
        "chosen_normalization": report.chosen_normalization,
        "runner_up": report.runner_up,
        "margin": report.margin,
        "resolution": report.resolution,
        "resolvable": report.resolvable,
        "tie_break": report.tie_break,
        "decisions": [_decision_payload(decision) for decision in report.decisions],
        "n_dev_decisions": report.n_dev_decisions,
        "config": dict(report.config),
        "seed": report.seed,
        "code_version": report.code_version,
        "inherits_selection_digest": report.inherits_selection_digest,
        "inherits": dict(report.inherits),
        "dev_evaluations_spent": report.dev_evaluations_spent,
        "stop_threshold": report.stop_threshold,
        "max_iterations": report.max_iterations,
        "frozen_at": report.frozen_at,
    }


def save_expansion_selection(report: ExpansionSelection, directory: Path | str) -> Path:
    """Write the selection once, atomically, and never over an existing one."""
    report.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = expansion_path(directory)

    if path.exists():
        raise ExpansionSelectionError(
            f"a selection is already frozen at {path}; this phase writes one and never "
            "overwrites it. A second freeze is a deviation to be recorded, so move the "
            "existing artifact aside deliberately if that is what is meant"
        )

    payload = _payload_of(report)
    payload["digest"] = digest_of_payload(payload)
    write_text_atomic(path, serialize_payload(payload))
    return path


def expansion_selection_digest(report: ExpansionSelection) -> str:
    """The digest the frozen artifact of this report carries.

    Recorded in every Phase 4 result beside the Phase 3 one, so a measurement names
    both freezes it ran under rather than one of them. Computed from the report for
    the same reason Phase 3 computes its own that way: a digest read out of the file
    it is meant to verify proves nothing.
    """
    return digest_of_payload(_payload_of(report))


def load_expansion_selection(
    directory: Path | str,
    *,
    expected_unit_set_hash: str | None = None,
    known_dictionary_keys: Iterable[str] | None = None,
    inherits_selection_digest: str | None = None,
) -> ExpansionSelection:
    """Read the selection back, proving it is the one this run may use.

    Five checks, in the order the spec declares. The split the artifact claims comes
    first, before the digest: a file that says it was fitted on anything but dev is
    refused whatever its contents hash to. Then the digest, then the pool, then the
    dictionary keys, then the Phase 3 freeze it says it inherits.
    """
    path = expansion_path(directory)
    if not path.exists():
        raise ExpansionSelectionError(
            f"no expansion selection in {directory}; run `cer expand` to write one before "
            "any Phase 4 system is evaluated"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    claimed_split = str(payload.get("evaluated_on", ""))
    if claimed_split != DEV_SPLIT:
        raise ExpansionSelectionError(
            f"the selection at {path.name} declares it was fitted on {claimed_split!r}, not "
            f"on {DEV_SPLIT!r}; refusing to use it"
        )

    recorded = payload.get("digest")
    if recorded is not None:
        actual = digest_of_payload(payload)
        if actual != recorded:
            raise ExpansionSelectionError(
                f"the selection at {path.name} hashes to {actual[:16]} but records "
                f"{str(recorded)[:16]}; it has been modified"
            )

    frozen_config = dict(payload["config"])
    recorded_pool = str(frozen_config.get("unit_set_hash", ""))
    if expected_unit_set_hash is not None and recorded_pool != expected_unit_set_hash:
        raise ExpansionSelectionError(
            f"the selection was made over pool {recorded_pool}, not {expected_unit_set_hash}; "
            "refusing to use it"
        )

    report = ExpansionSelection(
        selection_metric=str(payload["selection_metric"]),
        selection_budget=int(payload["selection_budget"]),
        cells=[_cell_from_payload(cell) for cell in payload["cells"]],
        chosen=str(payload["chosen"]),
        chosen_restart=float(payload["chosen_restart"]),
        chosen_normalization=str(payload["chosen_normalization"]),
        runner_up=None if payload["runner_up"] is None else str(payload["runner_up"]),
        margin=float(payload["margin"]),
        resolution=float(payload["resolution"]),
        resolvable=bool(payload["resolvable"]),
        tie_break=None if payload["tie_break"] is None else str(payload["tie_break"]),
        decisions=[_decision_from_payload(entry) for entry in payload["decisions"]],
        n_dev_decisions=int(payload["n_dev_decisions"]),
        config=frozen_config,
        seed=int(payload["seed"]),
        code_version=str(payload["code_version"]),
        inherits_selection_digest=str(payload["inherits_selection_digest"]),
        inherits=dict(payload["inherits"]),
        dev_evaluations_spent=int(payload["dev_evaluations_spent"]),
        stop_threshold=float(payload["stop_threshold"]),
        max_iterations=int(payload["max_iterations"]),
        frozen_at=str(payload["frozen_at"]),
    )

    if known_dictionary_keys is not None:
        available = set(known_dictionary_keys)
        unknown = [key for key in report.dictionary_keys if key not in available]
        if unknown:
            raise ExpansionSelectionError(
                f"the selection names dictionaries that are not on disk: {unknown}; "
                "the artifacts it was measured over are gone or were never induced"
            )

    if (
        inherits_selection_digest is not None
        and report.inherits_selection_digest != inherits_selection_digest
    ):
        raise ExpansionSelectionError(
            f"this selection inherits the Phase 3 freeze "
            f"{report.inherits_selection_digest[:16]}, and the one on disk is "
            f"{inherits_selection_digest[:16]}; the configuration it was fitted under is "
            "not the configuration this run would apply"
        )

    return report


@runtime_checkable
class Inheriting(Protocol):
    """Anything that names the Phase 3 configuration it inherited.

    The frozen artifact satisfies it, and so does the sweep's own view of what it
    was handed - so the dictionary check is one function serving both, rather than
    one at evaluation time and a second, subtly different one at sweep time.
    """

    @property
    def inherits(self) -> Mapping[str, Any]: ...


def check_dictionary_agrees(report: Inheriting, dictionary_key: str) -> None:
    """Refuse a run whose dictionary is not the one the selection was fitted over.

    HU-3's last criterion in the direction that matters at evaluation time: the
    inherited configuration is read from an artifact and verified, and a space that
    is merely the right size is not the right space.
    """
    inherited = str(report.inherits["dictionary_key"])
    if inherited != dictionary_key:
        raise ExpansionSelectionError(
            f"the expansion selection was fitted over dictionary {inherited!r} and this run "
            f"loaded {dictionary_key!r}; refusing to measure one configuration under another"
        )


def spent_against_cap(spent: int, cap: int = config.DEV_EVALUATION_CAP) -> None:
    """The declared budget of HU-6, checked before an evaluation rather than after it."""
    if spent > cap:
        raise ExpansionSelectionError(
            f"this phase has spent {spent} dev evaluations against a declared cap of {cap}; "
            "exceeding it requires a deviation document, not a quiet second sweep"
        )


def sweep_cells(cells: Sequence[ExpansionCell]) -> dict[str, list[ExpansionCell]]:
    """The measured grid, split by seed arm, for a report that prints both."""
    return {arm: [cell for cell in cells if cell.seed_arm == arm] for arm in config.SEED_ARMS}


# --- The dev grid that fills the table ----------------------------------------


def build_grid_retrievers(
    *,
    matrix: ConceptMatrix,
    dictionary: ConceptDictionary,
    seed_retrievers: Mapping[str, Retriever],
    inherits: Mapping[str, Any],
    concept_weights: np.ndarray | None = None,
    restarts: Sequence[float] = config.RESTART_GRID,
    normalizations: Sequence[str] = config.NORMALIZATION_ARMS,
) -> list[DiffusionRetriever]:
    """Every cell of the grid as a retriever, with one operator per normalization arm.

    The operator is what rescaling `X` produces, and it depends on the normalization
    and on nothing else in the grid. Building it per cell would compute the same three
    matrices twelve times and would dominate the cost of the sweep, so it is built
    once per arm and handed to the four restarts that share it - which the retriever
    accepts only after checking that it is the operator it would have built itself.
    """
    _check_arms(seed_retrievers)
    check_dictionary_agrees(_AsReport(inherits), dictionary.key)

    retrievers: list[DiffusionRetriever] = []
    for normalization in normalizations:
        operator = DiffusionOperator(
            matrix.X, normalization=normalization, concept_weights=concept_weights
        )
        for seed_arm in config.SEED_ARMS:
            for restart in restarts:
                retrievers.append(
                    DiffusionRetriever(
                        matrix,
                        dictionary,
                        seed_retrievers[seed_arm],
                        seed_arm=seed_arm,
                        restart=restart,
                        normalization=normalization,
                        concept_weights=concept_weights,
                        inherits=dict(inherits),
                        operator=operator,
                    )
                )
    return retrievers


def build_expansion_retrievers(
    *,
    matrix: ConceptMatrix,
    dictionary: ConceptDictionary,
    seed_retrievers: Mapping[str, Retriever],
    restart: float,
    normalization: str,
    inherits: Mapping[str, Any],
    concept_weights: np.ndarray | None = None,
) -> list[DiffusionRetriever]:
    """The two arms of HU-3 at one cell: the same parameters, two seeds.

    Decision D8 chooses on the dense arm and applies the cell **unchanged** to the
    conceptual one. That is what this function is: one set of parameters, built twice,
    differing in the seed retriever and in the arm name derived from it. Choosing
    separately per arm would make the two differ in more than the seed, and the
    isolating variant would stop isolating anything.
    """
    _check_arms(seed_retrievers)
    check_dictionary_agrees(_AsReport(inherits), dictionary.key)

    operator = DiffusionOperator(
        matrix.X, normalization=normalization, concept_weights=concept_weights
    )
    return [
        DiffusionRetriever(
            matrix,
            dictionary,
            seed_retrievers[seed_arm],
            seed_arm=seed_arm,
            restart=restart,
            normalization=normalization,
            concept_weights=concept_weights,
            inherits=dict(inherits),
            operator=operator,
        )
        for seed_arm in config.SEED_ARMS
    ]


def cell_config(base_config: Mapping[str, Any], retriever: DiffusionRetriever) -> dict[str, Any]:
    """The shared provenance plus everything this phase adds to a result.

    Read off the retriever rather than passed alongside it, so that a recorded
    configuration cannot claim a restart, an arm or a damping the walk did not use.
    `mean_iterations` is not here: it is a measurement of the run and is merged in
    afterwards by decision D12, because the harness takes its config before it runs.
    """
    described = retriever.describe()
    cell = dict(base_config)
    cell.update(
        {
            "seed_arm": described["seed_arm"],
            "seed_system": described["seed_system"],
            "restart": described["restart"],
            "normalization": described["normalization"],
            "stop_threshold": described["stop_threshold"],
            "max_iterations": described["max_iterations"],
            "seed_top_k": described["seed_top_k"],
            "dictionary_key": described["dictionary_key"],
            "k": described["k"],
            "view": described["view"],
            "damping": described["damping"],
            "query_operator": retriever.inherits.get("query_operator"),
            "config_digest": retriever.config_digest,
        }
    )
    return cell


def sweep_expansion(
    *,
    matrix: ConceptMatrix,
    dictionary: ConceptDictionary,
    seed_retrievers: Mapping[str, Retriever],
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    base_config: Mapping[str, Any],
    inherits: Mapping[str, Any],
    concept_weights: np.ndarray | None = None,
    restarts: Sequence[float] = config.RESTART_GRID,
    normalizations: Sequence[str] = config.NORMALIZATION_ARMS,
    budgets: Sequence[int] = config.CONTEXT_BUDGETS,
    ks: Sequence[int] = config.RECALL_AT_K,
    cap: int = config.DEV_EVALUATION_CAP,
    spent: int = 0,
) -> tuple[list[ExpansionCell], int]:
    """D7's grid, entire: 4 restarts x 3 normalizations x 2 seed arms, on dev.

    One call of `evaluate_retriever` per cell, unmodified, so these figures are
    comparable with the Phase 1 baselines rather than merely similar to them. The
    evaluation counter is checked **before** each pass: a cap discovered after the
    spend is a note about something that already happened, not a budget.

    Returns the cells and the running total, so the caller can keep counting across
    the rest of the phase and write the number into the frozen artifact.
    """
    check_dev_only(questions)
    top_k = _check_top_k(base_config)

    retrievers = build_grid_retrievers(
        matrix=matrix,
        dictionary=dictionary,
        seed_retrievers=seed_retrievers,
        inherits=inherits,
        concept_weights=concept_weights,
        restarts=restarts,
        normalizations=normalizations,
    )

    cells: list[ExpansionCell] = []
    for retriever in retrievers:
        spent += 1
        spent_against_cap(spent, cap)

        retriever.reset_stats()
        result = evaluate_retriever(
            retriever,
            questions=questions,
            token_counts=token_counts,
            budgets=tuple(budgets),
            ks=tuple(ks),
            top_k=top_k,
            config=cell_config(base_config, retriever),
            split=DEV_SPLIT,
        )
        # Decision D12: the counters are a measurement of the run, and the harness
        # takes its configuration before the run exists. They are merged in here.
        counted = retriever.stats.describe()
        cells.append(
            ExpansionCell.from_result(replace(result, config={**result.config, **counted}))
        )
    return cells, spent


def _check_arms(seed_retrievers: Mapping[str, Retriever]) -> None:
    """Both arms of HU-3 or neither: one measured alone is not this phase's grid."""
    missing = [arm for arm in config.SEED_ARMS if arm not in seed_retrievers]
    if missing:
        raise ExpansionSelectionError(
            f"this phase measures both seed arms with identical machinery and is missing "
            f"{missing}; a gain measured on one arm alone cannot be told apart from the "
            "dense retriever having brought almost everything already"
        )


def _check_top_k(base_config: Mapping[str, Any]) -> int:
    """`top_k` comes out of the recorded configuration, so the two cannot disagree."""
    missing = sorted(key for key in REQUIRED_CONFIG_KEYS if key not in base_config)
    if missing:
        raise SelectionError(
            "a sweep is only evidence if the run behind it can be reproduced, and this "
            f"configuration is missing {missing}"
        )
    return int(base_config["top_k"])


@dataclass(frozen=True)
class _AsReport:
    """The one field `check_dictionary_agrees` reads, so the sweep can use it too."""

    inherits: Mapping[str, Any]


# --- The choice, the ledger and the freeze ------------------------------------


@dataclass(frozen=True)
class CellChoice:
    """D8's one decision: which cell, over which alternatives, and how it was decided.

    `runner_up` and `margin` are always the second-best cell by the selection figure
    and the gap to it, whether or not the tie-break fired, because the runner-up is
    part of how a dev figure should be read either way.
    """

    restart: float
    normalization: str
    label: str
    primary: float
    mean_iterations: float
    runner_up: str | None
    margin: float
    resolution: float
    resolvable: bool
    alternatives: list[str]
    tie_break: str | None

    def as_decision(self, name: str = EXPANSION_DECISION_ORDER[0]) -> Decision:
        """This choice, in the shape the ledger stores it."""
        return Decision(
            name=name,
            chosen=self.label,
            alternatives=self.alternatives,
            runner_up=self.runner_up,
            margin=self.margin,
        )


def choose_cell(
    cells: Sequence[ExpansionCell],
    *,
    metric: str = config.EXPANSION_SELECTION_METRIC,
    budget: int = config.EXPANSION_SELECTION_BUDGET,
) -> CellChoice:
    """D8's decision, taken on the dense arm and on the declared figure.

    The leader is the cell with the best Full Support at 2,048 tokens. Every other
    cell within `resolvable_margin` of it is a contender rather than a clear loser,
    and among the leader and its contenders **cost decides**: fewer mean iterations
    wins, then - on an exact tie of figure and cost - the larger restart, because more
    restart is less expansion and an undecided measurement must not arrive as evidence
    for the system under test.

    Both rules were declared in the phase plan before a number existed, which is the
    only thing that makes a tie-break something other than a choice made afterwards.
    """
    if not cells:
        raise ExpansionSelectionError("there is no cell to decide over; the grid measured nothing")

    dense = [cell for cell in cells if cell.seed_arm == DENSE_ARM]
    if not dense:
        raise ExpansionSelectionError(
            f"the choice is made on the {DENSE_ARM!r} arm and the grid holds none of it; "
            "the conceptual arm is measured and reported, never consulted for the choice"
        )

    counts = {cell.n_questions for cell in dense}
    if len(counts) != 1:
        raise ExpansionSelectionError(
            f"the cells were measured over different numbers of questions ({sorted(counts)}); "
            "a margin across them would not mean what it says"
        )
    n_questions = counts.pop()

    ordered = sorted(dense, key=lambda cell: (cell.restart, _ARM_ORDER[cell.normalization]))
    alternatives = [cell.label for cell in ordered]
    figures = {cell.label: cell.primary(metric, budget) for cell in ordered}

    leader = max(ordered, key=lambda cell: figures[cell.label])
    resolution = resolvable_margin(figures[leader.label], n_questions)
    contenders = [
        cell for cell in ordered if abs(figures[cell.label] - figures[leader.label]) <= resolution
    ]

    winner = leader
    tie_break: str | None = None
    if len(contenders) > 1:
        winner = min(
            contenders,
            key=lambda cell: (
                cell.mean_iterations,
                -figures[cell.label],
                -cell.restart,
                _ARM_ORDER[cell.normalization],
            ),
        )
        dearest = max(cell.mean_iterations for cell in contenders if cell is not winner)
        tie_break = (
            f"fewer iterations: {winner.mean_iterations:.2f} against {dearest:.2f} mean rounds"
        )

    others = [cell for cell in ordered if cell is not winner]
    runner_up = max(others, key=lambda cell: figures[cell.label]) if others else None
    margin = 0.0 if runner_up is None else abs(figures[winner.label] - figures[runner_up.label])

    return CellChoice(
        restart=winner.restart,
        normalization=winner.normalization,
        label=winner.label,
        primary=figures[winner.label],
        mean_iterations=winner.mean_iterations,
        runner_up=None if runner_up is None else runner_up.label,
        margin=margin,
        resolution=resolution,
        resolvable=margin > resolution,
        alternatives=alternatives,
        tie_break=tie_break,
    )


def freeze_expansion_selection(
    cells: Sequence[ExpansionCell],
    *,
    choice: CellChoice,
    base_config: Mapping[str, Any],
    inherits: Mapping[str, Any],
    inherits_selection_digest: str,
    dev_evaluations_spent: int,
    metric: str = config.EXPANSION_SELECTION_METRIC,
    budget: int = config.EXPANSION_SELECTION_BUDGET,
) -> ExpansionSelection:
    """Close the configuration: the cell, the ledger, and the moment it closed.

    `frozen_at` is stamped by the report's own default and nowhere else, and
    `save_expansion_selection` refuses to write over an artifact that exists - so a
    configuration freezes once and a second freeze has to be a deliberate, visible act.

    Everything the report states is derived from what this function is handed rather
    than passed again beside it, so no field of the artifact can disagree with the run
    it describes.
    """
    _check_top_k(base_config)

    frozen = dict(base_config)
    frozen.update(
        {
            "restart": choice.restart,
            "normalization": choice.normalization,
            "stop_threshold": config.STOP_THRESHOLD,
            "max_iterations": config.MAX_ITERATIONS,
            "seed_top_k": config.SEED_TOP_K,
            **{key: inherits[key] for key in INHERITED_KEYS if key in inherits},
        }
    )

    return ExpansionSelection(
        selection_metric=metric,
        selection_budget=budget,
        cells=list(cells),
        chosen=choice.label,
        chosen_restart=choice.restart,
        chosen_normalization=choice.normalization,
        runner_up=choice.runner_up,
        margin=choice.margin,
        resolution=choice.resolution,
        resolvable=choice.resolvable,
        tie_break=choice.tie_break,
        decisions=[choice.as_decision()],
        n_dev_decisions=1,
        config=frozen,
        seed=int(base_config["seed"]),
        code_version=str(base_config["code_version"]),
        inherits_selection_digest=inherits_selection_digest,
        inherits=dict(inherits),
        dev_evaluations_spent=dev_evaluations_spent,
        stop_threshold=config.STOP_THRESHOLD,
        max_iterations=config.MAX_ITERATIONS,
    )
