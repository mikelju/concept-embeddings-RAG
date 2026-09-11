"""What this phase decided against the dev split, written down so it can be checked.

Phase 2 induced four concept spaces and deliberately chose none of them. This module
holds the artifact that resolves that deferral: which K, which view and which query
operator retrieval runs through, plus every other choice fitted on the 600 dev
questions - the fusion scheme, its weight, the damping variant.

Three properties are what turn it from a note into evidence:

- **Verified on load.** The file is bound to its own contents by a digest, to the
  pool it was computed over, and to the dictionary keys it names. That is the rule
  the Phase 1 audit left behind: verify on load, do not merely record.
- **Never overwritten.** A configuration is frozen once. A second freeze is a
  deviation to be written down (`3.Y_name.md`), not a file to be replaced, so the
  writer refuses rather than truncating what is already there.
- **No field can be populated from a test-split number**, and that is a property of
  the API rather than of the caller's care: a table cell is built from a `RunResult`,
  and a `RunResult` measured on any split but dev is refused at the door. The split
  the report was fitted on is a class constant, not a parameter someone could set.

The first part is the schema, its invariants and its two I/O functions. The second
is the sweep that fills the table: it measures every cell through the unchanged
harness, one call per cell, so that System B's numbers and the Phase 1 baselines are
two readings of the same instrument rather than two instruments. The third fits the
fusion on that same split - the declared weight grid and the parameter-free scheme it
has to earn its degree of freedom against - through one function that serves System B
and the control it has to be falsifiable against.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, write_text_atomic
from concept_embeddings_rag.concepts.coding import ConceptMatrix
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.embeddings.backend import EmbeddingBackend
from concept_embeddings_rag.evaluation.harness import (
    REQUIRED_CONFIG_KEYS,
    RunResult,
    evaluate_retriever,
)
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.conceptual import (
    UNDAMPED,
    ConceptualRetriever,
    concept_support,
    resolve_query_operator,
)
from concept_embeddings_rag.retrieval.fusion import RRF, WEIGHTED, FusedRetriever

# The only split this phase fits anything on. It is a constant rather than an
# argument because a selection fitted on test is not a thing this project wants to
# be able to express, let alone write to disk.
DEV_SPLIT: str = "dev"

SELECTION_FILENAME: str = "selection.json"

# What a measured cell has to name about itself before it can enter the table. A
# row that cannot say which space produced it is a number without provenance.
CELL_KEYS: tuple[str, ...] = ("k", "dictionary_key", "view", "query_operator", "damping")
# The three coordinates that vary within one K. Ordered: the tie-break below reads
# them in this order, so two arms with the same figure always resolve the same way.
ARM_KEYS: tuple[str, ...] = ("query_operator", "view", "damping")


class SelectionError(Exception):
    """The selection artifact is missing, inconsistent, or not the one asked for."""


def _now() -> str:
    """The freeze stamp, to the second: a configuration closes at a moment, not an instant."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_primary(metrics: Mapping[str, Any], metric: str, budget: int) -> float:
    """The one figure a measurement is judged by, refused rather than defaulted.

    Shared by the sweep's cells and by the fusion curve, so that "gold recall at
    2,048 tokens" is the same reading of the same harness output wherever this
    phase takes it.
    """
    row = metrics.get(f"budget_{budget}")
    if not isinstance(row, Mapping):
        raise SelectionError(
            f"this was never measured at budget {budget}; it carries "
            f"{sorted(name for name in metrics if name.startswith('budget_'))}"
        )
    if metric not in row:
        raise SelectionError(f"there is no {metric!r} at budget {budget}; it has {sorted(row)}")
    return float(row[metric])


@dataclass(frozen=True)
class SweepCell:
    """One evaluation of one space on dev, kept whole.

    `metrics` is the harness's own output, unreduced: the full four-budget table
    plus the recall-at-k figures. HU-3 asks for it whole so that a ranking which
    only holds at the primary budget is visible as such rather than hidden behind
    the one number that chose the winner.
    """

    k: int
    dictionary_key: str
    view: str
    query_operator: str
    damping: str
    n_questions: int
    metrics: dict[str, Any]

    @classmethod
    def from_result(cls, result: RunResult) -> "SweepCell":
        """Turn one dev measurement into a table cell, refusing anything else.

        The cell describes itself out of the result's own config rather than out of
        labels passed alongside it, so a cell cannot claim a space that is not the
        one the harness measured.
        """
        if result.split != DEV_SPLIT:
            raise SelectionError(
                f"the selection is fitted on the {DEV_SPLIT!r} split only; this result was "
                f"measured on {result.split!r} and cannot enter the table"
            )

        missing = [key for key in CELL_KEYS if key not in result.config]
        if missing:
            raise SelectionError(
                f"a table cell must name the space that produced it; the result lacks {missing}"
            )

        return cls(
            k=int(result.config["k"]),
            dictionary_key=str(result.config["dictionary_key"]),
            view=str(result.config["view"]),
            query_operator=str(result.config["query_operator"]),
            damping=str(result.config["damping"]),
            n_questions=int(result.cost.get("n_questions", 0)),
            metrics=dict(result.metrics),
        )

    @property
    def arm(self) -> dict[str, str]:
        """The three coordinates that vary within one K."""
        return {key: str(getattr(self, key)) for key in ARM_KEYS}

    def primary(self, metric: str, budget: int) -> float:
        """The figure this cell is judged by, refused rather than defaulted if absent."""
        return _read_primary(self.metrics, metric, budget)


@dataclass(frozen=True)
class PerKEntry:
    """One row of the report: one K, every arm measured over it, and its structure.

    The structural columns are Phase 2's written instruction to this phase. They stay
    `None` until they are read from the `SpaceDiagnostics` artifacts, because a space
    can win on recall while being built out of dimensions the corpus barely uses - and
    an invented figure would hide exactly that.
    """

    k: int
    dictionary_key: str
    cells: list[SweepCell]
    best_arm: dict[str, str]
    primary: float
    hub_share: float | None = None
    hub_mass_share: float | None = None
    thin_concepts: int | None = None

    def __post_init__(self) -> None:
        if not self.cells:
            raise SelectionError(f"the row for K={self.k} has no cell; nothing was measured on it")
        spans = {(cell.k, cell.dictionary_key) for cell in self.cells}
        if spans != {(self.k, self.dictionary_key)}:
            raise SelectionError(
                "each row of the table is one K measured over one dictionary; these cells "
                f"span {sorted(spans)}"
            )

    @classmethod
    def from_cells(
        cls,
        cells: Sequence[SweepCell],
        *,
        metric: str = config.SELECTION_METRIC,
        budget: int = config.SELECTION_BUDGET,
        hub_share: float | None = None,
        hub_mass_share: float | None = None,
        thin_concepts: int | None = None,
    ) -> "PerKEntry":
        """The row for one K, with its winning arm derived rather than declared."""
        cells = list(cells)
        if not cells:
            raise SelectionError("a row of the table needs a measurement, and has no cell")

        best = min(cells, key=lambda cell: (-cell.primary(metric, budget), *cell.arm.values()))
        return cls(
            k=cells[0].k,
            dictionary_key=cells[0].dictionary_key,
            cells=cells,
            best_arm=best.arm,
            primary=best.primary(metric, budget),
            hub_share=hub_share,
            hub_mass_share=hub_mass_share,
            thin_concepts=thin_concepts,
        )

    @property
    def n_questions(self) -> int:
        """The one question count every cell of this row shares.

        A row is one K measured on one dev split, so its cells cannot disagree
        about how many questions that measurement covered - if they do, something
        upstream mixed two runs into one row, and that is refused here rather than
        silently averaged away or read off whichever cell happened to come first.
        """
        counts = {cell.n_questions for cell in self.cells}
        if len(counts) != 1:
            raise SelectionError(
                f"the row for K={self.k} was measured over different numbers of "
                f"questions across its cells: {sorted(counts)}"
            )
        return counts.pop()


@dataclass(frozen=True)
class Decision:
    """One choice made against dev: what won, against what, and by how much.

    HU-7 counts these. Six defensible choices over 600 questions are a
    multiple-comparisons problem whatever each one is worth on its own, and the
    honest response is to write down how many there were and what each one cost.
    """

    name: str
    chosen: str
    alternatives: list[str]
    runner_up: str | None
    margin: float

    def __post_init__(self) -> None:
        if self.chosen not in self.alternatives:
            raise SelectionError(
                f"decision {self.name!r} chose {self.chosen!r}, which is not among the "
                f"alternatives it was decided over: {self.alternatives}"
            )
        if self.runner_up is not None and self.runner_up not in self.alternatives:
            raise SelectionError(
                f"decision {self.name!r} names runner-up {self.runner_up!r}, which is not "
                f"among its alternatives: {self.alternatives}"
            )
        if self.margin < 0.0:
            raise SelectionError(
                f"decision {self.name!r} records a margin of {self.margin}: a margin is how "
                "far the winner won by, so a negative one means the loser was recorded"
            )


@dataclass(frozen=True)
class SelectionReport:
    """The frozen configuration of this phase, and the dev evidence that chose it.

    The spec's schema, with one split: `selection_metric` names the metric and
    `selection_budget` the budget, rather than packing both into one string that
    would have to be parsed back apart.

    `evaluated_on` is a class constant. It is written into the artifact so a reader
    does not have to trust the filename, and it is checked on load before anything
    else - no digest can make a selection fitted on test acceptable.
    """

    selected_k: int
    selected_dictionary_key: str
    selection_metric: str
    selection_budget: int
    per_k: dict[int, PerKEntry]
    decisions: list[Decision]
    n_dev_decisions: int
    tie_break: str | None
    config: dict
    seed: int
    code_version: str
    frozen_at: str = field(default_factory=_now)

    evaluated_on: ClassVar[str] = DEV_SPLIT

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Every invariant the spec states, enforced when the report is built and when it loads."""
        for k, entry in self.per_k.items():
            if entry.k != k:
                raise SelectionError(f"the table row filed under K={k} was measured at K={entry.k}")

        if self.selected_k not in self.per_k:
            raise SelectionError(
                f"the report selects K={self.selected_k}, which has no row in the table: "
                f"{sorted(self.per_k)}"
            )

        selected = self.per_k[self.selected_k]
        if selected.dictionary_key != self.selected_dictionary_key:
            raise SelectionError(
                f"the report selects dictionary {self.selected_dictionary_key!r} but the row "
                f"for K={self.selected_k} was measured over {selected.dictionary_key!r}"
            )

        if self.n_dev_decisions != len(self.decisions):
            raise SelectionError(
                f"n_dev_decisions says {self.n_dev_decisions} but the ledger holds "
                f"{len(self.decisions)}; the count is reported, not estimated"
            )

        missing = REQUIRED_CONFIG_KEYS - self.config.keys()
        if missing:
            raise SelectionError(
                "the frozen configuration could not reproduce the system it describes; "
                f"it lacks {sorted(missing)}"
            )

        if self.selection_budget <= 0:
            raise SelectionError(
                f"a selection budget is a token count, not {self.selection_budget}"
            )

        if self.tie_break is not None and not self.tie_break.strip():
            raise SelectionError("a tie-break is either a stated criterion or absent, not empty")

        try:
            datetime.fromisoformat(self.frozen_at)
        except ValueError as error:
            raise SelectionError(f"frozen_at {self.frozen_at!r} is not a timestamp") from error

    @property
    def dictionary_keys(self) -> list[str]:
        """Every dictionary this report names, selected or merely measured."""
        named = {self.selected_dictionary_key}
        for entry in self.per_k.values():
            named.add(entry.dictionary_key)
            named.update(cell.dictionary_key for cell in entry.cells)
        return sorted(named)


def selection_path(directory: Path | str) -> Path:
    """Where this phase's one selection artifact lives."""
    return Path(directory) / SELECTION_FILENAME


def _cell_payload(cell: SweepCell) -> dict[str, Any]:
    return {
        "k": cell.k,
        "dictionary_key": cell.dictionary_key,
        "view": cell.view,
        "query_operator": cell.query_operator,
        "damping": cell.damping,
        "n_questions": cell.n_questions,
        "metrics": cell.metrics,
    }


def _cell_from_payload(payload: Mapping[str, Any]) -> SweepCell:
    return SweepCell(
        k=int(payload["k"]),
        dictionary_key=str(payload["dictionary_key"]),
        view=str(payload["view"]),
        query_operator=str(payload["query_operator"]),
        damping=str(payload["damping"]),
        n_questions=int(payload["n_questions"]),
        metrics=dict(payload["metrics"]),
    )


def _entry_payload(entry: PerKEntry) -> dict[str, Any]:
    return {
        "k": entry.k,
        "dictionary_key": entry.dictionary_key,
        "cells": [_cell_payload(cell) for cell in entry.cells],
        "best_arm": dict(entry.best_arm),
        "primary": entry.primary,
        "hub_share": entry.hub_share,
        "hub_mass_share": entry.hub_mass_share,
        "thin_concepts": entry.thin_concepts,
    }


def _entry_from_payload(payload: Mapping[str, Any]) -> PerKEntry:
    return PerKEntry(
        k=int(payload["k"]),
        dictionary_key=str(payload["dictionary_key"]),
        cells=[_cell_from_payload(cell) for cell in payload["cells"]],
        best_arm={str(key): str(value) for key, value in payload["best_arm"].items()},
        primary=float(payload["primary"]),
        hub_share=_optional_float(payload.get("hub_share")),
        hub_mass_share=_optional_float(payload.get("hub_mass_share")),
        thin_concepts=_optional_int(payload.get("thin_concepts")),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


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


def _payload_of(report: SelectionReport) -> dict[str, Any]:
    return {
        "evaluated_on": SelectionReport.evaluated_on,
        "selected_k": report.selected_k,
        "selected_dictionary_key": report.selected_dictionary_key,
        "selection_metric": report.selection_metric,
        "selection_budget": report.selection_budget,
        "per_k": {str(k): _entry_payload(entry) for k, entry in report.per_k.items()},
        "decisions": [_decision_payload(decision) for decision in report.decisions],
        "n_dev_decisions": report.n_dev_decisions,
        "tie_break": report.tie_break,
        "config": dict(report.config),
        "seed": report.seed,
        "code_version": report.code_version,
        "frozen_at": report.frozen_at,
    }


def _serialize(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _digest_of_payload(payload: Mapping[str, Any]) -> str:
    """The digest is taken over the serialization, minus the digest field itself."""
    return digest_of(_serialize({key: value for key, value in payload.items() if key != "digest"}))


def save_selection(report: SelectionReport, directory: Path | str) -> Path:
    """Write the selection once, atomically, and never over an existing one."""
    report.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = selection_path(directory)

    if path.exists():
        raise SelectionError(
            f"a selection is already frozen at {path}; this phase writes one and never "
            "overwrites it. A second freeze is a deviation to be recorded, so move the "
            "existing artifact aside deliberately if that is what is meant"
        )

    payload = _payload_of(report)
    payload["digest"] = _digest_of_payload(payload)
    write_text_atomic(path, _serialize(payload))
    return path


def load_selection(
    directory: Path | str,
    *,
    expected_unit_set_hash: str | None = None,
    known_dictionary_keys: Iterable[str] | None = None,
) -> SelectionReport:
    """Read the selection back, proving it is the one this run may use.

    Four checks, in this order. The split the artifact claims comes first, before
    the digest: a file that says it was fitted on test is refused whatever its
    contents hash to. Then the digest, then the pool, then the dictionaries.
    """
    path = selection_path(directory)
    if not path.exists():
        raise SelectionError(
            f"no selection artifact in {directory}; run `cer select` to write one before "
            "any Phase 3 system is evaluated"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    claimed_split = str(payload.get("evaluated_on", ""))
    if claimed_split != DEV_SPLIT:
        raise SelectionError(
            f"the selection at {path.name} declares it was fitted on {claimed_split!r}, not "
            f"on {DEV_SPLIT!r}; refusing to use it"
        )

    recorded = payload.get("digest")
    if recorded is not None:
        actual = _digest_of_payload(payload)
        if actual != recorded:
            raise SelectionError(
                f"the selection at {path.name} hashes to {actual[:16]} but records "
                f"{str(recorded)[:16]}; it has been modified"
            )

    frozen_config = dict(payload["config"])
    recorded_pool = str(frozen_config.get("unit_set_hash", ""))
    if expected_unit_set_hash is not None and recorded_pool != expected_unit_set_hash:
        raise SelectionError(
            f"the selection was made over pool {recorded_pool}, not {expected_unit_set_hash}; "
            "refusing to use it"
        )

    report = SelectionReport(
        selected_k=int(payload["selected_k"]),
        selected_dictionary_key=str(payload["selected_dictionary_key"]),
        selection_metric=str(payload["selection_metric"]),
        selection_budget=int(payload["selection_budget"]),
        per_k={int(k): _entry_from_payload(entry) for k, entry in dict(payload["per_k"]).items()},
        decisions=[_decision_from_payload(entry) for entry in payload["decisions"]],
        n_dev_decisions=int(payload["n_dev_decisions"]),
        tie_break=None if payload["tie_break"] is None else str(payload["tie_break"]),
        config=frozen_config,
        seed=int(payload["seed"]),
        code_version=str(payload["code_version"]),
        frozen_at=str(payload["frozen_at"]),
    )

    if known_dictionary_keys is not None:
        available = set(known_dictionary_keys)
        unknown = [key for key in report.dictionary_keys if key not in available]
        if unknown:
            raise SelectionError(
                f"the selection names dictionaries that are not on disk: {unknown}; "
                "the artifacts it was measured over are gone or were never induced"
            )

    return report


# --- The dev sweep that fills the table ---------------------------------------


@dataclass(frozen=True)
class _MemoizingDictionary(ConceptDictionary):
    """The same dictionary, coding each distinct query vector once.

    The two views of one K differ in `X` alone: the concept vector of a question is
    the same for both, and `sparse_coding` pays a lasso solve for it. Coding it twice
    would double the cost of the most expensive arm of the sweep for a number that
    cannot change.

    It wraps rather than reimplements - every code still comes out of the dictionary
    it defers to - so the memo is an optimisation the arithmetic cannot notice, and
    the returned array is a copy so a caller that mutates its result cannot poison
    the next cell.
    """

    source: ConceptDictionary | None = None
    memo: dict[tuple[float | None, bytes], np.ndarray] = field(
        default_factory=dict, repr=False, compare=False
    )

    def encode(self, vectors: np.ndarray, alpha: float | None = None) -> np.ndarray:
        if self.source is None:
            raise SelectionError("a memoizing dictionary must wrap the dictionary it defers to")

        block = np.ascontiguousarray(vectors, dtype=np.float32)
        key = (None if alpha is None else float(alpha), block.tobytes())
        cached = self.memo.get(key)
        if cached is None:
            cached = np.asarray(self.source.encode(block, alpha=alpha), dtype=np.float32)
            self.memo[key] = cached
        return cached.copy()


def _memoizing(dictionary: ConceptDictionary) -> _MemoizingDictionary:
    """The same dictionary by every field that identifies it, so its key is unchanged."""
    return _MemoizingDictionary(
        atoms=dictionary.atoms,
        k=dictionary.k,
        seed=dictionary.seed,
        sparsity_param=dictionary.sparsity_param,
        max_iter=dictionary.max_iter,
        unit_set_hash=dictionary.unit_set_hash,
        model=dictionary.model,
        revision=dictionary.revision,
        code_version=dictionary.code_version,
        merged_from=dictionary.merged_from,
        merge_threshold=dictionary.merge_threshold,
        source=dictionary,
    )


@dataclass(frozen=True)
class SweepSpace:
    """One Phase 2 concept space as the sweep needs it: K, its dictionary, its views.

    The three are checked against each other on construction rather than trusted,
    because the failure they guard against is silent: a matrix coded against another
    dictionary still multiplies, and the row it produces would look like a measured
    result of a space that was never measured.
    """

    k: int
    dictionary: ConceptDictionary
    matrices: dict[str, ConceptMatrix]

    def __post_init__(self) -> None:
        if self.k != self.dictionary.k:
            raise SelectionError(
                f"the space says K={self.k} but its dictionary holds {self.dictionary.k} atoms"
            )
        for view, matrix in self.matrices.items():
            if matrix.dictionary_key != self.dictionary.key:
                raise SelectionError(
                    f"the {view!r} matrix was coded against dictionary "
                    f"{matrix.dictionary_key!r}, not the dictionary {self.dictionary.key!r} "
                    "this space is built on"
                )
            if matrix.view != view:
                raise SelectionError(
                    f"the matrix filed under {view!r} declares the view {matrix.view!r}; "
                    "a view is not renamed by the key it is stored under"
                )

    def matrix_for(self, view: str) -> ConceptMatrix:
        """The matrix for one view, refused by name rather than defaulted to another."""
        if view not in config.CONCEPT_VIEWS:
            raise SelectionError(
                f"unknown concept view: {view!r}; expected one of {sorted(config.CONCEPT_VIEWS)}"
            )
        matrix = self.matrices.get(view)
        if matrix is None:
            raise SelectionError(
                f"the space at K={self.k} carries no {view!r} matrix; it has "
                f"{sorted(self.matrices)}"
            )
        return matrix


@dataclass(frozen=True)
class SpaceStructure:
    """HU-3's three structural figures for one space, read off its own two views.

    Nothing here is recomputed from embeddings: `hub_concept` and `hub_share` come
    from the same sparsity pattern decision D7 damps by, and `hub_mass_share` comes
    from the row-normalized view Phase 2 wrote alongside the raw one. `hub_concept`
    is not one of the three columns the report table carries - the spec lists only
    `hub_share`, `hub_mass_share` and `thin_concepts` - but it is what ties the two
    shares together as being about the same concept, which is what a caller needs to
    explain a tie-break in words.
    """

    hub_concept: int
    hub_share: float
    hub_mass_share: float
    thin_concepts: int


def structure_of(
    space: SweepSpace, *, max_units: int = config.THIN_CONCEPT_MAX_UNITS
) -> SpaceStructure:
    """The hub, its two shares and the thin-concept count, for one space.

    The hub is the concept `concept_support` counts the most units against, on the
    raw view - `df_j` of decision D7, so the count and the damping weight it feeds
    agree by construction. Its mass share is read on the row-normalized view rather
    than on raw: row-normalizing gives every unit the same one unit of attention
    regardless of how many atoms coded it, so a share taken there compares units to
    each other. Taken on raw weights instead, a unit coded by few atoms would
    inflate whichever of them it used, and the resulting share would not be reading
    what the pool actually does with the concept - HU-3 asks for "its share of
    row-normalized mass", not of raw weight, for exactly this reason. A concept
    counts as thin when strictly fewer than `max_units` activate it; at the
    threshold itself it does not.
    """
    raw = space.matrix_for("raw")
    row_norm = space.matrix_for("row_normalized")

    support = concept_support(raw.X)
    n_units = raw.X.shape[0]
    hub = int(np.argmax(support))

    mass = np.asarray(row_norm.X.sum(axis=0)).ravel()

    return SpaceStructure(
        hub_concept=hub,
        hub_share=float(support[hub]) / n_units,
        hub_mass_share=float(mass[hub]) / n_units,
        thin_concepts=int(np.sum(support < max_units)),
    )


def _check_questions(questions: Sequence[Question]) -> None:
    """The guard HU-7 rests on, at the only door the sweep has."""
    if not questions:
        raise SelectionError(
            "the sweep has no question to measure on; the dev split must be loaded first"
        )
    intruders = sorted({question.split for question in questions if question.split != DEV_SPLIT})
    if intruders:
        raise SelectionError(
            f"the selection is fitted on the {DEV_SPLIT!r} split alone; questions on {intruders} "
            "were handed to the sweep and it refuses to measure them"
        )


def _check_config(base_config: Mapping[str, Any]) -> int:
    """Every cell records the configuration that produced it, and `top_k` comes from it.

    Returning the `top_k` the sweep will use, rather than taking it as a second
    argument, is what makes a recorded `top_k` unable to disagree with the one the
    harness was asked for.
    """
    missing = sorted(key for key in REQUIRED_CONFIG_KEYS if key not in base_config)
    if missing:
        raise SelectionError(
            "a sweep is only evidence if the run behind it can be reproduced, and this "
            f"configuration is missing {missing}"
        )
    return int(base_config["top_k"])


def _cell_config(base_config: Mapping[str, Any], retriever: ConceptualRetriever) -> dict[str, Any]:
    """The shared provenance plus the six keys this phase adds to every result."""
    cell = dict(base_config)
    cell.update(
        {
            "dictionary_key": retriever.dictionary_key,
            "k": retriever.k,
            "view": retriever.view,
            # The arm, not the primitive. Decision D4 makes truncation part of the
            # operator, so a cell recording `projection` would merge two of the three
            # alternatives into one row that cannot be read back. `describe()` keeps
            # the primitive; here `query_top_m` and `query_alpha` say which arm it was.
            "query_operator": retriever.arm,
            "query_alpha": retriever.query_alpha,
            "query_top_m": retriever.query_top_m,
            "damping": retriever.damping,
            # Conceptual-only: nothing was fused into this reading.
            "fusion_scheme": None,
        }
    )
    return cell


def sweep_space(
    space: SweepSpace,
    *,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    backend: EmbeddingBackend,
    base_config: Mapping[str, Any],
    arms: Sequence[str] = config.QUERY_OPERATORS,
    views: Sequence[str] = config.CONCEPT_VIEWS,
    damping: str = UNDAMPED,
    concept_weights: np.ndarray | None = None,
    budgets: Sequence[int] = config.CONTEXT_BUDGETS,
    ks: Sequence[int] = config.RECALL_AT_K,
) -> list[SweepCell]:
    """Measure one space over every arm and every view, through the unchanged harness.

    The same function serves D8's first decision and its second: the damping arm is
    this call with the weights supplied, not a second code path, so the two readings
    differ in the weights and in nothing else.

    Everything it could refuse is refused before the first question is retrieved -
    an unknown view, an absent view, an unknown arm, a configuration that could not
    reproduce the run - because a sweep that dies half way through has spent the
    expensive part of the work to produce a table it cannot write.
    """
    _check_questions(questions)
    top_k = _check_config(base_config)

    matrices = [space.matrix_for(view) for view in views]
    operators = [resolve_query_operator(arm) for arm in arms]

    # One memo per space: the two views of a K share it, and the key carries the
    # coding alpha, so a cell can never read a code computed at another resolution.
    dictionary = _memoizing(space.dictionary)

    cells: list[SweepCell] = []
    for operator in operators:
        # The view is not carried separately: `matrix.view` is what the retriever
        # records, so the cell can only ever be filed under the view it was measured on.
        for matrix in matrices:
            retriever = ConceptualRetriever(
                matrix,
                dictionary,
                backend,
                arm=operator.arm,
                damping=damping,
                concept_weights=concept_weights,
            )
            result = evaluate_retriever(
                retriever,
                questions=questions,
                token_counts=token_counts,
                budgets=tuple(budgets),
                ks=tuple(ks),
                top_k=top_k,
                config=_cell_config(base_config, retriever),
                split=DEV_SPLIT,
            )
            cells.append(SweepCell.from_result(result))
    return cells


def run_dev_sweep(
    spaces: Iterable[SweepSpace],
    *,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    backend: EmbeddingBackend,
    base_config: Mapping[str, Any],
    arms: Sequence[str] = config.QUERY_OPERATORS,
    views: Sequence[str] = config.CONCEPT_VIEWS,
    damping: str = UNDAMPED,
    concept_weights: np.ndarray | None = None,
    budgets: Sequence[int] = config.CONTEXT_BUDGETS,
    ks: Sequence[int] = config.RECALL_AT_K,
    metric: str = config.SELECTION_METRIC,
    budget: int = config.SELECTION_BUDGET,
) -> dict[int, PerKEntry]:
    """Decision D5's grid, entire: every K crossed with every arm and every view.

    The table comes back keyed by K and in ascending order whatever order the spaces
    arrived in, so two runs of the same sweep are comparable line by line. Nothing is
    pruned early on a leading figure: a space is measured over all its arms or not at
    all, because the reason for the four-budget table is that a winner at 2,048 which
    loses everywhere else has to be visible as such.
    """
    ordered = sorted(spaces, key=lambda space: space.k)
    if not ordered:
        raise SelectionError("the sweep has no space to measure; run `cer induce` first")

    seen: set[int] = set()
    for space in ordered:
        if space.k in seen:
            raise SelectionError(
                f"K={space.k} was handed to the sweep twice; one row of the table is one space"
            )
        seen.add(space.k)

    _check_questions(questions)
    _check_config(base_config)

    table: dict[int, PerKEntry] = {}
    for space in ordered:
        cells = sweep_space(
            space,
            questions=questions,
            token_counts=token_counts,
            backend=backend,
            base_config=base_config,
            arms=arms,
            views=views,
            damping=damping,
            concept_weights=concept_weights,
            budgets=budgets,
            ks=ks,
        )
        structure = structure_of(space)
        table[space.k] = PerKEntry.from_cells(
            cells,
            metric=metric,
            budget=budget,
            hub_share=structure.hub_share,
            hub_mass_share=structure.hub_mass_share,
            thin_concepts=structure.thin_concepts,
        )
    return table


def _space_label(entry: PerKEntry) -> str:
    """The way one row is named in a `Decision`'s alternatives: K, operator, view."""
    return f"k={entry.k}/{entry.best_arm['query_operator']}/{entry.best_arm['view']}"


def resolvable_margin(primary: float, n_questions: int) -> float:
    """How far apart two figures have to be before the dev split can tell them apart.

    `sqrt(p(1-p)/n)` is the standard error of a mean of values in [0, 1] measured
    over `n` questions - not a declared constant, because the dev split's resolving
    power is a property of the split, not a choice made about it. Used as a margin
    threshold it errs toward calling a difference unresolved rather than real, and
    that is the direction that hands a close call to the declared structural
    criterion instead of to whichever space happened to score a few thousandths
    higher on 600 questions.
    """
    if n_questions <= 0:
        raise SelectionError(f"cannot say what the dev split resolves from {n_questions} questions")
    return float(np.sqrt(primary * (1.0 - primary) / n_questions))


@dataclass(frozen=True)
class SpaceChoice:
    """D8's first decision: which K, on which arm, and why.

    `alternatives` names every row the choice was decided over, in K order, the way
    `Decision` records them; `runner_up` and `margin` are always the second-best row
    by recall and the gap to it, whether or not `tie_break` fired - HU-3 asks for
    the runner-up stated either way, not only when the leader changed hands.
    """

    k: int
    dictionary_key: str
    best_arm: dict[str, str]
    primary: float
    label: str
    runner_up: str | None
    margin: float
    alternatives: list[str]
    tie_break: str | None

    def as_decision(self, name: str = "space") -> Decision:
        """This choice, in the shape HU-7's ledger of decisions stores it."""
        return Decision(
            name=name,
            chosen=self.label,
            alternatives=self.alternatives,
            runner_up=self.runner_up,
            margin=self.margin,
        )


def choose_space(
    table: Mapping[int, PerKEntry],
    *,
    metric: str = config.SELECTION_METRIC,
    budget: int = config.SELECTION_BUDGET,
) -> SpaceChoice:
    """D8's first decision: the K with the best dev recall, a structural tie-break.

    The leader is the row with the highest primary figure. Every other row within
    `resolvable_margin` of it is a contender rather than a clear loser, and among
    the leader and its contenders the smaller hub decides -
    `(hub_share, hub_mass_share, thin_concepts, -primary, k)` in that order -
    because a decimal the dev split cannot tell apart from noise is not a reason to
    prefer the more concentrated space. `tie_break` records that criterion only
    when more than one row was in contention; otherwise it stays `None` and the
    leader's own recall is the whole story.

    `runner_up` and `margin` are computed against the row this function actually
    returns, not against the original leader, so the ledger states what the
    winner's margin over its nearest competitor is, not what it would have been had
    structure not intervened.
    """
    if not table:
        raise SelectionError("choose_space has no row to decide over; the sweep table is empty")

    ordered = sorted(table)
    question_counts = {table[k].n_questions for k in ordered}
    if len(question_counts) > 1:
        raise SelectionError(
            "the rows of this table were measured over different numbers of dev "
            f"questions ({sorted(question_counts)}); a margin across them would not "
            "mean what it says"
        )

    labels = {k: _space_label(table[k]) for k in ordered}
    alternatives = [labels[k] for k in ordered]

    leader_k = max(ordered, key=lambda k: table[k].primary)
    leader = table[leader_k]

    if len(ordered) == 1:
        return SpaceChoice(
            k=leader_k,
            dictionary_key=leader.dictionary_key,
            best_arm=leader.best_arm,
            primary=leader.primary,
            label=labels[leader_k],
            runner_up=None,
            margin=0.0,
            alternatives=alternatives,
            tie_break=None,
        )

    resolution = resolvable_margin(leader.primary, leader.n_questions)
    contenders = [k for k in ordered if abs(table[k].primary - leader.primary) <= resolution]

    winner_k = leader_k
    tie_break: str | None = None
    if len(contenders) > 1:
        for k in contenders:
            entry = table[k]
            if (
                entry.hub_share is None
                or entry.hub_mass_share is None
                or entry.thin_concepts is None
            ):
                raise SelectionError(
                    f"K={k} is within the dev split's resolving power of the leader "
                    "but carries no hub share, hub mass share or thin-concept count; "
                    "refusing to break the tie on the larger decimal"
                )

        winner_k = min(
            contenders,
            key=lambda k: (
                table[k].hub_share,
                table[k].hub_mass_share,
                table[k].thin_concepts,
                -table[k].primary,
                k,
            ),
        )
        winner = table[winner_k]
        runner_hub = max(cast(float, table[k].hub_share) for k in contenders if k != winner_k)
        tie_break = f"smaller hub: {winner.hub_share:.1%} vs {runner_hub:.1%} hub share"

    winner = table[winner_k]
    runner_up_k = max((k for k in ordered if k != winner_k), key=lambda k: table[k].primary)
    margin = abs(winner.primary - table[runner_up_k].primary)

    return SpaceChoice(
        k=winner_k,
        dictionary_key=winner.dictionary_key,
        best_arm=winner.best_arm,
        primary=winner.primary,
        label=labels[winner_k],
        runner_up=labels[runner_up_k],
        margin=margin,
        alternatives=alternatives,
        tie_break=tie_break,
    )


# --- Fitting the fusion on dev ------------------------------------------------


@dataclass(frozen=True)
class FusionFit:
    """One hybrid's fusion as dev decided it: the declared grid, the curve, the reference.

    `curve` is the weighted scheme's dev figure at every point of `grid`, keyed by the
    weight on the **dense** component - the `w` of decision D6's
    `w * dense + (1 - w) * other`. The endpoint `w = 1.0` is in the grid on purpose, so
    "the second signal contributed nothing" is read off the curve rather than inferred
    from its absence. `rrf_score` is the same figure for the parameter-free scheme,
    measured through the same harness call on the same questions.

    Both figures survive whatever the outcome. A weighted scheme that does not beat RRF
    has bought a degree of freedom it did not need, and HU-4 asks for that to be
    reported as the finding it is - which it cannot be if the losing number is dropped.

    The selected weight is derived from the curve rather than stored beside it: a field
    could disagree with the measurements it claims to summarise, and a property cannot.
    """

    components: tuple[str, str]
    metric: str
    budget: int
    n_questions: int
    grid: tuple[float, ...]
    curve: dict[float, float]
    rrf_score: float

    def __post_init__(self) -> None:
        if not self.grid:
            raise SelectionError("a fusion weight is fitted over a grid, and this one has no point")
        if len(set(self.grid)) != len(self.grid):
            raise SelectionError(
                f"the grid repeats a point: {sorted(self.grid)}; a curve over it would claim "
                "more measurements than were taken"
            )
        if set(self.curve) != set(self.grid):
            raise SelectionError(
                "the curve must cover exactly the grid it was fitted over: the curve holds "
                f"{sorted(self.curve)} and the grid {sorted(self.grid)}"
            )

    @property
    def best_weight(self) -> float:
        """The grid point with the best dev figure, ties broken toward the dense side.

        Neither D6 nor D8 says which of two equal weights to take, and the two
        directions are not equivalent. A tie broken toward the second signal would let
        a flat curve - the case where fusing changed nothing at all - come back as
        evidence that the second signal contributed something. Ties therefore go to the
        larger weight on dense, the system this phase has to beat, so an undecided curve
        can never be read as a win for the system under test. The same rule fits the
        HU-5 control, which is what decision D9 requires of it.
        """
        return max(self.grid, key=lambda weight: (self.curve[weight], weight))

    @property
    def best_weighted_score(self) -> float:
        """What the weighted scheme reaches on dev at the weight it selected."""
        return self.curve[self.best_weight]

    @property
    def winning_scheme(self) -> str:
        """D8's third decision, with a tie going to the scheme that fitted nothing.

        The weighted scheme spends a degree of freedom on the same 600 questions it is
        then scored on; RRF spends none. Matching it is therefore not beating it, and
        only a strict win takes the decision.
        """
        return WEIGHTED if self.best_weighted_score > self.rrf_score else RRF

    @property
    def scheme_margin(self) -> float:
        """How far the winning scheme's dev figure is from the other one's."""
        return abs(self.best_weighted_score - self.rrf_score)


def _fusion_config(base_config: Mapping[str, Any], hybrid: FusedRetriever) -> dict[str, Any]:
    """The shared provenance plus what this particular reading fused, and how."""
    fused = dict(base_config)
    fused.update(
        {
            "fusion_scheme": hybrid.scheme,
            "fusion_weights": None if hybrid.weights is None else dict(hybrid.weights),
        }
    )
    return fused


def fit_fusion_weight(
    components: Sequence[Retriever],
    *,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    base_config: Mapping[str, Any],
    grid: Sequence[float] = config.FUSION_WEIGHT_GRID,
    budgets: Sequence[int] = config.CONTEXT_BUDGETS,
    ks: Sequence[int] = config.RECALL_AT_K,
    metric: str = config.SELECTION_METRIC,
    budget: int = config.SELECTION_BUDGET,
) -> FusionFit:
    """D8's third and fourth decisions, measured in one pass over the dev split.

    One function, called with `[dense, conceptual]` for System B and with
    `[dense, bm25]` for the HU-5 control: decision D9 makes that a requirement rather
    than a convenience, because giving the control a shorter grid or a weaker
    reference is the single easiest way to manufacture this phase's headline. The
    components are an argument; everything else about the fitting is not.

    Every point of the grid and the RRF reading go through `evaluate_retriever`
    unmodified, so these figures are comparable with the Phase 1 baselines rather than
    merely similar to them.
    """
    _check_questions(questions)
    top_k = _check_config(base_config)
    if not grid:
        raise SelectionError("a fusion weight is fitted over a grid, and this one has no point")

    # Built first because it needs no weight, and because its component order is the
    # one the weighted hybrids are keyed by: the order belongs to the fusion, never to
    # whoever passed the components in.
    reference = FusedRetriever(list(components), scheme=RRF)
    dense_name, second_name = reference.components

    def measure(hybrid: FusedRetriever) -> float:
        result = evaluate_retriever(
            hybrid,
            questions=questions,
            token_counts=token_counts,
            budgets=tuple(budgets),
            ks=tuple(ks),
            top_k=top_k,
            config=_fusion_config(base_config, hybrid),
            split=DEV_SPLIT,
        )
        return _read_primary(result.metrics, metric, budget)

    rrf_score = measure(reference)

    curve: dict[float, float] = {}
    for point in grid:
        weight = float(point)
        hybrid = FusedRetriever(
            list(components),
            scheme=WEIGHTED,
            weights={dense_name: weight, second_name: 1.0 - weight},
        )
        curve[weight] = measure(hybrid)

    return FusionFit(
        components=(dense_name, second_name),
        metric=metric,
        budget=budget,
        n_questions=len(questions),
        grid=tuple(float(point) for point in grid),
        curve=curve,
        rrf_score=rrf_score,
    )
