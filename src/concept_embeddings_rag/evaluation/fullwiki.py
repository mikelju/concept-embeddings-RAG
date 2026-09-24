"""Phase 9: does Dense + Entity Hop stay useful over the whole HotpotQA FullWiki corpus?

One variable changes - corpus scale - and everything else is inherited: BGE-small, the
frozen Phase 7 GLiNER configuration, the one-hop query-blind Entity Hop, both fusion
weights, the budget ruler and the harness. The module holds what is new and no more:

1. **Cohorts and gold (S3).** The 7,405 validation qids split by the historical Phase 1
   subset, and each supporting title resolved to a FullWiki unit by title (D4): exact, then
   a unique `normalized_title`. An unresolved title becomes a sentinel gold id no unit can
   carry, so its question stays in the denominator and can never reach Full Support.
2. **The systems (D8).** P9-A Dense, P9-B Dense + BM25 and P9-C Dense + Entity Hop, built
   from the inherited retrievers with the weights read from the fits that chose them.
3. **One measurement path (S4, S6).** The same function measures the historical-pool
   reproduction and the FullWiki pass: the Phase 1 harness for the budget family, the rank
   metrics read off the same list, latency, and the Entity Hop's candidate record.
4. **The build (S5).** Token counts, BGE vectors, the BM25 index and the GLiNER entity index
   over the whole corpus, each with a manifest recording its identity, size, digest, wall
   time, attributable cost and peak memory.
5. **The probe, the single pass and the outcome (S6, S7).** Query time on a seeded sample of
   historical dev questions (rankings only), the one authorized pass writing each run once,
   and the exact McNemar test that labels the phase mechanically.

Nothing here reads the candidate contexts shipped with the FullWiki dev file (R2): the
questions come from the distractor validation source, and retrieval searches the corpus.
"""

import gzip
import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import (
    digest_of,
    savez_compressed_atomic,
    write_text_atomic,
)
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question
from concept_embeddings_rag.corpus.scale_corpus import (
    load_token_counts,
    normalized_title,
    token_counts_for,
)
from concept_embeddings_rag.embeddings.cache import (
    EmbeddingCache,
    cache_key,
    query_prompt_of,
    question_cache_key,
    question_set_hash,
)
from concept_embeddings_rag.embeddings.cache import unit_set_hash as corpus_hash
from concept_embeddings_rag.evaluation.budget import fill_context
from concept_embeddings_rag.evaluation.entity_diagnostics import (
    RecordingHybrid,
    RecordingRetriever,
    RecordingStage,
)
from concept_embeddings_rag.evaluation.harness import QuestionOutcome, evaluate_retriever
from concept_embeddings_rag.evaluation.metrics import full_support
from concept_embeddings_rag.nodes.extraction import ExtractionRecord
from concept_embeddings_rag.nodes.index import NodeIndex, build_node_index, save_node_index
from concept_embeddings_rag.retrieval.base import Hit, Retriever
from concept_embeddings_rag.retrieval.bm25 import BM25Retriever
from concept_embeddings_rag.retrieval.entity_hop import EntityHopStage
from concept_embeddings_rag.retrieval.fusion import WEIGHTED

QUESTIONS_FILENAME = "questions.json"
DATA_STOP = "DATA_STOP"
OPERATIONAL_STOP = "OPERATIONAL_STOP"
SOURCE = "hotpotqa distractor validation"

# A gold id no corpus unit can carry: unit ids are 16 hexadecimal characters.
UNRESOLVED_PREFIX = "unresolved:"

EXACT, NORMALIZED, UNMATCHED, AMBIGUOUS = "exact", "normalized", "unmatched", "ambiguous"


class FullWikiPhaseError(Exception):
    """A Phase 9 artifact or input is not the one this step may use."""


class DataStop(FullWikiPhaseError):
    """The frozen data contract does not hold: the spec's `DATA_STOP`."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"{DATA_STOP}: {reason}")


# --- Cohorts (D3) -----------------------------------------------------------------------


def cohorts_of(
    raw_qids: Sequence[str],
    historical_qids: Sequence[str],
    *,
    sizes: Mapping[str, int] = config.PHASE_9_COHORT_SIZES,
) -> dict[str, list[str]]:
    """The three cohorts by qid, in source order. Any count that does not reconcile stops."""
    if len(set(raw_qids)) != len(raw_qids):
        raise DataStop("the validation source repeats a qid")
    if len(set(historical_qids)) != len(historical_qids):
        raise DataStop("the historical Phase 1 subset repeats a qid")
    known = set(raw_qids)
    outside = [qid for qid in historical_qids if qid not in known]
    if outside:
        raise DataStop(f"{len(outside)} historical qids are not in the validation source")
    historical = set(historical_qids)
    cohorts = {
        config.PHASE_9_STANDARD: list(raw_qids),
        config.PHASE_9_HISTORICAL_OVERLAP: [qid for qid in raw_qids if qid in historical],
        config.PHASE_9_RETRIEVAL_UNSEEN: [qid for qid in raw_qids if qid not in historical],
    }
    for cohort, qids in cohorts.items():
        if len(qids) != sizes[cohort]:
            raise DataStop(f"{cohort} holds {len(qids)} qids, not the declared {sizes[cohort]}")
    return cohorts


def supporting_titles(raw: Mapping[str, Any]) -> tuple[str, str]:
    """A question's distinct supporting titles, in first-appearance order: exactly two."""
    titles = tuple(dict.fromkeys(str(title) for title, _index in raw["supporting_facts"]))
    if len(titles) != 2:
        raise DataStop(f"question {raw['_id']} has {len(titles)} supporting titles, not two")
    return titles[0], titles[1]


# --- Gold mapping (D4) ------------------------------------------------------------------


@dataclass(frozen=True)
class TitleResolution:
    title: str
    status: str
    unit_id: str | None
    candidates: int


def resolve_titles(
    titles: Iterable[str], units: Iterable[IndexingUnit]
) -> dict[str, TitleResolution]:
    """Each title to one FullWiki unit by title alone, or to a recorded failure.

    One pass over the corpus with only title-sized indexes in memory. An exact title wins;
    otherwise the deviation-8.1 `normalized_title` must name exactly one unit. Two units
    under the same exact or normalized title are ambiguous and never resolved to either.
    No fuzzy, embedding or model-assisted matching exists here.
    """
    wanted = set(titles)
    wanted_normalized = {normalized_title(title) for title in wanted}
    exact: dict[str, list[str]] = {}
    normalized: dict[str, list[str]] = {}
    for unit in units:
        if unit.title in wanted:
            exact.setdefault(unit.title, []).append(unit.unit_id)
        key = normalized_title(unit.title)
        if key in wanted_normalized:
            normalized.setdefault(key, []).append(unit.unit_id)

    resolved: dict[str, TitleResolution] = {}
    for title in sorted(wanted):
        exact_ids = exact.get(title, [])
        if len(exact_ids) == 1:
            resolved[title] = TitleResolution(title, EXACT, exact_ids[0], 1)
        elif exact_ids:
            resolved[title] = TitleResolution(title, AMBIGUOUS, None, len(exact_ids))
        else:
            normalized_ids = normalized.get(normalized_title(title), [])
            if len(normalized_ids) == 1:
                resolved[title] = TitleResolution(title, NORMALIZED, normalized_ids[0], 1)
            elif normalized_ids:
                resolved[title] = TitleResolution(title, AMBIGUOUS, None, len(normalized_ids))
            else:
                resolved[title] = TitleResolution(title, UNMATCHED, None, 0)
    return resolved


def _sentinel(qid: str, position: int) -> str:
    return f"{UNRESOLVED_PREFIX}{qid}:{position}"


def _question_digest(questions: Sequence[Question]) -> str:
    return digest_of(
        *(f"{q.qid}\t{q.split}\t{q.question}" for q in sorted(questions, key=lambda q: q.qid))
    )


def _mapping_digest(questions: Sequence[Question], corpus_unit_set_hash: str) -> str:
    return digest_of(
        corpus_unit_set_hash,
        *(
            f"{q.qid}\t{q.gold_unit_ids[0]}\t{q.gold_unit_ids[1]}"
            for q in sorted(questions, key=lambda q: q.qid)
        ),
    )


def build_questions(
    raws: Sequence[Mapping[str, Any]],
    historical_qids: Sequence[str],
    units: Iterable[IndexingUnit],
    *,
    corpus: Mapping[str, Any],
    sizes: Mapping[str, int] = config.PHASE_9_COHORT_SIZES,
    ceiling: int = config.PHASE_9_UNRESOLVED_CEILING,
) -> tuple[list[Question], dict[str, Any]]:
    """The 7,405 questions with their cohort and resolved gold, and the artifact body.

    `split` carries the cohort a question belongs to besides `standard-dev`: every question
    is in `standard-dev`, and exactly one of `historical-overlap` or `retrieval-unseen`.
    """
    cohorts = cohorts_of([str(raw["_id"]) for raw in raws], historical_qids, sizes=sizes)
    historical = set(cohorts[config.PHASE_9_HISTORICAL_OVERLAP])
    titles_by_qid = {str(raw["_id"]): supporting_titles(raw) for raw in raws}
    resolutions = resolve_titles(
        (title for pair in titles_by_qid.values() for title in pair), units
    )

    questions: list[Question] = []
    entries: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for raw in raws:
        qid = str(raw["_id"])
        gold: list[str] = []
        for position, title in enumerate(titles_by_qid[qid]):
            resolution = resolutions[title]
            if resolution.unit_id is None:
                gold.append(_sentinel(qid, position))
                unresolved.append(
                    {
                        "qid": qid,
                        "title": title,
                        "status": resolution.status,
                        "candidates": resolution.candidates,
                    }
                )
            else:
                gold.append(resolution.unit_id)
        cohort = (
            config.PHASE_9_HISTORICAL_OVERLAP
            if qid in historical
            else config.PHASE_9_RETRIEVAL_UNSEEN
        )
        question = Question(
            qid=qid,
            question=str(raw["question"]),
            answer=str(raw["answer"]),
            gold_unit_ids=(gold[0], gold[1]),
            supporting_facts=tuple((str(t), int(i)) for t, i in raw["supporting_facts"]),
            split=cohort,
        )
        questions.append(question)
        entries.append(
            {
                "qid": qid,
                "cohort": cohort,
                "question": question.question,
                "answer": question.answer,
                "supporting_titles": list(titles_by_qid[qid]),
                "supporting_facts": [[t, i] for t, i in question.supporting_facts],
                "gold_unit_ids": list(question.gold_unit_ids),
                "resolution": [resolutions[title].status for title in titles_by_qid[qid]],
            }
        )

    affected = len({entry["qid"] for entry in unresolved})
    status_counts = {
        status: sum(1 for title in resolutions.values() if title.status == status)
        for status in (EXACT, NORMALIZED, UNMATCHED, AMBIGUOUS)
    }
    body: dict[str, Any] = {
        "source": SOURCE,
        "n_standard_dev": len(cohorts[config.PHASE_9_STANDARD]),
        "n_historical_overlap": len(cohorts[config.PHASE_9_HISTORICAL_OVERLAP]),
        "n_retrieval_unseen": len(cohorts[config.PHASE_9_RETRIEVAL_UNSEEN]),
        "corpus_unit_set_hash": corpus["unit_set_hash"],
        "corpus_ordered_unit_digest": corpus["ordered_unit_digest"],
        "question_digest": _question_digest(questions),
        "mapping_digest": _mapping_digest(questions, str(corpus["unit_set_hash"])),
        "distinct_titles": len(resolutions),
        "title_resolution": status_counts,
        "unmatched_titles": sum(1 for entry in unresolved if entry["status"] == UNMATCHED),
        "ambiguous_titles": sum(1 for entry in unresolved if entry["status"] == AMBIGUOUS),
        "questions_with_unresolved_gold": affected,
        "unresolved_ceiling": ceiling,
        "unresolved": unresolved,
        "unresolved_rule": (
            "an unmatched or ambiguous gold title stays in the denominator as a sentinel id no "
            "unit carries: never retrieved by any system"
        ),
        "terminal_state": DATA_STOP if affected > ceiling else None,
        "questions": entries,
    }
    return questions, body


def write_questions(directory: Path | str, body: Mapping[str, Any]) -> Path:
    """Persist the frozen question set; a different mapping never replaces a recorded one."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / QUESTIONS_FILENAME
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        for key in ("question_digest", "mapping_digest"):
            if existing.get(key) != body[key]:
                raise FullWikiPhaseError(
                    f"{target} already records {key} {existing.get(key)}, not {body[key]}: a "
                    "frozen question set is never replaced in place"
                )
    write_text_atomic(target, json.dumps(dict(body), indent=2, ensure_ascii=False, sort_keys=True))
    return target


def load_questions(
    directory: Path | str, *, corpus_unit_set_hash: str
) -> tuple[list[Question], dict[str, Any]]:
    """The frozen questions, their digests recomputed, refused on a DATA_STOP or another corpus."""
    target = Path(directory) / QUESTIONS_FILENAME
    if not target.exists():
        raise FullWikiPhaseError(f"{target} does not exist: run 'fullwiki-questions' first")
    body: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    if body.get("terminal_state") == DATA_STOP:
        raise DataStop(
            f"{body['questions_with_unresolved_gold']} questions have an unresolved gold title, "
            f"past the ceiling of {body['unresolved_ceiling']}; no retrieval score is published"
        )
    if body["corpus_unit_set_hash"] != corpus_unit_set_hash:
        raise FullWikiPhaseError(
            f"{target.name} was mapped against corpus {body['corpus_unit_set_hash']}, not "
            f"{corpus_unit_set_hash}"
        )
    questions = [
        Question(
            qid=str(entry["qid"]),
            question=str(entry["question"]),
            answer=str(entry["answer"]),
            gold_unit_ids=(str(entry["gold_unit_ids"][0]), str(entry["gold_unit_ids"][1])),
            supporting_facts=tuple((str(t), int(i)) for t, i in entry["supporting_facts"]),
            split=str(entry["cohort"]),
        )
        for entry in body["questions"]
    ]
    if (
        _question_digest(questions) != body["question_digest"]
        or _mapping_digest(questions, corpus_unit_set_hash) != body["mapping_digest"]
    ):
        raise FullWikiPhaseError(f"{target.name} does not match its recorded digests")
    return questions, body


# --- The frozen weights and the three systems (D8) --------------------------------------


def read_frozen_weights(
    bm25_file: Path | str = config.PHASE_9_BM25_WEIGHTS_FILE,
    entity_file: Path | str = config.PHASE_9_ENTITY_WEIGHTS_FILE,
) -> dict[str, dict[str, float]]:
    """Both hybrids' weights, read from the fits that chose them and held to the frozen values.

    The recorded floats are returned as recorded - `0.30000000000000004`, not a retyped
    `0.3` - because they are what the historical runs fused with, and a reproduction has to
    fuse with the same numbers. A recorded fit that is not the frozen one refuses.
    """
    bm25 = json.loads(Path(bm25_file).read_text(encoding="utf-8"))["config"]
    entity = json.loads(Path(entity_file).read_text(encoding="utf-8"))["fit"]
    recorded = {
        "hybrid-bm25": (str(bm25["fusion_scheme"]), dict(bm25["fusion_weights"])),
        "hybrid-entity-hop": (str(entity["scheme"]), dict(entity["weights"])),
    }
    weights: dict[str, dict[str, float]] = {}
    for system, (scheme, values) in recorded.items():
        frozen = config.PHASE_9_FROZEN_WEIGHTS[system]
        agrees = (
            scheme == WEIGHTED
            and sorted(values) == sorted(frozen)
            and all(
                abs(float(values[name]) - weight) <= config.PHASE_9_WEIGHT_TOLERANCE
                for name, weight in frozen.items()
            )
        )
        if not agrees:
            raise FullWikiPhaseError(
                f"the recorded fit for {system} is {scheme} {values}, not the frozen weighted "
                f"{frozen}; Phase 9 fits nothing and fuses only with the inherited weights"
            )
        weights[system] = {name: float(value) for name, value in values.items()}
    return weights


def build_systems(
    dense: Retriever,
    bm25: Retriever,
    stage: EntityHopStage,
    weights: Mapping[str, Mapping[str, float]],
) -> dict[str, RecordingRetriever | RecordingHybrid]:
    """P9-A, P9-B and P9-C, each keeping the lists it produced for the per-question record."""
    return {
        "dense": RecordingRetriever(dense),
        "hybrid-bm25": RecordingHybrid(
            dense, bm25, scheme=WEIGHTED, weights=weights["hybrid-bm25"]
        ),
        "hybrid-entity-hop": RecordingHybrid(
            dense, stage, scheme=WEIGHTED, weights=weights["hybrid-entity-hop"]
        ),
    }


# --- The shared measurement path (S4 reproduction, S6 evaluation pass) -------------------
#
# One function measures a system for S4 and for S6, so the reproduction on the historical
# pool exercises exactly the code the FullWiki pass runs. The budget metrics go through
# the unchanged Phase 1 harness (its `outcomes=` record); the rank metrics are read off the
# very list the harness filled its context from.


class TimedRetriever:
    """Times each `retrieve` call. The boundary: from the call to the returned list, with
    the question vector already encoded (a cached lookup), as in every earlier phase."""

    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self.name = inner.name
        self.seconds: list[float] = []
        self.rankings: list[list[str]] = []

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        started = time.perf_counter()
        hits = self.inner.retrieve(query, top_k)
        self.seconds.append(time.perf_counter() - started)
        self.rankings.append([unit_id for unit_id, _score in hits])
        return hits


LATENCY_BOUNDARY = (
    "Retriever.retrieve wall time per question, question vector served from a cache; "
    "includes dense scoring, the second component and fusion; excludes question encoding"
)


def measure_system(
    name: str,
    system: RecordingRetriever | RecordingHybrid,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    *,
    run_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One record per question, in question order: budgets, rank metrics, latency, hop."""
    if system.name != name:
        raise FullWikiPhaseError(f"system {system.name!r} was handed in as {name!r}")
    timed = TimedRetriever(system)
    outcomes: list[QuestionOutcome] = []
    evaluate_retriever(
        timed,
        questions,
        token_counts,
        budgets=config.PHASE_9_BUDGETS,
        ks=config.PHASE_9_KS,
        top_k=config.PHASE_9_RANKING_DEPTH,
        config=dict(run_config or {}),
        split=config.PHASE_9_STANDARD,
        outcomes=outcomes,
    )
    expansions = (
        system.second.expansions
        if isinstance(system, RecordingHybrid) and isinstance(system.second, RecordingStage)
        else None
    )
    records: list[dict[str, Any]] = []
    for position, (question, outcome) in enumerate(zip(questions, outcomes, strict=True)):
        ranked = timed.rankings[position]
        gold = question.gold_unit_ids
        record: dict[str, Any] = {
            "qid": question.qid,
            "cohort": question.split,
            "budgets": {
                str(budget): {
                    "full_support": reading.full_support,
                    "gold_recall": reading.gold_recall,
                    "context_precision": reading.precision,
                    "units_included": reading.units_included,
                }
                for budget, reading in outcome.budgets.items()
            },
            "fs_at_k": {str(k): float(full_support(ranked[:k], gold)) for k in config.PHASE_9_KS},
            "gpr_at_k": {str(k): outcome.recall_at_k[k] for k in config.PHASE_9_KS},
            "latency_ms": timed.seconds[position] * 1000.0,
        }
        if expansions is not None and isinstance(system, RecordingHybrid):
            expansion = expansions[position]
            dense_ids = {unit_id for unit_id, _score in system.dense.calls[position]}
            context = fill_context(ranked, token_counts, config.PHASE_9_PRIMARY_BUDGET)
            outside = [unit_id for unit_id in gold if unit_id not in dense_ids]
            record["entity"] = {
                "p1": expansion.p1,
                "p1_entity_nodes": expansion.p1_entity_nodes,
                "positives": expansion.positives,
                "proposed": len(expansion.candidates),
                "gold_proposed_outside_dense_top100": sum(
                    1 for candidate in expansion.candidates if candidate.unit_id in outside
                ),
                "gold_introduced_top100": sum(1 for unit_id in outside if unit_id in ranked),
                "gold_introduced_context_2048": sum(1 for unit_id in outside if unit_id in context),
            }
        records.append(record)
    return records


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def cohort_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per cohort: the budget family, FS/GPR @K, and the primary count, as plain means."""
    members = {
        config.PHASE_9_STANDARD: list(records),
        config.PHASE_9_RETRIEVAL_UNSEEN: [
            r for r in records if r["cohort"] == config.PHASE_9_RETRIEVAL_UNSEEN
        ],
        config.PHASE_9_HISTORICAL_OVERLAP: [
            r for r in records if r["cohort"] == config.PHASE_9_HISTORICAL_OVERLAP
        ],
    }
    primary = str(config.PHASE_9_PRIMARY_BUDGET)
    metrics: dict[str, dict[str, Any]] = {}
    for cohort, rows in members.items():
        metrics[cohort] = {
            "n_questions": len(rows),
            "at_budget": {
                str(budget): {
                    name: _mean([r["budgets"][str(budget)][name] for r in rows])
                    for name in ("full_support", "gold_recall", "context_precision")
                }
                for budget in config.PHASE_9_BUDGETS
            },
            "full_support_at_k": {
                str(k): _mean([r["fs_at_k"][str(k)] for r in rows]) for k in config.PHASE_9_KS
            },
            "gold_recall_at_k": {
                str(k): _mean([r["gpr_at_k"][str(k)] for r in rows]) for k in config.PHASE_9_KS
            },
            "supported_at_primary_budget": int(
                sum(r["budgets"][primary]["full_support"] for r in rows)
            ),
        }
    return metrics


def _percentile(values: Sequence[float], share: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), share)) if values else 0.0


def latency_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(r["latency_ms"]) for r in records]
    return {
        "mean_ms": _mean(values),
        "median_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "max_ms": max(values) if values else 0.0,
        "boundary": LATENCY_BOUNDARY,
    }


def candidate_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """HU-5's candidate-count distribution and the gold the hop brings from outside Dense."""
    rows = [r["entity"] for r in records if "entity" in r]
    if not rows:
        return None
    positives = [float(row["positives"]) for row in rows]
    return {
        "n_questions": len(rows),
        "positives_mean": _mean(positives),
        "positives_median": _percentile(positives, 50),
        "positives_p90": _percentile(positives, 90),
        "positives_p95": _percentile(positives, 95),
        "positives_p99": _percentile(positives, 99),
        "positives_max": int(max(positives)),
        "zero_candidate_share": _mean([1.0 if p == 0 else 0.0 for p in positives]),
        "p1_without_entity": sum(1 for row in rows if row["p1_entity_nodes"] == 0),
        "questions_with_gold_introduced_top100": sum(
            1 for row in rows if row["gold_introduced_top100"] > 0
        ),
        "gold_introduced_top100": sum(int(row["gold_introduced_top100"]) for row in rows),
        "gold_introduced_context_2048": sum(
            int(row["gold_introduced_context_2048"]) for row in rows
        ),
        "gold_proposed_outside_dense_top100": sum(
            int(row["gold_proposed_outside_dense_top100"]) for row in rows
        ),
    }


def reproduction_verdict(
    measured: Mapping[str, int], recorded: Mapping[str, int]
) -> dict[str, Any]:
    """S4: the Phase 9 path must give the recorded Phase 7 dev counts, exactly, per system."""
    mismatches = {
        system: {"measured": int(measured.get(system, -1)), "recorded": int(count)}
        for system, count in recorded.items()
        if measured.get(system) != count
    }
    return {
        "measured": dict(measured),
        "recorded": dict(recorded),
        "reproduced": not mismatches,
        "mismatches": mismatches,
    }


# --- S5: the frozen FullWiki representations, one manifest each --------------------------
#
# Each build step runs as its own CLI invocation on the measurement host, so each records
# its own peak memory, and each writes one manifest beside its artifact. Every artifact is
# tied to the corpus by its unit-set hash; the entity index also by the extraction digest.

TOKENS_FILENAME = "token-counts.npz"
TOKENS_MANIFEST = "token-counts.json"
EMBEDDING_FILENAME = "embedding.json"
BM25_FILENAME = "bm25.json"
GLINER_DIRNAME = "gliner"
ENTITY_INDEX_FILENAME = "entity-index.json"
HUBS_RECORDED = 25


def peak_memory() -> dict[str, float | None]:
    """This process's RSS high-water mark (Linux) and peak CUDA allocation, when measurable."""
    rss: float | None = None
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                rss = int(line.split()[1]) / 1024.0
    vram: float | None = None
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        vram = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    return {"peak_rss_mb": rss, "peak_vram_mb": vram}


def _usd(seconds: float, hourly_rate_usd: float) -> float:
    return seconds / 3600.0 * hourly_rate_usd


def _write_manifest(path: Path, body: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_text_atomic(
        path, json.dumps(dict(body), indent=2, ensure_ascii=False, sort_keys=True)
    )


def build_token_counts(
    units: Sequence[IndexingUnit],
    counter: Any,
    directory: Path | str,
    *,
    unit_set_hash: str,
) -> dict[str, Any]:
    """The historical budget ruler over every unit, in bounded batches (R5)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    counts = token_counts_for(units, counter)
    seconds = time.perf_counter() - started
    unit_ids = [unit.unit_id for unit in units]
    values = np.array([counts[unit_id] for unit_id in unit_ids], dtype=np.int32)
    savez_compressed_atomic(
        directory / TOKENS_FILENAME, unit_ids=np.array(unit_ids, dtype=np.str_), counts=values
    )
    body = {
        "tokenizer_id": config.BUDGET_TOKENIZER_ID,
        "tokenizer_revision": config.BUDGET_TOKENIZER_REVISION,
        "n_units": len(unit_ids),
        "unit_set_hash": unit_set_hash,
        "counts_digest": digest_of(values),
        "total_tokens": int(values.sum()),
        "seconds": seconds,
        "memory": peak_memory(),
    }
    _write_manifest(directory / TOKENS_MANIFEST, body)
    return body


def load_phase9_token_counts(
    directory: Path | str, *, unit_set_hash: str, unit_ids: Sequence[str]
) -> dict[str, int]:
    """The budget ruler, refused unless it is keyed by exactly `unit_ids`, in that order, and
    its counts are the ones the build digested."""
    directory = Path(directory)
    manifest = json.loads((directory / TOKENS_MANIFEST).read_text(encoding="utf-8"))
    if manifest["unit_set_hash"] != unit_set_hash:
        raise FullWikiPhaseError("the token counts belong to another corpus")
    counts = load_token_counts(directory)
    if len(counts) != manifest["n_units"]:
        raise FullWikiPhaseError("the token counts do not cover the corpus they name")
    # SEC-034: the counts decide every budget metric, so they are verified like every other
    # input rather than trusted by length. The ids must be the corpus's, in the corpus's order
    # (a set check would let shuffled ids hand each unit another unit's count), and the values
    # in that order must be the ones digested at build time.
    if list(counts) != list(unit_ids):
        raise FullWikiPhaseError("the token counts are not keyed by the corpus units in order")
    if digest_of(np.array(list(counts.values()), dtype=np.int32)) != manifest["counts_digest"]:
        raise FullWikiPhaseError(
            f"{TOKENS_FILENAME} does not match the digest {TOKENS_MANIFEST} records"
        )
    return counts


def question_cache_key_for(questions: Sequence[Question], backend: Any) -> str:
    """Phase 9's own question key: the cohort name stands where a split did (the gotcha)."""
    return question_cache_key(
        backend.name,
        backend.revision,
        question_set_hash([question.qid for question in questions]),
        config.PHASE_9_STANDARD,
        normalized=bool(getattr(backend, "normalize", True)),
    )


EMBED_BLOCK = 262_144


def encode_blockwise(units: Sequence[IndexingUnit], backend: Any, *, block: int) -> np.ndarray:
    """`backend.encode` over `indexable_text`, block by block, into one float32 array."""
    if query_prompt_of(backend):
        raise FullWikiPhaseError("documents are encoded without a query prompt")
    vectors: np.ndarray | None = None
    for start in range(0, len(units), block):
        encoded = backend.encode([unit.indexable_text for unit in units[start : start + block]])
        if vectors is None:
            vectors = np.empty((len(units), encoded.shape[1]), dtype=np.float32)
        vectors[start : start + len(encoded)] = encoded
        print(f"[INFO] encoded {min(start + block, len(units))}/{len(units)} units", flush=True)
    if vectors is None:
        raise FullWikiPhaseError("no unit to encode")
    return vectors


def build_embedding(
    units: Sequence[IndexingUnit],
    questions: Sequence[Question],
    backend: Any,
    cache_dir: Path | str,
    question_cache_dir: Path | str,
    directory: Path | str,
    *,
    unit_set_hash: str,
    weights_sha256: str,
    resolved_revision: str,
    hardware: Mapping[str, Any],
    hourly_rate_usd: float,
    block: int = EMBED_BLOCK,
) -> dict[str, Any]:
    """BGE-small over every unit and every question, in Phase 9's own caches (D5, HU-3).

    The corpus is encoded in blocks into one preallocated float32 array, then saved under
    the key and sidecar fields `embed_units` writes. One call over 5.2M texts would hold the
    batch list, its stacked copy and the normalized copy at once - about 8 GB each on a host
    listed with 41 GB.
    """
    if (backend.name, backend.revision) != (config.EMBEDDING_MODEL, config.EMBEDDING_REVISION):
        raise FullWikiPhaseError(f"Phase 9 measures BGE-small only, not {backend.name}")
    if resolved_revision != config.EMBEDDING_REVISION:
        raise FullWikiPhaseError(
            f"the Hub served {resolved_revision}, not the pinned {config.EMBEDDING_REVISION}"
        )
    target = Path(directory) / EMBEDDING_FILENAME
    if target.exists():
        raise FullWikiPhaseError(f"{target} already records the FullWiki encoding")
    if corpus_hash([unit.unit_id for unit in units]) != unit_set_hash:
        raise FullWikiPhaseError("the units handed in are not the corpus the manifest names")
    cache = EmbeddingCache(Path(cache_dir))
    key = cache_key(backend.name, backend.revision, unit_set_hash, normalized=True)
    if cache.path_for(key).exists():
        raise FullWikiPhaseError(
            f"{cache.path_for(key)} exists without {target.name}: its wall time is unknown"
        )
    started = time.perf_counter()
    vectors = encode_blockwise(units, backend, block=block)
    corpus_seconds = time.perf_counter() - started
    unit_ids = [unit.unit_id for unit in units]
    cache.save(
        key,
        vectors,
        unit_ids,
        metadata={
            "model": backend.name,
            "revision": backend.revision,
            "resolved_revision": resolved_revision,
            "dim": int(vectors.shape[1]),
            "normalized": True,
        },
    )

    question_started = time.perf_counter()
    question_vectors = backend.encode([question.question for question in questions])
    question_seconds = time.perf_counter() - question_started
    question_key = question_cache_key_for(questions, backend)
    EmbeddingCache(Path(question_cache_dir)).save(
        question_key,
        question_vectors,
        [question.qid for question in questions],
        metadata={
            "model": backend.name,
            "revision": backend.revision,
            "dim": int(question_vectors.shape[1]),
            "normalized": True,
            "split": config.PHASE_9_STANDARD,
        },
    )
    body = {
        "model": backend.name,
        "revision": backend.revision,
        "resolved_revision": resolved_revision,
        "weights_sha256": weights_sha256,
        "dim": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
        "normalized": True,
        "query_prompt": "",
        "n_units": len(unit_ids),
        "unit_set_hash": unit_set_hash,
        "corpus_cache_key": key,
        "vectors_digest": vectors_digest(vectors),
        "vector_bytes": int(vectors.nbytes),
        "cache_file_bytes": cache.path_for(key).stat().st_size,
        "wall_seconds": corpus_seconds,
        "paragraphs_per_second": len(unit_ids) / corpus_seconds if corpus_seconds else 0.0,
        "question_cache_key": question_key,
        "n_questions": len(questions),
        "question_seconds": question_seconds,
        "hardware": dict(hardware),
        "hourly_rate_usd": hourly_rate_usd,
        "attributable_usd": _usd(corpus_seconds + question_seconds, hourly_rate_usd),
        "attributable_usd_basis": "derived from measured wall time x contracted hourly rate",
        "memory": peak_memory(),
    }
    _write_manifest(target, body)
    return body


def vectors_digest(vectors: np.ndarray) -> str:
    """sha256 over the float32 vector bytes, read in place: no 8 GB copy on a 41 GB host."""
    block = np.ascontiguousarray(vectors, dtype=np.float32)
    return hashlib.sha256(block.reshape(-1).view(np.uint8)).hexdigest()


def bm25_digest(retriever: BM25Retriever) -> tuple[str, int]:
    """A digest over the built BM25 index and its in-memory bytes (data, indices, indptr)."""
    index = retriever._index
    scores = index.scores
    arrays = [np.asarray(scores[name]) for name in ("data", "indices", "indptr")]
    vocabulary = json.dumps(sorted(index.vocab_dict.items()), ensure_ascii=True)
    digest = digest_of(*arrays, int(scores["num_docs"]), vocabulary, *retriever.unit_ids)
    return digest, int(sum(array.nbytes for array in arrays))


def build_bm25(
    units: Sequence[IndexingUnit], *, unit_set_hash: str
) -> tuple[BM25Retriever, dict[str, Any]]:
    """BM25 over the same `indexable_text` and unit ids, timed and digested (HU-3)."""
    started = time.perf_counter()
    retriever = BM25Retriever(units)
    seconds = time.perf_counter() - started
    digest, nbytes = bm25_digest(retriever)
    return retriever, {
        "library": "bm25s",
        "library_version": importlib.metadata.version("bm25s"),
        "stopwords": retriever.stopwords,
        "n_units": len(retriever.unit_ids),
        "unit_set_hash": unit_set_hash,
        "index_digest": digest,
        "index_bytes": nbytes,
        "index_bytes_basis": "in-memory score matrix arrays; the index is rebuilt, not stored",
        "build_seconds": seconds,
        "memory": peak_memory(),
    }


def entity_index_statistics(index: NodeIndex) -> dict[str, Any]:
    """HU-4's figures over the entity nodes: incidences, empty rows, DF percentiles, hubs."""
    columns = index.columns_of(config.ENTITY_HOP_TYPES)
    entity = index.incidence[:, columns].tocsc()
    df = np.diff(entity.indptr)
    per_row = np.diff(entity.tocsr().indptr)
    hubs = np.argsort(-df, kind="stable")[:HUBS_RECORDED]
    return {
        "n_units": len(index.unit_ids),
        "n_entity_nodes": int(columns.size),
        "n_incidences": int(entity.nnz),
        "zero_entity_units": int((per_row == 0).sum()),
        "failed_units": index.failed_units,
        "df_percentiles": {
            "p50": float(np.percentile(df, 50)) if df.size else 0.0,
            "p90": float(np.percentile(df, 90)) if df.size else 0.0,
            "p95": float(np.percentile(df, 95)) if df.size else 0.0,
            "p99": float(np.percentile(df, 99)) if df.size else 0.0,
            "max": int(df.max()) if df.size else 0,
        },
        "singleton_share": float((df == 1).mean()) if df.size else 0.0,
        "largest_hubs": [
            {"form": index.nodes[int(columns[column])].form, "df": int(df[column])}
            for column in hubs
        ],
    }


def build_entity_index(
    records: Mapping[str, ExtractionRecord],
    unit_ids: Sequence[str],
    directory: Path | str,
    *,
    extraction_digest: str,
    corpus_unit_set_hash: str,
) -> dict[str, Any]:
    """The Phase 5/7 node index over the whole corpus, in corpus row order (HU-4)."""
    directory = Path(directory)
    started = time.perf_counter()
    index = build_node_index(records, unit_ids, extraction_digest=extraction_digest)
    sidecar = save_node_index(index, directory)
    seconds = time.perf_counter() - started
    body = {
        "corpus_unit_set_hash": corpus_unit_set_hash,
        "extraction_digest": extraction_digest,
        "normalization": index.normalization_version,
        **entity_index_statistics(index),
        "index_bytes": sidecar.stat().st_size + sidecar.with_suffix(".npz").stat().st_size,
        "index_digest": index.digest,
        "build_seconds": seconds,
        "memory": peak_memory(),
    }
    _write_manifest(directory / ENTITY_INDEX_FILENAME, body)
    return body


# --- S6: the operational probe and the single evaluation pass ---------------------------

PROBE_FILENAME = "probe.json"
EVALUATION_MARKER = "pass.json"
REFERENCE_HOP = "reference"
COLUMNWISE_HOP = "columnwise"
# The plan's "near the ceiling": the exact column-wise hop replaces the reference when the
# reference P9-C probe mean reaches this share of the query ceiling. Either gives the same
# candidates and scores (tests/retrieval/test_columnwise_hop.py); only the time differs.
COLUMNWISE_SHARE = 0.8


def probe_sample(
    questions: Sequence[Question],
    dev_qids: Iterable[str],
    *,
    n: int = config.PHASE_9_PROBE_QUESTIONS,
    seed: int = config.PHASE_9_PROBE_SEED,
) -> list[Question]:
    """A seeded sample of historical dev questions, already used for dev in earlier phases."""
    dev = set(dev_qids)
    pool = sorted((q for q in questions if q.qid in dev), key=lambda q: q.qid)
    if len(pool) < n:
        raise FullWikiPhaseError(f"only {len(pool)} historical dev questions for a probe of {n}")
    return random.Random(seed).sample(pool, n)  # noqa: S311 - a sample, not a secret


def time_system(system: Retriever, questions: Sequence[Question]) -> list[float]:
    """Seconds per `retrieve` call, rankings only: nothing is scored against a gold."""
    seconds: list[float] = []
    for question in questions:
        started = time.perf_counter()
        system.retrieve(question.question, config.PHASE_9_RANKING_DEPTH)
        seconds.append(time.perf_counter() - started)
    return seconds


def hop_implementation(reference_mean_seconds: float) -> str:
    near = reference_mean_seconds >= COLUMNWISE_SHARE * config.PHASE_9_QUERY_SECONDS_CEILING
    return COLUMNWISE_HOP if near else REFERENCE_HOP


def probe_verdict(mean_seconds: Mapping[str, float]) -> dict[str, Any]:
    """D11: a system whose mean query time exceeds the ceiling is an OPERATIONAL_STOP."""
    over = [
        name
        for name, seconds in mean_seconds.items()
        if seconds > config.PHASE_9_QUERY_SECONDS_CEILING
    ]
    return {
        "mean_seconds": dict(mean_seconds),
        "ceiling_seconds": config.PHASE_9_QUERY_SECONDS_CEILING,
        "over_ceiling": over,
        "terminal_state": OPERATIONAL_STOP if over else None,
    }


def start_pass(directory: Path | str, *, probe_digest: str, code_commit: str) -> Path:
    """Mark the single evaluation pass as started. A second start is refused, whatever happened."""
    target = Path(directory) / EVALUATION_MARKER
    if target.exists():
        raise FullWikiPhaseError(
            f"{target} exists: the evaluation pass runs once and is never rerun because of what "
            "it showed; a failed pass is a decision for the author"
        )
    body = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "probe_digest": probe_digest,
        "code_commit": code_commit,
    }
    return _write_manifest(target, body)


def _run_names(system: str) -> tuple[str, str]:
    return f"run-{system}.json", f"outcomes-{system}.jsonl.gz"


def write_run(
    directory: Path | str,
    system: str,
    records: Sequence[Mapping[str, Any]],
    *,
    body: Mapping[str, Any],
) -> Path:
    """One system's run and its per-question outcomes, written once."""
    directory = Path(directory)
    run_name, outcomes_name = _run_names(system)
    target = directory / run_name
    if target.exists() or (directory / outcomes_name).exists():
        raise FullWikiPhaseError(f"{target} already exists; a Phase 9 run is written once")
    text = "".join(json.dumps(dict(r), sort_keys=True) + "\n" for r in records)
    archive = directory / outcomes_name
    temporary = archive.with_name(archive.name + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(text.encode("utf-8"))
    temporary.replace(archive)
    full = {**body, "outcomes_file": outcomes_name, "outcomes_digest": digest_of(text)}
    return _write_manifest(target, full)


def load_outcomes(directory: Path | str, system: str) -> list[dict[str, Any]]:
    directory = Path(directory)
    run_name, outcomes_name = _run_names(system)
    body = json.loads((directory / run_name).read_text(encoding="utf-8"))
    text = gzip.decompress((directory / outcomes_name).read_bytes()).decode("utf-8")
    if digest_of(text) != body["outcomes_digest"]:
        raise FullWikiPhaseError(f"{outcomes_name} does not match the digest {run_name} records")
    return [json.loads(line) for line in text.splitlines()]


# --- S7: the one inferential test and the mechanical label (D10) -------------------------


def exact_two_sided_p(wins: int, losses: int) -> float:
    """The exact McNemar test: two-sided binomial(n = wins + losses, 1/2) on the discordant."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / 2**n)


def primary_outcome(
    dense: Sequence[Mapping[str, Any]],
    entity: Sequence[Mapping[str, Any]],
    *,
    stop: str | None = None,
) -> dict[str, Any]:
    """P9-C against P9-A, Full Support @2,048 on `retrieval-unseen`, and the D10 label."""
    budget = str(config.PHASE_9_PRIMARY_BUDGET)
    cohort = config.PHASE_9_RETRIEVAL_UNSEEN
    a = {r["qid"]: r["budgets"][budget]["full_support"] for r in dense if r["cohort"] == cohort}
    c = {r["qid"]: r["budgets"][budget]["full_support"] for r in entity if r["cohort"] == cohort}
    if set(a) != set(c):
        raise FullWikiPhaseError("the two runs do not hold the same primary questions")
    wins = sum(1 for qid in a if c[qid] == 1.0 and a[qid] == 0.0)
    losses = sum(1 for qid in a if a[qid] == 1.0 and c[qid] == 0.0)
    p = exact_two_sided_p(wins, losses)
    if stop is not None:
        label = stop
    elif wins > losses and p < config.PHASE_9_ALPHA:
        label = "SCALE_SUPPORTED"
    elif losses > wins and p < config.PHASE_9_ALPHA:
        label = "SCALE_REGRESSION"
    else:
        label = "SCALE_NOT_SUPPORTED"
    n = len(a)
    dense_successes = int(sum(a.values()))
    entity_successes = int(sum(c.values()))
    return {
        "primary_cohort": cohort,
        "primary_metric": f"full_support@{budget}_tokens",
        "n_questions": n,
        "dense_successes": dense_successes,
        "entity_successes": entity_successes,
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
        "delta_percentage_points": 100.0 * (entity_successes - dense_successes) / n if n else 0.0,
        "exact_two_sided_p": p,
        "alpha": config.PHASE_9_ALPHA,
        "test": "exact two-sided McNemar (binomial on discordant questions, p = 1/2)",
        "terminal_state": label,
    }
