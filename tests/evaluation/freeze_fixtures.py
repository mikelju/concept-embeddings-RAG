"""The pieces of a Phase 6 freeze, built on toys and measured on dev (T13, T14, T15, T17).

Two fusion fits over scripted retrievers, each with the per-point results the observe-only
hook hands out; dev outcome artifacts of the three systems written to disk; a control
reproduction report, a continuity report, the protocol and the entity component. Everything
is dev: the builder refuses anything else.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from outcome_fixtures import vectors_from_counts, write_outcomes

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.continuity import (
    CHECK_NAMES,
    ContinuityCheck,
    ContinuityReport,
)
from concept_embeddings_rag.evaluation.control_reproduction import Check, ReproductionReport
from concept_embeddings_rag.evaluation.harness import RunResult
from concept_embeddings_rag.evaluation.replacement_freeze import MeasuredFit, point_label
from concept_embeddings_rag.evaluation.selection import fit_fusion_weight
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.fusion import FusedRetriever

DENSE_HITS: list[Hit] = [("g1", 1.0), ("x", 0.5), ("y", 0.0)]
SECOND_HITS: list[Hit] = [("g2", 2.0), ("x", 1.0)]
TOKENS = dict.fromkeys(["g1", "g2", "x", "y"], 1024)


class StubRetriever:
    def __init__(self, name: str, hits: list[Hit]) -> None:
        self.name = name
        self.hits = hits

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return self.hits[:top_k]


class StubStage:
    name = "entity-hop"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
        return self.hits[:top_k]


def a_config() -> dict:
    return {
        "model": "toy-model",
        "revision": "rev",
        "unit_set_hash": "pool",
        "seed": 42,
        "tokenizer": "toy-tokenizer",
        "code_version": "0.1.0",
        "top_k": 100,
    }


def dev_questions() -> list[Question]:
    return [Question("q1", "a question", "-", ("g1", "g2"), (("g1", 0),), "dev")]


def measured_fit(second, stage_hits: list[Hit] | None = None) -> MeasuredFit:
    points: dict[str, RunResult] = {}

    def keep(hybrid: FusedRetriever, result: RunResult) -> None:
        points[point_label(hybrid)] = result

    fit = fit_fusion_weight(
        [StubRetriever("dense", DENSE_HITS), second],
        questions=dev_questions(),
        token_counts=TOKENS,
        base_config=a_config(),
        on_measure=keep,
    )
    return MeasuredFit(fit=fit, points=points)


def control_fit() -> MeasuredFit:
    return measured_fit(StubRetriever("bm25", SECOND_HITS))


def entity_fit(hits: list[Hit] = SECOND_HITS) -> MeasuredFit:
    return measured_fit(StubStage(hits))


def dev_results(directory: Path) -> dict[str, tuple[RunResult, Any]]:
    outcomes, _ = write_outcomes(
        directory,
        vectors_from_counts({(1, 1, 1): 6, (0, 1, 1): 2, (0, 0, 1): 1, (0, 0, 0): 1}),
        split="dev",
        freeze_digest=None,
    )
    results: dict[str, tuple[RunResult, Any]] = {}
    for system, artifact in outcomes.items():
        payload = json.loads((directory / artifact.run_result["file"]).read_text(encoding="utf-8"))
        results[system] = (RunResult(**payload), artifact)
    return results


def a_reproduction(passed: bool = True) -> ReproductionReport:
    checks = [Check("control rrf_score", 0.5, 0.5), Check("dense/dev recall_at_2", 0.25, 0.25)]
    if not passed:
        checks.append(Check("dense/dev budget_2048 full_support", 0.5, 0.4))
    return ReproductionReport(checks=tuple(checks))


def a_continuity(passed: bool = True) -> ContinuityReport:
    checks = [ContinuityCheck(name, 3, 3, ()) for name in CHECK_NAMES]
    if not passed:
        checks[0] = ContinuityCheck(CHECK_NAMES[0], 2, 3, ("q1: read differs",))
    return ContinuityReport(checks=tuple(checks))


def a_protocol() -> dict[str, Any]:
    return {
        "unit_set_hash": "pool",
        "manifest": {"sha256": "0" * 64, "seed": 42, "n_units": 4},
        "split_sizes": {"dev": 600, "test": 1400},
        "question_set_hashes": {"dev": "a" * 16, "test": "b" * 16},
        "model": "toy-model",
        "revision": "rev",
        "tokenizer": "toy-tokenizer",
        "token_counts_sha256": "c" * 64,
        "budgets": list(config.CONTEXT_BUDGETS),
        "top_k": config.EVALUATION_TOP_K,
        "recall_depths": list(config.RECALL_AT_K),
        "metrics": ["gold_recall", "full_support", "precision"],
        "seed": config.DEFAULT_SEED,
        "code_version": "0.1.0",
    }


def an_entity_component() -> dict[str, Any]:
    return {
        "node_index_digest": "d" * 64,
        "extraction_digest": "e" * 64,
        "prompt_digest": "0123456789abcdef",
        "normalization_version": config.NORMALIZATION_VERSION,
        "types": list(config.ENTITY_HOP_TYPES),
        "weight_rule": "log(1 + N / (1 + df)) over binary node incidence",
        "read_depth": config.PILOT_READ_DEPTH,
        "max_depth": config.ENTITY_HOP_MAX_DEPTH,
        "dense_order": "descending score, ties by unit id",
    }


def historical_identities() -> dict[str, dict[str, Any]]:
    identities = {}
    for system in ("dense", "hybrid-bm25"):
        for split, stamp in (("dev", "20260911T133154+0000"), ("test", "20260911T133159+0000")):
            cfg = {"unit_set_hash": "pool", "tokenizer": "toy-tokenizer", "seed": 42, "top_k": 100}
            cfg.update({"model": "toy-model", "revision": "rev"})
            if system == "hybrid-bm25":
                cfg.update(
                    {"fusion_scheme": "weighted", "fusion_weights": {"dense": 0.5, "bm25": 0.5}}
                )
            identities[f"{system}/{split}"] = {
                "name": f"run-{system}-{split}-{stamp}.json",
                "sha256": "9" * 64,
                "system": system,
                "split": split,
                "created_at": "2026-09-11T13:31:54+00:00",
                "config": cfg,
            }
    return identities


def a_reproducibility(dev: dict[str, tuple[RunResult, Any]]) -> dict[str, Any]:
    from concept_embeddings_rag.evaluation.replacement_freeze import reproducibility_block

    first = {system: artifact for system, (_result, artifact) in dev.items()}
    return reproducibility_block(first, first)
