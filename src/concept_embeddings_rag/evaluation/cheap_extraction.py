"""Phase 7 dev measurement, using the existing index, hop, fusion and harness.

Each candidate owns its index, run and dev.json. This step measures candidates;
extractor selection and the held-out run belong to S5.
"""

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.embeddings.cache import unit_set_hash
from concept_embeddings_rag.evaluation.budget import fill_context
from concept_embeddings_rag.evaluation.entity_diagnostics import RecordingHybrid, records_of
from concept_embeddings_rag.evaluation.harness import evaluate_retriever
from concept_embeddings_rag.evaluation.second_hop import node_weights
from concept_embeddings_rag.evaluation.selection import (
    check_dev_only,
    digest_of_payload,
    fit_fusion_weight,
)
from concept_embeddings_rag.nodes.index import (
    build_node_index,
    fragmentation,
    load_node_index,
    save_node_index,
)
from concept_embeddings_rag.nodes.local_extraction import load_extraction
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.entity_hop import EntityHopStage

DEV_FILENAME = "dev.json"


class CheapEvaluationError(ValueError):
    """A candidate cannot be measured against the declared dev inputs."""


def evaluate_candidate(
    directory: Path,
    *,
    extractor_id: str,
    unit_ids: Sequence[str],
    dense: Retriever,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    run_config: Mapping[str, Any],
) -> Path:
    """Build one candidate's index, fit on dev, and persist the measured configuration."""
    check_dev_only(questions)
    directory = Path(directory)
    target = directory / DEV_FILENAME
    if target.exists():
        raise CheapEvaluationError(f"{target} already holds a dev measurement; not overwriting it")
    if any((directory.parent / name).exists() for name in ("selection.json", "test.json")):
        raise CheapEvaluationError("candidate selection already exists; dev fitting is closed")
    if run_config.get("unit_set_hash") != unit_set_hash(unit_ids):
        raise CheapEvaluationError("the run configuration belongs to another pool")

    records, manifest = load_extraction(directory, expected_unit_ids=unit_ids)
    if manifest["extractor_id"] != extractor_id:
        raise CheapEvaluationError("the extraction belongs to another candidate")
    if set(records) != set(unit_ids) or manifest["n_units"] != len(unit_ids):
        raise CheapEvaluationError("the extraction belongs to another pool")
    if any(record.concepts for record in records.values()):
        raise CheapEvaluationError("Phase 7 extraction records must contain entities only")

    # An existing index is verified, never silently replaced to hide a pool mismatch.
    if list(directory.glob("nodes-*.json")):
        index = load_node_index(
            directory, extraction_digest=manifest["digest"], expected_unit_ids=unit_ids
        )
    else:
        index = build_node_index(records, unit_ids, extraction_digest=manifest["digest"])
        save_node_index(index, directory)
    sidecar = directory / f"nodes-{manifest['digest'][:16]}.json"
    weights = node_weights(index)
    provenance = {
        **run_config,
        "phase": 7,
        "extractor_id": extractor_id,
        "extraction_digest": manifest["digest"],
        "extractor_configuration": manifest["configuration"],
        "configuration_digest": manifest["configuration_digest"],
        "node_index_digest": index.digest,
        "normalization_version": index.normalization_version,
        "read_depth": config.PILOT_READ_DEPTH,
        "entity_types": list(config.ENTITY_HOP_TYPES),
    }
    fit = fit_fusion_weight(
        [dense, EntityHopStage(index, weights)],
        questions=questions,
        token_counts=token_counts,
        base_config=provenance,
    )
    # A fresh stage prevents the fit's memo from flattering measured query latency.
    hybrid = RecordingHybrid(
        dense, EntityHopStage(index, weights), scheme=fit.winning_scheme, weights=fit.weights
    )
    result = evaluate_retriever(
        hybrid,
        questions=questions,
        token_counts=token_counts,
        budgets=config.CONTEXT_BUDGETS,
        ks=config.RECALL_AT_K,
        top_k=int(provenance["top_k"]),
        config={**provenance, **hybrid.describe()},
        split="dev",
    )
    recorded = records_of(hybrid, questions)
    expansions = [row.expansion for row in recorded if row.expansion is not None]
    positives = [expansion.positives for expansion in expansions]
    introduced = []
    for row in recorded:
        context = fill_context(
            [unit_id for unit_id, _ in row.fused], token_counts, config.SELECTION_BUDGET
        )
        supplied = {unit_id for unit_id, _ in row.dense}
        introduced.append(len(set(context) - supplied))
    entity_stats = fragmentation(index)["entity"]
    run_path = result.save(directory)
    payload = {
        "extractor_id": extractor_id,
        "split": "dev",
        "n_questions": len(questions),
        "config": result.config,
        "fit": {**asdict(fit), "scheme": fit.winning_scheme, "weights": fit.weights},
        "metrics": result.metrics,
        "cost": result.cost,
        "extraction": manifest,
        "index": {
            **entity_stats,
            "entity_incidences": int(index.incidence[:, index.columns_of(("entity",))].nnz),
            "bytes": sidecar.stat().st_size + sidecar.with_suffix(".npz").stat().st_size,
            "digest": index.digest,
        },
        "hop": {
            "positive_candidates_mean": statistics.fmean(positives),
            "positive_candidates_median": statistics.median(positives),
            "zero_candidate_share": sum(value == 0 for value in positives) / len(questions),
            "p1s_without_entity": sum(value.p1_entity_nodes == 0 for value in expansions),
            "introduced_context_units": sum(introduced),
            "introduced_context_units_mean": statistics.fmean(introduced),
            "introduced_context_budget": config.SELECTION_BUDGET,
            "introduced_context_definition": "fused context units outside Dense top_k",
        },
        "run_file": run_path.name,
        "run_digest": digest_of_payload(asdict(result)),
    }
    payload["digest"] = digest_of_payload(payload)
    return write_text_atomic(target, json.dumps(payload, indent=2, sort_keys=True))
