"""Phase 7 dev measurement, selection and the single held-out run.

Each candidate owns its index, run and dev.json. Selection reads those dev
readings, applies the rule the spec wrote before any of them existed, and names
the extractor Phase 8 inherits. Only then may test be read, once.
"""

import json
import math
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
from concept_embeddings_rag.retrieval.fusion import FusedRetriever

DEV_FILENAME = "dev.json"
SELECTION_FILENAME = "selection.json"
TEST_FILENAME = "test.json"
TEST_SPLIT = "test"
HEADLINE_BUDGET = config.SELECTION_BUDGET


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


# --- Selection (D11) --------------------------------------------------------------------


def _headline(payload: Mapping[str, Any]) -> float:
    return float(payload["metrics"][f"budget_{HEADLINE_BUDGET}"]["full_support"])


def retention_bar(n_questions: int) -> tuple[float, int]:
    """The bar as a share and as whole questions, from the inherited dev figures.

    The share is `dense + 0.75 * (claude entity hop - dense)` on the Phase 6 dev
    readings, which over the 600 dev questions is 517 of them - the number the spec
    writes as `0.8617`. The count is what the comparison runs on: a float test
    against the rounded share would reject a candidate sitting exactly on the bar.
    """
    reference = config.PHASE_7_REFERENCE_DEV_FULL_SUPPORT
    dense, entity_hop = reference["dense"], reference["hybrid-entity-hop"]
    share = dense + config.PHASE_7_RETENTION_SHARE * (entity_hop - dense)
    return share, math.ceil(round(share * n_questions, 6))


def _retention_of(payload: Mapping[str, Any]) -> dict[str, Any]:
    n_questions = int(payload["n_questions"])
    share, required = retention_bar(n_questions)
    full_support = _headline(payload)
    supported = round(full_support * n_questions)
    return {
        "full_support_at_headline": full_support,
        "headline_budget": HEADLINE_BUDGET,
        "supported_questions": supported,
        "n_questions": n_questions,
        "bar_share": share,
        "bar_questions": required,
        "retention_share": config.PHASE_7_RETENTION_SHARE,
        "passed": supported >= required,
    }


def _economics_of(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The four clauses of the economic bar: two numeric, two structural."""
    manifest = payload["extraction"]
    projection = manifest["projection_5m"]
    hours = float(projection["hours"])
    infrastructure = float(manifest["usd"])
    software = float(config.PHASE_7_SOFTWARE_COST_CEILING_USD)
    return {
        "third_party_api_calls_per_paragraph": 0,
        "api_call_basis": (
            "a declared local open-weight extractor; the pass makes no network call after "
            "the pinned model download, so the clause holds by construction, not by sampling"
        ),
        "software_cost_usd": software,
        "projected_fullwiki_hours": hours,
        "projected_fullwiki_usd": float(projection["usd"]),
        "infrastructure_usd": infrastructure,
        "hours_ceiling": config.PHASE_7_FULLWIKI_HOURS_CEILING,
        "infrastructure_ceiling_usd": config.PHASE_7_INFRASTRUCTURE_CEILING_USD,
        "passed": (
            hours <= config.PHASE_7_FULLWIKI_HOURS_CEILING
            and infrastructure <= config.PHASE_7_INFRASTRUCTURE_CEILING_USD
            and software <= config.PHASE_7_SOFTWARE_COST_CEILING_USD
        ),
    }


def _row(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = payload["extraction"]
    retention = _retention_of(payload)
    economics = _economics_of(payload)
    bm25_line = config.PHASE_7_REFERENCE_DEV_FULL_SUPPORT["hybrid-bm25"]
    return {
        "extractor_id": payload["extractor_id"],
        "model": manifest["model"],
        "revision": manifest["revision"],
        "labels": manifest["labels"],
        "configuration_digest": manifest["configuration_digest"],
        "extraction_digest": manifest["digest"],
        "node_index_digest": payload["index"]["digest"],
        "index": payload["index"],
        "hop": payload["hop"],
        "fusion": {"scheme": payload["fit"]["scheme"], "weights": payload["fit"]["weights"]},
        "dev_metrics": payload["metrics"],
        "dev_cost": payload["cost"],
        "extraction_cost": {
            "seconds": manifest["seconds"],
            "paragraphs_per_second": manifest["paragraphs_per_second"],
            "usd": manifest["usd"],
            "failures": manifest["failures"],
            "failure_rate": manifest["failure_rate"],
            "hardware": manifest["hardware"],
            "projection_5m": manifest["projection_5m"],
        },
        "retention": retention,
        "economics": economics,
        "bm25_dev_line": {
            "full_support": bm25_line,
            "below_line": retention["full_support_at_headline"] < bm25_line,
            "note": (
                "a reported qualification, never a rejection: a free lexical complement did "
                "at least as well on dev"
            ),
        },
        "passed_both_bars": retention["passed"] and economics["passed"],
    }


def _rank_key(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
    """Lowest projected FullWiki cost, then time; ties by dev headline, then by id."""
    projection = row["extraction_cost"]["projection_5m"]
    return (
        float(projection["usd"]),
        float(projection["hours"]),
        -float(row["retention"]["full_support_at_headline"]),
        str(row["extractor_id"]),
    )


def select_extractor(phase_7_dir: Path) -> Path:
    """D11: apply the rule as written, over every dev reading, before any test run."""
    phase_7_dir = Path(phase_7_dir)
    target = phase_7_dir / SELECTION_FILENAME
    if target.exists():
        raise CheapEvaluationError(f"{target} already names a selection; it is written once")
    if (phase_7_dir / TEST_FILENAME).exists():
        raise CheapEvaluationError("a held-out run already exists; selection cannot be rewritten")

    rows: list[dict[str, Any]] = []
    absent: list[str] = []
    for extractor_id in config.PHASE_7_EXTRACTORS:
        path = phase_7_dir / extractor_id / DEV_FILENAME
        if not path.exists():
            absent.append(extractor_id)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("digest") != digest_of_payload(payload):
            raise CheapEvaluationError(f"{path} does not match its digest; it was edited")
        if payload["extractor_id"] != extractor_id or payload["split"] != "dev":
            raise CheapEvaluationError(f"{path} is not the dev reading of {extractor_id}")
        rows.append(_row(payload))
    if not rows:
        raise CheapEvaluationError("no dev measurement exists: run 'cheap-eval' first")

    eligible = sorted((row for row in rows if row["passed_both_bars"]), key=_rank_key)
    selected = str(eligible[0]["extractor_id"]) if eligible else None
    if selected is None:
        reason = (
            "no candidate cleared both bars, so Phase 7 closes with the negative result the "
            "spec provides for: no cheap extractor measured here keeps enough of the Entity "
            "Hop gain at a viable extraction cost, and no held-out run follows"
        )
    else:
        reason = (
            f"{selected} cleared the retention bar and the economic bar, and ranked first on "
            "projected FullWiki cost and time among the candidates that did"
        )
    payload = {
        "phase": 7,
        "rule": {
            "retention_share": config.PHASE_7_RETENTION_SHARE,
            "retention_bar": config.PHASE_7_RETENTION_BAR,
            "retention_bar_questions": config.PHASE_7_RETENTION_BAR_QUESTIONS,
            "headline_metric": "full_support",
            "headline_budget": HEADLINE_BUDGET,
            "software_cost_ceiling_usd": config.PHASE_7_SOFTWARE_COST_CEILING_USD,
            "fullwiki_hours_ceiling": config.PHASE_7_FULLWIKI_HOURS_CEILING,
            "infrastructure_ceiling_usd": config.PHASE_7_INFRASTRUCTURE_CEILING_USD,
            "ranking": "lowest projected FullWiki usd, then hours, then higher dev full support",
            "bm25_dev_line": config.PHASE_7_REFERENCE_DEV_FULL_SUPPORT["hybrid-bm25"],
        },
        "reference_dev_full_support": dict(config.PHASE_7_REFERENCE_DEV_FULL_SUPPORT),
        "reference_dev_files": dict(config.PHASE_7_REFERENCE_DEV_FILES),
        "candidates": rows,
        "candidates_without_dev_measurement": absent,
        "ranking": [str(row["extractor_id"]) for row in eligible],
        "selected": selected,
        "reason": reason,
    }
    payload["digest"] = digest_of_payload(payload)
    return write_text_atomic(target, json.dumps(payload, indent=2, sort_keys=True))


def load_selection(phase_7_dir: Path) -> dict[str, Any]:
    """The written selection, verified against its own digest."""
    path = Path(phase_7_dir) / SELECTION_FILENAME
    if not path.exists():
        raise CheapEvaluationError(f"{path} does not exist: run 'cheap-eval' first")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("digest") != digest_of_payload(payload):
        raise CheapEvaluationError(f"{path} does not match its digest; it was edited")
    return payload


# --- The single held-out run (D12) -------------------------------------------------------


def measure_held_out(
    phase_7_dir: Path,
    *,
    unit_ids: Sequence[str],
    dense: Retriever,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    run_config: Mapping[str, Any],
) -> Path:
    """Measure the selected extractor on test, once, at the scheme and weight dev chose.

    Nothing is refitted and no other candidate is measured: this reads the held-out
    split exactly once, for the one configuration a written selection already names.
    """
    phase_7_dir = Path(phase_7_dir)
    target = phase_7_dir / TEST_FILENAME
    if target.exists():
        raise CheapEvaluationError(f"{target} already holds the held-out run; it is read once")
    selection = load_selection(phase_7_dir)
    selected = selection["selected"]
    if selected is None:
        raise CheapEvaluationError(
            "the selection names no extractor, so there is nothing to read test for; a "
            "negative phase closes on its dev figures"
        )
    intruders = sorted({question.split for question in questions if question.split != TEST_SPLIT})
    if intruders or not questions:
        raise CheapEvaluationError(
            f"the held-out run was handed questions of split(s) {intruders or ['none']}; it "
            "measures test and nothing else"
        )
    if run_config.get("unit_set_hash") != unit_set_hash(unit_ids):
        raise CheapEvaluationError("the run configuration belongs to another pool")

    directory = phase_7_dir / str(selected)
    dev = json.loads((directory / DEV_FILENAME).read_text(encoding="utf-8"))
    if dev.get("digest") != digest_of_payload(dev):
        raise CheapEvaluationError(f"{directory / DEV_FILENAME} does not match its digest")
    index = load_node_index(
        directory, extraction_digest=dev["extraction"]["digest"], expected_unit_ids=unit_ids
    )
    if index.digest != dev["index"]["digest"]:
        raise CheapEvaluationError("the node index on disk is not the one dev was measured on")

    scheme, weights = dev["fit"]["scheme"], dev["fit"]["weights"]
    provenance = {
        **run_config,
        "phase": 7,
        "extractor_id": selected,
        "extraction_digest": dev["extraction"]["digest"],
        "extractor_configuration": dev["extraction"]["configuration"],
        "configuration_digest": dev["extraction"]["configuration_digest"],
        "node_index_digest": index.digest,
        "normalization_version": index.normalization_version,
        "read_depth": config.PILOT_READ_DEPTH,
        "entity_types": list(config.ENTITY_HOP_TYPES),
        "selection_digest": selection["digest"],
        "fitted_on": "dev",
    }
    hybrid = FusedRetriever(
        [dense, EntityHopStage(index, node_weights(index))], scheme=scheme, weights=weights
    )
    result = evaluate_retriever(
        hybrid,
        questions=questions,
        token_counts=token_counts,
        budgets=config.CONTEXT_BUDGETS,
        ks=config.RECALL_AT_K,
        top_k=int(provenance["top_k"]),
        config={**provenance, **hybrid.describe()},
        split=TEST_SPLIT,
    )
    run_path = result.save(directory)
    payload = {
        "phase": 7,
        "extractor_id": selected,
        "split": TEST_SPLIT,
        "n_questions": len(questions),
        "config": result.config,
        "fusion": {"scheme": scheme, "weights": weights},
        "metrics": result.metrics,
        "cost": result.cost,
        "selection_digest": selection["digest"],
        "dev_digest": dev["digest"],
        "inherited_test_full_support": dict(config.PHASE_7_REFERENCE_HELD_OUT_FULL_SUPPORT),
        "inherited_test_files": {
            name: f"{config.REPLACEMENT_DIR.name}/{filename}"
            for name, filename in config.PHASE_7_REFERENCE_HELD_OUT_FILES.items()
        },
        "run_file": f"{selected}/{run_path.name}",
        "run_digest": digest_of_payload(asdict(result)),
    }
    payload["digest"] = digest_of_payload(payload)
    return write_text_atomic(target, json.dumps(payload, indent=2, sort_keys=True))
