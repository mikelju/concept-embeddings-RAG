"""The three Phase 6 stages: what `replace-check`, `replace-freeze` and `replace-test` run (D1).

Each stage verifies every inherited input first (D2, D3) and ends where the spec demands a stop
that may need a person:

- **`replace-check`** (dev, D11 steps 1-3): the inputs, the historical test files by identity
  only; continuity (a)-(d) with Phase 5, and on any mismatch a stop before any fit; then the
  control: `dense` on dev with outcomes, the re-executed fit of `[dense, bm25]` with every
  point's metrics, A at the historical configuration with outcomes, and the reproduction checks
  and mode of D12. `checks-dev.json` is written whatever the outcome; a failure exits non-zero,
  and no B figure exists.

Every harness call of these stages goes through one helper, and every fit through another, so
that both hybrids are measured and fitted with the same keywords (D10); `cli.py` holds neither
call. Every door that takes questions takes dev questions only.
"""

import importlib.metadata
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from concept_embeddings_rag import __version__, config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.embeddings.backend import EmbeddingBackend
from concept_embeddings_rag.evaluation.continuity import ContinuityReport, check_continuity
from concept_embeddings_rag.evaluation.control_reproduction import dev_reproduction
from concept_embeddings_rag.evaluation.entity_diagnostics import run_digest
from concept_embeddings_rag.evaluation.harness import QuestionOutcome, RunResult, evaluate_retriever
from concept_embeddings_rag.evaluation.outcomes import (
    PerQuestionOutcomes,
    build_outcomes,
    load_outcomes,
    save_outcomes,
)
from concept_embeddings_rag.evaluation.replacement_freeze import MeasuredFit, point_label
from concept_embeddings_rag.evaluation.replacement_inputs import (
    InputPaths,
    InputPins,
    ReplacementInputError,
    TokenCountFunction,
    VerifiedInputs,
    load_verified_inputs,
    read_dev_figures,
)
from concept_embeddings_rag.evaluation.second_hop import node_weights
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    FusionFit,
    check_dev_only,
    digest_of_payload,
    fit_fusion_weight,
    serialize_payload,
)
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.bm25 import BM25Retriever
from concept_embeddings_rag.retrieval.dense import DenseRetriever
from concept_embeddings_rag.retrieval.entity_hop import EntityHopStage
from concept_embeddings_rag.retrieval.fusion import FusedRetriever, SecondStage

CHECKS_FILENAME = "checks-dev.json"
LIBRARIES: tuple[str, ...] = ("numpy", "scipy", "bm25s", "transformers", "tokenizers")
BM25_STOPWORDS = "en"
DENSE, CONTROL, ENTITY = "dense", "hybrid-bm25", "hybrid-entity-hop"


class ReplacementRunError(Exception):
    """A stage refuses to run, or stops, for a reason it names."""


@dataclass(frozen=True)
class StageEnvironment:
    """What a stage reads and where it writes. The CLI builds it from `config`; tests from a toy."""

    paths: InputPaths
    pins: InputPins
    replacement_dir: Path
    backend: EmbeddingBackend
    token_counter: TokenCountFunction
    parse: Callable[[str], Mapping[str, Any]] = field(default=json.loads)


@dataclass(frozen=True)
class StageResult:
    passed: bool
    path: Path
    message: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


# --- Shared helpers: the one harness call, the one fit call, the one hybrid construction ------


def dev_questions(inputs: VerifiedInputs) -> list[Question]:
    """The dev split in pool order (D7), refused at the door if anything else is in it."""
    questions = [question for question in inputs.questions if question.split == DEV_SPLIT]
    check_dev_only(questions)
    return questions


def base_config(inputs: VerifiedInputs, question_cache_key: str) -> dict[str, Any]:
    """The provenance every Phase 6 result records (D22)."""
    pins = inputs.pins
    return {
        "model": pins.model,
        "revision": pins.revision,
        "resolved_revision": pins.revision,
        "unit_set_hash": inputs.pool_hash,
        "seed": pins.seed,
        "tokenizer": pins.tokenizer,
        "code_version": __version__,
        "top_k": pins.top_k,
        "n_units": len(inputs.unit_ids),
        "phase": 6,
        "libraries": library_versions(),
        "dense_identity": {
            "model": pins.model,
            "revision": pins.revision,
            "corpus_cache_key": inputs.corpus_cache_key,
            "question_cache_key": question_cache_key,
        },
        "bm25_identity": {
            "bm25s": library_versions()["bm25s"],
            "stopwords": BM25_STOPWORDS,
            "unit_set_hash": inputs.pool_hash,
        },
    }


def entity_identity(inputs: VerifiedInputs) -> dict[str, Any]:
    return {
        "node_index_digest": inputs.node_index.digest,
        "extraction_digest": inputs.pins.extraction_digest,
        "prompt_digest": inputs.pins.extraction_prompt_digest,
        "normalization_version": inputs.node_index.normalization_version,
        "types": list(config.ENTITY_HOP_TYPES),
        "weight_rule": "log(1 + N / (1 + df)) over binary node incidence (second_hop.node_weights)",
        "read_depth": config.PILOT_READ_DEPTH,
        "max_depth": config.ENTITY_HOP_MAX_DEPTH,
        "dense_order": "descending score, ties by unit id (DenseRetriever)",
    }


def build_components(
    inputs: VerifiedInputs, query_backend: EmbeddingBackend
) -> tuple[DenseRetriever, BM25Retriever, EntityHopStage]:
    dense = DenseRetriever(inputs.vectors, inputs.unit_ids, query_backend)
    bm25 = BM25Retriever(inputs.units, stopwords=BM25_STOPWORDS)
    stage = EntityHopStage(inputs.node_index, node_weights(inputs.node_index))
    return dense, bm25, stage


def build_hybrid(
    dense: Retriever, second: Retriever | SecondStage, fit_scheme: str, fit_weights: Any
) -> FusedRetriever:
    """The one construction both hybrids come out of, with the same keywords."""
    return FusedRetriever([dense, second], scheme=fit_scheme, weights=fit_weights)


def fit_hybrid(
    dense: Retriever,
    second: Retriever | SecondStage,
    *,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    run_config: Mapping[str, Any],
) -> MeasuredFit:
    """The one fitting call both hybrids go through, keeping every point's dev result (D10)."""
    check_dev_only(questions)
    points: dict[str, RunResult] = {}

    def keep(hybrid: FusedRetriever, result: RunResult) -> None:
        points[point_label(hybrid)] = result

    fit = fit_fusion_weight(
        [dense, second],
        questions=questions,
        token_counts=token_counts,
        base_config=run_config,
        on_measure=keep,
    )
    return MeasuredFit(fit=fit, points=points)


@dataclass(frozen=True)
class Measured:
    result: RunResult
    run_path: Path
    outcomes: PerQuestionOutcomes


def measure(
    retriever: Retriever,
    *,
    questions: Sequence[Question],
    split: str,
    token_counts: Mapping[str, int],
    run_config: Mapping[str, Any],
    directory: Path,
    freeze_digest: str | None,
    before_save: Callable[[RunResult], None] | None = None,
) -> Measured:
    """The one harness call of the stages: result and outcomes, written and verified."""
    records: list[QuestionOutcome] = []
    started = datetime.now(UTC)
    result = evaluate_retriever(
        retriever,
        questions=questions,
        token_counts=token_counts,
        budgets=config.CONTEXT_BUDGETS,
        ks=config.RECALL_AT_K,
        top_k=config.EVALUATION_TOP_K,
        config=dict(run_config),
        split=split,
        outcomes=records,
    )
    elapsed = (datetime.now(UTC) - started).total_seconds()
    print(f"[INFO] {retriever.name} on {split}: {len(questions)} questions in {elapsed:.1f} s")
    if before_save is not None:
        before_save(result)
    run_path = result.save(directory)
    order = [question.qid for question in questions]
    payload = build_outcomes(
        result, records, run_file=run_path.name, split_order=order, freeze_digest=freeze_digest
    )
    outcomes_path = save_outcomes(payload, directory)
    loaded = load_outcomes(outcomes_path, split_order=order, run_dir=directory)
    return Measured(result=result, run_path=run_path, outcomes=loaded)


def fit_payload(measured: MeasuredFit) -> dict[str, Any]:
    fit = measured.fit
    return {
        "components": list(fit.components),
        "metric": fit.metric,
        "budget": fit.budget,
        "n_questions": fit.n_questions,
        "grid": [float(weight) for weight in fit.grid],
        "curve": {repr(float(weight)): value for weight, value in fit.curve.items()},
        "rrf_score": fit.rrf_score,
        "winning_scheme": fit.winning_scheme,
        "best_weight": fit.best_weight,
        "weights": fit.weights,
        "points": {label: asdict(result) for label, result in sorted(measured.points.items())},
    }


def measured_fit_from_payload(payload: Mapping[str, Any]) -> MeasuredFit:
    dense_name, second_name = (str(name) for name in payload["components"])
    fit = FusionFit(
        components=(dense_name, second_name),
        metric=str(payload["metric"]),
        budget=int(payload["budget"]),
        n_questions=int(payload["n_questions"]),
        grid=tuple(float(weight) for weight in payload["grid"]),
        curve={float(weight): float(value) for weight, value in payload["curve"].items()},
        rrf_score=float(payload["rrf_score"]),
    )
    points = {label: RunResult(**result) for label, result in payload["points"].items()}
    return MeasuredFit(fit=fit, points=points)


def _write_once(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise ReplacementRunError(f"{path} already exists; it is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = {key: value for key, value in payload.items() if key != "digest"}
    sealed["digest"] = digest_of_payload(sealed)
    write_text_atomic(path, serialize_payload(sealed))
    return path


def load_checks(directory: Path) -> dict[str, Any]:
    path = Path(directory) / CHECKS_FILENAME
    if not path.exists():
        raise ReplacementRunError(f"no {CHECKS_FILENAME} in {directory}: run `cer replace-check`")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluated_on") != DEV_SPLIT:
        raise ReplacementRunError(f"{CHECKS_FILENAME} does not claim dev")
    if payload.get("digest") != digest_of_payload(payload):
        raise ReplacementRunError(f"{CHECKS_FILENAME} does not match its digest; it was modified")
    return payload


def verified_inputs(environment: StageEnvironment) -> VerifiedInputs:
    try:
        return load_verified_inputs(
            environment.paths,
            pins=environment.pins,
            backend=environment.backend,
            token_counter=environment.token_counter,
            parse=environment.parse,
        )
    except ReplacementInputError as error:
        raise ReplacementRunError(f"an inherited input does not verify: {error}") from error


def continuity_digests(inputs: VerifiedInputs) -> dict[str, str]:
    return {
        "pilot_digest": inputs.pins.pilot_digest,
        "hop_run_digest": inputs.pins.hop_run_digest,
        "traces_digest": inputs.pins.traces_digest,
    }


# --- replace-check ----------------------------------------------------------------------------


def run_continuity(
    inputs: VerifiedInputs, dense: Retriever, stage: EntityHopStage
) -> ContinuityReport:
    texts = {question.qid: question.question for question in dev_questions(inputs)}
    return check_continuity(
        inputs.pilot,
        inputs.hop_run,
        inputs.navigation_traces,
        dense=dense,
        stage=stage,
        question_text=texts,
    )


def run_check(environment: StageEnvironment) -> StageResult:
    """D11 steps 1-3 on dev. Writes `checks-dev.json` once; `passed` is False on any mismatch."""
    directory = Path(environment.replacement_dir)
    path = directory / CHECKS_FILENAME
    if path.exists():
        raise ReplacementRunError(
            f"{path} already exists; the checks run once. Move it aside deliberately, with a "
            "deviation, if a second check is meant"
        )
    inputs = verified_inputs(environment)
    questions = dev_questions(inputs)
    dense, bm25, stage = build_components(inputs, inputs.dev_query_backend)
    print(f"[OK] inputs verified: {inputs.counts}")

    continuity = run_continuity(inputs, dense, stage)
    payload: dict[str, Any] = {
        "evaluated_on": DEV_SPLIT,
        "created_at": _now(),
        "code_version": __version__,
        "identities": inputs.identities(),
        "continuity": {**continuity_digests(inputs), **continuity.as_payload()},
        "control": None,
        "runs": {},
        "passed": False,
    }
    for check in continuity.checks:
        status = "OK" if check.passed else "ERROR"
        print(f"[{status}] continuity {check.name}: {check.agreements} of {check.total} agree")
    if not continuity.passed:
        _write_once(path, payload)
        return StageResult(
            False,
            path,
            "continuity with Phase 5 failed; the phase stops before any fit and a deviation "
            "document records the mismatch",
        )

    run_config = base_config(inputs, inputs.dev_question_cache_key)
    selection_control = inputs.selection.control
    if selection_control is None:
        raise ReplacementRunError("the selection carries no control fit")

    dense_run = measure(
        dense,
        questions=questions,
        split=DEV_SPLIT,
        token_counts=inputs.token_counts,
        run_config=run_config,
        directory=directory,
        freeze_digest=None,
    )
    control_fit = fit_hybrid(
        dense, bm25, questions=questions, token_counts=inputs.token_counts, run_config=run_config
    )
    control = build_hybrid(dense, bm25, selection_control.winning_scheme, selection_control.weights)
    control_run = measure(
        control,
        questions=questions,
        split=DEV_SPLIT,
        token_counts=inputs.token_counts,
        run_config={
            **run_config,
            "fusion_scheme": control.scheme,
            "fusion_weights": control.weights,
            "fitted_on": control.fitted_on,
            "control_configuration": "historical, from selection.json",
        },
        directory=directory,
        freeze_digest=None,
    )

    historical_dev = {
        system: read_dev_figures(inputs.paths.results_dir, inputs.historical[(system, DEV_SPLIT)])
        for system in (DENSE, CONTROL)
    }
    report = dev_reproduction(
        frozen=selection_control,
        refit=control_fit.fit,
        historical_dev={
            system: {"metrics": figures.metrics, "cost": figures.cost}
            for system, figures in historical_dev.items()
        },
        observed_dev={
            DENSE: {"metrics": dense_run.result.metrics, "cost": dense_run.result.cost},
            CONTROL: {"metrics": control_run.result.metrics, "cost": control_run.result.cost},
        },
        historical_configs={
            f"{system}/{split}": identity.config
            for (system, split), identity in inputs.historical.items()
        },
        observed_configs={DENSE: dense_run.result.config, CONTROL: control_run.result.config},
    )
    payload["control"] = {
        "mode": report.mode,
        "reproduction": report.as_payload(),
        "fit": fit_payload(control_fit),
    }
    payload["runs"] = {
        system: {
            "run_file": measured.run_path.name,
            "run_digest": run_digest(measured.result),
            "outcomes_file": measured.outcomes.path.name,
            "outcomes_digest": measured.outcomes.digest,
        }
        for system, measured in ((DENSE, dense_run), (CONTROL, control_run))
    }
    payload["passed"] = report.passed
    _write_once(path, payload)
    for failed in report.failed():
        print(f"[ERROR] {failed.name}: expected {failed.expected!r}, observed {failed.observed!r}")
    if not report.passed:
        return StageResult(
            False,
            path,
            f"{len(report.failed())} control reproduction check(s) failed: the mode is "
            "re-measured; write the deviation document before `replace-freeze --control-mode "
            "re-measured --deviation <path>`",
        )
    return StageResult(True, path, "every check passed: the control mode is reused")
