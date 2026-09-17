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

import hashlib
import importlib.metadata
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from concept_embeddings_rag import __version__, config, decision_parameters
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.embeddings.backend import EmbeddingBackend
from concept_embeddings_rag.embeddings.cache import (
    EmbeddingCache,
    build_query_backend,
    question_cache_key,
    question_set_hash,
)
from concept_embeddings_rag.evaluation.continuity import (
    ContinuityCheck,
    ContinuityReport,
    check_continuity,
)
from concept_embeddings_rag.evaluation.control_reproduction import (
    REMEASURED,
    REUSED,
    Check,
    ReproductionReport,
    compare_test_reproduction,
    dev_reproduction,
)
from concept_embeddings_rag.evaluation.entity_diagnostics import (
    RecordingHybrid,
    build_diagnostics_and_traces,
    records_of,
    run_digest,
    save_diagnostics_and_traces,
)
from concept_embeddings_rag.evaluation.harness import QuestionOutcome, RunResult, evaluate_retriever
from concept_embeddings_rag.evaluation.outcomes import (
    PerQuestionOutcomes,
    build_outcomes,
    load_outcomes,
    save_outcomes,
)
from concept_embeddings_rag.evaluation.readout import build_readout, save_readout
from concept_embeddings_rag.evaluation.replacement_decision import (
    DECISION_FILENAME,
    DecisionError,
    check_parameters,
    compute_decision,
    save_decision,
)
from concept_embeddings_rag.evaluation.replacement_freeze import (
    FREEZE_PATTERN,
    FreezeError,
    MeasuredFit,
    Phase6Freeze,
    build_freeze,
    load_freeze,
    point_label,
    reproducibility_block,
    save_freeze,
)
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
    SelectionError,
    check_dev_only,
    check_freeze_precedes,
    digest_of_payload,
    fit_fusion_weight,
    serialize_payload,
)
from concept_embeddings_rag.evaluation.tango import TangoError, require_published_example
from concept_embeddings_rag.nodes.extraction import OK
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.bm25 import BM25Retriever
from concept_embeddings_rag.retrieval.dense import DenseRetriever
from concept_embeddings_rag.retrieval.entity_hop import EntityHopStage
from concept_embeddings_rag.retrieval.fusion import FusedRetriever

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


def _resolve_existing(path: str) -> bool:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = config.PROJECT_ROOT / candidate
    return candidate.is_file()


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
    dense: Retriever, second: Retriever | EntityHopStage, fit_scheme: str, fit_weights: Any
) -> RecordingHybrid:
    """The one construction both hybrids come out of, with the same keywords, recorded (D17)."""
    return RecordingHybrid(dense, second, scheme=fit_scheme, weights=fit_weights)


def fit_hybrid(
    dense: Retriever,
    second: Retriever | EntityHopStage,
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
    """The one harness call of the stages: result and outcomes, written and verified.

    Every question must be on the split the call names: a test question handed to a dev
    measurement, or hidden among dev ones, is refused before anything is retrieved.
    """
    intruders = sorted({question.split for question in questions if question.split != split})
    if intruders or not questions:
        raise ReplacementRunError(
            f"a {split} measurement was handed questions of split(s) {intruders or ['none']}; it "
            "measures the split it names and nothing else"
        )
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
        run_config=hybrid_config(
            run_config, control, control_configuration="historical, from selection.json"
        ),
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


# --- replace-freeze ---------------------------------------------------------------------------


def _load_run(directory: Path, run_file: str, expected_digest: str) -> RunResult:
    path = directory / run_file
    if not path.exists():
        raise ReplacementRunError(f"the dev result {run_file} named by the checks is missing")
    result = RunResult(**json.loads(path.read_text(encoding="utf-8")))
    if run_digest(result) != expected_digest:
        raise ReplacementRunError(f"the dev result {run_file} changed since the checks ran")
    return result


def _load_checked_run(
    directory: Path, entry: Mapping[str, Any], split_order: Sequence[str]
) -> tuple[RunResult, PerQuestionOutcomes]:
    result = _load_run(directory, str(entry["run_file"]), str(entry["run_digest"]))
    outcomes = load_outcomes(
        directory / str(entry["outcomes_file"]), split_order=split_order, run_dir=directory
    )
    if outcomes.digest != entry["outcomes_digest"]:
        raise ReplacementRunError(f"{entry['outcomes_file']} changed since the checks were written")
    return result, outcomes


def _in_memory_outcomes(
    result: RunResult, records: Sequence[QuestionOutcome], split_order: Sequence[str]
) -> PerQuestionOutcomes:
    """A second run's outcomes, built and verified like the first, and compared, never written."""
    payload = build_outcomes(
        result, records, run_file="not-written", split_order=split_order, freeze_digest=None
    )
    return PerQuestionOutcomes(
        system=str(payload["system"]),
        split=str(payload["split"]),
        questions=tuple(payload["questions"]),
        aggregates=dict(payload["aggregates"]),
        run_result=dict(payload["run_result"]),
        freeze_digest=None,
        digest=digest_of_payload(payload),
        path=Path("not-written"),
    )


def rerun(
    retriever: Retriever,
    *,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    run_config: Mapping[str, Any],
) -> PerQuestionOutcomes:
    """D19: the second dev run of a system, over unchanged inputs, kept in memory."""
    check_dev_only(questions)
    records: list[QuestionOutcome] = []
    result = evaluate_retriever(
        retriever,
        questions=questions,
        token_counts=token_counts,
        budgets=config.CONTEXT_BUDGETS,
        ks=config.RECALL_AT_K,
        top_k=config.EVALUATION_TOP_K,
        config=dict(run_config),
        split=DEV_SPLIT,
        outcomes=records,
    )
    return _in_memory_outcomes(result, records, [question.qid for question in questions])


def _continuity_report(payload: Mapping[str, Any]) -> ContinuityReport:
    return ContinuityReport(
        checks=tuple(
            ContinuityCheck(
                name=str(entry["name"]),
                agreements=int(entry["agreements"]),
                total=int(entry["total"]),
                mismatches=tuple(str(item) for item in entry["mismatches"]),
            )
            for entry in payload["checks"]
        )
    )


def _reproduction_report(payload: Mapping[str, Any]) -> ReproductionReport:
    return ReproductionReport(
        checks=tuple(
            Check(str(entry["name"]), entry["expected"], entry["observed"])
            for entry in payload["checks"]
        )
    )


def hybrid_config(
    run_config: Mapping[str, Any], hybrid: RecordingHybrid, **extra: Any
) -> dict[str, Any]:
    return {
        **run_config,
        "fusion_scheme": hybrid.scheme,
        "fusion_weights": hybrid.inner.weights,
        "fitted_on": hybrid.inner.fitted_on,
        **extra,
    }


def protocol_of(inputs: VerifiedInputs) -> dict[str, Any]:
    identities = inputs.identities()
    return {
        "unit_set_hash": inputs.pool_hash,
        "manifest": {
            "sha256": inputs.manifest.sha256,
            "seed": inputs.manifest.seed,
            "n_units": inputs.manifest.n_units,
            "unit_set_hash": inputs.manifest.unit_set_hash,
            "dataset": inputs.manifest.dataset,
        },
        "split_sizes": dict(inputs.manifest.split_sizes),
        "question_set_hashes": dict(inputs.question_set_hashes),
        "model": inputs.pins.model,
        "revision": inputs.pins.revision,
        "tokenizer": inputs.pins.tokenizer,
        "token_counts_sha256": inputs.token_counts_sha256,
        "embeddings": identities["embeddings"],
        "budgets": list(config.CONTEXT_BUDGETS),
        "top_k": inputs.pins.top_k,
        "recall_depths": list(config.RECALL_AT_K),
        "metrics": ["gold_recall", "full_support", "precision", "recall_at_k", "units_included"],
        "seed": inputs.pins.seed,
        "code_version": __version__,
        "libraries": library_versions(),
    }


def failed_extractions(inputs: VerifiedInputs) -> set[str]:
    return {unit_id for unit_id, record in inputs.extraction.items() if record.status != OK}


def _existing_freeze(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if FREEZE_PATTERN.match(path.name))


def run_freeze(
    environment: StageEnvironment,
    *,
    control_mode: str | None = None,
    deviation: str | None = None,
    supersede: bool = False,
) -> StageResult:
    """D11 steps 4-8 on dev: fit B, measure it, check reproducibility, freeze, dev readings.

    `supersede` is the one way back to dev after test (D14, OI-4): after a stop at step 5 whose
    cause invalidates the dev fit, a re-measured control and a new freeze superseding the valid
    one, named by an existing deviation document.
    """
    directory = Path(environment.replacement_dir)
    supersedes: str | None = None
    if supersede:
        if not _existing_freeze(directory):
            raise ReplacementRunError("there is no freeze to supersede")
        if control_mode != REMEASURED or deviation is None or not _resolve_existing(deviation):
            raise ReplacementRunError(
                "a superseding freeze needs --control-mode re-measured and an existing deviation "
                "document naming the cause"
            )
        try:
            supersedes = load_freeze(directory).digest
        except FreezeError as error:
            raise ReplacementRunError(
                f"the freeze to supersede does not verify: {error}"
            ) from error
    elif _existing_freeze(directory):
        raise ReplacementRunError(
            f"a freeze already exists in {directory}; the freeze is written once, and a "
            "superseding one only on the deviation branch of D14"
        )
    checks = load_checks(directory)
    if checks["continuity"]["passed"] is not True or checks["control"] is None:
        raise ReplacementRunError("replace-check did not pass continuity; no freeze is written")
    checked_mode = str(checks["control"]["mode"])
    mode = control_mode or REUSED
    if mode not in (REUSED, REMEASURED):
        raise ReplacementRunError(f"unknown control mode {mode!r}")
    if mode == REUSED and checked_mode != REUSED:
        raise ReplacementRunError(
            "replace-check did not reproduce the control; the freeze requires --control-mode "
            "re-measured --deviation <existing document>"
        )
    control_deviation: str | None = None
    if mode == REMEASURED:
        if checked_mode == REUSED and not supersede:
            raise ReplacementRunError(
                "every control check passed; re-measured applies only after a failed check"
            )
        if deviation is None or not _resolve_existing(deviation):
            raise ReplacementRunError(
                "re-measured needs an existing deviation document recording the mismatch"
            )
        control_deviation = deviation

    inputs = verified_inputs(environment)
    if checks["identities"] != json.loads(json.dumps(inputs.identities())):
        raise ReplacementRunError("the inputs differ from the ones replace-check verified")
    questions = dev_questions(inputs)
    order = [question.qid for question in questions]
    dense, bm25, stage = build_components(inputs, inputs.dev_query_backend)
    run_config = base_config(inputs, inputs.dev_question_cache_key)
    runs = checks["runs"]
    dense_result, dense_outcomes = _load_checked_run(directory, runs[DENSE], order)
    control_result, control_outcomes = _load_checked_run(directory, runs[CONTROL], order)
    control_fit = measured_fit_from_payload(checks["control"]["fit"])

    # Step 4: B's fit, by the same helper as the control's.
    entity_fit = fit_hybrid(
        dense, stage, questions=questions, token_counts=inputs.token_counts, run_config=run_config
    )
    fit = entity_fit.fit
    print(
        f"[INFO] B's fit: {fit.winning_scheme} at w={fit.best_weight:.1f} "
        f"({fit.best_weighted_score:.4f} weighted against {fit.rrf_score:.4f} rrf)"
    )

    # Step 5: B at the selected configuration, recorded; its figures must be the fit's reading.
    entity = build_hybrid(dense, stage, fit.winning_scheme, fit.weights)
    entity_run = measure(
        entity,
        questions=questions,
        split=DEV_SPLIT,
        token_counts=inputs.token_counts,
        run_config=hybrid_config(run_config, entity, entity_component=entity_identity(inputs)),
        directory=directory,
        freeze_digest=None,
    )
    reading = entity_fit.points[point_label(entity.inner)]
    if entity_run.result.metrics != reading.metrics:
        raise ReplacementRunError(
            "B's selected-configuration aggregates differ from the fit's reading at that point; "
            "the measurement is not deterministic and the phase stops"
        )

    control_hybrid_fit = control_fit.fit if mode == REMEASURED else inputs.selection.control
    if control_hybrid_fit is None:
        raise ReplacementRunError("the selection carries no control fit")
    if mode == REMEASURED:
        remeasured = build_hybrid(
            dense, bm25, control_hybrid_fit.winning_scheme, control_hybrid_fit.weights
        )
        control_run = measure(
            remeasured,
            questions=questions,
            split=DEV_SPLIT,
            token_counts=inputs.token_counts,
            run_config=hybrid_config(run_config, remeasured, control_mode=REMEASURED),
            directory=directory,
            freeze_digest=None,
        )
        control_result, control_outcomes = control_run.result, control_run.outcomes

    # Step 6: reproducibility, a second dev run of the three systems, compared per question.
    control_again = build_hybrid(
        dense, bm25, control_hybrid_fit.winning_scheme, control_hybrid_fit.weights
    )
    entity_again = build_hybrid(dense, stage, fit.winning_scheme, fit.weights)
    second = {
        DENSE: rerun(
            dense, questions=questions, token_counts=inputs.token_counts, run_config=run_config
        ),
        CONTROL: rerun(
            control_again,
            questions=questions,
            token_counts=inputs.token_counts,
            run_config=hybrid_config(run_config, control_again),
        ),
        ENTITY: rerun(
            entity_again,
            questions=questions,
            token_counts=inputs.token_counts,
            run_config=hybrid_config(run_config, entity_again),
        ),
    }
    first = {DENSE: dense_outcomes, CONTROL: control_outcomes, ENTITY: entity_run.outcomes}
    reproducibility = reproducibility_block(first, second)
    if not reproducibility["passed"]:
        raise ReplacementRunError(
            f"a second dev run did not reproduce every outcome ({reproducibility['systems']}); "
            "the source of nondeterminism is to be recorded and the phase stops"
        )

    # Step 7: the freeze.
    history = inputs.selection.control
    if history is None:
        raise ReplacementRunError("the selection carries no control fit")
    freeze = build_freeze(
        protocol=protocol_of(inputs),
        control_mode=mode,
        reproduction=_reproduction_report(checks["control"]["reproduction"]),
        selection={"digest": inputs.selection_digest, "frozen_at": inputs.selection.frozen_at},
        historical=inputs.identities()["historical"],
        history=history,
        control=control_fit,
        entity_component=entity_identity(inputs),
        continuity=_continuity_report(checks["continuity"]),
        continuity_digests=continuity_digests(inputs),
        entity=entity_fit,
        dev_results={
            DENSE: (dense_result, dense_outcomes),
            CONTROL: (control_result, control_outcomes),
            ENTITY: (entity_run.result, entity_run.outcomes),
        },
        reproducibility=reproducibility,
        seed=inputs.pins.seed,
        code_version=__version__,
        control_deviation=control_deviation,
        supersedes=supersedes,
        deviation=deviation if supersede else None,
    )
    freeze_path = save_freeze(freeze, directory)
    frozen = load_freeze(directory)
    print(f"[OK] freeze written at {frozen.frozen_at} -> {freeze_path}")

    # Step 8: dev diagnostics, traces and readout, bound to the freeze.
    write_dev_readings(
        inputs,
        directory=directory,
        frozen=frozen,
        questions=questions,
        entity=entity,
        control_recording=control_again,
        control=(control_result, control_outcomes),
        entity_measured=(entity_run.result, entity_run.outcomes),
        dense_measured=(dense_result, dense_outcomes),
        index=len(_existing_freeze(directory)),
    )
    return StageResult(True, freeze_path, f"frozen in {mode} mode; commit it before replace-test")


def write_dev_readings(
    inputs: VerifiedInputs,
    *,
    directory: Path,
    frozen: Phase6Freeze,
    questions: Sequence[Question],
    entity: RecordingHybrid,
    control_recording: RecordingHybrid,
    control: tuple[RunResult, PerQuestionOutcomes],
    entity_measured: tuple[RunResult, PerQuestionOutcomes],
    dense_measured: tuple[RunResult, PerQuestionOutcomes],
    index: int = 1,
) -> None:
    b_records = records_of(entity, questions)
    diagnostics, traces = build_diagnostics_and_traces(
        split=DEV_SPLIT,
        questions=questions,
        b_records=b_records,
        a_records=records_of(control_recording, questions),
        b_outcomes=entity_measured[1],
        a_outcomes=control[1],
        token_counts=inputs.token_counts,
        failed_units=failed_extractions(inputs),
        b_scheme=entity.scheme,
        freeze_digest=frozen.digest,
        run_digests={CONTROL: run_digest(control[0]), ENTITY: run_digest(entity_measured[0])},
        node_index_digest=inputs.node_index.digest,
        extraction_digest=inputs.pins.extraction_digest,
    )
    save_diagnostics_and_traces(diagnostics, traces, directory, index)
    readout = build_readout(
        split=DEV_SPLIT,
        outcomes={DENSE: dense_measured[1], CONTROL: control[1], ENTITY: entity_measured[1]},
        questions=questions,
        read_lists={
            record.qid: [unit for unit, _ in record.dense[: config.PILOT_READ_DEPTH]]
            for record in b_records
        },
        costs={
            DENSE: dense_measured[0].cost,
            CONTROL: control[0].cost,
            ENTITY: entity_measured[0].cost,
        },
        freeze={**frozen.payload, "digest": frozen.digest},
        pilot_qids=[entry.qid for entry in inputs.pilot.questions],
    )
    save_readout(readout, directory, index)
    comparison = readout["subgroup"]["pilot_comparison"]
    print(
        f"[INFO] dev bridge-like subgroup: {readout['subgroup']['n_questions']} questions; equal "
        f"to the pilot's: {comparison['equal']}"
    )


# --- replace-test -----------------------------------------------------------------------------

TEST_REPRODUCTION_FILENAME = "test-reproduction.json"
HELD_OUT = decision_parameters.DECISION_SPLIT
COUNT_TOLERANCE = 1e-9


def read_historical_test_figures(frozen: Phase6Freeze, results_dir: Path) -> dict[str, Any]:
    """The only reader of the historical test figures (D2), called once, at step 5 of D14.

    It takes the loaded, verified freeze and re-checks each file's bytes against the sha256 the
    freeze recorded before returning a figure.
    """
    if not isinstance(frozen, Phase6Freeze):
        raise ReplacementRunError("the historical test figures are read only under a loaded freeze")
    recorded = frozen.payload["control"]["historical_files"]
    figures: dict[str, Any] = {}
    for system in (DENSE, CONTROL):
        identity = recorded[f"{system}/{HELD_OUT}"]
        path = Path(results_dir) / str(identity["name"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != identity["sha256"]:
            raise ReplacementRunError(
                f"{identity['name']} no longer has the sha256 the freeze recorded"
            )
        document = json.loads(raw.decode("utf-8"))
        if document.get("system") != system or document.get("split") != HELD_OUT:
            raise ReplacementRunError(f"{identity['name']} is not the {system} {HELD_OUT} result")
        figures[system] = {"metrics": document["metrics"], "cost": document["cost"]}
    return figures


def confirm_delta_source(figures: Mapping[str, Any], parameters: Mapping[str, Any]) -> dict:
    """D13: the frozen M1 checked against its recorded source, as integer counts, never decimals."""
    budget, metric, n = int(parameters["budget"]), str(parameters["metric"]), int(parameters["n"])
    details: dict[str, Any] = {}
    consistent = True
    for system in (CONTROL, DENSE):
        mean = float(figures[system]["metrics"][f"budget_{budget}"][metric])
        n_questions = figures[system]["cost"].get("n_questions")
        raw = mean * n
        count = round(raw)
        integral = abs(raw - count) <= COUNT_TOLERANCE
        details[system] = {
            "mean": mean,
            "n_questions": n_questions,
            "count": count,
            "integral": integral,
        }
        consistent = consistent and integral and n_questions == n
    difference = details[CONTROL]["count"] - details[DENSE]["count"]
    delta = parameters["delta"]
    passed = (
        consistent
        and difference == int(delta["m1_numerator"])
        and int(delta["m1_denominator"]) == n
    )
    return {
        "control_count": details[CONTROL]["count"],
        "dense_count": details[DENSE]["count"],
        "difference": difference,
        "expected_difference": int(delta["m1_numerator"]),
        "n": n,
        "details": details,
        "passed": passed,
    }


def _files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern)) if directory.exists() else []


def _next_free(directory: Path, filename: str) -> Path:
    path = directory / filename
    index = 2
    while path.exists():
        path = directory / filename.replace(".json", f"-{index}.json")
        index += 1
    return path


def load_test_reproductions(directory: Path) -> list[dict[str, Any]]:
    """Every `test-reproduction*.json` record in the directory, each verified against its digest."""
    records: list[dict[str, Any]] = []
    for path in _files(directory, "test-reproduction*.json"):
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("digest") != digest_of_payload(payload):
            raise ReplacementRunError(f"{path.name} does not match its digest; it was modified")
        records.append(payload)
    return records


def _stopped_by_a_failed_reproduction(record: Mapping[str, Any]) -> bool:
    """A reused test run that confirmed delta's source, then failed its reproduction before B."""
    checks = record.get("checks")
    delta = record.get("delta_confirmation") or {}
    return (
        record.get("mode") == REUSED
        and delta.get("passed") is True
        and isinstance(checks, list)
        and any(check.get("passed") is False for check in checks)
        and record.get("passed") is False
        and record.get("stops_run") is True
    )


def check_deviation_branch(
    directory: Path,
    frozen: Phase6Freeze,
    *,
    control_mode: str | None,
    deviation: str | None,
) -> None:
    """`--control-mode re-measured` on test, admitted only on the deviation branch of D14 (OI-4).

    Admitted if and only if every one of these holds:

    - the valid freeze supersedes an earlier freeze (and so names an existing deviation, which
      `load_freeze` verifies);
    - a test run under that earlier freeze, in reused mode, confirmed delta's source, failed its
      reproduction and stopped before B, and the dense and A results it wrote still exist;
    - nothing has been read on test under the superseding freeze yet;
    - the deviation document passed exists.

    A superseding freeze is read on test in no other way, with or without the flag. A freeze that
    is re-measured because replace-check found the control incompatible on dev (HU-2) supersedes
    nothing: its first test read takes the mode from the freeze, and the flag is refused there.
    """
    superseded = frozen.payload.get("supersedes")
    requested = control_mode == REMEASURED
    if superseded is None:
        if requested:
            raise ReplacementRunError(
                "--control-mode re-measured on test is admitted only on the deviation branch of "
                "D14 (OI-4): the valid freeze supersedes no earlier freeze, so no test run has "
                "stopped on a failed reproduction under a freeze it replaces"
            )
        return
    if not requested:
        raise ReplacementRunError(
            "the valid freeze supersedes an earlier one: it is read on test only with "
            "--control-mode re-measured --deviation <document>, on the deviation branch of D14"
        )
    if deviation is None or not _resolve_existing(deviation):
        raise ReplacementRunError("re-measured on test needs an existing deviation document")
    records = load_test_reproductions(directory)
    if any(record.get("freeze_digest") == frozen.digest for record in records):
        raise ReplacementRunError(
            "the superseding freeze has already been read on test; a further re-read needs a new "
            "deviation document and a new superseding freeze"
        )
    stopped = [
        record
        for record in records
        if record.get("freeze_digest") == superseded and _stopped_by_a_failed_reproduction(record)
    ]
    if not stopped:
        raise ReplacementRunError(
            "re-measured on test needs a reused test run under the superseded freeze that "
            "confirmed delta's source and stopped on a failed reproduction before B; none is "
            "recorded"
        )
    observed = stopped[-1].get("observed_runs") or {}
    missing = [
        system
        for system in (DENSE, CONTROL)
        if not observed.get(system) or not (directory / str(observed[system])).exists()
    ]
    if missing:
        raise ReplacementRunError(
            f"the {HELD_OUT} results of the stopped run are missing for {missing}; the deviation "
            "branch re-reads a run that exists"
        )


def _check_inputs_against_freeze(inputs: VerifiedInputs, frozen: Phase6Freeze) -> None:
    payload = frozen.payload
    for label, identity in payload["control"]["historical_files"].items():
        system, split = label.split("/")
        current = inputs.historical[(system, split)]
        if current.sha256 != identity["sha256"] or current.name != identity["name"]:
            raise ReplacementRunError(
                f"the historical file {label} is not the one the freeze bound"
            )
    protocol = payload["protocol"]
    expected = {
        "unit_set_hash": inputs.pool_hash,
        "token_counts_sha256": inputs.token_counts_sha256,
        "question_set_hashes": dict(inputs.question_set_hashes),
    }
    for key, value in expected.items():
        if protocol[key] != value:
            raise ReplacementRunError(f"the input {key} differs from the one the freeze recorded")
    component = payload["entity_component"]
    if component["node_index_digest"] != inputs.node_index.digest:
        raise ReplacementRunError("the node index differs from the one the freeze recorded")
    if component["extraction_digest"] != inputs.pins.extraction_digest:
        raise ReplacementRunError("the extraction differs from the one the freeze recorded")


def held_out_query_backend(
    inputs: VerifiedInputs, environment: StageEnvironment
) -> tuple[Any, str]:
    """The held-out question vectors, from the cache only; built here and nowhere earlier."""
    questions = [question for question in inputs.questions if question.split == HELD_OUT]
    qids = [question.qid for question in questions]
    pins = inputs.pins
    key = question_cache_key(pins.model, pins.revision, question_set_hash(qids), HELD_OUT, True)
    cache = EmbeddingCache(Path(environment.paths.question_cache_dir))
    if cache.load(key, expected_unit_ids=qids) is None:
        raise ReplacementRunError(f"the {HELD_OUT} question embeddings are not cached ({key})")
    return build_query_backend(questions, environment.backend, cache), key


def run_test(
    environment: StageEnvironment,
    *,
    control_mode: str | None = None,
    deviation: str | None = None,
) -> StageResult:
    """The ordered protocol of D14, once: dense, A, the reproduction check, then B and the state."""
    directory = Path(environment.replacement_dir)

    # Step 1: the valid freeze, its parameters, the inputs it bound, the published example.
    try:
        frozen = load_freeze(directory)
    except FreezeError as error:
        raise ReplacementRunError(f"replace-test requires a verified freeze: {error}") from error
    parameters = frozen.payload["decision_parameters"]
    try:
        check_parameters(parameters)
    except DecisionError as error:
        raise ReplacementRunError(str(error)) from error
    inputs = verified_inputs(environment)
    _check_inputs_against_freeze(inputs, frozen)
    try:
        require_published_example()
    except TangoError as error:
        raise ReplacementRunError(str(error)) from error
    if (directory / DECISION_FILENAME).exists():
        raise ReplacementRunError(f"{DECISION_FILENAME} exists; the decision is computed once")
    if _files(directory, f"run-{ENTITY}-{HELD_OUT}-*.json") or _files(
        directory, f"outcomes-{ENTITY}-{HELD_OUT}*.json"
    ):
        raise ReplacementRunError(f"a {ENTITY} {HELD_OUT} file exists; B is evaluated once")

    # Step 2: the mode, and the one admissible re-read of dense and A.
    freeze_mode = frozen.control_mode
    mode = freeze_mode if control_mode is None else control_mode
    if mode not in (REUSED, REMEASURED):
        raise ReplacementRunError(f"unknown control mode {mode!r}")
    check_deviation_branch(directory, frozen, control_mode=control_mode, deviation=deviation)
    if mode == REUSED and freeze_mode != REUSED:
        raise ReplacementRunError("the freeze records a re-measured control")
    existing = [
        *_files(directory, f"run-{DENSE}-{HELD_OUT}-*.json"),
        *_files(directory, f"run-{CONTROL}-{HELD_OUT}-*.json"),
    ]
    if existing and control_mode != REMEASURED:
        raise ReplacementRunError(
            f"a {HELD_OUT} result of dense or A already exists ({existing[0].name}); a re-read "
            "needs --control-mode re-measured --deviation <existing document>"
        )

    # Step 3: dense on test. The held-out question backend is built here and nowhere earlier.
    questions = [question for question in inputs.questions if question.split == HELD_OUT]
    query_backend, question_key = held_out_query_backend(inputs, environment)
    dense, bm25, stage = build_components(inputs, query_backend)
    run_config = {
        **base_config(inputs, question_key),
        "freeze_digest": frozen.digest,
        "frozen_at": frozen.frozen_at,
    }

    def precedes(result: RunResult) -> None:
        try:
            check_freeze_precedes(frozen, result)
        except SelectionError as error:
            raise ReplacementRunError(str(error)) from error

    dense_run = measure(
        dense,
        questions=questions,
        split=HELD_OUT,
        token_counts=inputs.token_counts,
        run_config=run_config,
        directory=directory,
        freeze_digest=frozen.digest,
        before_save=precedes,
    )

    # Step 4: A at the frozen configuration.
    control_fit = frozen.payload["control"]["fit"]
    control = build_hybrid(dense, bm25, control_fit["winning_scheme"], control_fit["weights"])
    control_run = measure(
        control,
        questions=questions,
        split=HELD_OUT,
        token_counts=inputs.token_counts,
        run_config=hybrid_config(run_config, control, control_mode=mode),
        directory=directory,
        freeze_digest=frozen.digest,
        before_save=precedes,
    )

    # Step 5: the historical test figures, first read here; delta's source; the reproduction.
    historical = read_historical_test_figures(frozen, inputs.paths.results_dir)
    delta_record = confirm_delta_source(historical, parameters)
    record: dict[str, Any] = {
        "split": HELD_OUT,
        "freeze_digest": frozen.digest,
        "mode": mode,
        "delta_confirmation": delta_record,
        "observed_runs": {
            DENSE: dense_run.run_path.name,
            CONTROL: control_run.run_path.name,
        },
    }
    reproduction_path = _next_free(directory, TEST_REPRODUCTION_FILENAME)
    if not delta_record["passed"]:
        _write_once(
            reproduction_path, {**record, "checks": None, "passed": False, "stops_run": True}
        )
        return StageResult(
            False,
            reproduction_path,
            "the historical Full Support counts do not confirm the frozen delta's source; the run "
            "stops before B, in either mode, and a deviation document identifies the cause",
        )
    held_out = compare_test_reproduction(
        {
            DENSE: {"metrics": dense_run.result.metrics, "cost": dense_run.result.cost},
            CONTROL: {"metrics": control_run.result.metrics, "cost": control_run.result.cost},
        },
        historical,
        mode=mode,
    )
    reproduction = held_out.as_payload()
    _write_once(reproduction_path, {**record, **reproduction})
    if held_out.stops_run:
        return StageResult(
            False,
            reproduction_path,
            f"{sum(1 for c in held_out.checks if not c.passed)} test reproduction check(s) failed "
            "in reused mode; no B figure exists. Write the deviation document before any re-run",
        )

    # Step 6: B on test, once.
    entity_fit = frozen.payload["entity_fit"]
    entity = build_hybrid(dense, stage, entity_fit["winning_scheme"], entity_fit["weights"])
    entity_run = measure(
        entity,
        questions=questions,
        split=HELD_OUT,
        token_counts=inputs.token_counts,
        run_config=hybrid_config(run_config, entity, entity_component=entity_identity(inputs)),
        directory=directory,
        freeze_digest=frozen.digest,
        before_save=precedes,
    )

    # Step 7: diagnostics and traces from the recorded lists; no second retrieval.
    b_records = records_of(entity, questions)
    diagnostics, traces = build_diagnostics_and_traces(
        split=HELD_OUT,
        questions=questions,
        b_records=b_records,
        a_records=records_of(control, questions),
        b_outcomes=entity_run.outcomes,
        a_outcomes=control_run.outcomes,
        token_counts=inputs.token_counts,
        failed_units=failed_extractions(inputs),
        b_scheme=entity.scheme,
        freeze_digest=frozen.digest,
        run_digests={
            CONTROL: run_digest(control_run.result),
            ENTITY: run_digest(entity_run.result),
        },
        node_index_digest=inputs.node_index.digest,
        extraction_digest=inputs.pins.extraction_digest,
    )
    save_diagnostics_and_traces(diagnostics, traces, directory)

    # Step 8: the decision, from the three outcome artifacts as read back from disk.
    order = [question.qid for question in questions]
    outcomes = {
        system: load_outcomes(measured.outcomes.path, split_order=order, run_dir=directory)
        for system, measured in ((DENSE, dense_run), (CONTROL, control_run), (ENTITY, entity_run))
    }
    decision_freeze = {
        **frozen.payload,
        "digest": frozen.digest,
        "control": {**frozen.payload["control"], "mode": mode},
    }
    decision = compute_decision(decision_freeze, outcomes, reproduction)
    decision_path = save_decision(decision, directory)

    # Step 9: the readout, descriptive only.
    readout = build_readout(
        split=HELD_OUT,
        outcomes=outcomes,
        questions=questions,
        read_lists={
            record.qid: [unit for unit, _ in record.dense[: config.PILOT_READ_DEPTH]]
            for record in b_records
        },
        costs={
            DENSE: dense_run.result.cost,
            CONTROL: control_run.result.cost,
            ENTITY: entity_run.result.cost,
        },
        freeze=decision_freeze,
    )
    save_readout(readout, directory)
    route = decision["route"] or "-"
    return StageResult(True, decision_path, f"state {decision['state']}, route {route}")
