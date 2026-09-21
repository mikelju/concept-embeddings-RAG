"""Deviation 8.1, S4-S5: exact retrieval over corpus prefixes, and the reproduction gate.

This module is the 8.1 mirror of `strong_dense.py`: it holds the pins, a named retriever
wrapper, and the gate that decides whether anything larger may be measured at all. It adds
no retrieval method and touches neither the harness nor `DenseRetriever`.

**Retrieval semantics are the existing ones, unchanged.** Corpus vectors are L2-normalized
upstream, the query vector is L2-normalized, the score is a dot product and therefore the
cosine, `EVALUATION_TOP_K` is 100, and ties break by unit id. Every evaluated corpus is an
index **prefix** of one frozen ordering, so `C19 < C100 < C250 < C500` holds by construction
rather than by assertion afterwards, and row `i` means the same paragraph at every size. No
approximate index appears here: no ANN, no HNSW, no IVF, no quantization.

**The reproduction gate, level 1.** The naive single gate would compare a number produced by
new code *and* new vectors against one produced by old code and old vectors - and when it
failed it could not say which had moved. So level 1 runs this new path over the unchanged
frozen C19 corpus using the **existing historical caches**, read-only. The vectors are then
bit-identical to the recorded runs, the only variable is the code, and both counts must match
exactly:

```text
BGE-small   487 / 600
Qwen        446 / 600
```

Any difference is a defect in the new path, not a tolerance to widen: it is recorded, with
`terminal_state = "reproduction_stop"`, before any implementation is changed. Level 2 - C19
as a prefix of the new C500 encode, under a 1-question tolerance - belongs to S6 and is a
different check with different consequences. Keeping the two counts in separate fields is
what makes the diagnosis possible at all.

Nothing here loads a model or reaches the network. `MetadataBackend` carries a model's
identity for cache keying and raises if anything asks it to encode, because at level 1 an
encode would mean the vectors were not the historical ones.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.embeddings.cache import CachedQueryBackend, unit_set_hash
from concept_embeddings_rag.evaluation.harness import RunResult, evaluate_retriever
from concept_embeddings_rag.evaluation.selection import check_dev_only, digest_of_payload
from concept_embeddings_rag.retrieval.base import Hit, Retriever
from concept_embeddings_rag.retrieval.dense import DenseRetriever

REPRODUCTION_FILENAME = "reproduction.json"

REPRODUCTION_STOP = "reproduction_stop"

HEADLINE_BUDGET = config.SELECTION_BUDGET


class ScaleSensitivityError(Exception):
    """A refusal on the 8.1 measurement path that must not be worked around."""


@dataclass(frozen=True)
class ModelPins:
    """One encoder's identity, read from `config` rather than retyped.

    Two declarations of one pin are two places for it to disagree with itself, so the
    values come from the constants Phases 1-8 already recorded.
    """

    key: str
    model: str
    revision: str
    dim: int
    query_prompt: str

    @classmethod
    def bge(cls) -> "ModelPins":
        return cls(
            key="bge",
            model=config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_REVISION,
            dim=config.EMBEDDING_DIM,
            # Historical symmetric Phase 1-7 behaviour: no query instruction at all.
            query_prompt="",
        )

    @classmethod
    def qwen(cls) -> "ModelPins":
        return cls(
            key="qwen",
            model=config.PHASE_8_DENSE_MODEL,
            revision=config.PHASE_8_DENSE_REVISION,
            dim=config.PHASE_8_DENSE_DIM,
            query_prompt=config.PHASE_8_QUERY_PROMPT,
        )

    @classmethod
    def for_key(cls, key: str) -> "ModelPins":
        if key == "bge":
            return cls.bge()
        if key == "qwen":
            return cls.qwen()
        raise ScaleSensitivityError(
            f"{key!r} is not one of the two encoders 8.1 compares: {config.PHASE_8_1_MODELS}"
        )


class MetadataBackend:
    """A model's identity without the model: enough to key a cache, unable to encode.

    Level 1 exists to hold the vectors constant. If anything on that path tried to embed a
    text, the vectors would no longer be the historical ones and the gate would be
    measuring something else, so this raises rather than quietly re-encoding.
    """

    def __init__(self, *, name: str, revision: str, query_prompt: str = "") -> None:
        self.name = name
        self.revision = revision
        self.query_prompt = query_prompt
        self.normalize = config.NORMALIZE_EMBEDDINGS

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise ScaleSensitivityError(
            f"{self.name} was asked to encode {len(texts)} texts, but this path reads a "
            "cache: an encode here would mean the vectors are not the historical ones"
        )


class ScaleDense:
    """A named `DenseRetriever`, so every (model, scale) run file is identifiable.

    The harness names a run after its system, and `RunResult.validate` requires that name to
    be filename-safe. Wrapping rather than subclassing keeps `DenseRetriever` untouched.
    """

    def __init__(self, inner: DenseRetriever, name: str) -> None:
        self.inner = inner
        self.name = name

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return self.inner.retrieve(query, top_k)


def run_name(model: str, size: int) -> str:
    """`dense-qwen-c500` and friends: one stable, filename-safe name per reading."""
    sizes = config.PHASE_8_1_CORPUS_SIZES
    if size not in sizes:
        raise ScaleSensitivityError(f"{size} is not one of the four declared corpus sizes {sizes}")
    label = config.PHASE_8_1_CORPUS_LABELS[sizes.index(size)]
    return f"dense-{model}-{label}"


def prefix_retriever(
    vectors: np.ndarray,
    unit_ids: Sequence[str],
    *,
    size: int,
    backend: Any,
    name: str,
) -> ScaleDense:
    """The existing exact retriever, pointed at the first `size` rows of one ordering."""
    if size > len(unit_ids):
        raise ScaleSensitivityError(
            f"a corpus of {size} was asked for but only {len(unit_ids)} vectors are cached"
        )
    if vectors.shape[0] != len(unit_ids):
        raise ScaleSensitivityError(
            f"{vectors.shape[0]} vectors for {len(unit_ids)} unit ids: rows and ids must align"
        )
    return ScaleDense(DenseRetriever(vectors[:size], list(unit_ids[:size]), backend), name)


def historical_retriever(
    vectors: np.ndarray,
    unit_ids: Sequence[str],
    *,
    questions: Sequence[Question],
    query_vectors: np.ndarray,
    qids: Sequence[str],
    pins: ModelPins,
    name: str,
) -> ScaleDense:
    """A retriever over cached corpus and cached query vectors. Nothing is embedded.

    Both caches were written by earlier phases and are opened read-only; the query backend
    serves them by lookup, exactly as Phase 6 and Phase 8 already do.
    """
    by_qid = {question.qid: question.question for question in questions}
    missing = [qid for qid in qids if qid not in by_qid]
    if missing:
        raise ScaleSensitivityError(
            f"the question cache holds {len(missing)} qids this split does not, first {missing[0]}"
        )
    backend = CachedQueryBackend(
        texts=[by_qid[qid] for qid in qids],
        vectors=query_vectors,
        name=pins.model,
        revision=pins.revision,
    )
    return prefix_retriever(vectors, unit_ids, size=len(unit_ids), backend=backend, name=name)


def required_counts() -> dict[str, int]:
    """The two counts level 1 must reproduce, as recorded in `config`."""
    return dict(config.PHASE_8_1_C19_DEV_SUPPORTED)


def _reading(
    result: RunResult, *, n_questions: int, required: int, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """One model's level-1 reading, with every budget kept beside the headline."""
    full_support = {
        str(budget): float(result.metrics[f"budget_{budget}"]["full_support"])
        for budget in config.CONTEXT_BUDGETS
    }
    supported = round(full_support[str(HEADLINE_BUDGET)] * n_questions)
    return {
        "system": result.system,
        "model": provenance.get("model"),
        "revision": provenance.get("revision"),
        "dim": provenance.get("dim"),
        "corpus_cache_key": provenance.get("corpus_cache_key"),
        "question_cache_key": provenance.get("question_cache_key"),
        "headline_budget": HEADLINE_BUDGET,
        "full_support": full_support,
        "recall_at_k": {
            str(k): float(result.metrics[f"recall_at_{k}"]) for k in config.RECALL_AT_K
        },
        "supported": supported,
        "required": required,
        "matches": supported == required,
        "n_questions": n_questions,
        "mean_latency_ms": float(result.cost["mean_latency_ms"]),
    }


def measure_level_1(
    directory: Path | str,
    *,
    retrievers: Mapping[str, Retriever],
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    provenance: Mapping[str, Mapping[str, Any]],
    host: Mapping[str, Any],
    required: Mapping[str, int] | None = None,
    expected_questions: int = config.N_DEV,
) -> Path:
    """Run the new evaluation path over frozen C19 and the historical caches.

    Blocking: S6 refuses to encode anything without a passing `reproduction.json`.
    """
    check_dev_only(questions)
    if len(questions) != expected_questions:
        raise ScaleSensitivityError(
            f"level 1 measures exactly the {config.N_DEV} dev questions, not {len(questions)}"
        )
    bars = dict(required) if required is not None else required_counts()
    unknown = sorted(set(retrievers) - set(bars))
    if unknown:
        raise ScaleSensitivityError(f"no required count is declared for {unknown}")

    directory = Path(directory)
    target = directory / REPRODUCTION_FILENAME
    if target.exists():
        raise ScaleSensitivityError(
            f"{target} already holds a reproduction reading; a gate is not re-run in place"
        )

    readings: dict[str, Any] = {}
    for key in sorted(retrievers):
        model_provenance = dict(provenance[key])
        result = evaluate_retriever(
            retrievers[key],
            questions=questions,
            token_counts=token_counts,
            budgets=config.CONTEXT_BUDGETS,
            ks=config.RECALL_AT_K,
            top_k=int(model_provenance["top_k"]),
            config=model_provenance,
            split="dev",
        )
        run_path = result.save(directory)
        reading = _reading(
            result,
            n_questions=len(questions),
            required=bars[key],
            provenance=model_provenance,
        )
        reading["run_file"] = run_path.name
        readings[key] = reading

    matches = all(reading["matches"] for reading in readings.values())
    body = {
        "level": 1,
        "corpus": "c19",
        "corpus_units": config.EXPECTED_N_UNITS,
        "questions": len(questions),
        "required": bars,
        "models": readings,
        "matches": matches,
        "terminal_state": None if matches else REPRODUCTION_STOP,
        "host": dict(host),
        "basis": (
            "the new 8.1 evaluation path over the unchanged frozen C19 corpus and the "
            "existing historical caches, so the only variable is the code"
        ),
    }
    write_text_atomic(target, json.dumps(body, indent=2, sort_keys=True))
    return target


def load_reproduction(directory: Path | str) -> dict[str, Any]:
    """The recorded level-1 reading, or a refusal naming the stage that writes it."""
    target = Path(directory) / REPRODUCTION_FILENAME
    if not target.exists():
        raise ScaleSensitivityError(f"{target} does not exist: run 'scale-repro' first")
    body: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return body


def check_reproduction(directory: Path | str) -> dict[str, Any]:
    """What S6 calls before it encodes anything. A stop is not negotiable here."""
    body = load_reproduction(directory)
    if body.get("terminal_state") == REPRODUCTION_STOP or not body.get("matches"):
        failed = [
            key
            for key, reading in sorted(body.get("models", {}).items())
            if not reading.get("matches")
        ]
        raise ScaleSensitivityError(
            f"the C19 reproduction gate recorded {REPRODUCTION_STOP} for {failed}: "
            "diagnose the new evaluation path before any larger corpus is interpreted"
        )
    return body


def embedding_filename(model: str) -> str:
    """One immutable C500 encoding artifact per model."""
    ModelPins.for_key(model)
    return f"embedding-{model}.json"


def write_embedding(
    directory: Path | str,
    *,
    model: str,
    resolved_revision: str,
    weights_sha256: str,
    query_prompt: str,
    corpus_cache_key: str,
    question_cache_key: str,
    unit_ids: Sequence[str],
    n_questions: int,
    dim: int,
    max_seq_length: int | None,
    selection_digest: str,
    hardware: Mapping[str, Any],
    embedding_seconds: float,
    question_seconds: float,
    bytes_on_disk: Mapping[str, int],
    hourly_rate_usd: float,
) -> Path:
    """Record the one C500 encode that every scale reading reuses."""
    pins = ModelPins.for_key(model)
    target = Path(directory) / embedding_filename(model)
    if target.exists():
        raise ScaleSensitivityError(
            f"{target.name} already records {model}'s C500 encoding; it is not rewritten"
        )
    if resolved_revision != pins.revision:
        raise ScaleSensitivityError(
            f"{model} resolved to {resolved_revision}, not the pinned {pins.revision}"
        )
    if query_prompt != pins.query_prompt:
        raise ScaleSensitivityError(
            f"{model} resolved a different query prompt from the frozen 8.1 prompt"
        )
    if dim != pins.dim:
        raise ScaleSensitivityError(
            f"{model} produced {dim}-dimensional vectors, not the pinned {pins.dim}"
        )
    expected_units = config.PHASE_8_1_CORPUS_SIZES[-1]
    if len(unit_ids) != expected_units:
        raise ScaleSensitivityError(
            f"{model} embedding contains {len(unit_ids)} units, not C500={expected_units}"
        )
    if n_questions != config.N_DEV:
        raise ScaleSensitivityError(
            f"{model} embedding contains {n_questions} questions, not the frozen {config.N_DEV}"
        )
    if corpus_cache_key == question_cache_key:
        raise ScaleSensitivityError("corpus and question embeddings cannot share one cache key")

    paragraphs_per_second = (
        len(unit_ids) / embedding_seconds if embedding_seconds > 0.0 else 0.0
    )
    embedding_usd = hourly_rate_usd * embedding_seconds / 3600.0
    body: dict[str, Any] = {
        "model": model,
        "model_id": pins.model,
        "revision": pins.revision,
        "resolved_revision": resolved_revision,
        "weights_sha256": weights_sha256,
        "query_prompt": query_prompt,
        "dim": dim,
        "max_seq_length": max_seq_length,
        "corpus_cache_key": corpus_cache_key,
        "question_cache_key": question_cache_key,
        "corpus_units": len(unit_ids),
        "corpus_unit_set_hash": unit_set_hash(unit_ids),
        "selection_digest": selection_digest,
        "n_questions": n_questions,
        "hardware": dict(hardware),
        "embedding_seconds": embedding_seconds,
        "paragraphs_per_second": paragraphs_per_second,
        "question_seconds": question_seconds,
        "hourly_rate_usd": hourly_rate_usd,
        "embedding_usd": embedding_usd,
        "bytes_on_disk": dict(bytes_on_disk),
        "cost_basis": (
            "embedding_usd is the machine-rate cost attributable to the C500 paragraph "
            "encoding pass only; the total rented-session charge is recorded separately "
            "in 8.1_results.md"
        ),
    }
    body["digest"] = digest_of_payload(body)
    Path(directory).mkdir(parents=True, exist_ok=True)
    write_text_atomic(target, json.dumps(body, indent=2, sort_keys=True))
    return target


def load_embedding(directory: Path | str, model: str) -> dict[str, Any]:
    """Load and verify the immutable C500 encoding artifact."""
    target = Path(directory) / embedding_filename(model)
    if not target.exists():
        raise ScaleSensitivityError(
            f"{target} does not exist: run 'scale-run --model {model}' first"
        )
    body: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    if body.get("digest") != digest_of_payload(body):
        raise ScaleSensitivityError(f"{target} does not match its digest; it was edited")
    return body


def measure_level_one_on_measurement_host(
    *,
    model: str,
    retriever: Retriever,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    provenance: Mapping[str, Any],
    host: Mapping[str, Any],
    required: int,
    expected_questions: int = config.N_DEV,
) -> dict[str, Any]:
    """Repeat level 1 on the measurement host without rewriting the laptop gate artifact."""
    check_dev_only(questions)
    if len(questions) != expected_questions:
        raise ScaleSensitivityError(
            f"measurement-host level 1 measures exactly the {config.N_DEV} dev questions, "
            f"not {len(questions)}"
        )
    result = evaluate_retriever(
        retriever,
        questions=questions,
        token_counts=token_counts,
        budgets=config.CONTEXT_BUDGETS,
        ks=config.RECALL_AT_K,
        top_k=int(provenance["top_k"]),
        config=dict(provenance),
        split="dev",
    )
    reading = _reading(
        result, n_questions=len(questions), required=required, provenance=provenance
    )
    reading["host"] = dict(host)
    reading["run_digest"] = digest_of_payload(asdict(result))
    reading["basis"] = (
        "D-L1 repeats the historical-vector level-1 reading on the measurement host before "
        "any new model is loaded, so a later level-2 difference can be localized more tightly."
    )
    if not reading["matches"]:
        raise ScaleSensitivityError(
            f"measurement-host level 1 for {model} reproduced {reading['supported']} / "
            f"{reading['n_questions']}, not the required {required}: reproduction_stop before "
            "model load"
        )
    return reading


# --- S6: the three larger scales, and the level-2 gate -----------------------


def scale_filename(model: str) -> str:
    return f"scale-{model}.json"


OUTCOME_FILENAME = "outcome.json"

PERMITTED_READINGS: dict[str, str] = {
    "crossover": (
        "The Dense ordering measured at C19 does not generalize across the tested "
        "retrieval-corpus scales."
    ),
    "convergence": (
        "Corpus size materially affects the Dense-model comparison over the tested range."
    ),
    "both_degrade": (
        "Both encoders degraded substantially over the tested range, and the narrowed "
        "deficit at C500 cannot be distinguished from a floor effect. The scale "
        "sensitivity of the ordering is NOT established."
    ),
    "stable_ranking": (
        "No evidence was found up to 500k paragraphs that Qwen's Phase 8 deficit was an "
        "artefact of the 19k retrieval pool."
    ),
}


def _scale_reading(
    result: RunResult, *, n_questions: int, units: int, run_file: str
) -> dict[str, Any]:
    """One (model, corpus size) reading. No required count: nothing is being gated here."""
    full_support = {
        str(budget): float(result.metrics[f"budget_{budget}"]["full_support"])
        for budget in config.CONTEXT_BUDGETS
    }
    return {
        "system": result.system,
        "units": units,
        "headline_budget": HEADLINE_BUDGET,
        "full_support": full_support,
        "recall_at_k": {
            str(k): float(result.metrics[f"recall_at_{k}"]) for k in config.RECALL_AT_K
        },
        "supported": round(full_support[str(HEADLINE_BUDGET)] * n_questions),
        "n_questions": n_questions,
        "mean_latency_ms": float(result.cost["mean_latency_ms"]),
        "run_file": run_file,
        "run_digest": digest_of_payload(asdict(result)),
    }


def measure_scales(
    directory: Path | str,
    *,
    model: str,
    retriever_for_size: Callable[[int], Retriever],
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    provenance: Mapping[str, Any],
    level_1_supported: int,
    retrieval_cost: Mapping[str, Any],
    host: Mapping[str, Any],
    sizes: Sequence[int] = config.PHASE_8_1_CORPUS_SIZES,
    level_one_on_measurement_host: Mapping[str, Any] | None = None,
    expected_questions: int = config.N_DEV,
    peak_rss_mb: Callable[[], float] | None = None,
) -> Path:
    """One model's readings at every corpus size, plus the level-2 reproduction check.

    Every size is a prefix of one encode, so the four readings are nested by construction.
    Level 2 compares the smallest reading - C19 from the prefix of the new vectors - against
    the level-1 count from the historical ones: identical is a confirmation, one question is
    tolerated and recorded as a caveat on every figure, and two or more is
    `reproduction_stop`. D-L1 is used to localize a tolerated difference when it can; otherwise
    the cause remains residual numerical/host/re-encoding variation.
    """
    check_dev_only(questions)
    if len(questions) != expected_questions:
        raise ScaleSensitivityError(
            f"the scale run measures exactly the {config.N_DEV} dev questions, not {len(questions)}"
        )
    directory = Path(directory)
    target = directory / scale_filename(model)
    if target.exists():
        raise ScaleSensitivityError(
            f"{target.name} already holds a scale reading for {model}; a measured run is "
            "not replaced in place"
        )

    ordered = sorted(sizes)
    if not ordered:
        raise ScaleSensitivityError("the scale run was given no corpus sizes")

    readings: dict[str, Any] = {}

    def measure_one(size: int) -> None:
        run_provenance = dict(provenance)
        unit_set_hashes = run_provenance.get("unit_set_hashes")
        if isinstance(unit_set_hashes, Mapping):
            prefix_hash = unit_set_hashes.get(str(size))
            if not isinstance(prefix_hash, str) or not prefix_hash:
                raise ScaleSensitivityError(
                    f"no unit-set hash was recorded for the declared corpus size {size}"
                )
            run_provenance["unit_set_hash"] = prefix_hash
        result = evaluate_retriever(
            retriever_for_size(size),
            questions=questions,
            token_counts=token_counts,
            budgets=config.CONTEXT_BUDGETS,
            ks=config.RECALL_AT_K,
            top_k=int(provenance["top_k"]),
            config={**run_provenance, "corpus_units": size},
            split="dev",
        )
        run_path = result.save(directory)
        readings[str(size)] = _scale_reading(
            result, n_questions=len(questions), units=size, run_file=run_path.name
        )

    # Level 2 is a gate, not a post-hoc annotation. Measure C19 first and do not spend
    # another retrieval pass if the fresh C500 prefix already fails reproduction.
    measure_one(ordered[0])
    smallest = str(ordered[0])
    supported = readings[smallest]["supported"]
    difference = supported - level_1_supported
    within = abs(difference) <= config.PHASE_8_1_REENCODE_TOLERANCE
    caveat = None
    if within and difference != 0:
        host_supported = (
            int(level_one_on_measurement_host["supported"])
            if level_one_on_measurement_host is not None
            and "supported" in level_one_on_measurement_host
            else None
        )
        if host_supported == supported and host_supported != level_1_supported:
            attribution = (
                "D-L1 moved the unchanged historical-vector reading to the same count on this "
                "host, localizing the difference to host-side evaluation rather than re-encoding."
            )
        elif host_supported == level_1_supported:
            attribution = (
                "D-L1 reproduced the laptop count exactly, so host-only evaluation did not move "
                "the historical vectors; the remaining difference is residual "
                "numerical/host-re-encoding interaction."
            )
        else:
            attribution = (
                "D-L1 does not uniquely localize the source; it is reported as residual "
                "numerical/host/re-encoding variation."
            )
        caveat = (
            f"{model}: C19 from the new C500 prefix supports {supported} against level 1's "
            f"{level_1_supported}, a difference of {difference} question within the declared "
            f"tolerance of 1. {attribution} This is a caveat on every scale figure, not a "
            "licence to widen the gate."
        )

    if within:
        for size in ordered[1:]:
            measure_one(size)

    measured_retrieval_cost = dict(retrieval_cost)
    if within:
        c500 = str(ordered[-1])
        c500_seconds = float(readings[c500]["mean_latency_ms"]) / 1000.0
        measured_retrieval_cost.update(
            {
                "probe_corpus": ordered[-1],
                "mean_seconds_per_query": c500_seconds,
                "query_seconds_ceiling": config.PHASE_8_1_QUERY_SECONDS_CEILING,
                "exceeds_query_seconds_ceiling": (
                    c500_seconds > config.PHASE_8_1_QUERY_SECONDS_CEILING
                ),
            }
        )
        if peak_rss_mb is not None:
            measured_retrieval_cost["peak_rss_mb"] = float(peak_rss_mb())
    else:
        measured_retrieval_cost["probe_status"] = "not_run_due_to_reproduction_stop"

    largest_measured = str(max(int(size) for size in readings))
    body = {
        "model": model,
        "questions": len(questions),
        "sizes": [int(size) for size in readings],
        "scales": readings,
        "level_2": {
            "corpus": ordered[0],
            "supported": supported,
            "level_1_supported": level_1_supported,
            "difference": difference,
            "tolerance": config.PHASE_8_1_REENCODE_TOLERANCE,
            "within_tolerance": within,
            "caveat": caveat,
            "level_one_on_measurement_host": (
                dict(level_one_on_measurement_host)
                if level_one_on_measurement_host is not None
                else None
            ),
            "basis": (
                "level 1 held the vectors constant and varied the code; level 2 holds the "
                "code constant and varies the vectors. The two counts are stored separately "
                "because that separation is the whole diagnostic."
            ),
        },
        "own_drop_c19_to_c500": (
            supported - readings[largest_measured]["supported"] if within else None
        ),
        "retrieval_cost": measured_retrieval_cost,
        "host": dict(host),
        "provenance": dict(provenance),
        "terminal_state": None if within else REPRODUCTION_STOP,
    }
    write_text_atomic(target, json.dumps(body, indent=2, sort_keys=True))
    return target


def load_scale(directory: Path | str, model: str) -> dict[str, Any]:
    """One model's recorded scale readings, or a refusal naming the stage that writes them."""
    target = Path(directory) / scale_filename(model)
    if not target.exists():
        raise ScaleSensitivityError(
            f"{target} does not exist: run 'scale-run --model {model}' first"
        )
    body: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return body


# --- S7: the pre-declared outcome -------------------------------------------


def _supported_by_size(body: Mapping[str, Any]) -> dict[int, int]:
    return {int(size): int(reading["supported"]) for size, reading in body["scales"].items()}


def classify_outcome(
    directory: Path | str,
    *,
    bge: Mapping[str, Any],
    qwen: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
) -> Path:
    """Assign exactly one pre-declared outcome, in the declared order, on whole questions.

    No significance test, no non-inferiority margin, no state machine. The three constants
    this reads - the 20-question convergence threshold, the 244/600 BGE retention floor and
    the 1-question re-encode tolerance - were frozen in `config` before the first
    measurement and are not adjusted after seeing any scale result.

    There is no branch that can produce one of the four outcomes when an upstream artifact
    records a stop: a stop is not silently converted into a weaker success criterion.
    """
    upstream = reconciliation.get("terminal_state")
    if upstream:
        raise ScaleSensitivityError(
            f"the reconciliation recorded {upstream}: no scale result may be interpreted, "
            "and no outcome is assigned"
        )
    for body in (bge, qwen):
        stop = body.get("terminal_state")
        if stop:
            raise ScaleSensitivityError(
                f"the {body.get('model')} scale run recorded {stop}: diagnose it before any "
                "outcome is assigned"
            )

    bge_supported = _supported_by_size(bge)
    qwen_supported = _supported_by_size(qwen)
    expected_sizes = tuple(config.PHASE_8_1_CORPUS_SIZES)
    if (
        tuple(sorted(bge_supported)) != expected_sizes
        or tuple(sorted(qwen_supported)) != expected_sizes
    ):
        raise ScaleSensitivityError(
            "the outcome requires complete measurements at exactly "
            f"{list(expected_sizes)}; got BGE {sorted(bge_supported)} and "
            f"Qwen {sorted(qwen_supported)}"
        )
    if int(bge["questions"]) != config.N_DEV or int(qwen["questions"]) != config.N_DEV:
        raise ScaleSensitivityError(
            f"the outcome requires exactly {config.N_DEV} dev questions for both models; "
            f"got BGE {bge.get('questions')} and Qwen {qwen.get('questions')}"
        )
    bge_question_hash = bge.get("provenance", {}).get("question_set_hash")
    qwen_question_hash = qwen.get("provenance", {}).get("question_set_hash")
    if (
        not isinstance(bge_question_hash, str)
        or not bge_question_hash
        or not isinstance(qwen_question_hash, str)
        or not qwen_question_hash
        or bge_question_hash != qwen_question_hash
    ):
        raise ScaleSensitivityError(
            "the outcome requires BGE and Qwen to use exactly the same frozen dev "
            "question set; their question_set_hash values are missing or differ"
        )
    sizes = list(expected_sizes)
    n_questions = config.N_DEV
    largest = sizes[-1]

    headline = [
        {
            "corpus": size,
            "bge": bge_supported[size],
            "qwen": qwen_supported[size],
            "n_questions": n_questions,
            "deficit_questions": qwen_supported[size] - bge_supported[size],
            "deficit_points": (qwen_supported[size] - bge_supported[size]) / n_questions * 100,
        }
        for size in sizes
    ]

    # C19 is the inherited reading, not one of the scales the ordering is tested at.
    larger = sizes[1:]
    crossover_at = [size for size in larger if qwen_supported[size] >= bge_supported[size]]
    deficit = qwen_supported[largest] - bge_supported[largest]
    floor = config.PHASE_8_1_BGE_RETENTION_FLOOR
    above_floor = bge_supported[largest] >= floor
    narrowed = abs(deficit) <= config.PHASE_8_1_CONVERGENCE_QUESTIONS

    if crossover_at:
        outcome, rule = "crossover", "C"
    elif narrowed and above_floor:
        outcome, rule = "convergence", "B"
    elif narrowed:
        outcome, rule = "both_degrade", "BD"
    else:
        outcome, rule = "stable_ranking", "A"

    if outcome not in config.PHASE_8_1_OUTCOMES:
        raise ScaleSensitivityError(f"{outcome} is not one of {config.PHASE_8_1_OUTCOMES}")

    caveats = {
        model: body["level_2"]["caveat"]
        for model, body in (("bge", bge), ("qwen", qwen))
        if body["level_2"].get("caveat")
    }

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / OUTCOME_FILENAME
    if target.exists():
        raise ScaleSensitivityError(f"{target.name} already records an outcome")

    body_out = {
        "terminal_state": outcome,
        "rule": rule,
        "permitted_reading": PERMITTED_READINGS[outcome],
        "headline": headline,
        "sizes": sizes,
        "deficit_c500_questions": deficit,
        "deficit_c500_points": deficit / n_questions * 100,
        "crossover_at": crossover_at,
        "convergence_threshold_questions": config.PHASE_8_1_CONVERGENCE_QUESTIONS,
        "floor_guard": {
            "bge_c500": bge_supported[largest],
            "floor": floor,
            "above_floor": above_floor,
            "basis": (
                "half of BGE's C19 reading, rounded up. It decides only between convergence "
                "and both_degrade; crossover and stable_ranking are unaffected by it."
            ),
        },
        # Reported for every outcome, so a reader always sees the absolute level any
        # difference sits on rather than the deficit alone.
        "own_drops": {
            "bge": bge["own_drop_c19_to_c500"],
            "qwen": qwen["own_drop_c19_to_c500"],
        },
        "measurement_caveats": caveats,
        "bounds": (
            "This does not establish which encoder performs better at 5M paragraphs, that "
            "one model is generally superior, or that Entity Hop scales. Its claim is "
            "limited to whether the C19 ordering survives the tested range."
        ),
    }
    write_text_atomic(target, json.dumps(body_out, indent=2, sort_keys=True))
    return target
