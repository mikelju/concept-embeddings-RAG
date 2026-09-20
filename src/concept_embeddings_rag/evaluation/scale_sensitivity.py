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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.embeddings.cache import CachedQueryBackend
from concept_embeddings_rag.evaluation.harness import RunResult, evaluate_retriever
from concept_embeddings_rag.evaluation.selection import check_dev_only
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
