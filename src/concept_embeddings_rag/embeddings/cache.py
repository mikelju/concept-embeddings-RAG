"""On-disk embedding cache, keyed by the configuration that produced it.

Embedding the corpus is the expensive operation of this project, so nothing is
ever recomputed. Two rules keep the cache trustworthy: artifacts are keyed by
(model, revision, corpus, normalization) so a changed parameter writes a new file
instead of overwriting an old one, and vectors are always validated against their
id list before anything downstream trusts the alignment.

Storage is `.npz` plus a JSON sidecar. Never pickle: loading a pickle executes
code, and a cache file is exactly what an attacker would tamper with.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from concept_embeddings_rag.corpus.pool import IndexingUnit, Question
from concept_embeddings_rag.embeddings.backend import EmbeddingBackend, QueryPromptError


class CacheAlignmentError(Exception):
    """Vectors and unit ids do not line up, or are not the ones expected."""


def unit_set_hash(unit_ids: Sequence[str]) -> str:
    """Identify a corpus by its set of unit ids, independent of ordering."""
    joined = "\n".join(sorted(unit_ids))
    return hashlib.sha1(joined.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def cache_key(model: str, revision: str, corpus_hash: str, normalized: bool) -> str:
    payload = f"{model}|{revision}|{corpus_hash}|{int(normalized)}"
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


class EmbeddingCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.directory / f"embeddings-{key}.npz"

    def sidecar_for(self, key: str) -> Path:
        return self.directory / f"embeddings-{key}.json"

    def save(
        self,
        key: str,
        vectors: np.ndarray,
        unit_ids: Sequence[str],
        metadata: dict,
    ) -> Path:
        if vectors.shape[0] != len(unit_ids):
            raise CacheAlignmentError(f"{vectors.shape[0]} vectors for {len(unit_ids)} unit ids")
        path = self.path_for(key)
        np.savez_compressed(
            path,
            vectors=vectors.astype(np.float32),
            unit_ids=np.array(list(unit_ids), dtype=np.str_),
        )
        self.sidecar_for(key).write_text(
            json.dumps(
                {"key": key, "n_units": len(unit_ids), **metadata}, indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
        return path

    def load(
        self,
        key: str,
        expected_unit_ids: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, list[str]] | None:
        path = self.path_for(key)
        if not path.exists():
            return None

        with np.load(path, allow_pickle=False) as payload:
            vectors = payload["vectors"]
            unit_ids = [str(uid) for uid in payload["unit_ids"]]

        if vectors.shape[0] != len(unit_ids):
            raise CacheAlignmentError(
                f"cache {path.name}: {vectors.shape[0]} vectors for {len(unit_ids)} ids"
            )
        if expected_unit_ids is not None and unit_ids != list(expected_unit_ids):
            raise CacheAlignmentError(
                f"cache {path.name} does not hold the expected units; refusing to use it"
            )
        return vectors, unit_ids


def resolved_revision(backend: EmbeddingBackend) -> str:
    """The commit the backend actually loaded, or the requested revision.

    Recorded next to every cached artifact so a rerun can be checked against
    the pin instead of trusting it.
    """
    resolver = getattr(backend, "resolved_revision", None)
    return resolver() if callable(resolver) else backend.revision


def query_prompt_of(backend: EmbeddingBackend) -> str:
    """The instruction prefix a backend encodes questions with, or `""` for none.

    Phase 8, decision 5. Every backend written before Phase 8 has no such attribute
    and answers `""`, which is what keeps every key and sidecar of Phases 1-7
    byte-identical.
    """
    prompt = getattr(backend, "resolved_query_prompt", "")
    return str(prompt) if prompt else ""


def embed_units(
    units: Sequence[IndexingUnit],
    backend: EmbeddingBackend,
    cache: EmbeddingCache,
) -> tuple[np.ndarray, list[str]]:
    """Return embeddings for `units`, computing them only if they are not cached.

    A backend carrying a query prompt is refused rather than used: documents are
    encoded without an instruction prefix, and making that structural is cheaper than
    trusting every future caller to remember it.
    """
    if query_prompt_of(backend):
        raise QueryPromptError(
            "this backend carries a query prompt and documents are encoded without one; "
            "build a second, unprompted backend for the corpus pass"
        )
    unit_ids = [unit.unit_id for unit in units]
    key = cache_key(
        backend.name,
        backend.revision,
        unit_set_hash(unit_ids),
        normalized=getattr(backend, "normalize", True),
    )

    cached = cache.load(key, expected_unit_ids=unit_ids)
    if cached is not None:
        print(f"[INFO] using cached embeddings for {len(unit_ids)} units")
        return cached

    print(f"[INFO] embedding {len(unit_ids)} units with {backend.name}")
    vectors = backend.encode([unit.indexable_text for unit in units])
    cache.save(
        key,
        vectors,
        unit_ids,
        metadata={
            "model": backend.name,
            "revision": backend.revision,
            "resolved_revision": resolved_revision(backend),
            "dim": int(vectors.shape[1]),
            "normalized": bool(getattr(backend, "normalize", True)),
        },
    )
    return vectors, unit_ids


# --- Question embeddings (Phase 3) ------------------------------------------
# The dev sweep makes dozens of passes over the same 600 questions, and a dense
# encode costs ~30.7 ms here: re-encoding them every pass would cost hours and
# change no number. So the questions are embedded once, exactly like the corpus,
# and every retriever of the sweep is handed a backend that only reads that cache
# (decision D1 of the phase plan).


class QueryCacheMiss(KeyError):
    """A query was asked for that the question cache does not hold.

    Deliberately fatal. Falling back to the model on a miss would make the sweep
    slow instead of wrong, which is the same thing as making the bug invisible.
    """


def question_set_hash(qids: Sequence[str]) -> str:
    """Identify a question set by its ids, independent of ordering."""
    joined = "\n".join(sorted(qids))
    return hashlib.sha1(joined.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def question_cache_key(
    model: str,
    revision: str,
    question_set: str,
    split: str,
    normalized: bool,
    query_prompt: str = "",
) -> str:
    """Key a question artifact by everything that decides what is in it.

    The split is part of the key and not merely of the metadata: dev and test must
    never be able to land in the same file, whatever their ids happen to hash to.

    `query_prompt` joins the payload **only when it is non-empty** (Phase 8, decision
    5). An asymmetric model encodes the same question differently under two prompts, so
    the prompt has to be part of the key; appending it unconditionally would instead
    change every key Phases 1-7 wrote and orphan every question cache on disk, which is
    an invalidation by arithmetic rather than by deletion.
    """
    payload = f"question|{model}|{revision}|{question_set}|{split}|{int(normalized)}"
    if query_prompt:
        payload = f"{payload}|{query_prompt}"
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _single_split(questions: Sequence[Question]) -> str:
    splits = sorted({question.split for question in questions})
    if len(splits) != 1:
        raise ValueError(f"questions span more than one split: {splits}")
    return splits[0]


def _question_artifact_key(questions: Sequence[Question], backend: EmbeddingBackend) -> str:
    return question_cache_key(
        backend.name,
        backend.revision,
        question_set_hash([question.qid for question in questions]),
        _single_split(questions),
        normalized=bool(getattr(backend, "normalize", True)),
        query_prompt=query_prompt_of(backend),
    )


def embed_questions(
    questions: Sequence[Question],
    backend: EmbeddingBackend,
    cache: EmbeddingCache,
) -> tuple[np.ndarray, list[str]]:
    """Return embeddings for `questions`, computing them only if they are not cached."""
    if not questions:
        raise ValueError("no questions to embed")

    split = _single_split(questions)
    qids = [question.qid for question in questions]
    key = _question_artifact_key(questions, backend)

    cached = cache.load(key, expected_unit_ids=qids)
    if cached is not None:
        print(f"[INFO] using cached embeddings for {len(qids)} {split} questions")
        return cached

    print(f"[INFO] embedding {len(qids)} {split} questions with {backend.name}")
    vectors = backend.encode([question.question for question in questions])
    metadata = {
        "model": backend.name,
        "revision": backend.revision,
        "resolved_revision": resolved_revision(backend),
        "dim": int(vectors.shape[1]),
        "normalized": bool(getattr(backend, "normalize", True)),
        "split": split,
    }
    # Written only when there is one, so a Phase 1-7 sidecar keeps its exact shape.
    prompt = query_prompt_of(backend)
    if prompt:
        metadata["query_prompt"] = prompt
    cache.save(key, vectors, qids, metadata=metadata)
    return vectors, qids


class CachedQueryBackend:
    """An `EmbeddingBackend` that answers from a table instead of from a model.

    It is a backend and not a retriever on purpose: `DenseRetriever` and every
    other system under test are handed this and need no change at all, so what the
    sweep measures is still the retriever the final report describes.
    """

    def __init__(
        self,
        texts: Sequence[str],
        vectors: np.ndarray,
        name: str,
        revision: str,
    ) -> None:
        if vectors.shape[0] != len(texts):
            raise CacheAlignmentError(f"{vectors.shape[0]} vectors for {len(texts)} queries")
        if vectors.ndim != 2:
            raise CacheAlignmentError(f"expected a 2-D vector block, got shape {vectors.shape}")

        table: dict[str, np.ndarray] = {}
        for text, vector in zip(texts, vectors, strict=True):
            previous = table.get(text)
            if previous is not None and not np.array_equal(previous, vector):
                raise CacheAlignmentError(
                    "two different vectors are cached for the same query text; refusing to pick one"
                )
            table[text] = vector

        self._table = table
        self.name = name
        self.revision = revision
        self.dim = int(vectors.shape[1])

    def __len__(self) -> int:
        return len(self._table)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = self._table.get(text)
            if vector is None:
                raise QueryCacheMiss(
                    f"query not in the cache and this backend never calls the model: {text!r}"
                )
            rows.append(vector)
        if not rows:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack(rows).astype(np.float32)


def build_query_backend(
    questions: Sequence[Question],
    backend: EmbeddingBackend,
    cache: EmbeddingCache,
) -> CachedQueryBackend:
    """Embed `questions` once and return a backend that serves them by lookup."""
    vectors, qids = embed_questions(questions, backend, cache)
    key = _question_artifact_key(questions, backend)
    _verify_question_sidecar(cache, key, backend, vectors, qids, _single_split(questions))

    by_qid = {question.qid: question.question for question in questions}
    return CachedQueryBackend(
        texts=[by_qid[qid] for qid in qids],
        vectors=vectors,
        name=backend.name,
        revision=backend.revision,
    )


def _verify_question_sidecar(
    cache: EmbeddingCache,
    key: str,
    backend: EmbeddingBackend,
    vectors: np.ndarray,
    qids: Sequence[str],
    split: str,
) -> None:
    """Compare the artifact against what it claims, rather than merely recording it.

    A sidecar is only worth writing if something reads it back and disagrees when
    it must - that was the whole content of the Phase 1 audit.
    """
    path = cache.sidecar_for(key)
    if not path.exists():
        raise CacheAlignmentError(f"question cache {key} has no sidecar; refusing to use it")

    recorded = json.loads(path.read_text(encoding="utf-8"))
    expected: dict[str, object] = {
        "model": backend.name,
        "revision": backend.revision,
        "dim": int(vectors.shape[1]),
        "n_units": len(qids),
        "split": split,
    }
    # Checked only when a prompt is in force: an empty one leaves this verification
    # byte-identical to what Phases 1-7 ran, and a cache whose sidecar disagrees with
    # the prompt in hand is refused rather than used.
    prompt = query_prompt_of(backend)
    if prompt:
        expected["query_prompt"] = prompt
    for field, value in expected.items():
        if recorded.get(field) != value:
            raise CacheAlignmentError(
                f"question cache {key}: sidecar says {field}={recorded.get(field)!r}, "
                f"the artifact says {value!r}"
            )
