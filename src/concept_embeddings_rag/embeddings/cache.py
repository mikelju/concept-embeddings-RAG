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

from concept_embeddings_rag.corpus.pool import IndexingUnit
from concept_embeddings_rag.embeddings.backend import EmbeddingBackend


class CacheAlignmentError(Exception):
    """Vectors and unit ids do not line up, or are not the ones expected."""


def unit_set_hash(unit_ids: Sequence[str]) -> str:
    """Identify a corpus by its set of unit ids, independent of ordering."""
    joined = "\n".join(sorted(unit_ids))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def cache_key(model: str, revision: str, corpus_hash: str, normalized: bool) -> str:
    payload = f"{model}|{revision}|{corpus_hash}|{int(normalized)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


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
            raise CacheAlignmentError(
                f"{vectors.shape[0]} vectors for {len(unit_ids)} unit ids"
            )
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


def embed_units(
    units: Sequence[IndexingUnit],
    backend: EmbeddingBackend,
    cache: EmbeddingCache,
) -> tuple[np.ndarray, list[str]]:
    """Return embeddings for `units`, computing them only if they are not cached."""
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
            "dim": int(vectors.shape[1]),
            "normalized": bool(getattr(backend, "normalize", True)),
        },
    )
    return vectors, unit_ids
