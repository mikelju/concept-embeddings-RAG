"""Embedding backends behind a single interface.

The experiment compares retrieval systems, so swapping the embedding model must
never require touching retrieval or evaluation code. The model is loaded lazily:
constructing a backend downloads nothing, which is what keeps the test suite fast
and offline.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from concept_embeddings_rag import config


@runtime_checkable
class EmbeddingBackend(Protocol):
    name: str
    revision: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape [len(texts), dim]."""
        ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length, leaving zero rows untouched.

    A zero row would divide by zero and poison the whole matrix with NaNs, which
    then propagates silently into every similarity score.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return (vectors / safe).astype(np.float32)


class SentenceTransformerBackend:
    """sentence-transformers backend, loaded on first use."""

    def __init__(
        self,
        name: str = config.EMBEDDING_MODEL,
        revision: str = config.EMBEDDING_REVISION,
        dim: int = config.EMBEDDING_DIM,
        normalize: bool = config.NORMALIZE_EMBEDDINGS,
        batch_size: int = 64,
    ) -> None:
        self.name = name
        self.revision = revision
        self.dim = dim
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = None

    def resolved_revision(self) -> str:
        """The commit the Hub actually served, or the requested revision if offline.

        A pin is only worth as much as the ability to check it was honoured, and
        `revision` alone records an intention rather than an outcome.
        """
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(self.name, revision=self.revision)
        except Exception:  # offline, or a local path: the pin is all we can report
            return self.revision
        from pathlib import Path as _Path

        return _Path(path).name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"[INFO] loading embedding model {self.name} (revision {self.revision})")
            self._model = SentenceTransformer(self.name, revision=self.revision)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        ).astype(np.float32)
        return l2_normalize(vectors) if self.normalize else vectors
