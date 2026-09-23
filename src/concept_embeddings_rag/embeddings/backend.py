"""Embedding backends behind a single interface.

The experiment compares retrieval systems, so swapping the embedding model must
never require touching retrieval or evaluation code. The model is loaded lazily:
constructing a backend downloads nothing, which is what keeps the test suite fast
and offline.

Phase 8 adds one optional attribute, `query_prompt`, because the Dense model it
measures is asymmetric: queries carry an instruction prefix and documents do not.
It is a declared configuration item rather than a free parameter, so the backend
reads the prompt back off the loaded model and refuses any model whose own prompt
differs - before a single vector exists. A silently different prompt would change
every query vector while leaving the artifact's provenance looking correct.
"""

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from concept_embeddings_rag import config


class BackendError(ValueError):
    """The model in hand is not the one the configuration declares."""


class QueryPromptError(BackendError):
    """The query prompt in force is not the declared one, or was applied to documents."""


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


def check_query_prompt(prompts: Mapping[str, str] | None, expected: str) -> str:
    """Return the model's own query prompt, refusing anything but an exact match.

    Exact string equality, and a missing key is a mismatch like any other: the
    alternative is encoding 600 or 1,400 queries under a prompt nobody declared.
    An empty `expected` declares that no prompt is in force, and then nothing is read.
    """
    if not expected:
        return ""
    resolved = (prompts or {}).get("query")
    if resolved != expected:
        raise QueryPromptError(
            "the loaded model's query prompt is not the declared one; refusing to embed "
            f"anything (declared {expected!r}, model carries {resolved!r})"
        )
    return str(resolved)


def weights_sha256(directory: Path | str, filename: str = config.PHASE_8_WEIGHTS_FILE) -> str:
    """The sha256 of the weight file in the snapshot that was actually served.

    A revision names a commit; only the digest binds an artifact to the weights that
    produced its numbers. Read in blocks because the file is gigabytes.
    """
    path = Path(directory) / filename
    if not path.is_file():
        raise BackendError(
            f"{path} does not exist, so no weight digest can be recorded for this run"
        )
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def snapshot_directory(name: str, revision: str) -> Path:
    """The local snapshot the Hub serves for `name` at `revision`.

    Its directory name is the commit actually served, which is what makes the pin
    checkable instead of merely stated.
    """
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name, revision=revision))


class SentenceTransformerBackend:
    """sentence-transformers backend, loaded on first use."""

    def __init__(
        self,
        name: str = config.EMBEDDING_MODEL,
        revision: str = config.EMBEDDING_REVISION,
        dim: int = config.EMBEDDING_DIM,
        normalize: bool = config.NORMALIZE_EMBEDDINGS,
        batch_size: int = 64,
        query_prompt: str = "",
    ) -> None:
        self.name = name
        self.revision = revision
        self.dim = dim
        self.normalize = normalize
        self.batch_size = batch_size
        # Empty for every model Phases 1-7 used, and for the Phase 8 corpus pass.
        self.query_prompt = query_prompt
        self._resolved_query_prompt: str | None = None
        self._model = None

    @property
    def resolved_query_prompt(self) -> str:
        """The prompt the vectors were produced under: read from the model once loaded.

        Before the load it is the declared value, and the load refuses any model whose
        own prompt differs, so the two can never disagree in a run that produced a
        vector. Reading it therefore never forces a model load, which is what lets a
        cache-only path build the same key without touching the weights.
        """
        if self._resolved_query_prompt is not None:
            return self._resolved_query_prompt
        return self.query_prompt

    def max_seq_length(self) -> int | None:
        """The window the loaded model actually resolves, never the model-card figure.

        Phase 8, decision 6: the pinned revision's own files disagree with each other,
        so the run records what it observed rather than what was published.
        """
        value = getattr(self._load(), "max_seq_length", None)
        return None if value is None else int(value)

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
            # No `trust_remote_code`: the models this project loads do not need it, and
            # enabling it would execute code fetched from the Hub at load time.
            model = SentenceTransformer(self.name, revision=self.revision)
            self._resolved_query_prompt = check_query_prompt(
                getattr(model, "prompts", None), self.query_prompt
            )
            self._model = model
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        arguments: dict[str, Any] = {
            "batch_size": self.batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": len(texts) > 1000,
        }
        if self.query_prompt:
            # The model's own prompt, already proved equal to the declared one above.
            arguments["prompt_name"] = "query"
        vectors = model.encode(list(texts), **arguments).astype(np.float32)
        return l2_normalize(vectors) if self.normalize else vectors
