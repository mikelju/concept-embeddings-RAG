"""Dense retrieval by cosine similarity over the whole pool.

Vectors are L2-normalized upstream, so a dot product is the cosine. Ties are
broken by unit id rather than left to array order, because otherwise a ranking
could change between runs for reasons that have nothing to do with relevance.
"""

from collections.abc import Sequence

import numpy as np

from concept_embeddings_rag.embeddings.backend import EmbeddingBackend
from concept_embeddings_rag.retrieval.base import Hit


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        vectors: np.ndarray,
        unit_ids: Sequence[str],
        backend: EmbeddingBackend,
    ) -> None:
        if vectors.shape[0] != len(unit_ids):
            raise ValueError(f"{vectors.shape[0]} vectors for {len(unit_ids)} unit ids")
        self.vectors = vectors
        self.unit_ids = list(unit_ids)
        self.backend = backend

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        query_vector = self.backend.encode([query])[0]
        scores = self.vectors @ query_vector

        k = min(top_k, len(self.unit_ids))
        if k < len(self.unit_ids):
            candidates = np.argpartition(-scores, k - 1)[:k]
        else:
            candidates = np.arange(len(self.unit_ids))

        ordered = sorted(candidates, key=lambda i: (-float(scores[i]), self.unit_ids[i]))
        return [(self.unit_ids[i], float(scores[i])) for i in ordered]
