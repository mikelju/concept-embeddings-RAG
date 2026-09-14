"""Shared builders for the Phase 4 diffusion tests.

Not a test module: it holds the toy corpus that T4 to T8 all walk over, so that
four files do not each invent a slightly different one and then disagree about
what the diffusion is supposed to do on it.

The corpus is five units over three concepts, shaped so that a second hop is
visible with the naked eye:

    u0: [2, 0, 0]      u0 and u1 share concept 0
    u1: [1, 1, 0]      u1, u2 and u3 share concept 1
    u2: [0, 3, 0]      u3 and u4 share concept 2
    u3: [0, 1, 2]
    u4: [0, 0, 1]

A seed that returns u0 alone therefore reaches u1 in one round and u2 and u3 in
two - none of which the seed ever ranked. No unit has an empty row and no concept
an empty column, which is what the operator requires and what Phase 2 measured to
be true of the real space at every K.
"""

import numpy as np
from scipy import sparse

from concept_embeddings_rag.concepts.coding import ConceptMatrix
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary
from concept_embeddings_rag.retrieval.base import Hit

ROWS: list[list[float]] = [
    [2.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 3.0, 0.0],
    [0.0, 1.0, 2.0],
    [0.0, 0.0, 1.0],
]
UNIT_IDS: list[str] = ["u0", "u1", "u2", "u3", "u4"]

INHERITED = {
    "selection_digest": "0" * 16,
    "k": 3,
    "view": "raw",
    "damping": "idf",
    "query_operator": "projection_full",
}


def a_dictionary(k: int = 3, dim: int = 3) -> ConceptDictionary:
    """The identity, so a projection is readable straight off the query string."""
    return ConceptDictionary(
        atoms=np.eye(k, dim, dtype=np.float32),
        k=k,
        seed=42,
        sparsity_param=0.05,
        max_iter=3,
        unit_set_hash="101f564fdcca620c",
        model="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c",
    )


def a_matrix(
    key: str,
    rows: list[list[float]] | None = None,
    unit_ids: list[str] | None = None,
    view: str = "raw",
) -> ConceptMatrix:
    X = sparse.csr_matrix(np.array(rows if rows is not None else ROWS, dtype=np.float32))
    return ConceptMatrix(
        X=X,
        unit_ids=list(unit_ids if unit_ids is not None else UNIT_IDS),
        dictionary_key=key,
        view=view,
        coding_alpha=0.07,
        mean_active_per_unit=float(X.nnz) / X.shape[0],
        reconstruction_error=0.0,
    )


class SeedStub:
    """A `Retriever` that returns a fixed list, so a test controls where the walk starts.

    It records the depth it was asked for, because decision D5 ties that depth to
    the one the harness reads and a test has to be able to see what was requested.
    """

    def __init__(self, hits: list[Hit], name: str = "seed") -> None:
        self.name = name
        self._hits = list(hits)
        self.asked_for: list[int] = []
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        self.asked_for.append(top_k)
        self.queries.append(query)
        return list(self._hits[:top_k])
