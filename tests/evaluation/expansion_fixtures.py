"""Shared builders for the Phase 4 sweep, choice and freeze tests.

Not a test module. It holds the toy space the three files walk over, so that they
cannot disagree about what the grid is supposed to measure.

The corpus is five units over three concepts, with no empty row and no empty column -
which is what the `symmetric` and `stochastic` arms require and what Phase 2 measured
to be true of the real space at every K:

    u0: [2, 0, 0]      u0 and u1 share concept 0
    u1: [1, 1, 0]      u1, u2 and u3 share concept 1
    u2: [0, 3, 0]      u3 and u4 share concept 2
    u3: [0, 1, 2]
    u4: [0, 0, 1]
"""

from collections.abc import Sequence

import numpy as np
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.concepts.coding import ConceptMatrix
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.retrieval.base import Hit

ROWS = [
    [2.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 3.0, 0.0],
    [0.0, 1.0, 2.0],
    [0.0, 0.0, 1.0],
]
UNIT_IDS = ["u0", "u1", "u2", "u3", "u4"]
TOKENS = dict.fromkeys(UNIT_IDS, 100)
POOL = "101f564fdcca620c"
PHASE_3_DIGEST = "91daa10ef0a75b6e" + "0" * 48

INHERITED = {
    "k": 3,
    "dictionary_key": "",  # filled in by `inherited_for`, which knows the key
    "view": "raw",
    "damping": "idf",
    "query_operator": "projection_full",
}


def a_dictionary(k: int = 3, dim: int = 3) -> ConceptDictionary:
    return ConceptDictionary(
        atoms=np.eye(k, dim, dtype=np.float32),
        k=k,
        seed=42,
        sparsity_param=0.05,
        max_iter=3,
        unit_set_hash=POOL,
        model="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c",
    )


def a_matrix(key: str, view: str = "raw") -> ConceptMatrix:
    X = sparse.csr_matrix(np.array(ROWS, dtype=np.float32))
    return ConceptMatrix(
        X=X,
        unit_ids=list(UNIT_IDS),
        dictionary_key=key,
        view=view,
        coding_alpha=0.07,
        mean_active_per_unit=float(X.nnz) / X.shape[0],
        reconstruction_error=0.0,
    )


def inherited_for(dictionary: ConceptDictionary) -> dict:
    return {**INHERITED, "dictionary_key": dictionary.key}


class SeedStub:
    """A retriever with a fixed answer, so a test controls where every walk starts."""

    def __init__(self, name: str, hits: Sequence[Hit]) -> None:
        self.name = name
        self._hits = list(hits)
        self.asked_for: list[int] = []

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        self.asked_for.append(top_k)
        return list(self._hits[:top_k])


def seeds() -> dict[str, SeedStub]:
    """One seed per arm of HU-3, deliberately different so a mix-up would show."""
    return {
        "dense": SeedStub("dense", [("u0", 0.9), ("u1", 0.4)]),
        "conceptual": SeedStub("conceptual", [("u2", 0.7), ("u4", 0.2)]),
    }


def a_question(qid: str, split: str = "dev", gold: tuple[str, ...] = ("u1", "u3")) -> Question:
    return Question(
        qid=qid,
        question=f"question {qid}",
        answer="-",
        gold_unit_ids=gold,
        supporting_facts=(("T", 0),),
        split=split,
    )


def questions_on(split: str = "dev", n: int = 4) -> list[Question]:
    return [a_question(f"q{index}", split) for index in range(n)]


def a_config(**overrides) -> dict:
    base = {
        "model": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c",
        "unit_set_hash": POOL,
        "seed": config.DEFAULT_SEED,
        "tokenizer": "fake-tokenizer",
        "code_version": "0.1.0",
        "top_k": 5,
    }
    base.update(overrides)
    return base
