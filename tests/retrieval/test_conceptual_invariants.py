"""T6: what the concept space guarantees, stated rather than discovered.

A unit whose row of `X` is all zeros shares no concept with anything. Its score is
`0 @ c = 0` for every query there is, so it can never be retrieved on merit: it is
ranked below every unit that activates a concept the question activates, whatever
the query and whatever the `top_k`.

What it is not is absent from the list. Like `dense.py` and `bm25.py`, this
retriever returns the best `top_k` it has, and once the units with a positive score
run out the remainder is filler at zero - which at a fixed context budget still
costs tokens. Filtering zeros here would make System B fill less context than the
baselines it is compared against, so the honest invariant is the one asserted below:
unreachable on merit, never ahead of a unit that shares something with the question.

Phase 2 measured zero orphan units at every K, so this is a property being written
down, not a limitation being introduced.
"""

import numpy as np
import pytest
from scipy import sparse

from concept_embeddings_rag.concepts.coding import ConceptMatrix
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary
from concept_embeddings_rag.retrieval.conceptual import ConceptualRetriever

DIM = 12
K = 24
ORPHAN = "orphan"


class TableBackend:
    """Serves a stored vector per query string; nothing is derived from the text."""

    name = "fake"
    revision = "v0"
    dim = DIM
    normalize = True

    def __init__(self, table: dict[str, np.ndarray]) -> None:
        self._table = table

    def encode(self, texts):
        return np.stack([self._table[text] for text in texts]).astype(np.float32)


def a_dictionary() -> ConceptDictionary:
    rng = np.random.default_rng(0)
    atoms = rng.normal(size=(K, DIM)).astype(np.float32)
    atoms = atoms / np.linalg.norm(atoms, axis=1, keepdims=True)
    return ConceptDictionary(
        atoms=atoms,
        k=K,
        seed=42,
        sparsity_param=0.05,
        max_iter=3,
        unit_set_hash="101f564fdcca620c",
        model="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c",
    )


def a_pool_with_an_orphan() -> tuple[ConceptMatrix, ConceptDictionary]:
    """Six units that activate concepts, and one that activates none."""
    dictionary = a_dictionary()
    rng = np.random.default_rng(1)

    rows = np.abs(rng.normal(size=(6, K))).astype(np.float32)
    rows[rows < 0.8] = 0.0  # sparse, the way a coded corpus is
    rows = np.vstack([rows, np.zeros((1, K), dtype=np.float32)])

    unit_ids = [f"u{i}" for i in range(6)] + [ORPHAN]
    X = sparse.csr_matrix(rows)
    return (
        ConceptMatrix(
            X=X,
            unit_ids=unit_ids,
            dictionary_key=dictionary.key,
            view="raw",
            coding_alpha=0.07,
            mean_active_per_unit=float(X.nnz) / X.shape[0],
            reconstruction_error=0.0,
        ),
        dictionary,
    )


def some_queries(n: int = 20) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(n, DIM)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return {f"q{i}": vector for i, vector in enumerate(vectors)}


QUERIES = some_queries()


def a_retriever(arm: str = "projection_full") -> ConceptualRetriever:
    matrix, dictionary = a_pool_with_an_orphan()
    return ConceptualRetriever(matrix, dictionary, TableBackend(QUERIES), arm=arm)


@pytest.mark.parametrize("arm", ["projection_full", "projection_top16", "sparse_coding"])
@pytest.mark.parametrize("query", sorted(QUERIES))
def test_a_unit_that_activates_no_concept_scores_exactly_zero(arm: str, query: str):
    hits = a_retriever(arm).retrieve(query, top_k=7)

    assert dict(hits)[ORPHAN] == 0.0


@pytest.mark.parametrize("query", sorted(QUERIES))
def test_the_orphan_never_outranks_a_unit_that_shares_a_concept(query: str):
    hits = a_retriever().retrieve(query, top_k=7)
    ranked = [unit_id for unit_id, _score in hits]
    positive = [unit_id for unit_id, score in hits if score > 0.0]

    assert ranked.index(ORPHAN) >= len(positive)


@pytest.mark.parametrize("query", sorted(QUERIES))
def test_the_orphan_is_out_of_reach_of_any_top_k_the_positives_can_fill(query: str):
    retriever = a_retriever()
    positive = [unit_id for unit_id, score in retriever.retrieve(query, top_k=7) if score > 0.0]
    assert positive, "the fixture must produce positives, or the guard proves nothing"

    for top_k in range(1, len(positive) + 1):
        assert ORPHAN not in dict(retriever.retrieve(query, top_k=top_k))


@pytest.mark.parametrize("query", sorted(QUERIES))
def test_beyond_the_positives_the_list_is_filler_and_says_so_with_a_zero(query: str):
    """It is returned, and at a fixed budget it costs tokens. The score does not lie."""
    hits = a_retriever().retrieve(query, top_k=7)
    positive = [unit_id for unit_id, score in hits if score > 0.0]

    for _unit_id, score in hits[len(positive) :]:
        assert score == 0.0


def test_the_orphan_cannot_be_reached_by_damping_either():
    """Rarity multiplies the query, so a row of zeros stays a row of zeros."""
    matrix, dictionary = a_pool_with_an_orphan()
    weights = np.full(K, 1000.0, dtype=np.float32)
    retriever = ConceptualRetriever(
        matrix,
        dictionary,
        TableBackend(QUERIES),
        arm="projection_full",
        damping="idf",
        concept_weights=weights,
    )

    for query in sorted(QUERIES):
        assert dict(retriever.retrieve(query, top_k=7))[ORPHAN] == 0.0


def test_the_whole_pool_is_searched_rather_than_a_per_question_candidate_list():
    """If a candidate list existed, "unreachable" would be a property of the list."""
    retriever = a_retriever()

    assert len(retriever.retrieve("q0", top_k=100)) == 7
