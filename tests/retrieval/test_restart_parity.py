"""T5: at `restart = 1.0` the walk is its seed, exactly.

This is the property the whole phase's attribution rests on (HU-2). If the pipeline
that runs the diffusion changes anything about the ranking even when no expansion
is asked for - a different depth, a different tie-break, a dropped hit - then every
difference measured at a lower restart is a difference of two things at once, and no
number the phase produces can be attributed to the expansion.

It is asserted against **both real seeds**, not against a stub, because the stub is
exactly the object that cannot disagree with itself.
"""

import numpy as np
import pytest
from diffusion_fixtures import INHERITED, ROWS, UNIT_IDS, SeedStub, a_dictionary, a_matrix

from concept_embeddings_rag.retrieval.conceptual import ConceptualRetriever
from concept_embeddings_rag.retrieval.dense import DenseRetriever
from concept_embeddings_rag.retrieval.diffusion import NONE, STOCHASTIC, DiffusionRetriever


class DirectionBackend:
    """Encodes a query as the literal vector written in the query string."""

    name = "fake"
    revision = "v0"
    dim = 3
    normalize = True

    def encode(self, texts):
        return np.array([[float(x) for x in t.split(",")] for t in texts], dtype=np.float32)


def a_dense_seed() -> DenseRetriever:
    # Unit vectors along each concept direction, so a query picks a unit out directly.
    vectors = np.array(ROWS, dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return DenseRetriever(vectors=vectors, unit_ids=UNIT_IDS, backend=DirectionBackend())


def a_conceptual_seed() -> ConceptualRetriever:
    dictionary = a_dictionary()
    return ConceptualRetriever(
        a_matrix(dictionary.key),
        dictionary,
        DirectionBackend(),
        arm="projection_full",
    )


def walking(seed, *, seed_arm: str, restart: float, normalization: str = NONE):
    dictionary = a_dictionary()
    return DiffusionRetriever(
        matrix=a_matrix(dictionary.key),
        dictionary=dictionary,
        seed_retriever=seed,
        seed_arm=seed_arm,
        restart=restart,
        normalization=normalization,
        inherits=INHERITED,
    )


@pytest.mark.parametrize("normalization", [NONE, STOCHASTIC])
def test_a_full_restart_reproduces_the_dense_seed_ranking(normalization):
    seed = a_dense_seed()
    query = "1,0.2,0"

    expected = [unit_id for unit_id, _score in seed.retrieve(query, top_k=5)]
    walked = walking(seed, seed_arm="dense", restart=1.0, normalization=normalization)

    assert [unit_id for unit_id, _score in walked.retrieve(query, top_k=5)] == expected


def test_a_full_restart_reproduces_the_conceptual_seed_ranking():
    seed = a_conceptual_seed()
    query = "1,0.2,0"

    expected = [unit_id for unit_id, _score in seed.retrieve(query, top_k=5)]
    walked = walking(seed, seed_arm="conceptual", restart=1.0)

    assert [unit_id for unit_id, _score in walked.retrieve(query, top_k=5)] == expected


def test_a_full_restart_keeps_a_seed_hit_whose_score_is_zero():
    """Decision D4's candidate set is what makes the parity exact rather than near.

    A zero-scoring hit carries no mass, so without the union with the seed's own list
    it would be displaced by whichever zero-mass unit of the pool sorts first.
    """
    seed = SeedStub([("u3", 0.9), ("u1", 0.4), ("u4", 0.0)])

    hits = walking(seed, seed_arm="dense", restart=1.0).retrieve("anything", top_k=3)

    assert [unit_id for unit_id, _score in hits] == ["u3", "u1", "u4"]
    assert hits[-1][1] == 0.0


def test_the_mass_of_a_full_restart_is_the_seed_scores_renormalized():
    """Not merely the same order: the same numbers, up to the one declared rescaling."""
    seed = SeedStub([("u0", 0.6), ("u2", 0.3), ("u3", 0.1)])

    hits = walking(seed, seed_arm="dense", restart=1.0).retrieve("anything", top_k=3)

    assert [score for _unit_id, score in hits] == pytest.approx([0.6, 0.3, 0.1])


def test_below_a_full_restart_the_ranking_does_move():
    """Otherwise the parity test above would be passing for the wrong reason."""
    seed = SeedStub([("u0", 0.9)])

    still = walking(seed, seed_arm="dense", restart=1.0).retrieve("anything", top_k=5)
    moved = walking(seed, seed_arm="dense", restart=0.4).retrieve("anything", top_k=5)

    assert still != moved
