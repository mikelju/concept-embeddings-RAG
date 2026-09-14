"""T5: the two arms of HU-3 differ in the seed and in nothing else.

Phase 3 needed a control because "the hybrid beats dense" could have been a property
of hybridizing rather than of the concept space. This phase needs the same guarantee
one level up: if the dense-seeded and conceptual-seeded walks were allowed to differ
in their operator, their restart or their stopping rule, then "the conceptual seed
behaves differently" would say nothing about the seed.

Giving one arm a better operator is the single easiest way to manufacture this
phase's headline, so the parity is a test rather than an intention.
"""

import numpy as np
from diffusion_fixtures import INHERITED, SeedStub, a_dictionary, a_matrix

from concept_embeddings_rag import config
from concept_embeddings_rag.retrieval.diffusion import NONE, DiffusionRetriever

# Everything but the seed, so that the two constructions below cannot drift apart
# in the test any more than they may in the phase.
SHARED = {
    "seed_arm": None,
    "restart": 0.4,
    "normalization": NONE,
    "stop_threshold": config.STOP_THRESHOLD,
    "max_iterations": config.MAX_ITERATIONS,
    "seed_top_k": config.SEED_TOP_K,
}


def an_arm(seed_arm: str, seed) -> DiffusionRetriever:
    dictionary = a_dictionary()
    parameters = dict(SHARED)
    parameters["seed_arm"] = seed_arm
    return DiffusionRetriever(
        matrix=a_matrix(dictionary.key),
        dictionary=dictionary,
        seed_retriever=seed,
        inherits=INHERITED,
        concept_weights=np.array([1.0, 2.0, 3.0]),
        **parameters,
    )


def test_the_two_arms_agree_on_every_recorded_field_but_the_seed():
    dense_arm = an_arm("dense", SeedStub([("u0", 0.9)], name="dense"))
    conceptual_arm = an_arm("conceptual", SeedStub([("u2", 0.9)], name="conceptual"))

    described_dense = dense_arm.describe()
    described_conceptual = conceptual_arm.describe()

    differing = {
        field
        for field in described_dense
        if described_dense[field] != described_conceptual.get(field)
    }
    assert differing == {"name", "seed_arm", "seed_system"}


def test_the_two_arms_share_the_operator_parameters_exactly():
    dense_arm = an_arm("dense", SeedStub([("u0", 0.9)]))
    conceptual_arm = an_arm("conceptual", SeedStub([("u2", 0.9)]))

    assert dense_arm.restart == conceptual_arm.restart
    assert dense_arm.normalization == conceptual_arm.normalization
    assert dense_arm.stop_threshold == conceptual_arm.stop_threshold
    assert dense_arm.max_iterations == conceptual_arm.max_iterations
    assert dense_arm.seed_top_k == conceptual_arm.seed_top_k
    assert np.array_equal(dense_arm.concept_weights, conceptual_arm.concept_weights)


def test_the_same_seed_through_both_arms_produces_the_same_ranking():
    """The strongest form of it: with the seed held fixed, the arm label changes nothing."""
    hits = [("u0", 0.7), ("u3", 0.3)]

    as_dense = an_arm("dense", SeedStub(hits)).retrieve("anything", top_k=5)
    as_conceptual = an_arm("conceptual", SeedStub(hits)).retrieve("anything", top_k=5)

    assert as_dense == as_conceptual


def test_both_arms_are_named_in_the_declared_set():
    assert {an_arm(arm, SeedStub([("u0", 0.9)])).seed_arm for arm in config.SEED_ARMS} == set(
        config.SEED_ARMS
    )
