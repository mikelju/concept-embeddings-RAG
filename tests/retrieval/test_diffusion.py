"""T4: the retriever that walks the corpus instead of ranking it once.

`DiffusionRetriever` is the phase's object. It implements the same `Retriever` the
harness has measured since Phase 1 - unchanged, so the harness cannot tell it apart
from the four rivals - and everything it records about itself has to agree with what
it actually did, because a mislabelled result is worse than a missing one.
"""

import numpy as np
import pytest
from diffusion_fixtures import INHERITED, UNIT_IDS, SeedStub, a_dictionary, a_matrix

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.harness import evaluate_retriever
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.diffusion import (
    NONE,
    STOCHASTIC,
    DiffusionError,
    DiffusionRetriever,
)


def a_retriever(
    seed: SeedStub | None = None,
    *,
    seed_arm: str = "dense",
    restart: float = 0.5,
    normalization: str = NONE,
    concept_weights: np.ndarray | None = None,
    max_iterations: int = config.MAX_ITERATIONS,
    matrix=None,
) -> DiffusionRetriever:
    dictionary = a_dictionary()
    return DiffusionRetriever(
        matrix=matrix if matrix is not None else a_matrix(dictionary.key),
        dictionary=dictionary,
        seed_retriever=seed if seed is not None else SeedStub([("u0", 0.9)]),
        seed_arm=seed_arm,
        restart=restart,
        normalization=normalization,
        concept_weights=concept_weights,
        max_iterations=max_iterations,
        inherits=INHERITED,
    )


def test_it_implements_the_unchanged_retriever_protocol():
    assert isinstance(a_retriever(), Retriever)


def test_the_harness_measures_it_without_modification():
    questions = [
        Question(
            qid="q1",
            question="anything",
            answer="x",
            split="dev",
            gold_unit_ids=("u2",),
            supporting_facts=(("T", 0),),
        )
    ]
    token_counts = dict.fromkeys(UNIT_IDS, 100)

    result = evaluate_retriever(
        a_retriever(),
        questions=questions,
        token_counts=token_counts,
        budgets=(512,),
        ks=(2,),
        top_k=5,
        config={
            "model": "m",
            "revision": "r",
            "unit_set_hash": "h",
            "seed": 42,
            "tokenizer": "t",
            "code_version": "0",
            "top_k": 5,
        },
        split="dev",
    )

    assert result.system == "expansion"
    assert result.metrics["budget_512"]["gold_recall"] >= 0.0


def test_the_name_is_derived_from_the_seed_arm_and_cannot_disagree_with_it():
    """A result filename that claims one arm while the other ran is unfixable later."""
    assert a_retriever(seed_arm="dense").name == "expansion"
    assert a_retriever(seed_arm="conceptual").name == "expansion-conceptual"


def test_an_unknown_seed_arm_is_refused():
    with pytest.raises(DiffusionError, match="seed arm"):
        a_retriever(seed_arm="bm25")


def test_the_walk_reaches_a_unit_the_seed_never_returned():
    """The second hop, which is the entire reason this phase exists."""
    seed = SeedStub([("u0", 0.9)])

    hits = a_retriever(seed, restart=0.4).retrieve("anything", top_k=5)

    reached = [unit_id for unit_id, score in hits if score > 0.0]
    assert "u1" in reached
    assert set(reached) - {"u0"}


def test_the_seed_is_asked_for_the_declared_depth_not_for_top_k():
    """Decision D5: the seed's depth is a constant of the phase, not the caller's top_k."""
    seed = SeedStub([("u0", 0.9), ("u1", 0.5)])

    a_retriever(seed).retrieve("anything", top_k=2)

    assert seed.asked_for == [config.SEED_TOP_K]


def test_it_returns_at_most_top_k_hits_best_first():
    hits = a_retriever(restart=0.3).retrieve("anything", top_k=3)

    assert len(hits) <= 3
    scores = [score for _unit_id, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_the_ranking_is_deterministic_across_runs():
    first = a_retriever(restart=0.3).retrieve("anything", top_k=5)
    second = a_retriever(restart=0.3).retrieve("anything", top_k=5)

    assert first == second


def test_ties_break_by_unit_id_exactly_as_every_other_retriever_does():
    """A symmetric corpus leaves two units with identical mass; the id decides."""
    dictionary = a_dictionary(k=2, dim=2)
    symmetric = a_matrix(
        dictionary.key,
        rows=[[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
        unit_ids=["ua", "uc", "ub"],
    )
    retriever = DiffusionRetriever(
        matrix=symmetric,
        dictionary=dictionary,
        seed_retriever=SeedStub([("ua", 1.0)]),
        seed_arm="dense",
        restart=0.5,
        normalization=NONE,
        inherits=INHERITED,
    )

    hits = retriever.retrieve("anything", top_k=3)
    tied = [unit_id for unit_id, score in hits if unit_id in {"ub", "uc"}]

    assert tied == ["ub", "uc"]


def test_a_matrix_coded_against_another_dictionary_is_refused():
    dictionary = a_dictionary()
    with pytest.raises(DiffusionError, match="dictionary"):
        DiffusionRetriever(
            matrix=a_matrix("some-other-key"),
            dictionary=dictionary,
            seed_retriever=SeedStub([("u0", 0.9)]),
            seed_arm="dense",
            restart=0.5,
            normalization=NONE,
            inherits=INHERITED,
        )


def test_a_dictionary_of_the_wrong_size_is_refused():
    dictionary = a_dictionary(k=4, dim=4)
    matrix = a_matrix(dictionary.key)  # three concepts, not four
    with pytest.raises(DiffusionError, match="concepts"):
        DiffusionRetriever(
            matrix=matrix,
            dictionary=dictionary,
            seed_retriever=SeedStub([("u0", 0.9)]),
            seed_arm="dense",
            restart=0.5,
            normalization=NONE,
            inherits=INHERITED,
        )


@pytest.mark.parametrize("restart", [-0.1, 1.1])
def test_a_restart_outside_the_unit_interval_is_refused(restart):
    with pytest.raises(DiffusionError, match="restart"):
        a_retriever(restart=restart)


def test_describe_records_every_field_the_data_contract_names():
    retriever = a_retriever(restart=0.4, normalization=STOCHASTIC)

    described = retriever.describe()

    assert described["name"] == "expansion"
    assert described["seed_arm"] == "dense"
    assert described["restart"] == 0.4
    assert described["normalization"] == STOCHASTIC
    assert described["stop_threshold"] == config.STOP_THRESHOLD
    assert described["max_iterations"] == config.MAX_ITERATIONS
    assert described["seed_top_k"] == config.SEED_TOP_K
    assert described["dictionary_key"] == a_dictionary().key
    assert described["k"] == 3
    assert described["view"] == "raw"
    assert described["damping"] == "idf"
    assert described["inherits"] == INHERITED


def test_the_restart_is_applied_at_every_round_not_only_the_first():
    """HU-2. With a restart of 1.0 every round returns to the seed, so two rounds
    of a walk that would otherwise drift still sit exactly on the seed's mass."""
    seed = SeedStub([("u0", 0.6), ("u4", 0.4)])

    always = a_retriever(seed, restart=1.0, max_iterations=4).retrieve("anything", top_k=5)

    assert [unit_id for unit_id, _score in always[:2]] == ["u0", "u4"]
    assert always[0][1] == pytest.approx(0.6)
    assert always[1][1] == pytest.approx(0.4)


def test_the_damping_travels_into_the_walk():
    """The inherited idf weight has to reach the operator, not merely be recorded."""
    weights = np.array([1.0, 5.0, 1.0])
    seed = SeedStub([("u1", 1.0)])

    undamped = a_retriever(seed, restart=0.2).retrieve("anything", top_k=5)
    damped = a_retriever(seed, restart=0.2, concept_weights=weights).retrieve("anything", top_k=5)

    assert undamped != damped
