"""T6: the walk stops on its own, and says what it cost.

HU-4 asks for three things that are easy to approximate and worth not approximating:
the stop is **per question** rather than a constant number of rounds, the threshold
is declared rather than fitted, and the iterations are reported as a distribution
rather than as a mean. A system that stops at round 1 on nine questions in ten is a
different system from one that runs to the cap, and a mean of 1.4 hides both.

The counters are mutable state on a retriever, which is a smell. It is bounded here
rather than argued away: they reset per run, and they are never read by the path
that produces a score - both asserted below.
"""

import numpy as np
import pytest
from diffusion_fixtures import INHERITED, SeedStub, a_dictionary, a_matrix

from concept_embeddings_rag import config
from concept_embeddings_rag.retrieval.diffusion import (
    CAP,
    CONVERGED,
    NONE,
    STOCHASTIC,
    THRESHOLD,
    DiffusionRetriever,
)


def walking(
    seed: SeedStub | None = None,
    *,
    restart: float = 0.5,
    normalization: str = NONE,
    stop_threshold: float = config.STOP_THRESHOLD,
    max_iterations: int = config.MAX_ITERATIONS,
) -> DiffusionRetriever:
    dictionary = a_dictionary()
    return DiffusionRetriever(
        matrix=a_matrix(dictionary.key),
        dictionary=dictionary,
        seed_retriever=seed if seed is not None else SeedStub([("u0", 0.9)]),
        seed_arm="dense",
        restart=restart,
        normalization=normalization,
        stop_threshold=stop_threshold,
        max_iterations=max_iterations,
        inherits=INHERITED,
    )


def test_a_full_restart_converges_on_the_first_round():
    """Nothing moves when every round returns the whole mass to the seed."""
    retriever = walking(restart=1.0)

    retriever.retrieve("anything", top_k=5)

    assert retriever.stats.iterations == [1]
    assert retriever.stats.stop_reasons[CONVERGED] == 1


def test_a_settling_walk_stops_on_the_threshold_before_the_cap():
    retriever = walking(restart=0.5, stop_threshold=1e-3, max_iterations=50)

    retriever.retrieve("anything", top_k=5)

    assert retriever.stats.stop_reasons[THRESHOLD] == 1
    assert retriever.stats.iterations[0] < 50


def test_a_walk_that_has_not_settled_stops_at_the_cap_and_says_so():
    """An impossible threshold is the only way to see the cap fire deterministically."""
    retriever = walking(restart=0.5, stop_threshold=1e-30, max_iterations=3)

    retriever.retrieve("anything", top_k=5)

    assert retriever.stats.iterations == [3]
    assert retriever.stats.stop_reasons[CAP] == 1
    assert retriever.stats.cap_hits == 1


@pytest.mark.parametrize("restart", config.RESTART_GRID)
@pytest.mark.parametrize("normalization", config.NORMALIZATION_ARMS)
def test_the_cap_is_never_exceeded_at_any_grid_cell(restart, normalization):
    retriever = walking(restart=restart, normalization=normalization, max_iterations=4)

    retriever.retrieve("anything", top_k=5)

    assert max(retriever.stats.iterations) <= 4


def test_the_stop_is_per_question_not_per_run():
    """Two questions that settle at different speeds must be recorded separately."""
    retriever = walking(restart=1.0)
    retriever.retrieve("anything", top_k=5)

    retriever.seed_retriever = SeedStub([("u1", 0.5), ("u3", 0.5)])
    retriever.restart = 0.2
    retriever.retrieve("anything else", top_k=5)

    assert len(retriever.stats.iterations) == 2
    assert retriever.stats.iterations[0] != retriever.stats.iterations[1]


def test_the_histogram_accounts_for_every_question_asked():
    retriever = walking(restart=0.4)

    for _ in range(5):
        retriever.retrieve("anything", top_k=5)

    assert sum(retriever.stats.iterations_histogram.values()) == 5
    assert retriever.stats.n_questions == 5


def test_the_mean_is_reported_beside_the_distribution_not_instead_of_it():
    retriever = walking(restart=0.4)
    retriever.retrieve("anything", top_k=5)

    assert retriever.stats.mean_iterations == pytest.approx(
        float(np.mean(retriever.stats.iterations))
    )
    assert set(retriever.stats.iterations_histogram) == set(retriever.stats.iterations)


def test_two_sparse_products_are_counted_per_round_and_no_more():
    """Decision D2's whole cost model: a round is two matvecs, never a unit x unit product."""
    retriever = walking(restart=0.5, stop_threshold=1e-30, max_iterations=3)

    retriever.retrieve("anything", top_k=5)

    assert retriever.stats.sparse_products == 2 * 3


def test_a_clipped_seed_hit_is_counted_and_reaches_the_statistics():
    retriever = walking(SeedStub([("u0", 0.9), ("u1", -0.2)]), restart=0.5)

    retriever.retrieve("anything", top_k=5)

    assert retriever.stats.clipped_seed_hits == 1


def test_the_counters_reset_per_run():
    retriever = walking(restart=0.4)
    retriever.retrieve("anything", top_k=5)

    retriever.reset_stats()

    assert retriever.stats.iterations == []
    assert retriever.stats.sparse_products == 0
    assert retriever.stats.clipped_seed_hits == 0
    assert retriever.stats.n_questions == 0


def test_the_counters_never_reach_the_score():
    """The same question, asked twice with statistics accumulated in between."""
    retriever = walking(restart=0.4)

    first = retriever.retrieve("anything", top_k=5)
    for _ in range(4):
        retriever.retrieve("anything", top_k=5)
    again = retriever.retrieve("anything", top_k=5)

    assert first == again


def test_the_statistics_are_reported_in_the_shape_a_result_config_records():
    retriever = walking(restart=0.4, normalization=STOCHASTIC)
    retriever.retrieve("anything", top_k=5)

    described = retriever.stats.describe()

    assert described["mean_iterations"] == retriever.stats.mean_iterations
    assert described["iterations_histogram"] == {
        str(k): v for k, v in retriever.stats.iterations_histogram.items()
    }
    assert described["cap_hits"] == retriever.stats.stop_reasons[CAP]
    assert described["clipped_seed_hits"] == 0
    assert described["n_questions"] == 1
    assert described["mean_sparse_products"] == pytest.approx(
        retriever.stats.sparse_products / retriever.stats.n_questions
    )


def test_describing_an_empty_run_is_refused_rather_than_reported_as_zero():
    """A mean of zero iterations would read as a measurement; it is an absence."""
    retriever = walking()

    with pytest.raises(ValueError, match="no question"):
        retriever.stats.describe()
