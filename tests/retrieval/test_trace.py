"""T8: the trace the walk leaves behind, and what it may not cost.

HU-5 asks for a per-query record of which concept entered at which iteration and
which unit it brought along, so that a recovered question can be explained rather
than celebrated. Two properties are what make such a record worth reading, and
both are asserted here rather than intended:

- **It is produced inside the loop.** A trace reconstructed afterwards from the
  final ranking is a second computation, and a second computation can disagree
  with the run it claims to explain while looking perfectly plausible.
- **Producing it changes no score.** The whole point of a diagnostic is that the
  system diagnosed is the system measured, so the ranking is compared with tracing
  on and with it off, on the same question, and the two are identical - not close.

The third property is about what the trace holds: concept **ids**, never labels.
`retrieval/` may not import the labelling artifact at all, and a trace that carried
a label would be the one way that rule could be broken from inside the data.
"""

import numpy as np
import pytest
from diffusion_fixtures import INHERITED, UNIT_IDS, SeedStub, a_dictionary, a_matrix

from concept_embeddings_rag import config
from concept_embeddings_rag.retrieval.diffusion import (
    CAP,
    NONE,
    STOCHASTIC,
    SYMMETRIC,
    DiffusionError,
    DiffusionRetriever,
    ExpansionTrace,
    IterationTrace,
    seed_mass,
)

ARMS = (NONE, SYMMETRIC, STOCHASTIC)


def a_retriever(
    seed: SeedStub | None = None,
    *,
    seed_arm: str = "dense",
    restart: float = 0.5,
    normalization: str = NONE,
    concept_weights: np.ndarray | None = None,
    max_iterations: int = config.MAX_ITERATIONS,
    stop_threshold: float = config.STOP_THRESHOLD,
) -> DiffusionRetriever:
    dictionary = a_dictionary()
    return DiffusionRetriever(
        matrix=a_matrix(dictionary.key),
        dictionary=dictionary,
        seed_retriever=seed if seed is not None else SeedStub([("u0", 0.9)]),
        seed_arm=seed_arm,
        restart=restart,
        normalization=normalization,
        concept_weights=concept_weights,
        stop_threshold=stop_threshold,
        max_iterations=max_iterations,
        inherits=INHERITED,
    )


# --- Producing the trace changes no score -------------------------------------


@pytest.mark.parametrize("normalization", ARMS)
@pytest.mark.parametrize("restart", config.RESTART_GRID)
def test_the_ranking_is_identical_with_tracing_on_and_off(normalization: str, restart: float):
    """The criterion HU-5 states, over every cell of the grid this phase will measure."""
    plain = a_retriever(restart=restart, normalization=normalization)
    traced = a_retriever(restart=restart, normalization=normalization)

    without = plain.retrieve("q", top_k=5)
    with_it, _trace = traced.retrieve_with_trace("q", top_k=5, qid="q1")

    assert [unit for unit, _score in with_it] == [unit for unit, _score in without]
    for (_unit_a, score_a), (_unit_b, score_b) in zip(with_it, without, strict=True):
        assert score_a == score_b, "tracing moved a score, so the traced system is not the system"


def test_tracing_costs_the_same_walk_and_records_the_same_statistics():
    """Same rounds, same stop reason, same sparse products: one walk, watched or not."""
    plain = a_retriever(restart=0.2)
    traced = a_retriever(restart=0.2)

    plain.retrieve("q", top_k=5)
    _hits, trace = traced.retrieve_with_trace("q", top_k=5, qid="q1")

    assert traced.stats.describe() == plain.stats.describe()
    assert len(trace.iterations) == plain.stats.iterations[0]
    assert trace.stop_reason == next(iter(plain.stats.stop_reasons))


# --- It is produced inside the loop -------------------------------------------


def test_the_trace_holds_one_entry_per_round_the_walk_actually_ran():
    retriever = a_retriever(restart=0.2, max_iterations=3, stop_threshold=1e-12)
    _hits, trace = retriever.retrieve_with_trace("q", top_k=5, qid="q1")

    assert len(trace.iterations) == 3
    assert [row.index for row in trace.iterations] == [1, 2, 3]
    assert trace.stop_reason == CAP


def test_only_the_last_round_is_the_one_that_stopped_the_walk():
    retriever = a_retriever(restart=0.2, max_iterations=4)
    _hits, trace = retriever.retrieve_with_trace("q", top_k=5, qid="q1")

    stopped = [row.index for row in trace.iterations if row.stopped]
    assert stopped == [trace.iterations[-1].index]


@pytest.mark.parametrize("restart", config.RESTART_GRID)
def test_the_new_mass_of_each_round_agrees_with_the_reason_the_walk_stopped(restart: float):
    """`new_mass` is D6's `delta_t`, so it is what the threshold was read against.

    Whichever way the walk ends, the trace has to be readable against the rule: no
    round before the last may have been below the threshold, and the last one is
    below it if and only if the threshold is what stopped the walk.
    """
    retriever = a_retriever(restart=restart)
    _hits, trace = retriever.retrieve_with_trace("q", top_k=5, qid="q1")

    for row in trace.iterations[:-1]:
        assert row.new_mass >= config.STOP_THRESHOLD

    last = trace.iterations[-1]
    if trace.stop_reason == CAP:
        assert last.new_mass >= config.STOP_THRESHOLD
        assert len(trace.iterations) == config.MAX_ITERATIONS
    else:
        assert last.new_mass < config.STOP_THRESHOLD


def test_the_concepts_of_a_round_are_the_ones_the_operator_actually_moved_mass_through():
    """Read against the operator rather than against the trace's own arithmetic."""
    retriever = a_retriever(restart=0.5, normalization=NONE)
    seed = retriever.seed_retriever.retrieve("q", top_k=retriever.seed_top_k)
    start, _clipped = seed_mass(seed, retriever._index)
    expected = retriever.operator.to_concepts(start)

    _hits, trace = retriever.retrieve_with_trace("q", top_k=5, qid="q1")
    first = trace.iterations[0]

    for concept_id, mass in first.concepts:
        assert mass == pytest.approx(float(expected[concept_id]))
    assert [concept for concept, _mass in first.concepts] == sorted(
        (int(c) for c in np.nonzero(expected)[0]),
        key=lambda c: (-float(expected[c]), c),
    )[: len(first.concepts)]


def test_a_later_round_brings_in_a_unit_the_seed_never_returned():
    """The second hop, visible in the artifact: u2 and u3 are two rounds from u0."""
    retriever = a_retriever(SeedStub([("u0", 0.9)]), restart=0.2, normalization=NONE)
    _hits, trace = retriever.retrieve_with_trace("q", top_k=5, qid="q1")

    brought = {unit for row in trace.iterations for unit, _mass in row.units}
    assert brought - {"u0"}, "the walk recorded no unit beyond the one the seed returned"


def test_every_unit_the_trace_records_gained_mass_in_that_round():
    retriever = a_retriever(restart=0.3)
    _hits, trace = retriever.retrieve_with_trace("q", top_k=5, qid="q1")

    for row in trace.iterations:
        for _unit, mass in row.units:
            assert mass > 0.0


# --- What the trace says about the run that produced it -----------------------


def test_the_trace_names_the_question_the_arm_and_the_configuration():
    retriever = a_retriever(seed_arm="conceptual", restart=0.6, normalization=STOCHASTIC)
    _hits, trace = retriever.retrieve_with_trace("q", top_k=5, qid="5a8b57f25542995d1e6f1371")

    assert trace.qid == "5a8b57f25542995d1e6f1371"
    assert trace.seed_arm == "conceptual"
    assert trace.config_digest == retriever.config_digest


def test_two_configurations_that_differ_anywhere_do_not_share_a_digest():
    baseline = a_retriever(restart=0.2, normalization=NONE).config_digest

    assert a_retriever(restart=0.4, normalization=NONE).config_digest != baseline
    assert a_retriever(restart=0.2, normalization=SYMMETRIC).config_digest != baseline
    assert a_retriever(restart=0.2, normalization=NONE, seed_arm="conceptual").config_digest != (
        baseline
    )


def test_the_same_configuration_digests_the_same_way_on_every_construction():
    assert a_retriever().config_digest == a_retriever().config_digest


def test_a_trace_cannot_be_asked_for_without_naming_the_question_it_describes():
    with pytest.raises(DiffusionError, match="qid"):
        a_retriever().retrieve_with_trace("q", top_k=5, qid="")


# --- Concept ids, never labels ------------------------------------------------


def test_the_trace_records_concepts_as_ids():
    retriever = a_retriever()
    _hits, trace = retriever.retrieve_with_trace("q", top_k=5, qid="q1")

    for row in trace.iterations:
        for concept_id, _mass in row.concepts:
            assert isinstance(concept_id, int)
            assert 0 <= concept_id < retriever.matrix.X.shape[1]


def test_a_concept_named_rather_than_numbered_is_refused():
    """The one way a label could reach `retrieval/` is from inside the data."""
    with pytest.raises(DiffusionError, match="id"):
        IterationTrace(
            index=1,
            concepts=(("baseball players and Florida places", 0.4),),
            units=(("u0", 0.2),),
            new_mass=0.1,
            stopped=True,
        )


def test_a_trace_of_no_round_at_all_is_refused():
    """Every walk runs at least one round, so an empty trace describes nothing."""
    with pytest.raises(DiffusionError, match="round"):
        ExpansionTrace(
            qid="q1", seed_arm="dense", config_digest="0" * 16, iterations=(), stop_reason=CAP
        )


def test_a_trace_whose_rounds_are_not_consecutive_is_refused():
    rows = (
        IterationTrace(
            index=1, concepts=((0, 0.4),), units=(("u0", 0.2),), new_mass=0.1, stopped=False
        ),
        IterationTrace(
            index=3, concepts=((0, 0.4),), units=(("u0", 0.2),), new_mass=0.1, stopped=True
        ),
    )
    with pytest.raises(DiffusionError, match="round"):
        ExpansionTrace(
            qid="q1", seed_arm="dense", config_digest="0" * 16, iterations=rows, stop_reason=CAP
        )


def test_a_trace_naming_an_arm_this_phase_does_not_have_is_refused():
    rows = (
        IterationTrace(
            index=1, concepts=((0, 0.4),), units=(("u0", 0.2),), new_mass=0.1, stopped=True
        ),
    )
    with pytest.raises(DiffusionError, match="arm"):
        ExpansionTrace(
            qid="q1", seed_arm="hybrid", config_digest="0" * 16, iterations=rows, stop_reason=CAP
        )


def test_the_units_of_a_trace_are_ids_of_the_pool_it_walked():
    retriever = a_retriever()
    _hits, trace = retriever.retrieve_with_trace("q", top_k=5, qid="q1")

    for row in trace.iterations:
        for unit_id, _mass in row.units:
            assert unit_id in set(UNIT_IDS)
