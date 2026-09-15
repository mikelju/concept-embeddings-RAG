"""T8 of Phase 5: the node hop for the three arms, and the trace of every ranked candidate.

The toy pool is built so that the answers can be computed by hand and so that the two
ways of breaking a tie disagree: `zz` sits before `aa` in pool order, and the hop must
still rank `aa` first (decision D8).
"""

import math

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.second_hop import node_hop, node_weights
from concept_embeddings_rag.nodes.extraction import OK, PROMPT_DIGEST, ExtractionRecord
from concept_embeddings_rag.nodes.index import build_node_index

ENTITY, CONCEPT = config.NODE_TYPES
E = (ENTITY,)
C = (CONCEPT,)
EC = (ENTITY, CONCEPT)

UNIT_IDS = ["p1x", "zz", "aa", "mm", "rr"]
N = len(UNIT_IDS)


def record(unit_id: str, entities: list[str], concepts: list[str]) -> ExtractionRecord:
    return ExtractionRecord(
        unit_id=unit_id,
        model=config.EXTRACTION_MODEL,
        prompt_digest=PROMPT_DIGEST,
        status=OK,
        entities=tuple(entities),
        concepts=tuple(concepts),
        failure=None,
        input_tokens=1,
        output_tokens=1,
    )


def the_index():
    records = {
        "p1x": record("p1x", ["Alpha"], ["river", "bridge"]),
        "zz": record("zz", ["Alpha"], []),
        "aa": record("aa", ["Alpha"], []),
        "mm": record("mm", [], ["river"]),
        "rr": record("rr", ["Beta"], ["river", "bridge"]),
    }
    return build_node_index(records, UNIT_IDS, extraction_digest="toy")


def w(df: int) -> float:
    """The Phase 3 rarity weight, written out: log(1 + N / (1 + df))."""
    return math.log(1 + N / (1 + df))


W_ALPHA, W_BRIDGE, W_RIVER = w(3), w(2), w(3)
P1 = 0


def ranked(types, read=frozenset({P1}), depth=10):
    index = the_index()
    candidates, positives = node_hop(
        index, node_weights(index), types=types, p1=P1, read=read, depth=depth
    )
    return candidates, positives


def test_the_weights_are_the_rarity_formula_over_node_document_frequency():
    index = the_index()
    by_form = {(node.type, node.form): node.node_id for node in index.nodes}

    weights = node_weights(index)

    assert weights[by_form[(ENTITY, "alpha")]] == pytest.approx(W_ALPHA)
    assert weights[by_form[(ENTITY, "beta")]] == pytest.approx(w(1))
    assert weights[by_form[(CONCEPT, "bridge")]] == pytest.approx(W_BRIDGE)
    assert weights[by_form[(CONCEPT, "river")]] == pytest.approx(W_RIVER)


def test_entity_only_scores_shared_entities_and_breaks_the_tie_by_unit_id():
    candidates, positives = ranked(E)

    assert [c.unit_id for c in candidates] == ["aa", "zz"]
    assert [c.score for c in candidates] == pytest.approx([W_ALPHA, W_ALPHA])
    assert positives == 2


def test_concept_only_scores_shared_concepts_heaviest_first():
    candidates, positives = ranked(C)

    assert [c.unit_id for c in candidates] == ["rr", "mm"]
    assert candidates[0].score == pytest.approx(W_BRIDGE + W_RIVER)
    assert [n.form for n in candidates[0].nodes] == ["bridge", "river"]
    assert positives == 2


def test_entity_plus_concept_uses_both_namespaces_with_their_own_weights():
    candidates, positives = ranked(EC)

    assert [c.unit_id for c in candidates] == ["rr", "aa", "mm", "zz"]
    assert positives == 4
    assert {(n.type, n.form) for n in candidates[0].nodes} == {
        (CONCEPT, "bridge"),
        (CONCEPT, "river"),
    }


def test_ties_follow_unit_ids_not_pool_rows():
    candidates, _ = ranked(E)
    rows = [c.row for c in candidates]

    # `aa` is row 2 and `zz` row 1: a positional tie-break would put `zz` first.
    assert rows == [2, 1]


@pytest.mark.parametrize("types", [E, C, EC])
def test_contributions_sum_to_the_score_of_every_ranked_candidate(types):
    candidates, _ = ranked(types)

    assert candidates
    for candidate in candidates:
        assert math.fsum(n.weight for n in candidate.nodes) == pytest.approx(
            candidate.score, rel=1e-12
        )
        weights = [n.weight for n in candidate.nodes]
        assert weights == sorted(weights, reverse=True)


@pytest.mark.parametrize("types", [E, C, EC])
def test_p1_and_read_paragraphs_never_appear(types):
    candidates, _ = ranked(types, read=frozenset({P1, 2}))

    assert "p1x" not in {c.unit_id for c in candidates}
    assert "aa" not in {c.unit_id for c in candidates}


def test_the_depth_bounds_the_ranking_but_not_the_positive_count():
    candidates, positives = ranked(EC, depth=1)

    assert [c.unit_id for c in candidates] == ["rr"]
    assert positives == 4


def test_a_p1_with_no_node_of_the_arm_reaches_nothing():
    index = the_index()

    candidates, positives = node_hop(
        index, node_weights(index), types=E, p1=3, read=frozenset({3}), depth=10
    )

    assert candidates == [] and positives == 0
