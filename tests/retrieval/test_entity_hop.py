"""Phase 6, T5: `EntityHopStage`, the Phase 5 entity hop as a second-stage generator (D6).

For a dense list `D(q)`, the stage reads its first ten units, starts from the first, and
returns the Phase 5 entity hop from there: candidates scored by the rarity weights of the
entity nodes they share with `p1`, the read units excluded, positive scores only, ties by
unit id, cut at the depth it is asked for and never padded. The toy below is small enough to
compute every figure by hand.
"""

import math

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.second_hop import node_hop, node_weights
from concept_embeddings_rag.nodes.extraction import FAILED, OK, PROMPT_DIGEST, ExtractionRecord
from concept_embeddings_rag.nodes.index import build_node_index
from concept_embeddings_rag.retrieval.entity_hop import (
    EntityExpansion,
    EntityHopError,
    EntityHopStage,
)

FILLERS = [f"f{index:02d}" for index in range(8)]
UNIT_IDS = ["p1", "r2", *FILLERS, "zz", "aa", "bb", "cc", "dd", "ff"]
N = len(UNIT_IDS)


def record(unit_id: str, entities=(), concepts=(), status: str = OK) -> ExtractionRecord:
    return ExtractionRecord(
        unit_id=unit_id,
        model=config.EXTRACTION_MODEL,
        prompt_digest=PROMPT_DIGEST,
        status=status,
        entities=tuple(entities) if status == OK else (),
        concepts=tuple(concepts) if status == OK else (),
        failure=None if status == OK else "invalid_json",
        input_tokens=1,
        output_tokens=1,
    )


def the_index():
    records = {unit_id: record(unit_id) for unit_id in UNIT_IDS}
    records.update(
        {
            "p1": record("p1", ["Alpha", "Beta"], ["river"]),
            "r2": record("r2", ["Alpha"], []),
            "zz": record("zz", ["Alpha"], []),
            "aa": record("aa", ["Alpha"], []),
            "bb": record("bb", ["Beta", "Alpha"], []),
            "cc": record("cc", [], ["river"]),
            "dd": record("dd", ["Gamma"], []),
            "ff": record("ff", status=FAILED),
        }
    )
    return build_node_index(records, UNIT_IDS, extraction_digest="toy")


def w(df: int) -> float:
    return math.log(1 + N / (1 + df))


W_ALPHA = w(5)  # p1, r2, zz, aa, bb
W_BETA = w(2)  # p1, bb


def dense_list(head: list[str], tail: list[str] = ()) -> list[tuple[str, float]]:
    """A `D(q)` in the declared order: descending score, all scores distinct."""
    units = [*head, *tail]
    return [(unit, 1.0 - index / 100) for index, unit in enumerate(units)]


READ = ["p1", "r2", *FILLERS]


@pytest.fixture(scope="module")
def stage() -> EntityHopStage:
    index = the_index()
    return EntityHopStage(index, node_weights(index))


# --- The hop, by hand ----------------------------------------------------------------------


def test_read_is_the_first_ten_units_and_p1_the_first(stage):
    expansion = stage.expand(dense_list(READ, ["zz", "cc"]), top_k=100)

    assert isinstance(expansion, EntityExpansion)
    assert expansion.read == tuple(READ)
    assert len(expansion.read) == config.PILOT_READ_DEPTH
    assert expansion.p1 == "p1"


def test_candidates_are_scored_by_shared_entity_weights_and_ranked_with_ties_by_unit_id(stage):
    hits = stage.propose(dense_list(READ), top_k=100)

    assert [unit for unit, _ in hits] == ["bb", "aa", "zz"]
    assert hits[0][1] == pytest.approx(W_ALPHA + W_BETA)
    assert hits[1][1] == pytest.approx(W_ALPHA)
    assert hits[2][1] == pytest.approx(W_ALPHA)


def test_positives_count_every_candidate_before_truncation_and_nothing_is_padded(stage):
    shallow = stage.expand(dense_list(READ), top_k=2)
    deep = stage.expand(dense_list(READ), top_k=100)

    assert [c.unit_id for c in shallow.candidates] == ["bb", "aa"]
    assert shallow.positives == deep.positives == 3
    assert len(deep.candidates) == 3


def test_the_read_units_are_excluded_even_when_they_share_the_heaviest_nodes(stage):
    hits = stage.propose(dense_list(READ), top_k=100)
    assert not {unit for unit, _ in hits} & set(READ)


def test_a_unit_below_the_read_depth_is_still_a_candidate(stage):
    """Units at ranks 11-100 of `D(q)` can be proposed: only the ten read ones are excluded."""
    hits = stage.propose(dense_list(READ, ["zz", "aa"]), top_k=100)
    assert {"zz", "aa"} <= {unit for unit, _ in hits}


def test_a_paragraph_sharing_only_a_concept_with_p1_is_never_proposed(stage):
    hits = stage.propose(dense_list(READ), top_k=100)
    assert "cc" not in {unit for unit, _ in hits}


def test_the_candidates_carry_their_shared_entity_nodes_heaviest_first(stage):
    expansion = stage.expand(dense_list(READ), top_k=100)
    bb = expansion.candidates[0]

    assert [node.form for node in bb.nodes] == ["beta", "alpha"]
    assert {node.type for node in bb.nodes} == {"entity"}
    assert math.fsum(node.weight for node in bb.nodes) == pytest.approx(bb.score, rel=1e-12)


def test_p1_with_no_entity_node_gives_an_empty_list(stage):
    head = ["cc", "r2", *FILLERS]
    expansion = stage.expand(dense_list(head), top_k=100)

    assert expansion.candidates == ()
    assert expansion.positives == 0
    assert expansion.p1_entity_nodes == 0


def test_p1_among_the_failed_extractions_gives_an_empty_list(stage):
    head = ["ff", "r2", *FILLERS]
    assert stage.propose(dense_list(head), top_k=100) == []
    assert stage.expand(dense_list(head), top_k=100).p1_entity_nodes == 0


def test_p1_entity_nodes_counts_only_entities(stage):
    assert stage.expand(dense_list(READ), top_k=100).p1_entity_nodes == 2


def test_a_list_shorter_than_the_read_depth_is_read_whole():
    index = the_index()
    stage = EntityHopStage(index, node_weights(index))
    expansion = stage.expand(dense_list(["p1", "zz"]), top_k=100)
    assert expansion.read == ("p1", "zz")
    assert [c.unit_id for c in expansion.candidates] == ["bb", "aa", "r2"]


# --- Unchanged Phase 5 semantics -------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 3, 100])
def test_the_output_equals_a_direct_phase_5_node_hop_call(depth: int):
    index = the_index()
    weights = node_weights(index)
    stage = EntityHopStage(index, weights)
    rows = [UNIT_IDS.index(unit) for unit in READ]

    direct, positives = node_hop(
        index, weights, types=("entity",), p1=rows[0], read=rows, depth=depth
    )
    expansion = stage.expand(dense_list(READ), top_k=depth)

    assert list(expansion.candidates) == direct
    assert expansion.positives == positives
    assert stage.propose(dense_list(READ), top_k=depth) == [(c.unit_id, c.score) for c in direct]


# --- Refusals -----------------------------------------------------------------------------


def test_a_dense_list_out_of_score_order_is_refused(stage):
    hits = dense_list(READ)
    hits[0], hits[1] = hits[1], hits[0]
    with pytest.raises(EntityHopError, match="order"):
        stage.propose(hits, top_k=100)


def test_a_dense_list_whose_tie_is_not_broken_by_unit_id_is_refused(stage):
    hits = [("r2", 0.9), ("p1", 0.9), *[(unit, 0.5 - i / 100) for i, unit in enumerate(FILLERS)]]
    with pytest.raises(EntityHopError, match="order"):
        stage.propose(hits, top_k=100)


def test_a_dense_list_holding_a_duplicate_is_refused(stage):
    hits = [("p1", 0.9), ("p1", 0.8), ("r2", 0.7)]
    with pytest.raises(EntityHopError, match="twice"):
        stage.propose(hits, top_k=100)


def test_an_empty_dense_list_is_refused(stage):
    with pytest.raises(EntityHopError, match="nothing"):
        stage.propose([], top_k=100)


def test_a_unit_outside_the_index_is_refused(stage):
    with pytest.raises(EntityHopError, match="index"):
        stage.propose([("nowhere", 1.0)], top_k=100)


def test_a_depth_of_zero_or_less_is_refused(stage):
    with pytest.raises(EntityHopError, match="top_k"):
        stage.propose(dense_list(READ), top_k=0)


# --- Memoization --------------------------------------------------------------------------


def test_memoized_expansions_equal_fresh_ones_and_are_copies():
    index = the_index()
    weights = node_weights(index)
    memoized = EntityHopStage(index, weights)
    first = memoized.expand(dense_list(READ), top_k=100)
    second = memoized.expand(dense_list(READ), top_k=100)
    fresh = EntityHopStage(index, weights).expand(dense_list(READ), top_k=100)

    assert first == second == fresh
    assert first is not second
    listed = memoized.propose(dense_list(READ), top_k=100)
    listed.clear()
    assert memoized.propose(dense_list(READ), top_k=100) == fresh_hits(index, weights)


def fresh_hits(index, weights):
    return EntityHopStage(index, weights).propose(dense_list(READ), top_k=100)


def test_the_memo_is_keyed_by_the_read_list_and_the_depth():
    index = the_index()
    stage = EntityHopStage(index, node_weights(index))
    stage.expand(dense_list(READ), top_k=100)
    stage.expand(dense_list(READ, ["zz"]), top_k=100)  # same read list: a hit
    stage.expand(dense_list(READ), top_k=2)  # another depth: a miss

    assert stage.memo_size == 2
