"""Phase 9, S6: the exact column-wise entity hop gives the Phase 5 node hop's output, bit for bit.

The column-wise hop exists only because the reference recomputes the entity column mask
over every node on every call, which is linear in the node vocabulary - millions of nodes at
FullWiki scale. It may replace the reference only if nothing a figure depends on can move:
the same candidates, in the same order, with the same float scores and the same shared
nodes, and the same positive count. Asserted here with `==`, never with a tolerance, on toy
indexes built to produce ties and on the Phase 7 GLiNER index itself when it is present.
"""

import random

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.second_hop import node_hop, node_hop_columnwise, node_weights
from concept_embeddings_rag.nodes.index import build_node_index, load_node_index
from concept_embeddings_rag.nodes.local_extraction import record_from_forms
from concept_embeddings_rag.retrieval.entity_hop import ColumnwiseEntityHopStage, EntityHopStage

TYPES = config.ENTITY_HOP_TYPES


def a_random_index(seed: int, n_units: int = 300, vocabulary: int = 40):
    """Few forms over many rows, so shared-node sets repeat and scores tie often."""
    rng = random.Random(seed)  # noqa: S311 - fixture layout, not security
    unit_ids = [f"{rng.getrandbits(64):016x}" for _ in range(n_units)]
    records = {
        unit_id: record_from_forms(
            unit_id,
            [f"Form {rng.randrange(vocabulary)}" for _ in range(rng.randrange(0, 5))],
            model="fixture",
            configuration_digest="f",
        )
        for unit_id in unit_ids
    }
    return build_node_index(records, unit_ids, extraction_digest=f"{seed:064d}")


def assert_same(index, weights, columnwise, *, p1, read, depth):
    expected = node_hop(index, weights, types=TYPES, p1=p1, read=read, depth=depth)
    observed = node_hop_columnwise(index, weights, columnwise, p1=p1, read=read, depth=depth)
    assert observed == expected


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("depth", [1, 3, 10, 100, 1000])
def test_the_columnwise_hop_equals_the_reference_on_tie_heavy_indexes(seed, depth):
    index = a_random_index(seed)
    weights = node_weights(index)
    columnwise = ColumnwiseEntityHopStage.columns_for(index)
    for p1 in range(len(index.unit_ids)):
        read = [p1, *((p1 + offset) % len(index.unit_ids) for offset in (1, 2, 7))]
        assert_same(index, weights, columnwise, p1=p1, read=read, depth=depth)


def test_the_columnwise_stage_expands_exactly_as_the_reference_stage():
    index = a_random_index(4)
    weights = node_weights(index)
    reference = EntityHopStage(index, weights)
    columnwise = ColumnwiseEntityHopStage(index, weights)
    ordered = sorted(index.unit_ids)
    for start in range(0, len(ordered) - 12, 5):
        dense = [
            (unit_id, 1.0 - position / 100)
            for position, unit_id in enumerate(ordered[start : start + 12])
        ]
        assert columnwise.expand(dense, 100) == reference.expand(dense, 100)
    assert columnwise.name == reference.name


def test_the_columnwise_hop_equals_the_reference_on_the_phase_7_gliner_index():
    """Data-dependent: every seventh row of the historical pool as p1, at the evaluation depth."""
    directory = config.PHASE_8_GLINER_DIR
    if not (directory / f"nodes-{config.PHASE_8_GLINER_EXTRACTION_DIGEST[:16]}.npz").exists():
        pytest.skip("the Phase 7 GLiNER index is not on this checkout")
    index = load_node_index(directory, extraction_digest=config.PHASE_8_GLINER_EXTRACTION_DIGEST)
    weights = node_weights(index)
    columnwise = ColumnwiseEntityHopStage.columns_for(index)
    n = len(index.unit_ids)
    for p1 in range(0, n, 7):
        read = [(p1 + offset) % n for offset in range(config.PILOT_READ_DEPTH)]
        assert_same(index, weights, columnwise, p1=p1, read=read, depth=config.EVALUATION_TOP_K)
