"""T2 of Phase 5: the second-hop protocol lives in `src`, and the diagnostic still reproduces.

The toy tests pin each piece of the protocol the pilot reuses. The data-dependent test is
the continuity check of HU-5: moving the code must not move a single figure of the
committed `data/diagnostics/second_hop-dev.json`.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import second_hop


class FakeBM25:
    """Returns a fixed ranking whatever the text, so the exclusion rules can be read off."""

    def __init__(self, hits: list[tuple[str, float]]) -> None:
        self.hits = hits
        self.asked: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        self.asked.append((query, top_k))
        return self.hits[:top_k]


def test_the_frozen_comparator_names_a_hop_the_protocol_defines():
    """Deferred from T1: the gate's comparator must be one of the protocol's hops."""
    assert config.GATE_COMPARATOR in second_hop.BASELINE_HOPS


def test_dense_order_is_descending_and_stable_on_ties():
    vectors = np.array([[1.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 1.0]])
    order = second_hop.dense_order(vectors, np.array([1.0, 0.0]))
    assert order.tolist() == [0, 2, 1, 3]


def test_read_and_p1_are_the_head_of_the_dense_order():
    order = np.array([4, 2, 0, 1, 3])
    read = second_hop.read_rows(order, depth=2)
    assert read == [4, 2]
    assert second_hop.origin(read) == 4


def test_ranking_never_returns_a_read_row_even_when_it_scores_highest():
    scores = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    ranked = second_hop.rank_scores(scores, read={0, 1}, positive_only=False, depth=3)
    assert ranked == [2, 3, 4]


def test_positive_only_drops_zero_and_negative_scores():
    scores = np.array([0.0, 3.0, -1.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    ranked = second_hop.rank_scores(scores, read=set(), positive_only=True, depth=5)
    assert ranked == [1, 4]


def test_ranking_respects_the_depth():
    scores = np.arange(10, dtype=np.float64)
    ranked = second_hop.rank_scores(scores, read=set(), positive_only=False, depth=4)
    assert ranked == [9, 8, 7, 6]


def test_dense_continue_reads_on_from_where_the_reader_stopped():
    order = np.arange(20)[::-1]
    assert second_hop.dense_continue(order, read_depth=5, depth=3) == [14, 13, 12]


def test_bm25_ranking_drops_read_paragraphs_and_non_positive_scores():
    index = {f"u{i}": i for i in range(6)}
    bm25 = FakeBM25([("u0", 5.0), ("u3", 4.0), ("u1", 3.0), ("u5", 0.0), ("u2", 1.0)])
    ranked = second_hop.rank_bm25(bm25, index, "text", read={0, 1}, read_depth=2, depth=10)
    assert ranked == [3, 2]
    # Asked deep enough that excluding the whole read list still leaves `depth` candidates.
    assert bm25.asked == [("text", 10 + 2 + 1)]


def test_the_concept_hop_scores_shared_concepts_weighted_and_reports_reach():
    X = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0],  # p1
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
    )
    weights = np.array([1.0, 10.0, 0.5])
    ranked, reach = second_hop.concept_hop(X, weights, p1=0, read={0}, depth=3)
    # scores: row1 = 1*1*1 = 1.0, row2 = 0, row3 = 2*0.5*1 = 1.0, row4 = 1 + 1.0 = 2.0
    assert ranked[0] == 4
    assert set(ranked) == {1, 3, 4}
    # Positive scores over the whole pool, p1 included: rows 0, 1, 3, 4.
    assert reach == pytest.approx(4 / 5)


def test_truncation_keeps_each_rows_strongest_concepts():
    X = sparse.csr_matrix(np.array([[0.1, 0.9, 0.5, 0.0], [0.3, 0.0, 0.0, 0.2]]))
    kept = second_hop.truncate_rows(X, 1).toarray()
    assert kept.tolist() == [[0.0, 0.9, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0]]
    assert second_hop.truncate_rows(X, None) is X


# --- Continuity with the committed diagnostic ---------------------------------------

DIAGNOSTIC = config.DATA_DIR / "diagnostics" / "second_hop-dev.json"
SCRIPT = config.PROJECT_ROOT / "scripts" / "second_hop_diagnostic.py"


def _real_inputs_present() -> bool:
    return (
        (config.DATA_DIR / "pool.json").exists()
        and (config.DATA_DIR / "hotpot_raw.json").exists()
        and any(config.CACHE_DIR.glob("*.npz"))
        and any(config.CONCEPTS_DIR.glob("matrix-*.npz"))
        and any(config.QUESTION_CACHE_DIR.glob("*.npz"))
    )


def test_the_refactored_diagnostic_reproduces_the_committed_figures_exactly(tmp_path: Path):
    if not _real_inputs_present():
        pytest.skip("the pool, embeddings or concept spaces are not on this machine")

    spec = importlib.util.spec_from_file_location("second_hop_diagnostic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = tmp_path / "second_hop-dev.json"
    module.main(out)

    committed = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    regenerated = json.loads(out.read_text(encoding="utf-8"))
    for key in ("hit_rates", "reach_from_p1", "structure", "n_missing_paragraphs"):
        assert regenerated[key] == committed[key], key
