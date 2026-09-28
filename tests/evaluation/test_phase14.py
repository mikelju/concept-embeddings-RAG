"""Phase 14, S1: similarity, normalization, mixing, ranking, grid, tie rule, gates and label.

These are the functions whose output can change a measured result: how a candidate's
similarity is taken, how both terms are scaled and mixed, how the list is ranked and cut,
which grid point dev selects, whether the integrity check and the dev gate pass, and which
label the held-out comparison gets. Everything is fixture-based and fast.
"""

import random

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import phase14 as p14

# --- D1: candidate similarity ----------------------------------------------------------------


def test_candidate_similarity_is_the_float32_dot_product_of_the_gathered_rows():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((20, 8)).astype(np.float32)
    question = rng.standard_normal(8).astype(np.float32)
    rows = np.array([3, 17, 0, 3])
    got = p14.candidate_similarity(vectors, rows, question)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, vectors[rows] @ question)


# --- D2: normalization and mixing ------------------------------------------------------------


def test_min_max_scales_to_the_unit_interval_in_float64():
    got = p14.min_max(np.array([2.0, 4.0, 3.0], dtype=np.float32))
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, [0.0, 1.0, 0.5])


def test_min_max_of_a_constant_term_is_zero_everywhere():
    np.testing.assert_array_equal(p14.min_max(np.array([0.7, 0.7, 0.7])), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(p14.min_max(np.array([5.0])), [0.0])


def test_mixed_scores_at_alpha_one_equal_the_similarity_term():
    rng = np.random.default_rng(1)
    r_hat, s_hat = rng.random(50), rng.random(50)
    np.testing.assert_array_equal(p14.mixed_scores(r_hat, s_hat, 1.0), s_hat)
    np.testing.assert_array_equal(p14.mixed_scores(r_hat, s_hat, 0.25), 0.75 * r_hat + 0.25 * s_hat)


# --- D2: ranking and the cut -----------------------------------------------------------------


def test_rank_cut_breaks_ties_by_unit_id_not_by_row():
    unit_ids = ["zz", "aa", "mm", "bb"]
    rows = np.array([0, 1, 2, 3])
    scores = np.array([0.5, 0.5, 0.9, 0.5])
    assert p14.rank_cut(rows, scores, unit_ids, 10) == [(2, 0.9), (1, 0.5), (3, 0.5), (0, 0.5)]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rank_cut_equals_the_head_of_a_full_sort(seed):
    rng = random.Random(seed)  # noqa: S311 - fixture layout, not security
    n = 400
    unit_ids = [f"{rng.getrandbits(64):016x}" for _ in range(n)]
    rows = np.array(sorted(rng.sample(range(n), 250)))
    # Few distinct values, so the tie group straddling the cut is large.
    scores = np.array([rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) for _ in rows])
    full = sorted(
        zip(rows.tolist(), scores.tolist(), strict=True),
        key=lambda pair: (-pair[1], unit_ids[pair[0]]),
    )
    for depth in (1, 7, 50, 100, 249, 250, 400):
        assert p14.rank_cut(rows, scores, unit_ids, depth) == full[:depth]


# --- D4: the grid, the tie rule and the dev gate ---------------------------------------------


def test_the_grid_has_330_points_and_contains_p10c():
    points = p14.grid_points()
    assert len(points) == 330 == len(set(points))
    assert (config.PHASE_14_P10C_ALPHA, config.PHASE_14_P10C_WEIGHTS) in points
    assert {alpha for alpha, _weights in points} == set(config.PHASE_14_ALPHAS)


def test_alpha_key_names_each_alpha_distinctly():
    keys = [p14.alpha_key(alpha) for alpha in config.PHASE_14_ALPHAS]
    assert len(set(keys)) == 5
    assert p14.alpha_key(0.25) == "relevance-hop@alpha=0.25"


def _point(supported, recall, alpha, weights):
    return {"supported": supported, "gold_recall_sum": recall, "alpha": alpha, "weights": weights}


def test_the_tie_rule_applies_its_keys_in_order():
    # 1. Full Support decides first, whatever the rest.
    best = _point(4900, 1.0, 1.0, (0.0, 0.0, 1.0))
    assert p14.choose_point([_point(4899, 9e9, 0.0, (1.0, 0.0, 0.0)), best]) == best
    # 2. Then gold recall.
    best = _point(4900, 10.0, 1.0, (0.0, 0.0, 1.0))
    assert p14.choose_point([_point(4900, 9.0, 0.0, (1.0, 0.0, 0.0)), best]) == best
    # 3. Then the smaller alpha.
    best = _point(4900, 10.0, 0.25, (0.0, 0.0, 1.0))
    assert p14.choose_point([_point(4900, 10.0, 0.5, (1.0, 0.0, 0.0)), best]) == best
    # 4. Then the larger w_dense.
    best = _point(4900, 10.0, 0.5, (0.6, 0.0, 0.4))
    assert p14.choose_point([_point(4900, 10.0, 0.5, (0.5, 0.5, 0.0)), best]) == best
    # 5. Then the larger w_bm25.
    best = _point(4900, 10.0, 0.5, (0.5, 0.3, 0.2))
    assert p14.choose_point([_point(4900, 10.0, 0.5, (0.5, 0.2, 0.3)), best]) == best


def test_the_dev_gate_is_4835():
    assert config.PHASE_14_DEV_BAR == 4835
    assert p14.dev_gate({"supported": 4834}) == p14.DEV_STOP
    assert p14.dev_gate({"supported": 4835}) is None


# --- D1: the integrity check -----------------------------------------------------------------


def test_the_integrity_check_passes_at_the_tolerance_and_fails_just_past_it():
    recorded = {"q1": 0.0, "q2": 0.5}
    at = p14.integrity_verdict({"q1": 1e-5, "q2": 0.5}, recorded, 1e-5)
    assert at["passed"] and at["offending_qids"] == []
    assert at["max_abs_difference"] == pytest.approx(1e-5)
    past = p14.integrity_verdict({"q1": 1.1e-5, "q2": 0.5}, recorded, 1e-5)
    assert not past["passed"] and past["offending_qids"] == ["q1"]
    assert past["questions"] == 2 and past["tolerance"] == 1e-5


def test_the_integrity_check_refuses_different_question_sets():
    with pytest.raises(p14.Phase14Error, match="same questions"):
        p14.integrity_verdict({"q1": 0.0}, {"q2": 0.0}, 1e-5)


# --- D8: the Dense-rank split ----------------------------------------------------------------


def test_dense_rank_classes():
    assert [p14.dense_rank_class(r) for r in (1, 10, 11, 100, 101, None)] == [
        "1-10",
        "1-10",
        "11-100",
        "11-100",
        "beyond-100",
        "beyond-100",
    ]


def test_coverage_changes_list_only_gold_that_entered_one_context_but_not_the_other():
    got = p14.coverage_changes(
        gold=["g1", "g2", "g3", "g4"],
        candidate_context={"g1", "g2", "x"},
        control_context={"g2", "g3"},
        dense_ranks={"g1": 42, "g3": 4, "g4": 7},
    )
    assert got == [
        {"unit_id": "g1", "change": "gained", "dense_rank": 42, "dense_class": "11-100"},
        {"unit_id": "g3", "change": "lost", "dense_rank": 4, "dense_class": "1-10"},
    ]
    missing = p14.coverage_changes(
        gold=["g9"], candidate_context={"g9"}, control_context=set(), dense_ranks={}
    )
    assert missing == [
        {"unit_id": "g9", "change": "gained", "dense_rank": None, "dense_class": "beyond-100"}
    ]


def test_the_dense_rank_split_tallies_won_and_lost_questions_only():
    dense = [f"d{i}" for i in range(1, 101)]
    dense[11] = "g12"  # rank 12
    dense[2] = "g3"  # rank 3
    entries = [
        # won: both gold enter P14's context; one at Dense rank 12, one beyond 100
        {
            "qid": "won",
            "gold": ["g12", "gx"],
            "candidate_context": ["g12", "gx"],
            "control_context": ["d1"],
            "dense_ids": dense,
        },
        # lost: g3 leaves the context; a3 is covered by both, so it did not change
        {
            "qid": "lost",
            "gold": ["g3", "a3"],
            "candidate_context": ["a3"],
            "control_context": ["g3", "a3"],
            "dense_ids": dense,
        },
        # a tie with a coverage change still moves no question, so it is not counted
        {
            "qid": "tie",
            "gold": ["g3", "gx"],
            "candidate_context": ["g3"],
            "control_context": ["gx"],
            "dense_ids": dense,
        },
    ]
    got = p14.dense_rank_split(entries)
    assert got["won"] == {
        "questions": 1,
        "gold_changed": {"1-10": 0, "11-100": 1, "beyond-100": 1},
        "gold_changed_total": 2,
    }
    assert got["lost"] == {
        "questions": 1,
        "gold_changed": {"1-10": 1, "11-100": 0, "beyond-100": 0},
        "gold_changed_total": 1,
    }
    assert got["total"] == {
        "questions": 2,
        "gold_changed": {"1-10": 1, "11-100": 1, "beyond-100": 1},
        "gold_changed_total": 3,
    }
    assert [(q["qid"], q["outcome"]) for q in got["per_question"]] == [
        ("won", "won"),
        ("lost", "lost"),
    ]
    assert [c["dense_rank"] for c in got["per_question"][0]["changes"]] == [12, None]


# --- D6: the label ---------------------------------------------------------------------------


def test_the_label_on_constructed_counts():
    assert p14.label(60, 40, 0.049) == "CANDIDATE_RELEVANCE_SUPPORTED"
    assert p14.label(40, 60, 0.049) == "CANDIDATE_RELEVANCE_REGRESSION"
    assert p14.label(60, 40, 0.05) == "CANDIDATE_RELEVANCE_NOT_SUPPORTED"
    assert p14.label(40, 60, 0.05) == "CANDIDATE_RELEVANCE_NOT_SUPPORTED"
    assert p14.label(50, 50, 0.0) == "CANDIDATE_RELEVANCE_NOT_SUPPORTED"
    assert set(config.PHASE_14_TERMINAL_STATES) == {
        p14.SUPPORTED,
        p14.NOT_SUPPORTED,
        p14.REGRESSION,
        p14.DEV_STOP,
    }


# --- D3: the reproduction ---------------------------------------------------------------------


def test_the_reproduction_passes_only_on_the_exact_count_with_nothing_moved():
    assert p14.reproduction_verdict(4801, [], [])["passed"]
    assert p14.reproduction_verdict(4801, [], [])["point"] == {
        "alpha": 0.0,
        "weights": {"dense": 0.5, "bm25": 0.3, "relevance-hop": 0.2},
    }
    assert not p14.reproduction_verdict(4800, [], [])["passed"]
    assert not p14.reproduction_verdict(4801, ["q2", "q1"], [])["passed"]
    assert p14.reproduction_verdict(4801, ["q2", "q1"], [])["differing_qids"] == ["q1", "q2"]
    assert not p14.reproduction_verdict(4801, [], ["q3"])["passed"]
