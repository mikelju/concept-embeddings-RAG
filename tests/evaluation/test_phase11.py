"""Phase 11: question-entity matching, the D3 verdict, the grid, the tie rule, the gate, the label.

What can change a Phase 11 figure is tested here: which nodes a question seeds, the grid the
fit searches, which point D2's order selects, where the D4 gate falls, and the D5 label.
"""

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import phase10 as p10
from concept_embeddings_rag.evaluation import phase11 as p11


def test_spans_are_normalized_like_the_index_and_unmatched_forms_are_kept():
    forms = {"first for women": 3, "arthur's magazine": 1}
    matched = p11.match_spans(["First for Women", "The Arthur's Magazine", "Nowhere", "  "], forms)
    assert matched["node_ids"] == [1, 3]
    assert matched["unmatched"] == ["nowhere"]
    assert matched["spans"] == ["First for Women", "The Arthur's Magazine", "Nowhere", "  "]
    assert p11.match_spans([], forms)["node_ids"] == []


def test_question_entities_are_written_once_and_checked_on_load(tmp_path):
    rows = [{"qid": "a", "spans": ["X"], "node_ids": [1], "unmatched": [], "seconds": 0.1}]
    p11.write_question_entities(tmp_path, "dev", rows, extractor={"digest": "d"})
    assert p11.load_question_entities(tmp_path, "dev")["questions"] == rows
    with pytest.raises(p11.Phase11Error, match="already"):
        p11.write_question_entities(tmp_path, "dev", rows, extractor={"digest": "d"})
    with pytest.raises(p11.Phase11Error, match="unknown"):
        p11.write_question_entities(tmp_path, "test-10", rows, extractor={})
    path = tmp_path / p11.entities_filename("dev")
    path.write_text(
        path.read_text().replace('"node_ids": [\n        1', '"node_ids": [\n        2')
    )
    with pytest.raises(p11.Phase11Error, match="digest"):
        p11.load_question_entities(tmp_path, "dev")


def test_phase_10_still_refuses_test_11(tmp_path):
    with pytest.raises(p10.Phase10Error, match="reserved"):
        p10.load_set(tmp_path, "test-11", corpus_unit_set_hash="x")


def test_the_reproduction_passes_only_on_the_exact_count_with_nothing_moved():
    assert p11.reproduction_verdict(4801, [], [])["passed"]
    assert not p11.reproduction_verdict(4800, [], [])["passed"]
    assert not p11.reproduction_verdict(4801, ["q1", "q2"], [])["passed"]  # two moves cancel
    assert not p11.reproduction_verdict(4801, [], ["q3"])["passed"]


def test_the_grid_has_286_quadruples_1430_points_and_contains_p10c():
    grid = p11.weight_grid()
    assert len(grid) == 286 and len(set(grid)) == 286
    assert all(abs(sum(w) - 1.0) < 1e-9 and min(w) >= 0.0 for w in grid)
    points = p11.grid_points()
    assert len(points) == 1430
    p10c = tuple(config.PHASE_11_P10C_WEIGHTS[name] for name in p11.QUAD_NAMES)
    assert (None, p10c) in points
    assert p10c == (0.5, 0.3, 0.2, 0.0)


def a_point(supported, recall=0.0, cap=None, weights=(0.5, 0.3, 0.2, 0.0)):
    return {"supported": supported, "gold_recall_sum": recall, "p1_df_cap": cap, "weights": weights}


def test_the_tie_rule_applies_in_the_spec_order():
    assert p11.choose_point([a_point(10), a_point(11)])["supported"] == 11
    assert p11.choose_point([a_point(10, 1.0), a_point(10, 2.0)])["gold_recall_sum"] == 2.0
    # No question weight wins a tie on the first two keys.
    with_q = a_point(10, 1.0, None, (0.5, 0.3, 0.1, 0.1))
    without_q = a_point(10, 1.0, 1000, (0.5, 0.3, 0.2, 0.0))
    assert p11.choose_point([with_q, without_q]) == without_q
    # Then the larger cap, uncapped largest.
    capped, uncapped = a_point(10, 1.0, 30_000), a_point(10, 1.0, None)
    assert p11.choose_point([capped, uncapped]) == uncapped
    assert p11.choose_point([a_point(10, 1.0, 1000), capped]) == capped
    # Then the larger dense, bm25 and p1 weights.
    a = a_point(10, 1.0, None, (0.6, 0.2, 0.2, 0.0))
    b = a_point(10, 1.0, None, (0.5, 0.4, 0.1, 0.0))
    assert p11.choose_point([b, a]) == a
    c = a_point(10, 1.0, None, (0.5, 0.2, 0.3, 0.0))
    assert p11.choose_point([c, b]) == b


def test_the_dev_gate_falls_at_4838():
    assert p11.dev_gate(a_point(4838)) is None
    assert p11.dev_gate(a_point(4837)) == "DEV_STOP"


def outcomes(values):
    return [
        {"qid": f"q{i}", "budgets": {"2048": {"full_support": v}}} for i, v in enumerate(values)
    ]


def test_the_label_follows_the_primary_comparison(monkeypatch):
    monkeypatch.setitem(config.PHASE_10_SET_SIZES, "test-11", 100)
    control = outcomes([0.0] * 50 + [1.0] * 50)
    better = outcomes([1.0] * 30 + [0.0] * 20 + [1.0] * 50)
    primary = p11.paired(control, better)
    assert (primary["wins"], primary["losses"], primary["ties"]) == (30, 0, 70)
    assert primary["delta_percentage_points"] == 30.0
    assert p11.label(primary) == p11.SUPPORTED
    assert p11.label(p11.paired(better, control)) == p11.REGRESSION
    close = outcomes([1.0] * 2 + [0.0] * 48 + [1.0] * 49 + [0.0])
    assert p11.label(p11.paired(control, close)) == p11.NOT_SUPPORTED
    with pytest.raises(p11.Phase11Error, match="same test-11"):
        p11.paired(control, outcomes([1.0] * 99))
