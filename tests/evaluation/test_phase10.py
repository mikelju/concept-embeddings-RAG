"""Phase 10: the held-out draw, its gold mapping, the probe, the fit and the outcome rule.

What can change a Phase 10 figure is tested here: which train questions land in which set,
that the sets never overlap each other or the validation questions, what an unresolved title
does, the contamination probe verdict, the dev grid and its gate, and the terminal label.
"""

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import IndexingUnit, unit_id_for
from concept_embeddings_rag.evaluation import phase10 as p10


def a_unit(title: str, text: str = "Some text.") -> IndexingUnit:
    sentences = (text,)
    return IndexingUnit(unit_id=unit_id_for(title, sentences), title=title, sentences=sentences)


def a_raw(qid: str, titles: tuple[str, str], level: str = "hard") -> dict:
    return {
        "_id": qid,
        "question": f"question {qid}?",
        "answer": "a",
        "type": "bridge",
        "level": level,
        "supporting_facts": [[titles[0], 0], [titles[1], 0]],
    }


SIZES = {"probe": 2, "test-10": 3, "test-11": 3}


def test_the_draw_is_deterministic_disjoint_and_sized():
    hard = [f"q{i:03d}" for i in range(40)]

    first = p10.draw_sets(hard, validation_qids=[], seed=10, sizes=SIZES)
    second = p10.draw_sets(list(reversed(hard)), validation_qids=[], seed=10, sizes=SIZES)

    assert first == second  # the source order does not move the draw
    assert list(first) == ["probe", "test-10", "test-11"]
    assert [len(qids) for qids in first.values()] == [2, 3, 3]
    drawn = [qid for qids in first.values() for qid in qids]
    assert len(set(drawn)) == len(drawn)
    assert p10.draw_sets(hard, validation_qids=[], seed=11, sizes=SIZES) != first


def test_a_train_qid_that_is_also_a_validation_qid_stops():
    with pytest.raises(p10.DataStop, match="validation"):
        p10.draw_sets(
            ["a", "b", "c", "d", "e", "f", "g", "h"], validation_qids=["c"], seed=10, sizes=SIZES
        )


def test_too_few_hard_questions_stops():
    with pytest.raises(p10.DataStop, match="hard questions"):
        p10.draw_sets(["a", "b", "c"], validation_qids=[], seed=10, sizes=SIZES)


def test_only_hard_questions_are_drawn_and_the_validation_level_is_checked():
    raws = [a_raw("h1", ("A", "B")), a_raw("m1", ("A", "B"), level="medium")]
    assert p10.hard_qids(raws) == ["h1"]
    with pytest.raises(p10.DataStop, match="not all 'hard'"):
        p10.check_validation_level([a_raw("v1", ("A", "B"), level="medium")])


def test_sets_map_gold_under_the_phase_9_rule_and_record_the_ceiling():
    units = [a_unit("A"), a_unit("B"), a_unit("C")]
    raws = [a_raw("x1", ("A", "B")), a_raw("x2", ("A", "Missing")), a_raw("x3", ("B", "C"))]
    sets = {"probe": ["x1"], "test-10": ["x2"], "test-11": ["x3"]}
    corpus = {"unit_set_hash": "h", "ordered_unit_digest": "o"}

    questions, body = p10.build_sets(raws, sets, units, corpus=corpus, share=0.01)

    assert [q.split for q in questions["test-10"]] == ["test-10"]
    assert questions["probe"][0].gold_unit_ids == (units[0].unit_id, units[1].unit_id)
    missing = questions["test-10"][0].gold_unit_ids[1]
    assert missing.startswith("unresolved:")  # stays in the denominator, never retrievable
    assert body["sets"]["test-10"]["questions_with_unresolved_gold"] == 1
    # One of one question is 100 % > 1 %: the stop is written, not raised.
    assert body["terminal_state"] == p10.DATA_STOP
    assert body["sets"]["probe"]["questions_with_unresolved_gold"] == 0


def test_set_digests_differ_per_set_and_are_stable():
    units = [a_unit("A"), a_unit("B")]
    raws = [a_raw("x1", ("A", "B")), a_raw("x2", ("A", "B")), a_raw("x3", ("B", "A"))]
    sets = {"probe": ["x1"], "test-10": ["x2"], "test-11": ["x3"]}
    corpus = {"unit_set_hash": "h", "ordered_unit_digest": "o"}

    _q, one = p10.build_sets(raws, sets, units, corpus=corpus, share=0.01)
    _q, two = p10.build_sets(raws, sets, units, corpus=corpus, share=0.01)

    digests = [one["sets"][name]["question_digest"] for name in sets]
    assert len(set(digests)) == 3
    assert one == two
    assert one["terminal_state"] is None


def test_the_configured_sizes_are_the_spec_sizes():
    assert config.PHASE_10_SET_SIZES == {"probe": 1000, "test-10": 5000, "test-11": 5000}
    assert list(config.PHASE_10_SET_SIZES) == ["probe", "test-10", "test-11"]


# --- D3 probe -------------------------------------------------------------------------------


def test_the_probe_stops_only_strictly_past_the_margin():
    # dev 50.0 %; probe 55.0 % is exactly +5.0 points: not past the margin.
    at = p10.probe_verdict(550, 1000, 5000, 10000)
    assert at["difference_pp"] == pytest.approx(5.0)
    assert at["terminal_state"] is None
    past = p10.probe_verdict(551, 1000, 5000, 10000)
    assert past["terminal_state"] == p10.DATA_STOP
    below = p10.probe_verdict(400, 1000, 5000, 10000)
    assert below["terminal_state"] is None  # a harder probe is not contamination


# --- D5 grid, tie rule and gate ----------------------------------------------------------------


def test_the_grid_has_66_convex_points_on_tenths():
    grid = p10.weight_grid()
    assert len(grid) == 66
    assert all(abs(sum(point) - 1.0) < 1e-9 and min(point) >= 0.0 for point in grid)
    assert (0.5, 0.5, 0.0) in grid and (1.0, 0.0, 0.0) in grid and (0.0, 0.0, 1.0) in grid
    assert len(set(grid)) == 66


def point(weights, supported, recall):
    return {"weights": weights, "supported": supported, "gold_recall_sum": recall}


def test_the_tie_rule_orders_support_recall_dense_then_bm25():
    curve = [
        point((0.4, 0.4, 0.2), 10, 5.0),
        point((0.5, 0.3, 0.2), 10, 5.0),  # same support and recall, larger dense
        point((0.5, 0.4, 0.1), 10, 5.0),  # same, same dense, larger bm25: wins
        point((0.3, 0.3, 0.4), 10, 4.0),
    ]
    assert p10.choose_point(curve)["weights"] == (0.5, 0.4, 0.1)
    curve.append(point((0.1, 0.1, 0.8), 10, 6.0))  # more recall beats any weight
    assert p10.choose_point(curve)["weights"] == (0.1, 0.1, 0.8)
    curve.append(point((0.6, 0.4, 0.0), 11, 0.0))  # more support beats everything
    assert p10.choose_point(curve)["weights"] == (0.6, 0.4, 0.0)


def test_the_dev_gate_has_two_stopping_branches():
    assert p10.dev_gate(point((0.5, 0.5, 0.0), 5000, 0.0), 4536) == p10.DEV_STOP
    assert p10.dev_gate(point((0.4, 0.4, 0.2), 4536, 0.0), 4536) == p10.DEV_STOP  # a tie
    assert p10.dev_gate(point((0.4, 0.4, 0.2), 4537, 0.0), 4536) is None


# --- D6 label -------------------------------------------------------------------------------


def records(pattern):
    return [
        {"qid": f"q{i}", "budgets": {"2048": {"full_support": value}}}
        for i, value in enumerate(pattern)
    ]


def test_the_label_follows_wins_losses_and_p(monkeypatch):
    monkeypatch.setitem(config.PHASE_10_SET_SIZES, "test-10", 40)
    control = records([0.0] * 30 + [1.0] * 10)
    better = records([1.0] * 30 + [1.0] * 10)  # 30 wins, 0 losses
    assert p10.three_way_outcome(control, better)["terminal_state"] == "THREE_WAY_SUPPORTED"
    worse = records([0.0] * 30 + [0.0] * 10)  # 0 wins, 10 losses
    assert p10.three_way_outcome(control, worse)["terminal_state"] == "THREE_WAY_REGRESSION"
    close = records([1.0] + [0.0] * 29 + [1.0] * 10)  # 1 win, 0 losses: p = 1
    outcome = p10.three_way_outcome(control, close)
    assert outcome["terminal_state"] == "THREE_WAY_NOT_SUPPORTED"
    assert (outcome["wins"], outcome["losses"], outcome["ties"]) == (1, 0, 39)


def test_the_label_refuses_runs_that_do_not_cover_test_10():
    with pytest.raises(p10.Phase10Error, match="test-10"):
        p10.three_way_outcome(records([1.0, 0.0]), records([1.0, 0.0]))


# --- Replay and dev lists --------------------------------------------------------------------


def test_replay_serves_in_order_and_refuses_another_query():
    replay = p10.ReplayRetriever("x", [("a?", [("u1", 1.0), ("u2", 0.5)]), ("b?", [])])
    assert replay.retrieve("a?", 1) == [("u1", 1.0)]
    with pytest.raises(p10.Phase10Error, match="out of order"):
        replay.retrieve("c?", 5)


def test_dev_lists_round_trip_and_refuse_a_modified_file(tmp_path):
    rows = [{"qid": "q1", "dense": [["u1", 0.5]], "bm25": [["u2", 3.0]], "entity-hop": []}]
    p10.write_dev_lists(tmp_path, rows, provenance={"note": "t"})
    loaded = p10.load_dev_lists(tmp_path)
    assert loaded[0]["dense"] == [("u1", 0.5)]
    with pytest.raises(p10.Phase10Error, match="already"):
        p10.write_dev_lists(tmp_path, rows, provenance={})
    (tmp_path / p10.DEV_LISTS_MANIFEST).write_text('{"digest": "x"}', encoding="utf-8")
    with pytest.raises(p10.Phase10Error, match="digest"):
        p10.load_dev_lists(tmp_path)


def test_the_reproduction_verdict_is_exact():
    ok = p10.reproduction_verdict({"dense": 4125, "hybrid-bm25": 4536, "hybrid-entity-hop": 4441})
    assert ok["passed"]
    off = p10.reproduction_verdict({"dense": 4125, "hybrid-bm25": 4535, "hybrid-entity-hop": 4441})
    assert not off["passed"]
