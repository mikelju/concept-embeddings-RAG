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
