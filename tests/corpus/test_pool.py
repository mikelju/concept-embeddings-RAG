"""T4: the unified pool of indexing units.

A unit is one benchmark paragraph and is never split, merged or truncated. Its id
comes from its content, so the same paragraph appearing in ten questions collapses
into one unit and a rebuilt corpus yields identical ids.
"""

import json
from pathlib import Path

from concept_embeddings_rag.corpus.pool import build_pool, unit_id_for

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hotpot_sample.json"


def raw_questions() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_unit_id_depends_only_on_content():
    first = unit_id_for("Torre Vieja", ("a sentence.", "another one."))
    second = unit_id_for("Torre Vieja", ("a sentence.", "another one."))
    different = unit_id_for("Torre Vieja", ("a sentence.",))

    assert first == second
    assert first != different
    assert len(first) == 16


def test_duplicate_paragraphs_collapse_into_one_unit():
    units, _ = build_pool({"dev": raw_questions()})
    titles = [unit.title for unit in units]

    # "Torre Vieja", "Cabo Norte", "Coastal navigation" and "Stone masonry" each
    # appear in two questions of the fixture.
    assert len(titles) == len(set(titles))
    assert "Torre Vieja" in titles


def test_units_preserve_sentences_verbatim_and_are_never_truncated():
    units, _ = build_pool({"dev": raw_questions()})
    by_title = {unit.title: unit for unit in units}

    torre = by_title["Torre Vieja"]
    assert torre.sentences == ("Torre Vieja is a lighthouse.", "It was built in 1852.")
    assert torre.text == "Torre Vieja is a lighthouse. It was built in 1852."


def test_every_question_keeps_its_split_and_metadata():
    units, questions = build_pool({"dev": raw_questions()[:2], "test": raw_questions()[2:]})

    splits = {q.qid: q.split for q in questions}
    assert splits["q0001"] == "dev"
    assert splits["q0003"] == "test"
    assert len(questions) == 4
    assert units


def test_pool_is_shared_across_splits():
    """Dev and test must search the same space, or the comparison is rigged."""
    units_together, _ = build_pool({"dev": raw_questions()[:2], "test": raw_questions()[2:]})
    units_all_dev, _ = build_pool({"dev": raw_questions()})

    assert {u.unit_id for u in units_together} == {u.unit_id for u in units_all_dev}


def test_ids_are_stable_across_builds():
    first, _ = build_pool({"dev": raw_questions()})
    second, _ = build_pool({"dev": raw_questions()})
    assert [u.unit_id for u in first] == [u.unit_id for u in second]


def test_loading_a_pool_whose_content_was_edited_under_its_id_is_refused(tmp_path):
    """SEC-003: unit ids are a pure function of content, so re-deriving them costs
    one hash each and catches the edit that would otherwise move every metric."""
    import pytest

    from concept_embeddings_rag.corpus.pool import PoolIntegrityError, load_pool, save_pool

    units, questions = build_pool({"dev": raw_questions()})
    path = tmp_path / "pool.json"
    save_pool(units, questions, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["units"][0]["sentences"] = ["Something else entirely."]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PoolIntegrityError, match="does not hash to its content"):
        load_pool(path)


def test_an_untouched_pool_round_trips(tmp_path):
    """The integrity check must not reject what the pipeline itself produced."""
    from concept_embeddings_rag.corpus.pool import load_pool, save_pool

    units, questions = build_pool({"dev": raw_questions()})
    path = tmp_path / "pool.json"
    save_pool(units, questions, path)

    loaded_units, loaded_questions = load_pool(path)
    assert [u.unit_id for u in loaded_units] == [u.unit_id for u in units]
    assert [q.qid for q in loaded_questions] == [q.qid for q in questions]


def test_loading_a_pool_whose_gold_points_at_nothing_is_refused(tmp_path):
    """SEC-010: a gold id resolving to no unit is not a retrieval failure, but every
    metric would score it as one. build_pool guarantees this; loading must too."""
    import pytest

    from concept_embeddings_rag.corpus.pool import GoldResolutionError, load_pool, save_pool

    units, questions = build_pool({"dev": raw_questions()})
    path = tmp_path / "pool.json"
    save_pool(units, questions, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["questions"][0]["gold_unit_ids"] = ["0" * 16]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GoldResolutionError, match="not in the pool"):
        load_pool(path)
