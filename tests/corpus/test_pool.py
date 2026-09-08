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
