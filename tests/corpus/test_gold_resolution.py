"""T5: gold resolution fails loudly or not at all.

A question whose gold paragraphs cannot be resolved must raise. Dropping it
silently would quietly shrink the evaluation set and inflate every metric
computed afterwards.
"""

import copy
import json
from pathlib import Path

import pytest

from concept_embeddings_rag.corpus.pool import GoldResolutionError, build_pool

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hotpot_sample.json"


def raw_questions() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_every_question_resolves_to_exactly_two_gold_units():
    units, questions = build_pool({"dev": raw_questions()})
    unit_ids = {unit.unit_id for unit in units}

    for question in questions:
        assert len(question.gold_unit_ids) == 2
        assert set(question.gold_unit_ids) <= unit_ids


def test_gold_ids_point_at_the_expected_titles():
    units, questions = build_pool({"dev": raw_questions()})
    title_by_id = {unit.unit_id: unit.title for unit in units}

    first = next(q for q in questions if q.qid == "q0001")
    assert {title_by_id[uid] for uid in first.gold_unit_ids} == {"Torre Vieja", "Cabo Norte"}


def test_an_unresolvable_supporting_fact_raises():
    broken = copy.deepcopy(raw_questions())
    broken[0]["supporting_facts"] = [["Nonexistent Article", 0], ["Cabo Norte", 1]]

    with pytest.raises(GoldResolutionError) as excinfo:
        build_pool({"dev": broken})

    assert "q0001" in str(excinfo.value)


def test_a_question_with_a_single_gold_paragraph_raises():
    broken = copy.deepcopy(raw_questions())
    broken[0]["supporting_facts"] = [["Torre Vieja", 0], ["Torre Vieja", 1]]

    with pytest.raises(GoldResolutionError):
        build_pool({"dev": broken})


def test_supporting_facts_are_kept_with_their_sentence_indices():
    _, questions = build_pool({"dev": raw_questions()})
    first = next(q for q in questions if q.qid == "q0001")

    assert ("Torre Vieja", 0) in first.supporting_facts
    assert ("Cabo Norte", 1) in first.supporting_facts
