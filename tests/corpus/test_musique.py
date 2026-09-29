"""Phase 15, S1: the MuSiQue corpus and its gold, on synthetic rows.

What can change a Phase 15 result here is the corpus and the gold: which paragraphs are
units, which units each question must retrieve, and whether a question with 3 or 4
supporting paragraphs keeps all of them through the digest, the file and the loader.
Phase 9's question code keeps exactly two gold ids, so these tests hold the n-gold path to
account. Every file is written in `tmp_path`; nothing reaches the network or `data/`.
"""

import gzip
from pathlib import Path
from typing import Any

import pytest

from concept_embeddings_rag.corpus import fullwiki as fullwiki_corpus
from concept_embeddings_rag.corpus import musique
from concept_embeddings_rag.corpus.pool import Question, unit_id_for
from concept_embeddings_rag.evaluation import fullwiki as phase9


def paragraph(idx: int, title: str, text: str, supporting: bool = False) -> dict[str, Any]:
    return {"idx": idx, "title": title, "paragraph_text": text, "is_supporting": supporting}


def row(qid: str, paragraphs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": qid,
        "question": f"question {qid}?",
        "answer": f"answer {qid}",
        "paragraphs": paragraphs,
        "answerable": True,
    }


def uid(title: str, text: str) -> str:
    return unit_id_for(title, (text,))


def pooled(rows: list[dict[str, Any]], tmp_path: Path) -> tuple[list, dict[str, Any]]:
    units = musique.pool_units(rows)
    manifest = musique.write_corpus(
        units, tmp_path, paragraphs_read=sum(len(r["paragraphs"]) for r in rows)
    )
    return units, manifest


# --- The pool --------------------------------------------------------------------------


def test_one_paragraph_attached_to_two_questions_is_one_unit():
    shared = paragraph(0, "Paris", "Paris is the capital of France.")
    rows = [
        row("q1", [shared, paragraph(1, "Lyon", "Lyon is a city.")]),
        row("q2", [{**shared, "idx": 5}, paragraph(6, "Nice", "Nice is a city.")]),
    ]

    units = musique.pool_units(rows)

    assert len(units) == 3
    assert [u.unit_id for u in units] == sorted(u.unit_id for u in units)
    paris = next(u for u in units if u.title == "Paris")
    assert paris.unit_id == uid("Paris", "Paris is the capital of France.")
    assert paris.indexable_text == "Paris. Paris is the capital of France."


def test_a_paragraph_gold_for_one_question_is_one_unit_and_gold_only_for_that_question(
    tmp_path: Path,
):
    text = "Paris is the capital of France."
    rows = [
        row("q1", [paragraph(0, "Paris", text, True), paragraph(1, "Lyon", "Lyon.", True)]),
        row("q2", [paragraph(0, "Paris", text), paragraph(1, "Nice", "Nice.", True)]),
    ]
    units, manifest = pooled(rows, tmp_path)

    questions, _body = musique.validation_questions(rows, units, corpus=manifest)

    assert len(units) == 3
    gold = {q.qid: q.gold_unit_ids for q in questions}
    assert uid("Paris", text) in gold["q1"]
    assert uid("Paris", text) not in gold["q2"]
    assert gold["q2"] == (uid("Nice", "Nice."),)


def test_two_paragraphs_with_the_same_title_and_different_text_are_two_units():
    rows = [
        row(
            "q1",
            [
                paragraph(0, "Paris", "Paris is the capital of France."),
                paragraph(1, "Paris", "Paris is a city in Texas."),
            ],
        )
    ]

    units = musique.pool_units(rows)

    assert len(units) == 2
    assert {u.text for u in units} == {
        "Paris is the capital of France.",
        "Paris is a city in Texas.",
    }


# --- The gold ---------------------------------------------------------------------------


def test_gold_keeps_three_and_four_paragraphs_through_the_file_and_the_loader(tmp_path: Path):
    three = [paragraph(i, f"T{i}", f"text {i}", True) for i in range(3)]
    four = [paragraph(i, f"F{i}", f"fact {i}", True) for i in range(4)]
    rows = [
        row("q3", [*three, paragraph(3, "X", "distractor")]),
        row("q4", [paragraph(0, "Y", "distractor"), *four]),
    ]
    units, manifest = pooled(rows, tmp_path)
    questions, body = musique.validation_questions(rows, units, corpus=manifest)
    musique.write_questions(tmp_path, body)

    loaded, _ = musique.load_questions(tmp_path, corpus_unit_set_hash=manifest["unit_set_hash"])

    expected = {
        "q3": tuple(uid(f"T{i}", f"text {i}") for i in range(3)),
        "q4": tuple(uid(f"F{i}", f"fact {i}") for i in range(4)),
    }
    assert {q.qid: q.gold_unit_ids for q in questions} == expected
    assert {q.qid: q.gold_unit_ids for q in loaded} == expected
    assert all(q.split == musique.VALIDATION_SPLIT for q in loaded)


def test_supporting_paragraphs_that_collapse_into_one_unit_are_recorded(tmp_path: Path):
    rows = [
        row(
            "q1",
            [
                paragraph(0, "Paris", "Paris is the capital.", True),
                paragraph(1, "Paris", "Paris is the capital.", True),
                paragraph(2, "Lyon", "Lyon is a city.", True),
            ],
        ),
        row("q2", [paragraph(0, "A", "a", True), paragraph(1, "B", "b", True)]),
    ]
    units, manifest = pooled(rows, tmp_path)

    questions, body = musique.validation_questions(rows, units, corpus=manifest)

    q1 = next(q for q in questions if q.qid == "q1")
    assert q1.gold_unit_ids == (
        uid("Paris", "Paris is the capital."),
        uid("Lyon", "Lyon is a city."),
    )
    assert [entry["qid"] for entry in body["collapsed_gold"]] == ["q1"]
    assert body["collapsed_gold"][0]["supporting"] == 3
    assert body["collapsed_gold"][0]["gold"] == 2


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_the_mapping_digest_changes_when_any_gold_id_changes(position: int):
    gold = ("a" * 16, "b" * 16, "c" * 16, "d" * 16)
    original = Question(
        qid="q1", question="?", answer="!", gold_unit_ids=gold, supporting_facts=(), split="s"
    )
    changed_gold = tuple("e" * 16 if i == position else g for i, g in enumerate(gold))
    changed = Question(
        qid="q1",
        question="?",
        answer="!",
        gold_unit_ids=changed_gold,
        supporting_facts=(),
        split="s",
    )

    assert musique.mapping_digest([original], "h") != musique.mapping_digest([changed], "h")


def many_rows(n: int, unmapped: int) -> list[dict[str, Any]]:
    return [
        row(
            f"q{i:03d}",
            [
                paragraph(0, f"A{i}", f"a {i}", True),
                paragraph(1, f"B{i}", "missing" if i < unmapped else f"b {i}", True),
            ],
        )
        for i in range(n)
    ]


@pytest.mark.parametrize(("n", "state"), [(100, None), (99, phase9.DATA_STOP)])
def test_one_question_with_unmapped_gold_goes_at_one_percent_and_stops_past_it(
    n: int, state: str | None, tmp_path: Path
):
    rows = many_rows(n, unmapped=1)
    missing = uid("B0", "missing")
    units = [u for u in musique.pool_units(rows) if u.unit_id != missing]
    manifest = musique.write_corpus(units, tmp_path, paragraphs_read=2 * n)

    questions, body = musique.validation_questions(rows, units, corpus=manifest)

    assert body["questions_with_unmapped_gold"] == 1
    assert body["terminal_state"] == state
    first = next(q for q in questions if q.qid == "q000")
    assert len(first.gold_unit_ids) == 2
    assert first.gold_unit_ids[1].startswith(phase9.UNRESOLVED_PREFIX)
    if state is not None:
        musique.write_questions(tmp_path, body)
        with pytest.raises(phase9.DataStop):
            musique.load_questions(tmp_path, corpus_unit_set_hash=manifest["unit_set_hash"])


# --- The corpus file --------------------------------------------------------------------


def test_the_existing_fullwiki_loader_reads_the_corpus_and_refuses_an_edited_line(
    tmp_path: Path,
):
    rows = [
        row("q1", [paragraph(0, "Paris", "Paris is the capital.", True), paragraph(1, "L", "l")]),
        row("q2", [paragraph(0, "Nice", "Nice is a city.", True), paragraph(1, "L", "l")]),
    ]
    units, manifest = pooled(rows, tmp_path)

    loaded = fullwiki_corpus.load_corpus(tmp_path)

    assert loaded == units
    assert manifest["n_units"] == 3
    assert manifest["duplicate_paragraphs_collapsed"] == 1

    corpus_file = tmp_path / manifest["corpus_file"]
    text = gzip.decompress(corpus_file.read_bytes()).decode("utf-8")
    corpus_file.write_bytes(
        gzip.compress(text.replace("Nice is a city.", "Nice is a town.").encode())
    )
    with pytest.raises(fullwiki_corpus.FullWikiError):
        fullwiki_corpus.load_corpus(tmp_path)
