"""T3 of Phase 5: the pilot population, frozen by rule (HU-1) and refusing test (HU-8)."""

import json
from pathlib import Path

import numpy as np
import pytest

from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.pilot import (
    PILOT_RULE,
    PilotError,
    build_pilot,
    check_pilot_against,
    load_pilot,
    pilot_path,
    save_pilot,
)
from concept_embeddings_rag.evaluation.selection import (
    SelectionError,
    digest_of_payload,
    serialize_payload,
)

UNIT_IDS = [f"u{index}" for index in range(8)]
POOL = "pool-hash-for-tests"
MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "pinned-revision"


class FixedQueryBackend:
    """Maps each question text to a vector chosen by the test, so the dense order is known."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.vectors[text] for text in texts])


def corpus() -> np.ndarray:
    """Eight units on eight axes: the question vector alone decides the dense order."""
    return np.eye(8, dtype=np.float32)


def weights(*ranked: int) -> np.ndarray:
    """A question vector under which `ranked` come first, in that order."""
    vector = np.zeros(8, dtype=np.float32)
    for position, row in enumerate(ranked):
        vector[row] = 10.0 - position
    return vector


def a_question(qid: str, gold: tuple[int, ...], split: str = "dev") -> Question:
    return Question(
        qid=qid,
        question=f"question {qid}",
        answer="-",
        gold_unit_ids=tuple(UNIT_IDS[row] for row in gold),
        supporting_facts=tuple((f"T{row}", 0) for row in gold),
        split=split,
    )


def a_pilot(questions: list[Question], backend: FixedQueryBackend, read_depth: int = 2):
    return build_pilot(
        questions,
        unit_ids=UNIT_IDS,
        vectors=corpus(),
        query_backend=backend,
        model=MODEL,
        revision=REVISION,
        unit_set_hash=POOL,
        read_depth=read_depth,
    )


def three_questions() -> tuple[list[Question], FixedQueryBackend]:
    questions = [
        a_question("q2", gold=(5, 6)),  # both gold outside the read top-2
        a_question("q0", gold=(0, 1)),  # both gold read: not in the pilot
        a_question("q1", gold=(2, 7)),  # one read, one missing
    ]
    backend = FixedQueryBackend(
        {
            "question q0": weights(0, 1, 2),
            "question q1": weights(2, 3, 7),
            "question q2": weights(4, 3, 5),
        }
    )
    return questions, backend


def test_the_rule_selects_exactly_the_questions_with_a_gold_outside_the_read_depth():
    questions, backend = three_questions()

    pilot = a_pilot(questions, backend)

    assert [q.qid for q in pilot.questions] == ["q1", "q2"]
    by_id = {q.qid: q for q in pilot.questions}
    assert by_id["q1"].read == ("u2", "u3")
    assert by_id["q1"].p1 == "u2"
    assert by_id["q1"].missing == ("u7",)
    assert by_id["q2"].missing == ("u5", "u6")
    assert pilot.n_missing == 3
    assert pilot.rule == PILOT_RULE
    assert pilot.split == "dev"


def test_building_the_pilot_twice_writes_byte_identical_artifacts(tmp_path: Path):
    questions, backend = three_questions()

    first = save_pilot(a_pilot(questions, backend), tmp_path / "a")
    second = save_pilot(a_pilot(list(reversed(questions)), backend), tmp_path / "b")

    assert first.read_bytes() == second.read_bytes()


def test_a_saved_pilot_loads_back_identical(tmp_path: Path):
    questions, backend = three_questions()
    pilot = a_pilot(questions, backend)

    save_pilot(pilot, tmp_path)

    assert load_pilot(tmp_path, expected_unit_set_hash=POOL) == pilot


def test_a_second_freeze_refuses_rather_than_overwriting(tmp_path: Path):
    questions, backend = three_questions()
    save_pilot(a_pilot(questions, backend), tmp_path)

    with pytest.raises(PilotError, match="already frozen"):
        save_pilot(a_pilot(questions, backend), tmp_path)


def test_a_test_split_question_is_refused_at_the_builder():
    questions, backend = three_questions()
    questions.append(a_question("q9", gold=(6,), split="test"))
    backend.vectors["question q9"] = weights(0, 1)

    with pytest.raises(SelectionError):
        a_pilot(questions, backend)


def _rewrite(tmp_path: Path, mutate) -> None:
    path = pilot_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    payload["digest"] = digest_of_payload(payload)
    path.write_text(serialize_payload(payload), encoding="utf-8")


def test_a_modified_artifact_is_refused(tmp_path: Path):
    questions, backend = three_questions()
    save_pilot(a_pilot(questions, backend), tmp_path)
    path = pilot_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["questions"][0]["missing"] = ["u4"]
    path.write_text(serialize_payload(payload), encoding="utf-8")

    with pytest.raises(PilotError, match="modified"):
        load_pilot(tmp_path)


def test_a_pilot_over_another_pool_is_refused(tmp_path: Path):
    questions, backend = three_questions()
    save_pilot(a_pilot(questions, backend), tmp_path)

    with pytest.raises(PilotError, match="pool"):
        load_pilot(tmp_path, expected_unit_set_hash="another-pool")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["questions"][0].update(p1="u3"), "p1"),
        (lambda p: p["questions"][0].update(missing=[]), "missing"),
        (lambda p: p["questions"][0].update(missing=["u2"]), "read"),
        (lambda p: p.update(split="test"), "dev"),
        (lambda p: p["questions"].reverse(), "sorted"),
    ],
)
def test_invariants_are_enforced_on_load_even_under_a_valid_digest(tmp_path: Path, mutate, message):
    questions, backend = three_questions()
    save_pilot(a_pilot(questions, backend), tmp_path)
    _rewrite(tmp_path, mutate)

    with pytest.raises(PilotError, match=message):
        load_pilot(tmp_path)


def test_every_missing_paragraph_must_be_a_gold_of_its_own_question():
    questions, backend = three_questions()
    pilot = a_pilot(questions, backend)
    check_pilot_against(pilot, questions)

    impostor = [a_question("q1", gold=(2, 6)) if q.qid == "q1" else q for q in questions]
    with pytest.raises(PilotError, match="gold"):
        check_pilot_against(pilot, impostor)
