"""The unified pool of indexing units, and the questions that point into it.

An indexing unit is one benchmark paragraph: a complete unit of meaning, never a
fixed-length window and never split. Its id is derived from its content, so the
same paragraph always gets the same id, deduplication comes for free, and a pool
rebuilt from the manifest is identical to the original.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class GoldResolutionError(Exception):
    """A question's gold paragraphs could not be resolved against the pool."""


class PoolIntegrityError(Exception):
    """A stored unit id does not match the content stored beside it."""


@dataclass(frozen=True)
class IndexingUnit:
    unit_id: str
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def indexable_text(self) -> str:
        """What every retriever indexes: title plus paragraph.

        The title carries the entity a multi-hop question usually needs to bridge
        on, and both retrievers must see the same text or the comparison measures
        the input rather than the method.
        """
        return f"{self.title}. {self.text}"


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    answer: str
    gold_unit_ids: tuple[str, ...]
    supporting_facts: tuple[tuple[str, int], ...]
    split: str


def unit_id_for(title: str, sentences: tuple[str, ...]) -> str:
    """Content-derived identifier: same paragraph, same id, always."""
    payload = title + "\n" + " ".join(sentences)
    # Content addressing, not integrity: collisions are a correctness concern here,
    # not a security one, and the flag says so to hashlib and to the SAST gate alike.
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def build_pool(
    questions_by_split: dict[str, list[dict]],
) -> tuple[list[IndexingUnit], list[Question]]:
    """Build the unified pool and resolve every question's gold paragraphs.

    All splits contribute to a single pool: dev and test must search the same
    space, otherwise one of them enjoys an easier problem and the comparison
    between them means nothing.
    """
    units_by_id: dict[str, IndexingUnit] = {}
    questions: list[Question] = []

    for split, raw_questions in questions_by_split.items():
        for raw in raw_questions:
            qid = raw["_id"]
            id_by_title: dict[str, str] = {}

            for title, sentences in raw["context"]:
                sentence_tuple = tuple(sentences)
                uid = unit_id_for(title, sentence_tuple)
                units_by_id.setdefault(
                    uid, IndexingUnit(unit_id=uid, title=title, sentences=sentence_tuple)
                )
                id_by_title[title] = uid

            supporting_facts = tuple(
                (title, int(index)) for title, index in raw["supporting_facts"]
            )
            gold_ids = _resolve_gold(qid, supporting_facts, id_by_title)

            questions.append(
                Question(
                    qid=qid,
                    question=raw["question"],
                    answer=raw["answer"],
                    gold_unit_ids=gold_ids,
                    supporting_facts=supporting_facts,
                    split=split,
                )
            )

    units = sorted(units_by_id.values(), key=lambda unit: unit.unit_id)
    return units, questions


def _resolve_gold(
    qid: str,
    supporting_facts: tuple[tuple[str, int], ...],
    id_by_title: dict[str, str],
) -> tuple[str, ...]:
    """Map supporting facts onto unit ids, raising rather than dropping anything."""
    resolved: list[str] = []
    for title, _sentence_index in supporting_facts:
        if title not in id_by_title:
            raise GoldResolutionError(
                f"question {qid}: supporting fact refers to '{title}', "
                "which is not among its context paragraphs"
            )
        uid = id_by_title[title]
        if uid not in resolved:
            resolved.append(uid)

    if len(resolved) != 2:
        raise GoldResolutionError(
            f"question {qid}: expected exactly 2 gold paragraphs, resolved {len(resolved)}"
        )
    return tuple(resolved)


def save_pool(units: list[IndexingUnit], questions: list[Question], path: Path | str) -> None:
    """Persist the pool as JSON, so later stages never re-derive it."""
    payload = {
        "units": [
            {"unit_id": u.unit_id, "title": u.title, "sentences": list(u.sentences)} for u in units
        ],
        "questions": [
            {
                "qid": q.qid,
                "question": q.question,
                "answer": q.answer,
                "gold_unit_ids": list(q.gold_unit_ids),
                "supporting_facts": [[t, i] for t, i in q.supporting_facts],
                "split": q.split,
            }
            for q in questions
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_pool(path: Path | str) -> tuple[list[IndexingUnit], list[Question]]:
    """Load the pool, re-deriving every id so a modified file cannot pass as intact.

    Unit ids are a pure function of content, so checking them costs one hash each and
    catches exactly what would otherwise be invisible: a paragraph edited under an id
    that every later stage keeps trusting. Gold ids are scored against these, so a
    silent mismatch here moves every metric in the project.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    units: list[IndexingUnit] = []
    for entry in payload["units"]:
        sentences = tuple(entry["sentences"])
        expected = unit_id_for(entry["title"], sentences)
        if expected != entry["unit_id"]:
            raise PoolIntegrityError(
                f"unit {entry['unit_id']} does not hash to its content (got {expected}); "
                f"{path} has been modified - rebuild it with 'build'"
            )
        units.append(
            IndexingUnit(unit_id=entry["unit_id"], title=entry["title"], sentences=sentences)
        )
    questions = [
        Question(
            qid=q["qid"],
            question=q["question"],
            answer=q["answer"],
            gold_unit_ids=tuple(q["gold_unit_ids"]),
            supporting_facts=tuple((t, int(i)) for t, i in q["supporting_facts"]),
            split=q["split"],
        )
        for q in payload["questions"]
    ]

    # build_pool guarantees this when it constructs the pool; loading has to prove it
    # again, because gold ids are what every metric is scored against. A gold pointing
    # at nothing is not a retrieval failure, but it would be counted as one.
    known = {unit.unit_id for unit in units}
    for question in questions:
        missing = [uid for uid in question.gold_unit_ids if uid not in known]
        if missing:
            raise GoldResolutionError(
                f"question {question.qid}: gold {missing} is not in the pool; "
                f"{path} is inconsistent - rebuild it with 'build'"
            )
    return units, questions
