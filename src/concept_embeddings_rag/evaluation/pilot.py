"""The pilot population of Phase 5: frozen by a rule, before any text-derived hop runs.

HU-1 of the Phase 5 spec. A question is in the pilot when dense, asked the question, leaves
at least one of its gold paragraphs outside the top-10 a reader has read. That is the whole
rule: no sample, no manual choice, and the same population the Phase 4 diagnostic measured,
so the two sets of figures compare directly.

The artifact records, per question, what every hop starts from - `p1` and the read top-10 -
and what a hop has to find: the missing gold paragraphs. It is written once, with a digest,
through the serializer the Phase 3 and Phase 4 freezes use, and it is refused on load if the
digest, the pool or any invariant disagrees.

`p1` and the read list are computed from the question and the embeddings alone. The gold is
read only afterwards, to decide which questions qualify and what they miss.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import second_hop
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    check_dev_only,
    digest_of_payload,
    serialize_payload,
)

PILOT_FILENAME = "pilot.json"
PILOT_RULE = (
    "every dev question for which at least one gold paragraph is outside dense's top-10 on the "
    "question; no sampling, no manual inclusion or exclusion"
)


class PilotError(Exception):
    """The pilot artifact is missing, modified, or not the one this run may use."""


class QueryEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class PilotQuestion:
    qid: str
    p1: str
    read: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class PilotSet:
    questions: tuple[PilotQuestion, ...]
    unit_set_hash: str
    model: str
    revision: str
    read_depth: int = config.PILOT_READ_DEPTH
    rule: str = PILOT_RULE
    split: str = DEV_SPLIT

    @property
    def n_missing(self) -> int:
        return sum(len(question.missing) for question in self.questions)


def build_pilot(
    questions: Sequence[Question],
    *,
    unit_ids: Sequence[str],
    vectors: np.ndarray,
    query_backend: QueryEncoder,
    model: str,
    revision: str,
    unit_set_hash: str,
    read_depth: int = config.PILOT_READ_DEPTH,
) -> PilotSet:
    """Apply the HU-1 rule to `questions`, which must all be dev questions."""
    check_dev_only(questions)
    if len(unit_ids) != vectors.shape[0]:
        raise PilotError(f"{vectors.shape[0]} embedding rows for {len(unit_ids)} unit ids")
    index = {unit_id: row for row, unit_id in enumerate(unit_ids)}

    selected: list[PilotQuestion] = []
    for question in sorted(questions, key=lambda q: q.qid):
        order = second_hop.dense_order(vectors, query_backend.encode([question.question])[0])
        read = second_hop.read_rows(order, read_depth)
        read_set = set(read)
        # The gold enters here and only here: after `p1` and the read list exist.
        for gold in question.gold_unit_ids:
            if gold not in index:
                raise PilotError(f"question {question.qid} names gold unit {gold}, not in the pool")
        missing = tuple(gold for gold in question.gold_unit_ids if index[gold] not in read_set)
        if not missing:
            continue
        selected.append(
            PilotQuestion(
                qid=question.qid,
                p1=unit_ids[second_hop.origin(read)],
                read=tuple(unit_ids[row] for row in read),
                missing=missing,
            )
        )

    return PilotSet(
        questions=tuple(selected),
        unit_set_hash=unit_set_hash,
        model=model,
        revision=revision,
        read_depth=read_depth,
    )


def pilot_path(directory: Path | str) -> Path:
    return Path(directory) / PILOT_FILENAME


def _payload(pilot: PilotSet) -> dict[str, Any]:
    return {
        "rule": pilot.rule,
        "split": pilot.split,
        "read_depth": pilot.read_depth,
        "unit_set_hash": pilot.unit_set_hash,
        "model": pilot.model,
        "revision": pilot.revision,
        "n_questions": len(pilot.questions),
        "n_missing": pilot.n_missing,
        "questions": [
            {
                "qid": question.qid,
                "p1": question.p1,
                "read": list(question.read),
                "missing": list(question.missing),
            }
            for question in pilot.questions
        ],
    }


def pilot_digest(pilot: PilotSet) -> str:
    """The digest the frozen artifact of this pilot carries, computed from the pilot itself."""
    return digest_of_payload(_payload(pilot))


def save_pilot(pilot: PilotSet, directory: Path | str) -> Path:
    """Write the pilot once, atomically, and never over an existing one."""
    _check_invariants(pilot)
    path = pilot_path(directory)
    if path.exists():
        raise PilotError(
            f"a pilot is already frozen at {path}; this phase writes one and never overwrites "
            "it. A second freeze is a deviation to be recorded, so move the existing artifact "
            "aside deliberately if that is what is meant"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(pilot)
    payload["digest"] = digest_of_payload(payload)
    write_text_atomic(path, serialize_payload(payload))
    return path


def load_pilot(directory: Path | str, *, expected_unit_set_hash: str | None = None) -> PilotSet:
    """Read the pilot back, proving it is the frozen one and that it still holds together."""
    path = pilot_path(directory)
    if not path.exists():
        raise PilotError(f"no pilot artifact in {directory}; run `cer pilot` to freeze one")

    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("digest")
    if recorded is None or digest_of_payload(payload) != recorded:
        raise PilotError(
            f"the pilot at {path.name} does not match its digest; it has been modified"
        )

    pilot = PilotSet(
        questions=tuple(
            PilotQuestion(
                qid=str(entry["qid"]),
                p1=str(entry["p1"]),
                read=tuple(str(unit) for unit in entry["read"]),
                missing=tuple(str(unit) for unit in entry["missing"]),
            )
            for entry in payload["questions"]
        ),
        unit_set_hash=str(payload["unit_set_hash"]),
        model=str(payload["model"]),
        revision=str(payload["revision"]),
        read_depth=int(payload["read_depth"]),
        rule=str(payload["rule"]),
        split=str(payload["split"]),
    )
    if expected_unit_set_hash is not None and pilot.unit_set_hash != expected_unit_set_hash:
        raise PilotError(
            f"the pilot was frozen over pool {pilot.unit_set_hash}, not {expected_unit_set_hash}; "
            "refusing to use it"
        )
    _check_invariants(pilot)
    return pilot


def _check_invariants(pilot: PilotSet) -> None:
    """What every pilot artifact must satisfy, whatever its digest says."""
    if pilot.split != DEV_SPLIT:
        raise PilotError(f"the pilot declares split {pilot.split!r}; it is drawn from dev alone")
    if pilot.rule != PILOT_RULE:
        raise PilotError("the pilot records a rule other than the one HU-1 declares")
    qids = [question.qid for question in pilot.questions]
    if qids != sorted(qids) or len(set(qids)) != len(qids):
        raise PilotError("the pilot's questions must be unique and sorted by question id")
    for question in pilot.questions:
        if len(question.read) != pilot.read_depth:
            raise PilotError(
                f"question {question.qid} has {len(question.read)} read paragraphs, "
                f"not {pilot.read_depth}"
            )
        if question.p1 != question.read[0]:
            raise PilotError(f"question {question.qid}: p1 is not the first read paragraph")
        if not question.missing:
            raise PilotError(f"question {question.qid} has no missing paragraph to find")
        if set(question.missing) & set(question.read):
            raise PilotError(
                f"question {question.qid} lists a missing paragraph it has already read"
            )


def check_pilot_against(pilot: PilotSet, questions: Sequence[Question]) -> None:
    """Every pilot question is a dev question, and every missing paragraph one of its gold."""
    by_id: Mapping[str, Question] = {question.qid: question for question in questions}
    for entry in pilot.questions:
        question = by_id.get(entry.qid)
        if question is None:
            raise PilotError(f"pilot question {entry.qid} is not in the pool's questions")
        if question.split != DEV_SPLIT:
            raise PilotError(f"pilot question {entry.qid} is on split {question.split!r}")
        if not set(entry.missing) <= set(question.gold_unit_ids):
            raise PilotError(
                f"question {entry.qid} lists a missing paragraph that is not one of its gold"
            )
