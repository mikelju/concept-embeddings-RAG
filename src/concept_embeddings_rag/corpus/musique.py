"""Phase 15, S1: the MuSiQue-Ans corpus and its validation questions.

MuSiQue ships each question with up to 20 paragraphs, some marked `is_supporting`. The
corpus is every paragraph of every train and validation question, pooled (D1); the gold of
a validation question is its own supporting paragraphs, mapped to corpus units (D2).

Everything that can be reused is: a paragraph becomes an `IndexingUnit` through
`unit_id_for`, so its id is the Phase 1 content id over title and text and deduplication
by title and text is the id itself; the corpus is written in the line format of
`corpus.fullwiki` and read back by the existing `corpus.fullwiki.load_corpus`; an unmapped
gold becomes Phase 9's sentinel id, so it stays in the denominator as a miss.

What is new is the number of gold paragraphs. Phase 9's question code keeps exactly two,
and MuSiQue has 2, 3 or 4, so the questions are written and read here, never with it, and
the mapping digest covers every gold id of a question.
"""

import gzip
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, write_text_atomic
from concept_embeddings_rag.corpus.fullwiki import (
    CORPUS_MANIFEST,
    CORPUS_NAME,
    _corpus_line,
    _unit_set_hash,
)
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question, unit_id_for
from concept_embeddings_rag.evaluation import fullwiki as phase9

QUESTIONS_FILENAME = "questions.json"
# The `split` every validation question carries.
VALIDATION_SPLIT = "musique-validation"

# --- The source pin (D1, author decision 1) ----------------------------------------------
# The Hugging Face copy's release files, as read through the Hub tree API on 2026-09-29
# (metadata only). That their bytes equal the official Google Drive zip's is not verified.

SOURCE_DATASET = "bdsaglam/musique"
SOURCE_REVISION = "22873a405dd809893b22ada0b499299fb612d2df"  # pragma: allowlist secret
SOURCE_LICENSE = "CC BY 4.0 (the MuSiQue release's stated license; not re-read at this pin)"
TRAIN_FILE = "musique_ans_v1.0_train.jsonl"
# MuSiQue's `dev` file is the validation split.
VALIDATION_FILE = "musique_ans_v1.0_dev.jsonl"
TRAIN_SHA256 = (
    "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"  # pragma: allowlist secret
)
VALIDATION_SHA256 = (
    "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"  # pragma: allowlist secret
)
SOURCE_FILES: dict[str, dict[str, Any]] = {
    TRAIN_FILE: {"bytes": 241_046_755, "sha256": TRAIN_SHA256},
    VALIDATION_FILE: {"bytes": 30_439_728, "sha256": VALIDATION_SHA256},
}


def source_url(name: str) -> str:
    return f"https://huggingface.co/datasets/{SOURCE_DATASET}/resolve/{SOURCE_REVISION}/{name}"


def source_pin() -> dict[str, Any]:
    """The pin every Phase 15 corpus and question artifact records."""
    return {
        "dataset": SOURCE_DATASET,
        "revision": SOURCE_REVISION,
        "license": SOURCE_LICENSE,
        "files": {
            name: {"url": source_url(name), **identity} for name, identity in SOURCE_FILES.items()
        },
    }


class MusiqueError(Exception):
    """A Phase 15 corpus or question artifact is not the one this step may use."""


# --- The corpus (D1) ---------------------------------------------------------------------


def unit_of(paragraph: Mapping[str, Any]) -> IndexingUnit:
    """One whole MuSiQue paragraph as one unit: `indexable_text = f"{title}. {text}"`."""
    title = str(paragraph["title"])
    sentences = (str(paragraph["paragraph_text"]),)
    return IndexingUnit(unit_id=unit_id_for(title, sentences), title=title, sentences=sentences)


def pool_units(rows: Iterable[Mapping[str, Any]]) -> list[IndexingUnit]:
    """Every paragraph of every row, deduplicated by unit id and sorted by it, as `build_pool`
    orders its pool: the corpus does not depend on the order of the files or their rows."""
    units: dict[str, IndexingUnit] = {}
    for row in rows:
        for paragraph in row["paragraphs"]:
            unit = unit_of(paragraph)
            units.setdefault(unit.unit_id, unit)
    return sorted(units.values(), key=lambda unit: unit.unit_id)


def write_corpus(
    units: Sequence[IndexingUnit], directory: Path | str, *, paragraphs_read: int
) -> dict[str, Any]:
    """`corpus.jsonl.gz` and `corpus.json`, written once, in the form `load_corpus` reads."""
    directory = Path(directory)
    target = directory / CORPUS_MANIFEST
    corpus_path = directory / CORPUS_NAME
    if target.exists() or corpus_path.exists():
        raise MusiqueError(f"{directory} already holds a corpus; a Phase 15 corpus is written once")
    directory.mkdir(parents=True, exist_ok=True)
    text = "".join(_corpus_line(unit) + "\n" for unit in units)
    temporary = corpus_path.with_name(corpus_path.name + ".tmp")
    # mtime=0 so the same units always compress to the same bytes.
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(text.encode("utf-8"))
    temporary.replace(corpus_path)
    unit_ids = [unit.unit_id for unit in units]
    manifest: dict[str, Any] = {
        "phase": 15,
        "source": source_pin(),
        "n_units": len(unit_ids),
        "paragraphs_read": paragraphs_read,
        "duplicate_paragraphs_collapsed": paragraphs_read - len(unit_ids),
        "empty_text_units": sum(1 for unit in units if not unit.text),
        "indexable_text_bytes": sum(len(unit.indexable_text.encode("utf-8")) for unit in units),
        "unit_set_hash": _unit_set_hash(unit_ids),
        "ordered_unit_digest": digest_of(*unit_ids),
        "corpus_file": CORPUS_NAME,
        "corpus_file_bytes": corpus_path.stat().st_size,
        "unit_contract": (
            "one MuSiQue paragraph per unit, sentences = (paragraph_text,), indexable_text = "
            'f"{title}. {paragraph_text}"; deduplicated by title and text, sorted by unit id'
        ),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_text_atomic(target, json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return manifest


# --- The validation questions and their gold (D2) ----------------------------------------


def question_digest(questions: Sequence[Question]) -> str:
    """Phase 9's question digest: qid, split and text, gold excluded."""
    return phase9._question_digest(questions)


def mapping_digest(questions: Sequence[Question], corpus_unit_set_hash: str) -> str:
    """Every gold id of every question. For two gold ids it equals Phase 9's `_mapping_digest`."""
    return digest_of(
        corpus_unit_set_hash,
        *("\t".join((q.qid, *q.gold_unit_ids)) for q in sorted(questions, key=lambda q: q.qid)),
    )


def validation_questions(
    rows: Sequence[Mapping[str, Any]],
    units: Iterable[IndexingUnit],
    *,
    corpus: Mapping[str, Any],
    share: float = config.PHASE_15_UNMAPPED_SHARE,
) -> tuple[list[Question], dict[str, Any]]:
    """Each row's supporting paragraphs mapped to corpus units, in paragraph order, and the body.

    Gold is per question: the ids of its own `is_supporting` paragraphs, deduplicated. A
    supporting paragraph with no unit in the corpus becomes Phase 9's sentinel, so the
    question stays in the denominator and can never reach Full Support; more than `share`
    of the questions with one is `DATA_STOP`. Supporting paragraphs that collapse into one
    unit are recorded.
    """
    known = {unit.unit_id for unit in units}
    questions: list[Question] = []
    entries: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    collapsed: list[dict[str, Any]] = []
    for row in rows:
        qid = str(row["id"])
        supporting = [p for p in row["paragraphs"] if p["is_supporting"]]
        gold: list[str] = []
        for position, paragraph in enumerate(supporting):
            unit_id = unit_of(paragraph).unit_id
            if unit_id not in known:
                unit_id = phase9._sentinel(qid, position)
                unmapped.append(
                    {"qid": qid, "title": str(paragraph["title"]), "idx": int(paragraph["idx"])}
                )
            gold.append(unit_id)
        gold = list(dict.fromkeys(gold))
        if len(gold) != len(supporting):
            collapsed.append({"qid": qid, "supporting": len(supporting), "gold": len(gold)})
        question = Question(
            qid=qid,
            question=str(row["question"]),
            answer=str(row["answer"]),
            gold_unit_ids=tuple(gold),
            supporting_facts=tuple((str(p["title"]), int(p["idx"])) for p in supporting),
            split=VALIDATION_SPLIT,
        )
        questions.append(question)
        entries.append(
            {
                "qid": qid,
                "question": question.question,
                "answer": question.answer,
                "supporting_facts": [[t, i] for t, i in question.supporting_facts],
                "gold_unit_ids": list(question.gold_unit_ids),
            }
        )

    affected = len({entry["qid"] for entry in unmapped})
    unit_set_hash = str(corpus["unit_set_hash"])
    body: dict[str, Any] = {
        "phase": 15,
        "source": source_pin(),
        "split": VALIDATION_SPLIT,
        "n_questions": len(questions),
        "corpus_unit_set_hash": unit_set_hash,
        "corpus_ordered_unit_digest": corpus["ordered_unit_digest"],
        "question_digest": question_digest(questions),
        "mapping_digest": mapping_digest(questions, unit_set_hash),
        "supporting_counts": _counts(len(q.supporting_facts) for q in questions),
        "gold_counts": _counts(len(q.gold_unit_ids) for q in questions),
        "collapsed_gold": collapsed,
        "questions_with_unmapped_gold": affected,
        "unmapped_ceiling_share": share,
        "unmapped": unmapped,
        "unmapped_rule": (
            "a supporting paragraph with no corpus unit stays in the denominator as a sentinel "
            "id no unit carries: never retrieved by any system"
        ),
        "terminal_state": phase9.DATA_STOP if affected > share * len(questions) else None,
        "questions": entries,
    }
    return questions, body


def _counts(values: Iterable[int]) -> dict[str, int]:
    """Questions by number of paragraphs, keyed as JSON writes them."""
    return {str(n): count for n, count in sorted(Counter(values).items())}


def write_questions(directory: Path | str, body: Mapping[str, Any]) -> Path:
    """Freeze the validation questions once. A second write over an existing file is refused."""
    directory = Path(directory)
    target = directory / QUESTIONS_FILENAME
    if target.exists():
        raise MusiqueError(f"{target} already freezes the Phase 15 questions")
    directory.mkdir(parents=True, exist_ok=True)
    write_text_atomic(target, json.dumps(dict(body), indent=2, ensure_ascii=False, sort_keys=True))
    return target


def load_questions(
    directory: Path | str, *, corpus_unit_set_hash: str
) -> tuple[list[Question], dict[str, Any]]:
    """The frozen questions, both digests recomputed, refused on a DATA_STOP or another corpus."""
    target = Path(directory) / QUESTIONS_FILENAME
    body: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    if body["terminal_state"] is not None:
        raise phase9.DataStop(
            f"{body['questions_with_unmapped_gold']} questions have an unmapped gold paragraph, "
            f"past the share of {body['unmapped_ceiling_share']}; no retrieval score is published"
        )
    if body["corpus_unit_set_hash"] != corpus_unit_set_hash:
        raise MusiqueError(
            f"{target.name} was mapped against corpus {body['corpus_unit_set_hash']}, not "
            f"{corpus_unit_set_hash}"
        )
    questions = [
        Question(
            qid=str(entry["qid"]),
            question=str(entry["question"]),
            answer=str(entry["answer"]),
            gold_unit_ids=tuple(str(unit_id) for unit_id in entry["gold_unit_ids"]),
            supporting_facts=tuple((str(t), int(i)) for t, i in entry["supporting_facts"]),
            split=str(body["split"]),
        )
        for entry in body["questions"]
    ]
    if (
        question_digest(questions) != body["question_digest"]
        or mapping_digest(questions, corpus_unit_set_hash) != body["mapping_digest"]
    ):
        raise MusiqueError(f"{target.name} does not match its recorded digests")
    return questions, body
