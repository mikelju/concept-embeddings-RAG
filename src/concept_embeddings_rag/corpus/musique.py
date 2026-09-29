"""Phase 15, S1-S2: the MuSiQue-Ans corpus, its validation questions and the live-path set.

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

S2 adds the reading of the two files (the train file streamed, only its paragraphs and each
row's id and question text kept), the checks against the examined source, the live-path
block (D4: train qids and text, never gold), and two counts the corpus manifest declares:
near-duplicate gold (author decision 5) and units past GLiNER's window (author decision 3).
"""

import gzip
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
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
# The `split` of the D4 live-path questions: train qids with their text only, never gold.
LIVE_PATH_SPLIT = "musique-train-live-path"

# --- The source pin (D1, author decision 1) ----------------------------------------------
# The Hugging Face copy's release files, as read through the Hub tree API on 2026-09-29
# (metadata only). That their bytes equal the official Google Drive zip's is not verified.

SOURCE_DATASET = "bdsaglam/musique"
SOURCE_REVISION = "22873a405dd809893b22ada0b499299fb612d2df"  # pragma: allowlist secret
SOURCE_LICENSE = (
    "the license stated by the MuSiQue release; not read or verified at this pin, and not "
    "recorded here"
)
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


# --- Reading the two files (S2) ----------------------------------------------------------


def read_rows(path: Path | str) -> Iterator[dict[str, Any]]:
    """One JSON object per line, parsed with the standard library, one line at a time."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@dataclass(frozen=True)
class Source:
    """What S2 keeps of the two files: the pooled units, the validation rows whole, and of
    each train row only its id and question text (its paragraphs are pooled, nothing else)."""

    units: list[IndexingUnit]
    validation: list[dict[str, Any]]
    train: list[tuple[str, str]]
    train_unanswerable: int
    paragraphs_read: int


def read_source(train_path: Path | str, validation_path: Path | str) -> Source:
    """Both files, the train file streamed: a train row is dropped once its paragraphs are
    pooled, so memory holds the units and the train qids, not the 241 MB of rows (D2)."""
    validation = list(read_rows(validation_path))
    train: list[tuple[str, str]] = []
    tally = {"paragraphs": 0, "train_unanswerable": 0}

    def rows() -> Iterator[Mapping[str, Any]]:
        for entry in validation:
            tally["paragraphs"] += len(entry["paragraphs"])
            yield entry
        for entry in read_rows(train_path):
            train.append((str(entry["id"]), str(entry["question"])))
            tally["train_unanswerable"] += entry["answerable"] is not True
            tally["paragraphs"] += len(entry["paragraphs"])
            yield {"paragraphs": entry["paragraphs"]}

    units = pool_units(rows())
    return Source(
        units=units,
        validation=validation,
        train=train,
        train_unanswerable=tally["train_unanswerable"],
        paragraphs_read=tally["paragraphs"],
    )


def source_problems(
    source: Source,
    *,
    train_rows: int,
    validation_rows: int,
    supporting_counts: Mapping[int, int],
) -> list[str]:
    """Every way the parsed files differ from the source examined on 2026-09-29. Any one
    means the files are not that source: the stage stops and the author decides."""
    problems: list[str] = []
    if len(source.train) != train_rows:
        problems.append(f"the train file holds {len(source.train)} rows, not {train_rows}")
    if len(source.validation) != validation_rows:
        problems.append(
            f"the validation file holds {len(source.validation)} rows, not {validation_rows}"
        )
    unanswerable = source.train_unanswerable + sum(
        row["answerable"] is not True for row in source.validation
    )
    if unanswerable:
        problems.append(f"{unanswerable} rows are not answerable")
    qids = [qid for qid, _text in source.train] + [str(row["id"]) for row in source.validation]
    if len(set(qids)) != len(qids):
        problems.append(f"the qids are not unique ({len(qids) - len(set(qids))} repeated)")
    found = _counts(
        sum(bool(p["is_supporting"]) for p in r["paragraphs"]) for r in source.validation
    )
    expected = {str(n): count for n, count in sorted(supporting_counts.items())}
    if found != expected:
        problems.append(f"validation supporting counts {found}, not {expected}")
    return problems


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


def corpus_identity(units: Sequence[IndexingUnit]) -> dict[str, str]:
    """The two corpus digests `corpus.json` records, known before it is written, so the gold
    can be mapped and counted into the write-once manifest."""
    unit_ids = [unit.unit_id for unit in units]
    return {"unit_set_hash": _unit_set_hash(unit_ids), "ordered_unit_digest": digest_of(*unit_ids)}


def write_corpus(
    units: Sequence[IndexingUnit],
    directory: Path | str,
    *,
    paragraphs_read: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """`corpus.jsonl.gz` and `corpus.json`, written once, in the form `load_corpus` reads.

    `extra` adds recorded fields to the manifest (S2's near-duplicate and long-unit counts).
    """
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
    manifest: dict[str, Any] = {
        **(extra or {}),
        "phase": 15,
        "source": source_pin(),
        "n_units": len(units),
        "paragraphs_read": paragraphs_read,
        "duplicate_paragraphs_collapsed": paragraphs_read - len(units),
        "empty_text_units": sum(1 for unit in units if not unit.text),
        "indexable_text_bytes": sum(len(unit.indexable_text.encode("utf-8")) for unit in units),
        **corpus_identity(units),
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


def units_over_window(
    units: Sequence[IndexingUnit],
    count_words: Callable[[str], int],
    *,
    max_len: int,
    share: float,
    splitter: str,
) -> dict[str, Any]:
    """Units whose `indexable_text` holds more than `max_len` of GLiNER's words (author
    decision 3). GLiNER truncates its input at `max_len` words and a MuSiQue unit is not
    sentence-split, so these are the units the extractor would read only in part. Counted on
    the whole text the extractor is handed, title included, so the count is an upper bound.
    Past `share` of the units the stage stops and the author decides."""
    over = []
    for unit in units:
        words = count_words(unit.indexable_text)
        if words > max_len:
            over.append({"unit_id": unit.unit_id, "words": words})
    return {
        "max_len": max_len,
        "splitter": splitter,
        "counted_on": "indexable_text",
        "n_units": len(units),
        "units_over": len(over),
        "share_over": len(over) / len(units) if units else 0.0,
        "ceiling_share": share,
        "stop": len(over) > share * len(units),
        "over": over,
    }


def near_duplicate_gold(
    units: Sequence[IndexingUnit], questions: Sequence[Question], *, prefix_chars: int
) -> dict[str, Any]:
    """Gold units with another unit of the same title whose text starts with the same
    `prefix_chars` characters (author decision 5). Counted and declared: gold stays the
    marked paragraph, nothing is merged, no metric changes."""
    keys = {unit.unit_id: (unit.title, unit.text[:prefix_chars]) for unit in units}
    sharing = Counter(keys.values())
    gold = {g for q in questions for g in q.gold_unit_ids if g in keys}
    near = sorted(g for g in gold if sharing[keys[g]] > 1)
    marked = set(near)
    pairs = [(q.qid, g) for q in questions for g in q.gold_unit_ids if g in keys]
    return {
        "rule": (
            f"a mapped gold unit with another unit of the same title whose text starts with the "
            f"same {prefix_chars} characters; counted, never merged"
        ),
        "prefix_chars": prefix_chars,
        "gold_units": len(gold),
        "gold_units_with_near_duplicate": len(near),
        "gold_pairs": len(pairs),
        "gold_pairs_with_near_duplicate": sum(g in marked for _qid, g in pairs),
        "unit_ids": near,
        "qids": sorted({qid for qid, g in pairs if g in marked}),
    }


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


# --- The live-path questions (D4) --------------------------------------------------------


def _live_question(qid: str, text: str) -> Question:
    """A train question as the live path sees it: its text, and no gold or answer."""
    return Question(
        qid=qid,
        question=text,
        answer="",
        gold_unit_ids=(),
        supporting_facts=(),
        split=LIVE_PATH_SPLIT,
    )


def live_path_block(train: Iterable[tuple[str, str]], *, size: int) -> dict[str, Any]:
    """The first `size` train qids sorted by id (Python string order), with their question
    text only: D4 runs the live retrievers on them and computes no metric."""
    chosen = sorted(train)[:size]
    return {
        "split": LIVE_PATH_SPLIT,
        "rule": (
            f"the first {size} MuSiQue-Ans train qids sorted by id; question text only, no gold, "
            "no answer; no metric of them is computed (D4)"
        ),
        "n_questions": len(chosen),
        "question_digest": question_digest([_live_question(q, t) for q, t in chosen]),
        "questions": [{"qid": qid, "question": text} for qid, text in chosen],
    }


def live_path_questions(body: Mapping[str, Any]) -> list[Question]:
    """The live-path questions of a loaded `questions.json`, their digest recomputed."""
    block = body["live_path"]
    questions = [_live_question(str(e["qid"]), str(e["question"])) for e in block["questions"]]
    if question_digest(questions) != block["question_digest"]:
        raise MusiqueError("the live-path questions do not match their recorded digest")
    return questions


def write_questions(
    directory: Path | str, body: Mapping[str, Any], *, live_path: Mapping[str, Any]
) -> Path:
    """Freeze the validation questions and the live-path block once. A second write over an
    existing file is refused."""
    directory = Path(directory)
    target = directory / QUESTIONS_FILENAME
    if target.exists():
        raise MusiqueError(f"{target} already freezes the Phase 15 questions")
    directory.mkdir(parents=True, exist_ok=True)
    frozen = {**body, "live_path": dict(live_path)}
    write_text_atomic(target, json.dumps(frozen, indent=2, ensure_ascii=False, sort_keys=True))
    return target


def load_questions(
    directory: Path | str, *, corpus_unit_set_hash: str
) -> tuple[list[Question], dict[str, Any]]:
    """The frozen questions, every digest recomputed (the live path's too), refused on a
    DATA_STOP or another corpus. The live-path questions come from `live_path_questions`."""
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
    live_path_questions(body)
    return questions, body
