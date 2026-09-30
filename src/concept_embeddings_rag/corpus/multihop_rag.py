"""Phase 16, S1: the MultiHop-RAG corpus, its answerable queries and the live-path set.

MultiHop-RAG ships 609 news articles (`corpus.json`) and 2,556 queries (`MultiHopRAG.json`)
whose evidence is a list of sentences copied verbatim from the articles. The corpus is every
non-empty newline paragraph of every article body (D1, boilerplate included); the gold of a
query is the set of units holding its facts (D2).

Reused from Phase 15 (`corpus.musique`), never copied: a paragraph becomes an
`IndexingUnit` through `unit_id_for`, so a verbatim repeat inside an article collapses into
one unit; the corpus is written by `musique.write_corpus` in the line format of
`corpus.fullwiki`, each line carrying its article's metadata under a key `load_corpus`
ignores; questions are written and read by `musique.write_questions` / `load_questions`,
whose digests already cover 2-4 gold ids; an unmapped fact becomes Phase 9's sentinel, so
it stays in the denominator as a miss.

What is new is the gold mapping: a fact is located in its article's body, and one that
crosses a newline maps to the paragraph holding its first line (author decision 2). The
`DATA_STOP` share is counted over facts, not queries (D2). The `answer` field is never read.
"""

import bisect
import gzip
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of
from concept_embeddings_rag.corpus import musique
from concept_embeddings_rag.corpus.fullwiki import CORPUS_MANIFEST, CORPUS_NAME
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question, unit_id_for
from concept_embeddings_rag.evaluation import fullwiki as phase9

QUESTIONS_FILENAME = musique.QUESTIONS_FILENAME
# The `split` every answerable query carries.
ANSWERABLE_SPLIT = "multihop-rag-answerable"
# The `split` of the D4 live-path questions: the `null` queries, text only, never gold.
LIVE_PATH_SPLIT = "multihop-rag-null-live-path"

# --- The source pin (D1) -------------------------------------------------------------------
# Measured on 2026-09-30 (pre-spec notes); licence ODC-BY as stated on the dataset card.

SOURCE_DATASET = "yixuantt/MultiHopRAG"
SOURCE_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"  # pragma: allowlist secret
SOURCE_LICENSE = "ODC-BY"
CORPUS_FILE = "corpus.json"
QUERIES_FILE = "MultiHopRAG.json"
CORPUS_SHA256 = (
    "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f"  # pragma: allowlist secret
)
QUERIES_SHA256 = (
    "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff"  # pragma: allowlist secret
)
SOURCE_FILES: dict[str, dict[str, Any]] = {
    CORPUS_FILE: {"bytes": 6_785_567, "sha256": CORPUS_SHA256},
    QUERIES_FILE: {"bytes": 5_171_312, "sha256": QUERIES_SHA256},
}

UNIT_CONTRACT = (
    "one newline paragraph of a MultiHop-RAG article body per unit: body.split('\\n'), every "
    "piece whose strip() is non-empty, its text piece.strip(); sentences = (text,), "
    'indexable_text = f"{title}. {text}"; deduplicated by title and text, sorted by unit id; '
    "no minimum length, boilerplate kept; article metadata in the line, never indexed"
)
METADATA_FIELDS = ("index", "url", "source", "category", "published_at")


def source_url(name: str) -> str:
    return f"https://huggingface.co/datasets/{SOURCE_DATASET}/resolve/{SOURCE_REVISION}/{name}"


def source_pin() -> dict[str, Any]:
    """The pin every Phase 16 corpus and question artifact records."""
    return {
        "dataset": SOURCE_DATASET,
        "revision": SOURCE_REVISION,
        "license": SOURCE_LICENSE,
        "files": {
            name: {"url": source_url(name), **identity} for name, identity in SOURCE_FILES.items()
        },
    }


class MultiHopRagError(musique.MusiqueError):
    """A Phase 16 corpus or question artifact is not the one this step may use. A subclass
    of Phase 15's error, so a stage catching `MusiqueError` catches both."""


def read_source(
    corpus_path: Path | str, queries_path: Path | str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The two pinned files, parsed with the standard `json`: articles and queries in order."""
    articles = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    queries = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    return list(articles), list(queries)


def qid_of(position: int) -> str:
    """A query's id: its position in the pinned `MultiHopRAG.json`, which has no id."""
    return f"mhr-{position:04d}"


def is_answerable(query: Mapping[str, Any]) -> bool:
    """D2: a query with evidence is answerable; the `null` ones have none."""
    return bool(query["evidence_list"])


# --- The corpus (D1) -----------------------------------------------------------------------


@dataclass(frozen=True)
class Paragraph:
    """One non-empty newline piece of a body: its position among the article's paragraphs,
    its stripped text, and the span of that text in the body."""

    position: int
    text: str
    start: int
    end: int


def paragraphs_of(body: str) -> list[Paragraph]:
    """`body.split("\\n")`, keeping each piece whose `strip()` is non-empty, its text the piece
    with outer whitespace removed (the inside untouched), with its span in the body."""
    paragraphs: list[Paragraph] = []
    offset = 0
    for piece in body.split("\n"):
        text = piece.strip()
        if text:
            start = offset + len(piece) - len(piece.lstrip())
            paragraphs.append(Paragraph(len(paragraphs), text, start, start + len(text)))
        offset += len(piece) + 1
    return paragraphs


def unit_of(title: str, text: str) -> IndexingUnit:
    """One newline paragraph as one unit: `indexable_text = f"{title}. {text}"`."""
    sentences = (text,)
    return IndexingUnit(unit_id=unit_id_for(title, sentences), title=title, sentences=sentences)


def _article_units(
    articles: Sequence[Mapping[str, Any]],
) -> Iterator[tuple[int, Paragraph, IndexingUnit]]:
    for index, article in enumerate(articles):
        title = str(article["title"])
        for paragraph in paragraphs_of(str(article["body"])):
            yield index, paragraph, unit_of(title, paragraph.text)


def pool_units(
    articles: Sequence[Mapping[str, Any]],
) -> tuple[list[IndexingUnit], dict[str, int]]:
    """Every paragraph of every article, deduplicated by unit id and sorted by it, and each
    unit's article index. A unit id claimed by two articles is refused (titles are distinct
    on the pinned source and the title is inside the id, so it cannot happen there)."""
    units: dict[str, IndexingUnit] = {}
    article_of: dict[str, int] = {}
    for index, _paragraph, unit in _article_units(articles):
        if article_of.setdefault(unit.unit_id, index) != index:
            raise MultiHopRagError(
                f"unit {unit.unit_id} belongs to articles {article_of[unit.unit_id]} and {index}"
            )
        units.setdefault(unit.unit_id, unit)
    return sorted(units.values(), key=lambda unit: unit.unit_id), article_of


def corpus_profile(
    articles: Sequence[Mapping[str, Any]],
    *,
    boilerplate_max_words: int = config.PHASE_16_BOILERPLATE_MAX_WORDS,
    window_words: int = config.GLINER_MAX_LEN,
) -> dict[str, Any]:
    """The D1 counts over the newline paragraphs read (repeats included, as measured on
    2026-09-30): boilerplate (at most `boilerplate_max_words` whitespace words) and paragraphs
    whose text holds more than `window_words` whitespace words. Counted, never filtered."""
    newline = boilerplate = 0
    over: list[dict[str, Any]] = []
    for index, paragraph, unit in _article_units(articles):
        newline += 1
        words = len(paragraph.text.split())
        boilerplate += words <= boilerplate_max_words
        if words > window_words:
            over.append(
                {
                    "article": index,
                    "position": paragraph.position,
                    "unit_id": unit.unit_id,
                    "words": words,
                }
            )
    return {
        "articles": len(articles),
        "newline_paragraphs": newline,
        "boilerplate_max_words": boilerplate_max_words,
        "boilerplate_paragraphs": boilerplate,
        "window_words": window_words,
        "over_window_counted_on": "paragraph text, whitespace words",
        "over_window_paragraphs": len(over),
        "over_window": over,
    }


def article_metadata(article: Mapping[str, Any], index: int) -> dict[str, Any]:
    """What a unit keeps of its article (D1, D8): never indexed."""
    return {
        "index": index,
        "url": str(article["url"]),
        "source": str(article["source"]),
        "category": str(article["category"]),
        "published_at": str(article["published_at"]),
    }


def _metadata_digest(unit_ids: Sequence[str], by_unit: Mapping[str, Mapping[str, Any]]) -> str:
    return digest_of(
        *(
            "\t".join([unit_id, *(str(by_unit[unit_id][f]) for f in METADATA_FIELDS)])
            for unit_id in unit_ids
        )
    )


def write_corpus(
    units: Sequence[IndexingUnit],
    directory: Path | str,
    *,
    articles: Sequence[Mapping[str, Any]],
    article_of: Mapping[str, int],
    paragraphs_read: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """`corpus.jsonl.gz` and `corpus.json`, written once through `musique.write_corpus`, each
    line with an `article` key and the manifest with `article_metadata_digest`."""
    by_unit = {
        unit.unit_id: article_metadata(articles[article_of[unit.unit_id]], article_of[unit.unit_id])
        for unit in units
    }
    try:
        return musique.write_corpus(
            units,
            directory,
            paragraphs_read=paragraphs_read,
            extra={
                **(extra or {}),
                "articles": len(articles),
                "article_metadata_fields": list(METADATA_FIELDS),
                "article_metadata_digest": _metadata_digest(
                    [unit.unit_id for unit in units], by_unit
                ),
            },
            phase=16,
            source=source_pin(),
            unit_contract=UNIT_CONTRACT,
            line_extra={unit_id: {"article": meta} for unit_id, meta in by_unit.items()},
        )
    except musique.MusiqueError as error:
        raise MultiHopRagError(str(error)) from error


def load_articles(directory: Path | str) -> dict[str, dict[str, Any]]:
    """`unit_id -> article metadata`, in corpus order, its digest recomputed against
    `corpus.json`. The units themselves are read by `fullwiki.load_corpus`."""
    directory = Path(directory)
    manifest = json.loads((directory / CORPUS_MANIFEST).read_text(encoding="utf-8"))
    by_unit: dict[str, dict[str, Any]] = {}
    with gzip.open(directory / CORPUS_NAME, "rt", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            by_unit[str(entry["unit_id"])] = dict(entry["article"])
    if _metadata_digest(list(by_unit), by_unit) != manifest["article_metadata_digest"]:
        raise MultiHopRagError(
            f"the article metadata in {CORPUS_NAME} does not match the digest {CORPUS_MANIFEST} "
            "records"
        )
    return by_unit


# --- The answerable queries and their gold (D2) --------------------------------------------


def _locate(
    fact: str, body: str, paragraphs: Sequence[Paragraph]
) -> tuple[Paragraph | None, int, bool]:
    """The paragraph holding the fact's first non-whitespace character, the number of
    paragraphs its span touches, and whether it occurs more than once. `None` if absent."""
    start = body.find(fact)
    if start < 0 or not fact.strip():
        return None, 0, False
    end = start + len(fact)
    first = start + len(fact) - len(fact.lstrip())
    starts = [p.start for p in paragraphs]
    slot = bisect.bisect_right(starts, first) - 1
    holder = paragraphs[slot] if slot >= 0 and first < paragraphs[slot].end else None
    crossed = sum(1 for p in paragraphs if p.start < end and p.end > start)
    return holder, crossed, body.count(fact) > 1


def answerable_questions(
    queries: Sequence[Mapping[str, Any]],
    articles: Sequence[Mapping[str, Any]],
    units: Sequence[IndexingUnit],
    *,
    corpus: Mapping[str, Any],
    share: float = config.PHASE_16_UNMAPPED_FACT_SHARE,
) -> tuple[list[Question], dict[str, Any]]:
    """Each answerable query's facts mapped to corpus units, in fact order, and the body.

    A fact names its article by `title` (its `url` must agree). It is found with
    `body.find(fact)`, first occurrence. Inside one paragraph it maps to that unit; across a
    newline it maps to the paragraph holding its first line and is listed. Gold ids are
    deduplicated in fact order. An unmapped fact is Phase 9's sentinel, in the denominator;
    more than `share` of the facts unmapped is `DATA_STOP`. `answer` is never read.
    """
    known = {unit.unit_id for unit in units}
    by_title = {str(a["title"]): i for i, a in enumerate(articles)}
    paragraphs = [paragraphs_of(str(a["body"])) for a in articles]
    questions: list[Question] = []
    types: list[str] = []
    entries: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    straddling: list[dict[str, Any]] = []
    repeated: list[dict[str, Any]] = []
    evidence_counts: list[int] = []
    articles_per_query: list[int] = []
    same_article_qids: list[str] = []
    one_article_qids: list[str] = []
    facts = inside = 0
    for position, query in enumerate(queries):
        if not is_answerable(query):
            continue
        qid = qid_of(position)
        gold: list[str] = []
        supporting: list[tuple[str, int]] = []
        titles: list[str] = []
        gold_articles: set[int] = set()
        for number, piece in enumerate(query["evidence_list"]):
            facts += 1
            title = str(piece["title"])
            titles.append(title)
            index = by_title.get(title)
            reason = None
            holder: Paragraph | None = None
            if index is None:
                reason = "no article with this title"
            elif piece.get("url") and piece["url"] != articles[index]["url"]:
                reason = "the url does not match the article's"
            else:
                text = str(articles[index]["body"])
                holder, crossed, twice = _locate(str(piece["fact"]), text, paragraphs[index])
                if holder is None:
                    reason = "fact not in the article body"
            if holder is None or index is None:
                unit_id = phase9._sentinel(qid, number)
                unmapped.append({"qid": qid, "evidence": number, "title": title, "reason": reason})
                supporting.append((title, -1))
            else:
                unit_id = unit_of(title, holder.text).unit_id
                supporting.append((title, holder.position))
                gold_articles.add(index)
                if "\n" in str(piece["fact"]):
                    straddling.append(
                        {
                            "qid": qid,
                            "evidence": number,
                            "article": index,
                            "unit_id": unit_id,
                            "position": holder.position,
                            "paragraphs_crossed": crossed,
                        }
                    )
                else:
                    inside += 1
                if twice:
                    repeated.append({"qid": qid, "evidence": number, "article": index})
            gold.append(unit_id)
        gold = list(dict.fromkeys(gold))
        evidence_counts.append(len(query["evidence_list"]))
        articles_per_query.append(len(set(titles)))
        if len(set(titles)) < len(titles):
            same_article_qids.append(qid)
        if len(gold_articles) == 1 and all(g in known for g in gold):
            one_article_qids.append(qid)
        question = Question(
            qid=qid,
            question=str(query["query"]),
            answer="",
            gold_unit_ids=tuple(gold),
            supporting_facts=tuple(supporting),
            split=ANSWERABLE_SPLIT,
        )
        questions.append(question)
        types.append(str(query["question_type"]))
        entries.append(
            {
                "qid": qid,
                "question": question.question,
                "answer": "",
                "question_type": types[-1],
                "supporting_facts": [[t, i] for t, i in supporting],
                "gold_unit_ids": list(question.gold_unit_ids),
            }
        )

    unit_set_hash = str(corpus["unit_set_hash"])
    body: dict[str, Any] = {
        "phase": 16,
        "source": source_pin(),
        "split": ANSWERABLE_SPLIT,
        "n_questions": len(questions),
        "corpus_unit_set_hash": unit_set_hash,
        "corpus_ordered_unit_digest": corpus["ordered_unit_digest"],
        "question_digest": musique.question_digest(questions),
        "mapping_digest": musique.mapping_digest(questions, unit_set_hash),
        "question_type_digest": _type_digest(entries),
        "type_counts": dict(sorted(Counter(types).items())),
        "evidence_counts": musique._counts(evidence_counts),
        "gold_counts": musique._counts(len(q.gold_unit_ids) for q in questions),
        "distinct_gold_units": len({g for q in questions for g in q.gold_unit_ids if g in known}),
        "facts": facts,
        "facts_inside": inside,
        "straddling_rule": (
            "a fact whose matched span contains a newline maps to the unit of the paragraph "
            "holding its first non-whitespace character (its first line; author decision 2)"
        ),
        "straddling": straddling,
        "straddling_digest": digest_of(*(json.dumps(e, sort_keys=True) for e in straddling)),
        "facts_found_more_than_once": repeated,
        "unmapped_facts": len(unmapped),
        "unmapped_ceiling_share": share,
        "unmapped": unmapped,
        "questions_with_unmapped_gold": len({e["qid"] for e in unmapped}),
        "unmapped_rule": (
            "a fact whose article or text is not found stays in the denominator as a sentinel "
            "id no unit carries: never retrieved by any system; more than the ceiling share of "
            "the facts is DATA_STOP"
        ),
        "context": {
            "articles_per_query": musique._counts(articles_per_query),
            "queries_with_two_facts_from_one_article": len(same_article_qids),
            "queries_with_two_facts_from_one_article_qids": same_article_qids,
            "queries_gold_in_one_article": len(one_article_qids),
            "queries_gold_in_one_article_qids": one_article_qids,
        },
        "answer_rule": "the answer field is never copied or read",
        "terminal_state": phase9.DATA_STOP if len(unmapped) > share * facts else None,
        "questions": entries,
    }
    return questions, body


def _type_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    return digest_of(*(f"{e['qid']}\t{e['question_type']}" for e in entries))


# --- The live-path questions (D4) ----------------------------------------------------------


def live_path_block(queries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every `null` query (no evidence), in file order, with its question text only: D4 runs
    the live retrievers on them and computes no metric."""
    chosen = [
        (qid_of(position), str(query["query"]))
        for position, query in enumerate(queries)
        if not is_answerable(query)
    ]
    return {
        "split": LIVE_PATH_SPLIT,
        "rule": (
            "every MultiHop-RAG query with an empty evidence_list, in file order; question text "
            "only, no gold, no answer; no metric of them is computed (D4)"
        ),
        "n_questions": len(chosen),
        "question_digest": musique.question_digest(
            [musique._live_question(qid, text, LIVE_PATH_SPLIT) for qid, text in chosen]
        ),
        "questions": [{"qid": qid, "question": text} for qid, text in chosen],
    }


def live_path_questions(body: Mapping[str, Any]) -> list[Question]:
    """The live-path questions of a loaded `questions.json`, their digest recomputed."""
    try:
        return musique.live_path_questions(body)
    except musique.MusiqueError as error:
        raise MultiHopRagError(str(error)) from error


def write_questions(
    directory: Path | str, body: Mapping[str, Any], *, live_path: Mapping[str, Any]
) -> Path:
    """Freeze the answerable queries and the live-path block once."""
    try:
        return musique.write_questions(directory, body, live_path=live_path, phase=16)
    except musique.MusiqueError as error:
        raise MultiHopRagError(str(error)) from error


def load_questions(
    directory: Path | str, *, corpus_unit_set_hash: str
) -> tuple[list[Question], dict[str, Any]]:
    """The frozen answerable queries, every digest recomputed (question type and straddling
    list included), refused on a `DATA_STOP` or another corpus. `question_type` is in the
    body's entries; the live-path questions come from `live_path_questions`."""
    target = Path(directory) / QUESTIONS_FILENAME
    stored: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    if stored["terminal_state"] is not None:
        raise phase9.DataStop(
            f"{stored['unmapped_facts']} of {stored['facts']} facts are unmapped, past the "
            f"share of {stored['unmapped_ceiling_share']}; no retrieval score is published"
        )
    try:
        questions, body = musique.load_questions(
            directory, corpus_unit_set_hash=corpus_unit_set_hash
        )
    except musique.MusiqueError as error:
        raise MultiHopRagError(str(error)) from error
    straddling = digest_of(*(json.dumps(e, sort_keys=True) for e in body["straddling"]))
    if (
        _type_digest(body["questions"]) != body["question_type_digest"]
        or straddling != body["straddling_digest"]
    ):
        raise MultiHopRagError(f"{target.name} does not match its recorded digests")
    return questions, body
