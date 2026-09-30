"""Phase 16, S1: the MultiHop-RAG corpus, its gold and the live path, on hand-built articles.

What can change a Phase 16 result here is the corpus and the gold: which newline pieces of an
article body become units (D1: every non-empty one, boilerplate included, a verbatim repeat
inside an article collapsing into one unit), which unit each evidence fact maps to (D2: the
paragraph holding it, or for a fact that straddles a break the paragraph holding its first
line), the sentinel that keeps an unmapped fact in the denominator, the 1 % `DATA_STOP`
counted over facts, and that a `null` query is never measured. Every file is written in
`tmp_path`; nothing reaches the network or `data/`.
"""

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from concept_embeddings_rag.corpus import fullwiki as fullwiki_corpus
from concept_embeddings_rag.corpus import multihop_rag
from concept_embeddings_rag.corpus.pool import unit_id_for
from concept_embeddings_rag.evaluation import fullwiki as phase9

ADVERTISEMENT = "Advertisement"


def article(title: str, body: str, index: int = 0) -> dict[str, Any]:
    return {
        "title": title,
        "author": "someone",
        "source": "Some Paper",
        "published_at": "2023-10-01T00:00:00+00:00",
        "category": "sports",
        "url": f"https://example.org/{index}",
        "body": body,
    }


def evidence(source: dict[str, Any], fact: str) -> dict[str, Any]:
    return {
        "title": source["title"],
        "url": source["url"],
        "source": source["source"],
        "category": source["category"],
        "published_at": source["published_at"],
        "author": source["author"],
        "fact": fact,
    }


def query(text: str, pieces: list[dict[str, Any]], kind: str = "comparison_query") -> dict:
    return {
        "query": text,
        "answer": f"answer to {text}",
        "question_type": kind if pieces else "null_query",
        "evidence_list": pieces,
    }


def uid(title: str, text: str) -> str:
    return unit_id_for(title, (text,))


ALPHA_BODY = (
    "Alpha won the cup on Sunday. The coach praised the defence.\n"
    "\n"
    "   \n"
    f"{ADVERTISEMENT}\n"
    "The final score was 3-1 after extra time\n"
    "and the fans stayed until midnight.\n"
    f"{ADVERTISEMENT}\n"
    "  Beta lost the semi-final.  "
)
GAMMA_BODY = f"Gamma opened a new stadium.\n{ADVERTISEMENT}\nIt seats forty thousand people."


def articles() -> list[dict[str, Any]]:
    return [
        {**article("Alpha wins the cup", ALPHA_BODY, 0)},
        {**article("Gamma opens a stadium", GAMMA_BODY, 1)},
    ]


def corpus_of(tmp_path: Path) -> tuple[list[Any], dict[str, Any]]:
    source = articles()
    units, article_of = multihop_rag.pool_units(source)
    manifest = multihop_rag.write_corpus(
        units,
        tmp_path,
        articles=source,
        article_of=article_of,
        paragraphs_read=multihop_rag.corpus_profile(source)["newline_paragraphs"],
    )
    return units, manifest


# --- Splitting (D1) ------------------------------------------------------------------------


def test_empty_and_whitespace_only_pieces_are_dropped_and_outer_whitespace_stripped():
    paragraphs = multihop_rag.paragraphs_of(ALPHA_BODY)

    assert [p.text for p in paragraphs] == [
        "Alpha won the cup on Sunday. The coach praised the defence.",
        ADVERTISEMENT,
        "The final score was 3-1 after extra time",
        "and the fans stayed until midnight.",
        ADVERTISEMENT,
        "Beta lost the semi-final.",
    ]
    assert [p.position for p in paragraphs] == list(range(6))
    for p in paragraphs:
        assert ALPHA_BODY[p.start : p.end] == p.text


def test_a_repeat_inside_one_article_is_one_unit_and_across_articles_two():
    units, article_of = multihop_rag.pool_units(articles())

    ids = {u.unit_id for u in units}
    assert uid("Alpha wins the cup", ADVERTISEMENT) in ids
    assert uid("Gamma opens a stadium", ADVERTISEMENT) in ids
    assert len(units) == 8  # 6 + 3 newline paragraphs, one repeat collapsed
    assert [u.unit_id for u in units] == sorted(ids)
    assert article_of[uid("Gamma opens a stadium", ADVERTISEMENT)] == 1
    assert article_of[uid("Alpha wins the cup", ADVERTISEMENT)] == 0


def test_the_profile_counts_newline_paragraphs_boilerplate_and_long_paragraphs():
    long_text = " ".join(["word"] * 385)
    source = [*articles(), article("Delta", f"Short one.\n{long_text}", 2)]

    profile = multihop_rag.corpus_profile(source, boilerplate_max_words=5, window_words=384)

    assert profile["articles"] == 3
    assert profile["newline_paragraphs"] == 11
    # Advertisement x3, "Beta lost the semi-final.", "Gamma opened a new stadium.",
    # "It seats forty thousand people." and "Short one."; six words is not boilerplate.
    assert profile["boilerplate_paragraphs"] == 7
    assert profile["over_window_paragraphs"] == 1
    assert profile["over_window"] == [
        {"article": 2, "position": 1, "unit_id": uid("Delta", long_text), "words": 385}
    ]


def test_the_corpus_keeps_article_metadata_out_of_indexable_text_and_reads_back(
    tmp_path: Path,
):
    units, manifest = corpus_of(tmp_path)

    assert manifest["n_units"] == 8
    assert manifest["paragraphs_read"] == 9
    assert manifest["phase"] == 16
    assert manifest["source"] == multihop_rag.source_pin()
    for unit in units:
        assert "example.org" not in unit.indexable_text
        assert "sports" not in unit.indexable_text
    assert units[0].indexable_text == f"{units[0].title}. {units[0].text}"

    assert fullwiki_corpus.load_corpus(tmp_path) == units
    by_unit = multihop_rag.load_articles(tmp_path)
    alpha = uid("Alpha wins the cup", ADVERTISEMENT)
    assert by_unit[alpha] == {
        "index": 0,
        "url": "https://example.org/0",
        "source": "Some Paper",
        "category": "sports",
        "published_at": "2023-10-01T00:00:00+00:00",
    }


def test_an_edited_corpus_line_is_refused(tmp_path: Path):
    corpus_of(tmp_path)
    path = tmp_path / fullwiki_corpus.CORPUS_NAME
    lines = gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()
    entry = json.loads(lines[0])
    entry["sentences"] = ["edited"]
    lines[0] = json.dumps(entry)
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))

    with pytest.raises(fullwiki_corpus.FullWikiError, match="modified"):
        fullwiki_corpus.load_corpus(tmp_path)


def test_edited_article_metadata_is_refused(tmp_path: Path):
    corpus_of(tmp_path)
    path = tmp_path / fullwiki_corpus.CORPUS_NAME
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    path.write_bytes(gzip.compress(text.replace("sports", "health", 1).encode("utf-8")))

    with pytest.raises(multihop_rag.MultiHopRagError, match="metadata"):
        multihop_rag.load_articles(tmp_path)


def test_the_corpus_is_written_once(tmp_path: Path):
    corpus_of(tmp_path)

    with pytest.raises(multihop_rag.MultiHopRagError, match="already"):
        corpus_of(tmp_path)


# --- Gold (D2) -----------------------------------------------------------------------------


def questions_of(
    queries: list[dict[str, Any]], tmp_path: Path, share: float = 0.01
) -> tuple[list[Any], dict[str, Any]]:
    source = articles()
    units, manifest = corpus_of(tmp_path)
    return multihop_rag.answerable_questions(queries, source, units, corpus=manifest, share=share)


def test_facts_map_to_their_paragraphs_and_a_straddling_fact_to_its_first_line(
    tmp_path: Path,
):
    alpha, gamma = articles()
    straddling = "The final score was 3-1 after extra time\nand the fans stayed"
    queries = [
        query(
            "Who won?",
            [
                evidence(alpha, "The coach praised the defence."),
                evidence(gamma, "It seats forty thousand people."),
                evidence(alpha, straddling),
            ],
            "inference_query",
        )
    ]

    questions, body = questions_of(queries, tmp_path)

    (question,) = questions
    assert question.qid == "mhr-0000"
    assert question.gold_unit_ids == (
        uid(alpha["title"], "Alpha won the cup on Sunday. The coach praised the defence."),
        uid(gamma["title"], "It seats forty thousand people."),
        uid(alpha["title"], "The final score was 3-1 after extra time"),
    )
    assert question.supporting_facts == (
        (alpha["title"], 0),
        (gamma["title"], 2),
        (alpha["title"], 2),
    )
    assert question.answer == ""
    assert body["facts"] == 3
    assert body["facts_inside"] == 2
    (entry,) = body["straddling"]
    assert entry["qid"] == "mhr-0000"
    assert entry["evidence"] == 2
    assert entry["article"] == 0
    assert entry["unit_id"] == uid(alpha["title"], "The final score was 3-1 after extra time")
    assert entry["paragraphs_crossed"] == 2
    assert body["gold_counts"] == {"3": 1}
    assert body["type_counts"] == {"inference_query": 1}
    assert body["questions"][0]["question_type"] == "inference_query"
    assert body["terminal_state"] is None
    assert "answer to" not in json.dumps(body)


def test_two_facts_in_one_paragraph_count_once(tmp_path: Path):
    alpha, gamma = articles()
    queries = [
        query(
            "Q?",
            [
                evidence(alpha, "Alpha won the cup on Sunday."),
                evidence(alpha, "The coach praised the defence."),
                evidence(gamma, "Gamma opened a new stadium."),
            ],
        )
    ]

    questions, body = questions_of(queries, tmp_path)

    assert len(questions[0].gold_unit_ids) == 2
    assert body["gold_counts"] == {"2": 1}
    assert body["evidence_counts"] == {"3": 1}
    assert body["distinct_gold_units"] == 2
    assert body["context"]["queries_with_two_facts_from_one_article"] == 1


def test_a_missing_fact_and_an_unknown_article_become_sentinels_in_the_denominator(
    tmp_path: Path,
):
    alpha, gamma = articles()
    unknown = article("No such article", "Nothing here.", 9)
    queries = [
        query(
            "Q?",
            [
                evidence(alpha, "Alpha won the cup on Sunday."),
                evidence(gamma, "This sentence is not in the body."),
                evidence(unknown, "Nothing here."),
            ],
        )
    ]

    questions, body = questions_of(queries, tmp_path, share=1.0)

    gold = questions[0].gold_unit_ids
    assert gold[1:] == (phase9._sentinel("mhr-0000", 1), phase9._sentinel("mhr-0000", 2))
    assert body["unmapped_facts"] == 2
    assert [(e["evidence"], e["reason"]) for e in body["unmapped"]] == [
        (1, "fact not in the article body"),
        (2, "no article with this title"),
    ]
    assert body["distinct_gold_units"] == 1
    assert body["terminal_state"] is None


@pytest.mark.parametrize(("facts", "stop"), [(100, False), (99, True)])
def test_data_stop_counts_facts_and_starts_just_past_one_percent(
    facts: int, stop: bool, tmp_path: Path
):
    alpha, gamma = articles()
    good = [evidence(alpha, "Alpha won the cup"), evidence(gamma, "Gamma opened")]
    queries = [query(f"Q{i}?", good) for i in range(49)]
    lost = [evidence(alpha, "Not in any body.")]
    if facts % 2 == 0:
        lost.append(evidence(gamma, "Gamma opened"))
    queries.append(query("lost?", lost))
    assert sum(len(q["evidence_list"]) for q in queries) == facts

    _questions, body = questions_of(queries, tmp_path)

    assert body["unmapped_facts"] == 1
    assert body["terminal_state"] == (phase9.DATA_STOP if stop else None)


# --- Null queries (D2, D4) -----------------------------------------------------------------


def test_a_null_query_is_never_measured_and_forms_the_live_path(tmp_path: Path):
    alpha, gamma = articles()
    queries = [
        query("Answerable?", [evidence(alpha, "Alpha won"), evidence(gamma, "Gamma opened")]),
        query("Unanswerable?", []),
    ]

    questions, body = questions_of(queries, tmp_path)
    block = multihop_rag.live_path_block(queries)

    assert [q.qid for q in questions] == ["mhr-0000"]
    assert body["n_questions"] == 1
    assert block["n_questions"] == 1
    assert block["questions"] == [{"qid": "mhr-0001", "question": "Unanswerable?"}]
    multihop_rag.write_questions(tmp_path, body, live_path=block)
    manifest = json.loads((tmp_path / "corpus.json").read_text(encoding="utf-8"))
    loaded, loaded_body = multihop_rag.load_questions(
        tmp_path, corpus_unit_set_hash=manifest["unit_set_hash"]
    )
    assert loaded == questions
    live = multihop_rag.live_path_questions(loaded_body)
    assert [(q.qid, q.question) for q in live] == [("mhr-0001", "Unanswerable?")]
    assert all(q.gold_unit_ids == () and q.answer == "" for q in live)
    assert {q.split for q in live} == {multihop_rag.LIVE_PATH_SPLIT}
    assert {q.split for q in loaded} == {multihop_rag.ANSWERABLE_SPLIT}


def test_the_loader_refuses_a_data_stop_an_edited_type_and_another_corpus(tmp_path: Path):
    alpha, gamma = articles()
    queries = [query("Q?", [evidence(alpha, "Alpha won"), evidence(gamma, "Gamma opened")])]
    _questions, body = questions_of(queries, tmp_path)
    manifest = json.loads((tmp_path / "corpus.json").read_text(encoding="utf-8"))
    block = multihop_rag.live_path_block(queries)
    path = multihop_rag.write_questions(tmp_path, body, live_path=block)

    with pytest.raises(multihop_rag.MultiHopRagError, match="corpus"):
        multihop_rag.load_questions(tmp_path, corpus_unit_set_hash="0" * 16)

    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["questions"][0]["question_type"] = "temporal_query"
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(multihop_rag.MultiHopRagError, match="digest"):
        multihop_rag.load_questions(tmp_path, corpus_unit_set_hash=manifest["unit_set_hash"])

    edited["terminal_state"] = phase9.DATA_STOP
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(phase9.DataStop):
        multihop_rag.load_questions(tmp_path, corpus_unit_set_hash=manifest["unit_set_hash"])

    with pytest.raises(multihop_rag.MultiHopRagError, match="already"):
        multihop_rag.write_questions(tmp_path, body, live_path=block)
