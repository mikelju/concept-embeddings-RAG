"""Phase 16 CLI, S2: `p16-data` turns the two pinned MultiHop-RAG files into the corpus, the
answerable queries with their gold, the live-path set and the token counts.

What can change a Phase 16 result here: which bytes are parsed (the pin), which units and gold
come out of them, what the live path carries (the `null` queries, text only), and the check of
every count against the measurement of 2026-09-30, which stops the stage before anything is
written. The source is a small synthetic pair of JSON files handed to the stage by an injected
fetcher, with the pins and the expected counts replaced by theirs; the budget tokenizer is a
stub. Nothing reaches the network or `data/`.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from concept_embeddings_rag import cli
from concept_embeddings_rag.corpus import fullwiki as fullwiki_corpus
from concept_embeddings_rag.corpus import multihop_rag
from concept_embeddings_rag.corpus.pool import IndexingUnit, unit_id_for
from concept_embeddings_rag.evaluation import fullwiki as phase9

ALPHA = "Alpha wins the cup"
GAMMA = "Gamma opens a stadium"
ALPHA_BODY = (
    "Alpha won the cup on Sunday after a long and tense final.\n\n"
    "Advertisement\n"
    "The final score was 3-1 after extra time\n"
    "and the fans stayed until midnight in the square.\n"
    "Advertisement"
)
GAMMA_BODY = "Gamma opened a new stadium in the north of the city.\nIt seats forty thousand people."


def article(title: str, body: str, index: int) -> dict[str, Any]:
    return {
        "title": title,
        "author": "someone",
        "source": "Some Paper",
        "published_at": "2023-10-01T00:00:00+00:00",
        "category": "sports",
        "url": f"https://example.org/{index}",
        "body": body,
    }


def articles() -> list[dict[str, Any]]:
    return [article(ALPHA, ALPHA_BODY, 0), article(GAMMA, GAMMA_BODY, 1)]


def evidence(index: int, fact: str) -> dict[str, Any]:
    source = articles()[index]
    return {k: source[k] for k in ("title", "url", "source", "category", "published_at")} | {
        "author": source["author"],
        "fact": fact,
    }


def queries() -> list[dict[str, Any]]:
    return [
        {
            "query": "Who won and where do they play?",
            "answer": "secret answer one",
            "question_type": "comparison_query",
            "evidence_list": [
                evidence(0, "Alpha won the cup on Sunday"),
                evidence(1, "It seats forty thousand people."),
            ],
        },
        {
            "query": "Nobody knows?",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
        {
            "query": "What was the score?",
            "answer": "secret answer two",
            "question_type": "inference_query",
            "evidence_list": [
                evidence(0, "extra time\nand the fans stayed"),
                evidence(0, "Alpha won the cup"),
                evidence(1, "Gamma opened a new stadium"),
            ],
        },
    ]


EXPECTED: dict[str, Any] = {
    "articles": 2,
    "newline_paragraphs": 7,
    "boilerplate_paragraphs": 3,  # Advertisement x2, "It seats forty thousand people."
    "over_window_paragraphs": 0,
    "queries": 3,
    "null_queries": 1,
    "answerable_queries": 2,
    "type_counts": {"comparison_query": 1, "inference_query": 1},
    "facts": 5,
    "facts_straddling": 1,
    "gold_counts": {2: 1, 3: 1},
    "distinct_gold_units": 4,
    "same_article_fact_queries": 1,
}


class StubCounter:
    """The budget ruler's interface, without its tokenizer (which would need the network)."""

    def count_units(self, units: list[IndexingUnit]) -> dict[str, int]:
        return {unit.unit_id: len(unit.indexable_text.split()) for unit in units}


class Fetcher:
    """Serves the synthetic files by URL and logs every request."""

    def __init__(self, by_url: dict[str, bytes]) -> None:
        self.by_url = by_url
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        return self.by_url[url]


def source(monkeypatch: pytest.MonkeyPatch) -> Fetcher:
    """Pin the synthetic files in place of the real pins and return their fetcher."""
    payloads = {
        multihop_rag.CORPUS_FILE: json.dumps(articles()).encode("utf-8"),
        multihop_rag.QUERIES_FILE: json.dumps(queries()).encode("utf-8"),
    }
    monkeypatch.setattr(
        multihop_rag,
        "SOURCE_FILES",
        {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in payloads.items()
        },
    )
    return Fetcher({multihop_rag.source_url(name): p for name, p in payloads.items()})


def run(target: Path, fetcher: Fetcher, **overrides: Any) -> dict[str, Any]:
    expected = {**EXPECTED, **overrides}
    return cli.cmd_p16_data(target, fetcher=fetcher, counter=StubCounter(), expected=expected)


def uid(title: str, text: str) -> str:
    return unit_id_for(title, (text,))


def test_the_stage_writes_the_units_the_gold_and_the_live_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run(tmp_path, source(monkeypatch))

    units = fullwiki_corpus.load_corpus(tmp_path)
    corpus = json.loads((tmp_path / "corpus.json").read_text(encoding="utf-8"))
    assert corpus["n_units"] == len(units) == 6  # Advertisement twice in one article
    assert corpus["paragraphs_read"] == 7
    assert corpus["phase"] == 16
    assert corpus["source"] == multihop_rag.source_pin()
    assert corpus["corpus_profile"]["boilerplate_paragraphs"] == 3
    assert corpus["gliner_window"]["units_over"] == 0
    assert (tmp_path / "source" / multihop_rag.CORPUS_FILE).exists()
    assert (
        multihop_rag.load_articles(tmp_path)[uid(GAMMA, "It seats forty thousand people.")]["index"]
        == 1
    )

    questions, body = multihop_rag.load_questions(
        tmp_path, corpus_unit_set_hash=corpus["unit_set_hash"]
    )
    first_alpha = uid(ALPHA, "Alpha won the cup on Sunday after a long and tense final.")
    assert {q.qid: q.gold_unit_ids for q in questions} == {
        "mhr-0000": (first_alpha, uid(GAMMA, "It seats forty thousand people.")),
        "mhr-0002": (
            uid(ALPHA, "The final score was 3-1 after extra time"),
            first_alpha,
            uid(GAMMA, "Gamma opened a new stadium in the north of the city."),
        ),
    }
    assert [e["qid"] for e in body["straddling"]] == ["mhr-0002"]
    assert body["terminal_state"] is None
    assert body["context"]["queries_with_two_facts_from_one_article"] == 1

    live = multihop_rag.live_path_questions(body)
    assert [(q.qid, q.question) for q in live] == [("mhr-0001", "Nobody knows?")]
    text = (tmp_path / "questions.json").read_text(encoding="utf-8")
    assert "secret answer" not in text and "Insufficient" not in text

    counts = phase9.load_phase9_token_counts(
        tmp_path, unit_set_hash=corpus["unit_set_hash"], unit_ids=[u.unit_id for u in units]
    )
    assert counts == StubCounter().count_units(units)


@pytest.mark.parametrize("field", ["sha256", "bytes"])
def test_a_wrong_sha256_or_byte_count_is_refused_before_parsing(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fetcher = source(monkeypatch)
    pins = {name: dict(pin) for name, pin in multihop_rag.SOURCE_FILES.items()}
    pin = pins[multihop_rag.QUERIES_FILE]
    pin[field] = "0" * 64 if field == "sha256" else pin["bytes"] + 1
    monkeypatch.setattr(multihop_rag, "SOURCE_FILES", pins)

    def never_parse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may be parsed")

    monkeypatch.setattr(multihop_rag, "read_source", never_parse)

    with pytest.raises(SystemExit, match="MultiHopRAG.json"):
        run(tmp_path, fetcher)

    assert not (tmp_path / "corpus.json").exists()
    assert not (tmp_path / "questions.json").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"newline_paragraphs": 8}, "newline_paragraphs"),
        ({"gold_counts": {2: 2}}, "gold_counts"),
        ({"facts_straddling": 0}, "facts_straddling"),
    ],
)
def test_a_count_other_than_the_measured_one_stops_before_anything_is_written(
    change: dict[str, Any], message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with pytest.raises(SystemExit, match=message):
        run(tmp_path, source(monkeypatch), **change)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["source"]


def test_the_stage_refuses_a_second_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fetcher = source(monkeypatch)
    run(tmp_path, fetcher)
    before = (tmp_path / "questions.json").read_bytes()
    calls = len(fetcher.calls)

    with pytest.raises(SystemExit, match="already"):
        run(tmp_path, fetcher)

    assert (tmp_path / "questions.json").read_bytes() == before
    assert len(fetcher.calls) == calls


def test_the_p16_data_stage_is_registered():
    parser = cli.build_parser()
    assert parser.parse_args(["p16-data"]).command == "p16-data"
