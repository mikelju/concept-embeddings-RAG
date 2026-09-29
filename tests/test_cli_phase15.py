"""Phase 15 CLI, S2: `p15-data` turns the two pinned MuSiQue-Ans files into the corpus, the
validation questions with their gold, the live-path set and the token counts.

What can change a Phase 15 result here: which bytes are parsed (the pin), which units and gold
come out of them, what the live path carries (question text only), and the two stops, unmapped
gold (D2) and units longer than GLiNER's window (author decision 3). The source is a small
synthetic pair of JSON lines files handed to the stage by an injected fetcher, with the pins
replaced by theirs; the budget tokenizer is a stub. Nothing reaches the network or `data/`.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from concept_embeddings_rag import cli
from concept_embeddings_rag.corpus import fullwiki as fullwiki_corpus
from concept_embeddings_rag.corpus import musique
from concept_embeddings_rag.corpus.pool import IndexingUnit, unit_id_for
from concept_embeddings_rag.evaluation import fullwiki as phase9

# A gold paragraph and a near-copy under the same title: equal first 80 characters,
# different ending, so two units (author decision 5 counts the gold one).
LONG_TEXT = "Paris is the capital and most populous city of France, with an estimated population"
assert len(LONG_TEXT) > 80
NEAR_COPY = LONG_TEXT[:80] + " of two million residents in a later census."


def paragraph(idx: int, title: str, text: str, supporting: bool = False) -> dict[str, Any]:
    return {"idx": idx, "title": title, "paragraph_text": text, "is_supporting": supporting}


def row(qid: str, paragraphs: list[dict[str, Any]], answerable: bool = True) -> dict[str, Any]:
    return {
        "id": qid,
        "question": f"question {qid}?",
        "question_decomposition": [{"id": 1, "question": "never read"}],
        "answer": f"answer {qid}",
        "answer_aliases": ["alias"],
        "paragraphs": paragraphs,
        "answerable": answerable,
    }


def validation_rows() -> list[dict[str, Any]]:
    return [
        row(
            "2hop__v1",
            [
                paragraph(0, "Paris", LONG_TEXT, True),
                paragraph(1, "Lyon", "Lyon is a city.", True),
                paragraph(2, "Nice", "Nice is a city."),
            ],
        ),
        row(
            "3hop__v2",
            [
                paragraph(0, "A", "a text", True),
                paragraph(1, "B", "b text", True),
                paragraph(2, "C", "c text", True),
                paragraph(3, "Lyon", "Lyon is a city."),
            ],
        ),
        row(
            "2hop__v3",
            [paragraph(0, "D", "d text", True), paragraph(1, "E", "e text", True)],
        ),
    ]


def train_rows() -> list[dict[str, Any]]:
    # Deliberately out of id order: the live path is the first qids *sorted by id*.
    return [
        row("2hop__t3", [paragraph(0, "Paris", NEAR_COPY, True), paragraph(1, "F", "f", True)]),
        row("2hop__t1", [paragraph(0, "Nice", "Nice is a city.", True), paragraph(1, "G", "g")]),
        row("2hop__t2", [paragraph(0, "H", "h", True), paragraph(1, "A", "a text", True)]),
    ]


def jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")


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


def source(
    monkeypatch: pytest.MonkeyPatch,
    train: list[dict[str, Any]] | None = None,
    validation: list[dict[str, Any]] | None = None,
) -> Fetcher:
    """Pin the synthetic files in place of the real pins and return their fetcher."""
    payloads = {
        musique.TRAIN_FILE: jsonl(train_rows() if train is None else train),
        musique.VALIDATION_FILE: jsonl(validation_rows() if validation is None else validation),
    }
    monkeypatch.setattr(
        musique,
        "SOURCE_FILES",
        {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in payloads.items()
        },
    )
    return Fetcher({musique.source_url(name): payload for name, payload in payloads.items()})


def run(target: Path, fetcher: Fetcher, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "fetcher": fetcher,
        "counter": StubCounter(),
        "train_rows": 3,
        "validation_rows": 3,
        "supporting_counts": {2: 2, 3: 1},
        "live_path_size": 2,
    }
    arguments.update(overrides)
    return cli.cmd_p15_data(target, **arguments)


def uid(title: str, text: str) -> str:
    return unit_id_for(title, (text,))


def never_parse(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("no line may be parsed")


# --- The stage writes the corpus, the gold, the live path and the token counts ----------


def test_the_stage_writes_the_pooled_units_the_gold_and_the_live_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fetcher = source(monkeypatch)

    run(tmp_path, fetcher)

    units = fullwiki_corpus.load_corpus(tmp_path)
    corpus = json.loads((tmp_path / "corpus.json").read_text(encoding="utf-8"))
    every = [p for r in validation_rows() + train_rows() for p in r["paragraphs"]]
    assert {u.unit_id for u in units} == {uid(p["title"], p["paragraph_text"]) for p in every}
    assert corpus["n_units"] == len(units) == 12
    assert corpus["paragraphs_read"] == len(every) == 15
    assert corpus["source"] == musique.source_pin()
    assert (tmp_path / "source" / musique.TRAIN_FILE).exists()

    questions, body = musique.load_questions(tmp_path, corpus_unit_set_hash=corpus["unit_set_hash"])
    assert {q.qid: q.gold_unit_ids for q in questions} == {
        "2hop__v1": (uid("Paris", LONG_TEXT), uid("Lyon", "Lyon is a city.")),
        "3hop__v2": (uid("A", "a text"), uid("B", "b text"), uid("C", "c text")),
        "2hop__v3": (uid("D", "d text"), uid("E", "e text")),
    }
    assert body["supporting_counts"] == {"2": 2, "3": 1}
    assert body["terminal_state"] is None
    assert body["questions_with_unmapped_gold"] == 0

    # D4: the first train qids sorted by id, question text only; no gold, no answer.
    block = body["live_path"]
    assert [entry["qid"] for entry in block["questions"]] == ["2hop__t1", "2hop__t2"]
    assert all(set(entry) == {"qid", "question"} for entry in block["questions"])
    live = musique.live_path_questions(body)
    assert [q.question for q in live] == ["question 2hop__t1?", "question 2hop__t2?"]
    assert all(q.gold_unit_ids == () and q.answer == "" for q in live)
    assert {q.split for q in live} == {musique.LIVE_PATH_SPLIT} != {musique.VALIDATION_SPLIT}
    text = (tmp_path / "questions.json").read_text(encoding="utf-8")
    assert "answer 2hop__t" not in text and "never read" not in text

    counts = phase9.load_phase9_token_counts(
        tmp_path,
        unit_set_hash=corpus["unit_set_hash"],
        unit_ids=[u.unit_id for u in units],
    )
    assert counts == StubCounter().count_units(units)


def test_near_duplicate_gold_and_long_units_are_recorded_in_the_corpus_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fetcher = source(monkeypatch)

    run(tmp_path, fetcher)

    corpus = json.loads((tmp_path / "corpus.json").read_text(encoding="utf-8"))
    near = corpus["near_duplicate_gold"]
    assert near["prefix_chars"] == 80
    assert near["gold_units"] == 7
    assert near["gold_units_with_near_duplicate"] == 1
    assert near["unit_ids"] == [uid("Paris", LONG_TEXT)]
    assert near["qids"] == ["2hop__v1"]
    window = corpus["gliner_window"]
    assert window["max_len"] == 384
    assert window["units_over"] == 0
    assert window["stop"] is False


# --- The pin: refused before a line is parsed --------------------------------------------


@pytest.mark.parametrize("field", ["sha256", "bytes"])
def test_a_wrong_sha256_or_byte_count_is_refused_before_any_line_is_parsed(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fetcher = source(monkeypatch)
    pins = {name: dict(pin) for name, pin in musique.SOURCE_FILES.items()}
    pin = pins[musique.VALIDATION_FILE]
    pin[field] = "0" * 64 if field == "sha256" else pin["bytes"] + 1
    monkeypatch.setattr(musique, "SOURCE_FILES", pins)
    monkeypatch.setattr(musique, "read_rows", never_parse)

    with pytest.raises(SystemExit, match=musique.VALIDATION_FILE if field == "bytes" else "hash"):
        run(tmp_path, fetcher)

    assert not (tmp_path / "corpus.json").exists()
    assert not (tmp_path / "questions.json").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"train_rows": 4}, "train"),
        ({"supporting_counts": {2: 1, 3: 2}}, "supporting"),
    ],
)
def test_a_source_whose_counts_differ_from_the_pins_is_refused_before_anything_is_written(
    change: dict[str, Any], message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fetcher = source(monkeypatch)

    with pytest.raises(SystemExit, match=message):
        run(tmp_path, fetcher, **change)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["source"]


def test_an_unanswerable_row_or_a_repeated_qid_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    train = train_rows()
    train[0] = {**train[0], "answerable": False}
    train[1] = {**train[1], "id": train[2]["id"]}
    fetcher = source(monkeypatch, train=train)

    with pytest.raises(SystemExit, match="answerable") as raised:
        run(tmp_path, fetcher)

    assert "unique" in str(raised.value)
    assert not (tmp_path / "corpus.json").exists()


# --- Written once ------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["corpus.json", "questions.json"])
def test_a_second_run_over_an_existing_corpus_or_questions_file_is_refused(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fetcher = source(monkeypatch)
    (tmp_path / name).write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="already"):
        run(tmp_path, fetcher)

    assert fetcher.calls == []
    assert sorted(p.name for p in tmp_path.iterdir()) == [name]


def test_the_stage_refuses_to_run_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fetcher = source(monkeypatch)
    run(tmp_path, fetcher)
    before = (tmp_path / "questions.json").read_bytes()

    with pytest.raises(SystemExit, match="already"):
        run(tmp_path, fetcher)

    assert (tmp_path / "questions.json").read_bytes() == before


# --- The stops ---------------------------------------------------------------------------


def test_unmapped_gold_past_the_share_writes_data_stop_and_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fetcher = source(monkeypatch)
    pool = musique.pool_units
    missing = uid("D", "d text")
    monkeypatch.setattr(
        musique, "pool_units", lambda rows: [u for u in pool(rows) if u.unit_id != missing]
    )

    with pytest.raises(SystemExit, match=phase9.DATA_STOP):
        run(tmp_path, fetcher)

    body = json.loads((tmp_path / "questions.json").read_text(encoding="utf-8"))
    assert body["terminal_state"] == phase9.DATA_STOP
    assert body["questions_with_unmapped_gold"] == 1
    assert [entry["qid"] for entry in body["unmapped"]] == ["2hop__v3"]
    assert not (tmp_path / phase9.TOKENS_MANIFEST).exists()


def test_units_past_the_gliner_window_share_stop_with_the_evidence_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    train = train_rows()
    train[2]["paragraphs"][0]["paragraph_text"] = " ".join(["word"] * 400)
    fetcher = source(monkeypatch, train=train)

    with pytest.raises(SystemExit, match="GLiNER"):
        run(tmp_path, fetcher)

    corpus = json.loads((tmp_path / "corpus.json").read_text(encoding="utf-8"))
    window = corpus["gliner_window"]
    assert window["units_over"] == 1
    assert window["stop"] is True
    assert window["over"] == [{"unit_id": uid("H", " ".join(["word"] * 400)), "words": 402}]
    assert not (tmp_path / phase9.TOKENS_MANIFEST).exists()


@pytest.mark.parametrize(("n", "over"), [(382, 0), (383, 1)])
def test_gliner_words_are_counted_with_its_own_splitter_at_the_window_edge(n: int, over: int):
    # "T" and "." are two of GLiNER's words, so the unit holds n + 2 of them.
    unit = IndexingUnit(unit_id="u", title="T", sentences=(" ".join(["w"] * n),))

    window = musique.units_over_window(
        [unit], cli._p15_gliner_words(), max_len=384, share=0.01, splitter="whitespace"
    )

    assert window["units_over"] == over


def test_the_loader_refuses_an_edited_live_path_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fetcher = source(monkeypatch)
    run(tmp_path, fetcher)
    path = tmp_path / "questions.json"
    corpus = json.loads((tmp_path / "corpus.json").read_text(encoding="utf-8"))
    body = json.loads(path.read_text(encoding="utf-8"))
    body["live_path"]["questions"][0]["question"] = "another question?"
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(musique.MusiqueError, match="live"):
        musique.load_questions(tmp_path, corpus_unit_set_hash=corpus["unit_set_hash"])
