"""Phase 15 CLI, S2: `p15-data` turns the two pinned MuSiQue-Ans files into the corpus, the
validation questions with their gold, the live-path set and the token counts. S3:
`p15-build` builds BM25, the vectors, the GLiNER records and the entity index over it.

What can change a Phase 15 result here: which bytes are parsed (the pin), which units and gold
come out of them, what the live path carries (question text only), and the two stops, unmapped
gold (D2) and units longer than GLiNER's window (author decision 3). The source is a small
synthetic pair of JSON lines files handed to the stage by an injected fetcher, with the pins
replaced by theirs; the budget tokenizer is a stub. Nothing reaches the network or `data/`.

For S3: which extractor configuration may write Phase 15 records (the Phase 9 pod digest, D4),
that the Phase 9 extraction path is unchanged by its generalization, and that every artifact is
written under the Phase 15 directory and names the Phase 15 corpus. The encoder and GLiNER are
stubs.

For S4: the live systems are built from the verified Phase 15 inputs built by the S3 stages
over the same synthetic corpus, `p15-integrity` writes `integrity.json` once, a miss is a
written `DATA_STOP`, and no metric of a live-path question is computed or stored. The HotpotQA
code-identity half reads the real Phase 9-14 artifacts and is replaced by a stub here.

For S5: `p15-eval` refuses without the flag, a committed passing `integrity.json`, committed
fits, or after a first start; over the synthetic corpus (its gold, not MuSiQue's) the stored
component lists re-fuse to the stored fused list, whose Full Support is the outcome record's,
and the hop lists at the five `alpha` agree with the live P10-C and P14 lists or nothing is
written. Git is replaced by a stub: nothing under `tmp_path` is tracked.

For S6 and S7: `p15-outcome` writes the stop-only form on a recorded `DATA_STOP`, reads the
synthetic pass back only after checking it whole, labels it from P14 against P10-B and carries
the comparison against P10-C beside it; `p15-refit` refuses without the outcome or twice, and
writes nothing unless the frozen points re-fused from the stored rankings reproduce the stored
outcomes. The HotpotQA `test-11` run files the ladder reads are synthetic stand-ins.
"""

import gzip
import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from concept_embeddings_rag import cli, config
from concept_embeddings_rag.artifacts import digest_of
from concept_embeddings_rag.corpus import fullwiki as fullwiki_corpus
from concept_embeddings_rag.corpus import musique
from concept_embeddings_rag.corpus.pool import IndexingUnit, unit_id_for
from concept_embeddings_rag.embeddings.cache import EmbeddingCache
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation import phase14, phase15
from concept_embeddings_rag.nodes import local_extraction
from concept_embeddings_rag.nodes.index import load_node_index
from concept_embeddings_rag.nodes.local_extraction import LocalExtractor
from concept_embeddings_rag.retrieval.entity_hop import RelevanceHopStage

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


# --- S3: `p15-build` ---------------------------------------------------------------------

PINNED = config.PHASE_9_GLINER_CONFIGURATION_DIGEST


def phase15_data(target: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The synthetic Phase 15 data directory S2 writes: corpus, questions, token counts."""
    run(target, source(monkeypatch))
    corpus: dict[str, Any] = json.loads((target / "corpus.json").read_text(encoding="utf-8"))
    return corpus


def titles(texts: Sequence[str]) -> list[list[str]]:
    """A stub GLiNER: each paragraph's title is its one entity."""
    return [[text.split(".")[0]] for text in texts]


def stub_extractor(digest: str, spans: Any = titles) -> LocalExtractor:
    """GLiNER's carrier with a stub reader and a chosen configuration digest."""
    return LocalExtractor(
        extractor_id=config.GLINER_EXTRACTOR,
        model=config.GLINER_MODEL,
        revision=config.GLINER_REVISION,
        labels=config.GLINER_LABELS,
        parameters={},
        library_versions={"gliner": "stub"},
        spans=spans,
        _digest=[digest],
    )


def serve_extractor(monkeypatch: pytest.MonkeyPatch, extractor: LocalExtractor) -> list[Path]:
    """Hand `extractor` to whoever loads GLiNER and log the model directory it was asked for."""
    loaded: list[Path] = []

    def load(directory: Path | str) -> tuple[LocalExtractor, float]:
        loaded.append(Path(directory))
        return extractor, 0.5

    monkeypatch.setattr(local_extraction, "gliner_extractor", load)
    return loaded


def weights_in(model_dir: Path) -> str:
    model_dir.mkdir(parents=True)
    (model_dir / config.PHASE_9_GLINER_WEIGHTS_FILE).write_bytes(b"weights")
    return hashlib.sha256(b"weights").hexdigest()


class StubBGE:
    """The pinned BGE-small's identity, with constant unit vectors."""

    name = config.EMBEDDING_MODEL
    revision = config.EMBEDDING_REVISION
    normalize = True
    query_prompt = ""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.ones((len(texts), 4), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def files_under(root: Path) -> set[Path]:
    return {path for path in root.rglob("*") if path.is_file()}


def test_extract_refuses_a_digest_other_than_the_phase_9_pod_one_before_any_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    phase15_data(tmp_path, monkeypatch)
    asked: list[Sequence[str]] = []

    def spans(texts: Sequence[str]) -> list[list[str]]:
        asked.append(texts)
        return [[] for _ in texts]

    loaded = serve_extractor(monkeypatch, stub_extractor("0" * 16, spans))

    with pytest.raises(SystemExit, match="configuration digest"):
        cli.cmd_p15_build("extract", target_dir=tmp_path)

    assert asked == []
    assert loaded == [tmp_path / "models" / config.GLINER_EXTRACTOR]
    assert not (tmp_path / "nodes").exists()


def test_extract_refuses_a_corpus_whose_long_units_stopped_the_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    phase15_data(tmp_path, monkeypatch)
    path = tmp_path / "corpus.json"
    corpus = json.loads(path.read_text(encoding="utf-8"))
    corpus["gliner_window"]["stop"] = True
    path.write_text(json.dumps(corpus), encoding="utf-8")
    loaded = serve_extractor(monkeypatch, stub_extractor(PINNED))

    with pytest.raises(SystemExit, match="window"):
        cli.cmd_p15_build("extract", target_dir=tmp_path)

    assert loaded == []
    assert not (tmp_path / "nodes").exists()


def test_the_phase_9_defaults_of_the_extraction_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    loaded = serve_extractor(monkeypatch, stub_extractor(PINNED))
    hashed: list[Path] = []

    def weights(directory: Path | str, filename: str) -> str:
        hashed.append(Path(directory))
        return "w" * 64

    monkeypatch.setattr(cli, "weights_sha256", weights)
    units = [IndexingUnit(uid("A", "a"), "A", ("a",)), IndexingUnit(uid("B", "b"), "B", ("b",))]

    manifest = cli._fullwiki_extract(units, tmp_path, "corpus", 0.0, None)

    phase_9_models = config.PHASE_9_DIR / "models" / config.GLINER_EXTRACTOR
    assert loaded == hashed == [phase_9_models]
    assert manifest["phase"] == 9
    assert "gliner_window" not in manifest
    assert sorted(path.name for path in tmp_path.iterdir()) == [phase9.GLINER_DIRNAME]
    assert (tmp_path / phase9.GLINER_DIRNAME / local_extraction.MANIFEST_NAME).exists()


def test_extract_then_index_name_the_phase_15_corpus_and_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus = phase15_data(tmp_path, monkeypatch)
    weights = weights_in(tmp_path / "models" / config.GLINER_EXTRACTOR)
    serve_extractor(monkeypatch, stub_extractor(PINNED))

    extraction = cli.cmd_p15_build("extract", target_dir=tmp_path, hourly_rate_usd=0.74)
    body = cli.cmd_p15_build("index", target_dir=tmp_path)

    assert extraction["phase"] == 15
    assert extraction["configuration_digest"] == PINNED
    assert extraction["configuration_digest_matches"] is True
    assert extraction["corpus_unit_set_hash"] == corpus["unit_set_hash"]
    assert extraction["weights_sha256"] == weights
    assert extraction["gliner_window"] == corpus["gliner_window"]
    assert body["corpus_unit_set_hash"] == corpus["unit_set_hash"]
    assert body["extraction_digest"] == extraction["digest"]
    nodes = tmp_path / "nodes"
    assert json.loads((nodes / local_extraction.MANIFEST_NAME).read_text("utf-8")) == extraction
    assert (nodes / phase9.ENTITY_INDEX_FILENAME).exists()
    assert not (tmp_path / phase9.GLINER_DIRNAME).exists()
    unit_ids = [unit.unit_id for unit in fullwiki_corpus.load_corpus(tmp_path)]
    index = load_node_index(
        nodes, extraction_digest=extraction["digest"], expected_unit_ids=unit_ids
    )
    assert index.digest == body["index_digest"]
    assert index.incidence.shape[0] == corpus["n_units"]


def test_a_smoke_extract_on_another_stack_records_the_mismatch_under_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    phase15_data(tmp_path, monkeypatch)
    weights_in(tmp_path / "models" / config.GLINER_EXTRACTOR)
    serve_extractor(monkeypatch, stub_extractor("0" * 16))

    manifest = cli.cmd_p15_build("extract", target_dir=tmp_path, smoke=4)

    assert manifest["configuration_digest_matches"] is False
    assert manifest["smoke"] == 4 and manifest["n_units"] == 4
    assert (tmp_path / "smoke" / "nodes" / local_extraction.MANIFEST_NAME).exists()
    assert not (tmp_path / "nodes").exists()


def test_embed_and_bm25_write_only_under_the_phase_15_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "phase15"
    corpus = phase15_data(target, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    for name in ("DATA_DIR", "CACHE_DIR", "QUESTION_CACHE_DIR", "PHASE_9_DIR"):
        monkeypatch.setattr(config, name, elsewhere / name)
    snapshot = tmp_path / "hub" / config.EMBEDDING_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"bge")
    monkeypatch.setattr(cli, "snapshot_directory", lambda name, revision: snapshot)
    monkeypatch.setattr(cli, "SentenceTransformerBackend", StubBGE)
    before = files_under(tmp_path)

    embedding = cli.cmd_p15_build("embed", target_dir=target, hourly_rate_usd=0.74)
    bm25 = cli.cmd_p15_build("bm25", target_dir=target)

    written = files_under(tmp_path) - before
    cache = target / "cache"
    assert all(path.is_relative_to(cache) or path.parent == target / "bm25" for path in written)
    assert cache / phase9.EMBEDDING_FILENAME in written
    assert any(path.parent == cache / "questions" for path in written)
    assert target / "bm25" / phase9.BM25_FILENAME in written
    assert not elsewhere.exists()
    assert embedding["unit_set_hash"] == bm25["unit_set_hash"] == corpus["unit_set_hash"]

    # The question vectors cover the validation questions, then the live path.
    questions, body = musique.load_questions(target, corpus_unit_set_hash=corpus["unit_set_hash"])
    qids = [q.qid for q in questions] + [q.qid for q in musique.live_path_questions(body)]
    assert embedding["n_questions"] == len(qids) == 5
    loaded = EmbeddingCache(cache / "questions").load(
        embedding["question_cache_key"], expected_unit_ids=qids
    )
    assert loaded is not None


# --- S4: `p15-integrity` -----------------------------------------------------------------

FROZEN = {
    "weights": {
        "hybrid-bm25": {"dense": 0.5, "bm25": 0.5},
        "hybrid-bm25-entity-hop": {"dense": 0.5, "bm25": 0.3, "entity-hop": 0.2},
        "hybrid-bm25-seeded-hop": {"dense": 0.5, "bm25": 0.3, "seeded-hop": 0.2},
    },
    "alpha": 0.75,
    "sources": {},
    "fit_digests": {"hybrid-bm25-entity-hop": "c" * 16, "hybrid-bm25-seeded-hop": "d" * 16},
}
METRIC_KEYS = {
    "budgets",
    "full_support",
    "gold_recall",
    "context_precision",
    "fs_at_k",
    "gpr_at_k",
    "supported",
    "recall",
    "gold",
    "latency_ms",
}


def keys_of(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_of(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_of(v)}
    return set()


def no_measure(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("no metric of a live-path question may be computed")


def built(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: type[StubBGE] = StubBGE,
    spans: Any = titles,
) -> Path:
    """Every Phase 15 input S4 reads, built over the synthetic corpus with the S3 stages."""
    target = tmp_path / "phase15"
    phase15_data(target, monkeypatch)
    snapshot = tmp_path / "hub" / config.EMBEDDING_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"bge")
    monkeypatch.setattr(cli, "snapshot_directory", lambda name, revision: snapshot)
    monkeypatch.setattr(cli, "SentenceTransformerBackend", backend)
    weights_in(target / "models" / config.GLINER_EXTRACTOR)
    serve_extractor(monkeypatch, stub_extractor(PINNED, spans))
    for stage in ("embed", "bm25", "extract", "index"):
        cli.cmd_p15_build(stage, target_dir=target)
    monkeypatch.setattr(phase15, "frozen_weights", lambda **kwargs: FROZEN)
    return target


def phase9_extraction(tmp_path: Path, **versions: str) -> Path:
    """The Phase 9 pod manifest the extractor identity is compared against."""
    directory = tmp_path / "phase9"
    (directory / phase9.GLINER_DIRNAME).mkdir(parents=True)
    body = {"configuration_digest": PINNED, "library_versions": {"gliner": "stub", **versions}}
    (directory / phase9.GLINER_DIRNAME / local_extraction.MANIFEST_NAME).write_text(
        json.dumps(body), encoding="utf-8"
    )
    return directory


def passing_code_identity(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"passed": True, "checks": {}}


def test_the_live_systems_are_built_from_the_verified_phase_15_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = built(tmp_path, monkeypatch)
    monkeypatch.setattr(phase9, "measure_system", no_measure)

    components = cli._p15_inputs(target)
    systems = cli._p15_systems(components)

    assert list(systems) == list(config.PHASE_15_SYSTEMS)
    p14 = systems["hybrid-bm25-seeded-hop"]
    assert isinstance(p14.hop, phase15.RecordingRelevanceStage)
    assert p14.hop.name == "seeded-hop" and p14.hop.inner.alpha == 0.75
    assert p14.fusion_weights == (0.5, 0.3, 0.2)
    assert [q.qid for q in components["live_path"]] == ["2hop__t1", "2hop__t2"]
    assert all(q.gold_unit_ids == () for q in components["live_path"])
    corpus_ids = set(components["dense"].unit_ids)

    result = phase15.run_live_path(systems, components["live_path"], corpus_ids, depth=100)

    assert result["passed"] is True, result
    hop = p14.hop.expansions
    assert len(hop) == 2 and all(e.alpha == 0.75 for e in hop)
    assert not keys_of(result) & METRIC_KEYS


def test_the_stage_writes_integrity_once_with_no_metric_of_the_live_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = built(tmp_path, monkeypatch)
    phase9_dir = phase9_extraction(tmp_path)
    monkeypatch.setattr(cli, "_p15_code_identity", passing_code_identity)
    monkeypatch.setattr(phase9, "measure_system", no_measure)

    path = cli.cmd_p15_integrity(target, phase9_dir=phase9_dir)

    body = json.loads(path.read_text(encoding="utf-8"))
    assert path == target / phase15.INTEGRITY_FILENAME
    assert body["terminal_state"] is None and body["stop_reasons"] == []
    assert body["extractor_identity"]["passed"] is True
    live = body["live_path"]
    assert live["qids"] == ["2hop__t1", "2hop__t2"]
    assert list(live["systems"]) == list(config.PHASE_15_SYSTEMS)
    assert not keys_of(body) & METRIC_KEYS
    assert not (target / "run-dense.json").exists()
    assert not list(target.glob("rankings-*"))

    with pytest.raises(SystemExit, match="already"):
        cli.cmd_p15_integrity(target, phase9_dir=phase9_dir)


def test_an_extractor_other_than_the_phase_9_pod_one_writes_data_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = built(tmp_path, monkeypatch)
    phase9_dir = phase9_extraction(tmp_path, torch="2.13.0+cu126")
    monkeypatch.setattr(cli, "_p15_code_identity", passing_code_identity)

    with pytest.raises(SystemExit, match=phase9.DATA_STOP):
        cli.cmd_p15_integrity(target, phase9_dir=phase9_dir)

    body = json.loads((target / phase15.INTEGRITY_FILENAME).read_text(encoding="utf-8"))
    assert body["terminal_state"] == phase9.DATA_STOP
    assert body["extractor_identity"]["library_versions_differing"] == ["torch"]


def test_missing_phase_15_inputs_stop_the_stage_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    phase15_data(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_p15_code_identity", passing_code_identity)

    with pytest.raises(SystemExit):
        cli.cmd_p15_integrity(tmp_path, phase9_dir=phase9_extraction(tmp_path / "p9"))

    assert not (tmp_path / phase15.INTEGRITY_FILENAME).exists()


def test_the_p15_integrity_stage_is_registered():
    assert cli.build_parser().parse_args(["p15-integrity"]).command == "p15-integrity"


# --- S5: `p15-eval --authorized-pass` -----------------------------------------------------

REAL_FROZEN_WEIGHTS = phase15.frozen_weights
P10A, P10B, P10C, P14 = config.PHASE_15_SYSTEMS
N_TOY = 3  # the synthetic validation questions


class VariedBGE(StubBGE):
    """The pinned BGE-small's identity, with a distinct unit vector per text."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = [
            np.frombuffer(hashlib.sha256(text.encode("utf-8")).digest()[:8], dtype=np.uint8)
            for text in texts
        ]
        vectors = np.asarray(rows, dtype=np.float32).reshape(len(texts), 8) - 127.5
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def shared_spans(texts: Sequence[str]) -> list[list[str]]:
    """A stub GLiNER: each paragraph's title, and one entity every paragraph shares, so the
    hop has candidates outside the Dense units it read."""
    return [[text.split(".")[0], "the shared city"] for text in texts]


def passable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The synthetic Phase 15 inputs with a passing, 'committed' `integrity.json`."""
    target = built(tmp_path, monkeypatch, backend=VariedBGE, spans=shared_spans)
    monkeypatch.setattr(cli, "_p15_code_identity", passing_code_identity)
    cli.cmd_p15_integrity(target, phase9_dir=phase9_extraction(tmp_path))
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: True)
    return target


def evaluate(target: Path, **overrides: Any) -> list[Path]:
    arguments: dict[str, Any] = {"authorized": True, "target_dir": target, "n_questions": N_TOY}
    arguments.update(overrides)
    return cli.cmd_p15_eval(**arguments)


def pass_files(target: Path) -> list[str]:
    patterns = ("pass.json", "run-*", "outcomes-*", "rankings-*")
    return sorted(path.name for pattern in patterns for path in target.glob(pattern))


def test_the_pass_refuses_without_the_authorization_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = passable(tmp_path, monkeypatch)

    with pytest.raises(SystemExit, match="--authorized-pass"):
        evaluate(target, authorized=False)

    assert pass_files(target) == []


def test_the_pass_refuses_without_integrity_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = passable(tmp_path, monkeypatch)
    (target / phase15.INTEGRITY_FILENAME).unlink()

    with pytest.raises(SystemExit, match=phase15.INTEGRITY_FILENAME):
        evaluate(target)

    assert pass_files(target) == []


def test_the_pass_refuses_after_a_d4_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = passable(tmp_path, monkeypatch)
    path = target / phase15.INTEGRITY_FILENAME
    body = json.loads(path.read_text(encoding="utf-8"))
    body["terminal_state"] = phase9.DATA_STOP
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(SystemExit, match=phase9.DATA_STOP):
        evaluate(target)

    assert pass_files(target) == []


def test_the_pass_refuses_an_integrity_json_git_does_not_track_unmodified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = passable(tmp_path, monkeypatch)
    integrity = target / phase15.INTEGRITY_FILENAME
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: Path(path) != integrity)

    with pytest.raises(SystemExit, match="committed"):
        evaluate(target)

    assert pass_files(target) == []


def test_the_pass_refuses_a_frozen_fit_git_does_not_track_unmodified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = passable(tmp_path, monkeypatch)
    monkeypatch.setattr(phase15, "frozen_weights", REAL_FROZEN_WEIGHTS)
    phase10_dir, phase14_dir = tmp_path / "p10", tmp_path / "p14"
    for directory in (phase10_dir, phase14_dir):
        directory.mkdir()
        (directory / "fit.json").write_text("{}", encoding="utf-8")
    fit14 = phase14_dir / "fit.json"
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: Path(path) != fit14)

    with pytest.raises(SystemExit, match="not committed unmodified"):
        evaluate(target, phase10_dir=phase10_dir, phase14_dir=phase14_dir)

    assert pass_files(target) == []


def test_the_pass_refuses_a_second_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = passable(tmp_path, monkeypatch)
    (target / "pass.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="already started"):
        evaluate(target)

    assert pass_files(target) == ["pass.json"]


def test_the_pass_refuses_inputs_other_than_those_d4_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = passable(tmp_path, monkeypatch)
    path = target / phase15.INTEGRITY_FILENAME
    body = json.loads(path.read_text(encoding="utf-8"))
    body["provenance"]["bm25_index_digest"] = "0" * 16
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(SystemExit, match="D4"):
        evaluate(target)

    assert pass_files(target) == []


def test_the_pass_measures_the_four_systems_and_stores_their_rankings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    target = passable(tmp_path, monkeypatch)
    capsys.readouterr()

    written = evaluate(target)

    printed = capsys.readouterr().out
    assert written == [target / f"run-{name}.json" for name in config.PHASE_15_SYSTEMS]
    assert "supported" not in printed.lower() and "TRANSFER" not in printed
    marker = json.loads((target / "pass.json").read_text(encoding="utf-8"))
    integrity_text = (target / phase15.INTEGRITY_FILENAME).read_text(encoding="utf-8")
    assert marker["integrity_digest"] == digest_of(integrity_text)
    assert marker["systems"] == list(config.PHASE_15_SYSTEMS)
    assert marker["n_questions"] == N_TOY

    components = cli._p15_inputs(target)
    questions = components["questions"]
    by_qid = {question.qid: question for question in questions}
    rankings: dict[str, list[dict[str, Any]]] = {}
    for position, name in enumerate(config.PHASE_15_SYSTEMS):
        body = json.loads((target / f"run-{name}.json").read_text(encoding="utf-8"))
        outcomes = phase9.load_outcomes(target, name)
        rankings[name] = phase15.load_rankings(target, name)
        assert body["order"] == position and body["set"] == musique.VALIDATION_SPLIT
        assert body["rankings_file"] == f"rankings-{name}.jsonl.gz"
        assert body["provenance"]["integrity_digest"] == marker["integrity_digest"]
        assert set(body["memory_before"]) == {"working_set_mb", "peak_memory_mb"}
        assert [r["qid"] for r in outcomes] == [r["qid"] for r in rankings[name]]
        assert [r["qid"] for r in outcomes] == [q.qid for q in questions]
        weights = body["fusion_weights"]
        for outcome, stored in zip(outcomes, rankings[name], strict=True):
            assert set(outcome["budgets"]) == {"512", "1024", "2048", "4096"}
            assert set(outcome["fs_at_k"]) == set(outcome["gpr_at_k"]) == {"2", "5", "10", "20"}
            # The file's keys are sorted: the fusion order is the one the run records.
            assert set(stored["components"]) == set(body["ranking_components"])
            ordered = [stored["components"][c] for c in body["ranking_components"]]
            fused = ordered[0] if weights is None else cli.fuse_lists(ordered, weights, top_k=100)
            assert fused == stored["fused"]
            question = by_qid[stored["qid"]]
            replayed = cli._p10_replay_records(
                name, [(question.question, fused)], [question], components["token_counts"]
            )
            assert (
                replayed[0]["budgets"]["2048"]["full_support"]
                == outcome["budgets"]["2048"]["full_support"]
            )

    # The hop lists at the five alpha: alpha 0 is P10-C's live list, 0.75 is P14's.
    zero, frozen = phase14.alpha_key(0.0), phase14.alpha_key(0.75)
    hops = rankings[P14]
    assert all(
        list(r["hop_alphas"]) == [phase14.alpha_key(a) for a in config.PHASE_14_ALPHAS]
        for r in hops
    )
    assert any(r["components"]["relevance-hop"] for r in hops)
    for p14, p10c in zip(hops, rankings[P10C], strict=True):
        assert p14["hop_alphas"][zero] == p10c["components"]["entity-hop"]
        assert p14["hop_alphas"][frozen] == p14["components"]["relevance-hop"]
        assert "sim_seconds" in p14["hop"]
    assert all("hop_alphas" not in r for name in (P10A, P10B, P10C) for r in rankings[name])

    with pytest.raises(SystemExit, match="already started"):
        evaluate(target)


def test_a_hop_list_that_differs_from_the_live_one_stops_before_any_ranking_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = passable(tmp_path, monkeypatch)
    live_hop_all = RelevanceHopStage.hop_all

    def moved(self: RelevanceHopStage, *args: Any, **kwargs: Any) -> Any:
        lists = live_hop_all(self, *args, **kwargs)
        return {alpha: replace(e, candidates=()) for alpha, e in lists.items()}

    monkeypatch.setattr(RelevanceHopStage, "hop_all", moved)

    with pytest.raises(SystemExit, match="hop"):
        evaluate(target)

    assert pass_files(target) == ["pass.json"]


def test_the_p15_eval_stage_is_registered():
    args = cli.build_parser().parse_args(["p15-eval", "--authorized-pass"])

    assert args.command == "p15-eval" and args.authorized is True
    assert cli.build_parser().parse_args(["p15-eval"]).authorized is False


# --- S6: `p15-outcome` -------------------------------------------------------------------

HOTPOTQA_TEST_11 = {P10A: 2852, P10B: 3079, P10C: 3285, P14: 3530}
STOP_KEYS = {
    "phase",
    "primary_comparison",
    "primary_metric",
    "terminal_state",
    "stop_reasons",
    "code_commit",
    "created_at",
}


def phase14_runs(directory: Path, counts: dict[str, int] = HOTPOTQA_TEST_11) -> Path:
    """The four Phase 14 run files the ladder reads, reduced to what it reads."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, supported in counts.items():
        body = {
            "system": name,
            "set": config.PHASE_14_TEST,
            "metrics": {"n_questions": 5000, "supported_at_primary_budget": supported},
        }
        (directory / f"run-{name}.json").write_text(json.dumps(body), encoding="utf-8")
    return directory


def measured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The synthetic pass, run once: runs, outcomes and rankings of the four systems."""
    target = passable(tmp_path, monkeypatch)
    evaluate(target)
    return target


def outcome_of(target: Path, tmp_path: Path, **overrides: Any) -> Path:
    arguments: dict[str, Any] = {
        "phase14_dir": phase14_runs(tmp_path / "p14"),
        "n_questions": N_TOY,
    }
    arguments.update(overrides)
    return cli.cmd_p15_outcome(target, **arguments)


def test_the_outcome_labels_the_pass_and_carries_both_comparisons_of_the_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = measured(tmp_path, monkeypatch)

    path = outcome_of(target, tmp_path)

    body = json.loads(path.read_text(encoding="utf-8"))
    assert path == target / "outcome.json"
    primary = body["d5_primary_p14_vs_p10b"]
    assert (primary["control"], primary["candidate"]) == ("P10-B", "P14")
    assert primary["n_questions"] == N_TOY
    assert body["terminal_state"] == phase15.label(
        primary["wins"], primary["losses"], primary["exact_two_sided_p"]
    )
    assert body["terminal_state"] != phase9.DATA_STOP and body["stop_reasons"] == []
    secondary = body["d6_secondary"]
    assert set(secondary) == {"p14_vs_p10c", "p10c_vs_p10b", "p10b_vs_p10a"}
    assert (secondary["p14_vs_p10c"]["control"], secondary["p14_vs_p10c"]["candidate"]) == (
        "P10-C",
        "P14",
    )
    ladder = body["d6_ladder"]
    assert [r["hotpotqa_test_11"]["supported"] for r in ladder["rungs"]] == list(
        HOTPOTQA_TEST_11.values()
    )
    groups = body["d7_by_supporting"]["groups"]
    assert (groups["2"]["n_questions"], groups["3"]["n_questions"]) == (2, 1)
    assert groups["4"] == {"n_questions": 0}
    assert set(groups["2"]["comparisons"]) == {key for key, _a, _b in phase15.COMPARISONS}
    d8 = body["d8_reach"]
    assert set(d8["dense_rank_split"]) == {"p14_vs_p10b", "p14_vs_p10c"}
    assert set(d8["systems"]) == set(config.PHASE_15_SYSTEMS)
    assert "candidate_set" in d8["systems"][P14] and "candidate_set" not in d8["systems"][P10B]
    d10 = body["d10_latency"]
    assert d10["run_order"] == list(config.PHASE_15_SYSTEMS)
    assert set(d10["systems"][P10A]["memory_before"]) == {"working_set_mb", "peak_memory_mb"}
    assert d10["p14_sim"]["n_questions"] == N_TOY
    integrity = body["integrity"]
    assert integrity["n_questions"] == N_TOY and integrity["same_qid_order"] is True
    assert integrity["refusion_consistent"] is True
    assert set(integrity["rankings_digests"]) == set(config.PHASE_15_SYSTEMS)

    with pytest.raises(SystemExit, match="already"):
        outcome_of(target, tmp_path)


def test_the_outcome_refuses_a_rankings_file_its_run_does_not_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = measured(tmp_path, monkeypatch)
    path = target / f"rankings-{P10C}.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"qid": "x"}\n', mtime=0))

    with pytest.raises(SystemExit, match="digest"):
        outcome_of(target, tmp_path)

    assert not (target / "outcome.json").exists()


def test_the_outcome_refuses_hotpotqa_figures_other_than_the_recorded_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = measured(tmp_path, monkeypatch)
    moved = {**HOTPOTQA_TEST_11, P14: 3531}

    with pytest.raises(SystemExit, match="test-11"):
        outcome_of(target, tmp_path, phase14_dir=phase14_runs(tmp_path / "moved", moved))

    assert not (target / "outcome.json").exists()


def test_the_outcome_refuses_when_no_pass_ran(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = passable(tmp_path, monkeypatch)

    with pytest.raises(SystemExit, match="pass"):
        outcome_of(target, tmp_path)

    assert not (target / "outcome.json").exists()


def test_a_questions_data_stop_writes_the_stop_only_outcome(
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

    path = cli.cmd_p15_outcome(tmp_path)

    body = json.loads(path.read_text(encoding="utf-8"))
    assert set(body) == STOP_KEYS
    assert body["terminal_state"] == phase9.DATA_STOP
    assert len(body["stop_reasons"]) == 1 and "unmapped" in body["stop_reasons"][0]


def test_an_integrity_data_stop_writes_the_stop_only_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    phase15_data(tmp_path, monkeypatch)
    integrity = {"terminal_state": phase9.DATA_STOP, "stop_reasons": ["live path: ['dense']"]}
    (tmp_path / phase15.INTEGRITY_FILENAME).write_text(json.dumps(integrity), encoding="utf-8")

    body = json.loads(cli.cmd_p15_outcome(tmp_path).read_text(encoding="utf-8"))

    assert set(body) == STOP_KEYS
    assert body["terminal_state"] == phase9.DATA_STOP
    assert body["stop_reasons"] == ["integrity: live path: ['dense']"]

    with pytest.raises(SystemExit, match="already"):
        cli.cmd_p15_outcome(tmp_path)


# --- S7: `p15-refit` ---------------------------------------------------------------------


def test_the_refit_refuses_without_outcome_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = measured(tmp_path, monkeypatch)

    with pytest.raises(SystemExit, match="outcome.json"):
        cli.cmd_p15_refit(target, n_questions=N_TOY)

    assert not (target / "refit.json").exists()


def test_the_refit_refuses_when_refit_json_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = measured(tmp_path, monkeypatch)
    outcome_of(target, tmp_path)
    (target / "refit.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="already"):
        cli.cmd_p15_refit(target, n_questions=N_TOY)

    assert (target / "refit.json").read_text(encoding="utf-8") == "{}"


def test_the_refit_refuses_after_a_stop_only_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    phase15_data(tmp_path, monkeypatch)
    integrity = {"terminal_state": phase9.DATA_STOP, "stop_reasons": ["x"]}
    (tmp_path / phase15.INTEGRITY_FILENAME).write_text(json.dumps(integrity), encoding="utf-8")
    cli.cmd_p15_outcome(tmp_path)

    with pytest.raises(SystemExit, match=phase9.DATA_STOP):
        cli.cmd_p15_refit(tmp_path, n_questions=N_TOY)

    assert not (tmp_path / "refit.json").exists()


def test_the_refit_reproduces_the_stored_outcomes_at_the_frozen_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = measured(tmp_path, monkeypatch)
    outcome_of(target, tmp_path)

    path = cli.cmd_p15_refit(target, n_questions=N_TOY)

    body = json.loads(path.read_text(encoding="utf-8"))
    assert path == target / "refit.json"
    assert body["exploratory"] is True
    assert "upper bound" in body["caution"] and "never" in body["caution"]
    assert body["n_points"] == len(body["curve"]) == 330
    consistency = body["consistency"]
    assert consistency["passed"] is True
    recorded = {name: phase10_supported(target, name) for name in (P10C, P14)}
    assert consistency["frozen_p14"]["supported"] == recorded[P14]
    assert consistency["frozen_p14"]["differing_qids"] == []
    assert consistency["p10c_at_alpha_0"]["supported"] == recorded[P10C]
    frozen = next(
        p for p in body["curve"] if p["alpha"] == 0.75 and p["weights"] == [0.5, 0.3, 0.2]
    )
    assert frozen["supported"] == recorded[P14]
    assert body["best"] == phase14.choose_point(body["curve"])

    with pytest.raises(SystemExit, match="already"):
        cli.cmd_p15_refit(target, n_questions=N_TOY)


def phase10_supported(target: Path, name: str) -> int:
    return int(
        sum(r["budgets"]["2048"]["full_support"] for r in phase9.load_outcomes(target, name))
    )


def test_the_refit_writes_nothing_when_the_frozen_point_moves_a_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = measured(tmp_path, monkeypatch)
    outcome_of(target, tmp_path)
    replay = cli._p10_replay_records

    def moved(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        records = replay(*args, **kwargs)
        first = records[0]["budgets"]["2048"]
        first["full_support"] = 1.0 - first["full_support"]
        return records

    monkeypatch.setattr(cli, "_p10_replay_records", moved)

    with pytest.raises(SystemExit, match="moved"):
        cli.cmd_p15_refit(target, n_questions=N_TOY)

    assert not (target / "refit.json").exists()


def test_the_p15_outcome_and_refit_stages_are_registered():
    assert cli.build_parser().parse_args(["p15-outcome"]).command == "p15-outcome"
    assert cli.build_parser().parse_args(["p15-refit"]).command == "p15-refit"
