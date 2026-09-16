"""T11 of Phase 5: the contamination audit, in one place (HU-8, decision D13).

Each assertion is made on a toy case that always runs, and - where the real artifact can be
on this machine - again on the real one, skipped when it is absent:

- a test-split question is refused by the pilot builder and by the navigation run;
- the extraction cache holds no question string of either split;
- `p1` and the read top-10 do not move when the gold annotations do;
- an extraction that misses a pool unit is refused before any node is built.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question, save_pool, unit_id_for
from concept_embeddings_rag.evaluation import second_hop
from concept_embeddings_rag.evaluation.navigation import run_navigation
from concept_embeddings_rag.evaluation.pilot import PilotError, build_pilot, load_pilot
from concept_embeddings_rag.evaluation.selection import SelectionError
from concept_embeddings_rag.nodes.extraction import (
    OK,
    PROMPT_DIGEST,
    ExtractionRecord,
    RawResponse,
    run_full,
    run_sample,
)
from concept_embeddings_rag.nodes.index import build_node_index

N = 16
UNIT_IDS = [f"u{index:02d}" for index in range(N)]


def eye() -> np.ndarray:
    return np.eye(N, dtype=np.float32)


def a_question(qid: str, text: str, gold: tuple[str, ...], split: str = "dev") -> Question:
    return Question(qid, text, "-", gold, tuple((unit, 0) for unit in gold), split)


class Encoder:
    """Each question text picks its own dense order; the gold plays no part in it."""

    def encode(self, texts):
        rows = []
        for text in texts:
            seed = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(seed, "big"))
            rows.append(rng.random(N).astype(np.float32))
        return np.stack(rows)


def a_pilot(questions, read_depth=3):
    return build_pilot(
        questions,
        unit_ids=UNIT_IDS,
        vectors=eye(),
        query_backend=Encoder(),
        model="toy",
        revision="toy",
        unit_set_hash="toy-pool",
        read_depth=read_depth,
    )


# --- Test questions are refused at every door ------------------------------------------


def test_the_pilot_builder_refuses_a_test_question():
    questions = [
        a_question("q0", "dev question", ("u10",)),
        a_question("q1", "test question", ("u11",), split="test"),
    ]

    with pytest.raises(SelectionError):
        a_pilot(questions)


def test_the_navigation_run_refuses_a_pilot_question_that_is_on_test():
    dev = [a_question(f"q{i}", f"question {i}", ("u15",)) for i in range(3)]
    pilot = a_pilot(dev)
    relabelled = {q.qid: a_question(q.qid, q.question, q.gold_unit_ids, split="test") for q in dev}
    records = {
        u: ExtractionRecord(u, config.EXTRACTION_MODEL, PROMPT_DIGEST, OK, (), (), None, 0, 0)
        for u in UNIT_IDS
    }

    with pytest.raises((PilotError, SelectionError)):
        run_navigation(
            pilot,
            relabelled,
            unit_ids=UNIT_IDS,
            texts=UNIT_IDS,
            vectors=eye(),
            query_backend=Encoder(),
            bm25=type("NoBM25", (), {"retrieve": lambda self, query, top_k: []})(),
            index=build_node_index(records, UNIT_IDS, extraction_digest="toy"),
            concept_matrix=sparse.csr_matrix((N, 2)),
            concept_weights=np.ones(2),
            provenance={},
        )


def test_the_real_pilot_holds_dev_questions_only():
    pool_path = config.DATA_DIR / "pool.json"
    pilot_file = config.PILOT_DIR / "pilot.json"
    if not (pool_path.exists() and pilot_file.exists()):
        pytest.skip("the pool or the frozen pilot is not on this machine")
    from concept_embeddings_rag.corpus.pool import load_pool

    _, questions = load_pool(pool_path)
    split_of = {question.qid: question.split for question in questions}

    pilot = load_pilot(config.PILOT_DIR)

    assert pilot.questions
    assert {split_of[entry.qid] for entry in pilot.questions} == {"dev"}


# --- The extraction never sees a question ----------------------------------------------


def question_pattern(questions) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(question.question) for question in questions))


def test_a_sample_cache_holds_no_question_of_either_split(tmp_path: Path):
    units = [
        IndexingUnit(unit_id_for(f"T{i}", (f"Body {i}.",)), f"T{i}", (f"Body {i}.",))
        for i in range(6)
    ]
    questions = [
        a_question("qd", "Which body follows body three?", (units[3].unit_id,)),
        a_question("qt", "Where was the fifth body written?", (units[5].unit_id,), split="test"),
    ]
    seen_prompts: list[str] = []

    class Recording:
        model = config.EXTRACTION_MODEL

        def extract(self, request):
            seen_prompts.append(json.dumps(request))
            return RawResponse("end_turn", json.dumps({"entities": [], "concepts": []}), 1, 1)

    run_sample(
        units,
        Recording(),
        cache_dir=tmp_path,
        extraction_dir=tmp_path / "extraction",
        token_counts={unit.unit_id: 5 for unit in units},
        size=len(units),
    )

    pattern = question_pattern(questions)
    assert seen_prompts and not any(pattern.search(prompt) for prompt in seen_prompts)
    for entry in (tmp_path / "extraction-cache").glob("*.json"):
        assert not pattern.search(entry.read_text(encoding="utf-8"))


def test_the_real_extraction_cache_holds_no_question_of_either_split():
    cache = config.CACHE_DIR / "extraction-cache"
    pool_path = config.DATA_DIR / "pool.json"
    if not (cache.exists() and pool_path.exists()):
        pytest.skip("no extraction cache on this machine yet")
    from concept_embeddings_rag.corpus.pool import load_pool

    _, questions = load_pool(pool_path)
    pattern = question_pattern(questions)
    entries = list(cache.glob("*.json"))

    assert entries
    offending = [entry.name for entry in entries if pattern.search(entry.read_text("utf-8"))]
    assert offending == []


# --- The hop origin does not depend on the gold ----------------------------------------


def test_p1_and_the_read_list_do_not_move_when_the_gold_does():
    rng = np.random.default_rng(config.DEFAULT_SEED)
    texts = [f"question number {i}" for i in range(12)]
    first = [a_question(f"q{i:02d}", text, ("u15",)) for i, text in enumerate(texts)]
    shuffled = [
        a_question(q.qid, q.question, tuple(rng.choice(UNIT_IDS, size=2, replace=False)))
        for q in first
    ]
    read_depth = 3

    for original, moved in zip(first, shuffled, strict=True):
        order_a = second_hop.dense_order(eye(), Encoder().encode([original.question])[0])
        order_b = second_hop.dense_order(eye(), Encoder().encode([moved.question])[0])
        assert second_hop.read_rows(order_a, read_depth) == second_hop.read_rows(
            order_b, read_depth
        )

    pilot_a = {e.qid: e for e in a_pilot(first, read_depth).questions}
    pilot_b = {e.qid: e for e in a_pilot(shuffled, read_depth).questions}
    for qid in pilot_a.keys() & pilot_b.keys():
        assert (pilot_a[qid].p1, pilot_a[qid].read) == (pilot_b[qid].p1, pilot_b[qid].read)


# --- No node without the whole pool extracted ------------------------------------------


def test_cer_nodes_refuses_an_extraction_that_misses_a_pool_unit(tmp_path: Path):
    from concept_embeddings_rag.cli import cmd_nodes

    units = [
        IndexingUnit(unit_id_for(f"T{i}", (f"Body {i}.",)), f"T{i}", (f"Body {i}.",))
        for i in range(8)
    ]

    class Client:
        model = config.EXTRACTION_MODEL

        def extract(self, request):
            return RawResponse("end_turn", json.dumps({"entities": ["X"], "concepts": ["y"]}), 1, 1)

    class NoBatches:
        model = config.EXTRACTION_MODEL

    extracted = units[:7]
    run_sample(
        extracted,
        Client(),
        cache_dir=tmp_path / "cache",
        extraction_dir=tmp_path / "extraction",
        token_counts={unit.unit_id: 5 for unit in extracted},
        size=len(extracted),
    )
    run_full(
        extracted,
        NoBatches(),  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        extraction_dir=tmp_path / "extraction",
        wait=lambda: None,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    question = a_question("q0", "Unused?", (units[0].unit_id,))
    save_pool(units, [question], data_dir / "pool.json")

    with pytest.raises(SystemExit) as excinfo:
        cmd_nodes(
            data_dir=data_dir,
            extraction_dir=tmp_path / "extraction",
            nodes_dir=tmp_path / "nodes",
        )

    assert "cover" in str(excinfo.value)
    assert not (tmp_path / "nodes").exists()
