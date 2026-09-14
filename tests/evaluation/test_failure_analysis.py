"""T16: the table Phase 1 computed by hand, regenerated from code.

Phase 1's failure analysis is the finding this project turns on: at 2,048 tokens on
dev, 72 of 600 questions defeat dense and BM25 alike, and all of them fail the same
way - the retriever finds the entity the question names and never reaches the bridge
entity it does not. HU-8 asks the same table of System B, and "the same table" only
means anything if the same code produces both.

So the headline test here is a reproduction: dense against BM25, on the real pool,
must come back 393 / 94 / 41 / 72. It is skipped where those artifacts do not exist
and it never builds them - in particular it never embeds, because a test that
silently spends fifteen seconds on the model is a test someone will delete.

The criterion is **Full Support**, not gold recall: a HotpotQA question needs both of
its paragraphs and half the evidence answers nothing. That is not a detail of
bookkeeping - (393 + 94) / 600 is 81.2%, which is exactly the Full Support the dense
baseline recorded in Phase 1, and it is what makes this table and that one the same
measurement rather than two similar ones.
"""

import ast
import json
from pathlib import Path

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question, load_pool
from concept_embeddings_rag.embeddings.backend import SentenceTransformerBackend
from concept_embeddings_rag.embeddings.cache import (
    EmbeddingCache,
    build_query_backend,
    cache_key,
    question_cache_key,
    question_set_hash,
    unit_set_hash,
)
from concept_embeddings_rag.evaluation.failure_analysis import (
    CrossTab,
    FailureAnalysisError,
    cross_tabulate,
    per_question_success,
)
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.bm25 import BM25Retriever
from concept_embeddings_rag.retrieval.dense import DenseRetriever

SOURCE = Path(__file__).resolve().parents[2] / "src" / "concept_embeddings_rag"

# Three units of ten tokens each and a budget for two of them, so what reaches the
# context is the top of the ranking and the arithmetic is checkable by eye.
TOKENS = {"g1": 10, "g2": 10, "x": 10}
TWO_UNITS = 20

# Four questions, one for each cell of the table. Both golds are needed every time.
ANSWERS: dict[str, dict[str, list[Hit]]] = {
    "a": {
        "both": [("g1", 0.9), ("g2", 0.8), ("x", 0.1)],
        "only_a": [("g1", 0.9), ("g2", 0.8), ("x", 0.1)],
        "only_b": [("x", 0.9), ("g1", 0.8), ("g2", 0.1)],
        "neither": [("x", 0.9), ("g1", 0.8), ("g2", 0.1)],
    },
    "b": {
        "both": [("g2", 0.9), ("g1", 0.8), ("x", 0.1)],
        "only_a": [("x", 0.9), ("g2", 0.8), ("g1", 0.1)],
        "only_b": [("g1", 0.9), ("g2", 0.8), ("x", 0.1)],
        "neither": [("x", 0.9), ("g2", 0.8), ("g1", 0.1)],
    },
}


class StubRetriever:
    """Answers each question from a table written out above."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.answers = ANSWERS[name]
        self.asked_for: list[int] = []

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        self.asked_for.append(top_k)
        return self.answers[query][:top_k]


def a_question(qid: str, gold: tuple[str, ...] = ("g1", "g2")) -> Question:
    """The qid doubles as the query string, so a stub can answer per question."""
    return Question(
        qid=qid,
        question=qid,
        answer="-",
        gold_unit_ids=gold,
        supporting_facts=(("g1", 0), ("g2", 0)),
        split="dev",
    )


def the_questions() -> list[Question]:
    return [a_question(qid) for qid in ("both", "only_a", "only_b", "neither")]


def success_of(name: str) -> dict[str, bool]:
    return per_question_success(StubRetriever(name), the_questions(), TOKENS, TWO_UNITS, top_k=3)


def a_table() -> CrossTab:
    return cross_tabulate(
        success_of("a"), success_of("b"), names=("dense", "bm25"), budget=TWO_UNITS
    )


# --- What counts as answering a question --------------------------------------


def test_a_question_is_answered_when_every_gold_paragraph_is_in_the_context():
    assert success_of("a") == {
        "both": True,
        "only_a": True,
        "only_b": False,
        "neither": False,
    }


def test_half_the_evidence_is_not_half_an_answer():
    """Full Support, not gold recall: one of two gold paragraphs answers nothing.

    `only_b` is the case. Its ranking puts a distractor first and one gold second, so
    the two units that fit carry exactly one gold. Gold recall would score that 0.5;
    the question is still unanswered, and the table says so.
    """
    assert success_of("a")["only_b"] is False


def test_the_budget_is_what_decides_how_much_context_there_is():
    """The same ranking answers the question at a budget that admits the second gold."""
    wider = per_question_success(StubRetriever("a"), [a_question("only_b")], TOKENS, 30, top_k=3)

    assert wider["only_b"] is True


def test_the_ranking_is_asked_for_the_top_k_it_was_given():
    retriever = StubRetriever("a")

    per_question_success(retriever, the_questions(), TOKENS, TWO_UNITS, top_k=100)

    assert retriever.asked_for == [100] * 4


def test_a_question_measured_twice_is_refused_rather_than_counted_once():
    with pytest.raises(FailureAnalysisError, match="twice"):
        per_question_success(
            StubRetriever("a"), [a_question("both"), a_question("both")], TOKENS, TWO_UNITS, top_k=3
        )


def test_a_budget_that_admits_nothing_is_refused():
    with pytest.raises(FailureAnalysisError, match="budget"):
        per_question_success(StubRetriever("a"), the_questions(), TOKENS, 0, top_k=3)


def test_a_top_k_of_zero_is_refused():
    with pytest.raises(FailureAnalysisError, match="top_k"):
        per_question_success(StubRetriever("a"), the_questions(), TOKENS, TWO_UNITS, top_k=0)


# --- The 2x2 ------------------------------------------------------------------


def test_the_four_cells_are_the_four_outcomes_phase_1_reported():
    assert a_table().counts == {"both": 1, "only_dense": 1, "only_bm25": 1, "neither": 1}


def test_the_cells_list_the_questions_and_not_only_how_many_there_were():
    """The counts say how many were lost; only the qids say which."""
    table = a_table()

    assert table.both == ("both",)
    assert table.only_a == ("only_a",)
    assert table.only_b == ("only_b",)
    assert table.neither == ("neither",)


def test_the_counts_are_keyed_by_the_systems_that_earned_them():
    assert sorted(a_table().counts) == ["both", "neither", "only_bm25", "only_dense"]


def test_every_question_falls_in_exactly_one_cell():
    table = a_table()

    assert table.n_questions == len(the_questions())


def test_the_qids_come_back_in_a_stable_order_rather_than_a_dict_s():
    """Two runs must produce the same lists, or the table cannot be diffed against itself."""
    assert a_table() == a_table()


def test_two_systems_measured_over_different_questions_cannot_be_tabulated():
    """A table over mismatched sets still renders, and every cell of it is wrong."""
    partial = {qid: value for qid, value in success_of("b").items() if qid != "neither"}

    with pytest.raises(FailureAnalysisError, match="different question sets"):
        cross_tabulate(success_of("a"), partial, names=("dense", "bm25"), budget=TWO_UNITS)


def test_a_system_cannot_be_cross_tabulated_against_itself():
    with pytest.raises(FailureAnalysisError, match="two systems"):
        cross_tabulate(success_of("a"), success_of("a"), names=("dense", "dense"), budget=TWO_UNITS)


def test_a_question_cannot_be_filed_in_two_cells():
    with pytest.raises(FailureAnalysisError, match="exactly one cell"):
        CrossTab(
            a="dense",
            b="bm25",
            budget=TWO_UNITS,
            both=("q1",),
            only_a=("q1",),
            only_b=(),
            neither=(),
        )


# --- Which side of the line this module lives on ------------------------------


def test_it_reads_gold_and_therefore_lives_beside_the_measurer():
    """`evaluation/` is the measurer's side: reading the answer key is its job there."""
    source = (SOURCE / "evaluation" / "failure_analysis.py").read_text(encoding="utf-8")

    assert "gold_unit_ids" in source


def test_nothing_in_the_retrieval_path_imports_it():
    """A retriever that could see this module could see the answer key through it."""
    for path in sorted((SOURCE / "retrieval").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            for name in (node.module, *(f"{node.module}.{a.name}" for a in node.names))
        }

        assert not any("failure_analysis" in name for name in imported), (
            f"{path.name} imports the failure analysis, which reads gold annotations"
        )


# --- The reproduction ---------------------------------------------------------


def the_real_dev_split():
    """The Phase 1 artifacts, or a skip that names the stage which builds each one.

    Nothing here is computed: the pool, the corpus embeddings and the dev question
    embeddings are all read from disk, and a missing one skips rather than being
    rebuilt. Embedding 600 questions costs fifteen seconds, and a test suite that
    quietly spends them is a test suite someone starts skipping.
    """
    pool_path = config.DATA_DIR / "pool.json"
    tokens_path = config.DATA_DIR / "token_counts.json"
    if not pool_path.exists() or not tokens_path.exists():
        pytest.skip("no pool on this machine; `cer build` and `cer embed` produce it")

    units, questions = load_pool(pool_path)
    dev = [question for question in questions if question.split == "dev"]
    token_counts = json.loads(tokens_path.read_text(encoding="utf-8"))

    backend = SentenceTransformerBackend()
    normalized = bool(getattr(backend, "normalize", True))
    corpus = EmbeddingCache(Path(config.CACHE_DIR)).load(
        cache_key(
            backend.name,
            backend.revision,
            unit_set_hash([unit.unit_id for unit in units]),
            normalized=normalized,
        ),
        expected_unit_ids=[unit.unit_id for unit in units],
    )
    if corpus is None:
        pytest.skip("the corpus embeddings are not cached here; `cer embed` writes them")

    question_cache = EmbeddingCache(config.QUESTION_CACHE_DIR)
    cached_questions = question_cache.load(
        question_cache_key(
            backend.name,
            backend.revision,
            question_set_hash([question.qid for question in dev]),
            "dev",
            normalized=normalized,
        )
    )
    if cached_questions is None:
        pytest.skip("the dev question embeddings are not cached here; `cer select` writes them")

    vectors, unit_ids = corpus
    dense = DenseRetriever(
        vectors=vectors,
        unit_ids=unit_ids,
        backend=build_query_backend(dev, backend, question_cache),
    )
    return dense, BM25Retriever(units), dev, token_counts


def test_the_phase_1_failure_table_regenerates_from_code():
    """393 / 94 / 41 / 72, the table `1.results.md` printed, recomputed rather than copied.

    It is also the arithmetic that ties the two together: (393 + 94) / 600 = 81.2% is
    the dense Full Support Phase 1 recorded, and (393 + 41) / 600 = 72.3% is BM25's.
    """
    dense, bm25, dev, token_counts = the_real_dev_split()

    table = cross_tabulate(
        per_question_success(dense, dev, token_counts, config.SELECTION_BUDGET, top_k=100),
        per_question_success(bm25, dev, token_counts, config.SELECTION_BUDGET, top_k=100),
        names=("dense", "bm25"),
        budget=config.SELECTION_BUDGET,
    )

    assert table.counts == {"both": 393, "only_dense": 94, "only_bm25": 41, "neither": 72}
    assert table.n_questions == config.N_DEV


def test_the_seventy_two_are_named_so_phase_4_can_be_asked_about_them():
    """HU-8 asks how many of these System B recovers, which needs the list, not the count."""
    dense, bm25, dev, token_counts = the_real_dev_split()

    table = cross_tabulate(
        per_question_success(dense, dev, token_counts, config.SELECTION_BUDGET, top_k=100),
        per_question_success(bm25, dev, token_counts, config.SELECTION_BUDGET, top_k=100),
        names=("dense", "bm25"),
        budget=config.SELECTION_BUDGET,
    )

    assert len(table.neither) == 72
    assert len(set(table.neither)) == 72
    assert set(table.neither) <= {question.qid for question in dev}
