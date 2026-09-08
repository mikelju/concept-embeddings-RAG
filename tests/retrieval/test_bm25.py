"""T9: BM25 over the same pool, behind the same interface.

BM25 is not a formality: on HotpotQA it is a hard baseline, and a conceptual
system that cannot beat it has not proved anything.
"""

from concept_embeddings_rag.corpus.pool import IndexingUnit
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.bm25 import BM25Retriever


def a_retriever() -> BM25Retriever:
    units = [
        IndexingUnit(
            unit_id="lighthouse",
            title="Torre Vieja",
            sentences=("Torre Vieja is a lighthouse built from granite.",),
        ),
        IndexingUnit(
            unit_id="museum",
            title="Museo del Mar",
            sentences=("The museum exhibits maritime lamps and charts.",),
        ),
        IndexingUnit(
            unit_id="cooking",
            title="Paella",
            sentences=("Paella is a rice dish from Valencia.",),
        ),
    ]
    return BM25Retriever(units)


def test_an_exact_term_match_ranks_first():
    hits = a_retriever().retrieve("granite lighthouse", top_k=3)
    assert hits[0][0] == "lighthouse"


def test_an_unrelated_query_does_not_rank_the_lighthouse_first():
    hits = a_retriever().retrieve("rice dish Valencia", top_k=3)
    assert hits[0][0] == "cooking"


def test_results_are_deterministic():
    retriever = a_retriever()
    assert retriever.retrieve("maritime lamps", top_k=3) == retriever.retrieve(
        "maritime lamps", top_k=3
    )


def test_top_k_is_respected():
    assert len(a_retriever().retrieve("lighthouse", top_k=2)) == 2


def test_a_query_matching_nothing_returns_no_crash():
    hits = a_retriever().retrieve("zzzz nonexistent token", top_k=3)
    assert isinstance(hits, list)


def test_it_satisfies_the_retriever_interface():
    retriever: Retriever = a_retriever()
    assert retriever.name == "bm25"
