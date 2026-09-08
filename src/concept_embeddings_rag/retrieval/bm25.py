"""BM25 retrieval over the same pool, behind the same interface.

BM25 is the baseline that keeps the project honest: on multi-hop benchmarks it
routinely beats dense retrieval, and a conceptual system that does not clear it
has demonstrated nothing.

It indexes exactly the same text the dense side embeds (`indexable_text`), so any
difference in the results comes from the method and not from one of them having
been fed more of the document.
"""

from collections.abc import Sequence

import bm25s

from concept_embeddings_rag.corpus.pool import IndexingUnit
from concept_embeddings_rag.retrieval.base import Hit


class BM25Retriever:
    name = "bm25"

    def __init__(self, units: Sequence[IndexingUnit], stopwords: str = "en") -> None:
        self.unit_ids = [unit.unit_id for unit in units]
        self.stopwords = stopwords

        corpus_tokens = bm25s.tokenize(
            [unit.indexable_text for unit in units],
            stopwords=stopwords,
            show_progress=False,
        )
        self._index = bm25s.BM25()
        self._index.index(corpus_tokens, show_progress=False)

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        k = min(top_k, len(self.unit_ids))
        query_tokens = bm25s.tokenize(query, stopwords=self.stopwords, show_progress=False)
        indices, scores = self._index.retrieve(query_tokens, k=k, show_progress=False)

        # Hits scoring zero are kept: they occupy the tail of the ranking and are
        # never gold, so dropping them would only make BM25 look artificially
        # precise without changing recall.
        return [
            (self.unit_ids[int(index)], float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
        ]
