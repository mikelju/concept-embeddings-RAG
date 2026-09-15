"""The second-hop protocol: when dense has read its top-10, where does a hop from `p1` lead?

The Phase 4 diagnostic defined it as a script. Phase 5 measures new hops against the same
protocol, so the protocol moves here and the script becomes one of its callers (decision
D8 of the Phase 5 plan). Its pieces:

- the **dense order** of the pool for a question, and the **read** top-10 a reader has seen;
- **`p1`**, the first paragraph of that order, which every hop starts from;
- a **ranking** that never returns a read paragraph, to a declared depth;
- the five hops the diagnostic compared: continue the dense list, BM25 on the question,
  dense neighbours of `p1`, BM25 with `p1`'s text, and the pooled-embedding concept hop.

Nothing here reads a gold annotation: the protocol decides where a hop leads, and the
caller decides what counts as found. Ties in the existing hops break exactly as the
diagnostic broke them, because its figures must reproduce to the last digit.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Protocol

import numpy as np
from scipy import sparse

# The hops the diagnostic compared that owe nothing to a concept space, by the names its
# output uses. The gate's comparator is one of them (`config.GATE_COMPARATOR`).
DENSE_CONTINUE = "dense-continue"
BM25_QUESTION = "bm25-question"
DENSE_NEIGHBOURS = "dense-neighbours(p1)"
BM25_FOLLOW = "bm25-follow(p1)"
BASELINE_HOPS: tuple[str, ...] = (DENSE_CONTINUE, BM25_QUESTION, DENSE_NEIGHBOURS, BM25_FOLLOW)


class TextRanker(Protocol):
    """What the protocol needs from BM25: a ranking of unit ids for a text."""

    def retrieve(self, query: str, top_k: int) -> Sequence[tuple[str, float]]: ...


def dense_order(vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Every pool row by descending dense score; equal scores keep pool order."""
    return np.argsort(-(vectors @ query_vector), kind="stable")


def read_rows(order: np.ndarray, depth: int) -> list[int]:
    """The paragraphs a reader has already read: the head of the dense order."""
    return [int(row) for row in order[:depth]]


def origin(read: Sequence[int]) -> int:
    """`p1`: the first paragraph read, and the one every hop starts from."""
    if not read:
        raise ValueError("a reader who has read nothing has no paragraph to hop from")
    return int(read[0])


def rank_scores(
    scores: np.ndarray,
    read: Collection[int],
    *,
    positive_only: bool,
    depth: int,
) -> list[int]:
    """Rows by descending score to `depth`, read rows excluded.

    Ties resolve as `argpartition` followed by a stable sort resolves them, which is what
    the diagnostic did. That is kept deliberately, for reproduction; the text-derived arms
    of Phase 5 do not use this function for their ranking (decision D8).
    """
    ranked = scores.astype(np.float64, copy=True)
    ranked[list(read)] = -np.inf
    if positive_only:
        ranked[ranked <= 0.0] = -np.inf
    top = np.argpartition(-ranked, depth)[:depth]
    top = top[np.argsort(-ranked[top], kind="stable")]
    return [int(row) for row in top if np.isfinite(ranked[row])]


def dense_continue(order: np.ndarray, *, read_depth: int, depth: int) -> list[int]:
    """No hop at all: keep reading the dense list where the reader stopped."""
    return [int(row) for row in order[read_depth : read_depth + depth]]


def rank_bm25(
    bm25: TextRanker,
    index: Mapping[str, int],
    text: str,
    read: Collection[int],
    *,
    read_depth: int,
    depth: int,
) -> list[int]:
    """BM25 on a text, read paragraphs and non-positive scores dropped, to `depth`.

    Asked for `depth + read_depth + 1` so that even a ranking that begins with every read
    paragraph still yields `depth` candidates.
    """
    hits = bm25.retrieve(text, top_k=depth + read_depth + 1)
    rows = [index[unit_id] for unit_id, score in hits if score > 0.0 and index[unit_id] not in read]
    return rows[:depth]


def dense_neighbours(
    vectors: np.ndarray, p1: int, read: Collection[int], *, depth: int
) -> list[int]:
    """The paragraphs whose embedding is closest to `p1`'s."""
    return rank_scores(vectors @ vectors[p1], read, positive_only=False, depth=depth)


def truncate_rows(X: sparse.csr_matrix, m: int | None) -> sparse.csr_matrix:
    """Describe every paragraph by its `m` strongest concepts and nothing else."""
    if m is None:
        return X
    truncated = X.tocsr(copy=True)
    for row in range(truncated.shape[0]):
        start, end = truncated.indptr[row], truncated.indptr[row + 1]
        if end - start > m:
            values = truncated.data[start:end]
            weakest = np.argsort(values)[: (end - start) - m]
            values[weakest] = 0.0
    truncated.eliminate_zeros()
    return truncated


def concept_hop(
    X: sparse.csr_matrix,
    weights: np.ndarray,
    *,
    p1: int,
    read: Collection[int],
    depth: int,
) -> tuple[list[int], float]:
    """The pooled-embedding concept hop: `score(v) = sum_c X[v, c] w_c X[p1, c]`.

    Returns the ranking of positive-scoring paragraphs and the **reach**: the share of the
    whole pool, `p1` and the read rows included, that receives a positive score.
    """
    row = X.getrow(p1)
    concepts = np.zeros(X.shape[1])
    concepts[row.indices] = row.data * weights[row.indices]
    scores = np.asarray(X @ concepts).ravel()
    reach = float((scores > 0).sum() / X.shape[0])
    return rank_scores(scores, read, positive_only=True, depth=depth), reach
