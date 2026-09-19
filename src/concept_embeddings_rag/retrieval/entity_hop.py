"""The entity hop of Phase 5 as a second-stage candidate generator (Phase 6, decision D6).

It never reads the question. For a question `q` it is handed `D(q)`, the dense list that
enters the fusion, and it proposes the paragraphs linked to dense's first paragraph by shared
entity names:

1. `read(q)` is the first `PILOT_READ_DEPTH` units of `D(q)`, and `p1(q)` the first of them;
2. `E(q)` is `second_hop.node_hop` over the **entity** nodes only, unchanged: a candidate
   scores the sum of the rarity weights of the entity nodes it shares with `p1`, every read
   unit is excluded, only positive scores are kept, ties break by unit id, and the list is cut
   at the depth the hybrid asks for. `P(q)` is the number of positive candidates before that
   cut. Nothing is padded: fewer positives give a shorter list, none gives an empty one.

The constructor takes the node index and the weights and nothing else. There is no
threshold, no minimum score, no expansion, no re-admission of read units, and no
canonicalization; the types come from `config.ENTITY_HOP_TYPES`.

The stage checks that it was handed a list in the dense order - descending score, ties by
unit id, no unit twice - rather than re-sorting it: it is conditioned on `D(q)` itself, and a
list in any other order would be a different `read(q)`.

`E(q)` is a pure function of the read list and the depth, and a fit asks the same question
once per grid point with the same `D(q)`, so expansions are memoized by `(read, depth)`. The
memo returns copies, so no caller can alter what the next one receives.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import second_hop
from concept_embeddings_rag.evaluation.second_hop import RankedCandidate
from concept_embeddings_rag.nodes.index import NodeIndex
from concept_embeddings_rag.retrieval.base import Hit

TYPES: tuple[str, ...] = config.ENTITY_HOP_TYPES
READ_DEPTH: int = config.PILOT_READ_DEPTH


class EntityHopError(ValueError):
    """The stage was handed something that is not a dense list it can be conditioned on."""


@dataclass(frozen=True)
class EntityExpansion:
    """Everything one hop produced for one dense list: the path, not only the ranking."""

    p1: str
    read: tuple[str, ...]
    candidates: tuple[RankedCandidate, ...]
    positives: int
    p1_entity_nodes: int


def check_dense_order(first: Sequence[Hit]) -> None:
    """Refuse a list that is not in the dense order: descending score, ties by unit id."""
    seen: set[str] = set()
    for position, (unit_id, _score) in enumerate(first):
        if unit_id in seen:
            raise EntityHopError(f"the dense list holds {unit_id!r} twice")
        seen.add(unit_id)
        if position == 0:
            continue
        previous_id, previous_score = first[position - 1]
        score = first[position][1]
        in_order = previous_score > score or (previous_score == score and previous_id < unit_id)
        if not in_order:
            raise EntityHopError(
                f"the dense list is not in the dense order at position {position}: "
                f"{previous_id!r} ({previous_score!r}) before {unit_id!r} ({score!r})"
            )


class EntityHopStage:
    """`D(q)` in, `E(q)` out, through the unchanged Phase 5 node hop."""

    name: str = config.ENTITY_HOP_NAME

    def __init__(self, index: NodeIndex, weights: np.ndarray) -> None:
        if weights.shape[0] != len(index.nodes):
            raise EntityHopError(
                f"{weights.shape[0]} weights for {len(index.nodes)} nodes; the weights belong to "
                "another index"
            )
        self._index = index
        self._weights = weights
        self._rows = {unit_id: row for row, unit_id in enumerate(index.unit_ids)}
        entity = np.zeros(len(index.nodes), dtype=bool)
        entity[index.columns_of(TYPES)] = True
        self._entity_columns = entity
        self._memo: dict[tuple[tuple[str, ...], int], EntityExpansion] = {}

    @property
    def memo_size(self) -> int:
        return len(self._memo)

    def _row_of(self, unit_id: str) -> int:
        row = self._rows.get(unit_id)
        if row is None:
            raise EntityHopError(f"unit {unit_id!r} is not in the node index this stage reads")
        return row

    def expand(self, first: Sequence[Hit], top_k: int) -> EntityExpansion:
        """The whole hop for one dense list: `p1`, `read`, the ranked candidates and `P(q)`."""
        if top_k <= 0:
            raise EntityHopError(f"top_k must be positive, not {top_k}")
        if not first:
            raise EntityHopError("the dense list is empty: a reader who read nothing has no p1")
        check_dense_order(first)

        read = tuple(unit_id for unit_id, _score in first[:READ_DEPTH])
        key = (read, top_k)
        cached = self._memo.get(key)
        if cached is None:
            rows = [self._row_of(unit_id) for unit_id in read]
            p1 = second_hop.origin(rows)
            candidates, positives = second_hop.node_hop(
                self._index, self._weights, types=TYPES, p1=p1, read=rows, depth=top_k
            )
            incidence = self._index.incidence
            p1_nodes = incidence.indices[incidence.indptr[p1] : incidence.indptr[p1 + 1]]
            cached = EntityExpansion(
                p1=read[0],
                read=read,
                candidates=tuple(candidates),
                positives=positives,
                p1_entity_nodes=int(self._entity_columns[p1_nodes].sum()),
            )
            self._memo[key] = cached
        return replace(cached)

    def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
        """`E(q)` as hits, best first: what the fusion receives."""
        expansion = self.expand(first, top_k)
        return [(candidate.unit_id, candidate.score) for candidate in expansion.candidates]
