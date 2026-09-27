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

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.embeddings.backend import EmbeddingBackend
from concept_embeddings_rag.evaluation import phase12, second_hop
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
            candidates, positives = self._hop(p1, rows, top_k)
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

    def _hop(self, p1: int, rows: list[int], top_k: int) -> tuple[list[RankedCandidate], int]:
        return second_hop.node_hop(
            self._index, self._weights, types=TYPES, p1=p1, read=rows, depth=top_k
        )

    def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
        """`E(q)` as hits, best first: what the fusion receives."""
        expansion = self.expand(first, top_k)
        return [(candidate.unit_id, candidate.score) for candidate in expansion.candidates]


class ColumnwiseEntityHopStage(EntityHopStage):
    """The same stage over `second_hop.node_hop_columnwise`: same output, bit for bit.

    Phase 9, S6: used only when the operational probe shows the reference too slow at
    FullWiki scale. It changes how the scores are computed, never which units are eligible
    or what they score; `tests/retrieval/test_columnwise_hop.py` holds the two equal.
    """

    def __init__(self, index: NodeIndex, weights: np.ndarray) -> None:
        super().__init__(index, weights)
        self._columns = self.columns_for(index)

    @staticmethod
    def columns_for(index: NodeIndex) -> second_hop.ArmColumns:
        return second_hop.arm_columns(index, TYPES)

    def _hop(self, p1: int, rows: list[int], top_k: int) -> tuple[list[RankedCandidate], int]:
        return second_hop.node_hop_columnwise(
            self._index, self._weights, self._columns, p1=p1, read=rows, depth=top_k
        )


# --- Phase 11: a DF cap on P1's seeds, and the question's own entities as seeds ----------

QUESTION_HOP_NAME: str = "question-hop"


def capped_seeds(
    nodes: np.ndarray, mask: np.ndarray, df: np.ndarray, *, cap: int | None
) -> np.ndarray:
    """The arm's nodes among `nodes` whose DF is at most `cap`, ascending; all when uncapped."""
    nodes = np.unique(np.asarray(nodes, dtype=np.int64))
    kept = nodes[mask[nodes]]
    return kept if cap is None else kept[df[kept] <= cap]


class CappedEntityHopStage(EntityHopStage):
    """The Phase 9 hop from P1, seeded only by P1's entity nodes with DF <= `cap` (Phase 11, D1).

    `cap=None` is the Phase 9 hop itself. The column-wise copy and the DF are passed in, so
    several caps over one FullWiki index share them instead of each copying the incidence.
    """

    def __init__(
        self,
        index: NodeIndex,
        weights: np.ndarray,
        *,
        cap: int | None,
        columns: second_hop.ArmColumns,
        df: np.ndarray,
    ) -> None:
        super().__init__(index, weights)
        self.cap = cap
        self._columns = columns
        self._df = df

    def _hop(self, p1: int, rows: list[int], top_k: int) -> tuple[list[RankedCandidate], int]:
        incidence = self._index.incidence
        p1_nodes = incidence.indices[incidence.indptr[p1] : incidence.indptr[p1 + 1]]
        seeds = capped_seeds(p1_nodes, self._columns.mask, self._df, cap=self.cap)
        return second_hop.seeded_hop_columnwise(
            self._index, self._weights, self._columns, seeds=seeds, read=rows, depth=top_k
        )


class QuestionEntityHop:
    """The same hop seeded by the entity nodes GLiNER read in the question (Phase 11, D1).

    The nodes come from the frozen question-entity artifact, keyed by question text; a text
    with no record is refused rather than treated as a question without entities. Dense's
    `read(q)` is excluded exactly as for the P1 hop. No match gives an empty list.
    """

    name: str = QUESTION_HOP_NAME

    def __init__(
        self,
        index: NodeIndex,
        weights: np.ndarray,
        *,
        columns: second_hop.ArmColumns,
        nodes_by_question: Mapping[str, Sequence[int]],
    ) -> None:
        self._index = index
        self._weights = weights
        self._columns = columns
        self._nodes = {
            text: np.asarray(nodes, dtype=np.int64) for text, nodes in nodes_by_question.items()
        }
        self._rows = {unit_id: row for row, unit_id in enumerate(index.unit_ids)}

    def hop(
        self, query: str, first: Sequence[Hit], top_k: int
    ) -> tuple[list[RankedCandidate], int]:
        """The ranked candidates and the positive count before the cut."""
        if top_k <= 0:
            raise EntityHopError(f"top_k must be positive, not {top_k}")
        if query not in self._nodes:
            raise EntityHopError(f"no recorded question entities for {query!r}")
        check_dense_order(first)
        seeds = self._nodes[query]
        if seeds.size == 0:
            return [], 0
        read = []
        for unit_id, _score in first[:READ_DEPTH]:
            row = self._rows.get(unit_id)
            if row is None:
                raise EntityHopError(f"unit {unit_id!r} is not in the node index this hop reads")
            read.append(row)
        return second_hop.seeded_hop_columnwise(
            self._index, self._weights, self._columns, seeds=seeds, read=read, depth=top_k
        )

    def propose(self, query: str, first: Sequence[Hit], top_k: int) -> list[Hit]:
        candidates, _positives = self.hop(query, first, top_k)
        return [(candidate.unit_id, candidate.score) for candidate in candidates]


# --- Phase 12: choosing, with the question, which of P1's entities seed the hop -----------

SEEDED_HOP_NAME: str = "seeded-hop"


class SentenceSource(Protocol):
    """What a seeded hop needs about P1: its sentence texts, and their vectors."""

    def sentences_of(self, unit_id: str) -> Sequence[str]: ...

    def vectors_of(self, unit_id: str) -> np.ndarray: ...


@dataclass(frozen=True)
class SeededExpansion:
    """One seeded hop's whole result: the seeds it chose and what they reached."""

    seeds: tuple[int, ...]
    candidates: tuple[RankedCandidate, ...]
    positives: int
    p1_entity_nodes: int
    located_entity_nodes: int


class SentenceSeededHopStage:
    """The Phase 9 hop from P1, seeded by the P1 entities the question's own sentence
    relevance selects (Phase 12, D1/D2).

    `s` and `exclude` fix one of the 6 seed configurations. At `s = None, exclude = False`
    every one of P1's entity nodes is a seed, unlocated ones included, which is exactly the
    Phase 9 hop's own seed set - so this stage equals `ColumnwiseEntityHopStage` bit for bit
    at that configuration (identity held by `tests/retrieval/test_seed_selection.py`, and
    checked on the real index by the D4 reproduction).

    Sentence vectors are read lazily through `sentences`, and only when `s` is not `None`:
    the identity configuration never needs a question vector or a sentence encode at all.
    """

    name: str = SEEDED_HOP_NAME

    def __init__(
        self,
        index: NodeIndex,
        weights: np.ndarray,
        *,
        columns: second_hop.ArmColumns,
        s: int | None,
        exclude: bool,
        forms: Mapping[int, str],
        sentences: SentenceSource,
        question_vector: Callable[[str], np.ndarray],
    ) -> None:
        self._index = index
        self._weights = weights
        self._columns = columns
        self.s = s
        self.exclude = exclude
        self._forms = forms
        self._sentences = sentences
        self._question_vector = question_vector
        self._rows = {unit_id: row for row, unit_id in enumerate(index.unit_ids)}

    def hop(self, query: str, first: Sequence[Hit], top_k: int) -> SeededExpansion:
        if top_k <= 0:
            raise EntityHopError(f"top_k must be positive, not {top_k}")
        if not first:
            raise EntityHopError("the dense list is empty: a reader who read nothing has no p1")
        check_dense_order(first)

        read: list[int] = []
        for unit_id, _score in first[:READ_DEPTH]:
            row = self._rows.get(unit_id)
            if row is None:
                raise EntityHopError(f"unit {unit_id!r} is not in the node index this hop reads")
            read.append(row)
        p1_unit = first[0][0]
        p1_row = read[0]

        incidence = self._index.incidence
        p1_nodes = incidence.indices[incidence.indptr[p1_row] : incidence.indptr[p1_row + 1]]
        p1_entities = p1_nodes[self._columns.mask[p1_nodes]]
        p1_entity_ids = [int(node) for node in p1_entities]

        sentences = self._sentences.sentences_of(p1_unit)
        located = phase12.locate(p1_entity_ids, self._forms, sentences)
        excluded = (
            phase12.named_in_question(p1_entity_ids, self._forms, query) if self.exclude else set()
        )
        if self.s is None:
            seeds = phase12.choose_seeds(p1_entity_ids, located, [], s=None, exclude=excluded)
        else:
            vectors = self._sentences.vectors_of(p1_unit)
            question_vector = self._question_vector(query)
            rel = (vectors @ question_vector).tolist() if vectors.size else []
            seeds = phase12.choose_seeds(p1_entity_ids, located, rel, s=self.s, exclude=excluded)

        if not seeds:
            return SeededExpansion(
                seeds=(),
                candidates=(),
                positives=0,
                p1_entity_nodes=len(p1_entity_ids),
                located_entity_nodes=len(located),
            )
        candidates, positives = second_hop.seeded_hop_columnwise(
            self._index,
            self._weights,
            self._columns,
            seeds=np.array(seeds, dtype=np.int64),
            read=read,
            depth=top_k,
        )
        return SeededExpansion(
            seeds=tuple(seeds),
            candidates=tuple(candidates),
            positives=positives,
            p1_entity_nodes=len(p1_entity_ids),
            located_entity_nodes=len(located),
        )

    def propose(self, query: str, first: Sequence[Hit], top_k: int) -> list[Hit]:
        expansion = self.hop(query, first, top_k)
        return [(candidate.unit_id, candidate.score) for candidate in expansion.candidates]


class CachedSentences:
    """Serves P1 sentences from the corpus and their vectors from the S2 cache (Phase 12).

    A unit the cache does not hold is refused rather than silently skipped: the dev fit reads
    fixed vectors, never encodes on the fly.
    """

    def __init__(
        self,
        texts_by_unit: Mapping[str, Sequence[str]],
        unit_ids: Sequence[str],
        positions: Sequence[int],
        vectors: np.ndarray,
    ) -> None:
        self._texts = texts_by_unit
        rows: dict[str, list[tuple[int, int]]] = {}
        for row, (unit_id, position) in enumerate(zip(unit_ids, positions, strict=True)):
            rows.setdefault(unit_id, []).append((position, row))
        self._rows = {
            unit_id: [row for _position, row in sorted(entries)]
            for unit_id, entries in rows.items()
        }
        self._vectors = vectors

    def sentences_of(self, unit_id: str) -> Sequence[str]:
        if unit_id not in self._texts:
            raise EntityHopError(f"unit {unit_id!r} is not in the corpus this stage reads")
        return self._texts[unit_id]

    def vectors_of(self, unit_id: str) -> np.ndarray:
        rows = self._rows.get(unit_id)
        if rows is None:
            raise EntityHopError(f"unit {unit_id!r} is not in the sentence cache; rebuild it")
        return self._vectors[rows]


class OnlineSentences:
    """Encodes a unit's sentences on first use and keeps them (Phase 12, the S6 pass only).

    `calls` holds one entry per `vectors_of` call, in order: the seconds spent encoding, or
    0.0 for a unit already seen, so the pass can time the encoding separately from `retrieve`
    (D8) - a real system with a small cache would reuse the same P1 the same way.
    """

    def __init__(
        self, texts_by_unit: Mapping[str, Sequence[str]], backend: EmbeddingBackend
    ) -> None:
        self._texts = texts_by_unit
        self._backend = backend
        self._vectors: dict[str, np.ndarray] = {}
        self.calls: list[float] = []

    def sentences_of(self, unit_id: str) -> Sequence[str]:
        if unit_id not in self._texts:
            raise EntityHopError(f"unit {unit_id!r} is not in the corpus this stage reads")
        return self._texts[unit_id]

    def vectors_of(self, unit_id: str) -> np.ndarray:
        cached = self._vectors.get(unit_id)
        if cached is not None:
            self.calls.append(0.0)
            return cached
        sentences = self.sentences_of(unit_id)
        started = time.perf_counter()
        vectors = self._backend.encode(list(sentences))
        self.calls.append(time.perf_counter() - started)
        self._vectors[unit_id] = vectors
        return vectors
