"""Retrieval over the corpus-induced concept space (System B).

The question never becomes a word here. It arrives as an embedding and enters the
concept space by geometry alone: either projected onto the dictionary atoms or
coded against them under the same L1 penalty the corpus was coded with. Nothing in
this module reads a concept label, a question string, a split or a gold annotation -
that is what makes the conceptual score independent of the dense one it is later
fused with.

Decision D4 of the phase plan makes truncation part of the operator rather than a
knob swept beside it, so the three arms measured are one decision with three
alternatives. `resolve_query_operator` is the only place that mapping exists.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.concepts.coding import ConceptMatrix
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary
from concept_embeddings_rag.embeddings.backend import EmbeddingBackend
from concept_embeddings_rag.retrieval.base import Hit

# The two operators the spec records in `ConceptualRetriever.query_operator`. The
# arm names of `config.QUERY_OPERATORS` are what a sweep iterates over; these are
# what a result file says.
PROJECTION = "projection"
SPARSE_CODING = "sparse_coding"

# The two arms of `config.DAMPING_MODES`, named so that nothing has to spell either
# of them inline: `UNDAMPED` is the identity, `RARITY` the declared IDF-shaped weight
# of decision D7.
UNDAMPED, RARITY = config.DAMPING_MODES


@dataclass(frozen=True)
class QueryOperator:
    """One arm of decision D4, resolved into what the spec actually records.

    `arm` is the name swept and reported; `operator` and `top_m` are the parameters
    that produced the vector. Both travel together so a recorded configuration can
    never claim an arm whose truncation it did not use.
    """

    arm: str
    operator: str
    top_m: int | None


_ARMS: dict[str, QueryOperator] = {
    "projection_top16": QueryOperator(
        arm="projection_top16", operator=PROJECTION, top_m=config.QUERY_TOP_M
    ),
    "projection_full": QueryOperator(arm="projection_full", operator=PROJECTION, top_m=None),
    "sparse_coding": QueryOperator(arm="sparse_coding", operator=SPARSE_CODING, top_m=None),
}


def resolve_query_operator(arm: str) -> QueryOperator:
    """Turn one of `config.QUERY_OPERATORS` into the parameters it stands for."""
    resolved = _ARMS.get(arm)
    if resolved is None:
        raise ValueError(f"unknown query operator arm: {arm!r}; expected one of {sorted(_ARMS)}")
    return resolved


def query_concepts(
    vector: np.ndarray,
    dictionary: ConceptDictionary,
    operator: str,
    top_m: int | None,
    alpha: float | None,
) -> np.ndarray:
    """Express one query embedding as a non-negative vector over `dictionary`.

    Two inputs and nothing else: a vector and a dictionary. Every parameter is
    passed in explicitly rather than defaulted here, so the caller that records a
    configuration is the caller that chose it.

    `projection` is `atoms @ q` clipped at zero, optionally keeping only the `top_m`
    largest coordinates. Clipping before truncating and truncating before clipping
    give the same answer - clipping is order-preserving - so the order is a matter
    of reading, not of result. `sparse_coding` runs the dictionary's own coder at
    the penalty it is given, which is the corpus coding alpha of the matrix being
    searched: coding the query more or less sparsely than the corpus would compare
    two different resolutions.
    """
    if vector.ndim != 1:
        raise ValueError(f"a query is one vector: expected a 1-D array, got shape {vector.shape}")
    if vector.shape[0] != dictionary.dim:
        raise ValueError(
            f"query dimension {vector.shape[0]} does not match the dictionary dimension "
            f"{dictionary.dim}"
        )

    if operator == PROJECTION:
        if alpha is not None:
            raise ValueError("projection takes no coding alpha; it would have been ignored")
        return _project(vector, dictionary, top_m)

    if operator == SPARSE_CODING:
        if top_m is not None:
            raise ValueError("sparse coding takes no top_m; sparsity comes from the penalty")
        if alpha is None:
            raise ValueError("sparse coding needs the coding alpha of the matrix being searched")
        return _code(vector, dictionary, alpha)

    raise ValueError(
        f"unknown query operator: {operator!r}; expected {PROJECTION!r} or {SPARSE_CODING!r}"
    )


def _project(vector: np.ndarray, dictionary: ConceptDictionary, top_m: int | None) -> np.ndarray:
    scores = np.maximum(dictionary.atoms @ vector.astype(np.float32), 0.0)
    if top_m is None:
        return scores

    if not 1 <= top_m <= dictionary.k:
        raise ValueError(f"top_m must be between 1 and the dictionary size {dictionary.k}: {top_m}")

    # A stable sort rather than `argpartition`: at K <= 4096 a full sort costs
    # nothing, and ties then break by concept index instead of by whatever order
    # the partition happened to leave them in. Determinism is the point.
    keep = np.argsort(-scores, kind="stable")[:top_m]
    truncated = np.zeros_like(scores)
    truncated[keep] = scores[keep]
    return truncated


def _code(vector: np.ndarray, dictionary: ConceptDictionary, alpha: float) -> np.ndarray:
    codes = dictionary.encode(vector.astype(np.float32).reshape(1, -1), alpha=alpha)
    return np.asarray(codes[0], dtype=np.float32)


def concept_support(X: sparse.csr_matrix) -> np.ndarray:
    """How many units activate each concept: the `df_j` of decision D7.

    Read off the sparsity pattern of the matrix Phase 2 wrote and this phase loads
    hash-verified - nothing is recounted from the embeddings and nothing is recoded.
    Stored zeros are cleared first: a coder can leave an explicit zero behind, and a
    concept whose weight is zero is not a concept the unit activates.

    D7 names the Phase 2 diagnostics as the source, and the intent is respected
    exactly; what the persisted `SpaceDiagnostics` keeps under `units_per_concept`
    is the five-number summary rather than the vector, so the vector is read here
    and `tests/retrieval/test_damping.py` asserts it reproduces that summary.
    """
    stored = sparse.csr_matrix(X, copy=True)
    stored.eliminate_zeros()
    return np.bincount(stored.indices, minlength=stored.shape[1]).astype(np.int64)


def rarity_weights(support: np.ndarray, n_units: int) -> np.ndarray:
    """Decision D7's damping: `w_j = log(1 + N / (1 + df_j))`, one weight per concept.

    Declared before any number was seen, and shaped so that the two failure modes it
    could have are impossible rather than unlikely. `1 + df` in the denominator makes
    a concept no unit activates an ordinary case instead of a division by zero; the
    outer `1 +` keeps every weight strictly positive, so the hub is demoted and never
    deleted. Pruning concepts is a declared Phase 5 ablation, and a weight that could
    reach zero would smuggle it into this phase.

    The vector multiplies the query's concept vector, which is algebraically the same
    as scaling the columns of `X` but leaves the matrix and the dictionary exactly as
    they are on disk and in memory - the property HU-6 asks for.
    """
    df = np.asarray(support)
    if df.ndim != 1:
        raise ValueError(
            f"the support is one number per concept: expected a 1-D array, got shape {df.shape}"
        )
    if n_units <= 0:
        raise ValueError(f"the pool size must be positive: {n_units}")
    if df.size and df.min() < 0:
        raise ValueError(f"a concept support cannot be negative: {int(df.min())}")
    if df.size and df.max() > n_units:
        raise ValueError(
            f"a concept support of {int(df.max())} exceeds the pool size {n_units}; the "
            "support and the matrix it was read from describe different pools"
        )

    return np.log1p(n_units / (1.0 + df.astype(np.float64))).astype(np.float32)


class ConceptualRetriever:
    """System B: ranking by how much a unit and a question share concepts.

    The score is `X @ c`, a dot product and deliberately not a cosine (decision D2).
    A unit that activates a queried concept strongly should outrank one that barely
    activates it; removing that scale is what the `row_normalized` view is for, and
    which of the two wins is measured on dev rather than assumed here.

    Everything the retriever holds is the concept space: the matrix, the dictionary
    that induced it, and a backend that turns a question into an embedding. There is
    no unit embedding to fall back on and no lexical index to blend in, which is the
    only reason the fused numbers of HU-4 can be read as a conceptual contribution.
    """

    name = "conceptual"

    def __init__(
        self,
        matrix: ConceptMatrix,
        dictionary: ConceptDictionary,
        backend: EmbeddingBackend,
        *,
        arm: str,
        damping: str = UNDAMPED,
        concept_weights: np.ndarray | None = None,
    ) -> None:
        if matrix.dictionary_key != dictionary.key:
            raise ValueError(
                f"the matrix was coded against dictionary {matrix.dictionary_key!r}, "
                f"not {dictionary.key!r}"
            )
        if matrix.X.shape[1] != dictionary.k:
            raise ValueError(
                f"the matrix has {matrix.X.shape[1]} concepts but the dictionary has "
                f"{dictionary.k} atoms"
            )
        if matrix.X.shape[0] != len(matrix.unit_ids):
            raise ValueError(f"{matrix.X.shape[0]} rows for {len(matrix.unit_ids)} unit ids")
        if matrix.view not in config.CONCEPT_VIEWS:
            raise ValueError(
                f"unknown concept matrix view: {matrix.view!r}; "
                f"expected one of {sorted(config.CONCEPT_VIEWS)}"
            )

        operator = resolve_query_operator(arm)
        self._weights = _resolve_damping(damping, concept_weights, dictionary.k)

        self.matrix = matrix
        self.dictionary = dictionary
        self.backend = backend
        self.unit_ids = list(matrix.unit_ids)

        self.arm = operator.arm
        self.query_operator = operator.operator
        self.query_top_m = operator.top_m
        # Derived rather than passed: the query is coded at the resolution the corpus
        # was coded at, by construction, so a recorded alpha cannot disagree with the
        # matrix it was measured against.
        self.query_alpha = matrix.coding_alpha if operator.operator == SPARSE_CODING else None

        self.dictionary_key = matrix.dictionary_key
        self.k = dictionary.k
        self.view = matrix.view
        self.damping = damping
        self.concept_weights = self._weights

    def describe(self) -> dict:
        """The spec's data contract, to be recorded beside every metric."""
        return {
            "name": self.name,
            "dictionary_key": self.dictionary_key,
            "k": self.k,
            "view": self.view,
            "query_operator": self.query_operator,
            "query_alpha": self.query_alpha,
            "query_top_m": self.query_top_m,
            "damping": self.damping,
        }

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        query_vector = self.backend.encode([query])[0]
        concepts = query_concepts(
            query_vector,
            self.dictionary,
            self.query_operator,
            top_m=self.query_top_m,
            alpha=self.query_alpha,
        )
        if self._weights is not None:
            # Decision D7 applies rarity to the query, so `X` and the atoms stay the
            # artifacts Phase 2 wrote and every damped run is comparable to its own
            # undamped one without recoding anything.
            concepts = concepts * self._weights

        scores = np.asarray(self.matrix.X @ concepts).ravel()

        k = min(top_k, len(self.unit_ids))
        if k < len(self.unit_ids):
            candidates = np.argpartition(-scores, k - 1)[:k]
        else:
            candidates = np.arange(len(self.unit_ids))

        ordered = sorted(candidates, key=lambda i: (-float(scores[i]), self.unit_ids[i]))
        return [(self.unit_ids[i], float(scores[i])) for i in ordered]


def _resolve_damping(damping: str, concept_weights: np.ndarray | None, k: int) -> np.ndarray | None:
    """Check that the mode and the weights say the same thing, and return the vector.

    A retriever that records a damping it did not apply - or applies one it does not
    record - produces a mislabelled result, which is worse than a missing one.
    """
    if damping not in config.DAMPING_MODES:
        raise ValueError(
            f"unknown damping mode: {damping!r}; expected one of {sorted(config.DAMPING_MODES)}"
        )
    if damping == UNDAMPED:
        if concept_weights is not None:
            raise ValueError("concept weights were given but the damping mode is 'none'")
        return None

    if concept_weights is None:
        raise ValueError(f"damping {damping!r} names a weighting but no concept weights were given")

    # Copied rather than referenced: a weight vector the caller can still edit would
    # let a recorded configuration drift away from the run that produced it.
    weights = np.array(concept_weights, dtype=np.float32, copy=True)
    if weights.shape != (k,):
        raise ValueError(
            f"concept weights have shape {weights.shape} but the dictionary has {k} concepts"
        )
    return weights
