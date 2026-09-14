"""Expansion by diffusion over `X` (System C).

Phase 3 scored the concept space once per question and stopped, and the fitted
fusion weight came back at 1.0: nothing the concept space said was worth adding to
dense. This module does the other thing the space allows - it **walks**.

A question arrives as its seed retriever's ranking. That ranking becomes a mass
vector over the 19,366 units, and each round moves the mass unit -> concept ->
unit through `X` and `X`ᵀ, returning a declared fraction of it to the seed's own
signal every time. A unit the seed never returned can end up high if it shares
concepts with units the seed did return, which is the second hop no one-hop
retriever has a mechanism for, whatever space it scores in.

Three properties decide whether the numbers that come out mean anything, and all
three are decisions of the phase plan rather than implementation details:

- **The unit x unit matrix is never formed** (D2). `X W X`ᵀ has 375 million entries
  at this scale. Associativity makes the same arithmetic two sparse products, and
  this module never writes the other order.
- **At `restart = 1.0` the walk is its seed**, exactly (D1). That is what makes
  every later difference attributable to the expansion rather than to a pipeline
  that changed underneath it.
- **The rarity weight is applied to the concept vector in the middle of the round
  trip** (D2), which is where Phase 3 applies it, so the matrix and the atoms stay
  the artifacts Phase 2 wrote.

Nothing here reads a concept label, a gold annotation or a split.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of
from concept_embeddings_rag.concepts.coding import ConceptMatrix
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary
from concept_embeddings_rag.retrieval.base import Hit, Retriever

# The three arms of decision D3, named so nothing has to spell them inline.
NONE, SYMMETRIC, STOCHASTIC = config.NORMALIZATION_ARMS


class DiffusionError(Exception):
    """The diffusion was asked for something its inputs cannot support."""


def seed_mass(
    hits: Sequence[Hit],
    unit_index: Mapping[str, int],
) -> tuple[np.ndarray, int]:
    """Turn a seed retriever's hit list into the mass the walk starts from.

    Decision D1: the walk restarts onto this same vector, so it is the seed's own
    signal that the expansion keeps returning to - not the query's concept vector,
    which the dense arm's seed never used and which would have made the two arms
    two systems instead of one system with two seeds.

    Clipping at zero and L1-normalizing is the whole transformation. Normalizing by
    a positive scalar preserves the seed's order exactly, which is what makes
    `restart = 1.0` reproduce the seed rather than merely resemble it.

    Returns the mass and **how many hits were clipped**. A dense cosine can in
    principle be negative; on the top-100 of this corpus it is expected never to
    happen, so the count is recorded per run and a non-zero one is a finding to be
    written down rather than a repair to be performed silently.
    """
    if not hits:
        raise DiffusionError("the seed retriever returned no hits: there is no mass to start from")

    mass = np.zeros(len(unit_index), dtype=np.float64)
    clipped = 0
    seen: set[str] = set()
    for unit_id, score in hits:
        row = unit_index.get(unit_id)
        if row is None:
            raise DiffusionError(
                f"the seed returned unit {unit_id!r}, which is not in the pool this "
                "diffusion runs over; the two were built from different corpora"
            )
        if unit_id in seen:
            raise DiffusionError(
                f"the seed returned unit {unit_id!r} twice; one unit is one row of the "
                "mass vector, and adding a score twice invents mass"
            )
        seen.add(unit_id)
        if score > 0.0:
            mass[row] = float(score)
        else:
            clipped += 1

    total = float(mass.sum())
    if total <= 0.0:
        raise DiffusionError(
            f"the seed returned {len(hits)} hits and no positive score among them; "
            "spreading the mass uniformly instead would be a different system"
        )
    return mass / total, clipped


class DiffusionOperator:
    """One round of `unit -> concept -> unit`, as two sparse products.

    The two scaled copies of `X` are built once, at construction, because a sweep
    measures four restart fractions against the same normalization and rescaling
    the matrix per question would dominate the cost of the whole phase.

    Decision D3 renormalizes the propagated vector to sum to one before it is mixed
    with the restart, so that `restart` is a genuine fraction of unit mass and the
    stop threshold means the same thing in all three arms. `propagate_raw` is the
    same round without that last step, kept because the conservation property of the
    stochastic arm is only visible before it.
    """

    def __init__(
        self,
        X: sparse.csr_matrix,
        *,
        normalization: str,
        concept_weights: np.ndarray | None = None,
    ) -> None:
        if normalization not in config.NORMALIZATION_ARMS:
            raise DiffusionError(
                f"unknown normalization arm: {normalization!r}; expected one of "
                f"{sorted(config.NORMALIZATION_ARMS)}"
            )

        matrix = sparse.csr_matrix(X, copy=True)
        matrix.eliminate_zeros()
        n_units, n_concepts = matrix.shape

        weights = _resolve_weights(concept_weights, n_concepts)

        degree_units = np.asarray(matrix.sum(axis=1)).ravel()
        degree_concepts = np.asarray(matrix.sum(axis=0)).ravel()
        # Checked only where the arm would actually divide by a degree. `none` does
        # not, so an empty row is not an error there - it is the invariant the spec
        # states, that a unit no concept reaches gains no mass, and it is asserted in
        # `tests/retrieval/test_diffusion_invariants.py` rather than made unreachable.
        # Phase 2 measured zero orphan units and zero dead atoms at every K, so on the
        # real space neither branch fires.
        if normalization != NONE:
            _check_degrees(degree_units, "unit", "row of X is all zeros")
            _check_degrees(degree_concepts, "concept", "column of X is all zeros")

        # `left` is applied transposed (units -> concepts) and `right` as it stands
        # (concepts -> units). For two of the arms they are the same matrix; for the
        # random walk they are scaled on different sides, which is exactly what makes
        # it a walk rather than a rescaling.
        if normalization == NONE:
            left = matrix
            right = matrix
        elif normalization == SYMMETRIC:
            scaled = _scale(matrix, 1.0 / np.sqrt(degree_units), 1.0 / np.sqrt(degree_concepts))
            left = scaled
            right = scaled
        else:
            left = _scale(matrix, 1.0 / degree_units, None)
            right = _scale(matrix, None, 1.0 / degree_concepts)

        self.normalization = normalization
        self.left = left
        self.right = right
        self.concept_weights = weights
        self.n_units = int(n_units)
        self.n_concepts = int(n_concepts)

    def to_concepts(self, mass: np.ndarray) -> np.ndarray:
        """The first half of a round: where this mass sits in concept space, damped.

        Exposed rather than kept inside `propagate` because the trace of HU-5 has to
        name the concepts that entered at each round, and a trace that recomputed
        them could disagree with the round it claims to describe.
        """
        self._check_mass(mass)
        concepts = self.left.T @ mass
        if self.concept_weights is not None:
            concepts = concepts * self.concept_weights
        return np.asarray(concepts).ravel()

    def from_concepts(self, concepts: np.ndarray) -> np.ndarray:
        """The second half: the mass those concepts hand back to the units."""
        if concepts.shape != (self.n_concepts,):
            raise DiffusionError(
                f"a concept vector is one value per concept: expected {(self.n_concepts,)}, "
                f"got {concepts.shape}"
            )
        return np.asarray(self.right @ concepts).ravel()

    def propagate_raw(self, mass: np.ndarray) -> np.ndarray:
        """One full round, without the renormalization of decision D3."""
        return self.from_concepts(self.to_concepts(mass))

    def propagate(self, mass: np.ndarray) -> np.ndarray:
        """One full round, renormalized to sum to one."""
        return _normalize(self.propagate_raw(mass))

    def _check_mass(self, mass: np.ndarray) -> None:
        if mass.shape != (self.n_units,):
            raise DiffusionError(
                f"the mass is one value per unit: expected {(self.n_units,)}, got {mass.shape}"
            )


def _normalize(vector: np.ndarray) -> np.ndarray:
    """L1-normalize a non-negative vector, refusing the degenerate case.

    The mass cannot legitimately vanish: the rarity weight is strictly positive by
    construction, every unit activates at least one concept, and the incoming vector
    is non-negative and non-zero. A zero total is therefore a defect somewhere
    upstream, and reporting it as a uniform distribution would hide it inside a
    ranking that still looks plausible.
    """
    total = float(vector.sum())
    if total <= 0.0:
        raise DiffusionError(
            "the propagated mass sums to zero: the walk reached no unit at all, which "
            "the inputs of this phase should make impossible"
        )
    return vector / total


def _scale(
    matrix: sparse.csr_matrix,
    rows: np.ndarray | None,
    columns: np.ndarray | None,
) -> sparse.csr_matrix:
    """`diag(rows) @ matrix @ diag(columns)`, without forming either diagonal."""
    scaled = matrix
    if rows is not None:
        scaled = sparse.diags(rows) @ scaled
    if columns is not None:
        scaled = scaled @ sparse.diags(columns)
    return sparse.csr_matrix(scaled)


def _resolve_weights(concept_weights: np.ndarray | None, n_concepts: int) -> np.ndarray | None:
    if concept_weights is None:
        return None
    # Copied rather than referenced: a weight vector the caller can still edit would
    # let a recorded configuration drift away from the run that produced it.
    weights = np.array(concept_weights, dtype=np.float64, copy=True)
    if weights.shape != (n_concepts,):
        raise DiffusionError(
            f"concept weights have shape {weights.shape} but the matrix has {n_concepts} concepts"
        )
    if (weights < 0.0).any():
        raise DiffusionError(
            "a concept weight cannot be negative: the walk carries mass, and negative "
            "mass is not a thing this phase can interpret"
        )
    return weights


def _check_degrees(degrees: np.ndarray, label: str, cause: str) -> None:
    zeros = int((degrees <= 0.0).sum())
    if zeros:
        raise DiffusionError(
            f"{zeros} {label}(s) have a degree of zero, because the {cause}. Phase 2 measured "
            f"none of these at any K, so this is a broken input rather than a case to handle"
        )


# The two systems of HU-3, named by their seed. The mapping is one-way and total,
# so a result filename can never claim an arm the run did not use.
EXPANSION_NAMES: dict[str, str] = {
    "dense": "expansion",
    "conceptual": "expansion-conceptual",
}

# Why a round ended. `converged` is the degenerate case of `threshold` - the mass
# stopped moving entirely - and is kept apart from it because the two say different
# things about the walk: one settled, the other ran out of patience.
CONVERGED, THRESHOLD, CAP = "converged", "threshold", "cap"
STOP_REASONS: tuple[str, ...] = (CONVERGED, THRESHOLD, CAP)

# How much of a sha256 names a configuration. The same width the dictionary keys of
# Phase 2 use, for the same reason: it has to fit in a filename a human reads.
DIGEST_WIDTH: int = 16
# What stands in for the damping weights in the digest when there are none, so that
# "undamped" is a value the hash sees rather than an absence it cannot distinguish.
UNDAMPED_DIGEST_PART: str = "undamped"

# Below this, two consecutive mass vectors are the same vector and the difference is
# float noise rather than movement.
_EXACT = 1e-12


@dataclass
class RunStats:
    """What one pass over a question set cost, accumulated as it goes.

    Mutable state on a retriever is a smell, and this is the bounded version of it.
    `evaluate_retriever` takes its configuration **before** it runs and the harness is
    frozen for this phase, so a measurement of the run itself has no other door into
    the result (decision D12). The bound is that nothing here is ever read by the path
    that produces a score, and that a caller resets it per run.

    `iterations` is kept per question rather than summarised, because HU-4 asks for the
    distribution: a system that stops at round 1 on 90% of questions and one that runs
    to the cap can share a mean and are not the same system.
    """

    iterations: list[int] = field(default_factory=list)
    stop_reasons: Counter[str] = field(default_factory=Counter)
    sparse_products: int = 0
    clipped_seed_hits: int = 0

    def record(self, *, iterations: int, stop_reason: str, clipped: int) -> None:
        self.iterations.append(iterations)
        self.stop_reasons[stop_reason] += 1
        # Decision D2: one round is exactly two sparse matrix-vector products, and the
        # count is derived from the rounds rather than incremented inside the operator,
        # so an operator that quietly did more work would show up as a disagreement.
        self.sparse_products += 2 * iterations
        self.clipped_seed_hits += clipped

    @property
    def n_questions(self) -> int:
        return len(self.iterations)

    @property
    def mean_iterations(self) -> float:
        return float(sum(self.iterations) / len(self.iterations)) if self.iterations else 0.0

    @property
    def iterations_histogram(self) -> dict[int, int]:
        return dict(sorted(Counter(self.iterations).items()))

    @property
    def cap_hits(self) -> int:
        return int(self.stop_reasons[CAP])

    def describe(self) -> dict:
        """The statistics in the shape decision D12 records them in a result config.

        The histogram's keys become strings because a result is JSON and JSON has no
        integer keys; doing it here rather than at the writing end keeps the artifact
        and this object saying the same thing.
        """
        if not self.iterations:
            raise ValueError(
                "no question was asked of this retriever, so there is nothing to describe; "
                "a mean of zero iterations would read as a measurement rather than an absence"
            )
        return {
            "mean_iterations": self.mean_iterations,
            "iterations_histogram": {str(k): v for k, v in self.iterations_histogram.items()},
            "stop_reasons": dict(sorted(self.stop_reasons.items())),
            "cap_hits": self.cap_hits,
            "clipped_seed_hits": self.clipped_seed_hits,
            "mean_sparse_products": self.sparse_products / len(self.iterations),
            "n_questions": self.n_questions,
        }


@dataclass(frozen=True)
class IterationTrace:
    """One round of one walk, as HU-5 asks it to be readable.

    `concepts` are the dimensions the mass passed through on the way out, damped
    exactly as the operator damped them, and `units` are the ones that **gained**
    mass this round - which is what makes a second hop visible: at round two they
    are units the seed never returned.

    Both are truncated to the widths declared in `config`, because a round touches
    every concept the mass reaches and an artifact holding all of them per question
    per round is one nobody opens.
    """

    index: int
    concepts: tuple[tuple[int, float], ...]
    units: tuple[tuple[str, float], ...]
    new_mass: float
    stopped: bool

    def __post_init__(self) -> None:
        if self.index < 1:
            raise DiffusionError(f"rounds are counted from one, and this one says {self.index}")
        for concept, mass in self.concepts:
            # The one way a label could reach `retrieval/` is from inside the data,
            # so a concept that is not a number is refused where it would enter.
            if not isinstance(concept, int) or isinstance(concept, bool):
                raise DiffusionError(
                    f"a trace holds concept ids, and this round names {concept!r}; a label "
                    "belongs to the report that reads the trace, never to the trace"
                )
            if mass < 0.0:
                raise DiffusionError(f"concept {concept} is recorded with negative mass {mass}")
        for unit_id, mass in self.units:
            if not isinstance(unit_id, str):
                raise DiffusionError(f"a unit is named by its id, and this round names {unit_id!r}")
            if mass <= 0.0:
                raise DiffusionError(
                    f"unit {unit_id!r} is recorded as entering with {mass} mass; a unit that "
                    "gained nothing did not enter"
                )
        if self.new_mass < 0.0:
            raise DiffusionError(f"the mass that moved is a distance, not {self.new_mass}")


@dataclass(frozen=True)
class ExpansionTrace:
    """What one question's walk did, bound to the configuration that walked it.

    `config_digest` is the whole reason this is an artifact rather than a printout:
    a trace read against another run explains a ranking that was never produced,
    in a document that looks exactly as authoritative as a correct one.
    """

    qid: str
    seed_arm: str
    config_digest: str
    iterations: tuple[IterationTrace, ...]
    stop_reason: str

    def __post_init__(self) -> None:
        if not self.qid:
            raise DiffusionError("a trace describes one question and has to name which")
        if self.seed_arm not in config.SEED_ARMS:
            raise DiffusionError(
                f"unknown seed arm: {self.seed_arm!r}; expected one of {sorted(config.SEED_ARMS)}"
            )
        if self.stop_reason not in STOP_REASONS:
            raise DiffusionError(
                f"unknown stop reason: {self.stop_reason!r}; expected one of {sorted(STOP_REASONS)}"
            )
        if not self.iterations:
            raise DiffusionError(
                "every walk runs at least one round, so a trace of no round describes nothing"
            )
        indices = [row.index for row in self.iterations]
        if indices != list(range(1, len(indices) + 1)):
            raise DiffusionError(
                f"the rounds of a walk are consecutive from one, and these are {indices}"
            )
        if not any(row.stopped for row in self.iterations):
            raise DiffusionError("no round of this trace is the one that stopped the walk")


def _top_concepts(vector: np.ndarray, width: int) -> tuple[tuple[int, float], ...]:
    """The concepts this round moved mass through, heaviest first, by id on ties."""
    active = [int(index) for index in np.nonzero(vector > 0.0)[0]]
    active.sort(key=lambda index: (-float(vector[index]), index))
    return tuple((index, float(vector[index])) for index in active[:width])


def _entering_units(
    gained: np.ndarray, unit_ids: Sequence[str], width: int
) -> tuple[tuple[str, float], ...]:
    """The units that gained mass this round, by how much, by id on ties."""
    rows = [int(row) for row in np.nonzero(gained > 0.0)[0]]
    rows.sort(key=lambda row: (-float(gained[row]), unit_ids[row]))
    return tuple((unit_ids[row], float(gained[row])) for row in rows[:width])


def _agreeing(
    operator: DiffusionOperator,
    matrix: ConceptMatrix,
    normalization: str,
    concept_weights: np.ndarray | None,
) -> DiffusionOperator:
    """The operator handed in, if it is the one this retriever would have built."""
    if operator.normalization != normalization:
        raise DiffusionError(
            f"the operator was built for the {operator.normalization!r} arm and this "
            f"retriever records {normalization!r}"
        )
    if (operator.n_units, operator.n_concepts) != tuple(matrix.X.shape):
        raise DiffusionError(
            f"the operator walks a {(operator.n_units, operator.n_concepts)} matrix and this "
            f"retriever was built over {tuple(matrix.X.shape)}"
        )
    theirs, mine = operator.concept_weights, concept_weights
    if (theirs is None) != (mine is None) or (
        theirs is not None and mine is not None and not np.array_equal(theirs, mine)
    ):
        raise DiffusionError(
            "the operator was built with different concept weights from the ones this "
            "retriever records; the damping would not be the one the result claims"
        )
    return operator


class DiffusionRetriever:
    """System C: the ranking a question reaches by walking, not by scoring once.

    Everything it holds is the concept space plus one seed retriever, and the seed
    is the only thing that differs between this phase's two arms (HU-3). That is
    deliberate to the point of being enforced: the name is derived from the arm, so
    a result file cannot record `expansion` while a conceptual seed ran.

    The walk is

        s_0     = the seed's ranking, clipped and L1-normalized (decision D1)
        s_{t+1} = restart * s_0 + (1 - restart) * propagate(s_t)

    and it stops when a round moves less than `stop_threshold` of the mass, or at
    `max_iterations`, whichever comes first. Both mixture terms are non-negative and
    sum to `restart` and `1 - restart`, so `s_{t+1}` is a distribution by
    construction and never needs rescuing with a second normalization.
    """

    def __init__(
        self,
        matrix: ConceptMatrix,
        dictionary: ConceptDictionary,
        seed_retriever: Retriever,
        *,
        seed_arm: str,
        restart: float,
        normalization: str,
        concept_weights: np.ndarray | None = None,
        stop_threshold: float = config.STOP_THRESHOLD,
        max_iterations: int = config.MAX_ITERATIONS,
        seed_top_k: int = config.SEED_TOP_K,
        inherits: Mapping[str, object] | None = None,
        operator: "DiffusionOperator | None" = None,
    ) -> None:
        if seed_arm not in config.SEED_ARMS:
            raise DiffusionError(
                f"unknown seed arm: {seed_arm!r}; expected one of {sorted(config.SEED_ARMS)}"
            )
        if not 0.0 <= restart <= 1.0:
            raise DiffusionError(
                f"the restart is a fraction of the mass: expected it in [0, 1], got {restart}"
            )
        if matrix.dictionary_key != dictionary.key:
            raise DiffusionError(
                f"the matrix was coded against dictionary {matrix.dictionary_key!r}, "
                f"not {dictionary.key!r}"
            )
        if matrix.X.shape[1] != dictionary.k:
            raise DiffusionError(
                f"the matrix has {matrix.X.shape[1]} concepts but the dictionary has "
                f"{dictionary.k} atoms"
            )
        if matrix.X.shape[0] != len(matrix.unit_ids):
            raise DiffusionError(f"{matrix.X.shape[0]} rows for {len(matrix.unit_ids)} unit ids")
        if stop_threshold <= 0.0:
            raise DiffusionError(
                f"the stop threshold is a fraction of the mass: expected it above 0, "
                f"got {stop_threshold}"
            )
        if max_iterations < 1:
            raise DiffusionError(f"the walk runs at least one round, not {max_iterations}")

        self.name = EXPANSION_NAMES[seed_arm]
        self.matrix = matrix
        self.dictionary = dictionary
        self.seed_retriever = seed_retriever
        self.seed_arm = seed_arm
        self.restart = float(restart)
        self.normalization = normalization
        self.stop_threshold = float(stop_threshold)
        self.max_iterations = int(max_iterations)
        self.seed_top_k = int(seed_top_k)
        self.inherits = dict(inherits) if inherits is not None else {}

        self.unit_ids = list(matrix.unit_ids)
        self._index = {unit_id: row for row, unit_id in enumerate(self.unit_ids)}
        # A sweep measures four restart fractions against the same normalization, and
        # rescaling `X` once per cell would compute the same three matrices twelve
        # times. An operator may therefore be handed in - but only one that agrees
        # with everything this retriever claims about itself, because a shared
        # operator is the easiest way for a recorded configuration to stop describing
        # the arithmetic that ran.
        self.operator = (
            DiffusionOperator(
                matrix.X, normalization=normalization, concept_weights=concept_weights
            )
            if operator is None
            else _agreeing(operator, matrix, normalization, concept_weights)
        )
        self.concept_weights = self.operator.concept_weights
        self.stats = RunStats()

    def describe(self) -> dict:
        """The spec's data contract, to be recorded beside every metric."""
        return {
            "name": self.name,
            "seed_arm": self.seed_arm,
            "seed_system": self.seed_retriever.name,
            "restart": self.restart,
            "normalization": self.normalization,
            "stop_threshold": self.stop_threshold,
            "max_iterations": self.max_iterations,
            "seed_top_k": self.seed_top_k,
            "dictionary_key": self.matrix.dictionary_key,
            "k": int(self.matrix.X.shape[1]),
            "view": self.matrix.view,
            "damping": self.inherits.get("damping"),
            "inherits": dict(self.inherits),
        }

    def reset_stats(self) -> None:
        """Start counting a new run. A caller that forgets would report two runs as one."""
        self.stats = RunStats()

    @property
    def config_digest(self) -> str:
        """A short, stable name for exactly this configuration.

        It covers everything `describe()` records plus the damping weights, which
        `describe()` names but does not carry. A trace is read back against this, so
        a digest that ignored part of the configuration would let a diagnostic be
        read against a run it does not describe - which is the one failure a trace
        cannot survive.
        """
        return digest_of(
            json.dumps(self.describe(), sort_keys=True),
            self.concept_weights if self.concept_weights is not None else UNDAMPED_DIGEST_PART,
        )[:DIGEST_WIDTH]

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        hits, mass, _trace = self._walk(query, qid=None)
        return self._rank(mass, hits, top_k)

    def retrieve_with_trace(
        self, query: str, top_k: int, *, qid: str
    ) -> tuple[list[Hit], ExpansionTrace]:
        """The same walk, watched. HU-5 requires the watching to change nothing.

        It cannot, because there is only one walk: `retrieve` and this method run
        the identical arithmetic in the identical order and differ in whether the
        loop also writes down what it just did. The equality of the two rankings is
        asserted in `tests/retrieval/test_trace.py` over every cell of the grid
        anyway, because "cannot" is a claim until something checks it.

        The qid arrives here rather than through `retrieve` because the `Retriever`
        protocol takes a question string and this phase does not touch that
        protocol - and because a trace has to name the question it explains.
        """
        if not qid:
            raise DiffusionError("a trace describes one question and needs its qid to say which")
        hits, mass, trace = self._walk(query, qid=qid)
        if trace is None:  # pragma: no cover - `_walk` returns one whenever a qid is given
            raise DiffusionError("the walk was asked for a trace and produced none")
        return self._rank(mass, hits, top_k), trace

    def _walk(
        self, query: str, *, qid: str | None
    ) -> tuple[list[Hit], np.ndarray, ExpansionTrace | None]:
        """The walk itself: one loop, whether or not anyone is writing it down."""
        hits = self.seed_retriever.retrieve(query, top_k=self.seed_top_k)
        start, clipped = seed_mass(hits, self._index)
        mass = start
        reason = CAP
        rounds = 0
        rows: list[IterationTrace] = []

        for _round in range(self.max_iterations):
            # Decomposed rather than `propagate()` so that the traced path can name
            # the concepts this round moved through without recomputing them. It is
            # the same two products in the same order, so the arithmetic is not
            # merely equivalent to the untraced one - it is the same arithmetic.
            concepts = self.operator.to_concepts(mass)
            propagated = _normalize(self.operator.from_concepts(concepts))
            moved = self.restart * start + (1.0 - self.restart) * propagated
            delta = float(np.abs(moved - mass).sum())
            rounds += 1

            settled = delta <= _EXACT
            stopping = settled or delta < self.stop_threshold or rounds == self.max_iterations
            if qid is not None:
                rows.append(
                    IterationTrace(
                        index=rounds,
                        concepts=_top_concepts(concepts, config.TRACE_TOP_CONCEPTS),
                        units=_entering_units(moved - mass, self.unit_ids, config.TRACE_TOP_UNITS),
                        new_mass=delta,
                        stopped=stopping,
                    )
                )
            mass = moved

            if settled:
                reason = CONVERGED
                break
            if delta < self.stop_threshold:
                reason = THRESHOLD
                break

        self.stats.record(iterations=rounds, stop_reason=reason, clipped=clipped)
        trace = (
            None
            if qid is None
            else ExpansionTrace(
                qid=qid,
                seed_arm=self.seed_arm,
                config_digest=self.config_digest,
                iterations=tuple(rows),
                stop_reason=reason,
            )
        )
        return hits, mass, trace

    def _rank(self, mass: np.ndarray, seed_hits: Sequence[Hit], top_k: int) -> list[Hit]:
        """Decision D4: rank the positive-mass units and the seed's own hits, by id on ties.

        The union with the seed's hits is what makes the parity property of HU-2
        exact rather than exact-up-to-a-tie: a seed hit whose score was clipped to
        zero must not be displaced from the list by whichever zero-mass unit of the
        pool happens to sort first alphabetically.
        """
        n_units = len(self.unit_ids)
        k = min(top_k, n_units)
        best = np.argpartition(-mass, k - 1)[:k] if k < n_units else np.arange(n_units)

        candidates = {int(row) for row in best if mass[row] > 0.0}
        candidates.update(self._index[unit_id] for unit_id, _score in seed_hits)

        ordered = sorted(candidates, key=lambda row: (-float(mass[row]), self.unit_ids[row]))
        return [(self.unit_ids[row], float(mass[row])) for row in ordered[:top_k]]
