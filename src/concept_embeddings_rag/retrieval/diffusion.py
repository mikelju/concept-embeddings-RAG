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

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.retrieval.base import Hit

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
