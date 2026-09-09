"""What the concept space actually looks like, so it can be criticized here.

A bad space should be caught in this phase, not blamed on retrieval in the next
one. These six statistics are the ones able to show it:

- **active per unit** - a space collapsed toward one activation per unit has no
  co-activation, and is the regime that disqualified clustering;
- **units per concept** and **dead atoms** - dimensions the corpus never uses;
- **orphan units** - paragraphs no concept reaches, and therefore unreachable
  through the concept space at all. They are listed by id, because a count says
  a space has holes while a list says where they are;
- **top units per concept** - a concept read as its evidence rather than as a
  number, which is also what makes the labelling of HU-6 checkable;
- **co-activation** - the association graph of section 14 and the substrate
  Phase 4 diffuses over. If it is degenerate here, Phase 4 has nothing to travel
  across, and that is worth knowing before building the traversal.

None of those six finds the failure HU-7 is about. A concept activated by 400
units that have nothing to do with each other is not dead, is not orphan, and its
ten evidence units look plausible read one at a time. The quality block scores it:
how alike the units that activate a concept actually are, how close the concept
sits to its own evidence, and how its weight is spread across it. Every score is
reported against a **null baseline measured on this same pool**, because the raw
cosine says nothing on its own - the embedding space is anisotropic, and two
unrelated paragraphs of this corpus score far above zero.

The quality block describes the space and never touches it. Nothing here prunes,
merges or reweights a concept on the strength of a score: what to do about a bad
concept is a Phase 3 decision, made against dev recall.

Everything is computed over the whole pool. There is deliberately no split
parameter: the test split is not looked at in this phase, not even descriptively.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, write_text_atomic
from concept_embeddings_rag.concepts.coding import ConceptMatrix
from concept_embeddings_rag.concepts.dictionary import ConceptArtifactError
from concept_embeddings_rag.embeddings.cache import unit_set_hash

STAT_KEYS: tuple[str, ...] = ("mean", "median", "p5", "p95", "max")

# A null baseline drawn from a pool too small to vary has a spread that is zero up
# to floating point, not zero. Dividing by that would report an astronomical
# z-score for an ordinary concept, so anything under this counts as no spread.
NEGLIGIBLE_SD: float = 1e-9


@dataclass(frozen=True)
class DictionaryQuality:
    """How good the concepts are, per concept, against what randomness scores here.

    Three numbers per concept, because they fail differently: `coherence` is low
    when the units activating a concept have nothing in common; `atom_alignment`
    is low when the atom sits away from its own evidence, which is what an atom
    stretched between two unrelated groups looks like even when each group is
    internally tight; and `top_mass_share` separates a sharp concept from an
    over-general one whose weight is spread thinly over thousands of units.

    `coherence_z` is the one to read. A raw cosine of 0.31 is only meaningful
    against `null_mean`, which is what random units of this pool score.
    """

    top_units: int
    min_units: int
    null_samples: int
    seed: int
    null_mean: float
    null_sd: float
    coherence: dict[str, float]
    coherence_z: dict[str, float]
    atom_alignment: dict[str, float]
    top_mass_share: dict[str, float]
    coherence_stats: dict[str, float]
    undefined_concepts: list[int]
    below_null_concepts: list[int]


@dataclass(frozen=True)
class SpaceDiagnostics:
    """One inspection of one concept space, written as a versioned artifact."""

    dictionary_key: str
    view: str
    n_units: int
    n_concepts: int
    active_per_unit: dict[str, float]
    units_per_concept: dict[str, float]
    dead_atoms: list[int]
    dead_atom_min_units: int
    orphan_units: list[str]
    top_units_per_concept: dict[str, list[str]]
    top_coactivations: dict[str, list[list[int]]]
    quality: DictionaryQuality | None = None
    # Which pool the inspected space describes. Empty on an artifact written
    # before this field existed, which is what makes the check on load optional.
    unit_set_hash: str = ""


def _distribution(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return dict.fromkeys(STAT_KEYS, 0.0)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _indicator(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """The support of `X`: one where a concept is active, nothing where it is not."""
    support = X.tocsr(copy=True)
    support.eliminate_zeros()
    return sparse.csr_matrix(
        (np.ones(support.nnz, dtype=np.int32), support.indices, support.indptr),
        shape=support.shape,
    )


def coactivation_counts(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """How many units activate each pair of concepts together.

    Symmetric by construction, and its diagonal is cleared: a concept co-activates
    with itself in every unit that holds it, which says nothing about association
    and would dominate every ranking.
    """
    support = _indicator(X)
    counts = sparse.csr_matrix(support.T @ support)
    counts.setdiag(0)
    counts.eliminate_zeros()
    return counts


def _top_units(X: sparse.csr_matrix, unit_ids: list[str], how_many: int) -> dict[str, list[str]]:
    columns = X.tocsc()
    top: dict[str, list[str]] = {}
    for concept in range(columns.shape[1]):
        start, end = columns.indptr[concept], columns.indptr[concept + 1]
        rows = columns.indices[start:end]
        weights = columns.data[start:end]
        # Stable, so equal weights fall back to ascending unit position and the
        # artifact does not change between runs over the same matrix.
        order = np.argsort(-weights, kind="stable")[:how_many]
        top[str(concept)] = [unit_ids[int(rows[position])] for position in order]
    return top


def _top_coactivations(counts: sparse.csr_matrix, how_many: int) -> dict[str, list[list[int]]]:
    top: dict[str, list[list[int]]] = {}
    for concept in range(counts.shape[0]):
        start, end = counts.indptr[concept], counts.indptr[concept + 1]
        others = counts.indices[start:end]
        together = counts.data[start:end]
        order = np.argsort(-together, kind="stable")[:how_many]
        top[str(concept)] = [[int(others[position]), int(together[position])] for position in order]
    return top


def _l2_normalized(vectors: np.ndarray) -> np.ndarray:
    """Rows of unit norm, so a cosine is a dot product.

    The pipeline already normalizes its embeddings and its atoms; doing it again
    here costs nothing and keeps this function correct for anything handed to it.
    A zero row keeps its zeros instead of dividing by zero.
    """
    rows = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms == 0.0, 1.0, norms)


def _mean_pairwise_cosine(rows: np.ndarray) -> float:
    """Mean cosine over every distinct pair of `rows`, which must be unit-norm.

    The diagonal of the Gram matrix is every vector with itself, which is 1 by
    construction and says nothing about the group, so it is removed.
    """
    count = rows.shape[0]
    if count < 2:
        return 0.0
    gram = rows @ rows.T
    return float((gram.sum() - np.trace(gram)) / (count * (count - 1)))


def null_coherence(
    units: np.ndarray,
    *,
    sample_size: int,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """What a concept made of random units of this pool would score.

    This is the reference every coherence is read against, and the reason the
    diagnostic is worth computing at all: `bge-small-en-v1.5` is anisotropic, so
    two unrelated paragraphs of this corpus are nowhere near orthogonal. Without
    this number, a coherence of 0.31 could be terrible or excellent and there
    would be no way to tell.

    Seeded explicitly, so the baseline is the same on every run over the same pool.
    """
    total_units = units.shape[0]
    size = min(sample_size, total_units)
    if size < 2 or samples < 1:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            _mean_pairwise_cosine(units[rng.choice(total_units, size=size, replace=False)])
            for _ in range(samples)
        ]
    )
    return float(draws.mean()), float(draws.std())


def compute_quality(
    matrix: ConceptMatrix,
    vectors: np.ndarray,
    atoms: np.ndarray,
    *,
    top_units: int = config.COHERENCE_TOP_UNITS,
    min_units: int = config.COHERENCE_MIN_UNITS,
    null_samples: int = config.COHERENCE_NULL_SAMPLES,
    seed: int = config.CONCEPT_SEED,
) -> DictionaryQuality:
    """Score every concept of one space, and never change the space.

    `vectors` are the cached Phase 1 embeddings of the units of `matrix`, in the
    same order; `atoms` is the dictionary `matrix` was coded against. Both are
    checked against the matrix rather than trusted, as every artifact of this
    project is.

    A concept activated by fewer than `min_units` units gets no score: a pairwise
    mean over two units is a single pair, and ranking concepts on that would rank
    them by noise. Those concepts are listed as undefined, not scored as zero.
    """
    X = matrix.X.tocsr(copy=True)
    X.eliminate_zeros()
    n_units, n_concepts = X.shape

    if vectors.shape[0] != n_units:
        raise ConceptArtifactError(
            f"quality: {vectors.shape[0]} embeddings for {n_units} units of {matrix.dictionary_key}"
        )
    if atoms.shape[0] != n_concepts:
        raise ConceptArtifactError(
            f"quality: {atoms.shape[0]} atoms for {n_concepts} concepts of {matrix.dictionary_key}"
        )
    if atoms.shape[1] != vectors.shape[1]:
        raise ConceptArtifactError(
            f"quality: atoms have {atoms.shape[1]} dimensions and embeddings "
            f"{vectors.shape[1]}; they do not live in the same space"
        )

    units = _l2_normalized(vectors)
    concepts = _l2_normalized(atoms)
    null_mean, null_sd = null_coherence(
        units, sample_size=top_units, samples=null_samples, seed=seed
    )
    # A degenerate baseline would divide every z-score by ~zero. It can only happen
    # on a pool too small to sample, where the z-score means nothing anyway, so the
    # score falls back to a plain difference against the null.
    spread = null_sd if null_sd > NEGLIGIBLE_SD else 1.0

    columns = X.tocsc()
    coherence: dict[str, float] = {}
    coherence_z: dict[str, float] = {}
    atom_alignment: dict[str, float] = {}
    top_mass_share: dict[str, float] = {}
    undefined: list[int] = []

    for concept in range(n_concepts):
        start, end = columns.indptr[concept], columns.indptr[concept + 1]
        rows = columns.indices[start:end]
        weights = columns.data[start:end]
        if rows.size < min_units:
            undefined.append(concept)
            continue
        order = np.argsort(-weights, kind="stable")[:top_units]
        evidence = units[rows[order]]
        total = float(weights.sum())

        score = _mean_pairwise_cosine(evidence)
        coherence[str(concept)] = score
        coherence_z[str(concept)] = (score - null_mean) / spread
        atom_alignment[str(concept)] = float((evidence @ concepts[concept]).mean())
        top_mass_share[str(concept)] = float(weights[order].sum() / total) if total > 0 else 0.0

    scored = np.array(list(coherence.values()), dtype=np.float64)
    return DictionaryQuality(
        top_units=top_units,
        min_units=min_units,
        null_samples=null_samples,
        seed=seed,
        null_mean=null_mean,
        null_sd=null_sd,
        coherence=coherence,
        coherence_z=coherence_z,
        atom_alignment=atom_alignment,
        top_mass_share=top_mass_share,
        coherence_stats=_distribution(scored),
        undefined_concepts=undefined,
        below_null_concepts=[
            int(concept) for concept, score in coherence.items() if score < null_mean
        ],
    )


def compute_diagnostics(
    matrix: ConceptMatrix,
    *,
    vectors: np.ndarray | None = None,
    atoms: np.ndarray | None = None,
    dead_atom_min_units: int = config.DEAD_ATOM_MIN_UNITS,
    top_units: int = config.TOP_UNITS_PER_CONCEPT,
    top_coactivations: int = config.TOP_COACTIVATIONS,
    seed: int = config.CONCEPT_SEED,
) -> SpaceDiagnostics:
    """Inspect one concept space, over the whole pool it was built from.

    `vectors` and `atoms` add the HU-7 quality block: the unit embeddings the
    matrix was coded from, and the dictionary it was coded against. Without them
    the structural half is still computed and `quality` is `None`, which is what
    lets the shape of a space be checked on a synthetic matrix with no embeddings
    behind it. The real sweep always passes both.
    """
    X = matrix.X.tocsr()
    support = _indicator(X)
    per_unit = np.diff(support.indptr)
    per_concept = np.asarray(support.sum(axis=0)).ravel()

    quality = (
        compute_quality(matrix, vectors, atoms, seed=seed)
        if vectors is not None and atoms is not None
        else None
    )

    return SpaceDiagnostics(
        dictionary_key=matrix.dictionary_key,
        view=matrix.view,
        n_units=int(X.shape[0]),
        n_concepts=int(X.shape[1]),
        active_per_unit=_distribution(per_unit),
        units_per_concept=_distribution(per_concept),
        dead_atoms=[int(index) for index in np.flatnonzero(per_concept < dead_atom_min_units)],
        dead_atom_min_units=dead_atom_min_units,
        orphan_units=[matrix.unit_ids[int(position)] for position in np.flatnonzero(per_unit == 0)],
        top_units_per_concept=_top_units(X, list(matrix.unit_ids), top_units),
        top_coactivations=_top_coactivations(coactivation_counts(X), top_coactivations),
        quality=quality,
        unit_set_hash=unit_set_hash(list(matrix.unit_ids)),
    )


def _quality_payload(quality: DictionaryQuality | None) -> dict[str, object] | None:
    if quality is None:
        return None
    return {
        "top_units": quality.top_units,
        "min_units": quality.min_units,
        "null_samples": quality.null_samples,
        "seed": quality.seed,
        "null_mean": quality.null_mean,
        "null_sd": quality.null_sd,
        "coherence": quality.coherence,
        "coherence_z": quality.coherence_z,
        "atom_alignment": quality.atom_alignment,
        "top_mass_share": quality.top_mass_share,
        "coherence_stats": quality.coherence_stats,
        "undefined_concepts": quality.undefined_concepts,
        "below_null_concepts": quality.below_null_concepts,
    }


def _quality_from_payload(payload: Any) -> DictionaryQuality | None:
    """Read the quality block back. Absent on an artifact written before it existed."""
    if payload is None:
        return None

    def scores(key: str) -> dict[str, float]:
        return {str(concept): float(value) for concept, value in payload[key].items()}

    return DictionaryQuality(
        top_units=int(payload["top_units"]),
        min_units=int(payload["min_units"]),
        null_samples=int(payload["null_samples"]),
        seed=int(payload["seed"]),
        null_mean=float(payload["null_mean"]),
        null_sd=float(payload["null_sd"]),
        coherence=scores("coherence"),
        coherence_z=scores("coherence_z"),
        atom_alignment=scores("atom_alignment"),
        top_mass_share=scores("top_mass_share"),
        coherence_stats=scores("coherence_stats"),
        undefined_concepts=[int(concept) for concept in payload["undefined_concepts"]],
        below_null_concepts=[int(concept) for concept in payload["below_null_concepts"]],
    )


def _path_for(directory: Path, dictionary_key: str) -> Path:
    return Path(directory) / f"diagnostics-{dictionary_key}.json"


def _payload_of(diagnostics: SpaceDiagnostics) -> dict[str, Any]:
    return {
        "dictionary_key": diagnostics.dictionary_key,
        "view": diagnostics.view,
        "n_units": diagnostics.n_units,
        "n_concepts": diagnostics.n_concepts,
        "unit_set_hash": diagnostics.unit_set_hash,
        "active_per_unit": diagnostics.active_per_unit,
        "units_per_concept": diagnostics.units_per_concept,
        "dead_atoms": diagnostics.dead_atoms,
        "dead_atom_min_units": diagnostics.dead_atom_min_units,
        "orphan_units": diagnostics.orphan_units,
        "top_units_per_concept": diagnostics.top_units_per_concept,
        "top_coactivations": diagnostics.top_coactivations,
        "quality": _quality_payload(diagnostics.quality),
    }


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _digest_of_payload(payload: dict[str, Any]) -> str:
    """The digest is taken over the serialization, minus the digest field itself."""
    return digest_of(_serialize({key: value for key, value in payload.items() if key != "digest"}))


def save_diagnostics(diagnostics: SpaceDiagnostics, directory: Path | str) -> Path:
    """Write the diagnostics as an artifact, so the four spaces stay comparable."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = _path_for(directory, diagnostics.dictionary_key)
    payload = _payload_of(diagnostics)
    payload["digest"] = _digest_of_payload(payload)
    write_text_atomic(path, _serialize(payload))
    return path


def load_diagnostics(
    dictionary_key: str, directory: Path | str, expected_unit_set_hash: str | None = None
) -> SpaceDiagnostics:
    """Read back the diagnostics of one space, proving it is the one asked for.

    This artifact is not only descriptive. `top_units_per_concept` is what the
    labelling stage turns into the prompts it pays for, so whatever this file says
    decides what evidence reaches the API and what the report claims a concept
    means. It is verified the way the dictionary and the matrix beside it are:
    the file is bound to its own contents by a digest, to the key it is filed
    under, and to the pool it was computed over.

    `unit_set_hash` and `digest` are checked when present rather than required, so
    the artifacts written before those fields existed still load - and every
    artifact written from here on carries both.
    """
    path = _path_for(Path(directory), dictionary_key)
    if not path.exists():
        raise ConceptArtifactError(f"no diagnostics for dictionary {dictionary_key} in {directory}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload["dictionary_key"]) != dictionary_key:
        raise ConceptArtifactError(
            f"diagnostics {path.name} declare dictionary {payload['dictionary_key']}, "
            f"not {dictionary_key}; refusing to use them"
        )

    recorded = payload.get("digest")
    if recorded is not None:
        actual = _digest_of_payload(payload)
        if actual != recorded:
            raise ConceptArtifactError(
                f"diagnostics {path.name} hash to {actual[:16]} but the artifact records "
                f"{str(recorded)[:16]}; it has been modified"
            )

    recorded_pool = str(payload.get("unit_set_hash", ""))
    if (
        expected_unit_set_hash is not None
        and recorded_pool
        and recorded_pool != expected_unit_set_hash
    ):
        raise ConceptArtifactError(
            f"diagnostics {path.name} were computed over pool {recorded_pool}, "
            f"not {expected_unit_set_hash}; refusing to use them"
        )

    return SpaceDiagnostics(
        dictionary_key=str(payload["dictionary_key"]),
        view=str(payload["view"]),
        n_units=int(payload["n_units"]),
        n_concepts=int(payload["n_concepts"]),
        active_per_unit={key: float(value) for key, value in payload["active_per_unit"].items()},
        units_per_concept={
            key: float(value) for key, value in payload["units_per_concept"].items()
        },
        dead_atoms=[int(index) for index in payload["dead_atoms"]],
        dead_atom_min_units=int(payload["dead_atom_min_units"]),
        orphan_units=[str(unit_id) for unit_id in payload["orphan_units"]],
        top_units_per_concept={
            key: [str(unit_id) for unit_id in value]
            for key, value in payload["top_units_per_concept"].items()
        },
        top_coactivations={
            key: [[int(pair[0]), int(pair[1])] for pair in value]
            for key, value in payload["top_coactivations"].items()
        },
        quality=_quality_from_payload(payload.get("quality")),
        unit_set_hash=recorded_pool,
    )
