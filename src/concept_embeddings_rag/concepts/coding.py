"""Coding the corpus into X, and the artifact that carries what a weight means.

`X` is the transform of the corpus against the dictionary. It is never a
similarity matrix computed afterwards and never an assignment produced by some
other rule: the whole point is that a unit activates several concepts at once
with continuous, non-negative weights, because that co-activation is what
Phase 3 scores over and Phase 4 diffuses across.

Sparsity is an outcome of the L1 penalty rather than a setting, so the coding
`alpha` is calibrated against a declared target band before the corpus is coded.
The search reads unit embeddings only - no question, no gold, no split label.

Storage mirrors the embedding cache: `.npz` plus a JSON sidecar, read with
`allow_pickle=False`. The CSR components are written by hand rather than through
`scipy.sparse.save_npz`, whose loader calls `np.load(file, **PICKLE_KWARGS)` -
an internal constant this project neither controls nor can assert on.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, savez_compressed_atomic, write_text_atomic
from concept_embeddings_rag.concepts.dictionary import ConceptArtifactError, ConceptDictionary

WEIGHT_SEMANTICS: str = (
    "A weight X[i, j] is the non-negative coefficient of concept atom j in the sparse "
    "reconstruction of unit i's embedding, on the scale of L2-normalized embeddings. "
    "It is not a probability: the weights of a unit do not sum to 1 and are not bounded "
    "by 1. Comparing raw weights across units is invalid, because a unit whose embedding "
    "is explained by fewer atoms carries larger coefficients; use the row_normalized view "
    "for that. Within one unit, a larger weight means the concept explains more of that "
    "unit's embedding. A zero means the concept took no part in the reconstruction at all."
)


# The sidecar's `mean_active_per_unit` is recomputed from the matrix on load. Both
# sides count the same stored non-zeros, so the difference is float round-trip noise
# through JSON rather than a real margin; anything above this is a disagreement.
_MEAN_ACTIVE_TOLERANCE: float = 1e-6


class CalibrationError(Exception):
    """The requested sparsity band cannot be reached with this dictionary."""


@dataclass(frozen=True)
class ConceptMatrix:
    """`X = chunks x concepts`, plus everything needed to know what it means."""

    X: sparse.csr_matrix
    unit_ids: list[str]
    dictionary_key: str
    view: str
    coding_alpha: float
    mean_active_per_unit: float
    reconstruction_error: float

    @property
    def weight_semantics(self) -> str:
        return WEIGHT_SEMANTICS


def _active_per_unit(X: sparse.csr_matrix) -> np.ndarray:
    return np.diff(X.tocsr().indptr)


def calibrate_coding_alpha(
    dictionary: ConceptDictionary,
    vectors: np.ndarray,
    *,
    target_band: tuple[int, int] = config.SPARSITY_TARGET_BAND,
    seed: int = config.CONCEPT_SEED,
    sample_size: int = 2000,
    max_steps: int = 24,
) -> tuple[float, float]:
    """Search `alpha` until the mean active concepts per unit lands in `target_band`.

    Returns `(alpha, achieved_mean)`. Bisection on a log scale, because the mean
    falls monotonically as the penalty rises and the useful range spans orders of
    magnitude. Raises rather than returning a near miss: a silently missed target
    would put a number in the artifact that no criterion actually held.
    """
    low, high = target_band
    if low >= high:
        raise CalibrationError(f"target band {target_band} is empty")

    vectors = np.asarray(vectors)
    rng = np.random.default_rng(seed)
    size = min(sample_size, vectors.shape[0])
    sample = vectors[np.sort(rng.choice(vectors.shape[0], size=size, replace=False))]

    def mean_active(alpha: float) -> float:
        code = dictionary.encode(sample, alpha=alpha)
        return float((code > 0).sum(axis=1).mean())

    lo_alpha, hi_alpha = 1e-5, 2.0
    densest = mean_active(lo_alpha)
    if densest < low:
        raise CalibrationError(
            f"even at alpha={lo_alpha} the code holds only {densest:.2f} active concepts "
            f"per unit, below the target of {low}: the dictionary (k={dictionary.k}) cannot "
            "produce a code this dense"
        )
    if densest <= high:
        return lo_alpha, densest

    sparsest = mean_active(hi_alpha)
    if sparsest > high:
        raise CalibrationError(
            f"even at alpha={hi_alpha} the code holds {sparsest:.2f} active concepts per "
            f"unit, above the target of {high}"
        )
    if sparsest >= low:
        return hi_alpha, sparsest

    for _ in range(max_steps):
        mid = float(np.sqrt(lo_alpha * hi_alpha))
        achieved = mean_active(mid)
        if low <= achieved <= high:
            return mid, achieved
        if achieved > high:
            lo_alpha = mid
        else:
            hi_alpha = mid

    raise CalibrationError(
        f"no alpha in [{lo_alpha:.2e}, {hi_alpha:.2e}] lands the mean active concepts per "
        f"unit inside {target_band} after {max_steps} steps; widen the band or revisit k"
    )


def code_corpus(
    dictionary: ConceptDictionary,
    vectors: np.ndarray,
    unit_ids: Sequence[str],
    *,
    alpha: float,
    batch_size: int = config.CODING_BATCH_SIZE,
) -> ConceptMatrix:
    """Code every unit against the dictionary, batched, into a sparse `X`.

    `transform` returns a dense block, so the corpus is coded in batches and each
    batch is converted to CSR before the next one is computed: peak memory is set
    by the batch, not by the corpus.
    """
    vectors = np.asarray(vectors)
    if vectors.shape[0] != len(unit_ids):
        raise ValueError(f"{vectors.shape[0]} vectors for {len(unit_ids)} unit ids")

    blocks: list[sparse.csr_matrix] = []
    residual_sq = 0.0
    total_sq = 0.0
    for start in range(0, vectors.shape[0], batch_size):
        block = vectors[start : start + batch_size]
        code = dictionary.encode(block, alpha=alpha)
        reconstruction = code.astype(np.float64) @ dictionary.atoms.astype(np.float64)
        residual_sq += float(np.sum((block.astype(np.float64) - reconstruction) ** 2))
        total_sq += float(np.sum(block.astype(np.float64) ** 2))
        blocks.append(sparse.csr_matrix(code.astype(np.float32)))

    X = sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix((0, dictionary.k))
    X.eliminate_zeros()
    return ConceptMatrix(
        X=X,
        unit_ids=list(unit_ids),
        dictionary_key=dictionary.key,
        view="raw",
        coding_alpha=alpha,
        mean_active_per_unit=float(_active_per_unit(X).mean()) if X.shape[0] else 0.0,
        reconstruction_error=residual_sq / total_sq if total_sq else 0.0,
    )


def row_normalized(matrix: ConceptMatrix) -> ConceptMatrix:
    """The view that makes units comparable to each other, derived not stored.

    Raw weights sit on the reconstruction scale, so a unit explained by fewer
    atoms carries larger coefficients; dividing each row by its own mass removes
    that. An orphan row - all zeros - stays all zeros instead of dividing by zero
    and poisoning the matrix with NaNs.
    """
    X = matrix.X.tocsr(copy=True).astype(np.float32)
    sums = np.asarray(X.sum(axis=1)).ravel()
    safe = np.where(sums == 0.0, 1.0, sums)
    scaling = sparse.diags(1.0 / safe, dtype=np.float32)
    normalized = sparse.csr_matrix(scaling @ X)
    return ConceptMatrix(
        X=normalized,
        unit_ids=list(matrix.unit_ids),
        dictionary_key=matrix.dictionary_key,
        view="row_normalized",
        coding_alpha=matrix.coding_alpha,
        mean_active_per_unit=matrix.mean_active_per_unit,
        reconstruction_error=matrix.reconstruction_error,
    )


def _path_for(directory: Path, dictionary_key: str, view: str) -> Path:
    return Path(directory) / f"matrix-{dictionary_key}-{view}.npz"


def _sidecar_for(directory: Path, dictionary_key: str, view: str) -> Path:
    return Path(directory) / f"matrix-{dictionary_key}-{view}.json"


def _components_of(X: sparse.csr_matrix, unit_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """The exact arrays that go into the `.npz`, and that a reader gets back.

    The casts are here rather than at the call site so that the digest is computed
    over the same bytes on both sides: scipy is free to narrow `indptr` to int32
    when it rebuilds the matrix, and a digest taken before that cast would differ
    from one taken after it.
    """
    return {
        "data": X.data.astype(np.float32),
        "indices": X.indices.astype(np.int32),
        "indptr": X.indptr.astype(np.int64),
        "shape": np.array(X.shape, dtype=np.int64),
        "unit_ids": np.array(list(unit_ids), dtype=np.str_),
    }


def _matrix_digest(components: dict[str, np.ndarray]) -> str:
    """Bind `X` to its sidecar, the way the dictionary artifact already binds its atoms."""
    return digest_of(
        components["data"],
        components["indices"],
        components["indptr"],
        components["shape"],
        "\n".join(str(unit_id) for unit_id in components["unit_ids"]),
    )


def save_matrix(matrix: ConceptMatrix, directory: Path | str) -> Path:
    """Persist `X` as CSR components plus a sidecar, refusing a one-hot fit.

    The rejection is here rather than in a report because a fit below the floor is
    not a finding to be noted, it is a fit that must not reach disk: everything
    downstream would then be measuring the regime that disqualified clustering.
    """
    X = matrix.X.tocsr()
    if X.shape[0] != len(matrix.unit_ids):
        raise ConceptArtifactError(f"{X.shape[0]} rows for {len(matrix.unit_ids)} unit ids")
    if X.nnz and X.data.min() < 0:
        raise ConceptArtifactError(
            f"X holds a negative weight ({X.data.min():.4f}); the code must be non-negative"
        )
    if matrix.view == "raw" and matrix.mean_active_per_unit < config.SPARSITY_FLOOR:
        raise ConceptArtifactError(
            f"fit rejected: {matrix.mean_active_per_unit:.2f} mean active concepts per unit, "
            f"below the floor of {config.SPARSITY_FLOOR}. That is the one-hot regime - it "
            "destroys co-activation, unit-to-unit similarity in concept space, and expansion"
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = _path_for(directory, matrix.dictionary_key, matrix.view)
    components = _components_of(X, matrix.unit_ids)
    savez_compressed_atomic(path, **components)
    write_text_atomic(
        _sidecar_for(directory, matrix.dictionary_key, matrix.view),
        json.dumps(
            {
                "dictionary_key": matrix.dictionary_key,
                "view": matrix.view,
                "coding_alpha": matrix.coding_alpha,
                "n_units": len(matrix.unit_ids),
                "n_concepts": int(X.shape[1]),
                "nnz": int(X.nnz),
                "mean_active_per_unit": matrix.mean_active_per_unit,
                "reconstruction_error": matrix.reconstruction_error,
                "weight_semantics": WEIGHT_SEMANTICS,
                "sparsity_floor": config.SPARSITY_FLOOR,
                "digest": _matrix_digest(components),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    return path


def load_matrix(
    dictionary_key: str,
    directory: Path | str,
    view: str = "raw",
    expected_unit_ids: Sequence[str] | None = None,
) -> ConceptMatrix:
    """Load `X`, proving it describes the pool the caller is working on."""
    directory = Path(directory)
    path = _path_for(directory, dictionary_key, view)
    sidecar_path = _sidecar_for(directory, dictionary_key, view)
    if not path.exists() or not sidecar_path.exists():
        raise ConceptArtifactError(
            f"no {view} matrix artifact for dictionary {dictionary_key} in {directory}"
        )

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as payload:
        X = sparse.csr_matrix(
            (payload["data"], payload["indices"], payload["indptr"]),
            shape=tuple(int(v) for v in payload["shape"]),
            dtype=np.float32,
        )
        unit_ids = [str(uid) for uid in payload["unit_ids"]]

    # The constructor only runs `check_format(full_check=False)`: it agrees the
    # component lengths and dtypes are consistent, and never looks at the values.
    # A column index past `n_concepts`, or an `indptr` that decreases, therefore
    # survives construction and is dereferenced later inside scipy's C routines,
    # outside the buffer that was allocated for it. This is the first thing done
    # with the matrix, before anything indexes into it.
    try:
        X.check_format(full_check=True)
    except (ValueError, IndexError) as error:
        raise ConceptArtifactError(
            f"matrix {path.name} is not a well-formed CSR matrix: {error}"
        ) from error

    if X.shape[0] != len(unit_ids):
        raise ConceptArtifactError(f"matrix {path.name}: {X.shape[0]} rows for {len(unit_ids)} ids")
    if expected_unit_ids is not None and unit_ids != list(expected_unit_ids):
        raise ConceptArtifactError(
            f"matrix {path.name} does not describe the expected pool; refusing to use it"
        )

    _verify_sidecar(sidecar, X, unit_ids, path, dictionary_key=dictionary_key, view=view)

    return ConceptMatrix(
        X=X,
        unit_ids=unit_ids,
        dictionary_key=str(sidecar["dictionary_key"]),
        view=str(sidecar["view"]),
        coding_alpha=float(sidecar["coding_alpha"]),
        mean_active_per_unit=float(sidecar["mean_active_per_unit"]),
        reconstruction_error=float(sidecar["reconstruction_error"]),
    )


def _verify_sidecar(
    sidecar: dict[str, object],
    X: sparse.csr_matrix,
    unit_ids: list[str],
    path: Path,
    *,
    dictionary_key: str,
    view: str,
) -> None:
    """Hold the sidecar against the matrix it claims to describe.

    Everything a caller reads off a loaded matrix - which dictionary coded it, at
    what `alpha`, how sparse the fit came out - comes from the sidecar, so an
    unchecked sidecar decides what the numbers of the next phase mean. The
    one-hot rejection of `save_matrix` is re-applied here for the same reason: on
    the write path alone it is a rule the read path can simply walk around.
    """
    if str(sidecar["dictionary_key"]) != dictionary_key or str(sidecar["view"]) != view:
        raise ConceptArtifactError(
            f"matrix {path.name} carries a sidecar for "
            f"{sidecar['dictionary_key']}/{sidecar['view']}; refusing to use it"
        )
    for field, actual in (
        ("n_units", len(unit_ids)),
        ("n_concepts", int(X.shape[1])),
        ("nnz", int(X.nnz)),
    ):
        if int(sidecar[field]) != actual:  # type: ignore[call-overload]
            raise ConceptArtifactError(
                f"matrix {path.name} holds {actual} {field} but its sidecar declares "
                f"{sidecar[field]}; the artifact and its sidecar disagree"
            )

    recorded = sidecar.get("digest")
    if recorded is not None:
        actual_digest = _matrix_digest(_components_of(X, unit_ids))
        if actual_digest != recorded:
            raise ConceptArtifactError(
                f"matrix {path.name} hashes to {actual_digest[:16]} but its sidecar records "
                f"{str(recorded)[:16]}; the artifact has been modified"
            )

    measured = float(_active_per_unit(X).mean()) if X.shape[0] else 0.0
    declared = float(sidecar["mean_active_per_unit"])  # type: ignore[arg-type]
    if abs(measured - declared) > _MEAN_ACTIVE_TOLERANCE:
        raise ConceptArtifactError(
            f"matrix {path.name} holds {measured:.4f} mean active concepts per unit but its "
            f"sidecar declares {declared:.4f}; the artifact and its sidecar disagree"
        )
    if view == "raw" and measured < config.SPARSITY_FLOOR:
        raise ConceptArtifactError(
            f"matrix {path.name} holds {measured:.2f} mean active concepts per unit, below "
            f"the floor of {config.SPARSITY_FLOOR}. That is the one-hot regime; it was "
            "rejected when written and is rejected again when read"
        )
