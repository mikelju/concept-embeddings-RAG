"""The concept dictionary: induction by non-negative sparse coding, and its artifact.

The dimensions of this space are learned from the corpus itself. Each atom is an
L2-normalized vector living in the embedding space, so Phase 3 maps a question to
concepts with a dot product and never through a generated name.

Two constraints, and only one of them is on the dictionary:

- `positive_code=True` is the one the hypothesis needs. Non-negative activations
  are what give co-activation, unit-to-unit similarity in concept space, and
  something for Phase 4's diffusion to travel across.
- `positive_dict` is **False**. Measured on the actual corpus, 51.3% of the energy
  of the `bge-small-en-v1.5` embeddings lives in negative coordinates, so a
  non-negative dictionary combined with a non-negative code cannot reconstruct
  them: a short fit gave 1.59 mean active concepts per unit - below the rejection
  floor of 3 - with 56% dead atoms. See decision D1 of the phase plan.

Storage is `.npz` plus a JSON sidecar, keyed by the configuration that produced
it. The fitted estimator is never serialized: only its learned array of atoms
travels, because `joblib.dump` is pickle under another name and loading a pickle
executes code.
"""

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder
from sklearn.exceptions import ConvergenceWarning

from concept_embeddings_rag import __version__, config
from concept_embeddings_rag.artifacts import savez_compressed_atomic, write_text_atomic
from concept_embeddings_rag.embeddings.backend import l2_normalize


class ConceptArtifactError(Exception):
    """An artifact is missing, malformed, or does not describe the pool it claims to."""


def dictionary_key(
    *,
    k: int,
    seed: int,
    alpha: float,
    unit_set_hash: str,
    model: str,
    revision: str,
    merge_threshold: float | None = None,
) -> str:
    """Identify a dictionary by everything that decides what it contains.

    Changing any of these writes a new artifact instead of overwriting one whose
    numbers are already recorded somewhere.

    `merge_threshold` is what separates a deduplicated dictionary from the one it
    was derived from. Without it, a merge that happened to remove nothing would
    produce the same key as its source and overwrite it - and a merge that removed
    plenty would still collide with a differently-thresholded run of the same K.
    """
    payload = f"{k}|{seed}|{alpha!r}|{unit_set_hash}|{model}|{revision}"
    if merge_threshold is not None:
        payload += f"|merged@{merge_threshold!r}"
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@dataclass(frozen=True)
class ConceptDictionary:
    """K concepts as unit-norm atoms, plus the configuration that produced them."""

    atoms: np.ndarray
    k: int
    seed: int
    sparsity_param: float
    max_iter: int
    unit_set_hash: str
    model: str
    revision: str
    code_version: str = __version__
    merged_from: list[list[int]] | None = field(default=None)
    merge_threshold: float | None = field(default=None)

    @property
    def key(self) -> str:
        return dictionary_key(
            k=self.k,
            seed=self.seed,
            alpha=self.sparsity_param,
            unit_set_hash=self.unit_set_hash,
            model=self.model,
            revision=self.revision,
            merge_threshold=self.merge_threshold,
        )

    @property
    def dim(self) -> int:
        return int(self.atoms.shape[1])

    def digest(self) -> str:
        """SHA-256 of the atoms, so reproducibility can be asserted bit for bit."""
        return hashlib.sha256(np.ascontiguousarray(self.atoms, dtype=np.float32)).hexdigest()

    def encode(self, vectors: np.ndarray, alpha: float | None = None) -> np.ndarray:
        """Code `vectors` against this dictionary, non-negatively.

        Returns the dense code for a block of vectors. The corpus-wide, batched
        and sparse version lives in `coding.py`; this is the primitive both it and
        the calibration search are built on.

        `alpha` defaults to the penalty the dictionary was induced with. It is a
        separate argument because the coding penalty is calibrated against a
        sparsity target after induction, and both values are recorded: the
        dictionary's on the dictionary, the coding one on the matrix.
        """
        coder = SparseCoder(
            dictionary=np.ascontiguousarray(self.atoms, dtype=np.float64),
            transform_algorithm="lasso_cd",
            transform_alpha=self.sparsity_param if alpha is None else alpha,
            positive_code=True,
            transform_max_iter=config.CONCEPT_MAX_ITER * 100,
        )
        with warnings.catch_warnings():
            # The inner lasso reports a duality gap around 1e-5 against a tolerance
            # derived from unit-norm vectors. Accepted deliberately (task T3): the
            # residual is numerically irrelevant at this scale, and silencing it
            # here keeps the pipeline's own output readable. It is scoped to this
            # call, never installed globally.
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            code = coder.transform(np.asarray(vectors, dtype=np.float64))
        return np.asarray(code, dtype=np.float32)


def induce_dictionary(
    vectors: np.ndarray,
    *,
    k: int,
    seed: int,
    alpha: float,
    max_iter: int = config.CONCEPT_MAX_ITER,
    batch_size: int = config.CONCEPT_BATCH_SIZE,
    unit_set_hash: str = "",
    model: str = config.EMBEDDING_MODEL,
    revision: str = config.EMBEDDING_REVISION,
) -> ConceptDictionary:
    """Learn `k` concepts from `vectors` by non-negative sparse coding.

    `fit_algorithm="cd"` is forced rather than chosen: scikit-learn's default LARS
    rejects the positivity constraint outright ("Positive constraint not supported
    for 'lars' coding method"). Sparsity is therefore controlled by the L1 penalty
    `alpha`, which is why it has to be calibrated against a target rather than set.
    """
    vectors = np.asarray(vectors)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {vectors.shape}")
    if k > vectors.shape[0]:
        raise ValueError(
            f"cannot induce {k} atoms from {vectors.shape[0]} samples: "
            "a dictionary larger than the corpus is not identifiable"
        )

    estimator = MiniBatchDictionaryLearning(
        n_components=k,
        alpha=alpha,
        fit_algorithm="cd",
        transform_algorithm="lasso_cd",
        positive_code=True,
        positive_dict=False,  # decision D1 - see the module docstring
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=seed,
        n_jobs=None,
        shuffle=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        estimator.fit(np.asarray(vectors, dtype=np.float64))

    # Only the learned array survives this function. The estimator is dropped here
    # deliberately: persisting it would mean joblib, and joblib is pickle.
    atoms = l2_normalize(np.asarray(estimator.components_, dtype=np.float32))
    return ConceptDictionary(
        atoms=atoms,
        k=k,
        seed=seed,
        sparsity_param=alpha,
        max_iter=max_iter,
        unit_set_hash=unit_set_hash,
        model=model,
        revision=revision,
    )


def _path_for(directory: Path, key: str) -> Path:
    return Path(directory) / f"dictionary-{key}.npz"


def _sidecar_for(directory: Path, key: str) -> Path:
    return Path(directory) / f"dictionary-{key}.json"


def save_dictionary(dictionary: ConceptDictionary, directory: Path | str) -> Path:
    """Persist the atoms as `.npz` plus a JSON sidecar carrying the configuration."""
    norms = np.linalg.norm(dictionary.atoms, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ConceptArtifactError(
            "atoms are not L2-normalized, so cosine between them is not a dot product "
            f"(norms range {norms.min():.4f} to {norms.max():.4f})"
        )
    if dictionary.atoms.shape[0] != dictionary.k:
        raise ConceptArtifactError(
            f"{dictionary.atoms.shape[0]} atoms for a dictionary declaring k={dictionary.k}"
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = _path_for(directory, dictionary.key)
    savez_compressed_atomic(path, atoms=dictionary.atoms.astype(np.float32))

    sidecar = {
        "key": dictionary.key,
        "k": dictionary.k,
        "seed": dictionary.seed,
        "sparsity_param": dictionary.sparsity_param,
        "max_iter": dictionary.max_iter,
        "unit_set_hash": dictionary.unit_set_hash,
        "model": dictionary.model,
        "revision": dictionary.revision,
        "code_version": dictionary.code_version,
        "dim": dictionary.dim,
        "digest": dictionary.digest(),
        "merged_from": dictionary.merged_from,
        "merge_threshold": dictionary.merge_threshold,
        "positive_code": True,
        "positive_dict": False,
    }
    write_text_atomic(
        _sidecar_for(directory, dictionary.key), json.dumps(sidecar, indent=2, sort_keys=True)
    )
    return path


def load_dictionary(
    key: str, directory: Path | str, expected_unit_set_hash: str | None = None
) -> ConceptDictionary:
    """Load a dictionary, verifying it describes the pool the caller is working on.

    Phase 1 established the rule the hard way: verify on load, do not merely
    record. A dictionary induced from a different corpus would produce an `X`
    whose rows mean nothing, and nothing downstream could tell.
    """
    directory = Path(directory)
    path = _path_for(directory, key)
    sidecar_path = _sidecar_for(directory, key)
    if not path.exists() or not sidecar_path.exists():
        raise ConceptArtifactError(f"no dictionary artifact for key {key} in {directory}")

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as payload:
        atoms = np.asarray(payload["atoms"], dtype=np.float32)

    if str(sidecar["key"]) != key:
        raise ConceptArtifactError(
            f"dictionary artifact {path.name} carries a sidecar for {sidecar['key']}; "
            "refusing to use it"
        )
    if atoms.shape[0] != sidecar["k"]:
        raise ConceptArtifactError(
            f"dictionary {key}: {atoms.shape[0]} atoms but the sidecar declares k={sidecar['k']}"
        )
    if expected_unit_set_hash is not None and sidecar["unit_set_hash"] != expected_unit_set_hash:
        raise ConceptArtifactError(
            f"dictionary {key} was induced from pool {sidecar['unit_set_hash']}, "
            f"not {expected_unit_set_hash}; refusing to use it"
        )

    dictionary = ConceptDictionary(
        atoms=atoms,
        k=int(sidecar["k"]),
        seed=int(sidecar["seed"]),
        sparsity_param=float(sidecar["sparsity_param"]),
        max_iter=int(sidecar["max_iter"]),
        unit_set_hash=str(sidecar["unit_set_hash"]),
        model=str(sidecar["model"]),
        revision=str(sidecar["revision"]),
        code_version=str(sidecar["code_version"]),
        merged_from=sidecar.get("merged_from"),
        merge_threshold=sidecar.get("merge_threshold"),
    )
    recorded = sidecar.get("digest")
    if recorded is not None and dictionary.digest() != recorded:
        raise ConceptArtifactError(
            f"dictionary {key} hashes to {dictionary.digest()[:16]} but its sidecar "
            f"records {recorded[:16]}; the artifact has been modified"
        )
    return dictionary
