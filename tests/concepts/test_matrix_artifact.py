"""T7: the ConceptMatrix artifact, and the fit this project refuses to ship.

Three rules, all of them consequences of earlier decisions:

- the meaning of a weight travels inside the artifact, because there is no
  universal meaning of 0.73;
- `unit_ids` are validated against the pool on load, exactly as the embedding
  cache is - verify, do not merely record;
- a fit below three active concepts per unit is the one-hot regime that
  disqualified clustering, and it is not written to disk at all.
"""

import json

import numpy as np
import pytest
from scipy import sparse

from concept_embeddings_rag.concepts.coding import (
    ConceptMatrix,
    load_matrix,
    save_matrix,
)
from concept_embeddings_rag.concepts.dictionary import ConceptArtifactError


def a_matrix(n: int = 6, k: int = 5, active: int = 4, **overrides) -> ConceptMatrix:
    rows, cols, values = [], [], []
    for i in range(n):
        for j in range(active):
            rows.append(i)
            cols.append((i + j) % k)
            values.append(float(j + 1))
    X = sparse.csr_matrix((values, (rows, cols)), shape=(n, k), dtype=np.float32)
    fields = {
        "X": X,
        "unit_ids": [f"unit{i:04d}" for i in range(n)],
        "dictionary_key": "abc123",
        "view": "raw",
        "coding_alpha": 0.01,
        "mean_active_per_unit": float(active),
        "reconstruction_error": 0.42,
    }
    fields.update(overrides)
    return ConceptMatrix(**fields)


def test_round_trip_preserves_the_matrix_exactly(tmp_path):
    original = a_matrix()
    save_matrix(original, tmp_path)

    loaded = load_matrix(original.dictionary_key, tmp_path)

    assert np.array_equal(loaded.X.toarray(), original.X.toarray())
    assert loaded.X.dtype == np.float32
    assert loaded.unit_ids == original.unit_ids
    assert loaded.coding_alpha == original.coding_alpha
    assert loaded.mean_active_per_unit == original.mean_active_per_unit
    assert loaded.reconstruction_error == original.reconstruction_error


def test_the_weight_semantics_are_written_into_the_artifact_not_only_the_docs(tmp_path):
    saved = a_matrix()
    save_matrix(saved, tmp_path)

    sidecar = json.loads(
        (tmp_path / f"matrix-{saved.dictionary_key}-raw.json").read_text(encoding="utf-8")
    )

    semantics = sidecar["weight_semantics"]
    assert "coefficient" in semantics
    assert "probabilit" in semantics  # it says what a weight is not
    assert "normaliz" in semantics  # and that cross-unit comparison needs it


def test_loading_against_a_different_pool_aborts(tmp_path):
    saved = a_matrix()
    save_matrix(saved, tmp_path)

    with pytest.raises(ConceptArtifactError):
        load_matrix(saved.dictionary_key, tmp_path, expected_unit_ids=["nope", "wrong"])


def test_loading_the_pool_it_describes_is_accepted(tmp_path):
    saved = a_matrix()
    save_matrix(saved, tmp_path)

    loaded = load_matrix(saved.dictionary_key, tmp_path, expected_unit_ids=saved.unit_ids)

    assert loaded.X.shape == saved.X.shape


def test_a_one_hot_fit_is_refused_rather_than_written(tmp_path):
    """Below three active concepts per unit there is no co-activation to expand over."""
    with pytest.raises(ConceptArtifactError, match="active"):
        save_matrix(a_matrix(active=2), tmp_path)

    assert list(tmp_path.glob("*.npz")) == []


def test_a_negative_entry_is_refused(tmp_path):
    X = sparse.csr_matrix(np.array([[1.0, -0.5, 0.0, 2.0, 1.0]] * 4, dtype=np.float32))
    with pytest.raises(ConceptArtifactError, match="non-negative"):
        save_matrix(a_matrix(X=X, unit_ids=[f"unit{i:04d}" for i in range(4)]), tmp_path)


def test_a_row_count_that_disagrees_with_the_unit_ids_is_refused(tmp_path):
    with pytest.raises(ConceptArtifactError):
        save_matrix(a_matrix(unit_ids=["only", "two"]), tmp_path)


def test_the_artifact_loads_without_pickle(tmp_path):
    saved = a_matrix()
    path = save_matrix(saved, tmp_path)

    with np.load(path, allow_pickle=False) as payload:
        assert payload["data"].shape[0] == saved.X.nnz


def test_the_two_views_are_separate_artifacts(tmp_path):
    save_matrix(a_matrix(view="raw"), tmp_path)
    save_matrix(a_matrix(view="row_normalized"), tmp_path)

    assert len(list(tmp_path.glob("matrix-*.npz"))) == 2


def test_a_missing_artifact_names_the_key(tmp_path):
    with pytest.raises(ConceptArtifactError, match="deadbeef"):
        load_matrix("deadbeef", tmp_path)
