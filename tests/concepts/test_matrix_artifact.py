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


def rewrite_sidecar(tmp_path, matrix: ConceptMatrix, **changes) -> None:
    """Edit the sidecar in place, the way anything that is not `save_matrix` would."""
    path = tmp_path / f"matrix-{matrix.dictionary_key}-{matrix.view}.json"
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    sidecar.update(changes)
    path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")


def test_a_matrix_that_is_not_well_formed_csr_is_refused_before_it_is_indexed(tmp_path):
    """SEC-013: the constructor runs `check_format(full_check=False)` and never looks

    at the values, so a column index past `n_concepts` survives it and is later
    dereferenced inside scipy's C routines, outside the allocated buffer.
    """
    saved = a_matrix()
    path = save_matrix(saved, tmp_path)
    with np.load(path, allow_pickle=False) as payload:
        components = {name: payload[name].copy() for name in payload.files}
    components["indices"][0] = 999  # past the last column, and past the buffer
    np.savez_compressed(path, **components)
    rewrite_sidecar(tmp_path, saved, digest=None)

    with pytest.raises(ConceptArtifactError, match="well-formed"):
        load_matrix(saved.dictionary_key, tmp_path)


def test_a_sidecar_filed_under_another_key_is_refused(tmp_path):
    """SEC-013: every number a caller reads off a loaded matrix comes from the sidecar."""
    saved = a_matrix()
    save_matrix(saved, tmp_path)
    rewrite_sidecar(tmp_path, saved, dictionary_key="somebody-elses-space")

    with pytest.raises(ConceptArtifactError, match="refusing to use it"):
        load_matrix(saved.dictionary_key, tmp_path)


def test_a_sidecar_that_disagrees_with_the_matrix_is_refused(tmp_path):
    """SEC-013: `n_units`, `n_concepts` and `nnz` describe the file beside them."""
    saved = a_matrix()
    save_matrix(saved, tmp_path)
    rewrite_sidecar(tmp_path, saved, nnz=1)

    with pytest.raises(ConceptArtifactError, match="disagree"):
        load_matrix(saved.dictionary_key, tmp_path)


def test_a_modified_matrix_is_caught_by_its_digest(tmp_path):
    """SEC-013: the dictionary artifact has been bound to its sidecar by a hash

    since Phase 1; `X` carried the same fields with nothing binding them.
    """
    saved = a_matrix()
    path = save_matrix(saved, tmp_path)
    with np.load(path, allow_pickle=False) as payload:
        components = {name: payload[name].copy() for name in payload.files}
    components["data"][0] = 99.0
    np.savez_compressed(path, **components)

    with pytest.raises(ConceptArtifactError, match="has been modified"):
        load_matrix(saved.dictionary_key, tmp_path)


def test_a_declared_sparsity_that_the_matrix_does_not_support_is_refused(tmp_path):
    """SEC-013: `mean_active_per_unit` is recomputed, not believed."""
    saved = a_matrix()
    save_matrix(saved, tmp_path)
    rewrite_sidecar(tmp_path, saved, mean_active_per_unit=12.0)

    with pytest.raises(ConceptArtifactError, match="mean active"):
        load_matrix(saved.dictionary_key, tmp_path)


def test_a_one_hot_fit_is_refused_on_load_as_well_as_on_save(tmp_path):
    """SEC-013: on the write path alone, the floor is a rule the read path walks around."""
    one_hot = a_matrix(active=1, mean_active_per_unit=1.0, view="row_normalized")
    save_matrix(one_hot, tmp_path)  # accepted: the floor applies to the raw view
    rewrite_sidecar(tmp_path, one_hot, view="raw")
    for suffix in (".npz", ".json"):
        source = tmp_path / f"matrix-{one_hot.dictionary_key}-row_normalized{suffix}"
        source.rename(tmp_path / f"matrix-{one_hot.dictionary_key}-raw{suffix}")

    with pytest.raises(ConceptArtifactError, match="one-hot"):
        load_matrix(one_hot.dictionary_key, tmp_path, view="raw")


def test_the_artifacts_this_phase_already_wrote_still_load(tmp_path):
    """SEC-013: a sidecar with no `digest` predates the field and is not rejected for it."""
    saved = a_matrix()
    save_matrix(saved, tmp_path)
    sidecar_path = tmp_path / f"matrix-{saved.dictionary_key}-raw.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    del sidecar["digest"]
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")

    loaded = load_matrix(saved.dictionary_key, tmp_path)

    assert np.array_equal(loaded.X.toarray(), saved.X.toarray())
