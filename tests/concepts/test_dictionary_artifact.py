"""T2: the concept dictionary artifact.

Two properties matter here, and they are the ones Phase 1's audit was about: a
changed parameter must write a new artifact instead of overwriting the old one,
and an artifact must be verified against the pool it claims to describe on load
rather than merely carrying a record of it.
"""

import json

import numpy as np
import pytest

from concept_embeddings_rag.concepts.dictionary import (
    ConceptArtifactError,
    ConceptDictionary,
    dictionary_key,
    load_dictionary,
    save_dictionary,
)


def some_dictionary(k: int = 4, dim: int = 8, **overrides) -> ConceptDictionary:
    rng = np.random.default_rng(0)
    atoms = rng.normal(size=(k, dim)).astype(np.float32)
    atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
    fields = {
        "atoms": atoms,
        "k": k,
        "seed": 42,
        "sparsity_param": 0.1,
        "max_iter": 3,
        "unit_set_hash": "101f564fdcca620c",
        "model": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c",
        "code_version": "0.1.0",
    }
    fields.update(overrides)
    return ConceptDictionary(**fields)


def test_key_changes_with_every_parameter_that_defines_the_artifact():
    base = dictionary_key(k=512, seed=42, alpha=0.1, unit_set_hash="abc", model="m", revision="r")
    assert base != dictionary_key(
        k=1024, seed=42, alpha=0.1, unit_set_hash="abc", model="m", revision="r"
    )
    assert base != dictionary_key(
        k=512, seed=7, alpha=0.1, unit_set_hash="abc", model="m", revision="r"
    )
    assert base != dictionary_key(
        k=512, seed=42, alpha=0.2, unit_set_hash="abc", model="m", revision="r"
    )
    assert base != dictionary_key(
        k=512, seed=42, alpha=0.1, unit_set_hash="other", model="m", revision="r"
    )
    assert base != dictionary_key(
        k=512, seed=42, alpha=0.1, unit_set_hash="abc", model="other", revision="r"
    )
    assert base != dictionary_key(
        k=512, seed=42, alpha=0.1, unit_set_hash="abc", model="m", revision="r2"
    )


def test_round_trip_preserves_the_atoms_bit_for_bit(tmp_path):
    original = some_dictionary()
    save_dictionary(original, tmp_path)

    loaded = load_dictionary(original.key, tmp_path)

    assert np.array_equal(loaded.atoms, original.atoms)
    assert loaded.atoms.dtype == np.float32
    assert loaded.k == original.k
    assert loaded.seed == original.seed
    assert loaded.sparsity_param == original.sparsity_param
    assert loaded.unit_set_hash == original.unit_set_hash
    assert loaded.model == original.model
    assert loaded.revision == original.revision
    assert loaded.code_version == original.code_version


def test_a_changed_parameter_writes_a_new_artifact_instead_of_overwriting(tmp_path):
    save_dictionary(some_dictionary(k=4), tmp_path)
    save_dictionary(some_dictionary(k=4, seed=7), tmp_path)

    assert len(list(tmp_path.glob("dictionary-*.npz"))) == 2


def test_loading_against_a_different_pool_aborts(tmp_path):
    saved = some_dictionary()
    save_dictionary(saved, tmp_path)

    with pytest.raises(ConceptArtifactError):
        load_dictionary(saved.key, tmp_path, expected_unit_set_hash="a-different-pool")


def test_loading_the_pool_it_was_induced_from_is_accepted(tmp_path):
    saved = some_dictionary()
    save_dictionary(saved, tmp_path)

    loaded = load_dictionary(saved.key, tmp_path, expected_unit_set_hash=saved.unit_set_hash)

    assert loaded.k == saved.k


def test_a_missing_artifact_reports_the_key_rather_than_raising_from_numpy(tmp_path):
    with pytest.raises(ConceptArtifactError, match="deadbeef"):
        load_dictionary("deadbeef", tmp_path)


def test_the_artifact_loads_without_pickle(tmp_path):
    saved = some_dictionary()
    path = save_dictionary(saved, tmp_path)

    with np.load(path, allow_pickle=False) as payload:
        assert payload["atoms"].shape == saved.atoms.shape


def test_the_sidecar_records_the_configuration_that_produced_it(tmp_path):
    saved = some_dictionary()
    save_dictionary(saved, tmp_path)

    sidecar = json.loads((tmp_path / f"dictionary-{saved.key}.json").read_text(encoding="utf-8"))

    for field in ("k", "seed", "sparsity_param", "unit_set_hash", "model", "revision"):
        assert field in sidecar
    assert sidecar["key"] == saved.key
    assert sidecar["code_version"] == saved.code_version


def test_a_dictionary_whose_atoms_are_not_unit_norm_is_refused(tmp_path):
    atoms = np.ones((3, 4), dtype=np.float32)  # norm 2, not 1
    with pytest.raises(ConceptArtifactError):
        save_dictionary(some_dictionary(atoms=atoms, k=3), tmp_path)


def test_merged_from_survives_the_round_trip(tmp_path):
    saved = some_dictionary(merged_from=[[0], [1, 3], [2]])
    save_dictionary(saved, tmp_path)

    loaded = load_dictionary(saved.key, tmp_path)

    assert loaded.merged_from == [[0], [1, 3], [2]]


def test_a_dictionary_without_a_merge_history_round_trips_as_none(tmp_path):
    saved = some_dictionary()
    save_dictionary(saved, tmp_path)

    assert load_dictionary(saved.key, tmp_path).merged_from is None
