"""T9: deduplicating the dictionary, and the log that makes the merge auditable.

A dictionary that carries the same concept three times inflates K without adding
a dimension, and splits the activation of a unit across near-identical atoms -
which is exactly the co-activation Phase 4 is meant to travel over.

Two rules matter here. The merge criterion is explicit and recorded, so anyone
can check the decisions instead of taking them on faith; and `X` is **recomputed**
against the merged dictionary rather than obtained by summing the columns of the
old one. A column sum is not the output of any coding, so the weight semantics of
HU-2 would stop holding the moment it were used.
"""

import numpy as np
import pytest

from concept_embeddings_rag.concepts.coding import code_corpus
from concept_embeddings_rag.concepts.dedup import (
    deduplicate_dictionary,
    find_duplicate_groups,
    load_merge_log,
    recode_after_merge,
    save_merge_log,
)
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary


def unit(index: int, dim: int = 8) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float64)
    vector[index] = 1.0
    return vector


def at_cosine(base: int, other: int, cosine: float, dim: int = 8) -> np.ndarray:
    """A vector at exactly `cosine` from `unit(base)`, in the (base, other) plane."""
    return cosine * unit(base, dim) + np.sqrt(1.0 - cosine**2) * unit(other, dim)


def a_dictionary(atoms: np.ndarray, **overrides) -> ConceptDictionary:
    atoms = np.asarray(atoms, dtype=np.float64)
    atoms = atoms / np.linalg.norm(atoms, axis=1, keepdims=True)
    fields = {
        "atoms": atoms.astype(np.float32),
        "k": atoms.shape[0],
        "seed": 42,
        "sparsity_param": 0.1,
        "max_iter": 3,
        "unit_set_hash": "101f564fdcca620c",
        "model": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c",
    }
    fields.update(overrides)
    return ConceptDictionary(**fields)


def near_duplicate_pair() -> ConceptDictionary:
    """Atoms 0 and 1 sit at cosine 0.91; atoms 2 and 3 are orthogonal to everything."""
    return a_dictionary(
        np.stack([unit(0), at_cosine(0, 1, 0.91), unit(2), unit(3)]),
    )


def test_two_duplicated_atoms_merge_and_two_unrelated_ones_do_not():
    groups = find_duplicate_groups(near_duplicate_pair().atoms, threshold=0.90)

    assert groups == [[0, 1]]


def test_a_pair_below_the_threshold_is_left_alone():
    """The threshold is the criterion, so it has to actually decide."""
    dictionary = near_duplicate_pair()

    assert find_duplicate_groups(dictionary.atoms, threshold=0.95) == []


def test_atoms_transitively_similar_land_in_one_group():
    atoms = np.stack([unit(0), at_cosine(0, 1, 0.93), at_cosine(0, 1, 0.97), unit(3)])

    groups = find_duplicate_groups(a_dictionary(atoms).atoms, threshold=0.90)

    assert groups == [[0, 1, 2]]


def test_the_merged_dictionary_is_smaller_and_still_unit_norm():
    merged, _ = deduplicate_dictionary(near_duplicate_pair(), threshold=0.90)

    assert merged.k == 3
    assert merged.atoms.shape == (3, 8)
    assert np.allclose(np.linalg.norm(merged.atoms, axis=1), 1.0, atol=1e-5)


def test_the_merged_dictionary_records_which_atoms_it_came_from():
    merged, _ = deduplicate_dictionary(near_duplicate_pair(), threshold=0.90)

    assert merged.merged_from == [[0, 1], [2], [3]]


def test_the_deduplicated_dictionary_gets_its_own_key():
    """Otherwise it would overwrite the dictionary it was derived from."""
    original = near_duplicate_pair()
    merged, _ = deduplicate_dictionary(original, threshold=0.90)

    assert merged.key != original.key


def test_a_dictionary_with_nothing_to_merge_keeps_every_atom():
    original = a_dictionary(np.stack([unit(0), unit(1), unit(2), unit(3)]))

    merged, log = deduplicate_dictionary(original, threshold=0.90)

    assert merged.k == original.k
    assert np.allclose(merged.atoms, original.atoms, atol=1e-6)
    assert log.groups == []
    assert merged.key != original.key


def test_the_merge_log_records_members_similarities_and_the_resulting_index():
    _, log = deduplicate_dictionary(near_duplicate_pair(), threshold=0.90)

    assert len(log.groups) == 1
    group = log.groups[0]
    assert group.members == [0, 1]
    assert group.resulting_index == 0
    assert len(group.similarities) == 1
    left, right, similarity = group.similarities[0]
    assert (left, right) == (0, 1)
    assert similarity == pytest.approx(0.91, abs=1e-3)


def test_the_merge_log_reports_the_atom_count_before_and_after():
    _, log = deduplicate_dictionary(near_duplicate_pair(), threshold=0.90)

    assert log.k_before == 4
    assert log.k_after == 3
    assert log.merged_fraction == pytest.approx(0.25)
    assert log.threshold == 0.90


def test_the_merge_log_is_a_versioned_artifact_not_console_output(tmp_path):
    merged, log = deduplicate_dictionary(near_duplicate_pair(), threshold=0.90)
    save_merge_log(log, tmp_path)

    loaded = load_merge_log(merged.key, tmp_path)

    assert loaded.k_before == log.k_before
    assert loaded.k_after == log.k_after
    assert loaded.threshold == log.threshold
    assert [g.members for g in loaded.groups] == [g.members for g in log.groups]
    assert [g.resulting_index for g in loaded.groups] == [g.resulting_index for g in log.groups]


def test_merging_beyond_the_budget_is_reported_as_a_finding_about_k():
    """An oversized dictionary is a fact about K, not a detail to absorb quietly."""
    atoms = np.stack([unit(0), at_cosine(0, 1, 0.99), at_cosine(0, 2, 0.99), unit(3)])

    _, log = deduplicate_dictionary(a_dictionary(atoms), threshold=0.90, max_merged_fraction=0.10)

    assert log.exceeded_merge_budget is True
    assert log.finding is not None
    assert "k" in log.finding.lower()


def test_a_merge_within_budget_reports_no_finding():
    _, log = deduplicate_dictionary(near_duplicate_pair(), threshold=0.90, max_merged_fraction=0.50)

    assert log.exceeded_merge_budget is False
    assert log.finding is None


def test_x_is_recomputed_as_a_transform_against_the_merged_dictionary():
    original = near_duplicate_pair()
    vectors = np.stack([unit(0), unit(2), unit(3), at_cosine(0, 1, 0.5)]).astype(np.float32)
    ids = [f"unit{i}" for i in range(len(vectors))]
    merged, _ = deduplicate_dictionary(original, threshold=0.90)

    recomputed = recode_after_merge(merged, vectors, ids, alpha=0.01)
    directly = code_corpus(merged, vectors, ids, alpha=0.01)

    assert np.array_equal(recomputed.X.toarray(), directly.X.toarray())
    assert recomputed.dictionary_key == merged.key
    assert recomputed.X.shape == (len(vectors), merged.k)


def test_the_recomputed_x_is_not_the_column_sum_of_the_old_one():
    """Summing columns would leave a matrix that is the output of no coding at all."""
    original = near_duplicate_pair()
    vectors = np.stack([unit(0), unit(2), unit(3), at_cosine(0, 1, 0.5)]).astype(np.float32)
    ids = [f"unit{i}" for i in range(len(vectors))]
    before = code_corpus(original, vectors, ids, alpha=0.01)
    merged, log = deduplicate_dictionary(original, threshold=0.90)

    recomputed = recode_after_merge(merged, vectors, ids, alpha=0.01)

    group = log.groups[0]
    column_sum = np.asarray(before.X.toarray()[:, group.members].sum(axis=1)).ravel()
    merged_column = np.asarray(recomputed.X.toarray()[:, group.resulting_index]).ravel()
    assert not np.allclose(column_sum, merged_column, atol=1e-3)
