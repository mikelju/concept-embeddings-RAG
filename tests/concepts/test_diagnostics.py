"""T10: the diagnostics that make the space inspectable before Phase 3 judges it.

A bad concept space should be caught here, not blamed on retrieval later. The
six statistics of the data contract are the ones that can show it: a space that
is one-hot, a space whose atoms are mostly dead, a space full of units no concept
reaches, or a co-activation graph so degenerate that Phase 4 has nothing to
travel across.

Everything is computed over the whole pool. There is no split parameter to break
it down by, because the test split is not looked at in this phase, not even
descriptively.
"""

import inspect
import json

import numpy as np
import pytest
from scipy import sparse

from concept_embeddings_rag.concepts.coding import ConceptMatrix
from concept_embeddings_rag.concepts.diagnostics import (
    coactivation_counts,
    compute_diagnostics,
    load_diagnostics,
    save_diagnostics,
)


def a_matrix(rows: list[list[float]], **overrides) -> ConceptMatrix:
    X = sparse.csr_matrix(np.array(rows, dtype=np.float32))
    fields = {
        "X": X,
        "unit_ids": [f"unit{i:04d}" for i in range(len(rows))],
        "dictionary_key": "abc123",
        "view": "raw",
        "coding_alpha": 0.01,
        "mean_active_per_unit": float((X.toarray() > 0).sum(axis=1).mean()),
        "reconstruction_error": 0.4,
    }
    fields.update(overrides)
    return ConceptMatrix(**fields)


# Four units, five concepts. Concept 4 is activated by nobody (a dead atom) and
# unit 3 activates nothing at all (an orphan unit).
SAMPLE = [
    [1.0, 2.0, 0.0, 0.0, 0.0],
    [0.0, 3.0, 1.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 4.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
]


def test_active_per_unit_carries_the_whole_distribution():
    diagnostics = compute_diagnostics(a_matrix(SAMPLE))

    stats = diagnostics.active_per_unit
    assert set(stats) == {"mean", "median", "p5", "p95", "max"}
    assert stats["max"] == 2.0
    assert stats["mean"] == pytest.approx(1.5)


def test_units_per_concept_carries_the_whole_distribution():
    diagnostics = compute_diagnostics(a_matrix(SAMPLE))

    stats = diagnostics.units_per_concept
    assert set(stats) == {"mean", "median", "p5", "p95", "max"}
    assert stats["max"] == 2.0


def test_a_concept_no_unit_activates_is_reported_dead():
    diagnostics = compute_diagnostics(a_matrix(SAMPLE))

    assert diagnostics.dead_atoms == [4]


def test_the_minimum_activation_count_is_declared_and_moves_the_answer():
    diagnostics = compute_diagnostics(a_matrix(SAMPLE), dead_atom_min_units=2)

    assert diagnostics.dead_atom_min_units == 2
    assert diagnostics.dead_atoms == [2, 3, 4]


def test_orphan_units_are_listed_by_id_never_merely_counted():
    diagnostics = compute_diagnostics(a_matrix(SAMPLE))

    assert diagnostics.orphan_units == ["unit0003"]


def test_top_units_per_concept_rank_by_weight_so_a_concept_reads_as_its_evidence():
    diagnostics = compute_diagnostics(a_matrix(SAMPLE), top_units=2)

    assert diagnostics.top_units_per_concept["1"] == ["unit0001", "unit0000"]
    assert diagnostics.top_units_per_concept["3"] == ["unit0002"]
    assert diagnostics.top_units_per_concept["4"] == []


def test_coactivation_is_symmetric_and_excludes_the_concept_itself():
    counts = coactivation_counts(a_matrix(SAMPLE).X)

    dense = counts.toarray()
    assert np.array_equal(dense, dense.T)
    assert np.array_equal(np.diag(dense), np.zeros(dense.shape[0]))


def test_coactivation_counts_the_units_in_which_both_concepts_are_active():
    counts = coactivation_counts(a_matrix(SAMPLE).X).toarray()

    assert counts[0][1] == 1  # unit0000 alone
    assert counts[1][2] == 1  # unit0001 alone
    assert counts[0][3] == 1  # unit0002 alone
    assert counts[0][2] == 0


def test_top_coactivations_are_ranked_and_carry_their_count():
    diagnostics = compute_diagnostics(a_matrix(SAMPLE), top_coactivations=3)

    assert diagnostics.top_coactivations["1"] == [[0, 1], [2, 1]]
    assert diagnostics.top_coactivations["4"] == []


def test_the_diagnostics_take_no_split_or_gold_argument():
    """A structural guard: there is nothing here to break down by split."""
    parameters = set(inspect.signature(compute_diagnostics).parameters)

    assert not parameters & {"split", "questions", "gold", "gold_unit_ids", "queries"}


def test_the_diagnostics_are_written_as_an_artifact_and_read_back(tmp_path):
    original = compute_diagnostics(a_matrix(SAMPLE))
    save_diagnostics(original, tmp_path)

    loaded = load_diagnostics(original.dictionary_key, tmp_path)

    assert loaded.dictionary_key == original.dictionary_key
    assert loaded.dead_atoms == original.dead_atoms
    assert loaded.orphan_units == original.orphan_units
    assert loaded.active_per_unit == original.active_per_unit
    assert loaded.top_units_per_concept == original.top_units_per_concept
    assert loaded.top_coactivations == original.top_coactivations


def test_the_artifact_names_the_dictionary_it_describes(tmp_path):
    diagnostics = compute_diagnostics(a_matrix(SAMPLE, dictionary_key="feedface"))
    path = save_diagnostics(diagnostics, tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["dictionary_key"] == "feedface"
    assert payload["view"] == "raw"
    assert payload["n_units"] == 4
    assert payload["n_concepts"] == 5


def test_two_dictionaries_produce_two_comparable_artifacts(tmp_path):
    save_diagnostics(compute_diagnostics(a_matrix(SAMPLE, dictionary_key="aaa")), tmp_path)
    save_diagnostics(compute_diagnostics(a_matrix(SAMPLE, dictionary_key="bbb")), tmp_path)

    assert len(list(tmp_path.glob("diagnostics-*.json"))) == 2


# --- T10b: dictionary quality against a null baseline (HU-7) -----------------
#
# The failure HU-5 cannot see: a concept activated by units that have nothing to
# do with each other is not dead, is not orphan, and reads plausibly one unit at
# a time. These tests are built so the metric can fail - a coherent concept and a
# mixed one, identical in every other respect - because a diagnostic that cannot
# rank those two would tell us nothing about the real space either.


def basis(index: int, dim: int = 8) -> list[float]:
    row = [0.0] * dim
    row[index] = 1.0
    return row


# Ten units. The first five all point the same way; the other five point five
# different ways. Concept 0 is activated by the first group, concept 1 by the
# second, and each atom is the first direction of its own group.
QUALITY_VECTORS = np.array([basis(0) for _ in range(5)] + [basis(i) for i in range(1, 6)])
QUALITY_ATOMS = np.array([basis(0), basis(1)])
QUALITY_ROWS = [[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5


def a_quality(rows=None, vectors=None, atoms=None, **overrides):
    from concept_embeddings_rag.concepts.diagnostics import compute_quality

    matrix = a_matrix(QUALITY_ROWS if rows is None else rows)
    return compute_quality(
        matrix,
        QUALITY_VECTORS if vectors is None else np.asarray(vectors),
        QUALITY_ATOMS if atoms is None else np.asarray(atoms),
        **overrides,
    )


def test_a_coherent_concept_scores_above_a_mixed_one():
    quality = a_quality()

    assert quality.coherence["0"] == pytest.approx(1.0)
    assert quality.coherence["1"] == pytest.approx(0.0)


def test_the_mixed_concept_is_the_one_listed_below_the_null():
    quality = a_quality()

    assert quality.below_null_concepts == [1]
    assert quality.coherence["1"] < quality.null_mean < quality.coherence["0"]


def test_the_null_baseline_is_measured_on_this_pool_and_travels_with_the_scores():
    quality = a_quality(top_units=3, null_samples=50, seed=7)

    assert quality.null_samples == 50
    assert quality.seed == 7
    assert quality.top_units == 3
    # Drawn from a pool that is neither identical nor orthogonal, so the baseline
    # sits strictly between the two extremes rather than at either of them.
    assert 0.0 < quality.null_mean < 1.0
    assert quality.null_sd > 0.0


def test_the_same_seed_gives_the_same_baseline_and_a_different_one_does_not():
    same = a_quality(top_units=3, seed=7).null_mean
    again = a_quality(top_units=3, seed=7).null_mean
    other = a_quality(top_units=3, seed=8).null_mean

    assert same == again
    assert same != other


def test_the_z_score_reads_the_coherence_against_the_null():
    quality = a_quality(top_units=3, seed=7)

    expected = (quality.coherence["0"] - quality.null_mean) / quality.null_sd
    assert quality.coherence_z["0"] == pytest.approx(expected)


def test_a_pool_too_small_to_sample_does_not_divide_by_zero():
    """Every draw is then the same set, so the spread is zero and z stays finite."""
    quality = a_quality(top_units=25)

    assert quality.null_sd == pytest.approx(0.0, abs=1e-9)
    assert quality.coherence_z["0"] == pytest.approx(quality.coherence["0"] - quality.null_mean)


def test_a_concept_below_the_minimum_is_undefined_never_scored_zero():
    rows = [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 2 + [[0.0, 0.0]] * 4

    quality = a_quality(rows, min_units=3)

    assert quality.undefined_concepts == [1]
    assert "1" not in quality.coherence
    assert "1" not in quality.coherence_z


def test_atom_alignment_falls_when_the_atom_sits_away_from_its_evidence():
    quality = a_quality()

    # Atom 0 is exactly what its five units are; atom 1 is one of the five
    # directions its units point in, so it matches a fifth of its own evidence.
    assert quality.atom_alignment["0"] == pytest.approx(1.0)
    assert quality.atom_alignment["1"] == pytest.approx(0.2)


def test_top_mass_share_separates_a_sharp_concept_from_a_spread_one():
    sharp = [[10.0, 1.0]] * 3 + [[0.01, 1.0]] * 7
    vectors = np.array([basis(0) for _ in range(10)])

    quality = a_quality(sharp, vectors=vectors, top_units=3)

    assert quality.top_mass_share["0"] > 0.99
    assert quality.top_mass_share["1"] == pytest.approx(0.3)


def test_the_quality_diagnostic_never_changes_the_space():
    from concept_embeddings_rag.concepts.diagnostics import compute_quality

    matrix = a_matrix(QUALITY_ROWS)
    before = matrix.X.toarray().copy()
    atoms = QUALITY_ATOMS.copy()

    quality = compute_quality(matrix, QUALITY_VECTORS, atoms)

    assert np.array_equal(matrix.X.toarray(), before)
    assert np.array_equal(atoms, QUALITY_ATOMS)
    assert len(quality.coherence) + len(quality.undefined_concepts) == before.shape[1]


def test_embeddings_that_do_not_describe_the_matrix_are_refused():
    from concept_embeddings_rag.concepts.dictionary import ConceptArtifactError

    with pytest.raises(ConceptArtifactError, match="embeddings"):
        a_quality(vectors=QUALITY_VECTORS[:4])
    with pytest.raises(ConceptArtifactError, match="atoms"):
        a_quality(atoms=QUALITY_ATOMS[:1])
    with pytest.raises(ConceptArtifactError, match="same space"):
        a_quality(atoms=np.array([basis(0, dim=4), basis(1, dim=4)]))


def test_diagnostics_without_embeddings_carry_no_quality():
    """The structural half still runs, which is what the synthetic tests above use."""
    assert compute_diagnostics(a_matrix(SAMPLE)).quality is None


def test_diagnostics_with_embeddings_carry_the_quality_block():
    diagnostics = compute_diagnostics(
        a_matrix(QUALITY_ROWS), vectors=QUALITY_VECTORS, atoms=QUALITY_ATOMS
    )

    assert diagnostics.quality is not None
    assert diagnostics.quality.below_null_concepts == [1]


def test_the_quality_block_survives_the_round_trip(tmp_path):
    original = compute_diagnostics(
        a_matrix(QUALITY_ROWS), vectors=QUALITY_VECTORS, atoms=QUALITY_ATOMS
    )
    save_diagnostics(original, tmp_path)

    loaded = load_diagnostics(original.dictionary_key, tmp_path)

    assert loaded.quality == original.quality


def test_the_quality_block_is_written_into_the_artifact(tmp_path):
    diagnostics = compute_diagnostics(
        a_matrix(QUALITY_ROWS), vectors=QUALITY_VECTORS, atoms=QUALITY_ATOMS
    )
    path = save_diagnostics(diagnostics, tmp_path)

    quality = json.loads(path.read_text(encoding="utf-8"))["quality"]

    assert quality["null_mean"] > 0.0
    assert quality["seed"] == diagnostics.quality.seed
    assert set(quality["coherence"]) == {"0", "1"}
