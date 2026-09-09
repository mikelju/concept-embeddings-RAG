"""T8: the row-normalized view.

Raw weights live on the reconstruction scale, so comparing them across units is
invalid. The normalized view is what makes unit-to-unit comparison meaningful,
and it is derived rather than stored: two persisted views can silently disagree.
"""

import numpy as np
from scipy import sparse

from concept_embeddings_rag.concepts.coding import ConceptMatrix, row_normalized


def a_matrix(rows: list[list[float]]) -> ConceptMatrix:
    X = sparse.csr_matrix(np.array(rows, dtype=np.float32))
    return ConceptMatrix(
        X=X,
        unit_ids=[f"unit{i}" for i in range(len(rows))],
        dictionary_key="abc123",
        view="raw",
        coding_alpha=0.01,
        mean_active_per_unit=float((X.toarray() > 0).sum(axis=1).mean()),
        reconstruction_error=0.4,
    )


def test_rows_sum_to_one():
    normalized = row_normalized(a_matrix([[1.0, 3.0, 0.0], [2.0, 2.0, 4.0]]))

    assert np.allclose(normalized.X.toarray().sum(axis=1), 1.0)


def test_an_orphan_row_stays_zero_instead_of_dividing_by_zero():
    normalized = row_normalized(a_matrix([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))

    dense = normalized.X.toarray()
    assert np.array_equal(dense[0], np.zeros(3, dtype=np.float32))
    assert not np.isnan(dense).any()


def test_the_raw_matrix_is_not_mutated():
    original = a_matrix([[1.0, 3.0, 0.0], [2.0, 2.0, 4.0]])
    before = original.X.toarray().copy()

    row_normalized(original)

    assert np.array_equal(original.X.toarray(), before)
    assert original.view == "raw"


def test_the_view_is_named_on_the_result():
    normalized = row_normalized(a_matrix([[1.0, 1.0, 0.0]]))

    assert normalized.view == "row_normalized"


def test_the_relative_order_within_a_unit_is_preserved():
    normalized = row_normalized(a_matrix([[1.0, 3.0, 2.0]]))

    dense = normalized.X.toarray()[0]
    assert dense[1] > dense[2] > dense[0]


def test_normalizing_twice_changes_nothing_further():
    once = row_normalized(a_matrix([[1.0, 3.0, 0.0], [2.0, 2.0, 4.0]]))
    twice = row_normalized(once)

    assert np.allclose(once.X.toarray(), twice.X.toarray())
