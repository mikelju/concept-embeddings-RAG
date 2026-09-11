"""T7: rarity damping, the shape declared in decision D7 before any number was seen.

`w_j = log(1 + N / (1 + df_j))`, with `df_j` the number of units that activate concept
`j` and `N` the pool size. It multiplies the *query's* concept vector, which is
algebraically the same as scaling the columns of `X` but leaves the matrix and the
dictionary exactly as Phase 2 wrote them - the property HU-6 asks for, and the reason
a damped run is comparable to its own undamped one without recoding anything.

Two things this file pins down beyond the arithmetic. The first is that `none` is the
identity and is provably so: not "close enough", bit-identical scores. The second is
that damping demotes and never prunes. Every weight is strictly positive, so the set of
units that share something with the question is the same before and after; what changes
is the order. The spec's anti-goal is explicit that pruning concepts belongs to a Phase 5
ablation, and a weight that could reach zero would smuggle it in here.

Where `df` comes from is worth stating. D7 says it is read from the Phase 2 diagnostics
rather than recomputed, and the intent - never re-derive it from the embeddings, never
recode anything - is respected exactly; but the persisted `SpaceDiagnostics` records
`units_per_concept` as a five-number summary, not the per-concept vector. So `df` is read
off the sparsity pattern of the `ConceptMatrix` that Phase 2 wrote and this phase loads
hash-verified. `test_the_support_agrees_with_what_the_phase_2_diagnostics_recorded` ties
the two together: the vector read here must reproduce the summary the artifact carries.
"""

import numpy as np
import pytest
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.concepts.coding import ConceptMatrix
from concept_embeddings_rag.concepts.diagnostics import compute_diagnostics
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary
from concept_embeddings_rag.retrieval.conceptual import (
    UNDAMPED,
    ConceptualRetriever,
    concept_support,
    rarity_weights,
)

# The pool size and the hub of decision D7, so the worked example in the plan is
# asserted rather than trusted.
POOL_SIZE = 19_366
HUB_SUPPORT = 19_250
THIN_SUPPORT = config.THIN_CONCEPT_MAX_UNITS  # 30


class TableBackend:
    """Serves a stored vector per query string; nothing is derived from the text."""

    name = "fake"
    revision = "v0"
    normalize = True

    def __init__(self, table: dict[str, np.ndarray]) -> None:
        self._table = table
        self.dim = int(next(iter(table.values())).shape[0])

    def encode(self, texts):
        return np.stack([self._table[text] for text in texts]).astype(np.float32)


def dictionary_of(atoms: np.ndarray) -> ConceptDictionary:
    atoms = np.asarray(atoms, dtype=np.float32)
    atoms = atoms / np.linalg.norm(atoms, axis=1, keepdims=True)
    return ConceptDictionary(
        atoms=atoms,
        k=atoms.shape[0],
        seed=42,
        sparsity_param=0.05,
        max_iter=3,
        unit_set_hash="101f564fdcca620c",
        model="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c",
    )


def matrix_of(
    rows: np.ndarray, unit_ids: list[str], dictionary: ConceptDictionary
) -> ConceptMatrix:
    X = sparse.csr_matrix(np.asarray(rows, dtype=np.float32))
    return ConceptMatrix(
        X=X,
        unit_ids=unit_ids,
        dictionary_key=dictionary.key,
        view="raw",
        coding_alpha=0.07,
        mean_active_per_unit=float(X.nnz) / X.shape[0],
        reconstruction_error=0.0,
    )


# --- A pool with a hub in it --------------------------------------------------

HUB = 0
RARE = 1
HUB_ONLY = "unit-on-the-hub"
RARE_ONLY = "unit-on-the-rare-concept"
# Wider than `config.QUERY_TOP_M`, so the truncating arm has something to truncate:
# a dictionary smaller than the truncation is a fixture that cannot run the arm.
CONCEPTS = 20


def a_pool_with_a_hub() -> tuple[ConceptMatrix, ConceptDictionary]:
    """Ninety-eight units of house style, one unit on the hub, one on a rare concept.

    Concept 0 is what the corpus says everywhere; concept 1 is what one unit says.
    The hub unit activates its concept twice as strongly as the rare unit activates
    its own, so undamped it wins on merit and nothing but rarity can reverse that.
    """
    dictionary = dictionary_of(np.eye(CONCEPTS))

    rows = np.zeros((100, CONCEPTS), dtype=np.float32)
    rows[:98, HUB] = 0.5
    rows[:98, 2] = 1.0
    rows[:98, 3] = 1.0
    rows[98, HUB] = 2.0
    rows[99, RARE] = 1.0

    unit_ids = [f"u{i:02d}" for i in range(98)] + [HUB_ONLY, RARE_ONLY]
    return matrix_of(rows, unit_ids, dictionary), dictionary


def a_query_that_asks_for_both() -> dict[str, np.ndarray]:
    """Projects equally onto the hub and onto the rare concept: 0.707 on each."""
    vector = np.zeros(CONCEPTS, dtype=np.float32)
    vector[HUB] = 1.0
    vector[RARE] = 1.0
    return {"q": vector / np.linalg.norm(vector)}


def a_mixed_pool(n_units: int = 60, k: int = 16, seed: int = 3) -> ConceptMatrix:
    """A coded corpus in miniature: sparse rows, a hub, and a concept nobody uses."""
    rng = np.random.default_rng(seed)
    rows = np.abs(rng.normal(size=(n_units, k))).astype(np.float32)
    rows[rows < 1.0] = 0.0
    rows[:, 0] = 1.0  # a hub: every unit activates it
    rows[:, 1] = 0.0  # a dead atom: no unit does
    return matrix_of(rows, [f"u{i}" for i in range(n_units)], dictionary_of(np.eye(k)))


# --- The declared function ----------------------------------------------------


def test_the_weight_is_the_declared_idf_shape_and_nothing_else():
    support = np.array([0, 1, 30, 500, 19_250], dtype=np.int64)

    weights = rarity_weights(support, POOL_SIZE)

    assert np.allclose(weights, np.log(1.0 + POOL_SIZE / (1.0 + support)), rtol=1e-6)


def test_a_concept_on_every_unit_is_demoted_relative_to_a_concept_on_thirty():
    """The headline criterion of T7, at the numbers decision D7 wrote down."""
    hub, thin = rarity_weights(np.array([HUB_SUPPORT, THIN_SUPPORT]), POOL_SIZE)

    assert hub < thin
    assert pytest.approx(float(hub), abs=0.01) == 0.69
    assert pytest.approx(float(thin), abs=0.05) == 6.4
    assert pytest.approx(float(thin / hub), abs=0.3) == 9.0


def test_the_weight_falls_as_the_support_rises_with_no_flat_stretch():
    weights = rarity_weights(np.arange(0, POOL_SIZE, 97), POOL_SIZE)

    assert (np.diff(weights) < 0.0).all()


def test_the_hub_is_demoted_and_never_deleted():
    """`log(1 + N/(1+df))` is positive for every df there is, including df = N.

    A weight that could reach zero would prune the concept, and the spec puts pruning
    in a Phase 5 ablation: this phase damps by a declared function and removes nothing.
    """
    weights = rarity_weights(np.arange(POOL_SIZE + 1), POOL_SIZE)

    assert (weights > 0.0).all()
    assert np.isfinite(weights).all()


def test_a_concept_no_unit_activates_gets_the_largest_weight_and_it_stays_finite():
    """`1 + df` in the denominator is what makes df = 0 an ordinary case."""
    weights = rarity_weights(np.array([0, 1]), POOL_SIZE)

    assert float(weights[0]) == pytest.approx(float(np.log1p(POOL_SIZE)))
    assert float(weights[0]) > float(weights[1])


def test_a_support_larger_than_the_pool_is_refused_rather_than_weighted():
    with pytest.raises(ValueError, match="support"):
        rarity_weights(np.array([POOL_SIZE + 1]), POOL_SIZE)


def test_a_negative_support_is_refused():
    with pytest.raises(ValueError, match="support"):
        rarity_weights(np.array([-1, 5]), POOL_SIZE)


def test_an_empty_pool_is_refused_rather_than_dividing_by_it():
    with pytest.raises(ValueError, match="pool"):
        rarity_weights(np.array([0]), 0)


def test_the_support_must_arrive_as_one_number_per_concept():
    with pytest.raises(ValueError, match="1-D"):
        rarity_weights(np.array([[1, 2], [3, 4]]), POOL_SIZE)


def test_the_weights_are_shaped_for_the_dictionary_they_will_multiply():
    matrix = a_mixed_pool()

    weights = rarity_weights(concept_support(matrix.X), matrix.X.shape[0])

    assert weights.shape == (matrix.X.shape[1],)


# --- Support read off the artifact, not recounted from the corpus -------------


def test_concept_support_counts_the_units_that_activate_each_concept():
    matrix = a_mixed_pool()

    support = concept_support(matrix.X)

    assert support[0] == matrix.X.shape[0]  # the hub
    assert support[1] == 0  # the dead atom
    assert (support <= matrix.X.shape[0]).all()


def test_a_stored_zero_is_not_support():
    """A concept whose weight happens to be zero is not a concept the unit activates."""
    X = sparse.csr_matrix(
        (
            np.array([1.0, 0.0], dtype=np.float32),  # the second is stored and still zero
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int32),
        ),
        shape=(2, 2),
    )
    assert X.nnz == 2, "the fixture must keep the explicit zero, or it proves nothing"

    support = concept_support(X)

    assert support.tolist() == [1, 0]


def test_the_support_agrees_with_what_the_phase_2_diagnostics_recorded():
    """D7 reads `df` from Phase 2. The artifact keeps a summary, so the summary must match.

    If these two ever disagreed, the damping would be weighting a space other than the
    one the diagnostics describe, and every structural column reported beside recall
    would be about a different matrix from the one that produced it.
    """
    matrix = a_mixed_pool()

    support = concept_support(matrix.X).astype(np.float64)
    recorded = compute_diagnostics(matrix).units_per_concept

    assert recorded["mean"] == pytest.approx(float(support.mean()))
    assert recorded["median"] == pytest.approx(float(np.median(support)))
    assert recorded["p5"] == pytest.approx(float(np.percentile(support, 5)))
    assert recorded["p95"] == pytest.approx(float(np.percentile(support, 95)))
    assert recorded["max"] == pytest.approx(float(support.max()))


def test_the_dead_atoms_the_diagnostics_names_are_the_ones_with_no_support():
    matrix = a_mixed_pool()

    diagnostics = compute_diagnostics(matrix, dead_atom_min_units=1)
    unsupported = np.flatnonzero(concept_support(matrix.X) == 0)

    assert diagnostics.dead_atoms == unsupported.tolist()


# --- `none` is the identity, and provably so ----------------------------------


def retriever(damping: str = UNDAMPED, weights: np.ndarray | None = None) -> ConceptualRetriever:
    matrix, dictionary = a_pool_with_a_hub()
    return ConceptualRetriever(
        matrix,
        dictionary,
        TableBackend(a_query_that_asks_for_both()),
        arm="projection_full",
        damping=damping,
        concept_weights=weights,
    )


def test_none_reproduces_the_undamped_ranking_exactly():
    """Not "close enough": the same units in the same order with the same scores."""
    matrix, dictionary = a_pool_with_a_hub()
    backend = TableBackend(a_query_that_asks_for_both())
    plain = ConceptualRetriever(matrix, dictionary, backend, arm="projection_full")
    declared = ConceptualRetriever(
        matrix, dictionary, backend, arm="projection_full", damping=UNDAMPED
    )

    assert plain.retrieve("q", top_k=100) == declared.retrieve("q", top_k=100)


@pytest.mark.parametrize("arm", config.QUERY_OPERATORS)
def test_none_is_the_identity_under_every_query_operator_arm(arm: str):
    matrix, dictionary = a_pool_with_a_hub()
    backend = TableBackend(a_query_that_asks_for_both())
    ones = np.ones(dictionary.k, dtype=np.float32)

    plain = ConceptualRetriever(matrix, dictionary, backend, arm=arm)
    flat = ConceptualRetriever(
        matrix, dictionary, backend, arm=arm, damping="idf", concept_weights=ones
    )

    assert plain.retrieve("q", top_k=100) == flat.retrieve("q", top_k=100)


def test_a_uniform_weight_rescales_every_score_without_reordering_anything():
    """Damping only matters because the weights differ; a flat one is a scale factor."""
    matrix, dictionary = a_pool_with_a_hub()
    backend = TableBackend(a_query_that_asks_for_both())
    plain = ConceptualRetriever(matrix, dictionary, backend, arm="projection_full")
    scaled = ConceptualRetriever(
        matrix,
        dictionary,
        backend,
        arm="projection_full",
        damping="idf",
        concept_weights=np.full(dictionary.k, 3.0, dtype=np.float32),
    )

    before = plain.retrieve("q", top_k=100)
    after = scaled.retrieve("q", top_k=100)

    assert [unit_id for unit_id, _ in before] == [unit_id for unit_id, _ in after]
    assert np.allclose([score for _, score in after], [3.0 * score for _, score in before])


# --- What the damping is for --------------------------------------------------


def damped_retriever() -> ConceptualRetriever:
    matrix, dictionary = a_pool_with_a_hub()
    weights = rarity_weights(concept_support(matrix.X), matrix.X.shape[0])
    return ConceptualRetriever(
        matrix,
        dictionary,
        TableBackend(a_query_that_asks_for_both()),
        arm="projection_full",
        damping="idf",
        concept_weights=weights,
    )


def test_undamped_the_house_style_wins_because_it_activates_more_strongly():
    ranked = [unit_id for unit_id, _ in retriever().retrieve("q", top_k=100)]

    assert ranked.index(HUB_ONLY) < ranked.index(RARE_ONLY)


def test_damped_the_rare_concept_wins_although_it_activates_half_as_strongly():
    """The whole point of HU-6, on a pool small enough to check by hand."""
    ranked = [unit_id for unit_id, _ in damped_retriever().retrieve("q", top_k=100)]

    assert ranked.index(RARE_ONLY) < ranked.index(HUB_ONLY)


def test_damping_reorders_the_pool_and_does_not_shrink_it():
    """Every unit that shared something with the question still does; the order moved."""
    plain = {unit_id for unit_id, score in retriever().retrieve("q", top_k=100) if score > 0.0}
    damped = {
        unit_id for unit_id, score in damped_retriever().retrieve("q", top_k=100) if score > 0.0
    }

    assert plain == damped
    assert plain == {HUB_ONLY, RARE_ONLY} | {f"u{i:02d}" for i in range(98)}


# --- The matrix and the dictionary are untouched ------------------------------


def bytes_of(matrix: ConceptMatrix, dictionary: ConceptDictionary) -> tuple[bytes, ...]:
    X = matrix.X.tocsr()
    return (
        X.data.tobytes(),
        X.indices.tobytes(),
        X.indptr.tobytes(),
        dictionary.atoms.tobytes(),
    )


def test_the_matrix_and_the_dictionary_are_byte_identical_after_a_damped_run():
    """HU-6: the damping is a retrieval parameter. No atom removed, no column scaled."""
    matrix, dictionary = a_pool_with_a_hub()
    weights = rarity_weights(concept_support(matrix.X), matrix.X.shape[0])
    before = bytes_of(matrix, dictionary)

    ConceptualRetriever(
        matrix,
        dictionary,
        TableBackend(a_query_that_asks_for_both()),
        arm="projection_full",
        damping="idf",
        concept_weights=weights,
    ).retrieve("q", top_k=100)

    assert bytes_of(matrix, dictionary) == before


def test_the_weight_vector_the_retriever_holds_cannot_be_edited_through_the_caller():
    matrix, dictionary = a_pool_with_a_hub()
    weights = rarity_weights(concept_support(matrix.X), matrix.X.shape[0])
    retrieval = ConceptualRetriever(
        matrix,
        dictionary,
        TableBackend(a_query_that_asks_for_both()),
        arm="projection_full",
        damping="idf",
        concept_weights=weights,
    )
    before = retrieval.retrieve("q", top_k=100)

    weights[:] = 1.0

    assert retrieval.retrieve("q", top_k=100) == before


# --- It travels in the recorded configuration ---------------------------------


def test_the_damping_mode_is_recorded_beside_the_metric_it_produced():
    assert retriever().describe()["damping"] == UNDAMPED
    assert damped_retriever().describe()["damping"] == "idf"


def test_the_recorded_mode_is_one_of_the_declared_arms():
    for description in (retriever().describe(), damped_retriever().describe()):
        assert description["damping"] in config.DAMPING_MODES


def test_a_damped_retriever_cannot_report_itself_as_undamped():
    """The failure this guards against is a mislabelled result, not a wrong one."""
    damped = damped_retriever()

    assert damped.describe()["damping"] != UNDAMPED
    assert damped.retrieve("q", top_k=100) != retriever().retrieve("q", top_k=100)
