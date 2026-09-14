"""T5: retrieval scored entirely in the concept space (System B).

The retriever holds `X`, a dictionary and a query backend, and nothing else. There
is no unit embedding to fall back on and no lexical index to blend in, so a hit is
a hit because the unit and the question activate the same concepts - which is the
only reason the fused numbers of HU-4 can be read as a conceptual contribution.

The score is `X @ c`, a dot product and deliberately not a cosine (decision D2): a
unit that activates a concept strongly should outrank one that activates it faintly,
and the `row_normalized` view is how that scale is removed when it is removed at all.
Both views are measured; neither is assumed.
"""

import inspect

import numpy as np
import pytest
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.concepts.coding import ConceptMatrix, row_normalized
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.harness import evaluate_retriever
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.conceptual import (
    PROJECTION,
    SPARSE_CODING,
    ConceptualRetriever,
)


class DirectionBackend:
    """Encodes a query as the literal vector written in the query string."""

    name = "fake"
    revision = "v0"
    dim = 4
    normalize = True

    def encode(self, texts):
        return np.array([[float(x) for x in t.split(",")] for t in texts], dtype=np.float32)


FIRST_CONCEPT = "1,0,0,0"


def a_dictionary(k: int = 4, dim: int = 4) -> ConceptDictionary:
    """The identity, so a projection is readable straight off the query string."""
    atoms = np.eye(k, dim, dtype=np.float32)
    return ConceptDictionary(
        atoms=atoms,
        k=k,
        seed=42,
        sparsity_param=0.05,
        max_iter=3,
        unit_set_hash="101f564fdcca620c",
        model="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c",
    )


def a_matrix(rows: list[list[float]], unit_ids: list[str], key: str) -> ConceptMatrix:
    X = sparse.csr_matrix(np.array(rows, dtype=np.float32))
    return ConceptMatrix(
        X=X,
        unit_ids=list(unit_ids),
        dictionary_key=key,
        view="raw",
        coding_alpha=0.07,
        mean_active_per_unit=float(X.nnz) / X.shape[0],
        reconstruction_error=0.0,
    )


def a_space() -> tuple[ConceptMatrix, ConceptDictionary]:
    """A pool of three units whose two views deliberately disagree.

    `broad` activates the first concept twice as hard as `focused` does, but spends
    most of its mass elsewhere. Raw weights rank it first; row-normalized weights
    rank it second. That disagreement is the whole reason both views are measured.
    """
    dictionary = a_dictionary()
    matrix = a_matrix(
        rows=[
            [1.0, 0.0, 0.0, 0.0],  # focused: all its mass on the queried concept
            [2.0, 6.0, 0.0, 0.0],  # broad: more of it, but a small share of itself
            [0.0, 0.0, 3.0, 0.0],  # absent: nothing to do with the query
        ],
        unit_ids=["focused", "broad", "absent"],
        key=dictionary.key,
    )
    return matrix, dictionary


def a_retriever(**overrides) -> ConceptualRetriever:
    matrix, dictionary = a_space()
    fields = {
        "matrix": matrix,
        "dictionary": dictionary,
        "backend": DirectionBackend(),
        "arm": "projection_full",
    }
    fields.update(overrides)
    return ConceptualRetriever(**fields)


# --- The interface the harness sees -------------------------------------------


def test_it_satisfies_the_retriever_interface():
    retriever: Retriever = a_retriever()

    assert isinstance(retriever, Retriever)
    assert retriever.name == "conceptual"


def test_retrieve_takes_a_query_and_a_top_k_and_has_nowhere_to_receive_candidates():
    parameters = set(inspect.signature(a_retriever().retrieve).parameters)

    assert parameters == {"query", "top_k"}


def test_the_harness_runs_it_with_harness_py_unmodified():
    questions = [
        Question(
            qid="q1",
            question=FIRST_CONCEPT,
            answer="an answer",
            gold_unit_ids=("focused",),
            supporting_facts=(("A", 0),),
            split="dev",
        )
    ]
    result = evaluate_retriever(
        a_retriever(),
        questions=questions,
        token_counts={"focused": 100, "broad": 100, "absent": 100},
        budgets=(400,),
        ks=(2,),
        top_k=10,
        config={
            "model": "fake",
            "revision": "v0",
            "unit_set_hash": "abc123",
            "seed": 42,
            "tokenizer": "fake-tokenizer",
            "code_version": "0.1.0",
            "top_k": 10,
        },
    )

    assert result.system == "conceptual"
    assert result.metrics["budget_400"]["gold_recall"] == 1.0


# --- The score is X @ c, and nothing else --------------------------------------


def test_the_score_is_the_matrix_row_against_the_query_concept_vector():
    hits = a_retriever().retrieve(FIRST_CONCEPT, top_k=3)

    assert hits == [("broad", 2.0), ("focused", 1.0), ("absent", 0.0)]


def test_the_two_views_rank_the_same_pool_differently():
    """If they ranked alike, measuring both would be a formality rather than a choice."""
    raw, dictionary = a_space()
    backend = DirectionBackend()

    on_raw = ConceptualRetriever(raw, dictionary, backend, arm="projection_full")
    on_normalized = ConceptualRetriever(
        row_normalized(raw), dictionary, backend, arm="projection_full"
    )

    assert [unit_id for unit_id, _ in on_raw.retrieve(FIRST_CONCEPT, top_k=2)] == [
        "broad",
        "focused",
    ]
    assert [unit_id for unit_id, _ in on_normalized.retrieve(FIRST_CONCEPT, top_k=2)] == [
        "focused",
        "broad",
    ]


def test_no_unit_embedding_and_no_lexical_index_is_anywhere_in_reach():
    """HU-2: the conceptual signal has to be independent of the one it is fused with."""
    parameters = set(inspect.signature(ConceptualRetriever.__init__).parameters)

    assert parameters == {
        "self",
        "matrix",
        "dictionary",
        "backend",
        "arm",
        "damping",
        "concept_weights",
    }

    retriever = a_retriever()
    assert not hasattr(retriever, "vectors")
    assert not hasattr(retriever, "index")


def test_a_concept_the_query_does_not_activate_contributes_nothing():
    """`absent` shares no concept with the query, so it scores exactly zero."""
    scores = dict(a_retriever().retrieve(FIRST_CONCEPT, top_k=3))

    assert scores["absent"] == 0.0


# --- Determinism ---------------------------------------------------------------


def test_top_k_limits_the_number_of_hits():
    assert len(a_retriever().retrieve(FIRST_CONCEPT, top_k=2)) == 2


def test_asking_for_more_than_the_pool_returns_the_whole_pool():
    assert len(a_retriever().retrieve(FIRST_CONCEPT, top_k=50)) == 3


def test_ties_are_broken_by_unit_id_not_by_matrix_order():
    dictionary = a_dictionary()
    matrix = a_matrix(
        rows=[[1.0, 0.0, 0.0, 0.0]] * 3,
        unit_ids=["zulu", "alpha", "mike"],
        key=dictionary.key,
    )
    retriever = ConceptualRetriever(matrix, dictionary, DirectionBackend(), arm="projection_full")

    hits = retriever.retrieve(FIRST_CONCEPT, top_k=3)

    assert [unit_id for unit_id, _ in hits] == ["alpha", "mike", "zulu"]


def test_a_truncated_ranking_over_ties_is_still_the_same_one_every_run():
    """Which tied units make the cut is `argpartition`'s answer, as in `dense.py`.

    Under a total tie no score can choose between the units, so the cut is whatever
    the partition leaves - deterministic for a given input, and the same rule the
    baselines already use, which is what keeps the systems comparable. What the
    contract does promise is that the answer never moves between runs and that the
    hits returned are ordered by unit id.
    """
    dictionary = a_dictionary()
    unit_ids = ["e", "d", "c", "b", "a"]
    rows = [[1.0, 0.0, 0.0, 0.0]] * 5

    first = ConceptualRetriever(
        a_matrix(rows, unit_ids, dictionary.key),
        dictionary,
        DirectionBackend(),
        arm="projection_full",
    )
    second = ConceptualRetriever(
        a_matrix(rows, unit_ids, dictionary.key),
        dictionary,
        DirectionBackend(),
        arm="projection_full",
    )

    hits = first.retrieve(FIRST_CONCEPT, top_k=2)
    assert hits == first.retrieve(FIRST_CONCEPT, top_k=2)
    assert hits == second.retrieve(FIRST_CONCEPT, top_k=2)
    assert [unit_id for unit_id, _ in hits] == sorted(unit_id for unit_id, _ in hits)


# --- What every result records -------------------------------------------------


def test_the_recorded_configuration_is_the_spec_s_data_contract():
    described = a_retriever().describe()

    assert set(described) == {
        "name",
        "dictionary_key",
        "k",
        "view",
        "query_operator",
        "query_alpha",
        "query_top_m",
        "damping",
    }
    assert described["name"] == "conceptual"
    assert described["k"] == 4
    assert described["view"] == "raw"
    assert described["damping"] == "none"


def test_projection_records_no_coding_penalty_and_the_truncation_of_its_arm():
    full = a_retriever(arm="projection_full").describe()
    truncated = a_retriever(arm="projection_top16").describe()

    assert full["query_operator"] == truncated["query_operator"] == PROJECTION
    assert full["query_alpha"] is None and truncated["query_alpha"] is None
    assert full["query_top_m"] is None
    assert truncated["query_top_m"] == config.QUERY_TOP_M


def test_the_query_alpha_is_the_corpus_alpha_rather_than_a_knob_of_its_own():
    """Coding the query at a different penalty compares two different resolutions."""
    matrix, _ = a_space()
    described = a_retriever(arm="sparse_coding").describe()

    assert described["query_operator"] == SPARSE_CODING
    assert described["query_alpha"] == matrix.coding_alpha


def test_the_view_recorded_is_the_one_the_matrix_was_loaded_as():
    raw, dictionary = a_space()
    normalized = ConceptualRetriever(
        row_normalized(raw), dictionary, DirectionBackend(), arm="projection_full"
    )

    assert normalized.describe()["view"] == "row_normalized"


# --- Refusals ------------------------------------------------------------------


def test_a_matrix_coded_against_a_different_dictionary_is_refused():
    matrix, dictionary = a_space()
    other = a_matrix(rows=[[1.0, 0.0, 0.0, 0.0]], unit_ids=["u"], key="not-this-dictionary")

    with pytest.raises(ValueError, match="dictionary"):
        ConceptualRetriever(other, dictionary, DirectionBackend(), arm="projection_full")
    assert matrix.dictionary_key == dictionary.key


def test_a_matrix_of_the_wrong_width_is_refused_rather_than_broadcast():
    dictionary = a_dictionary()
    narrow = a_matrix(rows=[[1.0, 0.0]], unit_ids=["u"], key=dictionary.key)

    with pytest.raises(ValueError, match="concepts"):
        ConceptualRetriever(narrow, dictionary, DirectionBackend(), arm="projection_full")


def test_an_unknown_arm_is_refused_at_construction_not_at_the_first_query():
    with pytest.raises(ValueError, match="unknown query operator"):
        a_retriever(arm="projection_top8")


def test_an_unknown_view_is_refused():
    dictionary = a_dictionary()
    matrix, _ = a_space()
    mislabelled = ConceptMatrix(
        X=matrix.X,
        unit_ids=matrix.unit_ids,
        dictionary_key=dictionary.key,
        view="l2_normalized",
        coding_alpha=matrix.coding_alpha,
        mean_active_per_unit=matrix.mean_active_per_unit,
        reconstruction_error=matrix.reconstruction_error,
    )

    with pytest.raises(ValueError, match="view"):
        ConceptualRetriever(mislabelled, dictionary, DirectionBackend(), arm="projection_full")


# --- Damping travels with the retriever, its shape arrives in T7 ----------------


def test_the_undamped_default_is_declared_rather_than_implied():
    retriever = a_retriever()

    assert retriever.damping == "none"
    assert retriever.concept_weights is None


def test_a_named_damping_without_its_weights_is_refused():
    """A retriever that records a damping it did not apply is a mislabelled result."""
    with pytest.raises(ValueError, match="damping"):
        a_retriever(damping="idf")


def test_weights_without_a_name_for_them_are_refused_too():
    with pytest.raises(ValueError, match="damping"):
        a_retriever(concept_weights=np.ones(4, dtype=np.float32))


def test_an_unknown_damping_mode_is_refused():
    with pytest.raises(ValueError, match="damping"):
        a_retriever(damping="sqrt", concept_weights=np.ones(4, dtype=np.float32))


def test_weights_of_the_wrong_length_are_refused():
    with pytest.raises(ValueError, match="concept"):
        a_retriever(damping="idf", concept_weights=np.ones(9, dtype=np.float32))


def test_weights_multiply_the_query_and_can_reorder_the_ranking():
    """Demoting the queried concept is enough to move a unit that leans on it."""
    matrix, dictionary = a_space()
    weights = np.array([0.1, 1.0, 1.0, 1.0], dtype=np.float32)
    damped = ConceptualRetriever(
        matrix,
        dictionary,
        DirectionBackend(),
        arm="projection_full",
        damping="idf",
        concept_weights=weights,
    )

    hits = damped.retrieve(FIRST_CONCEPT, top_k=3)

    assert dict(hits)["broad"] == pytest.approx(0.2)
    assert damped.describe()["damping"] == "idf"


def test_damping_leaves_the_matrix_and_the_atoms_untouched():
    """Decision D7: the weight is applied to the query, not baked into the space."""
    matrix, dictionary = a_space()
    before_x = matrix.X.data.tobytes()
    before_atoms = dictionary.atoms.tobytes()

    damped = ConceptualRetriever(
        matrix,
        dictionary,
        DirectionBackend(),
        arm="projection_full",
        damping="idf",
        concept_weights=np.array([0.1, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    damped.retrieve(FIRST_CONCEPT, top_k=3)

    assert matrix.X.data.tobytes() == before_x
    assert dictionary.atoms.tobytes() == before_atoms
