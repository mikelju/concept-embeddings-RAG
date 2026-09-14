"""T7: the three things the diffusion must never do, stated rather than discovered.

Each of these is an invariant of the spec's data contract, and each is the kind of
property that holds silently until the day it does not. A comment saying "we never
build the big matrix" is worth nothing the first time an intermediate is written the
other way round, so the first one is measured rather than asserted.
"""

import tracemalloc

import numpy as np
import pytest
from diffusion_fixtures import INHERITED, SeedStub, a_dictionary, a_matrix
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.retrieval.diffusion import NONE, DiffusionRetriever

# Big enough that the forbidden matrix is unmistakable: 2,000 units squared is four
# million float64 entries, which is 32 MB. Small enough that the test stays fast.
N_UNITS = 2_000
N_CONCEPTS = 50
FORBIDDEN_BYTES = N_UNITS * N_UNITS * 8


def a_wide_corpus(seed: int = 42) -> tuple[sparse.csr_matrix, list[str]]:
    """A sparse corpus at a scale where the unit x unit product would be visible."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(N_UNITS), 8)
    columns = rng.integers(0, N_CONCEPTS, size=rows.size)
    values = rng.random(rows.size) + 0.1
    X = sparse.csr_matrix((values, (rows, columns)), shape=(N_UNITS, N_CONCEPTS))
    X.sum_duplicates()
    return X, [f"u{row:05d}" for row in range(N_UNITS)]


def test_the_unit_by_unit_matrix_is_never_allocated():
    """Decision D2, measured: at this scale forming it would cost 32 MB in one object."""
    X, unit_ids = a_wide_corpus()
    dictionary = a_dictionary(k=N_CONCEPTS, dim=N_CONCEPTS)
    retriever = DiffusionRetriever(
        matrix=a_matrix(dictionary.key, rows=X.toarray().tolist(), unit_ids=unit_ids),
        dictionary=dictionary,
        seed_retriever=SeedStub([(unit_ids[0], 0.9), (unit_ids[7], 0.4)]),
        seed_arm="dense",
        restart=0.4,
        normalization=NONE,
        max_iterations=config.MAX_ITERATIONS,
        inherits=INHERITED,
    )

    tracemalloc.start()
    try:
        retriever.retrieve("anything", top_k=100)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < FORBIDDEN_BYTES / 8, (
        f"the walk peaked at {peak / 1e6:.1f} MB; the unit x unit matrix alone would be "
        f"{FORBIDDEN_BYTES / 1e6:.0f} MB, so something is being materialized that must not be"
    )


def test_the_operator_multiplies_nothing_of_unit_by_unit_shape():
    """The static half of the same rule: both operands keep the shape of `X`."""
    X, unit_ids = a_wide_corpus()
    dictionary = a_dictionary(k=N_CONCEPTS, dim=N_CONCEPTS)
    retriever = DiffusionRetriever(
        matrix=a_matrix(dictionary.key, rows=X.toarray().tolist(), unit_ids=unit_ids),
        dictionary=dictionary,
        seed_retriever=SeedStub([(unit_ids[0], 0.9)]),
        seed_arm="dense",
        restart=0.4,
        normalization=NONE,
        inherits=INHERITED,
    )

    for operand in (retriever.operator.left, retriever.operator.right):
        assert operand.shape == (N_UNITS, N_CONCEPTS)
        assert sparse.issparse(operand)


def test_a_unit_no_concept_reaches_gains_no_mass_from_the_walk():
    """The spec's invariant. Phase 2 measured zero orphan units, so it should hold."""
    dictionary = a_dictionary(k=2, dim=2)
    with_an_orphan = a_matrix(
        dictionary.key,
        rows=[[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 2.0]],
        unit_ids=["u0", "u1", "orphan", "u3"],
    )
    retriever = DiffusionRetriever(
        matrix=with_an_orphan,
        dictionary=dictionary,
        seed_retriever=SeedStub([("u0", 0.9)]),
        seed_arm="dense",
        restart=0.4,
        normalization=NONE,
        inherits=INHERITED,
    )

    hits = retriever.retrieve("anything", top_k=4)

    assert "orphan" not in [unit_id for unit_id, _score in hits]


def test_an_orphan_the_seed_itself_returned_still_gains_nothing_from_the_walk():
    """It keeps the restart's share of its own seed mass, and not one unit of anyone else's."""
    dictionary = a_dictionary(k=2, dim=2)
    with_an_orphan = a_matrix(
        dictionary.key,
        rows=[[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 2.0]],
        unit_ids=["u0", "u1", "orphan", "u3"],
    )
    retriever = DiffusionRetriever(
        matrix=with_an_orphan,
        dictionary=dictionary,
        seed_retriever=SeedStub([("u0", 0.5), ("orphan", 0.5)]),
        seed_arm="dense",
        restart=0.4,
        normalization=NONE,
        max_iterations=1,
        inherits=INHERITED,
    )

    hits = dict(retriever.retrieve("anything", top_k=4))

    # One round: restart * 0.5 of its own seed mass, and nothing propagated to it.
    assert hits["orphan"] == pytest.approx(0.4 * 0.5)


def test_the_arms_that_divide_by_a_degree_refuse_the_orphan_instead():
    """Two ways of honouring the same invariant, and the arm decides which applies."""
    from concept_embeddings_rag.retrieval.diffusion import DiffusionError, DiffusionOperator

    orphaned = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]]))

    for arm in ("symmetric", "stochastic"):
        with pytest.raises(DiffusionError):
            DiffusionOperator(orphaned, normalization=arm)

    assert DiffusionOperator(orphaned, normalization=NONE).n_units == 3


def test_the_retrieval_path_guard_covers_the_diffusion_module():
    """`retrieval/` is globbed by the labels guard, so this module is in it already."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "src" / "concept_embeddings_rag"
    covered = {path.name for path in (source / "retrieval").glob("*.py")}

    assert "diffusion.py" in covered


def test_the_diffusion_module_reads_no_label_no_gold_and_no_split():
    """The same three rules the rest of the retrieval path lives under, spelled out here."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "src" / "concept_embeddings_rag"
    text = (source / "retrieval" / "diffusion.py").read_text(encoding="utf-8")

    assert "labeling" not in text
    assert "labels-" not in text
    assert "gold_unit_ids" not in text
    assert "supporting_facts" not in text
