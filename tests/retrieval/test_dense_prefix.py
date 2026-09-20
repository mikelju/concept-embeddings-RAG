"""Deviation 8.1, S4: exact retrieval over a prefix of the frozen scale corpus.

Nothing here is a new retrieval method. The deviation fixes the semantics as *exactly*
those of the existing `DenseRetriever` - corpus vectors L2-normalized upstream, query
vector L2-normalized, score = dot product = cosine, `EVALUATION_TOP_K = 100`, tie-break by
unit id - and the only new thing 8.1 needs is to point that retriever at the first `n` rows
of one frozen ordering and to give each (model, scale) run a distinguishable name.

So these tests assert the two properties a prefix could silently break:

- **scores are the dot products** they claim to be, and the ranking is the prefix's own
  units ordered by `(-score, unit_id)`;
- **rows stay aligned with ids** at every prefix size, because row `i` meaning a different
  paragraph at C250 than at C100 would corrupt every figure without raising anything.

And one the deviation forbids outright: no approximate index may appear on this path.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import scale_sensitivity
from concept_embeddings_rag.retrieval.dense import DenseRetriever


class StubBackend:
    """A query encoder that returns a fixed vector. No model, no network."""

    name = "stub"
    revision = "stub"
    normalize = True

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector

    def encode(self, texts):
        return np.array([self.vector for _ in texts], dtype=np.float32)


def corpus(n: int, dim: int = 4) -> tuple[np.ndarray, list[str]]:
    """`n` L2-normalized rows and ids that sort in the opposite order to their scores."""
    rows = []
    for i in range(n):
        row = np.zeros(dim, dtype=np.float32)
        row[i % dim] = 1.0
        row[(i + 1) % dim] = float(n - i) / float(n)
        rows.append(row / np.linalg.norm(row))
    vectors = np.vstack(rows)
    unit_ids = [f"unit{i:04d}" for i in range(n)]
    return vectors, unit_ids


def test_prefix_scores_are_the_dot_products() -> None:
    vectors, unit_ids = corpus(12)
    query = vectors[3].copy()
    retriever = scale_sensitivity.prefix_retriever(
        vectors, unit_ids, size=8, backend=StubBackend(query), name="dense-bge-c19"
    )

    hits = retriever.retrieve("anything", top_k=8)

    expected = {unit_ids[i]: float(vectors[i] @ query) for i in range(8)}
    for unit_id, score in hits:
        assert score == pytest.approx(expected[unit_id])
    # Ranking is by descending score, then unit id - the project's deterministic rule.
    assert [unit_id for unit_id, _ in hits] == sorted(
        expected, key=lambda unit_id: (-expected[unit_id], unit_id)
    )


def test_prefix_only_sees_the_prefix() -> None:
    vectors, unit_ids = corpus(12)
    retriever = scale_sensitivity.prefix_retriever(
        vectors, unit_ids, size=5, backend=StubBackend(vectors[0].copy()), name="dense-bge-c19"
    )

    hits = retriever.retrieve("anything", top_k=12)

    assert len(hits) == 5
    assert {unit_id for unit_id, _ in hits} == set(unit_ids[:5])


def test_rows_stay_aligned_with_ids_at_every_prefix() -> None:
    """Row `i` is `unit_ids[i]` at every size, or a scale figure would measure noise."""
    vectors, unit_ids = corpus(20)

    for size in (4, 9, 20):
        retriever = scale_sensitivity.prefix_retriever(
            vectors, unit_ids, size=size, backend=StubBackend(vectors[0].copy()), name="dense-x-c19"
        )
        inner = retriever.inner
        assert inner.unit_ids == unit_ids[:size]
        assert np.array_equal(inner.vectors, vectors[:size])


def test_prefix_refuses_a_size_the_cache_cannot_serve() -> None:
    vectors, unit_ids = corpus(6)

    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="6"):
        scale_sensitivity.prefix_retriever(
            vectors, unit_ids, size=7, backend=StubBackend(vectors[0].copy()), name="dense-x-c19"
        )


def test_named_wrapper_is_harness_safe_and_delegates() -> None:
    """Each (model, scale) run needs its own valid name, and nothing else changes."""
    vectors, unit_ids = corpus(6)
    backend = StubBackend(vectors[0].copy())
    wrapped = scale_sensitivity.prefix_retriever(
        vectors, unit_ids, size=6, backend=backend, name="dense-qwen-c500"
    )
    plain = DenseRetriever(vectors, unit_ids, backend)

    assert wrapped.name == "dense-qwen-c500"
    assert scale_sensitivity.run_name("qwen", 500000) == "dense-qwen-c500"
    assert wrapped.retrieve("q", top_k=6) == plain.retrieve("q", top_k=6)


def test_no_approximate_index_is_imported() -> None:
    """The deviation excludes ANN / HNSW / IVF / quantized retrieval outright.

    Checked over the module's import statements rather than its text, so the docstring
    stays free to name what is forbidden.
    """
    tree = ast.parse(Path(scale_sensitivity.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for forbidden in ("faiss", "hnswlib", "annoy", "scann", "pynndescent", "nmslib"):
        assert not any(name.startswith(forbidden) for name in imported), forbidden
    # And the one retriever it does use is the project's own exact one.
    assert "concept_embeddings_rag.retrieval.dense" in imported


def test_run_names_cover_the_declared_sizes() -> None:
    for label, size in zip(
        config.PHASE_8_1_CORPUS_LABELS, config.PHASE_8_1_CORPUS_SIZES, strict=True
    ):
        assert scale_sensitivity.run_name("bge", size) == f"dense-bge-{label}"
