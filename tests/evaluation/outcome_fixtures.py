"""Per-question outcome artifacts written to disk from chosen Full Support vectors (T12, T15).

A toy test population whose outcomes at every budget are fixed by construction: each unit costs
1,000 tokens, so two units fit at 2,048 and the first two ranked units decide Full Support. A
system that should support a question ranks its two gold paragraphs first; one that should not
puts a distractor between them. The records go through the unchanged harness, the outcomes
builder and the writer, and come back through the verifying loader, exactly as the test stage's
outcomes will.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.harness import QuestionOutcome, evaluate_retriever
from concept_embeddings_rag.evaluation.outcomes import (
    PerQuestionOutcomes,
    build_outcomes,
    load_outcomes,
    save_outcomes,
)

FREEZE_DIGEST = "f" * 64
BUDGETS = (512, 1024, 2048, 4096)
KS = (2, 5, 10)
SYSTEMS = ("dense", "hybrid-bm25", "hybrid-entity-hop")


def qids_for(n: int) -> list[str]:
    # Pool order is deliberately not ascending question id.
    return [f"{(7919 * index) % 100003:024x}" for index in range(n)]


class VectorRetriever:
    def __init__(self, name: str, supported: Mapping[str, bool]) -> None:
        self.name = name
        self.supported = supported

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        qid = query.removeprefix("question ")
        first, second = f"{qid}-a", f"{qid}-b"
        ranking = [first, second, "x"] if self.supported[qid] else [first, "x", second]
        return [(unit, 1.0 / (rank + 1)) for rank, unit in enumerate(ranking)][:top_k]


def a_config() -> dict:
    return {
        "model": "toy-model",
        "revision": "rev",
        "unit_set_hash": "pool",
        "seed": 42,
        "tokenizer": "toy-tokenizer",
        "code_version": "0.1.0",
        "top_k": 100,
    }


def write_outcomes(
    directory: Path,
    vectors: Mapping[str, Sequence[int]],
    *,
    split: str = "test",
    freeze_digest: str | None = FREEZE_DIGEST,
) -> tuple[dict[str, PerQuestionOutcomes], list[str]]:
    """One outcomes artifact per system, from 0/1 Full Support vectors in pool order."""
    n = len(next(iter(vectors.values())))
    qids = qids_for(n)
    questions = [
        Question(qid, f"question {qid}", "-", (f"{qid}-a", f"{qid}-b"), (("t", 0),), split)
        for qid in qids
    ]
    tokens = {unit: 1000 for qid in qids for unit in (f"{qid}-a", f"{qid}-b")}
    tokens["x"] = 1000
    loaded: dict[str, PerQuestionOutcomes] = {}
    for system, vector in vectors.items():
        supported = {qid: bool(value) for qid, value in zip(qids, vector, strict=True)}
        records: list[QuestionOutcome] = []
        result = evaluate_retriever(
            VectorRetriever(system, supported),
            questions=questions,
            token_counts=tokens,
            budgets=BUDGETS,
            ks=KS,
            top_k=100,
            config=a_config(),
            split=split,
            outcomes=records,
        )
        run_path = result.save(directory)
        payload = build_outcomes(
            result,
            records,
            run_file=run_path.name,
            split_order=qids,
            freeze_digest=freeze_digest,
        )
        path = save_outcomes(payload, directory)
        loaded[system] = load_outcomes(path, split_order=qids, run_dir=directory)
    return loaded, qids


def vectors_from_counts(counts: Mapping[tuple[int, int, int], int]) -> dict[str, list[int]]:
    """Full Support vectors from counts of (dense, A, B) patterns."""
    dense: list[int] = []
    control: list[int] = []
    entity: list[int] = []
    for (d, a, b), k in sorted(counts.items()):
        dense.extend([d] * k)
        control.extend([a] * k)
        entity.extend([b] * k)
    return {"dense": dense, "hybrid-bm25": control, "hybrid-entity-hop": entity}
