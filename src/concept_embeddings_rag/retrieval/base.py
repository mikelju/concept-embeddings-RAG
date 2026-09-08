"""The retriever interface.

Every system in the experiment - dense, BM25, and later the conceptual ones -
exposes exactly this. The evaluation harness must not be able to tell them apart:
that is what makes their numbers comparable.
"""

from typing import Protocol, runtime_checkable

Hit = tuple[str, float]


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        """Return up to `top_k` (unit_id, score) pairs, best first.

        Ranking must be deterministic, ties included: the same query returns the
        same list on every run.
        """
        ...
