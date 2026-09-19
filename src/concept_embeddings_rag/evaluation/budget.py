"""Filling the context up to a fixed token budget.

This is what makes the comparison fair. The iterative system retrieves more
chunks than the dense one by design, so comparing them at the same K would be
meaningless; comparing them at the same number of tokens sent to the LLM is not.

The rule is deliberately dull and identical for every system: walk the ranking,
include a unit if it fits whole, skip it if it does not, and carry on.
"""

from collections.abc import Mapping, Sequence

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import IndexingUnit


def fill_context(
    ranked_unit_ids: Sequence[str],
    token_counts: Mapping[str, int],
    budget: int,
) -> list[str]:
    """Return the units that fit within `budget`, in rank order.

    Raises KeyError if a ranked unit has no known token count: silently treating
    it as free would let a system smuggle text past the budget.
    """
    selected: list[str] = []
    used = 0

    for unit_id in ranked_unit_ids:
        cost = token_counts[unit_id]
        if used + cost <= budget:
            selected.append(unit_id)
            used += cost

    return selected


def total_tokens(unit_ids: Sequence[str], token_counts: Mapping[str, int]) -> int:
    return sum(token_counts[unit_id] for unit_id in unit_ids)


class TokenCounter:
    """Counts tokens with a single declared tokenizer, loaded on first use.

    Which tokenizer it is matters far less than the fact that it never changes
    between systems: the budget is a shared ruler, not an absolute truth.

    The defaults name the budget tokenizer rather than the embedding model (Phase 8,
    restriction R1). They hold exactly the values the old defaults resolved to, so no
    recorded count moves; what changes is that repointing the Dense retriever can no
    longer move the ruler every inherited figure was measured with.
    """

    def __init__(
        self,
        tokenizer_id: str = config.BUDGET_TOKENIZER_ID,
        revision: str = config.BUDGET_TOKENIZER_REVISION,
    ) -> None:
        self.tokenizer_id = tokenizer_id
        self.revision = revision
        self._tokenizer = None

    def _load(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            print(f"[INFO] loading tokenizer {self.tokenizer_id}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id, revision=self.revision
            )
        return self._tokenizer

    def count(self, text: str) -> int:
        tokenizer = self._load()
        return len(tokenizer.encode(text, add_special_tokens=False))

    def count_units(self, units: Sequence[IndexingUnit]) -> dict[str, int]:
        """Token count per unit id, over the same text the retrievers index."""
        tokenizer = self._load()
        texts = [unit.indexable_text for unit in units]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return {unit.unit_id: len(ids) for unit, ids in zip(units, encoded, strict=True)}
