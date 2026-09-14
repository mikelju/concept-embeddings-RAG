"""Which questions a system answers, and where two systems disagree.

Phase 1 ended on a table that is the reason this project continues: at 2,048 tokens
on dev, dense and BM25 both answer 393 of the 600 questions, dense alone answers 94,
BM25 alone 41, and **72 defeat both**. Those 72 all look the same - the retriever
finds the entity the question names and never reaches the bridge entity it does not
name - and a one-hop retriever has no mechanism for the second hop, whatever space it
scores in.

That table was computed by hand once. This module regenerates it, so that System B's
can be read against it rather than beside it, and so that Phase 4 inherits a number
rather than an anecdote.

Success here is **Full Support**: every gold paragraph of the question inside the
budget. Not gold recall, which gives half a point for half the evidence - a HotpotQA
question needs both of its paragraphs and half of them answers nothing. Phase 1's own
table is in those terms, and (393 + 94) / 600 = 81.2% is exactly the Full Support the
dense baseline recorded, which is what says the two are the same measurement.

This module reads gold annotations, so it belongs on the measurer's side of the line
that `tests/concepts/test_labels_not_in_retrieval_path.py` draws: it lives in
`evaluation/`, beside the harness, and nothing in `retrieval/` may import it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.budget import fill_context
from concept_embeddings_rag.evaluation.metrics import full_support
from concept_embeddings_rag.retrieval.base import Retriever


class FailureAnalysisError(Exception):
    """Two systems were compared over question sets that are not the same."""


@dataclass(frozen=True)
class CrossTab:
    """The 2x2 of two systems over one question set, with the questions named.

    The qids are the point. A count says how many questions a system lost; the list
    says which, and only the list lets the next phase ask whether the ones it
    recovered are the ones that mattered - or check three of them by hand, as Phase 1
    did before concluding anything about what the failures have in common.
    """

    a: str
    b: str
    budget: int
    both: tuple[str, ...]
    only_a: tuple[str, ...]
    only_b: tuple[str, ...]
    neither: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise FailureAnalysisError(
                f"a cross-tabulation compares two systems, and both sides say {self.a!r}"
            )
        cells = [self.both, self.only_a, self.only_b, self.neither]
        seen = [qid for cell in cells for qid in cell]
        if len(seen) != len(set(seen)):
            raise FailureAnalysisError(
                "a question falls in exactly one cell of the table, and one of these "
                "appears in more than one"
            )

    @property
    def counts(self) -> dict[str, int]:
        """The four cells as Phase 1 reported them, in that order."""
        return {
            "both": len(self.both),
            f"only_{self.a}": len(self.only_a),
            f"only_{self.b}": len(self.only_b),
            "neither": len(self.neither),
        }

    @property
    def n_questions(self) -> int:
        return sum(self.counts.values())


def per_question_success(
    retriever: Retriever,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    budget: int,
    *,
    top_k: int,
) -> dict[str, bool]:
    """Whether each question is fully supported within `budget`, keyed by qid.

    The retrieval and budget filling are the harness's, unchanged: the same ranking,
    the same `top_k`, the same rule for what fits. A question is answered when every
    one of its gold paragraphs is in the context, and the budget is what decides how
    much context there is.
    """
    if budget <= 0:
        raise FailureAnalysisError(f"a budget is a token count, not {budget}")
    if top_k <= 0:
        raise FailureAnalysisError(f"top_k must be positive, not {top_k}")

    success: dict[str, bool] = {}
    for question in questions:
        if question.qid in success:
            raise FailureAnalysisError(
                f"question {question.qid} appears twice; one question is one row of the table"
            )
        hits = retriever.retrieve(question.question, top_k=top_k)
        context = fill_context([unit_id for unit_id, _score in hits], token_counts, budget)
        success[question.qid] = bool(full_support(context, question.gold_unit_ids))
    return success


def cross_tabulate(
    a: Mapping[str, bool],
    b: Mapping[str, bool],
    *,
    names: tuple[str, str],
    budget: int,
) -> CrossTab:
    """Phase 1's four outcomes over two systems measured on the same questions.

    The two mappings must cover exactly the same qids. Comparing a system measured on
    600 questions with one measured on 598 would still produce a table, and every cell
    of it would be wrong by an amount nobody could see.
    """
    if set(a) != set(b):
        only_in_a = sorted(set(a) - set(b))
        only_in_b = sorted(set(b) - set(a))
        raise FailureAnalysisError(
            "the two systems were measured over different question sets: "
            f"{len(only_in_a)} only in {names[0]}, {len(only_in_b)} only in {names[1]}"
        )

    ordered = sorted(a)
    return CrossTab(
        a=names[0],
        b=names[1],
        budget=budget,
        both=tuple(qid for qid in ordered if a[qid] and b[qid]),
        only_a=tuple(qid for qid in ordered if a[qid] and not b[qid]),
        only_b=tuple(qid for qid in ordered if b[qid] and not a[qid]),
        neither=tuple(qid for qid in ordered if not a[qid] and not b[qid]),
    )
