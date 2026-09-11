"""T18: the control is fitted by the same procedure as System B, or it is not a control.

HU-5 exists because "the hybrid beats dense" is a generic property of fusing two
signals: Phase 1 measured BM25 rescuing 6.8% of the dev questions dense loses, so the
headroom a hybrid exploits is available to any second signal. The dense+BM25 control
is what makes System B's number falsifiable - and only while the two are built and
fitted identically. Giving the control a shorter grid, a different scheme, or a
fitting that saw less of the dev split is the single easiest way to manufacture this
phase's headline, and it would not look like cheating in a diff.

So parity is asserted two ways. At runtime: one builder, one fitting function, and a
second signal that changes nothing else about how the hybrid is made. Structurally:
the two call sites in the CLI are compared keyword by keyword, and the check is shown
to fail on a source where one of them was given its own grid - because a parity test
that cannot fail proves nothing about the parity it claims to check.
"""

import ast
from pathlib import Path

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.selection import DEV_SPLIT, fit_fusion_weight
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.fusion import MIN_MAX, FusedRetriever

CLI = Path(__file__).resolve().parents[2] / "src" / "concept_embeddings_rag" / "cli.py"

TOKENS = {"g1": 10, "g2": 10, "x": 10}
ONE_UNIT = 10

DENSE_HITS: list[Hit] = [("g1", 1.0), ("x", 0.5), ("g2", 0.1)]
SECOND_HITS: list[Hit] = [("x", 9.0), ("g2", 5.0), ("g1", 1.0)]


class StubRetriever:
    def __init__(self, name: str, hits: list[Hit]) -> None:
        self.name = name
        self.hits = hits

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return self.hits[:top_k]


def a_question() -> Question:
    return Question(
        qid="q1",
        question="a question",
        answer="-",
        gold_unit_ids=("g1", "g2"),
        supporting_facts=(("g1", 0),),
        split=DEV_SPLIT,
    )


def a_config() -> dict:
    return {
        "model": "fake",
        "revision": "v0",
        "unit_set_hash": "poolhash0001",
        "seed": 42,
        "tokenizer": "fake-tokenizer",
        "code_version": "0.1.0",
        "top_k": 3,
    }


def a_fit(second_signal: str):
    return fit_fusion_weight(
        [StubRetriever("dense", DENSE_HITS), StubRetriever(second_signal, SECOND_HITS)],
        questions=[a_question()],
        token_counts=TOKENS,
        base_config=a_config(),
        budgets=(ONE_UNIT,),
        ks=(2,),
        budget=ONE_UNIT,
    )


# --- Same procedure at runtime ------------------------------------------------


def test_the_control_and_system_b_are_fitted_over_the_same_grid():
    assert a_fit("conceptual").grid == a_fit("bm25").grid == config.FUSION_WEIGHT_GRID


def test_a_second_signal_that_ranks_the_same_is_fitted_to_the_same_curve():
    """Only the name differs, so only the name may differ in the result."""
    system_b, control = a_fit("conceptual"), a_fit("bm25")

    assert control.curve == system_b.curve
    assert control.rrf_score == system_b.rrf_score
    assert (control.best_weight, control.winning_scheme) == (
        system_b.best_weight,
        system_b.winning_scheme,
    )
    assert control.components[1] != system_b.components[1]


def test_both_hybrids_come_out_of_the_same_constructor_with_the_same_normalization():
    weights = {"dense": 0.4, "conceptual": 0.6}
    system_b = FusedRetriever(
        [StubRetriever("dense", DENSE_HITS), StubRetriever("conceptual", SECOND_HITS)],
        scheme="weighted",
        weights=weights,
    )
    control = FusedRetriever(
        [StubRetriever("dense", DENSE_HITS), StubRetriever("bm25", SECOND_HITS)],
        scheme="weighted",
        weights={"dense": 0.4, "bm25": 0.6},
    )

    assert type(system_b) is type(control)
    assert system_b.normalization == control.normalization == MIN_MAX
    assert system_b.components[0] == control.components[0] == "dense"
    assert system_b.fitted_on == control.fitted_on


def test_the_same_ranking_fused_the_same_way_gives_the_same_ranking():
    """The control's hits are System B's here, so any difference would be procedural."""
    system_b = FusedRetriever(
        [StubRetriever("dense", DENSE_HITS), StubRetriever("conceptual", SECOND_HITS)],
        scheme="rrf",
    )
    control = FusedRetriever(
        [StubRetriever("dense", DENSE_HITS), StubRetriever("bm25", SECOND_HITS)],
        scheme="rrf",
    )

    assert system_b.retrieve("a query", top_k=3) == control.retrieve("a query", top_k=3)


# --- Same procedure in the source that runs it --------------------------------


def fitting_calls(source: str) -> list[set[str]]:
    """The keyword arguments of every `fit_fusion_weight` call, in source order."""
    return [
        {keyword.arg or "**" for keyword in node.keywords}
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fit_fusion_weight"
    ]


def hybrid_constructions(source: str) -> int:
    return sum(
        1
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FusedRetriever"
    )


def cli_source() -> str:
    return CLI.read_text(encoding="utf-8")


def test_the_stage_fits_exactly_twice_and_passes_both_the_same_arguments():
    """System B and the control differ in their components and in nothing else."""
    calls = fitting_calls(cli_source())

    assert len(calls) == 2, "the selection stage fits System B and the control, and nothing else"
    assert calls[0] == calls[1]


def test_the_check_fails_on_a_control_given_a_grid_of_its_own():
    """The negative proof: a parity check that cannot fail checks nothing."""
    control_call = "    control = fit_fusion_weight(\n        [dense, BM25Retriever(units)],"
    tampered = cli_source().replace(control_call, control_call + "\n        grid=(0.5,),", 1)
    calls = fitting_calls(tampered)

    assert len(calls) == 2
    assert calls[0] != calls[1], "the tampered source was not actually tampered with"


def test_one_constructor_serves_both_hybrids():
    """Two construction sites could drift apart; one cannot."""
    assert hybrid_constructions(cli_source()) == 1


def identifiers_in(source: str) -> set[str]:
    """Every name the code touches, docstrings and comments excluded.

    Read off the syntax tree rather than searched for in the text, for the reason
    the labels guard uses the same trick: `base.py` says in its own docstring that
    the conceptual retrievers will implement it, and a substring check would fail on
    the sentence promising exactly what this test is checking.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).rsplit(".", maxsplit=1)[-1])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.update(node.module.split("."))
    return names


@pytest.mark.parametrize("module", ["evaluation/harness.py", "retrieval/base.py"])
def test_the_measurer_and_the_interface_know_nothing_about_this_phase(module: str):
    """This phase adds config keys to a result, not code to the harness."""
    touched = identifiers_in((CLI.parent / module).read_text(encoding="utf-8"))

    for name in ("conceptual", "fusion", "selection", "dictionary_key", "FusedRetriever"):
        assert name not in touched, f"{module} reaches for {name!r}; it is supposed to be unchanged"
