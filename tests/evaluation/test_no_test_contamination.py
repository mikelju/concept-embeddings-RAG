"""T15: the test split cannot be reached from anywhere on the selection path.

Every other guard in this phase protects a number. This one protects the whole
phase: a hyperparameter chosen, adjusted or merely confirmed against test voids the
comparison the project exists to make, and no amount of care afterwards puts it
back. HU-7's last criterion is unconditional for that reason, and unconditional
rules are the ones worth asserting rather than remembering.

Three things are checked, and they fail differently on purpose.

**Every door refuses.** The selection path has three public entrances - the sweep of
one space, the sweep of all of them, and the fusion fitting - plus the door a
measured result comes through to become a table cell. A question from any split but
dev is refused at each of them, including when it is one question hidden among dev
ones, which is the shape the accident would actually take.

**Nothing on the path can name another split.** Checked statically over the whole
import closure of `evaluation/selection.py`, not over a hand-written list, so a
module that joins the path later is covered the moment it is imported rather than
when someone remembers to add it. `DEV_SPLIT` is the only split name any of those
modules holds.

**The guard is load-bearing.** With it neutralised, the sweep measures a test
question and files the result under `dev` without a murmur - which is exactly the
failure that would be invisible, and exactly why the guard is where it is rather
than left to the caller.
"""

import ast
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.concepts.coding import ConceptMatrix
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import selection
from concept_embeddings_rag.evaluation.expansion_selection import ExpansionCell, sweep_expansion
from concept_embeddings_rag.evaluation.harness import RunResult
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    SelectionError,
    SweepCell,
    SweepSpace,
    fit_fusion_weight,
    run_dev_sweep,
    sweep_space,
)
from concept_embeddings_rag.retrieval.base import Hit

SOURCE = Path(__file__).resolve().parents[2] / "src" / "concept_embeddings_rag"
PACKAGE = "concept_embeddings_rag"

# The splits this phase must not be able to reach from the selection path. `train`
# is in the list although the project does not use one: a guard written only against
# the split that exists today would not catch the next one.
OTHER_SPLITS = ("test", "train")

UNITS = ["u1", "u2", "u3"]
TOKENS = dict.fromkeys(UNITS, 100)
K = 32


class DirectionBackend:
    """Encodes a query as the literal vector written in the query string."""

    name = "fake"
    revision = "v0"
    dim = 4
    normalize = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([[float(x) for x in t.split(",")] for t in texts], dtype=np.float32)


class StubRetriever:
    """A retriever with a fixed answer, for the fusion door."""

    def __init__(self, name: str, hits: list[Hit]) -> None:
        self.name = name
        self.hits = hits

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return self.hits[:top_k]


def a_dictionary() -> ConceptDictionary:
    return ConceptDictionary(
        atoms=np.eye(K, 4, dtype=np.float32),
        k=K,
        seed=42,
        sparsity_param=0.05,
        max_iter=3,
        unit_set_hash="101f564fdcca620c",
        model="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c",
    )


def a_matrix(dictionary: ConceptDictionary, view: str) -> ConceptMatrix:
    rows = np.zeros((3, K), dtype=np.float32)
    rows[0, 0] = 1.0
    rows[1, 1] = 2.0
    rows[2, K - 1] = 3.0
    if view == "row_normalized":
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        rows = rows / np.where(norms == 0.0, 1.0, norms)

    X = sparse.csr_matrix(rows)
    return ConceptMatrix(
        X=X,
        unit_ids=list(UNITS),
        dictionary_key=dictionary.key,
        view=view,
        coding_alpha=0.07,
        mean_active_per_unit=float(X.nnz) / X.shape[0],
        reconstruction_error=0.0,
    )


def a_space() -> SweepSpace:
    dictionary = a_dictionary()
    return SweepSpace(
        k=K,
        dictionary=dictionary,
        matrices={view: a_matrix(dictionary, view) for view in config.CONCEPT_VIEWS},
    )


def a_question(qid: str, vector: str, split: str) -> Question:
    return Question(
        qid=qid,
        question=vector,
        answer="-",
        gold_unit_ids=("u1", "u2"),
        supporting_facts=(("u1", 0),),
        split=split,
    )


def questions_on(split: str) -> list[Question]:
    return [a_question("q1", "1,0,0,0", split), a_question("q2", "0,1,0,0", split)]


def a_config(**overrides) -> dict:
    base = {
        "model": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c",
        "unit_set_hash": "101f564fdcca620c",
        "seed": 42,
        "tokenizer": "cl100k_base",
        "code_version": "0.1.0",
        "top_k": 3,
    }
    base.update(overrides)
    return base


# --- The doors ----------------------------------------------------------------


def the_whole_sweep(questions: Sequence[Question]) -> object:
    return run_dev_sweep(
        [a_space()],
        questions=questions,
        token_counts=TOKENS,
        backend=DirectionBackend(),
        base_config=a_config(),
        arms=("projection_top16",),
        views=("raw",),
    )


def one_space(questions: Sequence[Question]) -> object:
    return sweep_space(
        a_space(),
        questions=questions,
        token_counts=TOKENS,
        backend=DirectionBackend(),
        base_config=a_config(),
        arms=("projection_top16",),
        views=("raw",),
    )


def the_fusion_fitting(questions: Sequence[Question]) -> object:
    return fit_fusion_weight(
        [
            StubRetriever("dense", [("u1", 0.9), ("u2", 0.5)]),
            StubRetriever("conceptual", [("u2", 3.0), ("u3", 1.0)]),
        ],
        questions=questions,
        token_counts=TOKENS,
        base_config=a_config(),
        grid=(0.0, 1.0),
        budgets=(config.SELECTION_BUDGET,),
        ks=(2,),
    )


DOORS: dict[str, Callable[[Sequence[Question]], object]] = {
    "run_dev_sweep": the_whole_sweep,
    "sweep_space": one_space,
    "fit_fusion_weight": the_fusion_fitting,
}


# --- Every door refuses -------------------------------------------------------


def test_the_guard_is_not_vacuous_because_every_door_opens_for_dev():
    """A refusal proves nothing if the call would have failed on dev as well."""
    for name, door in DOORS.items():
        assert door(questions_on(DEV_SPLIT)) is not None, f"{name} did not run on dev"


@pytest.mark.parametrize("door", DOORS, ids=list(DOORS))
@pytest.mark.parametrize("split", OTHER_SPLITS)
def test_no_door_of_the_selection_path_accepts_a_question_from_another_split(door: str, split: str):
    with pytest.raises(SelectionError, match=split):
        DOORS[door](questions_on(split))


@pytest.mark.parametrize("door", DOORS, ids=list(DOORS))
def test_one_test_question_hidden_among_the_dev_ones_is_enough_to_refuse(door: str):
    """The shape the accident would actually take: a filter that let one through."""
    contaminated = [*questions_on(DEV_SPLIT), a_question("q9", "0,0,0,1", "test")]

    with pytest.raises(SelectionError, match="test"):
        DOORS[door](contaminated)


@pytest.mark.parametrize("door", DOORS, ids=list(DOORS))
def test_every_door_goes_through_the_same_doorman(door: str, monkeypatch):
    """One guard, called by each entrance, rather than three copies to keep in step."""
    seen: list[int] = []
    real = selection._check_questions

    def spy(questions):
        seen.append(len(questions))
        return real(questions)

    monkeypatch.setattr(selection, "_check_questions", spy)
    DOORS[door](questions_on(DEV_SPLIT))

    assert seen, f"{door} never asked the guard"


def test_a_result_measured_on_test_cannot_become_a_table_cell():
    """The last door: even a finished measurement is refused entry to the table."""
    result = RunResult(
        system="conceptual",
        split="test",
        config=a_config(
            dictionary_key="key", k=K, view="raw", query_operator="projection_top16", damping="none"
        ),
        metrics={f"budget_{config.SELECTION_BUDGET}": {config.SELECTION_METRIC: 0.9}},
        cost={"n_questions": 2},
    )

    with pytest.raises(SelectionError, match="test"):
        SweepCell.from_result(result)


# --- Nothing on the path can name another split -------------------------------


def import_closure(start: str) -> list[str]:
    """Every project module reachable from `start`, computed rather than listed.

    A hand-written list is a list someone has to remember to extend. This walks the
    imports, so a module that joins the selection path is covered by the guard from
    the moment it is imported by something already on it.
    """
    found: set[str] = set()
    pending = [start]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        tree = ast.parse((SOURCE / current).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            named: list[str] = []
            if isinstance(node, ast.Import):
                named = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                named = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
            for name in named:
                if not name.startswith(PACKAGE):
                    continue
                relative = name[len(PACKAGE) :].strip(".").replace(".", "/")
                candidate = SOURCE / f"{relative}.py"
                if candidate.exists():
                    pending.append(candidate.relative_to(SOURCE).as_posix())
    return sorted(found)


SELECTION_CLOSURE = import_closure("evaluation/selection.py")


def string_literals(source: str) -> set[str]:
    """Every string constant in the module, docstrings included.

    The guard is equality against a split name, not a substring search, exactly so
    that a docstring explaining the rule does not trip the rule it explains.
    """
    return {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_the_closure_covers_the_modules_the_selection_actually_runs_through():
    """Another guard that would pass vacuously if the walk came back empty."""
    assert "evaluation/selection.py" in SELECTION_CLOSURE
    assert "retrieval/conceptual.py" in SELECTION_CLOSURE
    assert "retrieval/fusion.py" in SELECTION_CLOSURE
    assert "evaluation/harness.py" in SELECTION_CLOSURE
    assert len(SELECTION_CLOSURE) >= 10


@pytest.mark.parametrize("module", SELECTION_CLOSURE)
def test_no_module_on_the_selection_path_names_a_split_other_than_dev(module: str):
    named = string_literals((SOURCE / module).read_text(encoding="utf-8")) & set(OTHER_SPLITS)

    assert not named, (
        f"{module} names {sorted(named)}; nothing the selection imports may name a split "
        "it is not allowed to read"
    )


def test_the_one_split_this_path_names_is_the_one_it_fits_on():
    assert DEV_SPLIT == "dev"
    assert DEV_SPLIT in string_literals((SOURCE / "evaluation/selection.py").read_text("utf-8"))


def test_the_stage_that_does_know_about_test_is_outside_the_closure():
    """The CLI splits the pool; the selection never sees the word. That is the design."""
    assert "cli.py" not in SELECTION_CLOSURE


# --- The guard is load-bearing ------------------------------------------------


def test_with_the_guard_removed_the_sweep_measures_a_test_question_without_a_murmur(
    monkeypatch,
):
    """The negative proof: a guard nobody has seen fail is a guard nobody knows works.

    Worse than merely measuring it, the cell comes back filed under `dev`, because
    the sweep tells the harness which split it is measuring rather than asking the
    questions. So the contamination would not show up in the artifact at all - which
    is why the refusal lives at the door and not in a later check.
    """
    monkeypatch.setattr(selection, "_check_questions", lambda questions: None)

    cells = one_space(questions_on("test"))

    assert cells, "the guard was removed and the sweep still refused for another reason"
    assert cells[0].n_questions == 2


# --- T13: the same guard, extended to the Phase 4 selection path --------------
#
# Phase 4 fits two parameters on the same 600 dev questions, so it needs the same
# unconditional rule and gets it by the same mechanism rather than by a second one:
# its grid calls `check_dev_only`, which delegates to the doorman above. What is new
# is the surface - the diffusion retriever and the expansion selection join the
# import closure - and the static check below now walks that closure too, so a module
# that reaches the walk is covered the moment something on the path imports it.

EXPANSION_CLOSURE = import_closure("evaluation/expansion_selection.py")
BOTH_CLOSURES = sorted(set(SELECTION_CLOSURE) | set(EXPANSION_CLOSURE))

DIFFUSION_ROWS = [
    [2.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 3.0, 0.0],
]


def a_walkable_dictionary() -> ConceptDictionary:
    return ConceptDictionary(
        atoms=np.eye(3, 3, dtype=np.float32),
        k=3,
        seed=42,
        sparsity_param=0.05,
        max_iter=3,
        unit_set_hash="101f564fdcca620c",
        model="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c",
    )


def a_walkable_matrix(dictionary: ConceptDictionary) -> ConceptMatrix:
    X = sparse.csr_matrix(np.array(DIFFUSION_ROWS, dtype=np.float32))
    return ConceptMatrix(
        X=X,
        unit_ids=list(UNITS),
        dictionary_key=dictionary.key,
        view="raw",
        coding_alpha=0.07,
        mean_active_per_unit=float(X.nnz) / X.shape[0],
        reconstruction_error=0.0,
    )


def the_expansion_grid(questions: Sequence[Question]) -> object:
    """Phase 4's door: one cell of the grid is enough to ask the guard."""
    dictionary = a_walkable_dictionary()
    cells, _spent = sweep_expansion(
        matrix=a_walkable_matrix(dictionary),
        dictionary=dictionary,
        seed_retrievers={
            "dense": StubRetriever("dense", [("u1", 0.9), ("u2", 0.4)]),
            "conceptual": StubRetriever("conceptual", [("u3", 0.7), ("u2", 0.2)]),
        },
        questions=questions,
        token_counts=TOKENS,
        base_config=a_config(),
        inherits={
            "k": 3,
            "dictionary_key": dictionary.key,
            "view": "raw",
            "damping": "idf",
            "query_operator": "projection_full",
        },
        restarts=(0.2,),
        normalizations=("none",),
        budgets=(config.SELECTION_BUDGET,),
        ks=(2,),
    )
    return cells


# Deliberately not folded into `DOORS` above: that table is parametrized at import
# time and mutating it afterwards silently desynchronises the ids from the values.
# This door gets the same three tests instead, which is what the table was doing.


def test_the_phase_4_door_opens_for_dev_so_its_refusal_means_something():
    assert the_expansion_grid(questions_on(DEV_SPLIT))


@pytest.mark.parametrize("split", OTHER_SPLITS)
def test_the_phase_4_grid_refuses_a_question_from_another_split(split: str):
    with pytest.raises(SelectionError, match=split):
        the_expansion_grid(questions_on(split))


def test_one_foreign_question_hidden_among_the_dev_ones_is_enough_to_refuse_the_grid():
    """The shape the accident would actually take, on Phase 4's path."""
    contaminated = [*questions_on(DEV_SPLIT), a_question("q9", "0,0,0,1", "test")]

    with pytest.raises(SelectionError, match="test"):
        the_expansion_grid(contaminated)


def test_the_phase_4_grid_asks_the_same_doorman(monkeypatch):
    """One guard for both phases: `check_dev_only` delegates rather than re-checking."""
    seen: list[int] = []
    real = selection._check_questions

    def spy(questions):
        seen.append(len(questions))
        return real(questions)

    monkeypatch.setattr(selection, "_check_questions", spy)
    the_expansion_grid(questions_on(DEV_SPLIT))

    assert seen, "the expansion grid never asked the guard"


def test_with_the_guard_removed_the_phase_4_grid_measures_a_test_question_too(monkeypatch):
    """The negative proof again, on the new path: the cell comes back filed under dev."""
    monkeypatch.setattr(selection, "_check_questions", lambda questions: None)

    cells = the_expansion_grid(questions_on("test"))

    assert cells, "the guard was removed and the grid still refused for another reason"
    assert cells[0].n_questions == 2


def test_a_result_measured_on_test_cannot_become_an_expansion_cell():
    result = RunResult(
        system="expansion",
        split="test",
        config=a_config(seed_arm="dense", restart=0.4, normalization="none", mean_iterations=2.0),
        metrics={f"budget_{config.SELECTION_BUDGET}": {config.EXPANSION_SELECTION_METRIC: 0.9}},
        cost={"n_questions": 2},
    )

    with pytest.raises(SelectionError, match="test"):
        ExpansionCell.from_result(result)


def test_the_closure_covers_the_modules_the_expansion_actually_runs_through():
    assert "evaluation/expansion_selection.py" in EXPANSION_CLOSURE
    assert "retrieval/diffusion.py" in EXPANSION_CLOSURE
    assert "evaluation/harness.py" in EXPANSION_CLOSURE
    assert "evaluation/selection.py" in EXPANSION_CLOSURE


@pytest.mark.parametrize("module", BOTH_CLOSURES)
def test_no_module_on_either_selection_path_names_a_split_other_than_dev(module: str):
    named = string_literals((SOURCE / module).read_text(encoding="utf-8")) & set(OTHER_SPLITS)

    assert not named, (
        f"{module} names {sorted(named)}; nothing either selection imports may name a "
        "split it is not allowed to read"
    )


def test_the_walk_itself_cannot_name_a_split():
    """The retriever is on the path now, and it reads no split, no gold and no label."""
    named = string_literals((SOURCE / "retrieval/diffusion.py").read_text(encoding="utf-8"))

    assert not named & set(OTHER_SPLITS)
    assert "evaluation/failure_analysis.py" not in EXPANSION_CLOSURE
