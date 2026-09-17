"""Phase 6, T19: the test split and the answer key cannot leak into selection or retrieval (HU-9).

Five properties, each asserted on the code that runs:

- **Every dev door refuses a test question**, including one hidden among dev questions: the
  fitting helper, the measuring helper when asked for dev, and the second dev run.
- **The historical test figures have one reader and one call site**: `read_historical_test_figures`
  is called once, inside `run_test`, and nowhere else in the source; a tampered source that calls
  it from a dev stage fails the check. No test module reads a historical test file.
- **The answer key does not move a ranking**: shuffling gold annotations leaves `D(q)`, `E(q)` and
  every fused ranking unchanged.
- **No API key and no SDK**: no Phase 6 module imports the Anthropic SDK or `read_api_key`, and the
  dev stages run with the key unset and the SDK unimportable.
- **Phase 1-5 artifacts are read, never written**: no Phase 6 module calls a Phase 1-5 writer or
  names a Phase 1-5 directory outside the one place paths are read from `config`, and running the
  stages on the toy pipeline leaves every input byte-identical.
"""

import ast
import hashlib
import random
from dataclasses import replace
from pathlib import Path

import pytest
from replacement_fixtures import ToyPipeline, build_toy_pipeline, relocated, toy_counter

from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import replacement_run
from concept_embeddings_rag.evaluation.replacement_run import (
    ReplacementRunError,
    StageEnvironment,
    build_components,
    build_hybrid,
    fit_hybrid,
    measure,
    rerun,
    run_check,
    run_freeze,
    verified_inputs,
)
from concept_embeddings_rag.evaluation.selection import SelectionError

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "concept_embeddings_rag"
TESTS = ROOT / "tests"
PHASE_6_MODULES = (
    "decision_parameters.py",
    "evaluation/outcomes.py",
    "retrieval/entity_hop.py",
    "evaluation/replacement_inputs.py",
    "evaluation/continuity.py",
    "evaluation/control_reproduction.py",
    "evaluation/tango.py",
    "evaluation/replacement_decision.py",
    "evaluation/replacement_freeze.py",
    "evaluation/entity_diagnostics.py",
    "evaluation/readout.py",
    "evaluation/replacement_run.py",
)
HISTORICAL_TEST_FILES = (
    "run-dense-test-20260911T133159+0000.json",
    "run-hybrid-bm25-test-20260911T133204+0000.json",
)


@pytest.fixture(scope="module")
def pristine(tmp_path_factory) -> ToyPipeline:
    return build_toy_pipeline(tmp_path_factory.mktemp("contamination") / "toy")


@pytest.fixture
def toy(pristine, tmp_path) -> ToyPipeline:
    return relocated(pristine, tmp_path / "toy")


def environment(pipeline: ToyPipeline) -> StageEnvironment:
    return StageEnvironment(
        paths=pipeline.paths,
        pins=pipeline.pins,
        replacement_dir=pipeline.replacement_dir,
        backend=pipeline.backend,
        token_counter=toy_counter,
    )


def source_of(module: str) -> str:
    return (SOURCE / module).read_text(encoding="utf-8")


# --- Every dev door refuses a test question ---------------------------------------------------


@pytest.fixture
def dev_world(toy):
    inputs = verified_inputs(environment(toy))
    dense, bm25, stage = build_components(inputs, inputs.dev_query_backend)
    dev = [question for question in inputs.questions if question.split == "dev"]
    test = [question for question in inputs.questions if question.split == "test"]
    return inputs, dense, bm25, stage, dev, test


def config_of(inputs) -> dict:
    return replacement_run.base_config(inputs, inputs.dev_question_cache_key)


@pytest.mark.parametrize("hidden", [False, True])
def test_the_fitting_door_refuses_a_test_question(dev_world, hidden):
    inputs, dense, bm25, _stage, dev, test = dev_world
    questions = [*dev[:-1], test[0]] if hidden else test[:1]
    with pytest.raises(SelectionError, match="test"):
        fit_hybrid(
            dense,
            bm25,
            questions=questions,
            token_counts=inputs.token_counts,
            run_config=config_of(inputs),
        )


@pytest.mark.parametrize("hidden", [False, True])
def test_the_second_run_door_refuses_a_test_question(dev_world, hidden):
    inputs, dense, _bm25, _stage, dev, test = dev_world
    questions = [*dev[:-1], test[0]] if hidden else test[:1]
    with pytest.raises(SelectionError, match="test"):
        rerun(
            dense,
            questions=questions,
            token_counts=inputs.token_counts,
            run_config=config_of(inputs),
        )


@pytest.mark.parametrize("hidden", [False, True])
def test_the_measuring_door_refuses_a_question_of_another_split(dev_world, toy, hidden):
    inputs, dense, _bm25, _stage, dev, test = dev_world
    questions = [*dev[:-1], test[0]] if hidden else test[:1]
    with pytest.raises(ReplacementRunError, match="split"):
        measure(
            dense,
            questions=questions,
            split="dev",
            token_counts=inputs.token_counts,
            run_config=config_of(inputs),
            directory=toy.replacement_dir,
            freeze_digest=None,
        )
    assert (
        not list(toy.replacement_dir.glob("run-*.json")) if toy.replacement_dir.exists() else True
    )


def test_the_dev_question_door_hands_out_dev_questions_only(dev_world):
    inputs = dev_world[0]
    assert {question.split for question in replacement_run.dev_questions(inputs)} == {"dev"}


# --- One reader of historical test figures, one call site -------------------------------------


def call_sites(source: str, name: str) -> list[str]:
    """The enclosing top-level function of every call to `name`."""
    sites: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == name
                ):
                    sites.append(node.name)
    return sites


def reader_call_sites(source: str) -> list[str]:
    return call_sites(source, "read_historical_test_figures")


def test_the_reader_of_historical_test_figures_has_exactly_one_call_site_in_run_test():
    assert reader_call_sites(source_of("evaluation/replacement_run.py")) == ["run_test"]


def test_no_other_source_module_names_the_reader():
    for path in SOURCE.rglob("*.py"):
        if path.name == "replacement_run.py":
            continue
        assert "read_historical_test_figures" not in path.read_text(encoding="utf-8"), path


def test_the_check_fails_on_a_tampered_source_that_calls_the_reader_from_a_dev_stage():
    clean = source_of("evaluation/replacement_run.py")
    anchor = "    checks = load_checks(directory)\n"
    tampered = clean.replace(
        anchor, anchor + "    read_historical_test_figures(None, directory)\n", 1
    )
    assert tampered != clean
    assert reader_call_sites(tampered) == ["run_freeze", "run_test"]


def test_the_reader_is_called_after_dense_and_a_are_measured_in_run_test():
    tree = ast.parse(source_of("evaluation/replacement_run.py"))
    run_test = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_test")
    lines = {}
    for node in ast.walk(run_test):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            lines.setdefault(node.func.id, []).append(node.lineno)
    reader = lines["read_historical_test_figures"][0]
    measures = sorted(lines["measure"])
    assert measures[0] < measures[1] < reader < measures[2]


def test_the_identity_reader_is_the_only_code_reading_the_pinned_test_file_names():
    """`historical_test_files` flows only into identification in the inputs module."""
    source = source_of("evaluation/replacement_inputs.py")
    assert call_sites(source, "identify_result_file") == ["identify_historical_files"]
    tree = ast.parse(source)
    users = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Attribute) and inner.attr == "historical_test_files"
            for inner in ast.walk(node)
        )
    }
    assert users == {"identify_historical_files"}


READING_CALLS = {"read_text", "read_bytes", "open", "load", "loads"}


def reads_a_historical_test_file(source: str) -> list[int]:
    """Lines where a reading call's own expression names a historical test file."""
    flagged: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        )
        if name not in READING_CALLS:
            continue
        for inner in ast.walk(node):
            named = (
                isinstance(inner, ast.Constant)
                and isinstance(inner.value, str)
                and any(stem in inner.value for stem in ("-test-20260911", *HISTORICAL_TEST_FILES))
            ) or (
                isinstance(inner, ast.Attribute | ast.Name)
                and getattr(inner, "attr", getattr(inner, "id", ""))
                == "HISTORICAL_TEST_RESULT_FILES"
            )
            if named:
                flagged.append(node.lineno)
                break
    return flagged


def test_no_test_module_reads_a_historical_test_file():
    for path in TESTS.rglob("*.py"):
        assert reads_a_historical_test_file(path.read_text(encoding="utf-8")) == [], path


def test_the_test_module_check_fails_on_a_tampered_test():
    tampered = (
        "import json\nfrom pathlib import Path\n"
        "figures = json.loads(Path('data/results/run-dense-test-20260911T133159+0000.json')"
        ".read_text())\n"
    )
    assert reads_a_historical_test_file(tampered)


# --- Gold does not move a ranking -------------------------------------------------------------


def rankings(inputs, dense, bm25, stage, questions: list[Question]) -> list:
    control = build_hybrid(dense, bm25, "weighted", {"dense": 0.5, "bm25": 0.5})
    entity = build_hybrid(dense, stage, "rrf", None)
    readings = []
    for question in questions:
        first = dense.retrieve(question.question, 100)
        readings.append(
            (
                first,
                stage.propose(first, 100),
                control.retrieve(question.question, 100),
                entity.retrieve(question.question, 100),
            )
        )
    return readings


def test_shuffling_gold_annotations_leaves_every_ranking_unchanged(dev_world):
    inputs, dense, bm25, stage, dev, _test = dev_world
    before = rankings(inputs, dense, bm25, stage, dev)

    rng = random.Random(7)  # noqa: S311 - a seeded shuffle of toy annotations
    golds = [question.gold_unit_ids for question in dev]
    rng.shuffle(golds)
    shuffled = [
        replace(question, gold_unit_ids=gold, supporting_facts=(), answer="shuffled")
        for question, gold in zip(dev, golds, strict=True)
    ]
    assert [q.gold_unit_ids for q in shuffled] != [q.gold_unit_ids for q in dev]

    assert rankings(inputs, dense, bm25, stage, shuffled) == before


# --- No key and no SDK ------------------------------------------------------------------------


def imported_modules(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("module", PHASE_6_MODULES)
def test_no_phase_6_module_imports_the_sdk_or_reads_the_key(module):
    source = source_of(module)
    imported = imported_modules(source)
    assert "anthropic" not in imported
    assert "read_api_key" not in imported
    assert "read_api_key" not in source
    assert "ANTHROPIC_API_KEY" not in source


def test_the_dev_stages_run_with_the_key_unset_and_the_sdk_unimportable(toy, monkeypatch):
    import sys

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert run_check(environment(toy)).passed
    assert run_freeze(environment(toy)).passed


# --- Phase 1-5 artifacts are read, never written ----------------------------------------------

PHASE_1_TO_5_WRITERS = {
    "save_pool",
    "save_selection",
    "save_pilot",
    "save_node_index",
    "save_navigation",
    "save_gate",
    "save_matrix",
    "save_dictionary",
    "save_labels",
    "save_diagnostics",
    "save_expansion_selection",
    "run_full",
    "run_sample",
}
PHASE_1_TO_5_DIRECTORIES = {
    "NODES_DIR",
    "EXTRACTION_DIR",
    "NAVIGATION_DIR",
    "PILOT_DIR",
    "SELECTION_DIR",
    "RESULTS_DIR",
    "CACHE_DIR",
    "QUESTION_CACHE_DIR",
}


@pytest.mark.parametrize("module", PHASE_6_MODULES)
def test_no_phase_6_module_calls_a_phase_1_to_5_writer(module):
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(ast.parse(source_of(module)))
        if isinstance(node, ast.Call)
    }
    assert not called & PHASE_1_TO_5_WRITERS


@pytest.mark.parametrize("module", PHASE_6_MODULES)
def test_phase_1_to_5_directories_are_named_only_where_input_paths_are_read(module):
    tree = ast.parse(source_of(module))
    for node in tree.body:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Attribute) and inner.attr in PHASE_1_TO_5_DIRECTORIES:
                assert isinstance(node, ast.ClassDef) and node.name == "InputPaths", (
                    module,
                    inner.attr,
                )


def snapshot(pipeline: ToyPipeline) -> dict[str, str]:
    digests = {}
    for path in sorted(pipeline.paths.data_dir.rglob("*")):
        if path.is_file() and pipeline.replacement_dir not in path.parents:
            digests[str(path.relative_to(pipeline.root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digests


def test_running_the_dev_stages_leaves_every_input_byte_identical(toy):
    before = snapshot(toy)
    assert run_check(environment(toy)).passed
    assert run_freeze(environment(toy)).passed
    assert snapshot(toy) == before
