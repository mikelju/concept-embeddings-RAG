"""T16, extended by T6: a label never reaches the retrieval path, and neither does
a gold annotation or a split label.

The founding decision is that a concept's embedding is its own dictionary atom,
not the embedding of a name an LLM invented for it. That is what lets Phase 3 map
a question to concepts with a dot product, and it is why labelling can be
non-reproducible without contaminating anything.

The rule only holds if nothing downstream reads the labels, so it is asserted
here rather than remembered. `retrieval/` and `evaluation/` are globbed as whole
directories, so a module that lands there - `conceptual.py`, and `fusion.py` after
it - is covered the moment it exists rather than when someone remembers to add it.
The `concepts/` modules are named one by one, because only some of them are on the
scoring path and the choice of which deserves to be visible.

T6 adds two things. The first is proof of life: guards that have never been seen to
fail are guards nobody knows work, so the ones below are fed a real module with the
forbidden line inserted and must reject it. The second is the other half of the same
rule - a retrieval module must not read a gold annotation or a split label either,
which is what keeps a dev sweep from quietly learning the answer. `evaluation/` is
deliberately outside that second guard: `harness.py` is the measurer, and reading
gold and split is its job.
"""

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src" / "concept_embeddings_rag"

RETRIEVAL_PATH: tuple[Path, ...] = (
    SOURCE / "retrieval",
    SOURCE / "evaluation",
    SOURCE / "concepts" / "dictionary.py",
    SOURCE / "concepts" / "coding.py",
    SOURCE / "concepts" / "dedup.py",
    SOURCE / "concepts" / "diagnostics.py",
)

# The same path minus the measurer. `evaluation/harness.py` must read gold and split
# to compute a metric at all; every module below it must not, or the score would be
# reading the answer key it is being scored against.
SCORING_PATH: tuple[Path, ...] = tuple(
    entry for entry in RETRIEVAL_PATH if entry != SOURCE / "evaluation"
)

FORBIDDEN_NAMES = ("labeling", "labels", "ConceptLabels", "load_labels")

# Names that only appear in code that has an answer key or a split in its hands.
FORBIDDEN_ANNOTATIONS = (
    "gold",
    "gold_unit_ids",
    "supporting_facts",
    "split",
    "is_gold",
    "relevant_unit_ids",
)


def modules_on_the_retrieval_path() -> list[Path]:
    return modules_under(RETRIEVAL_PATH)


def modules_on_the_scoring_path() -> list[Path]:
    return modules_under(SCORING_PATH)


def modules_under(entries: tuple[Path, ...]) -> list[Path]:
    found: list[Path] = []
    for entry in entries:
        found.extend(sorted(entry.glob("*.py")) if entry.is_dir() else [entry])
    return found


def imported_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def identifiers_used(source: str) -> set[str]:
    """Every name the code actually touches: attributes, variables, parameters, keys.

    Built from the syntax tree rather than by searching the text, and the difference
    matters here: these modules state in their own docstrings that they read no gold
    and no split, so a substring guard would fail on the very sentence declaring the
    rule it enforces. Docstrings and comments are not identifiers, so they are simply
    not in this set.
    """
    used: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.arg | ast.keyword) and node.arg:
            used.add(node.arg)
        elif isinstance(node, ast.alias):
            used.add((node.asname or node.name).rsplit(".", maxsplit=1)[-1])
        # `record["split"]` reads a split just as surely as `record.split` does.
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            used.add(node.slice.value)
    return used


def test_the_declared_path_exists_and_is_not_empty():
    """A guard over nothing passes vacuously, which is the failure to avoid."""
    for entry in RETRIEVAL_PATH:
        assert entry.exists(), f"{entry} is named in the guard but does not exist"
    assert len(modules_on_the_retrieval_path()) >= 8


@pytest.mark.parametrize(
    "path", modules_on_the_retrieval_path(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_no_retrieval_module_imports_the_labelling_module(path: Path):
    imports = imported_names(path.read_text(encoding="utf-8"))

    assert not any("labeling" in name for name in imports), (
        f"{path.name} imports the labelling module; a concept's embedding is its atom, "
        "never the embedding of a generated name"
    )


@pytest.mark.parametrize(
    "path", modules_on_the_retrieval_path(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_no_retrieval_module_reads_the_labels_artifact(path: Path):
    source = path.read_text(encoding="utf-8")

    assert "labels-" not in source, f"{path.name} reaches for the labels artifact by name"
    for name in FORBIDDEN_NAMES:
        assert f"load_{name}" not in source


def test_the_labelling_module_is_the_only_place_the_labels_artifact_is_named():
    """Written by `labeling.py`, consumed by the CLI stage that reports. Nowhere else."""
    naming = {
        path.relative_to(SOURCE).as_posix()
        for path in SOURCE.rglob("*.py")
        if "labels-" in path.read_text(encoding="utf-8")
    }

    assert naming == {"concepts/labeling.py"}


# --- The guards proved to bite (T6) -------------------------------------------


def conceptual_source() -> str:
    """The System B module, which is the one T6 exists to keep clean."""
    return (SOURCE / "retrieval" / "conceptual.py").read_text(encoding="utf-8")


def test_the_module_the_guard_is_aimed_at_is_actually_covered_by_it():
    """`retrieval/` is globbed, so System B and the fusion module after it are in."""
    covered = {path.name for path in modules_on_the_retrieval_path()}

    assert "conceptual.py" in covered


def test_the_import_guard_fails_if_conceptual_py_imports_the_labelling_module():
    """Fed the one line the rule forbids, the predicate must reject a real module."""
    clean = conceptual_source()
    offending = "from concept_embeddings_rag.concepts.labeling import load_concept_labels\n" + clean

    assert not any("labeling" in name for name in imported_names(clean))
    assert any("labeling" in name for name in imported_names(offending))


def test_the_import_guard_fails_on_an_aliased_import_too():
    """`import ... as naming` hides the name from a reader, not from the syntax tree."""
    offending = "import concept_embeddings_rag.concepts.labeling as naming\n" + conceptual_source()

    assert any("labeling" in name for name in imported_names(offending))


def test_the_artifact_guard_fails_if_conceptual_py_reaches_for_the_labels_file():
    clean = conceptual_source()
    offending = clean.replace(
        'PROJECTION = "projection"',
        'LABELS = f"labels-{dictionary_key}.json"\nPROJECTION = "projection"',
    )

    assert offending != clean, "the mutation did not land; the guard was not exercised"
    assert "labels-" not in clean
    assert "labels-" in offending


# --- No gold annotation, no split label (T6) ----------------------------------


def test_the_scoring_path_exists_and_excludes_the_measurer():
    for entry in SCORING_PATH:
        assert entry.exists(), f"{entry} is named in the guard but does not exist"
    assert SOURCE / "evaluation" not in SCORING_PATH
    assert len(modules_on_the_scoring_path()) >= 4


@pytest.mark.parametrize(
    "path", modules_on_the_scoring_path(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_no_module_on_the_scoring_path_reads_a_gold_annotation_or_a_split_label(path: Path):
    """Tuning is done on dev; a retriever that could read the split could read either.

    The question reaches these modules as a string and nothing else. Whether it came
    from dev or from test is not knowable here, which is the structural version of the
    project's rule about never tuning against the evaluation set.
    """
    used = identifiers_used(path.read_text(encoding="utf-8"))

    trespassing = sorted(used & set(FORBIDDEN_ANNOTATIONS))
    assert not trespassing, (
        f"{path.name} touches {trespassing}; the retrieval path sees a query string, "
        "never the answer key it is about to be scored against"
    )


def test_the_measurer_is_outside_the_guard_because_reading_them_is_its_job():
    """The positive control: the same predicate, pointed at code that must trip it.

    If `identifiers_used` could not see `gold_unit_ids` and `split` here, the guard
    above would be passing because it detects nothing, not because nothing is wrong.
    """
    used = identifiers_used((SOURCE / "evaluation" / "harness.py").read_text(encoding="utf-8"))

    assert "gold_unit_ids" in used
    assert "split" in used


def test_the_gold_guard_fails_if_conceptual_py_starts_reading_the_answer_key():
    clean = conceptual_source()
    offending = clean.replace(
        "    def retrieve(self, query: str, top_k: int) -> list[Hit]:",
        "    def retrieve(self, query: str, top_k: int, gold_unit_ids: list[str]) -> list[Hit]:",
    )

    assert offending != clean, "the mutation did not land; the guard was not exercised"
    assert not identifiers_used(clean) & set(FORBIDDEN_ANNOTATIONS)
    assert identifiers_used(offending) & set(FORBIDDEN_ANNOTATIONS) == {"gold_unit_ids"}
