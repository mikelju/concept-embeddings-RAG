"""T16: a label never reaches the retrieval path.

The founding decision is that a concept's embedding is its own dictionary atom,
not the embedding of a name an LLM invented for it. That is what lets Phase 3 map
a question to concepts with a dot product, and it is why labelling can be
non-reproducible without contaminating anything.

The rule only holds if nothing downstream reads the labels, so it is asserted
here rather than remembered. The list of modules is written out deliberately: a
new retrieval module has to be added to it by hand, which is the moment someone
has to think about whether it reads a name.
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

FORBIDDEN_NAMES = ("labeling", "labels", "ConceptLabels", "load_labels")


def modules_on_the_retrieval_path() -> list[Path]:
    found: list[Path] = []
    for entry in RETRIEVAL_PATH:
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
