"""T13: the numerical path runs with neither optional dependency installed.

Reproducing the numbers of this experiment must never require an API key, an
account, or a network. Labelling is interpretability - it names atoms that
already exist - so `anthropic` and `python-dotenv` live in an optional group and
nothing in induction, coding, deduplication or diagnostics may reach for them.

The check runs in a subprocess and inspects `sys.modules`, because an import
somewhere three modules deep is exactly the kind that a reading of the source
misses and a transitive import does not.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NUMERICAL_PATH = """
import sys

import concept_embeddings_rag.cli
import concept_embeddings_rag.concepts.coding
import concept_embeddings_rag.concepts.dedup
import concept_embeddings_rag.concepts.diagnostics
import concept_embeddings_rag.concepts.dictionary

optional = {"anthropic", "dotenv", "httpx2", "pydantic"}
print(";".join(sorted(m for m in sys.modules if m.split(".")[0] in optional)))
"""


def test_the_numerical_path_imports_neither_optional_package():
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", NUMERICAL_PATH],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )

    leaked = result.stdout.strip()
    assert leaked == "", f"the numerical path pulled in optional packages: {leaked}"


def test_labelling_dependencies_live_in_their_own_group():
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    runtime = " ".join(manifest["project"]["dependencies"])
    labeling = " ".join(manifest["dependency-groups"]["labeling"])

    assert "anthropic" not in runtime
    assert "dotenv" not in runtime
    assert "anthropic" in labeling
    assert "python-dotenv" in labeling


def test_the_example_env_file_carries_the_name_and_no_value():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    lines = [line for line in example.splitlines() if line and not line.startswith("#")]
    assert lines == ["ANTHROPIC_API_KEY="]


def test_the_real_env_file_stays_ignored_and_the_example_does_not():
    ignore_rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignore_rules
    assert "!.env.example" in ignore_rules


def test_no_env_file_is_committed():
    """The example is versioned; the file holding an actual key never is."""
    tracked = subprocess.run(  # noqa: S603
        ["git", "ls-files", ".env", ".env.*"],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )

    assert tracked.stdout.split() in ([], [".env.example"])
