"""`_git_commit` marks a run whose code paths hold uncommitted changes (fix-2).

Every git call here runs in a throwaway repository with the caller's git environment removed:
`GIT_DIR`, `GIT_INDEX_FILE` and friends (set inside hooks and some IDEs) would otherwise point
the fixture at the real repository, and global or system config (identity, signing, hooks)
would change what it does. The fixture also checks it is inside its own repository before it
writes anything.
"""

import os
import subprocess
from pathlib import Path

import pytest

from concept_embeddings_rag import cli, config

_IDENTITY = ("-c", "user.name=test", "-c", "user.email=test@example.com")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", *_IDENTITY, "-c", "commit.gpgsign=false", *args],  # noqa: S607
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def isolated_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No inherited `GIT_*` variable, no global or system config, no repository above."""
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def repo(isolated_git: Path) -> Path:
    _git(isolated_git, "init", "-q")
    toplevel = Path(_git(isolated_git, "rev-parse", "--show-toplevel")).resolve()
    assert toplevel == isolated_git.resolve(), f"refusing to write into {toplevel}"
    (isolated_git / "src").mkdir()
    (isolated_git / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (isolated_git / "notes.md").write_text("notes\n", encoding="utf-8")
    _git(isolated_git, "add", ".")
    _git(isolated_git, "commit", "-q", "-m", "initial")
    return isolated_git


def test_clean_tree_records_the_bare_commit(repo: Path) -> None:
    assert cli._git_commit() == _git(repo, "rev-parse", "HEAD")


def test_modified_code_is_marked_dirty(repo: Path) -> None:
    (repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")
    assert cli._git_commit() == _git(repo, "rev-parse", "HEAD") + "-dirty"


def test_staged_code_is_marked_dirty(repo: Path) -> None:
    (repo / "src" / "module.py").write_text("x = 3\n", encoding="utf-8")
    _git(repo, "add", "src/module.py")
    assert cli._git_commit().endswith("-dirty")


def test_untracked_code_is_marked_dirty(repo: Path) -> None:
    (repo / "src" / "new_stage.py").write_text("y = 1\n", encoding="utf-8")
    assert cli._git_commit().endswith("-dirty")


def test_changes_outside_the_code_paths_are_not_dirty(repo: Path) -> None:
    (repo / "notes.md").write_text("edited\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("untracked\n", encoding="utf-8")
    assert cli._git_commit() == _git(repo, "rev-parse", "HEAD")


def test_outside_a_checkout_is_unknown(isolated_git: Path) -> None:
    assert cli._git_commit() == "unknown"
