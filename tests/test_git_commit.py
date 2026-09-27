"""`_git_commit` marks a run whose code paths hold uncommitted changes (fix-2)."""

import subprocess
from pathlib import Path

import pytest

from concept_embeddings_rag import cli, config


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("notes\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "initial")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_clean_tree_records_the_bare_commit(repo: Path) -> None:
    assert cli._git_commit() == _git(repo, "rev-parse", "HEAD")


def test_modified_code_is_marked_dirty(repo: Path) -> None:
    (repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")
    assert cli._git_commit() == _git(repo, "rev-parse", "HEAD") + "-dirty"


def test_untracked_code_is_marked_dirty(repo: Path) -> None:
    (repo / "src" / "new_stage.py").write_text("y = 1\n", encoding="utf-8")
    assert cli._git_commit().endswith("-dirty")


def test_changes_outside_the_code_paths_are_not_dirty(repo: Path) -> None:
    (repo / "notes.md").write_text("edited\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("untracked\n", encoding="utf-8")
    assert cli._git_commit() == _git(repo, "rev-parse", "HEAD")


def test_outside_a_checkout_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    assert cli._git_commit() == "unknown"
