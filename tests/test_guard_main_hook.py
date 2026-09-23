"""Regression tests for the PreToolUse guard in .claude/hooks/guard_main.py.

The hook is the local barrier behind the agent workflow: no pushes to main, no PR merges, no
.env reads, gh limited to an allowlist of subcommands and gh-axi to an allowlist of reads.
Each case runs the real hook script with a Claude Code style payload, against a scratch
repository on a working branch or on main.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "guard_main.py"
GIT = shutil.which("git")
BLOCKED, ALLOWED = 2, 0

pytestmark = pytest.mark.skipif(GIT is None, reason="git is required")


def _repo(path: Path, branch: str) -> Path:
    assert GIT is not None
    subprocess.run([GIT, "init", "-q", "-b", branch, str(path)], check=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [GIT, "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q"]
        + ["--allow-empty", "-m", "init"],
        check=True,
    )
    return path


def _run_hook(command: str, cwd: Path) -> int:
    payload = json.dumps({"tool_input": {"command": command}, "cwd": str(cwd)})
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(HOOK)], input=payload, text=True, capture_output=True, check=False
    )
    return result.returncode


@pytest.fixture(scope="module")
def work_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _repo(tmp_path_factory.mktemp("work"), "work")


@pytest.fixture(scope="module")
def main_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _repo(tmp_path_factory.mktemp("main"), "main")


AXI = "npx -y gh-axi@0.1.35"

ON_WORK_BRANCH = [
    # gh-axi allowlist: the reads /prime and deliver use.
    (f"{AXI} pr view 12", ALLOWED),
    (f"{AXI} pr checks 12", ALLOWED),
    (f"{AXI} pr list --state open --limit 5", ALLOWED),
    (f"{AXI} run list", ALLOWED),
    (f"{AXI} run view 35839937890", ALLOWED),
    ('npx -y "gh-axi@0.1.35" pr view 12', ALLOWED),
    # Review of PR #12: run is not read-only.
    (f"{AXI} run delete 123", BLOCKED),
    (f"{AXI} run rerun 123", BLOCKED),
    (f"{AXI} run cancel 123", BLOCKED),
    # Review of PR #12: stack merge merges, stack submit enables auto-merge.
    (f"{AXI} stack merge", BLOCKED),
    (f"{AXI} stack submit", BLOCKED),
    # Review of PR #12: a quoted package name must not hide the call.
    ('npx -y "gh-axi@0.1.35" pr merge 12', BLOCKED),
    ("npx -y 'gh-axi' pr merge 12", BLOCKED),
    ('npx -y gh-"axi"@0.1.35 pr merge 12', BLOCKED),
    # Anything outside the allowlist is blocked, known or not.
    ("gh-axi foo bar", BLOCKED),
    ("gh-axi", BLOCKED),
    (f"{AXI} pr merge 12", BLOCKED),
    (f"{AXI} api -X PUT repos/o/r/pulls/9/merge", BLOCKED),
    ("node dist/bin/gh-axi.js pr merge 1", BLOCKED),
    # Second review of PR #12: Windows paths must not glue gh-axi to its folder.
    (r".\node_modules\.bin\gh-axi.cmd pr merge 1", BLOCKED),
    (r"node C:\tools\gh-axi\dist\bin\gh-axi.js pr merge 1", BLOCKED),
    (r"gh\-axi pr merge 1", BLOCKED),
    (r"C:\tools\gh.exe pr merge 3", BLOCKED),
    # Second review of PR #12: a body executed as a script is still a command.
    (f"sh <<'EOF'\n{AXI} pr merge 12\nEOF", BLOCKED),
    ("Invoke-Expression @'\ngh pr merge 12\n'@", BLOCKED),
    ("bash <<EOF\ngit push origin main\nEOF", BLOCKED),
    ("npx -p gh-axi@0.1.35 gh-axi pr merge 1", BLOCKED),
    (f"{AXI} pr view 12; {AXI} pr merge 12", BLOCKED),
    # gh: merges blocked, quoted or not; reads and PR writes allowed.
    ("gh pr merge 9 --squash", BLOCKED),
    ('"gh" pr merge 9', BLOCKED),
    ("gh api -X PUT repos/o/r/pulls/9/merge", BLOCKED),
    ("gh pr view 9", ALLOWED),
    ("gh pr create --base main --title t --body-file body.md", ALLOWED),
    ("gh pr edit 12 --body-file body.md", ALLOWED),
    ("gh pr checks 12", ALLOWED),
    ("gh run view 35839937890 --log-failed", ALLOWED),
    ("gh repo view --json visibility", ALLOWED),
    # Third review of PR #12: gh is allowlisted too, so flags before the subcommand and
    # every api call (REST or GraphQL) are blocked.
    ("gh pr --repo o/r merge 9", BLOCKED),
    ("gh -R o/r pr merge 9", BLOCKED),
    (
        "gh api graphql -f query='mutation { mergePullRequest(input: {}) { clientMutationId } }'",
        BLOCKED,
    ),
    ("gh api graphql -f query=enablePullRequestAutoMerge", BLOCKED),
    ("gh api repos/o/r/pulls/9", BLOCKED),
    ("gh pr ready 9", BLOCKED),
    ("gh foo", BLOCKED),
    # git push on a working branch.
    ("git push -u origin work", ALLOWED),
    ("git push", ALLOWED),
    ("git push origin maintenance", ALLOWED),
    ("git push origin main", BLOCKED),
    ("git push origin 'main'", BLOCKED),
    ('"git" push origin main', BLOCKED),
    ("git push origin HEAD:main", BLOCKED),
    ("git -C . push origin main", BLOCKED),
    ("git push --all origin", BLOCKED),
    ("git push --force origin work", BLOCKED),
    ("git push origin +work", BLOCKED),
    ("uv run pytest && git push origin main", BLOCKED),
    # Inline text is matched, heredoc bodies included; messages go through files.
    ("git commit -F msg.txt", ALLOWED),
    ("git commit -F - <<'EOF'\nfix the parser\nEOF\ngit log -1", ALLOWED),
    ("git commit -F - <<'EOF'\nmentions gh pr merge\nEOF", BLOCKED),
    ("git commit -F - <<'EOF'\nmsg\nEOF\ngh pr merge 3", BLOCKED),
    # .env
    ("cat .env", BLOCKED),
    ('Get-Content ".env"', BLOCKED),
    ("type .env.local", BLOCKED),
    ("cat .env.example", ALLOWED),
    # Third review of PR #12: quotes or escapes splitting the name must not hide .env.
    ("cat .e'n'v", BLOCKED),
    ('cat .e"n"v', BLOCKED),
    (r"cat .e\nv", BLOCKED),
    (r"type C:\repo\.env", BLOCKED),
    ("ls", ALLOWED),
]

ON_MAIN = [
    ("git push", BLOCKED),
    ("git push origin HEAD", BLOCKED),
    ("git commit -m x", BLOCKED),
    ("git -C . commit -m x", BLOCKED),
    ("git merge work", BLOCKED),
    ("git switch -c work", ALLOWED),
    ("git pull", ALLOWED),
    (f"{AXI} pr view 12", ALLOWED),
]


@pytest.mark.parametrize(("command", "expected"), ON_WORK_BRANCH)
def test_on_working_branch(work_repo: Path, command: str, expected: int) -> None:
    assert _run_hook(command, work_repo) == expected


@pytest.mark.parametrize(("command", "expected"), ON_MAIN)
def test_on_main(main_repo: Path, command: str, expected: int) -> None:
    assert _run_hook(command, main_repo) == expected
