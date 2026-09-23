"""PreToolUse hook: keep agents off main.

Agents may commit and push working branches freely; main changes only through a PR the author
merges. Blocks (exit code 2) any shell command that would push to main, force-push, merge a PR,
or commit while main is checked out. Standard library only; ASCII output only.
"""

import json
import re
import shutil
import subprocess
import sys

PR_MERGE = re.compile(r"\bgh\s+pr\s+merge\b")
PUSH = re.compile(r"\bgit\s+push\b([^;&|\n]*)")
COMMIT = re.compile(r"\bgit\s+commit\b")
FORCE = re.compile(r"(^|\s)(-f|--force|--force-with-lease)(\s|=|$)|\s\+\S")
TARGETS_MAIN = re.compile(r"(^|\s|:)(refs/heads/)?main(\s|$)")
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n.*?^\s*\1\s*$", re.DOTALL | re.MULTILINE)
QUOTED = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"")


def strip_literals(command: str) -> str:
    """Drop heredoc bodies and quoted strings, so a commit message or PR body that mentions
    a blocked command is not mistaken for running it."""
    return QUOTED.sub("''", HEREDOC.sub("", command))


def current_branch(cwd: str | None) -> str:
    git = shutil.which("git")
    if git is None:
        return ""
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, resolved git path, no external input
            [git, "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def block(reason: str) -> None:
    print(f"[BLOCKED] {reason} main changes only through a PR the author merges.", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return
    command = strip_literals(str(payload.get("tool_input", {}).get("command", "")))
    if not command:
        return
    cwd = payload.get("cwd")

    if PR_MERGE.search(command):
        block("Agents do not merge pull requests.")

    for match in PUSH.finditer(command):
        args = match.group(1)
        if FORCE.search(args):
            block("Force-push is not allowed.")
        if TARGETS_MAIN.search(args):
            block("Pushing to main is not allowed.")
        refspecs = [a for a in args.split() if not a.startswith("-")]
        if len(refspecs) < 2 and current_branch(cwd) == "main":
            block("main is checked out; a bare push would update it.")

    if COMMIT.search(command) and current_branch(cwd) == "main":
        block("Do not commit on main; create a working branch first.")


if __name__ == "__main__":
    main()
