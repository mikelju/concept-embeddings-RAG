"""PreToolUse hook: keep agents off main and away from .env.

Agents may commit and push working branches freely; main changes only through a PR the author
merges. Blocks (exit code 2) any shell command that would push to main, push everything,
force-push, merge a PR, change history while main is checked out, or read a .env file.
It is a local guard, not a substitute for branch protection on the remote.
Standard library only; ASCII output only.
"""

import json
import re
import shutil
import subprocess
import sys

# "git" followed by global options (-C dir, -c k=v, --flag) and then the subcommand.
GIT_PREFIX = r"\bgit(?:\s+(?:-[Cc]\s+\S+|--?[\w-]+(?:=\S+)?))*\s+"
PUSH = re.compile(GIT_PREFIX + r"push\b([^;&|\n]*)")
HISTORY_ON_MAIN = re.compile(GIT_PREFIX + r"(commit|merge|cherry-pick|revert|rebase|am|reset)\b")
PR_MERGE = re.compile(r"\bgh\s+pr\s+merge\b|\bgh\s+api\b[^;&|\n]*/merge\b")
FORCE = re.compile(r"(^|\s)(-f|--force|--force-with-lease|--force-if-includes)(\s|=|$)|\s\+\S")
PUSH_ALL = re.compile(r"(^|\s)(--all|--mirror)(\s|=|$)")
TARGETS_MAIN = re.compile(r"(^|\s|:)(refs/heads/)?main(\s|$)")
READS_ENV = re.compile(r"(^|[\s'\"/\\=])\.env(\.(?!example\b)[\w.-]+)?(?=$|[\s'\";|&)])")
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n.*?^\s*\1\s*$", re.DOTALL | re.MULTILINE)
PS_HERESTRING = re.compile(r"@(['\"])\r?\n.*?\r?\n\1@", re.DOTALL)
QUOTED = re.compile(r"'[^'\n]*'|\"(?:\\.|[^\"\\\n])*\"")


def without_bodies(command: str) -> str:
    """Drop heredoc and PowerShell here-string bodies (commit messages, PR bodies)."""
    return PS_HERESTRING.sub("", HEREDOC.sub("", command))


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
    print(f"[BLOCKED] {reason}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return
    raw = without_bodies(str(payload.get("tool_input", {}).get("command", "")))
    if not raw:
        return
    # Where a command appears: quoted text (a -m message, a --body) does not count.
    code = QUOTED.sub("''", raw)
    # What its arguments say: quote characters removed, so 'main' still reads as main.
    args_view = raw.replace("'", " ").replace('"', " ")
    cwd = payload.get("cwd")
    rule = " main changes only through a PR the author merges."

    if PR_MERGE.search(code):
        block("Agents do not merge pull requests." + rule)

    if READS_ENV.search(args_view):
        block("Never read, print or copy .env files; secrets are loaded at runtime.")

    if PUSH.search(code):
        on_main = current_branch(cwd) == "main"
        for match in PUSH.finditer(args_view):
            args = match.group(1)
            if FORCE.search(args):
                block("Force-push is not allowed." + rule)
            if PUSH_ALL.search(args):
                block("Pushing all refs is not allowed." + rule)
            if TARGETS_MAIN.search(args):
                block("Pushing to main is not allowed." + rule)
        if on_main:
            block("main is checked out; switch to a working branch before pushing." + rule)

    if HISTORY_ON_MAIN.search(code) and current_branch(cwd) == "main":
        block("main is checked out; create a working branch first." + rule)


if __name__ == "__main__":
    main()
