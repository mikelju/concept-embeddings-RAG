"""PreToolUse hook: keep agents off main and away from .env.

Agents may commit and push working branches freely; main changes only through a PR the author
merges. Blocks (exit code 2) any shell command that would push to main, push everything,
force-push, change history while main is checked out, or read a .env file. GitHub CLIs are
allowlisted, not denylisted: gh may run only the subcommands in GH_ALLOWED and gh-axi only the
reads in GH_AXI_READS, written directly after the program name. Everything else - pr merge,
api (REST or GraphQL), stack, unknown subcommands, flags before the subcommand - is blocked.

Commands are matched on normalized views with quote and backtick characters removed, so quoting
a word ("gh-axi", 'main', .e'n'v) does not hide it. Backslashes are checked two ways: removed
(shell escapes such as gh\\-axi) and turned into "/" (Windows paths such as .\\bin\\gh-axi.cmd).
Heredoc and here-string bodies are matched like any other text, because sh <<EOF or
Invoke-Expression @'...'@ executes them. The price is that any inline text that mentions a
blocked command is blocked too: pass commit messages and PR bodies through files
(-F <file>, --body-file <file>).

Limits: this is a best-effort local guard, not a shell interpreter. Indirection through
variables, globs, command substitution or eval is out of its reach. The server-side barriers are
branch protection on main and the permission rules in .claude/settings.json.
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

# Any gh invocation (gh, gh.exe, a path to it) and any gh-axi invocation (direct,
# npx gh-axi@<version>, gh-axi.cmd/.js/.ps1, a path to it).
GH = re.compile(r"(?<![\w.-])gh(?:\.\w+)?(?![\w.-])")
GH_AXI = re.compile(r"(?<![\w-])gh-axi(?:\.\w+)?(?:@[^\s;&|()]*)?(?![\w-])")
GH_ALLOWED = {
    ("pr", "create"),
    ("pr", "edit"),
    ("pr", "view"),
    ("pr", "list"),
    ("pr", "checks"),
    ("pr", "diff"),
    ("run", "list"),
    ("run", "view"),
    ("repo", "view"),
    ("auth", "status"),
}
GH_AXI_READS = {
    ("pr", "view"),
    ("pr", "checks"),
    ("pr", "list"),
    ("run", "list"),
    ("run", "view"),
}
SEGMENT_END = re.compile(r"[;&|\n()]")
FORCE = re.compile(r"(^|\s)(-f|--force|--force-with-lease|--force-if-includes)(\s|=|$)|\s\+\S")
PUSH_ALL = re.compile(r"(^|\s)(--all|--mirror)(\s|=|$)")
TARGETS_MAIN = re.compile(r"(^|\s|:)(refs/heads/)?main(\s|$)")
READS_ENV = re.compile(r"(^|[\s/\\=])\.env(\.(?!example\b)[\w.-]+)?(?=$|[\s;|&)])")
QUOTE_CHARS = re.compile(r"['\"`]")


def normalized_views(raw: str) -> tuple[str, str]:
    """Quotes and backticks removed; backslashes removed in one view, made "/" in the other."""
    base = QUOTE_CHARS.sub("", raw)
    return base.replace("\\", ""), base.replace("\\", "/")


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


def call_violation(
    command: str, program: re.Pattern[str], allowed: set[tuple[str, str]], name: str
) -> str | None:
    """Return the offending call if any invocation of program is not an allowlisted subcommand.

    The subcommand must be the two words right after the program name, so flags placed before
    it (gh -R o/r pr merge, gh pr --repo o/r merge) are rejected rather than parsed.
    """
    for match in program.finditer(command):
        rest = SEGMENT_END.split(command[match.end() :], maxsplit=1)[0].split()
        if tuple(rest[:2]) not in allowed:
            return " ".join([name, *rest[:2]]).strip()
    return None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return
    raw = str(payload.get("tool_input", {}).get("command", ""))
    if not raw:
        return
    views = normalized_views(raw)
    cwd = payload.get("cwd")
    rule = " main changes only through a PR the author merges."

    for command in views:
        offending = call_violation(command, GH_AXI, GH_AXI_READS, "gh-axi")
        if offending is not None:
            allowed = ", ".join(" ".join(pair) for pair in sorted(GH_AXI_READS))
            block(f"'{offending}' is not an allowed gh-axi read ({allowed}). Writes use gh.")

        offending = call_violation(command, GH, GH_ALLOWED, "gh")
        if offending is not None:
            allowed = ", ".join(" ".join(pair) for pair in sorted(GH_ALLOWED))
            block(f"'{offending}' is not an allowed gh call ({allowed})." + rule)

        if READS_ENV.search(command):
            block("Never read, print or copy .env files; secrets are loaded at runtime.")

        if PUSH.search(command):
            for match in PUSH.finditer(command):
                args = match.group(1)
                if FORCE.search(args):
                    block("Force-push is not allowed." + rule)
                if PUSH_ALL.search(args):
                    block("Pushing all refs is not allowed." + rule)
                if TARGETS_MAIN.search(args):
                    block("Pushing to main is not allowed." + rule)
            if current_branch(cwd) == "main":
                block("main is checked out; switch to a working branch before pushing." + rule)

        if HISTORY_ON_MAIN.search(command) and current_branch(cwd) == "main":
            block("main is checked out; create a working branch first." + rule)


if __name__ == "__main__":
    main()
