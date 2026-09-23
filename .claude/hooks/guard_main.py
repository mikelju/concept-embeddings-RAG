"""PreToolUse hook: keep agents off main and away from .env.

Agents may commit and push working branches freely; main changes only through a PR the author
merges. Blocks (exit code 2) any shell command that would push to main, push everything,
force-push, merge a PR, change history while main is checked out, or read a .env file.
gh-axi is allowlisted: only the read subcommands in GH_AXI_READS may run, everything else is
blocked. It is a local guard, not a substitute for branch protection on the remote.

Commands are matched on normalized views with quote and backtick characters removed, so quoting
a word ("gh-axi", 'main', g"it") does not hide it. Backslashes are checked two ways: removed
(shell escapes such as gh\\-axi) and turned into "/" (Windows paths such as .\\bin\\gh-axi.cmd).
Heredoc and here-string bodies are matched like any other text, because sh <<EOF or
Invoke-Expression @'...'@ executes them. The price is that any inline text that mentions a
blocked command is blocked too: pass commit messages and PR bodies through files
(-F <file>, --body-file <file>). Standard library only; ASCII output only.
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
GH = r"(?<![\w-])gh(?:\.exe)?\s+"
GH_MERGE = re.compile(GH + r"pr\s+merge\b|" + GH + r"api\b[^;&|\n]*/merge\b")
# Any gh-axi invocation: direct, npx gh-axi@<version>, gh-axi.cmd/.js/.ps1, or a path to it.
GH_AXI = re.compile(r"(?<![\w-])gh-axi(?:\.\w+)?(?:@[^\s;&|()]*)?(?![\w-])")
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


def gh_axi_violation(command: str) -> str | None:
    """Return the offending gh-axi call if any invocation is not an allowlisted read."""
    for match in GH_AXI.finditer(command):
        rest = SEGMENT_END.split(command[match.end() :], maxsplit=1)[0].split()
        if tuple(rest[:2]) not in GH_AXI_READS:
            return " ".join(["gh-axi", *rest[:2]]).strip()
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
    # Paths keep their backslashes here, so C:\repo\.env still reads as .env.
    env_view = raw.replace("'", " ").replace('"', " ")
    cwd = payload.get("cwd")
    rule = " main changes only through a PR the author merges."

    for command in views:
        offending = gh_axi_violation(command)
        if offending is not None:
            allowed = ", ".join(" ".join(pair) for pair in sorted(GH_AXI_READS))
            block(f"'{offending}' is not an allowed gh-axi read ({allowed}). Writes use gh.")

        if GH_MERGE.search(command):
            block("Agents do not merge pull requests." + rule)

    if READS_ENV.search(env_view):
        block("Never read, print or copy .env files; secrets are loaded at runtime.")

    for command in views:
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
