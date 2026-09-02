# Tools Integration — Scanners, Bash Invocation, and MCP Roadmap

Reference for the `audit-code` skill **Phase 2 — Automated Scanning**. Explains:
1. Which tools to run, in what order.
2. Exact commands (Bash).
3. How to interpret output.
4. Planned MCP integration when servers become available.

---

## Ground rules

1. **Never install tools during an audit.** If a tool isn't installed, note it as a recommendation in the report's §5 Coverage — do NOT run `pip install bandit` unless the user explicitly authorizes it.
2. **Probe before invoking.** Use `command -v toolname` (bash) or `python -c "import bandit"` before running. Absence isn't an error — it's a finding for §5.
3. **JSON output where possible.** Easier to parse and to quote in the report. Fall back to human-readable only if JSON isn't supported.
4. **Never suppress findings silently.** If a tool flags something, either the report addresses it or the report says why it was dismissed.
5. **Tool output is a starting point, not the finding itself.** Every tool hit is confirmed by reading the flagged code before it goes in the report.

---

## Tool catalog — Python

### bandit — Python SAST (AST-based)

**Purpose**: AST-level scan for known insecure patterns (eval, pickle, weak crypto, shell=True, hardcoded passwords heuristics, etc.).

**Probe**:
```bash
command -v bandit
```

**Invoke**:
```bash
# Text output (for humans)
bandit -r src/ -ll -i

# JSON output (for the skill to parse)
bandit -r src/ -ll -i -f json -o bandit-report.json

# Skip known-false-positive IDs:
bandit -r src/ -ll -i --skip B101  # B101 = assert_used (noisy)

# With a config file to tune severity/confidence:
bandit -r src/ -c pyproject.toml
```

**Flags used above**:
- `-r` recurse
- `-l` low severity threshold (`-ll` = medium+)
- `-i` include confidence filter (can also pair with `-iii` for high-confidence only)
- `-f json` JSON formatter

**Common findings to confirm manually** (bandit test IDs):
| ID | Meaning | Action |
|----|---------|--------|
| B102 | `exec` used | Confirm; Critical. |
| B103 | Insecure file permission (`0o777`) | Confirm; usually High. |
| B105-B107 | Hardcoded password string / function arg / default | Confirm, redact, rotate. |
| B110 | `try/except/pass` | Low; usually code-quality but can hide security errors. |
| B301-B304 | pickle / cPickle / dill / shelve | Confirm; Critical if untrusted data. |
| B306 | `mktemp` | Low-Medium. |
| B307 | `eval` | Critical. |
| B308 | `mark_safe` | Confirm; High if on user data. |
| B310 | `urllib.urlopen` (potential file scheme) | SSRF risk. |
| B311 | `random` module for crypto | High. |
| B313-B320 | XML parsing without `defusedxml` | XXE — High. |
| B321 | FTP usage | Medium. |
| B322 | `input()` in Py2 | Critical on Py2. |
| B324 | `hashlib` weak algorithms | High for security uses. |
| B325 | `tempnam` | Medium. |
| B501 | `requests` with `verify=False` | High. |
| B502-B504 | SSL weak versions / bad defaults | High. |
| B505 | Weak RSA/DSA key size | High. |
| B506 | `yaml.load` without SafeLoader | Critical. |
| B601 | `paramiko` shell with untrusted data | High. |
| B602 | `subprocess` with `shell=True` | Critical if tainted. |
| B603-B604 | `subprocess` w/o `shell=True`, but with tainted args | Confirm. |
| B608 | SQL string construction | SQLi — High/Critical. |
| B609 | Wildcard in Linux commands | Medium. |
| B610-B611 | Django `.extra()` / `.raw()` | SQLi — High. |
| B701 | Jinja2 autoescape disabled | High. |
| B703 | Django `mark_safe` | High. |

### pip-audit — Python dependency CVE scan

**Purpose**: Match installed (or declared) dependency versions against CVE databases (PyPI Advisory DB, OSV).

**Probe**:
```bash
command -v pip-audit
```

**Invoke**:
```bash
# Audit installed environment
pip-audit --desc

# Audit requirements file without installing
pip-audit -r requirements.txt --desc

# JSON for parsing
pip-audit -r requirements.txt -f json -o pip-audit.json

# Audit from pyproject.toml (PEP 621) via an installed env
pip-audit --desc

# Fix automatically (DO NOT run during audit — recommend to user)
# pip-audit --fix
```

**Interpretation**:
- Each finding has a package, installed version, vulnerable range, fix version, CVE/GHSA ID.
- Check for: exploitable-in-this-project assessment (not every CVE is reachable from this code's usage). Note this in the finding.

### safety — alt. Python dependency scanner

**Probe**: `command -v safety`

**Invoke**:
```bash
safety check --json
safety check -r requirements.txt --json
```

Used as a complement to `pip-audit`; each may catch advisories the other misses.

### semgrep — multi-language SAST with rule packs

**Purpose**: Pattern-based detection across Python, JS/TS, Java, Go, and many more.

**Probe**: `command -v semgrep`

**Invoke**:
```bash
# Auto-config (recommended rule pack for detected languages)
semgrep --config=auto src/

# Focused security rulesets
semgrep --config=p/python --config=p/security-audit --config=p/secrets src/

# JSON output
semgrep --config=auto --json --output semgrep.json src/
```

**Relevant rule packs**:
- `p/security-audit`
- `p/owasp-top-ten`
- `p/cwe-top-25`
- `p/python`, `p/flask`, `p/django`, `p/fastapi`
- `p/react`, `p/javascript`, `p/typescript`
- `p/secrets`
- `p/dockerfile`

### ruff — Python linter with security rules

**Purpose**: Fast linter; `S` category ports Bandit-equivalent checks.

**Probe**: `command -v ruff`

**Invoke**:
```bash
# Run only the security (Bandit-port) category
ruff check --select=S src/

# Include pylint + flake8-security + bandit-ish
ruff check --select=S,B,PL,E9 src/
```

Useful because many projects already run ruff — findings are easy to actionable.

### detect-secrets / gitleaks / trufflehog — secret scanners

**Probe each**:
```bash
command -v detect-secrets
command -v gitleaks
command -v trufflehog
```

**Invoke**:
```bash
# detect-secrets
detect-secrets scan --all-files > .secrets.baseline
detect-secrets audit .secrets.baseline

# gitleaks — scans working dir AND git history
gitleaks detect --source . --no-git --report-format json --report-path gitleaks.json
gitleaks detect --source . --report-format json --report-path gitleaks-history.json

# trufflehog — finds verified secrets (attempts to authenticate)
trufflehog filesystem --json .
trufflehog git --json .
```

**Rules**:
- Run against the working tree AND the git history — secrets removed in HEAD are still recoverable.
- If any scanner is unavailable, note it in §5 Coverage as a recommendation.

---

## Tool catalog — JS/TS

### npm audit / pnpm audit / yarn audit

**Probe**:
```bash
test -f package-lock.json && command -v npm
test -f pnpm-lock.yaml && command -v pnpm
test -f yarn.lock && command -v yarn
```

**Invoke**:
```bash
npm audit --json
npm audit --audit-level=high
pnpm audit --json
yarn audit --json
```

### eslint-plugin-security

If the project has ESLint configured, check `.eslintrc*` for `eslint-plugin-security`. If present:
```bash
npx eslint src/
```
If not present, recommend adding it in §4 of the report.

### Snyk / Socket

If the project uses Snyk or Socket:
```bash
snyk test
socket scan
```
Mention only if already configured in the project.

### semgrep for JS/TS

See above. `--config=p/javascript`, `p/typescript`, `p/react`, `p/nextjs`.

---

## Tool catalog — Other

### hadolint — Dockerfile linting
```bash
hadolint Dockerfile
```

### checkov — IaC (Terraform, K8s, CloudFormation)
```bash
checkov -d .
```

### dockle / trivy — container image scanning (when images are in scope)
```bash
trivy fs .
trivy image myapp:latest
```

### osv-scanner — multi-ecosystem dependency CVE scan (Google)
```bash
osv-scanner -r .
```

---

## Putting it together — the Phase 2 recipe

```bash
# 1. Probe what's available. For each hit, run it.

for tool in bandit pip-audit safety semgrep ruff detect-secrets gitleaks trufflehog; do
  command -v "$tool" && echo "[present] $tool" || echo "[absent]  $tool"
done

# 2. Python stack
command -v bandit && bandit -r src/ -ll -f json -o .audit/bandit.json
command -v pip-audit && pip-audit -r requirements.txt -f json -o .audit/pip-audit.json
command -v ruff && ruff check --select=S src/ --output-format=json > .audit/ruff-s.json

# 3. JS/TS stack
test -f package-lock.json && npm audit --json > .audit/npm-audit.json

# 4. Secrets
command -v gitleaks && gitleaks detect --source . --report-format json --report-path .audit/gitleaks.json

# 5. Cross-language SAST if available
command -v semgrep && semgrep --config=auto --json --output .audit/semgrep.json src/
```

Then for each tool's JSON file, summarize and confirm each finding manually before writing it into the report.

---

## MCP Roadmap

The skill is designed so that MCP servers slot into Phase 2 without any workflow change. When/if MCP servers are added:

| Planned MCP server | Replaces this Bash invocation | Benefit |
|--------------------|-------------------------------|---------|
| `mcp-pip-audit`    | `pip-audit -f json`           | Structured tool call, no subprocess parsing; live CVE database access without local install. |
| `mcp-bandit`       | `bandit -r src/ -f json`      | Same; can call per-file for speed. |
| `mcp-semgrep`      | `semgrep --config=auto`       | Remote rule packs; no local semgrep install needed. |
| `mcp-gitleaks`     | `gitleaks detect`             | Same. |
| `mcp-npm-audit`    | `npm audit --json`            | Same. |
| `mcp-osv-scanner`  | `osv-scanner -r .`            | Cross-ecosystem. |
| `mcp-cve-lookup`   | (no bash equivalent — new capability) | On-demand CVE lookup for any package/version during manual review. |

### When MCP tools are configured

1. Prefer the MCP tool over the Bash equivalent (structured input/output, no shell quoting, no environment requirement on the user's machine).
2. The audit workflow and report format **do not change** — MCP is an invocation-layer detail.
3. If an MCP tool is configured but fails, fall back to the Bash equivalent (if installed) or note as a gap.

### How to wire a new MCP server later

User-side work (not part of this skill):
1. Add the server to `~/.claude.json` under `mcpServers`.
2. Restart Claude Code so it discovers the server.
3. Verify tools appear in the conversation (they will load automatically).

Skill-side work:
1. Update this file's "MCP Roadmap" table with the exact tool name exposed by the server.
2. Update `SKILL.md` Phase 2 if a tool offers capabilities that weren't reachable via Bash.

No code changes — the skill is purely instructional. The workflow absorbs new tools as they arrive.

---

## What to put in §5 Coverage of the report

Always record:
- Which tools were probed.
- Which tools ran successfully (with version, if trivially available).
- Which tools were absent — these become DEF-NNN recommendations.
- Raw findings count vs. confirmed-in-report count (if they differ, explain why).

Example block:
```markdown
### Tools run

| Tool | Status | Version | Findings | Confirmed |
|------|--------|---------|----------|-----------|
| bandit | Run | 1.7.9 | 7 | 4 (3 dismissed as FP) |
| pip-audit | Run | 2.7.3 | 2 | 2 |
| semgrep | Absent | — | — | Recommended (see DEF-002) |
| gitleaks | Absent | — | — | Recommended (see DEF-001) |
| manual review | Completed | — | 6 | 6 |
```
