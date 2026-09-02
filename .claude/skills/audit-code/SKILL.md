---
name: audit-code
description: Professional security code audit focused on Python (primary) and React/frontend (secondary). Use when the user asks to audit, review security, hunt vulnerabilities, check for bugs/CVEs, harden code, or prepare code for production. Triggers on phrases like "audita", "audit", "revisa seguridad", "check vulnerabilities", "OWASP", "security review", "pentest", "harden", "revisión de seguridad".
---

# Audit Code — Security Review Skill

Professional, defensive-minded code auditor. **Priority #1 is cybersecurity**: zero tolerance for unaddressed findings. Every suspicious pattern gets flagged; proof-of-safety is the auditor's burden, not the reverse.

## Philosophy

1. **Assume breach.** Treat every input as hostile: env vars, config files, filesystem paths, DB results, third-party APIs, and internal callers.
2. **Defense in depth.** A single control is a single point of failure. Layer validation, authorization, logging, and monitoring.
3. **Fail closed.** When uncertain, reject. Default-deny beats default-allow every time.
4. **False positives beat false negatives.** If a pattern *might* be exploitable, report it as HIGH until disproven.
5. **Trust nothing, verify everything.** Do not take a comment's word that code is safe — read the code and prove it.
6. **Minimize attack surface.** Unused code, dead imports, disabled controls, TODOs ("fix later"), and debug endpoints are all findings.

## When this skill runs

Trigger on any of these requests (Spanish or English):
- "audita este código" / "audit this code"
- "revisa la seguridad" / "security review"
- "busca vulnerabilidades" / "find vulnerabilities"
- "check OWASP" / "revisa contra OWASP"
- "harden this" / "prepara esto para producción"
- "pentest" / "análisis de seguridad"
- User mentions a specific vulnerability class ("check for SQL injection", "XSS?", etc.)

When triggered, load the full workflow below. Do not skip phases.

---

## Audit Workflow — Five Phases

### Phase 1 — Scope & Context

Before reading a single line of application code, answer:

1. **What are we auditing?** Ask the user if unclear:
   - Full codebase, single module, recent diff, PR, or specific file?
   - What's the threat model? (public web app, internal tool, CLI, desktop, data pipeline…)
   - Any compliance target? (GDPR, PCI-DSS, HIPAA, SOC2)
   - What data does it handle? (PII, payments, health, credentials, arbitrary user files…)

2. **Identify stack and entry points.** Read in this order:
   - `pyproject.toml` / `requirements.txt` / `package.json` / `Pipfile` / `poetry.lock`
   - `Dockerfile` / `docker-compose.yml` / CI config (`.github/workflows/`, `.gitlab-ci.yml`)
   - Main entry points: `__main__.py`, `app.py`, `main.py`, `manage.py`, Flask/FastAPI/Django routes, `index.tsx`, API route files
   - Configuration files: `.env.example`, `config.yaml`, `settings.py`

3. **Map trust boundaries.** Sketch (mentally or in the report) where data crosses:
   - User → server (request parsing, auth middleware)
   - Server → DB (ORM/query layer)
   - Server → external APIs / filesystem / shell / child processes
   - Server → rendered output (templates, JSON responses)

Write these down in the report's "Scope" section.

### Phase 2 — Automated Scanning

Run whatever is installed. If a tool isn't installed, note it in the report as a recommendation — do NOT install tools without asking. Expected tools (see `references/tools-integration.md` for full commands):

| Tool | Purpose | Command |
|------|---------|---------|
| `bandit` | Python SAST (AST-based) | `bandit -r src/ -ll -f json` |
| `pip-audit` | Python dependency CVEs | `pip-audit --desc` |
| `safety` | Alt. Python dep scanner | `safety check --json` |
| `semgrep` | Multi-language SAST with rule packs | `semgrep --config=auto src/` |
| `ruff` (security rules) | S-category rules | `ruff check --select=S src/` |
| `npm audit` | JS/TS dependency CVEs | `npm audit --json` |
| `eslint-plugin-security` | JS/TS SAST | via `.eslintrc` |
| `gitleaks` / `trufflehog` | Secret scanning | `gitleaks detect --source . --no-git` |

**Rules for automated tools:**
- Never suppress a finding without a documented reason.
- Treat tool warnings as starting points, not gospel — confirm each by reading the code.
- Aggregate results in the final report, deduplicated and ranked by severity.
- If a tool reports 0 findings, that is **not** proof of absence — continue to Phase 3.

### Phase 3 — Manual Review

Automated tools miss **logic flaws, auth bypasses, business-rule violations, race conditions, and context-dependent vulnerabilities**. This is where human audit value lives.

For each file in scope, walk through it asking:

1. **Inputs** — Where does data enter? Is it validated, typed, length-bounded, and charset-restricted? Is it *trusted* at the point of use? (Trust is not transitive.)
2. **Processing** — Are there dangerous sinks? (`eval`, `exec`, `subprocess` with `shell=True`, `pickle.loads`, `yaml.load`, raw SQL, `os.system`, `Runtime.exec`, template rendering with user input, `dangerouslySetInnerHTML`, etc.)
3. **Authorization** — Every protected action: is there a check? Is it before or after side effects? Can it be bypassed by ordering, race, or parameter tampering (IDOR)?
4. **Output** — Is data escaped for its context (HTML, URL, shell, SQL, JSON, log)? Are error messages leaking stack traces, paths, or secrets?
5. **Crypto** — MD5/SHA1 for anything security-relevant? `random` instead of `secrets`? Hardcoded IVs/keys? TLS verification disabled? Weak ciphers?
6. **Secrets** — Any hardcoded tokens, keys, passwords, connection strings? Any printed / logged secrets? `.env` committed?
7. **Session & Auth** — Token storage location, expiry, rotation, revocation. CSRF on state-changing routes. Rate limiting on auth endpoints. Password hashing algorithm (must be bcrypt/argon2/scrypt, never MD5/SHA256).
8. **Dependencies** — Any unmaintained packages? Packages with known CVEs? Typosquatting candidates? Install hooks? Unexpected native code?
9. **File system** — Path traversal in reads/writes/uploads? Zip-slip in archive extraction? Symlink attacks? Insufficient permissions on sensitive files?
10. **Concurrency** — Shared mutable state without locks? TOCTOU? Double-spend / double-submit?

**Load the language-specific checklist**:
- **Python** → always read `references/python-security.md` in full.
- **React/JS/TS** → read `references/react-security.md`.
- **Cross-cutting** → `references/owasp-top-10.md` (mapping to OWASP 2021 categories) and `references/secrets-patterns.md` (hardcoded credentials).

### Phase 4 — Exploit Reasoning

For each finding, reason about **exploitability**, not just presence:
- What does an attacker need? (network position, credentials, a crafted input)
- What do they gain? (RCE, data read, data write, DoS, auth bypass, privilege escalation)
- How noisy is the attack? (would it show in logs?)

If feasible, sketch a minimal PoC in the finding (NEVER run exploits — describe only). This helps the developer understand the severity.

**Do NOT write or execute malicious payloads against any system.** The audit is analytical and educational only.

### Phase 5 — Report

Produce the report using the format in `references/report-template.md`. Non-negotiable elements:
- Executive summary (3-5 sentences, non-technical)
- Findings table (severity, title, file:line, CWE, OWASP category)
- Detailed finding cards (one per issue): description, impact, reproduction/PoC sketch, fix (with code), references
- Defense-in-depth recommendations beyond specific findings
- Coverage statement: what was reviewed, what was skipped, and why

Use markdown with clickable file references (`[file.py:42](path/to/file.py#L42)`).

---

## Severity Rubric

Use this five-level scale. When in doubt, round **up**.

| Severity | CVSS-ish | Criteria |
|----------|----------|----------|
| **Critical** | 9.0-10 | Unauthenticated RCE, auth bypass affecting all users, exposed admin credentials, mass data exfiltration possible. Fix before the next commit. |
| **High** | 7.0-8.9 | Authenticated RCE, SQLi/XSS with significant impact, broken access control (IDOR), hardcoded secrets for production services, weak crypto protecting sensitive data. Fix before release. |
| **Medium** | 4.0-6.9 | Limited-scope XSS, information disclosure (stack traces, version headers), CSRF on low-impact endpoints, missing rate limiting, denial-of-service via large input. Fix in current sprint. |
| **Low** | 0.1-3.9 | Defense-in-depth improvements, missing security headers, verbose error messages, weak password policies (if other controls mitigate), logging gaps. Track and fix. |
| **Info** | — | Observations, code quality issues with marginal security impact, hardening suggestions. Document. |

Also tag each finding with:
- **CWE** (Common Weakness Enumeration) — use the most specific one (e.g., CWE-89 for SQLi, not CWE-20).
- **OWASP Top 10 2021 category** (A01…A10) — see `references/owasp-top-10.md`.
- **Confidence** (Confirmed / Likely / Possible) — distinguish verified issues from suspicions.

---

## Anti-patterns — things NOT to do during an audit

- **Do not install new packages** (`pip install`, `npm install`) without asking the user. Run only what's already available.
- **Do not modify application code** during audit — you are reading, not editing. If the user asks for fixes after the report, switch modes explicitly.
- **Do not execute user input against real systems** to "test" vulnerabilities. Describe the PoC; do not run it.
- **Do not skip the manual phase** because tools found nothing.
- **Do not downplay findings** to keep the report short. Every issue is reported, even if trivial — let the user prioritize.
- **Do not invent CWE/CVE numbers.** If you're unsure, say "CWE-unspecified, likely in the CWE-xxx family" and explain.
- **Do not leak secrets into the report.** If a finding involves a hardcoded credential, redact it: `API_KEY = "sk-••••••••" (redacted)`.
- **Do not claim a codebase is "secure"** at the end. Use "no high-severity findings under the reviewed scope" — absence of evidence is not evidence of absence.
- **Do not exit the audit without running the coverage check** (below).

---

## Coverage Self-Check (MANDATORY before closing the audit)

Before producing the final report, verify out loud:

- [ ] Did I read every entry point (CLI args, HTTP routes, message handlers)?
- [ ] Did I follow every external input to its sinks?
- [ ] Did I check auth and authz on every protected action?
- [ ] Did I review the dependency manifest for known CVEs?
- [ ] Did I check for hardcoded secrets (regex scan + visual review)?
- [ ] Did I check crypto use (hashing, symmetric, asymmetric, random)?
- [ ] Did I check error handling and logging for data leakage?
- [ ] Did I review file-system and subprocess usage?
- [ ] Did I check for the OWASP Top 10 categories applicable to this stack?
- [ ] For Python: did I read `references/python-security.md` end-to-end?
- [ ] For React/JS/TS: did I read `references/react-security.md` end-to-end?

Any unchecked item → either address it or explicitly call it out as "Out of scope" in the coverage section of the report.

---

## MCP Integration (future)

This skill is **MCP-ready**. When MCP servers for `pip-audit`, `bandit`, `semgrep`, or `npm audit` become available, they slot into **Phase 2 (Automated Scanning)** without changing the workflow. Current behavior: invoke the tools via `Bash` if installed, otherwise note in the report that a scan could not be run. See `references/tools-integration.md` for the planned MCP wiring.

---

## Output Contract

The final artifact is a markdown report (either inline in chat or written to `docs/security/audit-YYYY-MM-DD.md` if the user has a docs folder). Structure is defined in `references/report-template.md`. Do not deviate.

---

## Quality bar

A good audit report:
- Is **specific** — every finding names a file, a line, and a concrete fix.
- Is **actionable** — the developer can close the finding with the information given.
- Is **calibrated** — severities match impact, not drama.
- Is **complete** — coverage section is honest about what wasn't reviewed.
- Is **educational** — each finding briefly explains *why* it's dangerous, not just what rule it breaks.

A bad audit report:
- Dumps tool output without analysis.
- Says "this is secure" without qualification.
- Rates every issue as "HIGH" to seem thorough.
- Misses auth/authz logic flaws because they require reading, not scanning.
