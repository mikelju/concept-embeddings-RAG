# Audit Report — Output Template

The `audit-code` skill produces one final artifact: a security audit report in Markdown. This file defines the exact structure. **Follow it precisely.**

Output options:
- **Default** — Write to `docs/security/audit-YYYY-MM-DD.md` if `docs/` exists. Create the `security/` folder if missing.
- **Fallback** — Print the report inline in chat if no `docs/` folder exists or the user prefers inline.
- Ask the user which they want if it's ambiguous.

---

## Template (copy-paste verbatim, fill placeholders)

```markdown
# Security Audit — {{PROJECT NAME}}

**Date:** YYYY-MM-DD
**Auditor:** audit-code skill (Claude Code)
**Scope:** {{files / modules / commit range reviewed}}
**Threat model:** {{public web / internal tool / desktop / CLI / data pipeline / ...}}
**Stack:** {{Python 3.11, Flask 3.x, SQLAlchemy 2.x, …}}

---

## 1. Executive Summary

{{3-5 sentences, non-technical. State overall posture, count of findings by severity, top two or three things to fix first. Do NOT say "the code is secure"; use "no critical findings under the reviewed scope" or similar.}}

**Findings by severity:**

| Severity | Count |
|----------|-------|
| Critical | {{N}} |
| High     | {{N}} |
| Medium   | {{N}} |
| Low      | {{N}} |
| Info     | {{N}} |

---

## 2. Top Priorities

Ranked list of what to fix first (pull from Critical/High findings). Two-line summary each, linking to the detailed finding below.

1. **[{{ID}}] {{short title}}** — {{one-line consequence}}. → §3.{{N}}
2. **[{{ID}}] {{short title}}** — {{one-line consequence}}. → §3.{{N}}
3. ...

---

## 3. Findings

Each finding is a numbered subsection with the card structure below. Order by severity (Critical → Info), then by file path.

### 3.1 [SEC-001] {{Short, specific title}} — **Critical**

- **File:** [path/to/file.py:42](path/to/file.py#L42)
- **CWE:** CWE-89 (SQL Injection)
- **OWASP:** A03 — Injection
- **Confidence:** Confirmed / Likely / Possible
- **Tool:** manual review / bandit / semgrep / pip-audit / ...

**Description**

{{2-4 sentences. What is the vulnerable pattern, in plain terms? Why does it exist?}}

**Vulnerable code**

```python
# path/to/file.py:42
def get_user(name):
    return db.execute(f"SELECT * FROM users WHERE name = '{name}'")
```

**Impact**

{{What happens if exploited? Be specific: data read, data modification, RCE, DoS, auth bypass, scope of affected users/data.}}

**Reproduction / PoC sketch** (do NOT run)

{{Describe the input or steps. Example:}}
> Sending `name=' OR '1'='1` to the `/users?name=` endpoint would return all user rows, bypassing the intended filter.

**Remediation**

```python
# path/to/file.py:42 — fixed
def get_user(name: str):
    return db.execute(
        text("SELECT * FROM users WHERE name = :name"),
        {"name": name}
    )
```

{{Optional: Explain why the fix works and any defense-in-depth to add.}}

**References**
- [CWE-89](https://cwe.mitre.org/data/definitions/89.html)
- [OWASP SQL Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html)
- {{Stack-specific docs link if relevant}}

---

### 3.2 [SEC-002] ...

{{repeat}}

---

## 4. Defense-in-Depth Recommendations

Issues that are not specific bugs but harden the system. These are NOT findings against the current code — they're architectural improvements.

- **[DEF-001] Add pre-commit secret scanning.** Install `gitleaks` or `detect-secrets` as a pre-commit hook to prevent hardcoded secrets reaching the repo.
- **[DEF-002] Add CI dependency scanning.** Wire `pip-audit` / `npm audit` into CI; fail the build on High+ CVEs.
- **[DEF-003] Centralize authorization checks.** Current code repeats `if user != obj.owner` in multiple handlers — extract to a decorator or middleware to reduce the chance of missing one.
- **[DEF-004] Structured logging with redaction.** Move from `logger.info(f"…")` to structured fields + a redaction filter for known-sensitive keys (`password`, `token`, `ssn`).
- ...

---

## 5. Coverage

### What was reviewed
- `src/reme/mail/` — all files, full read
- `src/reme/web/` — all files, full read
- `pyproject.toml`, `config.example.yaml`
- Dependency versions checked against `pip-audit` database as of {{DATE}}

### What was NOT reviewed (and why)
- `docs/` — documentation only, no runtime code
- `tests/` — reviewed at a spot-check level; tests for security-critical paths were sampled but not fully audited
- Frontend (not present in this repo)
- Infrastructure-as-code / Terraform / K8s manifests (out of scope — no such files)

### Tools run
| Tool | Status | Findings |
|------|--------|----------|
| bandit | Run (`bandit -r src/ -ll`) | 3 low, 1 medium |
| pip-audit | Run | 0 CVEs |
| semgrep | Not installed | Recommended |
| gitleaks | Not installed | Recommended |
| manual review | Completed | 2 high, 4 medium, 3 low |

### Methodology
- Phase 1 — Scope & Context: {{brief notes}}
- Phase 2 — Automated Scanning: {{which tools}}
- Phase 3 — Manual Review: referenced `audit-code/references/python-security.md` and `owasp-top-10.md` in full
- Phase 4 — Exploit Reasoning: PoC sketches drafted, none executed
- Phase 5 — Report: this document

### Caveats
- This audit reflects the code at commit `{{short SHA}}`. Subsequent changes are not reviewed.
- Runtime behaviour (e.g., OS permissions, cloud IAM, network policies) is inferred from code and docs only; not verified against a live deployment.
- Business logic flaws are addressed to the extent the domain is understood from code comments and docstrings — recommend a second pass with a domain expert.

---

## 6. Next Steps

Suggested sequence:
1. Triage Critical and High findings with the team this week.
2. Fix Critical within 24-72h; rotate any exposed secrets immediately.
3. Fix High before the next release.
4. Track Medium/Low in the backlog with target dates.
5. Re-audit after fixes land, or add the CI scans from §4 to catch regressions automatically.

---

*Report generated by the `audit-code` skill. To re-run: invoke the skill on the same scope and compare reports.*
```

---

## Writing rules

- **Every finding gets an ID** (`SEC-001`, `SEC-002`, …). Use for cross-referencing and ticket creation.
- **Every finding names a file and line.** No vague "somewhere in the auth module."
- **Every finding has a fix.** Not "consider reviewing this" — show the corrected code.
- **Every code block is labeled with the file path in a comment**, so the reader can locate it.
- **Use markdown links** for file references: `[file.py:42](path/to/file.py#L42)`.
- **Redact secrets** — `sk-proj-abcd••••••••••`, never the full value.
- **No emojis** in the report body (they break in some terminals and PDFs).
- **Keep the exec summary executive.** No jargon, no CVE numbers, no CWE IDs. Those go in the findings.
- **Order by severity**, then by file path — makes the report predictable.
- **Use "Confidence" honestly.** "Possible" findings still get reported, but flagged as such so the reader knows.

## Severity ID scheme

- `SEC-NNN` — security finding (main body)
- `DEF-NNN` — defense-in-depth recommendation (not a bug)
- `OBS-NNN` — observation / code-quality note

Keep IDs unique across reports for a given project (use the date prefix if you need to regenerate).
