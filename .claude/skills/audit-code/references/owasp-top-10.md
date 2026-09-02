# OWASP Top 10 (2021) — Mapping for Code Audits

Reference for the `audit-code` skill. When a finding is reported, tag it with the applicable OWASP category. When auditing, use this as a checklist of attack classes to evaluate.

The 2021 edition is the latest *released* Top 10 at skill authoring time. When OWASP publishes a newer Top 10, update this file.

---

## A01 — Broken Access Control

**What to check:**
- Every route/handler that operates on a resource: does it verify the caller is authorized for *this specific* resource, not just authenticated in general?
- IDOR: `/invoice/123` — can user A fetch user B's invoice by changing the ID?
- Vertical privilege escalation: can a regular user hit admin-only endpoints by knowing the URL?
- Force-browsing: are admin endpoints discoverable via patterns (`/admin`, `/internal`, `/debug`)?
- Metadata tampering: JWT claims, form fields like `role=admin`, hidden inputs.
- CORS allowing credentialed requests from any origin.
- Missing function-level access checks on batch / GraphQL / aggregate endpoints (check ONE field, miss the other).

**Python signals**: Missing decorators (`@login_required`, `@permission_required`), unchecked `current_user` references, `QuerySet.all()` instead of `.filter(user=request.user)`.

**React signals**: Hiding UI without backend auth (client-side `if (user.role === "admin")` — always accompanied by server-side?).

**CWEs**: CWE-22, CWE-284, CWE-285, CWE-639, CWE-862.

---

## A02 — Cryptographic Failures

**What to check:**
- Data at rest: are sensitive fields encrypted? What key is used? Where is it stored?
- Data in transit: TLS enforced? `verify=False` anywhere? HSTS set?
- Passwords: hashed with a slow KDF (Argon2id, bcrypt, scrypt)? Not raw SHA/MD5?
- Tokens: generated with `secrets`, not `random`?
- Certificates: validated? Pinned for mobile/sensitive clients?
- Algorithms: no MD5/SHA1/DES/3DES/RC4 for security? No ECB mode? Using AES-GCM or equivalent AEAD?
- IVs/nonces: unique per message? Not hardcoded?
- Crypto library: modern (`cryptography`), not deprecated (`pycrypto`)?

**CWEs**: CWE-259, CWE-295, CWE-310, CWE-327, CWE-328, CWE-329, CWE-331, CWE-338, CWE-522, CWE-798, CWE-916.

---

## A03 — Injection

Not just SQL. Any place untrusted data flows into an interpreter.

**Taxonomy:**
- **SQLi**: ORM raw queries, string-formatted queries, `.extra()`, `.raw()`.
- **NoSQLi**: dict operators flowing into MongoDB filters.
- **Command injection**: `shell=True`, `os.system`, `os.popen`, backticks.
- **LDAP injection**: unescaped filter strings.
- **XPath injection**: string-built XPath.
- **Template injection (SSTI)**: user input rendered as template.
- **Header injection (CRLF)**: `\r\n` in response headers or log lines.
- **Log injection**: fake log lines via unescaped newlines.
- **XSS** (client-side injection): see A03's modern umbrella, also the React catalog.
- **Prompt injection** (LLM): user content reaching system prompts or agent tools without isolation.

**CWEs**: CWE-20, CWE-74, CWE-77, CWE-78, CWE-79, CWE-89, CWE-90, CWE-94, CWE-943.

---

## A04 — Insecure Design

Architectural-level gaps, distinct from implementation bugs.

**What to check:**
- Is there a threat model? Are known trust boundaries documented?
- Rate limiting on expensive operations (login, password reset, search, LLM calls, PDF rendering)?
- Business-logic flaws: can you order negative quantities? Can you race the checkout? Can the coupon stack with itself?
- Security requirements captured? (e.g., "all PII fields must be encrypted at rest")
- Failure modes: does the system fail open or closed? What happens when the auth service is down?
- Logging strategy: is there *anything* to investigate an incident?
- Privacy: data retention, right-to-erasure, data minimization.

**This category rewards stepping back from the code and asking "what could go wrong that isn't a bug."**

**CWEs**: CWE-209, CWE-256, CWE-501, CWE-522, CWE-840.

---

## A05 — Security Misconfiguration

**What to check:**
- Debug mode: `DEBUG=True`, `app.run(debug=True)`, `NODE_ENV=development` in production?
- Default credentials: admin/admin left anywhere?
- Verbose errors in production: stack traces to users?
- Unused features enabled: sample apps, management consoles, admin interfaces, directory listing?
- Security headers missing: CSP, HSTS, X-Content-Type-Options, X-Frame-Options/frame-ancestors, Permissions-Policy.
- CORS too permissive (`*` with credentials, origin reflection).
- Cloud config: public S3 buckets, overly permissive IAM, exposed management APIs.
- Container config: running as root, `--privileged`, mounted docker socket.
- Outdated OS packages / base images.
- TLS config: weak ciphers, old protocol versions.

**CWEs**: CWE-2, CWE-11, CWE-13, CWE-15, CWE-16, CWE-260, CWE-520, CWE-526, CWE-537, CWE-541, CWE-547.

---

## A06 — Vulnerable and Outdated Components

**What to check:**
- Inventory: is there a manifest of all direct and transitive dependencies? (lockfile, SBOM)
- Version hygiene: are deps pinned? Is `requirements.txt` reproducible?
- CVE scanning: `pip-audit`, `safety`, `npm audit`, Dependabot, Snyk.
- End-of-life runtimes: Python 2, Node 14, etc.
- Unmaintained packages (last release >2 years, archived repos).
- Bundled / vendored libraries that haven't been updated since copied in.
- System packages in Dockerfile (`apt-get install` without version pinning).

**CWEs**: CWE-1104, CWE-937.

---

## A07 — Identification and Authentication Failures

**What to check:**
- Password complexity vs. length (NIST favors length; allow long passphrases, check against breach lists).
- No default / weak / well-known passwords.
- Rate limiting on login, 2FA, password reset.
- Account recovery flow: can it be abused? Is the reset token strong, short-lived, single-use, invalidated on use?
- 2FA: if enabled, is it enforced? Can it be bypassed via "remember this device"?
- Session: regenerated on login? Invalidated on logout? Timeout?
- JWT: algorithm pinned? `none` rejected? Signature always verified?
- Credential storage: hashed with modern KDF?
- Timing-safe comparison for tokens/secrets (`hmac.compare_digest`, not `==`).
- OAuth: PKCE on public clients? State parameter? Redirect URI allowlist?

**CWEs**: CWE-259, CWE-287, CWE-288, CWE-290, CWE-294, CWE-295, CWE-297, CWE-300, CWE-302, CWE-304, CWE-306, CWE-307, CWE-346, CWE-384, CWE-521, CWE-613, CWE-620, CWE-640, CWE-798.

---

## A08 — Software and Data Integrity Failures

**What to check:**
- **Insecure deserialization**: `pickle.loads`, `yaml.load`, `marshal.loads` on untrusted data. Java/.NET equivalents. `JSON.parse` is safe, `eval` on JSON is not.
- CI/CD pipeline integrity: who can modify build scripts? Are artifacts signed?
- Auto-update mechanisms: signed updates? Rollback protection?
- CDN scripts without Subresource Integrity (SRI).
- Unsigned software installers or update packages.
- Dependency confusion: internal package names that could be hijacked on public registries.

**CWEs**: CWE-345, CWE-353, CWE-426, CWE-494, CWE-502, CWE-565, CWE-784, CWE-829, CWE-830.

---

## A09 — Security Logging and Monitoring Failures

**What to check:**
- Is anything logged at all? Is log format structured (JSON/KV)?
- Are critical events logged? (authn success/failure, authz failures, password changes, 2FA events, privilege escalation, high-value transactions, admin actions)
- Are logs tamper-resistant? Stored centrally? Retention policy?
- Are alerts configured for anomalies (repeated failed logins, spike in 5xx, suspicious user agents)?
- Are logs scrubbed of secrets/PII?
- Correlation IDs for cross-service tracing?
- Is there a runbook for suspected incidents?

Absence of logging is itself a finding.

**CWEs**: CWE-117, CWE-223, CWE-532, CWE-778.

---

## A10 — Server-Side Request Forgery (SSRF)

**What to check:**
- Any endpoint that fetches a user-supplied URL: webhooks, PDF renderers, URL previews, OAuth callbacks, avatar fetchers, OG tag scrapers.
- Are internal IP ranges blocked? (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8, 169.254.0.0/16, ::1, fe80::/10, fc00::/7)
- Cloud metadata endpoints blocked? (`169.254.169.254`, `metadata.google.internal`)
- DNS rebinding considered? (Resolve once, reject if private, then pass IP — or use an egress proxy.)
- Schemes restricted? (`file:`, `gopher:`, `dict:`, `ftp:` often unnecessary.)
- Redirects followed? If yes, re-validate the redirect target.

**CWEs**: CWE-918.

---

## Mapping CWE → OWASP (common ones)

When unsure which OWASP category fits a CWE:

| CWE | OWASP Category |
|-----|----------------|
| CWE-22 (Path Traversal) | A01 |
| CWE-78 (OS Command Injection) | A03 |
| CWE-79 (XSS) | A03 |
| CWE-89 (SQLi) | A03 |
| CWE-94 (Code Injection / SSTI) | A03 |
| CWE-117 (Log Injection) | A09 |
| CWE-200 (Info Disclosure) | A01 / A04 |
| CWE-209 (Error Message Info) | A04 / A05 |
| CWE-284 (Improper Access Control) | A01 |
| CWE-285 (Improper Authorization) | A01 |
| CWE-287 (Improper Authentication) | A07 |
| CWE-295 (Cert Validation) | A02 |
| CWE-306 (Missing Auth) | A07 |
| CWE-307 (No Brute-Force Protection) | A07 |
| CWE-327 (Weak Crypto) | A02 |
| CWE-338 (Weak PRNG) | A02 |
| CWE-347 (Improper Signature Verification) | A08 |
| CWE-352 (CSRF) | A01 |
| CWE-434 (Unrestricted Upload) | A04 |
| CWE-502 (Insecure Deserialization) | A08 |
| CWE-521 (Weak Password Requirements) | A07 |
| CWE-522 (Insuff. Credentials Protection) | A07 |
| CWE-532 (Log Sensitive Info) | A09 |
| CWE-601 (Open Redirect) | A01 |
| CWE-611 (XXE) | A05 |
| CWE-639 (IDOR) | A01 |
| CWE-798 (Hardcoded Creds) | A02 / A07 |
| CWE-829 (Untrusted Includes) | A08 |
| CWE-862 (Missing Authorization) | A01 |
| CWE-915 (Mass Assignment) | A08 |
| CWE-918 (SSRF) | A10 |
| CWE-922 (Insecure Storage) | A02 |
| CWE-1021 (Clickjacking) | A05 |
| CWE-1104 (Unmaintained 3rd-party) | A06 |
| CWE-1333 (ReDoS) | A04 |

For multiple applicable categories, use the most specific.
