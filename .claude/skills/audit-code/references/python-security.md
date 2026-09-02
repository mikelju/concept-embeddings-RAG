# Python Security — Vulnerability Catalog

Comprehensive reference for the `audit-code` skill. Read this file in full during **Phase 3 — Manual Review** when the target is Python.

Each entry has: **Pattern** (what to grep for), **Why it's dangerous**, **Vulnerable example**, **Secure fix**, **CWE**, **Severity baseline**.

---

## 1. Code Execution Sinks

### 1.1 `eval` / `exec` / `compile` with dynamic input
**Pattern**: `eval(`, `exec(`, `compile(` with any non-literal argument.
**Why**: Arbitrary code execution. Even "sanitized" input can escape via `__import__`, `__builtins__`, attribute access.
**Vulnerable**:
```python
result = eval(request.args["expr"])          # RCE
exec(f"config[{key}] = {value}")             # RCE via key or value
```
**Fix**: Replace with explicit parsing. For math: `ast.literal_eval` (still not safe for attacker-controlled input — limit length and use a whitelist parser like `simpleeval` with restricted functions). For config: structured formats (JSON, TOML, YAML safe-load).
**CWE-95** (Eval Injection). **Severity: Critical.**

### 1.2 `subprocess` with `shell=True`
**Pattern**: `subprocess.Popen/run/call/check_output(..., shell=True)`, `os.system(`, `os.popen(`.
**Why**: Shell metacharacters (`;`, `|`, `&&`, `$()`, backticks) enable command injection.
**Vulnerable**:
```python
subprocess.run(f"convert {user_file} out.png", shell=True)  # injection via user_file
os.system("rm " + path)                                     # injection via path
```
**Fix**: Pass a list, never a string; never use `shell=True` with untrusted input. Validate paths with `pathlib` + allowlist.
```python
subprocess.run(["convert", user_file, "out.png"], check=True)
```
**CWE-78** (OS Command Injection). **Severity: Critical.**

### 1.3 `pickle` / `marshal` / `shelve` on untrusted data
**Pattern**: `pickle.loads`, `pickle.load`, `cPickle.loads`, `marshal.loads`, `shelve.open`.
**Why**: Deserialization of pickled data executes arbitrary code (`__reduce__` gadget).
**Vulnerable**:
```python
user = pickle.loads(request.cookies["session"])   # RCE
```
**Fix**: Use JSON, msgpack, or Protobuf for untrusted data. If pickle is unavoidable, sign-then-verify (HMAC) before deserializing.
**CWE-502** (Deserialization of Untrusted Data). **Severity: Critical.**

### 1.4 `yaml.load` without `SafeLoader`
**Pattern**: `yaml.load(stream)` or `yaml.load(stream, Loader=FullLoader)`.
**Why**: Default loader instantiates arbitrary Python classes → RCE.
**Vulnerable**:
```python
config = yaml.load(open("config.yaml"))           # RCE via !!python/object
```
**Fix**: `yaml.safe_load(stream)` or explicit `Loader=yaml.SafeLoader`.
**CWE-502**. **Severity: Critical** (if input is untrusted), **High** (if file is controlled but still execution-risky).

### 1.5 `__import__` / `importlib.import_module` with dynamic name
**Pattern**: dynamic module import based on user input.
**Why**: Module side-effects execute on import. Allows loading attacker-chosen modules.
**Fix**: Whitelist of allowed module names; never let user input flow to `import_module`.
**CWE-470**. **Severity: High.**

---

## 2. Injection

### 2.1 SQL Injection
**Pattern**:
- String concatenation or `%`/f-string formatting inside `cursor.execute`, `session.execute`, `db.raw`, `text()`.
- Django `.raw()`, `.extra()`, `connection.cursor().execute()` with formatted strings.
- SQLAlchemy `text("SELECT … " + user_input)`.
**Why**: Classic SQLi → full database read/write, auth bypass, RCE in some DBs.
**Vulnerable**:
```python
cursor.execute(f"SELECT * FROM users WHERE name = '{name}'")
db.execute("SELECT * FROM t WHERE id = %s" % user_id)
User.objects.raw(f"SELECT * FROM users WHERE id={uid}")
```
**Fix**: Parameterized queries always. The driver handles quoting.
```python
cursor.execute("SELECT * FROM users WHERE name = %s", (name,))
db.session.execute(text("SELECT * FROM t WHERE id = :id"), {"id": user_id})
```
Also validate input types (`int(uid)`) before constructing even parameterized queries, to avoid second-order effects.
**CWE-89**. **Severity: Critical.**

### 2.2 NoSQL Injection (MongoDB, etc.)
**Pattern**: Passing a dict from user input directly into a query filter (`collection.find(request.json)`).
**Why**: `{"$ne": null}` operators enable auth bypass.
**Fix**: Validate shape and types before query; never let request bodies reach the DB driver unfiltered. Pydantic/Marshmallow schemas.
**CWE-943**. **Severity: High.**

### 2.3 LDAP Injection
**Pattern**: String interpolation into LDAP filters.
**Fix**: Escape with `ldap.filter.escape_filter_chars`.
**CWE-90**. **Severity: High.**

### 2.4 Template Injection (Jinja2, Mako, Django)
**Pattern**:
- `Template(user_input).render()` — **SSTI**.
- `render_template_string(user_input)` in Flask.
- Auto-escape disabled globally (`autoescape=False` in Jinja2) or locally (`{% autoescape false %}`).
- Using `| safe` on user data.
**Why**: SSTI → RCE via `{{ self.__init__.__globals__['__builtins__']['__import__']('os') }}`. Disabled auto-escape → XSS.
**Fix**: Never render user input as a template. Keep auto-escape on. Use `Markup` only for trusted HTML.
**CWE-94** (SSTI), **CWE-79** (XSS). **Severity: Critical / High.**

### 2.5 XML / XXE
**Pattern**: `xml.etree`, `xml.dom.minidom`, `lxml.etree.parse` without safe defaults.
**Why**: External entity resolution → file read, SSRF, DoS (billion laughs).
**Fix**: Use `defusedxml` (`pip install defusedxml`) and replace imports:
```python
import defusedxml.ElementTree as ET
```
**CWE-611**. **Severity: High.**

### 2.6 Log Injection
**Pattern**: Logging user input directly: `logger.info(f"user action: {user_input}")` without newline stripping.
**Why**: CRLF injection fakes log entries, confuses log parsers, can enable log-poisoning attacks.
**Fix**: Log as structured fields (key=value) or escape newlines: `user_input.replace("\n", "\\n").replace("\r", "\\r")`. Prefer structured logging (`logger.info("user action", user=user_input)`).
**CWE-117**. **Severity: Low-Medium.**

### 2.7 Header Injection / CRLF Injection
**Pattern**: `response.headers["X-Foo"] = user_input` or `redirect(user_url)` with untrusted input.
**Fix**: Validate and strip CR/LF. Use framework-native header APIs.
**CWE-113**. **Severity: Medium.**

---

## 3. Path Traversal & File System

### 3.1 Path traversal
**Pattern**: `open(base + user_path)`, `os.path.join(base, user_filename)` where `user_filename` may contain `..` or absolute paths.
**Why**: Read/write arbitrary files. `os.path.join("/safe", "/etc/passwd")` returns `/etc/passwd` — absolute second arg overrides.
**Vulnerable**:
```python
with open(f"uploads/{request.args['file']}") as f:  # ../../etc/passwd
    return f.read()
```
**Fix**:
```python
from pathlib import Path
base = Path("uploads").resolve()
target = (base / user_filename).resolve()
if base not in target.parents and base != target:
    raise PermissionError
```
Or use `werkzeug.utils.secure_filename` (strips path separators; does NOT fully prevent traversal on its own).
**CWE-22**. **Severity: High.**

### 3.2 Zip-slip / Tar-slip
**Pattern**: `zipfile.ZipFile(...).extractall(dest)` or `tarfile.open(...).extractall(dest)` without checking member names.
**Why**: Malicious archives with `../` entries write outside `dest`. Python 3.12+ adds `filter="data"` for tarfile, but older code is unsafe.
**Fix**: Validate each member name resolves inside the destination before extracting.
**CWE-22**. **Severity: High.**

### 3.3 Symlink following
**Pattern**: `open`, `shutil.copy` on attacker-writable directories without `O_NOFOLLOW`.
**Fix**: Use `os.open(path, os.O_NOFOLLOW)` or check `os.path.islink` before opening. Write to dedicated temp dirs created with `tempfile.mkdtemp`.
**CWE-59**. **Severity: Medium.**

### 3.4 TOCTOU on file operations
**Pattern**: `os.path.exists(p)` then `open(p)` — race window between check and use.
**Fix**: Use EAFP (try/except) pattern and atomic ops (`os.O_CREAT | os.O_EXCL` for new files).
**CWE-367**. **Severity: Medium.**

### 3.5 Insecure temporary files
**Pattern**: `tempfile.mktemp()` (deprecated), or manually constructing paths in `/tmp`.
**Fix**: `tempfile.mkstemp()`, `tempfile.NamedTemporaryFile(delete=False)`, `tempfile.mkdtemp()`.
**CWE-377**. **Severity: Medium.**

---

## 4. Cryptography

### 4.1 Weak hashing for security
**Pattern**: `hashlib.md5`, `hashlib.sha1`, `hashlib.new("md5")`, `Crypto.Hash.MD5`.
**Why**: Collision attacks broken for decades. MD5/SHA1 unacceptable for integrity, signatures, passwords.
**Exceptions**: Non-security uses (cache keys, content-addressing of trusted data) are OK but pass `usedforsecurity=False` (Python 3.9+).
**Fix**: Integrity → SHA-256/SHA-3. Passwords → `bcrypt`, `argon2-cffi`, or `scrypt` (never raw hash).
**CWE-327**. **Severity: High.**

### 4.2 Password hashing with fast hashes
**Pattern**: `hashlib.sha256(password.encode()).hexdigest()` stored in DB. `password + salt` then hashed.
**Why**: GPU can test billions of SHA-256/sec. Must use slow, memory-hard KDF.
**Fix**: `argon2.PasswordHasher().hash(password)` (Argon2id). Or bcrypt with cost ≥12.
**CWE-916**. **Severity: Critical.**

### 4.3 Insecure random for security-sensitive values
**Pattern**: `random.random()`, `random.randint()`, `random.choice()` used for tokens, session IDs, password reset codes, nonces, salts, API keys.
**Why**: Mersenne Twister is deterministic; state is recoverable from ~624 outputs.
**Fix**: `secrets` module: `secrets.token_urlsafe(32)`, `secrets.token_hex(16)`, `secrets.choice(…)`. For UUIDs in tokens, use `uuid.uuid4()` but prefer `secrets`.
**CWE-338**. **Severity: High.**

### 4.4 Hardcoded IV / nonce / key
**Pattern**: `AES.new(key, AES.MODE_CBC, iv=b"0000000000000000")`, keys as string literals.
**Why**: Static IVs break CBC semantic security. Reused GCM nonces are catastrophic (key recovery).
**Fix**: Generate a fresh random IV/nonce per message with `os.urandom(16)` or `secrets.token_bytes(16)`. Never reuse. Keys from a KMS, env var, or secure secret store — not literals.
**CWE-329** / **CWE-798**. **Severity: High/Critical.**

### 4.5 ECB mode / weak ciphers
**Pattern**: `AES.MODE_ECB`, `DES`, `3DES`, `RC4`, `Blowfish` for new code.
**Fix**: AES-GCM (authenticated) for symmetric. Use `cryptography` library, not `pycrypto` (abandoned) or low-level `pycryptodome` primitives without HMAC.
**CWE-327**. **Severity: High.**

### 4.6 Missing authenticated encryption
**Pattern**: Encrypting without MACing (CBC with no HMAC, CTR with no HMAC).
**Fix**: AES-GCM or ChaCha20-Poly1305. Or Encrypt-then-MAC with HMAC-SHA-256.
**CWE-353**. **Severity: High.**

### 4.7 TLS verification disabled
**Pattern**:
```python
requests.get(url, verify=False)
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context
httpx.Client(verify=False)
```
**Why**: MITM trivially possible on any network.
**Fix**: Never disable. For self-signed certs in dev, use a local CA and `verify=/path/to/ca.pem`. Never in production code paths, never behind an env flag that defaults to off.
**CWE-295**. **Severity: High.**

### 4.8 Weak JWT handling
**Pattern**:
- `jwt.decode(token, verify=False)` or `options={"verify_signature": False}`.
- `algorithms=["none"]` accepted.
- `algorithms` not specified (PyJWT <2.0 accepted all).
- Hardcoded secret, short secret (<256 bits for HS256).
- `algorithms=["HS256", "RS256"]` mixed (key confusion).
**Fix**: Always specify a single algorithm, verify signature, rotate secrets, use asymmetric (RS256/ES256) for public verification.
**CWE-347**. **Severity: Critical.**

---

## 5. Authentication & Authorization

### 5.1 Missing authorization check
**Pattern**: A route that operates on a resource (`GET /invoice/<id>`) without verifying the caller owns the resource.
**Why**: IDOR (Insecure Direct Object Reference) → horizontal privilege escalation.
**Fix**: After loading the object, check `obj.user_id == current_user.id` (or scoped query: `Invoice.query.filter_by(id=id, user_id=current_user.id)`).
**CWE-639 / OWASP A01**. **Severity: High.**

### 5.2 Broken session management
**Pattern**:
- Long-lived tokens with no expiry.
- Session ID predictable or stored client-side unencrypted.
- Password change doesn't invalidate existing sessions.
- No logout endpoint / logout doesn't revoke token server-side.
**Fix**: Short-lived access tokens + rotating refresh tokens. Server-side token revocation list. Invalidate on sensitive actions.
**CWE-384**. **Severity: High.**

### 5.3 Password policy
**Pattern**: No length check, accepts empty passwords, compares with `==` (timing attack).
**Fix**: Minimum 12 chars, passlist of breached passwords (HIBP API or `zxcvbn`). Compare hashes with the KDF's built-in verify (`argon2.verify`, `bcrypt.checkpw`). Never `==`.
**CWE-521**. **Severity: Medium.**

### 5.4 Missing rate limit / brute force protection
**Pattern**: Login, password reset, 2FA, token generation endpoints without rate limit.
**Fix**: Per-IP and per-account rate limits (`flask-limiter`, `django-ratelimit`, or gateway-level). Exponential backoff on failed logins. Account lockout with care (DoS vector).
**CWE-307**. **Severity: High.**

### 5.5 Timing-unsafe comparison
**Pattern**: `if token == stored_token`, `if hmac_received == hmac_computed`.
**Fix**: `hmac.compare_digest(a, b)`, `secrets.compare_digest(a, b)`.
**CWE-208**. **Severity: Medium.**

---

## 6. Web / HTTP

### 6.1 CSRF
**Pattern**: State-changing endpoint (POST/PUT/DELETE) without CSRF token check. Flask without `flask-wtf` or `flask-seasurf`. Django with `@csrf_exempt`. FastAPI (no built-in CSRF — must add).
**Fix**: Synchronizer token pattern or double-submit cookie. SameSite=Lax or Strict on session cookie (mitigates but doesn't replace).
**CWE-352**. **Severity: High.**

### 6.2 Open redirect
**Pattern**: `redirect(request.args["next"])` without validating `next`.
**Fix**: Whitelist allowed redirect targets or ensure URL is relative and same-origin.
**CWE-601**. **Severity: Medium.**

### 6.3 SSRF
**Pattern**: `requests.get(user_url)`, `urllib.request.urlopen(user_url)`, PDF/image fetchers, webhooks, URL previews.
**Why**: Server makes requests to internal metadata service (169.254.169.254), internal APIs, `localhost`, file:// URIs.
**Fix**:
- Allowlist of hostnames/domains.
- Resolve hostname first and reject RFC1918 / loopback / link-local / IPv6 equivalents.
- Disable redirects (`allow_redirects=False`) — attackers use 302 to metadata IP.
- Use a dedicated egress proxy.
**CWE-918**. **Severity: High.**

### 6.4 Missing security headers
**Pattern**: No `Content-Security-Policy`, `X-Content-Type-Options: nosniff`, `Strict-Transport-Security`, `X-Frame-Options` / `frame-ancestors`.
**Fix**: `flask-talisman`, Django `SecurityMiddleware`, or gateway config.
**Severity: Low-Medium.**

### 6.5 CORS misconfiguration
**Pattern**:
- `Access-Control-Allow-Origin: *` combined with `Access-Control-Allow-Credentials: true`.
- Reflecting `Origin` header without validation.
- Regex matching origins with weak patterns (`.example.com` matches `evilexample.com`).
**Fix**: Explicit allowlist of origins. Never reflect. Never `*` with credentials.
**CWE-942**. **Severity: High.**

### 6.6 Verbose error pages / debug mode in production
**Pattern**: `app.run(debug=True)`, `DEBUG = True` in Django, Flask/FastAPI default error handlers exposing stack traces.
**Why**: Leaks paths, versions, secrets in env vars, enables Werkzeug debugger PIN bypass → RCE.
**Fix**: Environment-gated debug. Production defaults to off. Catch-all exception handler that logs full trace server-side and returns a generic error to the user.
**CWE-209**. **Severity: Critical** (debug=True in prod), **Medium** otherwise.

### 6.7 Mass assignment
**Pattern**: `User(**request.json)` or `user.update(**request.json)` where the model has sensitive fields (`is_admin`, `role`).
**Fix**: Use Pydantic/Marshmallow schemas with explicit allowed fields. Never splat user input into model constructors.
**CWE-915**. **Severity: High.**

### 6.8 ReDoS (Regular Expression DoS)
**Pattern**: Regex with nested quantifiers (`(a+)+`, `(a|ab)+`) matched against user input.
**Fix**: Use `re` carefully; prefer `re2` (`pip install google-re2`) for untrusted patterns. Set input length limits. Use `regex` module with `TIMEOUT`.
**CWE-1333**. **Severity: Medium.**

---

## 7. Secrets & Credentials

### 7.1 Hardcoded secrets
**Pattern**: String literals matching API key, token, password shapes. See `references/secrets-patterns.md` for regex catalog.
**Fix**: Environment variables, secret manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault), `.env` with `python-dotenv` (file gitignored).
**CWE-798**. **Severity: Critical** (production creds), **High** (test creds that could enable lateral movement).

### 7.2 Secrets in logs
**Pattern**: `logger.info(f"Request: {request.headers}")`, logging request bodies, logging DB connection strings.
**Fix**: Redaction filter in logging config. Explicit allowlist of logged fields.
**CWE-532**. **Severity: High.**

### 7.3 Secrets in error messages
**Pattern**: Traceback containing environment variables, connection strings, tokens.
**Fix**: Global exception handler that strips frame locals before logging. Never return raw tracebacks to users.
**CWE-209**. **Severity: Medium.**

### 7.4 Secrets committed to git
**Pattern**: `.env` tracked, `secrets.yaml` committed, hardcoded creds in any versioned file.
**Fix**: `git filter-repo` to purge history, rotate all leaked credentials, add to `.gitignore`, add `gitleaks` or `detect-secrets` to pre-commit hooks.
**CWE-798**. **Severity: Critical** (if public repo), **High** (private).

---

## 8. Dependencies

### 8.1 Known-vulnerable packages
**Pattern**: Versions in `requirements.txt` / `pyproject.toml` / `poetry.lock` with CVEs.
**Tools**: `pip-audit`, `safety`, GitHub Dependabot, Snyk.
**Fix**: Upgrade to patched version. If no patch, evaluate risk and pin with documented reason.
**CWE-1104**. **Severity: depends on CVE.**

### 8.2 Unmaintained / abandoned packages
**Pattern**: Last release >2 years, no GitHub activity, known fork exists.
**Fix**: Migrate to actively maintained alternative.
**Severity: Medium.**

### 8.3 Typosquatting / malicious packages
**Pattern**: Packages with name variants (`reqests`, `python-dateutil` vs `dateutil`).
**Fix**: Verify package name letter-by-letter against official docs. Check download count, maintainers, repo link.
**CWE-829**. **Severity: Critical** if present.

### 8.4 Unpinned dependencies
**Pattern**: `requests` (no version) in requirements.txt.
**Fix**: Pin to a known-good version or range. Use `pip-compile` / Poetry lockfile for reproducible builds.
**Severity: Low-Medium** (not a direct vuln but enables supply chain attacks).

---

## 9. Concurrency & Logic

### 9.1 Race conditions on financial/stateful operations
**Pattern**: `if balance >= amount: balance -= amount` without transactional locking.
**Fix**: `SELECT … FOR UPDATE` (DB locks), atomic `UPDATE balance = balance - :amt WHERE id = :id AND balance >= :amt` with row count check.
**CWE-362**. **Severity: High.**

### 9.2 Double-submit / replay
**Pattern**: Idempotency-sensitive endpoints (payment, account creation) without idempotency keys.
**Fix**: Idempotency keys stored server-side with TTL. Unique constraint on sensitive operations.
**Severity: Medium-High.**

---

## 10. Python-specific gotchas

### 10.1 `assert` for security checks
**Pattern**: `assert user.is_admin, "forbidden"`.
**Why**: `python -O` strips asserts. Control flow disappears in production.
**Fix**: Explicit `if not: raise PermissionError(...)`.
**CWE-617**. **Severity: High.**

### 10.2 Mutable default arguments
**Pattern**: `def f(x=[]):` — not a vuln per se, but in security-sensitive contexts (caches, allow-lists) mutable defaults leak state between calls.
**Fix**: `def f(x=None): if x is None: x = []`.

### 10.3 `input()` in Python 2
(Unlikely today, but:) `input()` in Py2 calls `eval()`. Use `raw_input` there. In Py3, `input()` is safe (returns str).

### 10.4 `shelve` without authentication
**Pattern**: `shelve.open("state.db")` on attacker-writable paths.
**Why**: `shelve` uses pickle internally → untrusted shelve file = RCE.
**Fix**: Same rules as pickle; sign the shelve file or use a safer store.
**CWE-502**. **Severity: High.**

### 10.5 `xml.sax` with untrusted input
Use `defusedxml.sax`.

### 10.6 `requests.Session` without timeout
**Pattern**: No `timeout=` on HTTP calls.
**Why**: Slow-loris / hung upstream → resource exhaustion.
**Fix**: Always pass `timeout=(connect, read)` — e.g., `(5, 30)`. Set retries with exponential backoff.
**CWE-400**. **Severity: Low-Medium.**

### 10.7 `Pillow` / image library CVEs
**Pattern**: Processing user-uploaded images. Pillow has had many CVEs.
**Fix**: Stay patched. Set `Image.MAX_IMAGE_PIXELS` to prevent decompression bombs. Validate MIME before decoding.
**Severity: Medium.**

### 10.8 `pandas.read_pickle` / `joblib.load`
**Pattern**: Loading pickled DataFrames/models from untrusted sources.
**Why**: Pickle RCE through the backdoor.
**Fix**: Parquet/Arrow for data, versioned and signed model files, ONNX where possible.
**CWE-502**. **Severity: Critical.**

### 10.9 `flask.request.remote_addr` trust
**Pattern**: Rate-limiting or logging by `remote_addr` while behind a proxy.
**Why**: `remote_addr` is the proxy, not the client. `X-Forwarded-For` trust must be explicit and proxy-aware (`ProxyFix` with correct `x_for=` count).
**Fix**: Configure `ProxyFix` with exact number of trusted proxies. Never trust `X-Forwarded-For` blindly.
**CWE-290**. **Severity: Medium.**

### 10.10 `Flask.send_file` with user paths
**Pattern**: `send_file(user_input_path)`.
**Fix**: `send_from_directory(base, filename)` with `filename` passed through `secure_filename` and explicit base-dir containment check.
**CWE-22**. **Severity: High.**

---

## 11. Django-specific

- `QuerySet.extra()` with user input → SQLi. Use `.filter()` / `.annotate()` with expressions.
- `mark_safe` on user input → XSS. Only use on static strings.
- `@csrf_exempt` needs a comment explaining why. Audit every use.
- `ALLOWED_HOSTS = ['*']` in production → Host header attacks. Set explicit hostnames.
- `SECURE_SSL_REDIRECT`, `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE`, `SECURE_HSTS_SECONDS` should all be set in production.
- `DEBUG = True` in prod — critical.
- Admin at `/admin/` with default URL → brute-force target. Move it or add 2FA (`django-otp`).

## 12. Flask-specific

- `app.run(debug=True)` → Werkzeug debugger PIN may be bypassable → RCE. Never in prod.
- Custom session interface without signing → session tampering. Use default `itsdangerous`-based sessions with a strong `SECRET_KEY`.
- `render_template_string` with user input → SSTI.
- Missing `app.config["SECRET_KEY"]` or static value → session forgery.

## 13. FastAPI-specific

- No built-in CSRF — if using cookies for auth on browser clients, add CSRF middleware (`fastapi-csrf-protect`).
- `Depends` chains that short-circuit auth checks — verify every protected route has an auth dependency.
- Pydantic models with `extra = "allow"` → mass assignment vector.
- `response_model` missing on endpoints returning DB objects → may leak fields (e.g., password hash).

---

## 14. AI / ML-specific (if applicable)

- **Prompt injection** in LLM-facing code: user input concatenated into system prompts. Defense: separate user content from instructions, output filters, never grant LLM direct tool access without human-in-loop for sensitive ops.
- **Model file poisoning**: loading pickled `.pt` / `.pkl` models from untrusted sources → RCE (see 10.8).
- **Data exfil via prompts**: logging full LLM inputs/outputs may leak user PII or secrets.

---

## Quick-scan regex cheat sheet

Grep for these as a first pass:

```
eval\(|exec\(|compile\(
subprocess.*shell\s*=\s*True
os\.system|os\.popen
pickle\.loads?|cPickle|marshal\.loads
yaml\.load\s*\((?!.*Safe)
verify\s*=\s*False
MD5|SHA1|sha1\(|md5\(
random\.(random|randint|choice|uniform|sample)\b
assert\s+.*\bis_admin|assert\s+.*\brole
render_template_string
\.raw\(|\.extra\(|cursor\.execute\(.*%|cursor\.execute\(f"
password\s*==|token\s*==|secret\s*==
```

(These are starting points — every hit needs manual confirmation.)
