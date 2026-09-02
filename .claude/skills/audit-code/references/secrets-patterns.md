# Hardcoded Secrets — Detection Patterns

Reference for the `audit-code` skill. Use to scan for credentials, tokens, and keys committed in code, config, or logs.

## How to use

1. **Regex pass** — grep the codebase with patterns below (Grep tool with these as regexes).
2. **Visual pass** — open config files, env templates, and README for anything that *looks* like a secret (high-entropy strings, hex blobs, base64 lumps of 20+ chars).
3. **Tool pass** — if `gitleaks`, `trufflehog`, or `detect-secrets` are installed, run them across the repo (including git history).
4. **Redact in the report** — never quote a full secret. Show the first 4 chars + `•••` + service name.

## Secret format patterns (regex)

Tune these based on what the project actually uses. False positives are expected — confirm each match.

### Generic API / auth tokens
```
(?i)(api[_-]?key|apikey|token|secret|passwd|password|auth)[\s:=]+["']?[A-Za-z0-9_\-\.]{16,}
```

### AWS
```
AKIA[0-9A-Z]{16}                        # Access Key ID
(?i)aws(.{0,20})?(secret|private).{0,20}?[=:]\s*["']?[A-Za-z0-9/+=]{40}  # Secret Access Key
```

### GitHub
```
ghp_[A-Za-z0-9]{36}        # Personal access token (classic)
gho_[A-Za-z0-9]{36}        # OAuth
ghu_[A-Za-z0-9]{36}        # User-to-server
ghs_[A-Za-z0-9]{36}        # Server-to-server
ghr_[A-Za-z0-9]{36}        # Refresh
github_pat_[A-Za-z0-9_]{82}  # Fine-grained PAT
```

### Slack
```
xox[baprs]-[0-9A-Za-z\-]+
```

### Google
```
AIza[0-9A-Za-z\-_]{35}                     # Google API Key
ya29\.[0-9A-Za-z\-_]+                      # OAuth access token
[0-9]+-[0-9A-Za-z_]{32}\.apps\.googleusercontent\.com   # OAuth Client ID
```

### Stripe
```
sk_live_[0-9a-zA-Z]{24,}            # Secret key (live)
sk_test_[0-9a-zA-Z]{24,}            # Secret key (test)
pk_live_[0-9a-zA-Z]{24,}            # Publishable (not secret but still flag)
rk_live_[0-9a-zA-Z]{24,}            # Restricted
whsec_[0-9a-zA-Z]{32,}              # Webhook secret
```

### OpenAI / Anthropic
```
sk-[A-Za-z0-9]{32,}                 # OpenAI
sk-ant-[A-Za-z0-9\-_]{32,}          # Anthropic
```

### Microsoft / Azure
```
(?i)client[_-]?secret[\s:=]+["']?[A-Za-z0-9~._-]{34,}  # Azure AD client secret (shape varies)
```

### JWT tokens
```
eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*
```
(Flag, then decode header/payload to check if it's a demo token or live. Live JWTs in source = High.)

### Private keys
```
-----BEGIN (RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----
```

### Database connection strings
```
(?i)(postgres|postgresql|mysql|mongodb|redis)://[^\s:]+:[^\s@]+@
```
(The password lives between `:` and `@`.)

### Generic high-entropy heuristic
Strings ≥20 chars, ≥4 character classes (upper/lower/digit/symbol), Shannon entropy ≥4.0 bits/char are suspicious. No clean regex — requires a tool like `detect-secrets` or manual inspection.

## Locations where secrets hide

Check these specifically — secrets are often not where you'd expect:

| Location | What to look for |
|----------|------------------|
| Source code | Literal strings assigned to variables named `*_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD`, `api_key`, `bearer`, `auth`. |
| Config files | `.env`, `.env.local`, `config.yaml`, `config.json`, `settings.py`, `application.properties`. |
| Test files | Tests often use real keys "temporarily" — common source of leaks. |
| CI files | `.github/workflows/*.yml`, `.gitlab-ci.yml` — `env:` sections, `echo $SECRET` patterns. |
| Dockerfiles | `ENV API_KEY=...`, `ARG` that defaults to a secret. |
| Docstrings / comments | "the old key was abc123, don't use" — still leaked. |
| Git history | `git log -p | grep -E '(api_key|secret|token)'` — secrets removed in a later commit are still in history. |
| Frontend bundles | Bundled `process.env.*` inlined as literals in JS. |
| Mobile apps | Hardcoded in Android/iOS code, retrievable by decompilation. |
| README / docs | Example snippets with real keys. |
| Error logs | Connection strings, auth headers dumped on exception. |
| Gist / pastebin links | References to external paste of "temp" keys. |

## Reporting rules for secret findings

1. **Redact**. Never paste a full secret into the report. Example:
   ```
   API_KEY = "sk-proj-abcd••••••••••" (OpenAI, live)
   ```
2. **Rotate immediately**. Mark the finding with a prominent note: the secret is compromised the moment it's in a repo you don't fully control — rotate now, worry about root cause after.
3. **Check git history**. Removing from HEAD doesn't remove from history. Mention `git filter-repo` + force-push + rotate.
4. **Assess scope**. What does this key authorize? Production DB? Public read-only? Tag severity accordingly.
5. **Never echo to chat**. If the user paste-drops a secret to you during audit, tell them to rotate it and continue with the placeholder.

## Severity guidance

| Scenario | Severity |
|----------|----------|
| Live production credential in public repo | **Critical** |
| Live production credential in private repo | **Critical** (assume compromise; rotate) |
| Live credential in CI logs or build artifacts | **High** |
| Test/staging credential with privileges that enable lateral movement | **High** |
| Truly dummy / placeholder credential (e.g., `CHANGE_ME`, `xxx`) | **Info** (document, replace with env var reference) |
| Public client ID (OAuth, Stripe publishable key) | **Info** (not secret, but confirm it's intentional) |

## Related practices to check

During a secrets audit, also verify:
- `.gitignore` excludes `.env`, `*.pem`, `*.key`, `credentials*`.
- Pre-commit hook runs `gitleaks` / `detect-secrets` / `trufflehog`.
- Secrets manager is used for runtime (Vault, AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, Doppler, 1Password CLI).
- Key rotation policy exists and has been exercised.
- Least-privilege on the credentials themselves (scoped tokens, not root keys).
