# React / Frontend Security — Catalog

Reference for the `audit-code` skill when auditing React, Next.js, Vite, or general JS/TS frontends. Read in full when the target includes client-side code.

Remember: **frontend security is defense-in-depth**. The server must still validate everything — never rely on client-side checks for authorization, never trust the frontend to enforce business rules.

---

## 1. Cross-Site Scripting (XSS)

### 1.1 `dangerouslySetInnerHTML`
**Pattern**: `<div dangerouslySetInnerHTML={{ __html: userContent }} />`.
**Why**: Bypasses React's default escaping → DOM-based XSS.
**Fix**: Avoid entirely. If rich-text rendering is needed, sanitize with `DOMPurify.sanitize(html, { USE_PROFILES: { html: true } })` and apply strict allowlists.
**CWE-79 / OWASP A03**. **Severity: Critical** if user-controlled, **High** if CMS-controlled.

### 1.2 Rendering URLs from user input
**Pattern**: `<a href={userInput}>`, `<img src={userInput}>`, `<iframe src={userInput}>`.
**Why**: `javascript:alert(1)` as href → XSS on click. `data:` URIs can execute scripts.
**Fix**:
```ts
const safe = (url: string) => {
  const u = new URL(url, window.location.origin);
  if (!["http:", "https:", "mailto:"].includes(u.protocol)) return "#";
  return u.toString();
};
```
**Severity: High.**

### 1.3 `eval` / `Function` constructor / `setTimeout(string)`
**Pattern**: `eval(code)`, `new Function(code)`, `setTimeout("code", 1000)`, `setInterval("code", ...)`.
**Fix**: Replace with actual functions. Never string-form `setTimeout`/`setInterval`.
**CWE-95**. **Severity: Critical.**

### 1.4 Unsafe third-party HTML widgets
**Pattern**: Chat widgets, analytics, A/B testing tags injected without SRI.
**Fix**: Use Subresource Integrity (`integrity="sha384-…"`) on every `<script>` with external src. Review what the widget script can access.
**Severity: Medium.**

### 1.5 Content-Security-Policy (CSP) missing or weak
**Pattern**: No CSP header, or `unsafe-inline` / `unsafe-eval` in script-src.
**Fix**: Nonce-based CSP: `Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-{random}'; object-src 'none'; base-uri 'self'; frame-ancestors 'none';`. Remove `unsafe-inline` — refactor inline handlers to event listeners.
**CWE-693**. **Severity: Medium** (mitigating control is missing).

---

## 2. Authentication & Token Handling

### 2.1 Tokens in `localStorage` / `sessionStorage`
**Pattern**: `localStorage.setItem("token", jwt)`.
**Why**: Readable by any JavaScript → stolen on first XSS. No automatic attach to requests → every fetch must manually include → one mistake = auth bypass.
**Fix**: Use `HttpOnly; Secure; SameSite=Strict` cookies set by the server. Frontend never touches the token directly. Add CSRF protection separately (double-submit or origin check).
**CWE-922**. **Severity: High.**

### 2.2 JWT decoded client-side and trusted
**Pattern**: `jwt_decode(token).role === "admin"` used to gate features.
**Why**: Client can forge any claim; client-side decode is a UI hint only. The server must re-verify on every request.
**Fix**: Treat the token as opaque on the client. Make authorization decisions server-side. If the UI needs the role, return it as a separate API field.
**Severity: Medium** (UI concern; the server-side check is what actually matters).

### 2.3 OAuth implicit flow
**Pattern**: `response_type=token` in OAuth URLs.
**Fix**: Use Authorization Code with PKCE (`response_type=code`, `code_challenge`). Implicit flow is deprecated.
**CWE-522**. **Severity: High.**

### 2.4 Exposed API keys in bundled code
**Pattern**: `const API_KEY = "sk-…"` in a React app. Env vars prefixed `NEXT_PUBLIC_` / `VITE_` / `REACT_APP_` containing secrets.
**Why**: Anything bundled into the frontend is public. Shipped in the JS bundle, visible in DevTools.
**Fix**: Frontend calls your backend; backend holds the secret. Only publishable keys (Stripe publishable, Maps public) go in the frontend. Any key that can authorize writes or reads beyond public data must live server-side.
**CWE-798**. **Severity: Critical** (for write-capable keys).

---

## 3. Cross-Origin & Navigation

### 3.1 `window.postMessage` without origin check
**Pattern**: `window.addEventListener("message", e => { doThing(e.data); })` without checking `e.origin`.
**Fix**:
```ts
window.addEventListener("message", e => {
  if (e.origin !== "https://trusted.example.com") return;
  // process e.data
});
```
**CWE-346**. **Severity: High.**

### 3.2 `target="_blank"` without `rel`
**Pattern**: `<a target="_blank" href={url}>`.
**Why**: Pre-Chrome 88, the opened page could `window.opener.location = evil`. Modern browsers default to safe, but don't rely on it.
**Fix**: Always `rel="noopener noreferrer"`.
**CWE-1022**. **Severity: Low.**

### 3.3 Open redirect via router state
**Pattern**: `navigate(searchParams.get("next"))` without validating.
**Fix**: Allowlist or same-origin check before navigation.
**Severity: Medium.**

### 3.4 Iframe without `sandbox`
**Pattern**: `<iframe src={untrusted} />` without `sandbox` attribute.
**Fix**: `<iframe sandbox="allow-scripts" src="…">` — grant only what's needed.
**Severity: Medium.**

---

## 4. API Interactions

### 4.1 Client-side authorization gates
**Pattern**: Hiding admin buttons based on a `user.role` in state, without backend enforcement.
**Why**: Anyone can call the API directly; the UI is not a security boundary.
**Fix**: Authorization happens server-side on every endpoint. UI gating is UX only.
**CWE-602**. **Severity: depends on server-side — if server also misses the check, Critical.**

### 4.2 Sending credentials cross-origin
**Pattern**: `fetch(url, { credentials: "include" })` to arbitrary origins.
**Fix**: Only include credentials for trusted same-site APIs. Configure CORS on the backend to allowlist origins (no `*` with credentials).
**Severity: Medium.**

### 4.3 Unvalidated GraphQL introspection in production
**Pattern**: GraphQL introspection enabled in production.
**Fix**: Disable introspection in production, or require auth.
**Severity: Low-Medium.**

### 4.4 Missing CSRF protection for cookie-auth APIs
If using cookies (the recommended token storage), state-changing requests need CSRF protection: SameSite=Strict, or CSRF tokens, or custom headers (`X-Requested-With` with CORS preflight requirement).
**Severity: High** if missing.

---

## 5. Dependencies & Supply Chain

### 5.1 Known-vulnerable npm packages
**Tools**: `npm audit`, `yarn audit`, `pnpm audit`, Dependabot, Snyk, `socket.dev`.
**Fix**: Upgrade. If no patch, evaluate risk. Track in audit report.
**Severity: per-CVE.**

### 5.2 Typosquatted / malicious packages
**Pattern**: Typos in `package.json` (`reacat`, `loadash`), packages with suspicious install scripts, packages with unexpected postinstall hooks.
**Tools**: `socket.dev`, manual review of unfamiliar names.
**Severity: Critical** if present.

### 5.3 Packages with install scripts
`postinstall` scripts run arbitrary code during `npm install`. Review `package.json` `scripts.postinstall` of direct and transitive deps.
**Fix**: `npm ci --ignore-scripts` in CI; vet any package requiring install scripts.
**Severity: High** (supply chain risk).

### 5.4 Outdated React version
React <16 had known XSS issues. React <18 lacks some modern protections.
**Fix**: Stay on supported versions.

### 5.5 CDN dependencies without SRI
`<script src="https://cdn.example.com/lib.js">` without `integrity`.
**Fix**: Add `integrity="sha384-…" crossorigin="anonymous"`. Consider vendoring.
**CWE-353**. **Severity: Medium.**

---

## 6. State & Storage

### 6.1 Sensitive data in state / Redux devtools
**Pattern**: Passwords, SSNs, tokens in Redux store with devtools enabled in production.
**Fix**: `composeWithDevTools` gated by `NODE_ENV === "development"`. Never log sensitive fields.
**CWE-532**. **Severity: Medium.**

### 6.2 Sensitive data in URL
**Pattern**: `/reset?token=abc123` → logged by analytics, proxies, referer headers.
**Fix**: POST bodies or short-lived one-time codes. If GET is unavoidable, short TTL and invalidate on use.
**CWE-598**. **Severity: Medium.**

### 6.3 IndexedDB / localStorage persistence of PII
**Pattern**: Storing user PII in browser storage for offline use.
**Fix**: If required, encrypt with a key derived from user credentials (not stored). Provide clear-on-logout. Document in privacy policy.
**Severity: Medium.**

---

## 7. Build & Deployment

### 7.1 Source maps in production
**Pattern**: `.map` files served in production, exposing unminified source.
**Fix**: Generate source maps but don't upload to public CDN; upload to error-tracking service only.
**Severity: Low** (information disclosure).

### 7.2 Environment variables leaked into bundle
**Pattern**: Any `process.env.SECRET` referenced in client code. Build tools inline these at build time.
**Fix**: Strictly separate server-side and client-side env. Prefix public vars (`NEXT_PUBLIC_`, `VITE_`, `REACT_APP_`) and audit every use.
**Severity: Critical** if secret.

### 7.3 Debug routes / feature flags exposed
**Pattern**: `/debug`, `/dev/...` routes present in prod bundle.
**Fix**: Tree-shake via `NODE_ENV` guards; verify prod build doesn't include them.

### 7.4 Unpinned CDN scripts
(See 5.5.)

---

## 8. Next.js / SSR-specific

- **`getServerSideProps` returning sensitive data**: the server response body goes to the client. Strip sensitive fields before returning.
- **API routes without auth**: every `pages/api/**` or `app/api/**` route needs explicit auth middleware — they're not gated by default.
- **`Image` with `unoptimized`**: loads images without domain allowlist validation. Set `images.remotePatterns` strictly.
- **`next.config.js` `headers`**: configure CSP, HSTS, etc. here — don't rely on a reverse proxy alone.
- **ISR revalidation endpoints**: protect with a secret token; attackers triggering revalidation can DoS or poison cache.
- **Server Actions**: validate inputs; they're just POST endpoints with a friendly API.

---

## 9. Cookies

Cookies set from the server for auth should have:
- `HttpOnly` — inaccessible to JS
- `Secure` — HTTPS only
- `SameSite=Strict` (or `Lax` if you need cross-site top-level navigation)
- `Path=/` appropriate scope
- `__Host-` prefix for strongest guarantees (no Domain attribute, Path=/, Secure)

Missing any of these on an auth cookie → **High.**

---

## 10. Quick-scan regex cheat sheet

```
dangerouslySetInnerHTML
localStorage\.setItem|sessionStorage\.setItem
href=\{[^}]*\}|src=\{[^}]*\}
eval\(|new Function|setTimeout\(["'`]|setInterval\(["'`]
window\.postMessage|addEventListener\(["']message
target=["']_blank["'](?![^>]*rel=)
process\.env\.[A-Z_]+(?=.*(?:KEY|SECRET|TOKEN|PASSWORD))
credentials:\s*["']include["']
```

(Every hit gets read and confirmed — regex is a starting point.)
