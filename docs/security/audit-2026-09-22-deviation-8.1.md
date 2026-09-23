# Targeted security review — deviation 8.1

**Date:** 2026-09-22
**Mode:** targeted `/8-auditar` review
**CPU checkpoint reviewed:** `3b2c3f79dd1199b5a542d0d86d40f44697eb1627`

This is not a full-project audit. Its scope is exactly the surface declared before implementation in
`8.1_implementation_plan.md`:

- `src/concept_embeddings_rag/corpus/fullwiki.py`
- `src/concept_embeddings_rag/corpus/scale_corpus.py`
- `src/concept_embeddings_rag/evaluation/scale_sensitivity.py`

The questions fixed by the plan were source identity, byte-size/MD5 verification, local SHA-256,
safe tar handling, path-traversal prevention, executable serialization, and model
revision/weight provenance.

## Result

```text
Critical  0
High      0
Medium    0
Low       1
```

One Low finding was raised in the declared surface; there is no Critical, High or Medium finding.

## 1. Official archive identity

`fullwiki.verify_archive` measures the staged file before parsing it. A declared byte-size mismatch
or MD5 mismatch is a refusal. MD5 is explicitly marked `usedforsecurity=False`; it exists only to
compare with the release checksum carried into the deviation. The local integrity record is SHA-256.

The measured S1 artifact records:

```text
bytes       1,553,565,403
MD5         01edf64cd120ecc03a2745352779514c
SHA-256     1acca1c5cc93c4890ea51091d2bad7c3ef6987aead127ab88728dc9e26555729
bytes_agree true
md5_agree   true
page_checked false
```

`page_checked: false` matters: the live publisher page was not independently re-read during S1, so
the result report does not claim that it was.

## 2. Tar and decompression handling

The archive is streamed with `tarfile.open(..., "r|bz2")`; no member is extracted to disk.

The review confirmed:

- only regular files are parsed;
- member names must match the plain relative-path allow pattern and `..` segments are refused;
- symlinks, hard links, devices and directories are not followed;
- declared member size is bounded before read;
- the read itself is bounded to `max_member_bytes + 1`;
- nested bz2 payloads are incrementally decompressed against a byte ceiling;
- record lines and total records are separately bounded in the full record iterator;
- payloads are parsed with `json.loads`, not `pickle`, `eval`, `exec` or YAML loaders.

Path traversal is therefore structurally absent from the archive path: no archive-controlled path is
ever written to the filesystem.

### SEC-033 — Low — the schema probe bypasses the per-line ceiling

`src/concept_embeddings_rag/corpus/fullwiki.py:316-319` iterates
`payload.splitlines()` and calls `json.loads(line)` inside `probe_layout` without first applying
`PHASE_8_1_MAX_LINE_BYTES`. `iter_records` does apply that ceiling, and `iter_members` limits the
entire decompressed member to 256 MiB, so this is a bounded local memory/CPU denial-of-service
surface rather than an unbounded allocation or code-execution path.

The official staged archive completed the probe and matched the declared byte size and MD5, so the
finding does not alter the measured 8.1 result. It is left **open, prospective** rather than changing
the code behind an already measured checkpoint: before the FullWiki loader is reused in Phase 9,
the probe should share the same line-bounded record iterator (or enforce the same line ceiling before
`json.loads`), with a regression test for an oversized probe line.

## 3. Deterministic selection and local artifact loading

`scale_corpus.py` receives parsed records and writes project-produced JSON/gzip/NPZ artifacts.

The review confirmed:

- selected distractor IDs are content-derived;
- the C19 and distractor blocks must be disjoint;
- selected IDs are re-derived from title/sentences on load;
- the ordered-selection digest is re-derived before a received selection is trusted;
- token counts are fixed-width Unicode IDs plus integer arrays, not object/pickle payloads;
- no executable deserialization is introduced.

The FullWiki archive is the only untrusted **record stream parsed by new 8.1 code**. Model snapshot
bytes are external too, but their download/loading path is inherited from Phase 8 and is treated
separately below.

## 4. Model identity and weight provenance

`scale_sensitivity.ModelPins` reuses the already frozen BGE and Qwen identities rather than creating
new moving aliases.

Before an embedding artifact is accepted, `write_embedding` enforces:

- model key maps to the frozen model ID and revision;
- resolved revision equals the pin;
- query prompt equals the frozen prompt;
- vector width equals the pinned dimension;
- exactly 500,000 corpus vectors and 600 dev questions are recorded;
- corpus and question caches cannot share a cache key.

The S6 artifacts preserve the SHA-256 of the weight bytes actually used:

```text
BGE
revision  5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
weights   3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad

Qwen
revision  97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
weights   0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd
```

The model-download/snapshot helpers themselves are inherited unchanged from Phase 8; this targeted
review checks the new 8.1 enforcement and provenance surface, not a second audit of those inherited
helpers.

## 5. Split isolation

`scale_sensitivity` calls `check_dev_only` on both reproduction and scale-measurement doors and also
requires exactly 600 questions. The outcome classifier requires both models to carry the same
non-empty `question_set_hash`.

The 1,400 held-out questions were not opened, encoded or measured during 8.1.

## Operational preservation incident — not a security finding

The successful Qwen S6 run wrote and loaded `embedding-qwen.json`, then completed Level 2 and all
four scale readings. The later RunPod output tar nevertheless contained that file at zero bytes.

The zero-byte copy is not treated as valid provenance and no replacement is fabricated. The
surviving `scale-qwen.json` preserves the original embedding digest and model provenance, and the
incident is reported in `8.1_results.md`.

This is an artifact-preservation incident after the measured run, not evidence of unsafe
deserialization, model substitution or a changed scale result. No source change is made to a closed
measurement in order to recreate provenance that no longer exists.

## Validation

Final post-closeout validation, after measurement and documentation corrections and before commit:

```text
pytest      2799 passed, 1 skipped in 183.02 s
ruff        clean
mypy        clean (61 source files)
```

No source code changed during the GPU measurement or documentation closeout.

**Closure:** the declared 8.1 security review is complete with one Low finding (SEC-033) and no
Critical, High or Medium finding. SEC-033 does not block 8.1 closure; remediation is required before
the FullWiki loader is reused in Phase 9.
