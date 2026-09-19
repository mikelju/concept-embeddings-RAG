"""A detect-secrets content filter for the deterministic hex ids of Phase 6 artifacts (plan D20).

Phase 6 versions JSON artifacts under `data/replacement/` that carry thousands of machine-written
lowercase hex ids - question ids, unit ids, content hashes, sha256 digests, the pinned model
commit - which the `HexHighEntropyString` detector flags one by one. A path exclusion would also
stop the scanner from reading everything else those files hold (entity forms from an LLM
extraction, verbatim texts, file names, configuration), so this filter skips a finding if and only
if all four conditions hold:

1. **Path**: the file, with `\\` normalized to `/`, is directly inside `data/replacement/` and ends
   in `.json`.
2. **Detector**: the finding comes from `HexHighEntropyString`. Every other detector is kept.
3. **Line shape**: the whole line, surrounding whitespace stripped, is exactly `"<key>": "<hex>"`
   with an optional trailing comma, where `<key>` is on the allowlist below with `<hex>` of one of
   its lengths; or a bare list element `"<hex>"`, optional comma, of length 16 or 24. `<hex>` is
   lowercase `[0-9a-f]` only. The allowlist is the set of keys the Phase 6 writers really put such
   values under, measured on a run of the three stages, not a guess.
4. **Value**: the flagged secret is exactly that `<hex>`.

Anything else stays scanned: another detector, another length, uppercase, a key off the list, a
line holding anything more, a bare 40- or 64-character element, a nested file, a non-JSON file.
If this module or its function cannot be loaded, detect-secrets logs a warning and applies no
filter, so a broken filter brings findings back rather than hiding any.

Standard library only: the CI job installs detect-secrets and nothing else.
"""

import re

PATH = re.compile(r"data/replacement/[^/]+[.]json")
DETECTOR = "HexHighEntropyString"
# Hyphens are allowed in a key because the writers key digests by system name (`hybrid-bm25`).
KEYED = re.compile(r'"(?P<key>[A-Za-z0-9_-]+)": "(?P<hex>[0-9a-f]+)",?')
BARE = re.compile(r'"(?P<hex>[0-9a-f]+)",?')
BARE_LENGTHS = frozenset({16, 24})

# Exactly the (key, length) pairs under which the Phase 6 writers put a lowercase hex value that
# HexHighEntropyString flags (D20). `unit_id` is absent on purpose: detect-secrets' own
# `is_likely_id_string` heuristic already skips it. Extended only by a reviewed change to this
# module and its tests, never by regenerating the baseline; `tests/test_cli_replacement.py` runs the
# three stages on the toy pipeline and requires the emitted pairs to equal this mapping.
ALLOWLIST: dict[str, frozenset[int]] = {
    # Question and unit ids (outcomes, diagnostics, traces).
    "qid": frozenset({24}),
    "p1": frozenset({16}),
    # Protocol and input identities (run configs, checks-dev.json, the freeze).
    "unit_set_hash": frozenset({16}),
    "prompt_digest": frozenset({16}),
    "corpus_cache_key": frozenset({16}),
    "question_cache_key": frozenset({16}),
    "dev_question_cache_key": frozenset({16}),
    "dev": frozenset({16}),
    "test": frozenset({16}),
    "revision": frozenset({40}),
    "resolved_revision": frozenset({40}),
    "manifest_sha256": frozenset({64}),
    "token_counts_sha256": frozenset({64}),
    "sha256": frozenset({64}),
    # Reproduction checks whose figure is itself an id: `unit_set_hash` (16), `revision` (40).
    "expected": frozenset({16, 40}),
    "observed": frozenset({16, 40}),
    # Artifact digests and the bindings between artifacts.
    "digest": frozenset({64}),
    "freeze_digest": frozenset({64}),
    "supersedes": frozenset({64}),
    "node_index_digest": frozenset({64}),
    "extraction_digest": frozenset({64}),
    "pilot_digest": frozenset({64}),
    "hop_run_digest": frozenset({64}),
    "traces_digest": frozenset({64}),
    "run_digest": frozenset({64}),
    "outcomes_digest": frozenset({64}),
    "first_digest": frozenset({64}),
    "second_digest": frozenset({64}),
    # Digests keyed by system name (decision `outcome_digests`, readouts, diagnostics, traces).
    "dense": frozenset({64}),
    "hybrid-bm25": frozenset({64}),
    "hybrid-entity-hop": frozenset({64}),
}


def _detector_name(plugin: object) -> str:
    return type(plugin).__name__


def is_deterministic_replacement_id(filename: str, line: str, secret: str, plugin: object) -> bool:
    """True only for a hex id of an allowlisted shape, alone on its line, in a Phase 6 artifact."""
    if not PATH.fullmatch(str(filename).replace("\\", "/")):
        return False
    if _detector_name(plugin) != DETECTOR:
        return False
    text = str(line).strip()
    keyed = KEYED.fullmatch(text)
    if keyed is not None:
        allowed = ALLOWLIST.get(keyed.group("key"))
        value = keyed.group("hex")
        return allowed is not None and len(value) in allowed and secret == value
    bare = BARE.fullmatch(text)
    if bare is not None:
        value = bare.group("hex")
        return len(value) in BARE_LENGTHS and secret == value
    return False
