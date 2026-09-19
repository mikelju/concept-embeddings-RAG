"""The CI security gate, checked from the repository it is supposed to gate.

Both findings behind this file are the same failure in two places: a scanner that
runs, reports success, and was never looking at the thing it was supposed to cover.
A green check that means nothing is worse than no check, because it is read as
evidence.

These tests read the workflow and the baseline as data. They do not run the
scanners - CI does that - they assert that what CI runs covers what this repository
now contains.
"""

import ast
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "security.yml"
BASELINE = ROOT / ".secrets.baseline"


def synced_groups() -> set[str]:
    """The dependency groups the `pip-audit` job installs before scanning.

    Comment lines are skipped: the block above the command explains why every
    group is synced, and reading that explanation as if it were the command
    would let the test pass on a workflow that syncs nothing.
    """
    commands = [
        line
        for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
        if "uv sync" in line and not line.lstrip().startswith("#")
    ]
    return {group for line in commands for group in re.findall(r"--group\s+([\w.-]+)", line)}


def declared_groups() -> set[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return set(pyproject["dependency-groups"])


def test_the_dependency_scan_covers_every_optional_group_that_ships_code():
    """SEC-018: `uv sync` prunes whatever the named groups do not ask for, so a scan

    run with `--group security` alone audited an environment the `labeling`
    dependencies were absent from - the only ones in this project that open a
    network connection. The job reported no known vulnerabilities without having
    looked at them.

    `dev` is excluded on purpose: pytest, ruff and mypy are tooling and never run
    against corpus data or a network. Every other group is runtime, so adding one
    to `pyproject.toml` without adding it here fails this test.
    """
    assert declared_groups() - {"dev"} <= synced_groups()


def test_the_dependency_scan_audits_the_environment_not_a_requirements_file():
    """`pip-audit -r` resolves every pin against PyPI and dies on `torch==2.13.0+cpu`."""
    text = WORKFLOW.read_text(encoding="utf-8")
    uv_branch = text[text.index("if [ -f uv.lock ]") : text.index("elif [ -f requirements.txt ]")]

    assert "pip-audit --desc" in uv_branch
    assert "-r " not in uv_branch


def test_the_secret_scan_baseline_still_describes_this_repository():
    """SEC-019: the baseline was generated before this phase existed, so the gate

    would have failed the first pull request on findings nobody had audited - and
    the usual way that gets unblocked is regenerating the baseline in a hurry,
    which is how a real secret gets waved through.
    """
    results = json.loads(BASELINE.read_text(encoding="utf-8"))["results"]

    missing = [name for name in results if not (ROOT / name.replace("\\", "/")).exists()]
    assert not missing, f"the baseline still carries findings for files that are gone: {missing}"


def test_the_baseline_carries_the_findings_this_phase_introduced():
    """SEC-019: the five are false positives, and being audited is what makes them so.

    `API_KEY_VARIABLE` is the name of an environment variable, the dictionary key
    in the results document and the pool hash in two test fixtures are hex digests
    of public data, and the catalogue entry that explains the first one had to quote
    it verbatim to explain it. None of them is a secret, and none of them may reach
    CI as an unexplained finding.
    """
    results = set(json.loads(BASELINE.read_text(encoding="utf-8"))["results"])

    for path in (
        "src/concept_embeddings_rag/concepts/labeling.py",
        "docs/plans/phase_2/2.results.md",
        "tests/concepts/test_dedup.py",
        "tests/concepts/test_dictionary_artifact.py",
        "docs/security/README.md",
    ):
        assert path in results, f"{path} carries an audited false positive the baseline lacks"


def test_the_baseline_is_keyed_the_way_the_runner_reads_it():
    """SEC-019: the gate runs `detect-secrets-hook ... $(git ls-files)` on ubuntu,

    and `git ls-files` prints forward slashes there. A baseline regenerated on this
    Windows machine keys every finding with backslashes, so the hook matches none of
    them, reports 27 audited findings as new, and fails the job for no reason - which
    is precisely the pressure that gets a baseline regenerated without being read.
    """
    recorded = json.loads(BASELINE.read_text(encoding="utf-8"))["results"]

    for name, findings in recorded.items():
        assert "\\" not in name, f"{name} is keyed with backslashes; the ubuntu hook will miss it"
        for finding in findings:
            assert "\\" not in finding["filename"]


def excluded_file_patterns() -> list[str]:
    filters = json.loads(BASELINE.read_text(encoding="utf-8"))["filters_used"]
    return [
        pattern
        for entry in filters
        if entry["path"] == "detect_secrets.filters.regex.should_exclude_file"
        for pattern in entry["pattern"]
    ]


def test_the_path_exclusion_covers_only_the_phase_5_id_artifacts():
    """D12 of Phase 5: one narrow exclusion, on either separator, and nothing wider.

    An exclusion is the one baseline change that can hide a real secret without a
    finding ever appearing, so its scope is asserted file name by file name.
    """
    patterns = excluded_file_patterns()
    assert len(patterns) == 1
    exclusion = re.compile(patterns[0])

    excluded = [
        "data/pilot/pilot.json",
        r"data\pilot\pilot.json",
        "data/navigation/hop_run.json",
        r"data\navigation\gate_decision.json",
    ]
    scanned = [
        "data/results/run-dense-test.json",
        "data/selection/selection.json",
        "data/pilot/nested/pilot.json",
        "data/pilot/pilot.py",
        "data/extraction/extraction.jsonl.gz",
        "src/concept_embeddings_rag/config.py",
        ".env",
        "xdata/pilot/pilot.json",
        "data/replacement/x.json",
    ]
    for path in excluded:
        assert exclusion.search(path), f"{path} should be excluded"
    for path in scanned:
        assert not exclusion.search(path), f"{path} must still be scanned"


# --- Phase 6 (T20, plan D20): a content filter for deterministic hex ids, no path exclusion ------

FILTER = ROOT / ".github" / "detect_secrets_filters.py"
FILTER_PATH = "file://.github/detect_secrets_filters.py::is_deterministic_replacement_id"
REPLACEMENT = ROOT / "data" / "replacement"


def the_filter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detect_secrets_filters", FILTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HexHighEntropyString:
    """Stands in for the detector instance: the filter reads its class name only."""


class KeywordDetector:
    pass


def hex_of(length: int, seed: str = "") -> str:
    """A lowercase hex string the entropy detector flags (entropy above its 3.0 limit)."""
    counter = 0
    while True:
        value = hashlib.sha256(f"{seed}-{counter}".encode()).hexdigest()[:length]
        distinct = {char: value.count(char) for char in set(value)}
        entropy = -sum(n / length * math.log2(n / length) for n in distinct.values())
        if entropy > 3.3:
            return value
        counter += 1


def keyed(key: str, value: str, comma: bool = True) -> str:
    return f'    "{key}": "{value}"' + ("," if comma else "")


def bare(value: str, comma: bool = True) -> str:
    return f'      "{value}"' + ("," if comma else "")


def skips(filename: str, line: str, secret: str, plugin: object | None = None) -> bool:
    return the_filter().is_deterministic_replacement_id(
        filename=filename,
        line=line,
        secret=secret,
        plugin=plugin if plugin is not None else HexHighEntropyString(),
    )


ALLOWED = {
    "qid": (24,),
    "p1": (16,),
    "unit_set_hash": (16,),
    "prompt_digest": (16,),
    "corpus_cache_key": (16,),
    "question_cache_key": (16,),
    "dev_question_cache_key": (16,),
    "dev": (16,),
    "test": (16,),
    "revision": (40,),
    "resolved_revision": (40,),
    "manifest_sha256": (64,),
    "token_counts_sha256": (64,),
    "sha256": (64,),
    "expected": (16, 40),
    "observed": (16, 40),
    "digest": (64,),
    "freeze_digest": (64,),
    "supersedes": (64,),
    "node_index_digest": (64,),
    "extraction_digest": (64,),
    "pilot_digest": (64,),
    "hop_run_digest": (64,),
    "traces_digest": (64,),
    "run_digest": (64,),
    "outcomes_digest": (64,),
    "first_digest": (64,),
    "second_digest": (64,),
    "dense": (64,),
    "hybrid-bm25": (64,),
    "hybrid-entity-hop": (64,),
}

# The keys added on 2026-09-17, once the writers existed and had been run on the toy pipeline.
ADDED_FROM_THE_WRITERS = (
    "corpus_cache_key",
    "question_cache_key",
    "dev_question_cache_key",
    "dev",
    "test",
    "resolved_revision",
    "manifest_sha256",
    "token_counts_sha256",
    "expected",
    "observed",
    "supersedes",
    "outcomes_digest",
    "first_digest",
    "second_digest",
    "dense",
    "hybrid-bm25",
    "hybrid-entity-hop",
)
HEX_LENGTHS = (16, 24, 32, 40, 64)


def test_the_allowlist_is_the_one_d20_declares():
    assert {
        key: tuple(sorted(lengths)) for key, lengths in the_filter().ALLOWLIST.items()
    } == ALLOWED
    assert set(ADDED_FROM_THE_WRITERS) <= set(ALLOWED)


def test_unit_id_is_not_allowlisted_because_detect_secrets_already_skips_it():
    """Not added "just in case": the scanner's own id heuristic never flags a `unit_id` line."""
    heuristic = pytest.importorskip("detect_secrets.filters.heuristic")
    value = hex_of(16, "unit_id")

    assert "unit_id" not in the_filter().ALLOWLIST
    assert heuristic.is_likely_id_string(value, keyed("unit_id", value))


@pytest.mark.parametrize(
    ("key", "length"),
    [
        (key, length)
        for key, lengths in ALLOWED.items()
        for length in HEX_LENGTHS
        if length not in lengths
    ],
)
def test_every_allowlisted_key_at_any_other_length_is_reported(key, length):
    value = hex_of(length, f"{key}-other")
    assert not skips("data/replacement/freeze.json", keyed(key, value), value)


@pytest.mark.parametrize(
    ("key", "length"), [(key, length) for key, lengths in ALLOWED.items() for length in lengths]
)
@pytest.mark.parametrize("separator", ["/", "\\"])
def test_each_allowlisted_key_at_its_length_is_skipped(key, length, separator):
    value = hex_of(length, key)
    filename = separator.join(["data", "replacement", "freeze.json"])
    assert skips(filename, keyed(key, value), value)
    assert skips(filename, keyed(key, value, comma=False), value)


@pytest.mark.parametrize("length", [16, 24])
@pytest.mark.parametrize("separator", ["/", "\\"])
def test_a_bare_unit_id_or_qid_list_element_is_skipped(length, separator):
    value = hex_of(length, "bare")
    filename = separator.join(["data", "replacement", "outcomes-dense-dev.json"])
    assert skips(filename, bare(value), value)
    assert skips(filename, bare(value, comma=False), value)


@pytest.mark.parametrize(
    "filename",
    [
        "data/replacement/nested/freeze.json",
        "data/replacement/x.py",
        "xdata/replacement/freeze.json",
        "src/concept_embeddings_rag/config.py",
        "data/results/run-dense-dev.json",
    ],
)
def test_the_same_line_anywhere_else_is_reported(filename):
    value = hex_of(64, "path")
    assert not skips(filename, keyed("digest", value), value)


@pytest.mark.parametrize(
    ("line_of", "length"),
    [
        (lambda v: keyed("outcomes_digest_extra", v), 64),  # a key off the list
        (lambda v: keyed("hybrid-conceptual", v), 64),  # a hyphenated key off the list
        (lambda v: keyed("unit_id", v), 16),  # left to detect-secrets' own heuristic
        (lambda v: keyed("dev-", v), 16),
        (lambda v: keyed("digest", v), 40),  # a wrong length for the key
        (lambda v: keyed("qid", v), 16),
        (lambda v: keyed("secret", v), 64),
        (lambda v: bare(v), 64),  # a bare 64-character element
        (lambda v: bare(v), 40),
    ],
)
def test_a_key_off_the_list_a_wrong_length_or_a_long_bare_element_is_reported(line_of, length):
    value = hex_of(length, "shape")
    assert not skips("data/replacement/freeze.json", line_of(value), value)


# One allowed shape per group of the allowlist, the new keys among them, for the refusal checks.
SAMPLED = [
    ("digest", 64),
    ("corpus_cache_key", 16),
    ("dev", 16),
    ("resolved_revision", 40),
    ("token_counts_sha256", 64),
    ("expected", 40),
    ("observed", 16),
    ("hybrid-bm25", 64),
    ("supersedes", 64),
]


@pytest.mark.parametrize(("key", "length"), SAMPLED)
def test_uppercase_hex_is_reported(key, length):
    value = hex_of(length, f"{key}-upper").upper()
    assert not skips("data/replacement/freeze.json", keyed(key, value), value)


@pytest.mark.parametrize(("key", "length"), SAMPLED)
@pytest.mark.parametrize(
    "line_of",
    [
        lambda k, v: keyed(k, v) + ' "password": "hunter2"',
        lambda k, v: f'    "{k}": "{v}", "form": "an entity"',
        lambda k, v: f'    "note": "{k} {v}"',
        lambda k, v: f'    "{k}": "{v}x"',
    ],
)
def test_a_line_holding_anything_more_is_reported(key, length, line_of):
    value = hex_of(length, f"{key}-extra")
    assert not skips("data/replacement/traces-dev.json", line_of(key, value), value)


@pytest.mark.parametrize(("key", "length"), SAMPLED)
def test_another_detector_on_an_allowed_line_is_reported(key, length):
    value = hex_of(length, f"{key}-detector")
    line = keyed(key, value)
    assert not skips("data/replacement/freeze.json", line, value, plugin=KeywordDetector())


@pytest.mark.parametrize(("key", "length"), SAMPLED)
def test_a_secret_other_than_the_lines_value_is_reported(key, length):
    value, other = hex_of(length, f"{key}-value"), hex_of(length, f"{key}-other")
    assert not skips("data/replacement/freeze.json", keyed(key, value), other)


@pytest.mark.parametrize(("key", "length"), SAMPLED)
@pytest.mark.parametrize(
    "filename",
    ["data/replacement/nested/freeze.json", "data/replacement/x.py", "data/results/x.json"],
)
def test_a_new_key_is_skipped_only_inside_the_replacement_directory(key, length, filename):
    value = hex_of(length, f"{key}-path")
    assert skips("data/replacement/freeze.json", keyed(key, value), value)
    assert not skips(filename, keyed(key, value), value)


def test_the_filter_module_imports_only_the_standard_library():
    tree = ast.parse(FILTER.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported and imported <= set(sys.stdlib_module_names)


def test_ruff_covers_the_filter_module():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excluded = pyproject["tool"]["ruff"]["exclude"]
    assert ".github" not in excluded


def test_the_baseline_registers_exactly_this_one_custom_filter():
    filters = json.loads(BASELINE.read_text(encoding="utf-8"))["filters_used"]
    custom = [entry["path"] for entry in filters if entry["path"].startswith("file://")]
    assert custom == [FILTER_PATH]
    assert excluded_file_patterns() == ["^data[\\\\/](pilot|navigation)[\\\\/][^\\\\/]+[.]json$"]


def test_no_path_exclusion_covers_the_replacement_directory():
    exclusion = re.compile(excluded_file_patterns()[0])
    assert not exclusion.search("data/replacement/freeze.json")
    assert not exclusion.search(r"data\replacement\traces-test.json")


def test_completing_the_allowlist_widened_no_exclusion_of_the_baseline():
    """The allowlist grew; the baseline's exclusions did not: one path pattern, Phase 5's, and no
    line or secret exclusion that would reach beyond the content filter."""
    filters = json.loads(BASELINE.read_text(encoding="utf-8"))["filters_used"]
    paths = [entry["path"] for entry in filters]

    assert paths.count("detect_secrets.filters.regex.should_exclude_file") == 1
    assert "detect_secrets.filters.regex.should_exclude_line" not in paths
    assert "detect_secrets.filters.regex.should_exclude_secret" not in paths
    assert excluded_file_patterns() == ["^data[\\\\/](pilot|navigation)[\\\\/][^\\\\/]+[.]json$"]


# Which real artifacts carry each allowlisted key, taken from the writers. A key is required only
# once one of its artifacts exists, because the three stages write them in turn: `checks-dev.json`
# and the dev runs and outcomes at T21, the freeze and the dev readings at T22, the test readings
# and the decision at T23. `supersedes` carries a digest only in a superseding freeze, so it is
# required only on the deviation branch of D14 (OI-4).
IDENTITY_FILES = ("checks-dev.json", "freeze.json", "freeze-*.json")
RUN_FILES = ("run-*.json",)
ENTITY_FILES = ("run-hybrid-entity-hop-*.json", "diagnostics-*.json", "traces-*.json")
PER_QUESTION_FILES = ("outcomes-*.json", "diagnostics-*.json", "traces-*.json", "readout-*.json")
BOUND_FILES = (
    "diagnostics-*.json",
    "traces-*.json",
    "readout-*.json",
    "outcomes-*-test*.json",
    "run-*-test-*.json",
    "decision.json",
    "test-reproduction*.json",
)
SYSTEM_DIGEST_FILES = ("readout-*.json", "diagnostics-*.json", "traces-*.json", "decision.json")
WRITTEN_IN: dict[str, tuple[str, ...]] = {
    "qid": PER_QUESTION_FILES,
    "p1": ("diagnostics-*.json", "traces-*.json"),
    "unit_set_hash": IDENTITY_FILES + RUN_FILES,
    "prompt_digest": IDENTITY_FILES + ("run-hybrid-entity-hop-*.json",),
    "corpus_cache_key": IDENTITY_FILES + RUN_FILES,
    "question_cache_key": IDENTITY_FILES + RUN_FILES,
    "dev_question_cache_key": IDENTITY_FILES,
    "dev": IDENTITY_FILES,
    "test": IDENTITY_FILES,
    "revision": IDENTITY_FILES + RUN_FILES,
    "resolved_revision": IDENTITY_FILES + RUN_FILES,
    "manifest_sha256": ("checks-dev.json",),
    "token_counts_sha256": IDENTITY_FILES,
    "sha256": IDENTITY_FILES,
    "expected": IDENTITY_FILES,
    "observed": IDENTITY_FILES,
    "digest": ("*.json",),
    "freeze_digest": BOUND_FILES,
    "supersedes": ("freeze-*.json",),
    "node_index_digest": ("freeze.json", "freeze-*.json") + ENTITY_FILES,
    "extraction_digest": ("freeze.json", "freeze-*.json") + ENTITY_FILES,
    "pilot_digest": IDENTITY_FILES,
    "hop_run_digest": IDENTITY_FILES,
    "traces_digest": IDENTITY_FILES,
    "run_digest": ("checks-dev.json",),
    "outcomes_digest": IDENTITY_FILES,
    "first_digest": ("freeze.json", "freeze-*.json"),
    "second_digest": ("freeze.json", "freeze-*.json"),
    "dense": SYSTEM_DIGEST_FILES,
    "hybrid-bm25": SYSTEM_DIGEST_FILES,
    "hybrid-entity-hop": SYSTEM_DIGEST_FILES,
}


def test_every_allowlisted_key_is_declared_with_the_artifacts_that_write_it():
    """A key added to the allowlist has to say which artifact emits it, or this fails."""
    assert set(WRITTEN_IN) == set(ALLOWED)
    assert all(patterns for patterns in WRITTEN_IN.values())


def real_artifacts(patterns: tuple[str, ...]) -> list[Path]:
    if not REPLACEMENT.exists():
        return []
    return sorted({path for pattern in patterns for path in REPLACEMENT.glob(pattern)})


@pytest.mark.parametrize("key", sorted(ALLOWED))
def test_an_allowlisted_key_occurs_in_its_real_artifacts_once_they_exist(key):
    """Reconciled with the real artifacts as the stages write them (T21, T22, T23).

    The key is required only when one of the artifacts that writes it is on disk; a key no
    existing artifact emits fails, which is what keeps the allowlist from growing on guesses.
    """
    artifacts = real_artifacts(WRITTEN_IN[key])
    if not artifacts:
        pytest.skip(f"no artifact of {key} exists yet: {WRITTEN_IN[key]}")
    carriers = [
        path.name for path in artifacts if f'"{key}": "' in path.read_text(encoding="utf-8")
    ]
    assert carriers, f"{key} is allowlisted but absent from {[p.name for p in artifacts]}"


def test_the_real_artifacts_hold_no_hex_shape_the_allowlist_does_not_cover():
    """The direction the gate depends on: every hex id the stages really wrote is covered."""
    heuristic = pytest.importorskip("detect_secrets.filters.heuristic")
    artifacts = real_artifacts(("*.json",))
    if not artifacts:
        pytest.skip("data/replacement/ holds no JSON artifact yet")
    module = the_filter()
    keyed_line = re.compile(r'^\s*"(?P<key>[^"]+)": "(?P<hex>[0-9a-f]{16,})",?\s*$')
    bare_line = re.compile(r'^\s*"(?P<hex>[0-9a-f]{16,})",?\s*$')
    uncovered: set[tuple[str, int]] = set()
    bare_lengths: set[int] = set()
    for path in artifacts:
        for line in path.read_text(encoding="utf-8").splitlines():
            if (keyed_match := keyed_line.match(line)) is not None:
                shape = (keyed_match["key"], len(keyed_match["hex"]))
                covered = len(keyed_match["hex"]) in module.ALLOWLIST.get(shape[0], frozenset())
                if not covered and not heuristic.is_likely_id_string(keyed_match["hex"], line):
                    uncovered.add(shape)
            elif (bare_match := bare_line.match(line)) is not None:
                bare_lengths.add(len(bare_match["hex"]))

    assert uncovered == set(), f"hex shapes the gate would report: {sorted(uncovered)}"
    assert bare_lengths <= module.BARE_LENGTHS


def test_the_real_scanner_reports_exactly_the_cases_that_must_stay_reported(tmp_path, monkeypatch):
    """End to end: the installed scanner, configured from the real baseline, in a tree laid out
    like the repository."""
    detect_secrets = pytest.importorskip("detect_secrets")
    from detect_secrets.core.secrets_collection import SecretsCollection
    from detect_secrets.settings import transient_settings

    assert detect_secrets is not None
    (tmp_path / ".github").mkdir()
    shutil.copy(FILTER, tmp_path / ".github" / "detect_secrets_filters.py")
    skipped = {
        "data/replacement/freeze.json": keyed("freeze_digest", hex_of(64, "e2e-freeze")),
        "data/replacement/outcomes-dense-dev.json": keyed("qid", hex_of(24, "e2e-qid")),
        "data/replacement/traces-dev.json": bare(hex_of(16, "e2e-unit")),
        "data/replacement/checks-dev.json": keyed("revision", hex_of(40, "e2e-rev")),
        "data/replacement/decision.json": keyed("hybrid-entity-hop", hex_of(64, "e2e-system")),
        "data/replacement/freeze-2.json": keyed("expected", hex_of(40, "e2e-expected")),
        "data/replacement/run-dense-dev.json": keyed("corpus_cache_key", hex_of(16, "e2e-cache")),
    }
    reported = {
        "data/replacement/nested/freeze.json": keyed("digest", hex_of(64, "e2e-nested")),
        "data/replacement/x.py": f'DIGEST = "{hex_of(64, "e2e-py")}"',
        "xdata/replacement/freeze.json": keyed("digest", hex_of(64, "e2e-xdata")),
        "src/module.py": f'DIGEST = "{hex_of(64, "e2e-src")}"',
        "data/replacement/off-list.json": keyed("mystery", hex_of(64, "e2e-key")),
        "data/replacement/wrong-length.json": keyed("digest", hex_of(40, "e2e-len")),
        "data/replacement/extra.json": keyed("digest", hex_of(64, "e2e-extra")) + ' "x": 1',
        "data/replacement/bare-64.json": bare(hex_of(64, "e2e-bare")),
        "data/replacement/off-list-system.json": keyed("hybrid-conceptual", hex_of(64, "e2e-sys")),
        "data/replacement/expected-64.json": keyed("expected", hex_of(64, "e2e-exp64")),
        "data/replacement/upper.json": keyed("dense", hex_of(64, "e2e-upper").upper()),
        "data/results/run-dense-dev.json": keyed("corpus_cache_key", hex_of(16, "e2e-results")),
    }
    for relative, line in {**skipped, **reported}.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{\n" + line + "\n}\n", encoding="utf-8")

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    found = set()
    with transient_settings(
        {"plugins_used": baseline["plugins_used"], "filters_used": baseline["filters_used"]}
    ):
        collection = SecretsCollection()
        for relative in {**skipped, **reported}:
            collection.scan_file(relative)
        found = {filename.replace("\\", "/") for filename, _secret in collection}

    assert found == set(reported)
