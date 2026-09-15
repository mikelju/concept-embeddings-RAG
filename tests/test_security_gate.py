"""The CI security gate, checked from the repository it is supposed to gate.

Both findings behind this file are the same failure in two places: a scanner that
runs, reports success, and was never looking at the thing it was supposed to cover.
A green check that means nothing is worse than no check, because it is read as
evidence.

These tests read the workflow and the baseline as data. They do not run the
scanners - CI does that - they assert that what CI runs covers what this repository
now contains.
"""

import json
import re
import tomllib
from pathlib import Path

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
    ]
    for path in excluded:
        assert exclusion.search(path), f"{path} should be excluded"
    for path in scanned:
        assert not exclusion.search(path), f"{path} must still be scanned"
