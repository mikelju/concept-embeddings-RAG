"""T11: the rule against pickle, enforced by a scan rather than by reviewer memory.

Loading a pickle executes code, and a cached artifact is exactly what an attacker
would tamper with. `joblib.dump` is the idiomatic scikit-learn route and it is
pickle under another name, which is why the fitted estimator is never serialized
and only its learned array of atoms travels.

The scan reads source text, not the import table: `__import__("pickle")` never
appears in an import node. It does strip comments and docstrings first, so a
sentence explaining the rule does not trip the rule.
"""

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

CONCEPTS = Path(__file__).resolve().parents[2] / "src" / "concept_embeddings_rag" / "concepts"

FORBIDDEN: dict[str, str] = {
    r"\bpickle\b": "pickle: loading one executes code",
    r"\bjoblib\b": "joblib: pickle under another name",
    r"allow_pickle\s*=\s*True": "allow_pickle=True: the whole point is that it is False",
    r"\bmarshal\b": "marshal: the same hazard with a different name",
}


def _docstring_spans(source: str) -> list[tuple[int, int]]:
    """Offsets of every module, class and function docstring."""
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    spans: list[tuple[int, int]] = []
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, holders):
            continue
        head = node.body[0] if node.body else None
        if (
            isinstance(head, ast.Expr)
            and isinstance(head.value, ast.Constant)
            and isinstance(head.value.value, str)
            and head.end_lineno is not None
            and head.end_col_offset is not None
        ):
            spans.append(
                (
                    starts[head.lineno - 1] + head.col_offset,
                    starts[head.end_lineno - 1] + head.end_col_offset,
                )
            )
    return spans


def code_only(source: str) -> str:
    """The source with comments and docstrings blanked out, offsets preserved."""
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    spans = _docstring_spans(source)
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            spans.append(
                (
                    starts[token.start[0] - 1] + token.start[1],
                    starts[token.end[0] - 1] + token.end[1],
                )
            )

    masked = list(source)
    for start, end in spans:
        for index in range(start, end):
            if masked[index] != "\n":
                masked[index] = " "
    return "".join(masked)


def phase_two_sources() -> list[Path]:
    return sorted(CONCEPTS.glob("*.py"))


def test_the_scan_actually_has_sources_to_scan():
    """A scan over an empty list passes vacuously, which is the failure to avoid."""
    assert len(phase_two_sources()) >= 4


@pytest.mark.parametrize("path", phase_two_sources(), ids=lambda p: p.name)
def test_no_module_of_this_phase_reaches_for_a_pickle(path: Path):
    code = code_only(path.read_text(encoding="utf-8"))

    for pattern, reason in FORBIDDEN.items():
        assert not re.search(pattern, code), f"{path.name} uses {pattern} - {reason}"


def test_every_numpy_load_in_this_phase_disables_pickle():
    for path in phase_two_sources():
        code = code_only(path.read_text(encoding="utf-8"))
        for match in re.finditer(r"np\.load\(", code):
            tail = code[match.start() : match.start() + 400]
            assert "allow_pickle=False" in tail, f"{path.name}: np.load without allow_pickle=False"


def test_the_scan_fails_when_a_forbidden_call_is_introduced():
    """The guard has to be able to fail, or it guards nothing."""
    offending = 'import joblib\n\n\ndef save(model):\n    joblib.dump(model, "m.pkl")\n'

    code = code_only(offending)

    assert re.search(r"\bjoblib\b", code)
    assert any(re.search(pattern, code) for pattern in FORBIDDEN)


def test_a_docstring_mentioning_the_rule_does_not_trip_it():
    """The modules of this phase explain why they avoid pickle; that must be safe."""
    explaining = '"""joblib.dump is pickle under another name."""\n\nVALUE = 1\n'

    code = code_only(explaining)

    assert not any(re.search(pattern, code) for pattern in FORBIDDEN)


def test_a_comment_mentioning_the_rule_does_not_trip_it():
    explaining = "# never pickle an artifact: joblib.dump executes on load\nVALUE = 1\n"

    code = code_only(explaining)

    assert not any(re.search(pattern, code) for pattern in FORBIDDEN)


def test_a_dynamic_import_does_not_slip_past_the_scan():
    """An import-table check would miss this one; a source scan does not."""
    sneaky = 'MODULE = __import__("pickle")\n'

    code = code_only(sneaky)

    assert re.search(r"\bpickle\b", code)
