"""Phase 6, T1: the decision parameters are the spec's, held against its own text.

The spec fixes the metric, budget, split, margin, alpha, methods and the state table
before any Phase 6 number exists, and the freeze copies them from `config`. That copy is
only worth something if `config` itself says what the spec says, so every value is
asserted against a literal, and every text of the state table is read back out of
`6.spec.md` on disk: a text that drifted from the table by one character would make the
decision artifact record an open question the author never wrote.
"""

import ast
import json
from fractions import Fraction
from pathlib import Path

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag import decision_parameters as dp

SPEC = config.PROJECT_ROOT / "docs" / "plans" / "phase_6" / "6.spec.md"
SOURCES = config.PROJECT_ROOT / "src" / "concept_embeddings_rag"

EM_DASH = "\u2014"
SPEC_STATE_LABELS = {
    "ENTITY-REPLACEMENT-SUPPORTED": dp.STATE_REPLACEMENT_SUPPORTED,
    "ENTITY-PARTIAL / INVESTIGATE": dp.STATE_PARTIAL_INVESTIGATE,
    "STOP": dp.STATE_STOP,
}


def spec_text() -> str:
    return SPEC.read_text(encoding="utf-8")


def spec_state_rows(text: str) -> list[list[str]]:
    """The eight data rows of the spec's state table, as cells, read from the file."""
    lines = text.splitlines()
    header = "| S | V | N | State | Route | Recorded anomaly | Open question recorded |"
    start = lines.index(header)
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split(" | ")])
    return rows


def as_flag(cell: str) -> bool:
    assert cell in ("pass", "fail"), cell
    return cell == "pass"


def table_texts(table: tuple[dp.StateRow, ...]) -> list[str]:
    texts: list[str] = []
    for row in table:
        texts.extend(row.anomalies)
        if row.open_question is not None:
            texts.append(row.open_question)
    return texts


def missing_from_spec(table: tuple[dp.StateRow, ...], text: str) -> list[str]:
    return [piece for piece in table_texts(table) if piece not in text]


# --- Values, against literals ------------------------------------------------------------


def test_the_decision_is_full_support_at_2048_on_the_whole_test_split():
    assert dp.DECISION_METRIC == "full_support"
    assert dp.DECISION_BUDGET == 2048
    assert dp.DECISION_BUDGET in config.CONTEXT_BUDGETS
    assert dp.DECISION_SPLIT == "test"
    assert config.N_TEST == 1400


def test_delta_is_half_of_the_historical_test_effect_as_an_exact_fraction():
    delta, preserved = dp.DELTA, dp.PRESERVED_FRACTION
    assert dp.M1_NUMERATOR == 55
    assert dp.M1_DENOMINATOR == 1400
    assert preserved == Fraction(1, 2)
    assert isinstance(delta, Fraction)
    assert delta == Fraction(dp.M1_NUMERATOR, dp.M1_DENOMINATOR) * (1 - preserved)
    assert delta == Fraction(55, 2800)


def test_delta_has_no_mode_it_is_one_constant():
    """HU-2: the margin does not depend on the control mode."""
    assert not callable(dp.DELTA)


def test_the_margin_names_the_two_historical_test_files_it_derives_from():
    assert dp.M1_SOURCE_FILES == (
        "run-hybrid-bm25-test-20260911T133204+0000.json",
        "run-dense-test-20260911T133159+0000.json",
    )
    assert set(dp.M1_SOURCE_FILES) == set(config.HISTORICAL_TEST_RESULT_FILES.values())


def test_alpha_is_the_projects_only_declared_alpha():
    assert dp.DECISION_ALPHA == config.GATE_ALPHA == 0.05


def test_the_methods_and_references_are_declared():
    assert "Tango" in dp.NON_INFERIORITY_METHOD
    assert "one-sided" in dp.NON_INFERIORITY_METHOD
    assert "L > -delta" in dp.NON_INFERIORITY_METHOD
    assert dp.NON_INFERIORITY_REFERENCE.startswith("Tango, T. (1998).")
    assert "891-908" in dp.NON_INFERIORITY_REFERENCE
    assert "ties excluded" in dp.SIGN_TEST_METHOD
    assert "alternative = 'greater'" in dp.SIGN_TEST_METHOD
    assert dp.MULTIPLICITY_REFERENCE.startswith("Berger, R. L. (1982).")
    assert dp.STATE_PROCEDURE == (
        "1. S, V and N pass -> ENTITY_REPLACEMENT_SUPPORTED",
        "2. otherwise S passes -> ENTITY_PARTIAL_INVESTIGATE, route P1",
        "3. otherwise V and N pass -> ENTITY_PARTIAL_INVESTIGATE, route P2",
        "4. otherwise -> STOP",
    )


def test_the_state_and_route_names_are_the_data_contracts():
    assert dp.STATE_REPLACEMENT_SUPPORTED == "ENTITY_REPLACEMENT_SUPPORTED"
    assert dp.STATE_PARTIAL_INVESTIGATE == "ENTITY_PARTIAL_INVESTIGATE"
    assert dp.STATE_STOP == "STOP"
    assert (dp.ROUTE_P1, dp.ROUTE_P2) == ("P1", "P2")


def test_the_scenario_definitions_are_the_ones_decided_in_oi_3():
    assert dp.SCENARIO_DEFINITIONS == {
        "a": "the 90% interval contains 0 and L <= -delta",
        "b": "-delta < delta_hat < 0",
        "c": "S passes and L <= -delta",
        "d": "delta_hat > 0 and L <= 0",
        "e": "delta_hat < 0, B's Full Support count below dense's, and S fails",
    }


def test_the_qualitative_sample_rule_is_fixed_before_test():
    rule = dp.QUALITATIVE_SAMPLE_RULE
    assert rule["split"] == "test"
    assert rule["budget"] == 2048
    assert rule["per_group"] == config.TRACES_READ_PER_ARM == 5
    assert rule["order"] == "ascending question id"
    assert rule["groups"] == (
        "B succeeds where A fails",
        "A succeeds where B fails",
        "B succeeds where dense fails",
        "dense succeeds where B fails",
    )


def test_the_trace_tolerance_is_the_one_decided_in_oi_2():
    assert config.TRACE_WEIGHT_REL_TOL == 1e-12


# --- The state table, against the spec's text --------------------------------------------


def test_the_table_has_exactly_the_eight_combinations_each_once():
    keys = [(row.s, row.v, row.n) for row in dp.STATE_TABLE]
    assert len(keys) == 8
    assert len(set(keys)) == 8
    assert set(keys) == {
        (s, v, n) for s in (True, False) for v in (True, False) for n in (True, False)
    }


def test_every_row_equals_the_spec_tables_row_read_from_disk():
    rows = spec_state_rows(spec_text())
    assert len(rows) == 8

    by_key = {(row.s, row.v, row.n): row for row in dp.STATE_TABLE}
    for cells in rows:
        s, v, n, state, route, anomaly, question = cells
        row = by_key[(as_flag(s), as_flag(v), as_flag(n))]
        assert row.state == SPEC_STATE_LABELS[state]
        assert row.route == (None if route == EM_DASH else route)
        assert row.anomalies == (() if anomaly == EM_DASH else (anomaly,))
        assert row.open_question == (None if question == EM_DASH else question)


def test_the_table_is_in_the_spec_order():
    rows = spec_state_rows(spec_text())
    ordered = [(as_flag(c[0]), as_flag(c[1]), as_flag(c[2])) for c in rows]
    assert [(row.s, row.v, row.n) for row in dp.STATE_TABLE] == ordered


def test_every_anomaly_and_open_question_occurs_verbatim_in_the_spec():
    text = spec_text()
    assert table_texts(dp.STATE_TABLE), "a table with no text proves nothing"
    assert missing_from_spec(dp.STATE_TABLE, text) == []


def test_a_one_character_mutation_of_a_text_fails_the_check():
    """The negative proof: a verbatim check that cannot fail checks nothing."""
    text = spec_text()
    first = next(row for row in dp.STATE_TABLE if row.open_question is not None)
    mutated = first._replace(open_question=first.open_question.replace("\u03b4", "d", 1))
    table = tuple(mutated if row is first else row for row in dp.STATE_TABLE)

    assert mutated.open_question != first.open_question
    assert missing_from_spec(table, text) == [mutated.open_question]


def test_the_texts_keep_the_emphasis_and_the_delta_escape():
    """D21: stored as the table cell writes them, with the delta as a \\u03b4 escape."""
    questions = [row.open_question for row in dp.STATE_TABLE if row.open_question]
    assert all(question.startswith("*") for question in questions)
    assert sum("\u03b4" in question for question in questions) == 2


def test_the_table_is_serializable_as_ascii_json():
    payload = json.dumps([row._asdict() for row in dp.STATE_TABLE])
    assert payload.isascii()


@pytest.mark.parametrize("module", ["config.py", "decision_parameters.py"])
def test_the_sources_holding_the_parameters_are_ascii(module: str):
    assert (SOURCES / module).read_bytes().isascii()


def test_the_decision_split_is_named_outside_the_phase_3_selection_path():
    """D13 as implemented: `config.py` is imported by the Phase 3 selection, whose guard
    forbids a string naming a split other than dev, so the convention lives beside it."""
    tree = ast.parse((SOURCES / "config.py").read_text(encoding="utf-8"))
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert dp.DECISION_SPLIT not in strings
    assert "decision_parameters" not in imported
    assert dp.N_TEST == config.N_TEST == 1400


# --- Pinned digests, against the artifacts ------------------------------------------------


def recorded_digest(path: Path, field: str = "digest") -> str:
    if not path.exists():
        pytest.skip(f"{path.name} is not on disk")
    return str(json.loads(path.read_text(encoding="utf-8"))[field])


def test_the_pinned_selection_digest_is_the_artifacts():
    assert (
        recorded_digest(config.SELECTION_DIR / "selection.json") == config.PINNED_SELECTION_DIGEST
    )


def test_the_pinned_pilot_digest_is_the_artifacts():
    assert recorded_digest(config.PILOT_DIR / "pilot.json") == config.PINNED_PILOT_DIGEST


def test_the_pinned_hop_run_digest_is_the_artifacts():
    assert recorded_digest(config.NAVIGATION_DIR / "hop-run.json") == config.PINNED_HOP_RUN_DIGEST


def test_the_pinned_traces_digest_is_the_artifacts_and_bound_to_the_hop_run():
    path = config.NAVIGATION_DIR / "traces.json"
    assert recorded_digest(path) == config.PINNED_NAVIGATION_TRACES_DIGEST
    assert recorded_digest(path, "hop_run_digest") == config.PINNED_HOP_RUN_DIGEST


def test_the_pinned_extraction_digest_and_prompt_are_the_summarys():
    path = config.EXTRACTION_DIR / f"extraction-{config.PINNED_EXTRACTION_PROMPT_DIGEST}.json"
    assert recorded_digest(path) == config.PINNED_EXTRACTION_DIGEST
    assert recorded_digest(path, "prompt_digest") == config.PINNED_EXTRACTION_PROMPT_DIGEST


def test_the_pinned_node_index_digest_is_the_sidecars():
    path = config.NODES_DIR / f"nodes-{config.PINNED_EXTRACTION_DIGEST[:16]}.json"
    assert recorded_digest(path) == config.PINNED_NODE_INDEX_DIGEST
    assert recorded_digest(path, "extraction_digest") == config.PINNED_EXTRACTION_DIGEST
    assert recorded_digest(path, "normalization_version") == config.PINNED_NORMALIZATION_VERSION


def test_the_selection_frozen_at_is_the_one_the_spec_quotes():
    assert config.PINNED_SELECTION_FROZEN_AT == "2026-09-11T13:30:36+00:00"
    assert (
        recorded_digest(config.SELECTION_DIR / "selection.json", "frozen_at")
        == config.PINNED_SELECTION_FROZEN_AT
    )
