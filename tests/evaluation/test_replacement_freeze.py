"""Phase 6, T13: `Phase6Freeze`, written once on dev before test is read (HU-1, HU-5, HU-9, D13).

The freeze is what makes "the test configuration is frozen before the first test question is
evaluated" observable. Its builder accepts dev results only, so no field can hold a test figure
by construction; it records every fitted choice with its curve and per-point metrics, the
control mode and every reproduction check, the continuity checks, the identities of the inputs
(the historical files by name, sha256 and configuration, never by figure), and the decision
parameters copied from the spec. Its loader verifies the split claim first, then the digest,
the parameters, the derived selections and the supersession chain.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from freeze_fixtures import (
    a_continuity,
    a_protocol,
    a_reproducibility,
    a_reproduction,
    an_entity_component,
    control_fit,
    dev_results,
    entity_fit,
    historical_identities,
)

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.harness import RunResult
from concept_embeddings_rag.evaluation.replacement_decision import parameters_payload
from concept_embeddings_rag.evaluation.replacement_freeze import (
    FREEZE_FILENAME,
    FreezeError,
    build_freeze,
    load_freeze,
    point_label,
    reproducibility_block,
    save_freeze,
)
from concept_embeddings_rag.evaluation.selection import (
    FrozenReport,
    SelectionError,
    check_freeze_precedes,
    digest_of_payload,
)
from concept_embeddings_rag.retrieval.fusion import FusedRetriever

SELECTION = {"digest": "5" * 64, "frozen_at": "2026-09-11T13:30:36+00:00"}


@pytest.fixture(scope="module")
def pieces(tmp_path_factory):
    directory = tmp_path_factory.mktemp("dev-results")
    dev = dev_results(directory)
    return {
        "protocol": a_protocol(),
        "control_mode": "reused",
        "reproduction": a_reproduction(),
        "selection": SELECTION,
        "historical": historical_identities(),
        "history": control_fit().fit,
        "control": control_fit(),
        "entity_component": an_entity_component(),
        "continuity": a_continuity(),
        "continuity_digests": {
            "pilot_digest": "1" * 64,
            "hop_run_digest": "2" * 64,
            "traces_digest": "3" * 64,
        },
        "entity": entity_fit(),
        "dev_results": dev,
        "reproducibility": a_reproducibility(dev),
        "seed": config.DEFAULT_SEED,
        "code_version": "0.1.0",
    }


def build(pieces, **changes):
    arguments = dict(pieces)
    arguments.update(changes)
    return build_freeze(**arguments)


# --- The contract ---------------------------------------------------------------------------


def test_the_freeze_holds_every_field_of_the_contract(pieces):
    payload = build(pieces).payload
    assert set(payload) == {
        "protocol",
        "control",
        "entity_component",
        "continuity",
        "entity_fit",
        "dev_results",
        "decisions",
        "n_dev_decisions",
        "decision_parameters",
        "qualitative_sample_rule",
        "seed",
        "code_version",
        "evaluated_on",
        "frozen_at",
        "supersedes",
        "deviation",
    }
    assert payload["evaluated_on"] == "dev"
    assert payload["supersedes"] is None and payload["deviation"] is None


def test_the_protocol_holds_every_item_of_hu_1(pieces):
    protocol = build(pieces).payload["protocol"]
    for key in (
        "unit_set_hash",
        "manifest",
        "split_sizes",
        "model",
        "revision",
        "tokenizer",
        "budgets",
        "top_k",
        "metrics",
        "seed",
        "code_version",
    ):
        assert key in protocol


@pytest.mark.parametrize("missing", ["unit_set_hash", "tokenizer", "budgets", "code_version"])
def test_a_protocol_missing_an_item_is_refused(pieces, missing):
    protocol = dict(a_protocol())
    del protocol[missing]
    with pytest.raises(FreezeError, match=missing):
        build(pieces, protocol=protocol)


def test_a_top_k_other_than_the_declared_depth_is_refused(pieces):
    with pytest.raises(FreezeError, match="top_k"):
        build(pieces, protocol={**a_protocol(), "top_k": 50})


def test_the_control_block_holds_the_mode_every_check_and_the_historical_identity(pieces):
    control = build(pieces).payload["control"]
    assert control["mode"] == "reused"
    assert control["checks"] == [check.as_payload() for check in a_reproduction().checks]
    assert control["selection"] == SELECTION
    assert set(control["historical_files"]) == {
        "dense/dev",
        "dense/test",
        "hybrid-bm25/dev",
        "hybrid-bm25/test",
    }
    for identity in control["historical_files"].values():
        assert set(identity) == {"name", "sha256", "system", "split", "created_at", "config"}


def test_the_fits_hold_their_curves_derived_selections_and_per_point_metrics(pieces):
    payload = build(pieces).payload
    for block, measured in (
        (payload["control"]["fit"], pieces["control"]),
        (payload["entity_fit"], pieces["entity"]),
    ):
        fit = measured.fit
        assert block["components"] == list(fit.components)
        assert block["curve"] == {repr(w): v for w, v in fit.curve.items()}
        assert block["rrf_score"] == fit.rrf_score
        assert block["winning_scheme"] == fit.winning_scheme
        assert block["best_weight"] == fit.best_weight
        assert block["weights"] == fit.weights
        labels = {"rrf", *(f"w={w:.1f}" for w in config.FUSION_WEIGHT_GRID)}
        assert set(block["points"]) == labels
        for point in block["points"].values():
            assert point["split"] == "dev"
            assert set(point["metrics"]) >= {f"budget_{b}" for b in config.CONTEXT_BUDGETS}


def test_the_continuity_block_holds_the_four_checks_with_counts_and_the_digests(pieces):
    continuity = build(pieces).payload["continuity"]
    assert [check["name"] for check in continuity["checks"]] == [
        "a_read",
        "b_positives",
        "c_hits",
        "d_traces",
    ]
    assert all("agreements" in check and "total" in check for check in continuity["checks"])
    assert continuity["pilot_digest"] == "1" * 64
    assert continuity["passed"] is True


def test_dev_results_hold_the_tables_and_the_outcome_digests(pieces):
    dev = build(pieces).payload["dev_results"]
    assert set(dev["systems"]) == {"dense", "hybrid-bm25", "hybrid-entity-hop"}
    for system, entry in dev["systems"].items():
        result, outcomes = pieces["dev_results"][system]
        assert entry["metrics"] == result.metrics
        assert entry["outcomes_digest"] == outcomes.digest
        assert entry["split"] == "dev"
    assert dev["reproducibility"]["passed"] is True


def test_the_decision_parameters_and_the_sample_rule_are_copied_from_the_spec(pieces):
    payload = build(pieces).payload
    assert payload["decision_parameters"] == json.loads(json.dumps(parameters_payload()))
    assert payload["qualitative_sample_rule"]["per_group"] == 5


# --- Dev only ---------------------------------------------------------------------------------


def with_split(result: RunResult, split: str) -> RunResult:
    return replace(result, split=split)


def test_a_non_dev_point_result_is_refused(pieces):
    measured = entity_fit()
    points = dict(measured.points)
    points["w=0.5"] = with_split(points["w=0.5"], "test")
    with pytest.raises(FreezeError, match="dev"):
        build(pieces, entity=replace(measured, points=points))


def test_a_non_dev_dev_result_is_refused(pieces):
    dev = dict(pieces["dev_results"])
    result, outcomes = dev["dense"]
    dev["dense"] = (with_split(result, "test"), outcomes)
    with pytest.raises(FreezeError, match="dev"):
        build(pieces, dev_results=dev)


def test_a_non_dev_outcome_is_refused(pieces, tmp_path):
    from outcome_fixtures import vectors_from_counts, write_outcomes

    tests, _ = write_outcomes(tmp_path, vectors_from_counts({(1, 1, 1): 3}), split="test")
    dev = dict(pieces["dev_results"])
    dev["dense"] = (dev["dense"][0], tests["dense"])
    with pytest.raises(FreezeError, match="dev"):
        build(pieces, dev_results=dev)


def test_a_historical_identity_carrying_figures_is_refused(pieces):
    historical = historical_identities()
    historical["dense/test"]["metrics"] = {"budget_2048": {"full_support": 0.8}}
    with pytest.raises(FreezeError, match="metrics"):
        build(pieces, historical=historical)
    historical = historical_identities()
    historical["hybrid-bm25/test"]["config"]["cost"] = {"mean_latency_ms": 1.0}
    with pytest.raises(FreezeError, match="cost"):
        build(pieces, historical=historical)


def test_the_point_labels_name_rrf_and_each_weight():
    stub = [type("D", (), {"name": "dense", "retrieve": lambda self, q, k: []})()]
    stage = type("S", (), {"name": "entity-hop", "propose": lambda self, f, k: []})()
    assert point_label(FusedRetriever([stub[0], stage], scheme="rrf")) == "rrf"
    weighted = FusedRetriever(
        [stub[0], stage], scheme="weighted", weights={"dense": 0.3, "entity-hop": 0.7}
    )
    assert point_label(weighted) == "w=0.3"


# --- Decisions and the mode -------------------------------------------------------------------


def test_b_counts_two_decisions_when_weighted_wins_and_the_reused_control_none(pieces):
    freeze = build(pieces)
    assert pieces["entity"].fit.winning_scheme == "weighted"
    assert freeze.payload["n_dev_decisions"] == 2
    assert [d["hybrid"] for d in freeze.payload["decisions"]] == ["hybrid-entity-hop"] * 2


def test_b_counts_one_decision_when_rrf_wins(pieces):
    rrf = entity_fit(hits=[])
    assert rrf.fit.winning_scheme == "rrf"
    assert build(pieces, entity=rrf).payload["n_dev_decisions"] == 1


def test_the_re_measured_control_adds_its_own_decisions(pieces, tmp_path):
    deviation = tmp_path / "6.1_control_mismatch.md"
    deviation.write_text("# deviation", encoding="utf-8")
    freeze = build(
        pieces,
        control_mode="re-measured",
        reproduction=a_reproduction(passed=False),
        control_deviation=str(deviation),
    )
    assert freeze.payload["n_dev_decisions"] == 4
    assert [d["hybrid"] for d in freeze.payload["decisions"]] == ["hybrid-entity-hop"] * 2 + [
        "hybrid-bm25"
    ] * 2
    assert freeze.payload["control"]["deviation"] == str(deviation)


def test_a_mode_that_disagrees_with_the_checks_is_refused(pieces):
    with pytest.raises(FreezeError, match="mode"):
        build(pieces, control_mode="reused", reproduction=a_reproduction(passed=False))
    with pytest.raises(FreezeError, match="mode"):
        build(pieces, control_mode="re-measured", reproduction=a_reproduction(passed=True))


def test_re_measured_without_a_deviation_document_is_refused(pieces, tmp_path):
    with pytest.raises(FreezeError, match="deviation"):
        build(pieces, control_mode="re-measured", reproduction=a_reproduction(passed=False))
    with pytest.raises(FreezeError, match="deviation"):
        build(
            pieces,
            control_mode="re-measured",
            reproduction=a_reproduction(passed=False),
            control_deviation=str(tmp_path / "missing.md"),
        )


def test_a_reused_control_whose_fit_differs_from_the_history_is_refused(pieces):
    other = entity_fit(hits=[]).fit
    history = type(pieces["history"])(
        **{**pieces["history"].__dict__, "rrf_score": other.rrf_score + 1}
    )
    with pytest.raises(FreezeError, match="history"):
        build(pieces, history=history)


def test_a_failed_continuity_or_reproducibility_is_refused(pieces):
    with pytest.raises(FreezeError, match="continuity"):
        build(pieces, continuity=a_continuity(passed=False))
    broken = dict(pieces["reproducibility"])
    broken["passed"] = False
    with pytest.raises(FreezeError, match="reproducib"):
        build(pieces, reproducibility=broken)


def test_the_reproducibility_block_names_differing_outcomes(pieces, tmp_path):
    from outcome_fixtures import vectors_from_counts, write_outcomes

    first = {system: artifact for system, (_r, artifact) in pieces["dev_results"].items()}
    other, _ = write_outcomes(
        tmp_path,
        vectors_from_counts({(1, 1, 1): 5, (0, 1, 1): 3, (0, 0, 1): 1, (0, 0, 0): 1}),
        split="dev",
        freeze_digest=None,
    )
    block = reproducibility_block(first, other)
    assert block["passed"] is False
    assert block["systems"]["dense"]["identical"] is False
    assert block["systems"]["dense"]["differing_questions"] >= 1


# --- Written once, verified on load -----------------------------------------------------------


def test_the_freeze_round_trips_and_satisfies_frozen_report(pieces, tmp_path):
    freeze = build(pieces)
    path = save_freeze(freeze, tmp_path)
    assert path.name == FREEZE_FILENAME

    loaded = load_freeze(tmp_path)
    assert loaded.digest == digest_of_payload(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.payload["control"]["mode"] == "reused"
    assert isinstance(loaded, FrozenReport)
    assert path.read_bytes().isascii()


def test_check_freeze_precedes_accepts_a_later_result_and_refuses_an_earlier_one(pieces, tmp_path):
    save_freeze(build(pieces, frozen_at="2026-09-17T10:00:00+00:00"), tmp_path)
    loaded = load_freeze(tmp_path)
    result, _ = pieces["dev_results"]["dense"]
    check_freeze_precedes(loaded, replace(result, created_at="2026-09-17T10:00:01+00:00"))
    with pytest.raises(SelectionError):
        check_freeze_precedes(loaded, replace(result, created_at="2026-09-17T09:59:59+00:00"))


def test_overwriting_the_freeze_is_refused(pieces, tmp_path):
    save_freeze(build(pieces), tmp_path)
    with pytest.raises(FreezeError, match="already"):
        save_freeze(build(pieces), tmp_path)


def resealed(path: Path, change) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    payload["digest"] = digest_of_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_a_freeze_claiming_another_split_is_refused_before_its_digest(pieces, tmp_path):
    path = save_freeze(build(pieces), tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluated_on"] = "test"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FreezeError, match="dev"):
        load_freeze(tmp_path)


def test_a_modified_freeze_is_refused_by_its_digest(pieces, tmp_path):
    path = save_freeze(build(pieces), tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seed"] = 7
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FreezeError, match="digest"):
        load_freeze(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["decision_parameters"].update({"budget": 4096}),
        lambda p: p["decision_parameters"]["delta"].update({"numerator": 56}),
        lambda p: p["decision_parameters"].update({"alpha": 0.025}),
        lambda p: p["decision_parameters"]["state_table"][4].update({"route": "P1"}),
        lambda p: p["qualitative_sample_rule"].update({"per_group": 10}),
    ],
)
def test_a_decision_parameter_copy_differing_from_the_spec_is_refused_on_load(
    pieces, tmp_path, change
):
    path = save_freeze(build(pieces), tmp_path)
    resealed(path, change)
    with pytest.raises(FreezeError, match="decision parameters|sample rule"):
        load_freeze(tmp_path)


def test_a_stored_selection_that_does_not_re_derive_from_its_curve_is_refused(pieces, tmp_path):
    path = save_freeze(build(pieces), tmp_path)
    resealed(path, lambda p: p["entity_fit"].update({"best_weight": 0.9}))
    with pytest.raises(FreezeError, match="re-derive"):
        load_freeze(tmp_path)


def test_a_ledger_count_that_disagrees_with_the_rule_is_refused(pieces, tmp_path):
    path = save_freeze(build(pieces), tmp_path)
    resealed(path, lambda p: p.update({"n_dev_decisions": 3}))
    with pytest.raises(FreezeError, match="decisions"):
        load_freeze(tmp_path)


# --- Supersession (OI-4) ----------------------------------------------------------------------


def test_a_second_freeze_is_refused_without_supersedes_and_an_existing_deviation(pieces, tmp_path):
    first = save_freeze(build(pieces), tmp_path)
    first_digest = load_freeze(tmp_path).digest

    with pytest.raises(FreezeError, match="supersede"):
        save_freeze(build(pieces, deviation="docs/x.md"), tmp_path)
    with pytest.raises(FreezeError, match="deviation"):
        save_freeze(build(pieces, supersedes=first_digest), tmp_path)
    with pytest.raises(FreezeError, match="deviation"):
        save_freeze(
            build(pieces, supersedes=first_digest, deviation=str(tmp_path / "missing.md")), tmp_path
        )
    assert first.exists()


def test_a_superseding_freeze_names_the_latest_digest_and_becomes_the_valid_one(pieces, tmp_path):
    save_freeze(build(pieces), tmp_path)
    first_digest = load_freeze(tmp_path).digest
    deviation = tmp_path / "6.2_test_mismatch.md"
    deviation.write_text("# deviation", encoding="utf-8")

    with pytest.raises(FreezeError, match="supersede"):
        save_freeze(build(pieces, supersedes="0" * 64, deviation=str(deviation)), tmp_path)

    second = save_freeze(build(pieces, supersedes=first_digest, deviation=str(deviation)), tmp_path)
    assert second.name == "freeze-2.json"
    loaded = load_freeze(tmp_path)
    assert loaded.payload["supersedes"] == first_digest
    assert loaded.path == second


def test_a_broken_supersession_chain_is_refused_on_load(pieces, tmp_path):
    save_freeze(build(pieces), tmp_path)
    first_digest = load_freeze(tmp_path).digest
    deviation = tmp_path / "6.2_test_mismatch.md"
    deviation.write_text("# deviation", encoding="utf-8")
    second = save_freeze(build(pieces, supersedes=first_digest, deviation=str(deviation)), tmp_path)
    resealed(second, lambda p: p.update({"supersedes": "0" * 64}))
    with pytest.raises(FreezeError, match="chain"):
        load_freeze(tmp_path)


def test_loading_with_no_freeze_names_the_stage(tmp_path):
    with pytest.raises(FreezeError, match="replace-freeze"):
        load_freeze(tmp_path)
