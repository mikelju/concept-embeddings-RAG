"""Phase 6, T19: "the same opportunity to contribute", asserted on the code that runs (HU-4, HU-5).

Phase 3's parity tests compare the control's and System B's fitting calls in `cli.py` keyword by
keyword. Phase 6 keeps both hybrids out of `cli.py` (D10): every fit goes through one helper,
`fit_hybrid`, which holds the one `fit_fusion_weight` call, and every hybrid through one helper,
`build_hybrid`. The same comparison is applied to their call sites in `replacement_run.py`, and
shown to fail on a tampered source. At runtime, on the toy pipeline, both fits run over the same
grid on the same questions at `top_k = 100`, both hybrids fuse dense first through `fuse` with
`RRF_K`, and the stage is asked for the depth the dense component is.

Implementation note (T19): the plan wrote "the two `fit_fusion_weight` calls in
`replacement_run.py`"; the implementation has one call inside `fit_hybrid` and two call sites of the
helper, which is the stronger form of the same parity. Recorded under D10.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from replacement_fixtures import build_toy_pipeline, toy_counter  # noqa: E402

from concept_embeddings_rag import config  # noqa: E402
from concept_embeddings_rag.evaluation import replacement_run  # noqa: E402
from concept_embeddings_rag.evaluation.replacement_freeze import load_freeze  # noqa: E402
from concept_embeddings_rag.evaluation.replacement_run import (  # noqa: E402
    StageEnvironment,
    run_check,
    run_freeze,
)
from concept_embeddings_rag.retrieval import fusion  # noqa: E402

RUN = Path(replacement_run.__file__)
DIAGNOSTICS = RUN.parent / "entity_diagnostics.py"


def calls(source: str, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


def shape(call: ast.Call) -> tuple[int, frozenset[str]]:
    return len(call.args), frozenset(keyword.arg or "**" for keyword in call.keywords)


def run_source() -> str:
    return RUN.read_text(encoding="utf-8")


# --- Structural parity ------------------------------------------------------------------------


def test_one_fit_fusion_weight_call_serves_every_hybrid():
    assert len(calls(run_source(), "fit_fusion_weight")) == 1
    tree = ast.parse(run_source())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit_hybrid")
    assert len(calls(ast.unparse(helper), "fit_fusion_weight")) == 1


def test_the_two_fitting_call_sites_pass_the_same_keywords():
    sites = calls(run_source(), "fit_hybrid")
    assert len(sites) == 2, "the control and B are fitted, and nothing else"
    assert shape(sites[0]) == shape(sites[1])
    assert shape(sites[0])[1] == frozenset({"questions", "token_counts", "run_config"})


def test_the_fitting_check_fails_on_a_control_given_a_grid_of_its_own():
    clean = run_source()
    anchor = "    control_fit = fit_hybrid(\n        dense, bm25,"
    assert anchor in clean
    tampered = clean.replace(
        anchor, "    control_fit = fit_hybrid(\n        dense, bm25, grid=(0.5,),", 1
    )
    sites = calls(tampered, "fit_hybrid")
    assert shape(sites[0]) != shape(sites[1])


def test_every_hybrid_construction_goes_through_one_helper_with_the_same_arguments():
    sites = calls(run_source(), "build_hybrid")
    assert len(sites) >= 2
    assert {shape(site) for site in sites} == {(4, frozenset())}
    assert len(calls(run_source(), "FusedRetriever")) == 0
    assert len(calls(run_source(), "RecordingHybrid")) == 1
    diagnostics = DIAGNOSTICS.read_text(encoding="utf-8")
    constructions = calls(diagnostics, "FusedRetriever")
    assert len(constructions) == 1
    assert shape(constructions[0])[1] == frozenset({"scheme", "weights"})


def test_the_construction_check_fails_on_a_hybrid_built_with_its_own_keyword():
    clean = run_source()
    anchor = "entity = build_hybrid(dense, stage, fit.winning_scheme, fit.weights)"
    assert anchor in clean
    tampered = clean.replace(
        anchor, "entity = build_hybrid(dense, stage, fit.winning_scheme, fit.weights, top_k=50)", 1
    )
    assert len({shape(site) for site in calls(tampered, "build_hybrid")}) == 2


def test_both_hybrids_are_measured_by_the_one_harness_call_at_the_declared_depth():
    source = run_source()
    harness_calls = calls(source, "evaluate_retriever")
    tree = ast.parse(source)
    owners = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and calls(ast.unparse(node), "evaluate_retriever")
    }
    assert owners == {"measure", "rerun"}
    for call in harness_calls:
        top_k = next(keyword for keyword in call.keywords if keyword.arg == "top_k")
        assert ast.unparse(top_k.value) == "config.EVALUATION_TOP_K"


# --- Runtime parity on the toy pipeline -------------------------------------------------------


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    pipeline = build_toy_pipeline(tmp_path_factory.mktemp("parity") / "toy")
    env = StageEnvironment(
        paths=pipeline.paths,
        pins=pipeline.pins,
        replacement_dir=pipeline.replacement_dir,
        backend=pipeline.backend,
        token_counter=toy_counter,
    )
    fused: list[tuple[str, ...]] = []
    original = fusion.fuse

    def spy(components, **kwargs):
        fused.append(tuple(kwargs.items()))
        return original(components, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fusion, "fuse", spy)
        assert run_check(env).passed
        assert run_freeze(env).passed
    return pipeline, load_freeze(pipeline.replacement_dir), fused


def test_both_fits_use_the_same_grid_questions_metric_and_budget(frozen):
    _pipeline, freeze, _fused = frozen
    control, entity = freeze.payload["control"]["fit"], freeze.payload["entity_fit"]
    for key in ("grid", "metric", "budget", "n_questions"):
        assert control[key] == entity[key], key
    assert control["grid"] == list(config.FUSION_WEIGHT_GRID)
    assert control["metric"] == config.SELECTION_METRIC
    assert control["budget"] == config.SELECTION_BUDGET


def test_both_hybrids_fuse_dense_first(frozen):
    _pipeline, freeze, _fused = frozen
    assert freeze.payload["control"]["fit"]["components"] == ["dense", "bm25"]
    assert freeze.payload["entity_fit"]["components"] == ["dense", "entity-hop"]


def test_every_measurement_ran_at_the_declared_top_k(frozen):
    _pipeline, freeze, _fused = frozen
    for block in (freeze.payload["control"]["fit"], freeze.payload["entity_fit"]):
        for point in block["points"].values():
            assert point["metrics"]
    for entry in freeze.payload["dev_results"]["systems"].values():
        assert entry["config"]["top_k"] == config.EVALUATION_TOP_K == 100


def test_both_hybrids_go_through_fuse_with_the_same_arguments(frozen):
    _pipeline, _freeze, fused = frozen
    keywords = {tuple(name for name, _ in call) for call in fused}
    assert keywords == {("scheme", "top_k", "weights")}
    assert {dict(call)["top_k"] for call in fused} == {config.EVALUATION_TOP_K}
    assert config.RRF_K == 60


def test_the_stage_is_asked_for_the_depth_dense_is_asked_for(frozen, monkeypatch):
    pipeline, freeze, _fused = frozen
    from concept_embeddings_rag.evaluation.replacement_run import build_components, verified_inputs

    env = StageEnvironment(
        paths=pipeline.paths,
        pins=pipeline.pins,
        replacement_dir=pipeline.replacement_dir,
        backend=pipeline.backend,
        token_counter=toy_counter,
    )
    inputs = verified_inputs(env)
    dense, _bm25, stage = build_components(inputs, inputs.dev_query_backend)
    asked: list[tuple[str, int]] = []
    original_dense, original_stage = dense.retrieve, stage.expand

    def dense_spy(query, top_k):
        asked.append(("dense", top_k))
        return original_dense(query, top_k)

    def stage_spy(first, top_k):
        # The recording wrapper reads the whole expansion; `propose` is its projection.
        asked.append(("stage", top_k))
        return original_stage(first, top_k)

    monkeypatch.setattr(dense, "retrieve", dense_spy)
    monkeypatch.setattr(stage, "expand", stage_spy)
    fit = freeze.payload["entity_fit"]
    hybrid = replacement_run.build_hybrid(dense, stage, fit["winning_scheme"], fit["weights"])
    question = next(q for q in inputs.questions if q.split == "dev")
    hybrid.retrieve(question.question, config.EVALUATION_TOP_K)
    assert asked == [("dense", 100), ("stage", 100)]
