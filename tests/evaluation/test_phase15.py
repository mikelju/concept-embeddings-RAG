"""Phase 15, S4: the D4 verdicts, the frozen-weight reader and the live path's checks.

What can change a Phase 15 result here: which weights and `alpha` the four systems fuse with
(the reader refuses anything but the spec's table), whether the HotpotQA dev counts are
reproduced question by question, whether the pod's extractor is the Phase 9 one, and whether
the live path's ranking records are well formed, without a metric of a MuSiQue question ever
being computed. Everything is fixture-based and fast.

S5: the pass's rankings file is written once, without a timestamp, and read back only against
the digest its run records; the pass marker is written once; the hop lists at the five `alpha`
must equal the recorded P10-C and P14 lists and ride in P14's records through serialization.
"""

import gzip
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation import phase14, phase15
from concept_embeddings_rag.evaluation.entity_diagnostics import RecordingHybrid, RecordingRetriever
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.fusion import WEIGHTED

P10B, P10C, P14 = config.PHASE_15_SYSTEMS[1:]
N_DEV = config.PHASE_15_DEV_QUESTIONS

# --- The frozen-weight reader (D3, R4) -----------------------------------------------------

P10B_RECORDED = {"dense": 0.5, "bm25": 0.5}
P10C_FIT = {"terminal_state": None, "weights": {"dense": 0.5, "bm25": 0.3, "entity-hop": 0.2}}
P14_FIT = {
    "terminal_state": None,
    "alpha": 0.75,
    "weights": {"dense": 0.5, "bm25": 0.3, "relevance-hop": 0.2},
}


def fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    p10b: dict[str, float] = P10B_RECORDED,
    p10c: dict[str, Any] = P10C_FIT,
    p14: dict[str, Any] = P14_FIT,
) -> dict[str, Path]:
    """Both recorded fits in `tmp_path`, and Phase 9's reader answering with `p10b`."""
    paths = {"phase10_fit": tmp_path / "phase10-fit.json", "phase14_fit": tmp_path / "fit.json"}
    paths["phase10_fit"].write_text(json.dumps(p10c), encoding="utf-8")
    paths["phase14_fit"].write_text(json.dumps(p14), encoding="utf-8")
    monkeypatch.setattr(
        phase9,
        "read_frozen_weights",
        lambda *a, **k: {"hybrid-bm25": dict(p10b), "hybrid-entity-hop": {"dense": 0.7}},
    )
    return paths


def committed(path: Path) -> bool:
    return True


def test_the_reader_returns_the_spec_table_keyed_as_each_system_fuses(tmp_path, monkeypatch):
    paths = fits(tmp_path, monkeypatch)

    frozen = phase15.frozen_weights(is_committed=committed, **paths)

    assert frozen["weights"] == {
        P10B: {"dense": 0.5, "bm25": 0.5},
        P10C: {"dense": 0.5, "bm25": 0.3, "entity-hop": 0.2},
        P14: {"dense": 0.5, "bm25": 0.3, "seeded-hop": 0.2},
    }
    assert frozen["alpha"] == 0.75
    assert frozen["fit_digests"][P14] == phase15.fit_digest(P14_FIT)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"p10b": {"dense": 0.6, "bm25": 0.4}}, "P10-B"),
        (
            {"p10c": {**P10C_FIT, "weights": {"dense": 0.4, "bm25": 0.4, "entity-hop": 0.2}}},
            "P10-C",
        ),
        ({"p14": {**P14_FIT, "alpha": 0.5}}, "alpha"),
        ({"p14": {**P14_FIT, "weights": {"dense": 0.6, "bm25": 0.2, "relevance-hop": 0.2}}}, "P14"),
        ({"p14": {**P14_FIT, "weights": {"dense": 0.5, "bm25": 0.3, "hop": 0.2}}}, "P14"),
    ],
)
def test_the_reader_refuses_a_fit_whose_values_differ_from_the_spec_table(
    change: dict[str, Any], message: str, tmp_path, monkeypatch
):
    paths = fits(tmp_path, monkeypatch, **change)

    with pytest.raises(phase15.Phase15Error, match=message):
        phase15.frozen_weights(is_committed=committed, **paths)


def test_the_reader_refuses_a_fit_git_does_not_track_unmodified(tmp_path, monkeypatch):
    paths = fits(tmp_path, monkeypatch)

    with pytest.raises(phase15.Phase15Error, match="committed"):
        phase15.frozen_weights(is_committed=lambda path: path != paths["phase14_fit"], **paths)


def test_the_reader_refuses_a_fit_that_recorded_a_stop(tmp_path, monkeypatch):
    paths = fits(tmp_path, monkeypatch, p10c={**P10C_FIT, "terminal_state": "DEV_STOP"})

    with pytest.raises(phase15.Phase15Error, match="DEV_STOP"):
        phase15.frozen_weights(is_committed=committed, **paths)


# --- D4: code identity and the verdict ------------------------------------------------------


def outcomes(supported: int) -> dict[str, float]:
    """Full Support @2,048 over the 7,405 dev qids: the first `supported` questions pass."""
    return {f"q{i:05d}": float(i < supported) for i in range(N_DEV)}


def reproduced() -> dict[str, dict[str, float]]:
    return {
        P10C: outcomes(config.PHASE_15_P10C_DEV_SUPPORTED),
        P14: outcomes(config.PHASE_15_P14_DEV_SUPPORTED),
    }


PASSING_EXTRACTOR = {"passed": True}
PASSING_LIVE = {"passed": True, "systems": {}}


def test_the_d4_verdict_passes_on_the_recorded_counts_with_no_question_moved():
    code = phase15.code_identity_verdict(reproduced(), reproduced())

    assert code["passed"] is True
    assert code["checks"][P10C]["observed"] == 4801
    assert code["checks"][P14]["observed"] == 5224
    assert code["checks"][P14]["differing_qids"] == []
    verdict = phase15.d4_verdict(code, PASSING_EXTRACTOR, PASSING_LIVE)
    assert verdict == {"terminal_state": None, "stop_reasons": []}


def test_a_one_question_miss_is_a_data_stop_that_names_the_question():
    observed = reproduced()
    observed[P14]["q00017"] = 0.0

    code = phase15.code_identity_verdict(observed, reproduced())

    check = code["checks"][P14]
    assert (check["observed"], check["expected"]) == (5223, 5224)
    assert check["differing_qids"] == ["q00017"]
    assert code["passed"] is False and code["checks"][P10C]["passed"] is True
    verdict = phase15.d4_verdict(code, PASSING_EXTRACTOR, PASSING_LIVE)
    assert verdict["terminal_state"] == phase9.DATA_STOP
    assert any("q00017" in reason for reason in verdict["stop_reasons"])


def test_a_question_moved_between_the_two_routes_fails_even_at_the_recorded_count():
    reference = reproduced()
    reference[P10C]["q00000"], reference[P10C]["q07000"] = 0.0, 1.0

    code = phase15.code_identity_verdict(reproduced(), reference)

    assert code["checks"][P10C]["observed"] == 4801
    assert code["checks"][P10C]["differing_qids"] == ["q00000", "q07000"]
    assert code["passed"] is False


def test_an_extractor_or_live_path_miss_alone_is_a_data_stop():
    code = phase15.code_identity_verdict(reproduced(), reproduced())
    for extractor, live in (
        ({"passed": False}, PASSING_LIVE),
        (PASSING_EXTRACTOR, {"passed": False, "systems": {}}),
    ):
        assert phase15.d4_verdict(code, extractor, live)["terminal_state"] == phase9.DATA_STOP


# --- D4: extractor identity -----------------------------------------------------------------

POD = config.PHASE_9_GLINER_CONFIGURATION_DIGEST
VERSIONS = {"gliner": "0.2.29", "torch": "2.13.0+cu126", "transformers": "5.16.1"}


def manifest(digest: str = POD, **versions: str) -> dict[str, Any]:
    return {"configuration_digest": digest, "library_versions": {**VERSIONS, **versions}}


def test_the_extractor_is_the_phase_9_pod_one():
    verdict = phase15.extractor_identity_verdict(manifest(), manifest())

    assert verdict["passed"] is True
    assert verdict["library_versions_differing"] == []


def test_the_laptop_digest_is_not_the_pod_one():
    verdict = phase15.extractor_identity_verdict(manifest("390d0d8ae603fd1d"), manifest())

    assert verdict["passed"] is False


def test_a_library_version_differing_from_phase_9_fails_key_by_key():
    verdict = phase15.extractor_identity_verdict(manifest(torch="2.13.0+cpu"), manifest())

    assert verdict["passed"] is False
    assert verdict["library_versions_differing"] == ["torch"]


# --- The ranking record and its checks ------------------------------------------------------

CORPUS = {f"u{i:03d}" for i in range(200)}


def ranked(n: int) -> list[Hit]:
    return [(f"u{i:03d}", 1.0 - i / 1000.0) for i in range(n)]


def test_a_well_formed_ranking_has_no_problem():
    assert phase15.ranking_problems(ranked(100), CORPUS, depth=100) == []
    assert phase15.ranking_problems([], CORPUS, depth=100) == []


@pytest.mark.parametrize(
    ("hits", "problem"),
    [
        (ranked(3) + [("u001", 0.5)], "twice"),
        (ranked(101), "101"),
        (ranked(3) + [("elsewhere", 0.1)], "corpus"),
        ([("u000", 0.1), ("u001", 0.2)], "descending"),
    ],
)
def test_a_malformed_ranking_is_reported(hits: list[Hit], problem: str):
    problems = phase15.ranking_problems(hits, CORPUS, depth=100)

    assert any(problem in text for text in problems)


def test_the_ranking_records_round_trip_through_their_serialization():
    records = [
        {
            "qid": "t1",
            "system": P14,
            "fused": ranked(3),
            "components": {"dense": ranked(2), "bm25": [], "relevance-hop": ranked(1)},
            "hop": {"p1": "u000", "positives": 7, "sim_seconds": 0.001},
        }
    ]

    assert phase15.parse_rankings(phase15.serialize_rankings(records)) == records


# --- The live path --------------------------------------------------------------------------


class Stub:
    """A retriever answering every query with a fixed list, or failing on one query."""

    def __init__(self, name: str, hits: Sequence[Hit], fail_on: str | None = None) -> None:
        self.name = name
        self.hits = list(hits)
        self.fail_on = fail_on

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        if query == self.fail_on:
            raise RuntimeError("the index is gone")
        return list(self.hits[:top_k])


def live_systems(bm25_hits: Sequence[Hit] = ranked(5), fail_on: str | None = None):
    dense = Stub("dense", ranked(5), fail_on)
    alone = RecordingRetriever(dense)
    hybrid = RecordingHybrid(dense, Stub("bm25", bm25_hits), scheme=WEIGHTED, weights=P10B_RECORDED)
    return {
        "dense": phase15.LiveSystem("dense", alone, {"dense": alone}),
        P10B: phase15.LiveSystem(
            P10B,
            hybrid,
            {"dense": hybrid.dense, "bm25": hybrid.second},  # type: ignore[dict-item]
            fusion_weights=(0.5, 0.5),
        ),
    }


def live_questions(n: int = 3) -> list[Question]:
    return [Question(f"t{i}", f"train question {i}?", "", (), (), "live") for i in range(n)]


METRIC_KEYS = {
    "budgets",
    "full_support",
    "gold_recall",
    "context_precision",
    "fs_at_k",
    "gpr_at_k",
    "supported",
    "recall",
    "gold",
}


def keys_of(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_of(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_of(v)}
    return set()


def no_measure(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("no metric of a live-path question may be computed")


def test_the_live_path_checks_the_records_and_computes_no_metric(monkeypatch):
    monkeypatch.setattr(phase9, "measure_system", no_measure)

    result = phase15.run_live_path(live_systems(), live_questions(), CORPUS, depth=100)

    assert result["passed"] is True
    assert result["qids"] == ["t0", "t1", "t2"]
    for system in result["systems"].values():
        assert system["completed"] is True
        assert system["ranking_problems"] == []
        assert system["round_trip"] is True and system["fusion_consistent"] is True
        assert set(system["seconds"]) == {"total", "mean", "max"}
    assert not keys_of(result) & METRIC_KEYS


def test_a_component_ranking_outside_the_corpus_fails_the_live_path(monkeypatch):
    monkeypatch.setattr(phase9, "measure_system", no_measure)

    result = phase15.run_live_path(
        live_systems(bm25_hits=[*ranked(3), ("elsewhere", 0.0)]),
        live_questions(),
        CORPUS,
        depth=100,
    )

    assert result["passed"] is False
    problems = result["systems"][P10B]["ranking_problems"]
    assert any(text.startswith("t0 bm25:") and "corpus" in text for text in problems)
    assert result["systems"]["dense"]["passed"] is True


def test_a_component_listing_a_unit_twice_fails_the_live_path():
    # The fusion refuses a repeated unit mid-call, so the system does not complete.
    result = phase15.run_live_path(
        live_systems(bm25_hits=[*ranked(3), ("u000", 0.0)]), live_questions(), CORPUS, depth=100
    )

    assert result["passed"] is False
    assert result["systems"][P10B]["completed"] is False
    assert "twice" in result["systems"][P10B]["error"]


def test_a_call_that_does_not_complete_fails_the_live_path_and_names_the_question():
    result = phase15.run_live_path(
        live_systems(fail_on="train question 1?"), live_questions(), CORPUS, depth=100
    )

    assert result["passed"] is False
    dense = result["systems"]["dense"]
    assert dense["completed"] is False and dense["failed_qid"] == "t1"
    assert "the index is gone" in dense["error"]


def test_the_live_path_refuses_a_question_that_carries_gold():
    questions = [Question("v1", "validation?", "", ("u000", "u001"), (), "musique-validation")]

    with pytest.raises(phase15.Phase15Error, match="gold"):
        phase15.run_live_path(live_systems(), questions, CORPUS, depth=100)


# --- S5: the pass's rankings file, its marker and the hop lists at the five alpha ------------

ZERO, FROZEN_KEY = phase14.alpha_key(0.0), phase14.alpha_key(0.75)


def record(qid: str, system: str, hop_name: str, hop: list[Hit]) -> dict[str, Any]:
    return {
        "qid": qid,
        "system": system,
        "fused": ranked(3),
        "components": {"dense": ranked(3), "bm25": ranked(2), hop_name: hop},
        "hop": {"p1": "u000", "positives": len(hop)},
    }


def test_the_rankings_file_is_written_once_without_a_timestamp_and_matches_its_run(
    tmp_path: Path,
):
    records = [record("v1", P14, "relevance-hop", ranked(2))]

    written = phase15.write_rankings(tmp_path, P14, records)
    phase9.write_run(tmp_path, P14, [], body=written)

    path = tmp_path / f"rankings-{P14}.jsonl.gz"
    assert written["rankings_file"] == path.name
    assert path.read_bytes()[4:8] == b"\x00\x00\x00\x00"  # gzip mtime = 0
    assert phase15.load_rankings(tmp_path, P14) == records
    with pytest.raises(phase15.Phase15Error, match="written once"):
        phase15.write_rankings(tmp_path, P14, records)


def test_a_rankings_file_that_does_not_match_its_run_digest_is_refused(tmp_path: Path):
    phase9.write_run(
        tmp_path, P14, [], body=phase15.write_rankings(tmp_path, P14, [record("v1", P14, "h", [])])
    )
    path = tmp_path / f"rankings-{P14}.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"qid": "v2"}\n', mtime=0))

    with pytest.raises(phase15.Phase15Error, match="digest"):
        phase15.load_rankings(tmp_path, P14)


def test_the_pass_marker_is_written_once(tmp_path: Path):
    path = phase15.start_pass(tmp_path, {"code_commit": "abc"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"code_commit": "abc"}
    with pytest.raises(phase15.Phase15Error, match="already started"):
        phase15.start_pass(tmp_path, {"code_commit": "def"})


def test_the_hop_lists_must_equal_the_recorded_p10c_and_p14_lists():
    p10c = [record("v1", P10C, "entity-hop", ranked(2)), record("v2", P10C, "entity-hop", [])]
    p14 = [
        record("v1", P14, "relevance-hop", ranked(2)[::-1]),
        record("v2", P14, "relevance-hop", []),
    ]
    agreeing = [{ZERO: ranked(2), FROZEN_KEY: ranked(2)[::-1]}, {ZERO: [], FROZEN_KEY: []}]
    moved = [{ZERO: ranked(2)[::-1], FROZEN_KEY: ranked(2)}, {ZERO: [], FROZEN_KEY: []}]

    assert phase15.hop_alpha_mismatches(p14, p10c, agreeing, alpha=0.75) == []
    assert phase15.hop_alpha_mismatches(p14, p10c, moved, alpha=0.75) == [
        "v1 alpha=0.00",
        "v1 alpha=0.75",
    ]


def test_the_hop_lists_ride_in_p14s_records_and_survive_serialization():
    p14 = [record("v1", P14, "relevance-hop", ranked(2))]
    lists = [{key: ranked(1) for key in (ZERO, FROZEN_KEY)}]

    stored = phase15.with_hop_alphas(p14, lists)

    assert stored[0]["hop_alphas"] == lists[0] and "hop_alphas" not in p14[0]
    assert list(stored[0]["components"]) == ["dense", "bm25", "relevance-hop"]
    assert phase15.parse_rankings(phase15.serialize_rankings(stored)) == stored
