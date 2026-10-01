"""Phase 17, S1: the pool, the reorder rule, the union line, D6's metrics, D5's comparisons
and label, the shards and the write-once files, on hand-built fixtures."""

import gzip
import hashlib
import math

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation import phase10, phase17
from concept_embeddings_rag.evaluation.entity_diagnostics import RecordingRetriever

P10A, P10B, P10C, P14 = config.PHASE_17_SYSTEMS
LIGHT, STRONG = config.PHASE_17_LIGHT, config.PHASE_17_STRONG


# --- The pool (D1) -------------------------------------------------------------------------


def test_the_pool_is_the_union_with_each_systems_rank_ordered_by_best_rank_then_id():
    lists = {
        P10A: ["a", "b", "c"],
        P10B: ["b", "d"],
        P10C: ["c", "a"],
        P14: ["e"],
    }
    pool = phase17.pool_of(lists)

    assert pool == [
        ("a", {P10A: 1, P10C: 2}),
        ("b", {P10A: 2, P10B: 1}),
        ("c", {P10A: 3, P10C: 1}),
        ("e", {P14: 1}),
        ("d", {P10B: 2}),
    ]


def test_the_pool_reads_each_list_only_to_its_depth_and_refuses_a_repeated_unit():
    pool = phase17.pool_of({P10A: ["a", "b", "c"], P10B: ["c"]}, depth=2)
    assert [unit for unit, _ranks in pool] == ["a", "c", "b"]
    assert dict(pool)["c"] == {P10B: 1}

    with pytest.raises(phase17.Phase17Error, match="repeats"):
        phase17.pool_of({P10A: ["a", "a"]})


# --- The reorder rule and the union line (D1) ----------------------------------------------


def test_a_system_under_a_judge_is_its_own_top_by_score_ties_by_original_rank():
    top = ["u1", "u2", "u3", "u4"]
    scores = {"u1": 0.5, "u2": 2.0, "u3": 0.5, "u4": -1.0, "outside": 9.0}

    ranked = phase17.reorder(top, scores)

    assert ranked == [("u2", 2.0, 2), ("u1", 0.5, 1), ("u3", 0.5, 3), ("u4", -1.0, 4)]


def test_a_unit_outside_the_systems_top_never_enters_its_reordered_list():
    top = [f"u{i}" for i in range(101)]
    scores = {unit: float(i) for i, unit in enumerate(top)}

    ranked = phase17.reorder(top, scores, depth=100)

    assert len(ranked) == 100
    assert "u100" not in {unit for unit, _s, _r in ranked}
    assert sorted(unit for unit, _s, _r in ranked) == sorted(top[:100])


def test_a_unit_of_the_top_without_a_score_refuses():
    with pytest.raises(phase17.Phase17Error, match="no score"):
        phase17.reorder(["u1", "u2"], {"u1": 1.0})


def test_the_union_line_orders_the_whole_pool_by_score_then_best_rank_then_id():
    pool = [
        ("a", {P10A: 1}),
        ("b", {P10B: 1}),
        ("c", {P10A: 2, P14: 3}),
        ("d", {P10C: 5}),
    ]
    scores = {"a": 1.0, "b": 1.0, "c": 3.0, "d": 1.0}

    ranked = phase17.union_order(pool, scores)

    # a and b tie on score and best rank (1): unit id decides; d's best rank is 5.
    assert ranked == [("c", 3.0, 2), ("a", 1.0, 1), ("b", 1.0, 1), ("d", 1.0, 5)]


# --- D6: the comparable metrics and the ceiling ---------------------------------------------


def toy_questions() -> list[Question]:
    return [
        Question("q1", "first?", "", ("g1", "g2"), (), "dev"),
        Question("q2", "second?", "", ("h1", "sentinel"), (), "dev"),
        Question("q3", "third?", "", ("k1",), (), "dev"),
    ]


def toy_rankings() -> list[list[str]]:
    filler = [f"x{i}" for i in range(120)]
    return [
        ["g1", "x0", "g2", *filler[1:30]],
        ["y0", "y1", "h1", *filler[2:40]],
        [*filler[:25], "k1", *filler[25:90]],
    ]


def test_line_metrics_by_hand():
    metrics = phase17.line_metrics(toy_rankings(), toy_questions())

    assert metrics["n_questions"] == 3
    # Gold recall @2: q1 1/2, q2 0, q3 0. @5: q1 1, q2 1/2, q3 0. @100: q3 found at 26.
    assert metrics["gold_recall_at_k"]["2"] == pytest.approx((0.5 + 0 + 0) / 3)
    assert metrics["gold_recall_at_k"]["5"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert metrics["gold_recall_at_k"]["20"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert metrics["gold_recall_at_k"]["100"] == pytest.approx((1 + 0.5 + 1) / 3)
    # Full support @k: q1 from 3, q2 never (its sentinel), q3 from 26.
    assert metrics["full_support_at_k"]["2"] == 0.0
    assert metrics["full_support_at_k"]["5"] == pytest.approx(1 / 3)
    assert metrics["full_support_at_k"]["100"] == pytest.approx(2 / 3)
    ideal2 = 1 + 1 / math.log2(3)
    q1 = (1 + 1 / math.log2(4)) / ideal2
    q2 = (1 / math.log2(4)) / ideal2
    assert metrics["ndcg_at_10"] == pytest.approx((q1 + q2 + 0.0) / 3)
    assert sorted(metrics["gold_recall_at_k"]) == sorted(str(k) for k in config.PHASE_17_KS)


def test_line_metrics_equal_the_harness_at_every_k_it_measures():
    questions, rankings = toy_questions(), toy_rankings()
    tokens = {u: 50 for ranking in rankings for u in ranking}
    replay = phase10.ReplayRetriever(
        "dense",
        [(q.question, [(u, 1.0) for u in r]) for q, r in zip(questions, rankings, strict=True)],
    )
    records = phase9.measure_system("dense", RecordingRetriever(replay), questions, tokens)

    metrics = phase17.line_metrics(rankings, questions)

    for k in config.PHASE_9_KS:
        fs = sum(r["fs_at_k"][str(k)] for r in records) / len(records)
        gpr = sum(r["gpr_at_k"][str(k)] for r in records) / len(records)
        assert metrics["full_support_at_k"][str(k)] == pytest.approx(fs)
        assert metrics["gold_recall_at_k"][str(k)] == pytest.approx(gpr)
    assert phase17.harness_mismatches(rankings, questions, records) == []
    records[0]["fs_at_k"]["2"] = 1.0
    assert phase17.harness_mismatches(rankings, questions, records) == ["q1"]


def outcome(qid: str, supported: float) -> dict:
    return {"qid": qid, "budgets": {"2048": {"full_support": supported}}}


def test_the_ceiling_counts_questions_with_every_gold_inside_and_the_failures_among_them():
    questions = toy_questions()
    candidates = [["g1", "g2"], ["h1"], ["k1", "z"]]
    records = [outcome("q1", 1.0), outcome("q2", 0.0), outcome("q3", 0.0)]

    ceiling = phase17.ceiling(questions, candidates, records)

    assert ceiling == {
        "n_questions": 3,
        "all_gold_inside": 2,
        "full_support_2048": 1,
        "failures_inside": 1,
    }


# --- D5: the eleven comparisons and the label -----------------------------------------------


def runs_for(lines: dict[str, list[float]]) -> dict[str, list[dict]]:
    return {
        key: [outcome(f"q{i}", value) for i, value in enumerate(values)]
        for key, values in lines.items()
    }


def test_the_eleven_comparisons_are_oriented_control_then_candidate():
    base = [0.0, 0.0, 0.0, 0.0]
    lines: dict[str, list[float]] = {name: list(base) for name in config.PHASE_17_SYSTEMS}
    for judge in (LIGHT, STRONG):
        for name in config.PHASE_17_SYSTEMS:
            lines[phase17.line_key(judge, name)] = list(base)
    # Each candidate wins exactly one question over its control, on a distinct question set.
    lines[phase17.line_key(LIGHT, P14)] = [1.0, 0.0, 0.0, 0.0]
    lines[phase17.line_key(STRONG, P14)] = [1.0, 1.0, 0.0, 0.0]
    lines[phase17.line_key(LIGHT, P10B)] = [0.0, 0.0, 1.0, 0.0]
    lines[phase17.line_key(STRONG, P10B)] = [0.0, 0.0, 1.0, 1.0]

    compared = phase17.comparisons(runs_for(lines))

    assert sorted(compared) == sorted([LIGHT, STRONG, phase17.STRONG_VS_LIGHT])
    assert sum(len(compared[j]) for j in (LIGHT, STRONG)) + 1 == 11
    first = compared[LIGHT][phase17.PRIMARY]
    assert (first["control"], first["candidate"]) == ("J-light(P10-B)", "J-light(P14)")
    assert (first["wins"], first["losses"]) == (1, 1)
    strong_first = compared[STRONG][phase17.PRIMARY]
    assert (strong_first["wins"], strong_first["losses"]) == (2, 2)
    judged = compared[LIGHT]["j_p10b_vs_p10b"]
    assert (judged["control"], judged["candidate"]) == ("P10-B", "J-light(P10-B)")
    assert (judged["wins"], judged["losses"]) == (1, 0)
    for key, system in (("j_p14_vs_p14", "P14"), ("j_p10c_vs_p10c", "P10-C")):
        result = compared[STRONG][key]
        assert (result["control"], result["candidate"]) == (system, f"J-strong({system})")
    assert compared[STRONG]["j_p14_vs_p14"]["wins"] == 2
    p10a = compared[LIGHT]["j_p10a_vs_p10a"]
    assert (p10a["control"], p10a["candidate"]) == ("P10-A", "J-light(P10-A)")
    step = compared[phase17.STRONG_VS_LIGHT]
    assert (step["control"], step["candidate"]) == ("J-light(P10-B)", "J-strong(P10-B)")
    assert (step["wins"], step["losses"]) == (1, 0)


@pytest.mark.parametrize(
    ("wins", "losses", "p", "expected"),
    [
        (30, 10, 0.0499, "HOP_ADDS_UNDER_JUDGE"),
        (30, 10, 0.05, "HOP_NEUTRAL_UNDER_JUDGE"),
        (30, 10, 0.0501, "HOP_NEUTRAL_UNDER_JUDGE"),
        (10, 30, 0.0499, "HOP_HURTS_UNDER_JUDGE"),
        (10, 30, 0.05, "HOP_NEUTRAL_UNDER_JUDGE"),
        (20, 20, 0.0, "HOP_NEUTRAL_UNDER_JUDGE"),
    ],
)
def test_the_label_at_the_boundary_of_alpha(wins, losses, p, expected):
    result = {"wins": wins, "losses": losses, "exact_two_sided_p": p}
    assert phase17.hop_label(result) == expected
    assert expected in config.PHASE_17_LABELS


# --- Shards and files ------------------------------------------------------------------------


def test_the_shards_cover_every_question_once_in_order():
    assert phase17.shard_bounds(600, 250) == [(0, 250), (250, 500), (500, 600)]
    assert phase17.shard_bounds(500, 250) == [(0, 250), (250, 500)]
    assert phase17.shard_bounds(0, 250) == []


def test_a_written_file_is_reproducible_recorded_and_written_once(tmp_path):
    rows = [{"qid": "q1", "units": [["a", {P10A: 1}]]}, {"qid": "q2", "units": []}]
    first = phase17.write_jsonl_once(tmp_path / "one" / "pool-x.jsonl.gz", rows)
    second = phase17.write_jsonl_once(tmp_path / "two" / "pool-x.jsonl.gz", rows)

    data = (tmp_path / "one" / "pool-x.jsonl.gz").read_bytes()
    assert first == second
    assert first["sha256"] == hashlib.sha256(data).hexdigest()
    assert first["bytes"] == len(data)
    assert first["rows"] == 2
    assert gzip.decompress(data).decode("utf-8").count("\n") == 2
    assert phase17.read_jsonl(tmp_path / "one" / "pool-x.jsonl.gz", digest=first["digest"]) == rows
    with pytest.raises(phase17.Phase17Error, match="written once"):
        phase17.write_jsonl_once(tmp_path / "one" / "pool-x.jsonl.gz", rows)
    with pytest.raises(phase17.Phase17Error, match="digest"):
        phase17.read_jsonl(tmp_path / "one" / "pool-x.jsonl.gz", digest="0" * 64)


def test_pool_rows_and_their_file_round_trip(tmp_path):
    pool = phase17.pool_of({P10A: ["a", "b"], P14: ["b"]})
    record = phase17.write_pool(tmp_path, config.PHASE_17_MUSIQUE, [("q1", pool)])

    assert record["file"] == "pool-musique.jsonl.gz"
    back = phase17.read_pool(tmp_path, config.PHASE_17_MUSIQUE, digest=record["digest"])
    assert back == [("q1", pool)]


def test_the_pairs_join_keeps_each_text_beside_its_own_id():
    pool = phase17.pool_of({P10A: ["b", "a"], P10B: ["c"]})
    question = Question("q1", "Who?", "", ("a",), (), "dev")
    texts = {"a": "A. alpha", "b": "B. beta", "c": "C. gamma", "unused": "no"}

    rows = phase17.pair_rows([question], [("q1", pool)], texts)

    assert rows == [
        {
            "qid": "q1",
            "question": "Who?",
            "unit_ids": ["b", "c", "a"],
            "texts": ["B. beta", "C. gamma", "A. alpha"],
        }
    ]
    with pytest.raises(phase17.Phase17Error, match="no text"):
        phase17.pair_rows([question], [("q1", pool)], {"a": "A. alpha"})


def test_the_pool_summary_counts_sizes_pairs_and_systems_per_unit():
    pools = [
        ("q1", phase17.pool_of({P10A: ["a", "b"], P10B: ["a"], P10C: ["a"], P14: ["a"]})),
        ("q2", phase17.pool_of({P10A: ["c"], P10B: ["d"], P10C: ["e"], P14: ["c"]})),
    ]

    summary = phase17.pool_summary(pools)

    assert summary["questions"] == 2
    assert summary["pairs"] == 5
    assert summary["size"] == pytest.approx(
        {"mean": 2.5, "median": 2.5, "p90": 2.9, "min": 2, "max": 3}
    )
    assert summary["units_listed_by"] == {"1": 3, "2": 1, "3": 0, "4": 1}


def test_the_file_names_carry_the_set_judge_and_line():
    assert phase17.pool_name("hotpotqa") == "pool-hotpotqa.jsonl.gz"
    assert phase17.pairs_name("multihop-rag") == "pairs-multihop-rag.jsonl.gz"
    assert phase17.scores_name(LIGHT, "musique") == "scores-light-musique.jsonl.gz"
    assert phase17.scores_name(LIGHT, "musique", tag="r2") == "scores-light-musique-r2.jsonl.gz"
    assert (
        phase17.reordered_name(STRONG, "hotpotqa", config.PHASE_17_UNION)
        == "reordered-strong-hotpotqa-union.jsonl.gz"
    )
