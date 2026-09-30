"""Phase 16, S6: what the MultiHop-RAG outcome adds to Phase 15's.

What can change a reported Phase 16 figure here: the three-column ladder and whether each
step's sign agrees across the columns (D6); the two D7 groupings (gold count and query type)
through `phase15.group_summary`, which must leave Phase 15's `by_supporting` output byte for
byte; the same-article tally of D8 (which gold units the hop brought in, from P1's article or
another one); the count of deviation 16.1 (the other paragraph of a repeated fact in the
context while the gold one is not); and Hits@k over units with a sentinel gold in the
denominator (D9). Every input is a small hand-built fixture.
"""

import hashlib
import json
from typing import Any

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation import phase15, phase16
from concept_embeddings_rag.retrieval.base import Hit

P10A, P10B, P10C, P14 = config.PHASE_16_SYSTEMS


def outcome(qid: str, fs: float, recall: float | None = None) -> dict[str, Any]:
    """An outcome record reduced to its 2,048 reading."""
    return {
        "qid": qid,
        "budgets": {"2048": {"full_support": fs, "gold_recall": fs if recall is None else recall}},
    }


def question(qid: str, gold: tuple[str, ...], n_supporting: int | None = None) -> Question:
    n = len(gold) if n_supporting is None else n_supporting
    return Question(
        qid=qid,
        question=f"{qid}?",
        answer="",
        gold_unit_ids=gold,
        supporting_facts=tuple((f"T{i}", i) for i in range(n)),
        split="s",
    )


def hits(*ids: str) -> list[Hit]:
    return [(unit_id, 1.0 - i / 100) for i, unit_id in enumerate(ids)]


# --- D7: group_summary leaves Phase 15's by_supporting byte for byte ----------------------

# The digest of `json.dumps(by_supporting(...), indent=2, sort_keys=True)` on the fixture
# below, computed with the code before `group_summary` was extracted (commit 9fbf21d):
# `outcome.json` is written with that serialization.
BY_SUPPORTING_SHA256 = "26e61be235108418352351a2c141e61f193938c4ebd12a80be16885e73fbdc54"


def supporting_fixture() -> tuple[list[Question], dict[str, list[dict[str, Any]]]]:
    def q(qid: str, n: int, gold: int | None = None) -> Question:
        count = n if gold is None else gold
        return question(qid, tuple(f"{qid}-g{i}" for i in range(count)), n_supporting=n)

    questions = [q("a", 2), q("b", 2, gold=1), q("c", 3), q("d", 4), q("e", 5)]
    runs = {
        P10A: [
            outcome("a", 0.0, 0.5),
            outcome("b", 0.0, 0.0),
            outcome("c", 0.0, 1 / 3),
            outcome("d", 0.0, 0.25),
            outcome("e", 0.0, 0.2),
        ],
        P10B: [
            outcome("a", 1.0),
            outcome("b", 0.0, 0.0),
            outcome("c", 0.0, 2 / 3),
            outcome("d", 0.0, 0.5),
            outcome("e", 1.0),
        ],
        P10C: [
            outcome("a", 1.0),
            outcome("b", 1.0),
            outcome("c", 0.0, 2 / 3),
            outcome("d", 0.0, 0.75),
            outcome("e", 1.0),
        ],
        P14: [
            outcome("a", 1.0),
            outcome("b", 1.0),
            outcome("c", 1.0),
            outcome("d", 0.0, 0.5),
            outcome("e", 0.0, 0.4),
        ],
    }
    return questions, runs


def test_by_supporting_is_byte_identical_after_group_summary_is_extracted():
    questions, runs = supporting_fixture()

    text = json.dumps(phase15.by_supporting(questions, runs), indent=2, sort_keys=True)

    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == BY_SUPPORTING_SHA256


def test_group_summary_is_by_supportings_group_body():
    questions, runs = supporting_fixture()
    split = phase15.by_supporting(questions, runs)

    summary = phase15.group_summary({"a", "b"}, runs)

    expected = {k: v for k, v in split["groups"]["2"].items() if k != "gold_counts"}
    assert summary == expected
    assert phase15.group_summary(set(), runs) == {"n_questions": 0}


def test_d7_groups_by_gold_count_after_deduplication():
    # "b" has two supporting facts but one gold unit: it counts as a 1-gold query, "other".
    questions, runs = supporting_fixture()

    split = phase16.by_gold_count(questions, runs)

    assert list(split["groups"]) == ["2", "3", "4"]
    assert split["groups"]["2"] == phase15.group_summary({"a"}, runs)
    assert split["groups"]["3"]["n_questions"] == 1
    assert split["groups"]["3"]["comparisons"][phase15.PRIMARY]["wins"] == 1
    assert split["groups"]["4"]["systems"][P10C]["mean_gold_recall"] == pytest.approx(0.75)
    assert split["other"] == 2  # "b" (1 gold) and "e" (5 gold)
    assert "descriptive" in split["note"]


def test_d7_groups_by_question_type_with_an_empty_type_recorded():
    questions, runs = supporting_fixture()
    types = {
        "a": "comparison_query",
        "b": "comparison_query",
        "c": "inference_query",
        "d": "inference_query",
        "e": "inference_query",
    }

    split = phase16.by_question_type(questions, types, runs)

    assert list(split["groups"]) == list(config.PHASE_16_TYPE_COUNTS)
    comparison = split["groups"]["comparison_query"]
    assert comparison == phase15.group_summary({"a", "b"}, runs)
    assert comparison["systems"][P10B]["full_support"] == 1
    assert comparison["comparisons"]["p10c_vs_p10b"]["wins"] == 1
    assert split["groups"]["inference_query"]["n_questions"] == 3
    assert split["groups"]["temporal_query"] == {"n_questions": 0}
    assert split["other"] == 0


# --- D6: the ladder in three columns ------------------------------------------------------


def column(*supported: int, n: int) -> dict[str, dict[str, int]]:
    return {
        system: {"supported": s, "n_questions": n}
        for system, s in zip(config.PHASE_16_SYSTEMS, supported, strict=True)
    }


def test_the_ladder_has_three_columns_and_each_steps_sign_agreement():
    columns = {
        phase16.HOTPOTQA: column(2852, 3079, 3285, 3530, n=5000),
        phase16.MUSIQUE: column(436, 524, 669, 761, n=2417),
        # MultiHop-RAG: BM25 gains, the query-blind hop loses, the ordered hop gains back.
        phase16.MULTIHOP_RAG: column(10, 12, 11, 11, n=20),
    }

    ladder = phase16.ladder(columns)

    assert ladder["columns"] == [phase16.HOTPOTQA, phase16.MUSIQUE, phase16.MULTIHOP_RAG]
    assert [rung["system"] for rung in ladder["rungs"]] == list(config.PHASE_16_SYSTEMS)
    first = ladder["rungs"][0]
    assert first[phase16.HOTPOTQA]["full_support_percent"] == pytest.approx(57.04)
    assert first[phase16.MUSIQUE]["supported"] == 436
    assert first[phase16.MULTIHOP_RAG]["full_support_percent"] == 50.0
    steps = ladder["steps"]
    assert [(s["from"], s["to"]) for s in steps] == [
        ("P10-A", "P10-B"),
        ("P10-B", "P10-C"),
        ("P10-C", "P14"),
    ]
    bm25, hop, ordered = steps
    assert bm25["gain_pp"][phase16.MULTIHOP_RAG] == pytest.approx(10.0)
    assert bm25["gain_pp"][phase16.HOTPOTQA] == pytest.approx(4.54)
    assert bm25["signs"] == {phase16.HOTPOTQA: 1, phase16.MUSIQUE: 1, phase16.MULTIHOP_RAG: 1}
    assert bm25["all_same_sign"] is True
    assert hop["signs"][phase16.MULTIHOP_RAG] == -1 and hop["all_same_sign"] is False
    assert hop["new_column_agrees_with"] == {phase16.HOTPOTQA: False, phase16.MUSIQUE: False}
    assert ordered["signs"][phase16.MULTIHOP_RAG] == 0 and ordered["all_same_sign"] is False
    assert "direction" in ladder["note"]


def test_counts_reads_each_systems_full_support_at_2048():
    runs = {
        P10A: [outcome("a", 0.0), outcome("b", 1.0)],
        P10B: [outcome("a", 1.0), outcome("b", 1.0)],
        P10C: [outcome("a", 0.0), outcome("b", 0.0)],
        P14: [outcome("a", 1.0), outcome("b", 0.0)],
    }

    assert phase16.counts(runs) == column(1, 2, 0, 1, n=2)


# --- D8: the same-article tally -----------------------------------------------------------

ARTICLES = {
    "p1": {"index": 0},
    "g-same": {"index": 0},
    "g-other": {"index": 1},
    "g-reweighted": {"index": 2},
    "g-both": {"index": 3},
    "x": {"index": 4},
    "h-other": {"index": 5},
}


def ranking(qid: str, p1: str, hop: tuple[str, ...]) -> dict[str, Any]:
    return {
        "qid": qid,
        "components": {"dense": hits(p1, "x"), "bm25": hits("x"), "entity-hop": hits(*hop)},
        "hop": {"p1": p1, "positives": len(hop)},
        "fused": hits(p1, *hop),
    }


def test_the_same_article_tally_attributes_only_what_the_hop_listed():
    questions = [
        # Won: two gold units enter, one from P1's article, one from another, both listed by
        # the hop; a third enters by reweighting (not listed by the hop).
        question("won", ("p1", "g-same", "g-other", "g-reweighted")),
        # Not won: one hop-listed gold from another article enters, one gold stays out.
        question("lost", ("g-other", "h-other")),
        # Already in both contexts: nothing entered.
        question("tie", ("g-both",)),
    ]
    rankings = [
        ranking("won", "p1", ("g-same", "g-other")),
        ranking("lost", "p1", ("g-other", "h-other")),
        ranking("tie", "p1", ("g-both",)),
    ]
    candidate = {
        "won": ["p1", "g-same", "g-other", "g-reweighted"],
        "lost": ["p1", "g-other"],
        "tie": ["g-both"],
    }
    control = {"won": ["p1"], "lost": ["p1"], "tie": ["g-both"]}
    types = {"won": "comparison_query", "lost": "inference_query", "tie": "comparison_query"}

    tally = phase16.same_article(
        questions,
        rankings,
        hop_component="entity-hop",
        candidate=candidate,
        control=control,
        articles=ARTICLES,
        types=types,
    )

    everything, won = tally["all"], tally["won"]
    assert everything["queries"] == 3 and won["queries"] == 1
    assert everything["through_hop_p1_article"] == 1
    assert everything["through_hop_other_article"] == 2
    assert everything["without_hop"] == 1
    assert everything["queries_with_hop_entered_gold"] == 2
    assert everything["p1_article_share"] == pytest.approx(1 / 3)
    assert (won["through_hop_p1_article"], won["through_hop_other_article"]) == (1, 1)
    assert won["without_hop"] == 1
    by_type = tally["by_question_type"]
    assert by_type["comparison_query"]["all"]["queries"] == 2
    assert by_type["comparison_query"]["won"]["through_hop_p1_article"] == 1
    assert by_type["inference_query"]["all"]["through_hop_other_article"] == 1
    assert by_type["inference_query"]["won"]["queries"] == 0
    assert by_type["temporal_query"]["all"]["queries"] == 0
    assert tally["hop_component"] == "entity-hop"


def test_the_article_context_shares():
    questions = [
        question("one", ("g-same", "p1")),  # both in article 0, P1's article holds gold
        question("two", ("g-other", "h-other")),  # two articles, P1's (0) holds none
        question("sentinel", ("g-other", "missing")),  # an unmapped gold: not in one article
    ]
    dense = [
        {"qid": "one", "components": {"dense": hits("p1", "x")}},
        {"qid": "two", "components": {"dense": hits("p1")}},
        {"qid": "sentinel", "components": {"dense": hits("g-other")}},
    ]

    context = phase16.article_context(questions, dense, ARTICLES)

    assert context["n_queries"] == 3
    assert context["gold_in_one_article"] == 1
    assert context["gold_in_one_article_share"] == pytest.approx(1 / 3)
    assert context["p1_article_holds_gold"] == 2
    assert context["p1_article_holds_gold_share"] == pytest.approx(2 / 3)


# --- Deviation 16.1: the other paragraph of a repeated fact -------------------------------


def test_the_repeated_fact_count_needs_the_other_paragraph_in_and_the_gold_out():
    repeated = [
        {"qid": "q1", "unit_id": "gold1", "other_unit_ids": ["other1"]},
        {"qid": "q2", "unit_id": "gold2", "other_unit_ids": ["other2a", "other2b"]},
        {"qid": "q3", "unit_id": "gold3", "other_unit_ids": ["other3"]},
    ]
    contexts = {
        P10A: {"q1": ["other1"], "q2": ["other2b"], "q3": ["gold3", "other3"]},
        P10B: {"q1": ["gold1"], "q2": [], "q3": []},
        P10C: {"q1": ["gold1", "other1"], "q2": ["other2a"], "q3": ["x"]},
        P14: {"q1": [], "q2": [], "q3": []},
    }

    count = phase16.repeated_other(repeated, contexts)

    assert (count["n_facts"], count["n_queries"]) == (3, 3)
    assert count["systems"][P10A]["queries"] == 2
    assert count["systems"][P10A]["qids"] == ["q1", "q2"]
    assert count["systems"][P10B]["queries"] == 0
    assert count["systems"][P10C]["qids"] == ["q2"]
    assert count["systems"][P14]["queries"] == 0
    assert count["systems"][P10C]["label"] == "P10-C"
    assert "deciding nothing" in count["note"]


# --- D9: Hits@k over units ----------------------------------------------------------------


def test_hits_at_k_is_micro_averaged_with_a_sentinel_gold_in_the_denominator():
    sentinel = phase9._sentinel("q2", 1)
    questions = [question("q1", ("a", "b")), question("q2", ("c", sentinel))]
    rankings = [
        {"qid": "q1", "fused": hits("a", "x", "y", "z", "b")},
        {"qid": "q2", "fused": hits("x", "c")},
    ]

    at_4 = phase16.hits_at(rankings, questions, 4)
    at_10 = phase16.hits_at(rankings, questions, 10)

    assert (at_4["gold_found"], at_4["gold_units"]) == (2, 4)
    assert at_4["hits"] == 0.5
    assert at_10["hits"] == 0.75


def test_hits_at_k_refuses_rankings_in_another_order():
    questions = [question("q1", ("a",)), question("q2", ("b",))]
    rankings = [{"qid": "q2", "fused": []}, {"qid": "q1", "fused": []}]

    with pytest.raises(phase16.Phase16Error, match="order"):
        phase16.hits_at(rankings, questions, 4)


def test_the_hits_report_sits_beside_the_papers_figures_with_the_differences_named():
    questions = [question("q1", ("a", "b"))]
    rankings = {name: [{"qid": "q1", "fused": hits("a")}] for name in config.PHASE_16_SYSTEMS}

    report = phase16.hits_report(rankings, questions)

    assert report["paper"] == {"4": 0.6625, "10": 0.7467}
    assert len(report["differences"]) == 3
    assert report["systems"][P14]["hits"]["4"]["hits"] == 0.5
    assert report["systems"][P14]["label"] == "P14"
    assert "not comparable" in report["note"]
