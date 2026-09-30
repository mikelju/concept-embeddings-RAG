"""Phase 16: the frozen systems on MultiHop-RAG. S6: what the outcome adds to Phase 15's.

`16.spec.md` (approved and frozen 2026-09-30) fixes every rule here before any Phase 16
number exists. The paired comparisons, the label, the recorded contexts, the reach and the
Dense-rank split are Phase 15's functions, reused unchanged (`evaluation.phase15`). This
module holds only what MultiHop-RAG adds, all of it descriptive and deciding nothing:

- **the ladder in three columns (D6)**: P10-A -> P10-B -> P10-C -> P14 on HotpotQA `test-11`,
  MuSiQue validation and MultiHop-RAG, each step's gain per column and whether its sign agrees;
- **D7**: `phase15.group_summary` by gold count (2 / 3 / 4) and by `question_type`;
- **D8, the same-article tally**: each gold unit that entered a hop system's 2,048-token
  context through the hop, from P1's own article or another one, and the two context shares;
- **deviation 16.1**: for the queries with a repeated fact, how often the fact's other
  paragraph was in the context while its gold unit was not;
- **D9**: Hits@k over units, beside the paper's chunk-level figures, never compared with them.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import phase15

P10A, P10B, P10C, P14 = config.PHASE_16_SYSTEMS
BUDGET = phase15.BUDGET


class Phase16Error(Exception):
    """A Phase 16 outcome input is inconsistent."""


# --- D6: the ladder in three columns ------------------------------------------------------

HOTPOTQA, MUSIQUE, MULTIHOP_RAG = "hotpotqa_test_11", "musique_validation", "multihop_rag"
# D6's MuSiQue column: the four systems' Full Support @2,048 on the 2,417 validation
# questions, measured in Phase 15 and read back from its committed `outcome.json`.
MUSIQUE_VALIDATION_QUESTIONS = 2417
MUSIQUE_VALIDATION_SUPPORTED: dict[str, int] = {P10A: 436, P10B: 524, P10C: 669, P14: 761}


def counts(runs: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, int]]:
    """Each system's Full Support @2,048 count and question count, in system order."""
    return {
        system: {"supported": phase15._supported(runs[system]), "n_questions": len(runs[system])}
        for system in config.PHASE_16_SYSTEMS
    }


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def ladder(
    columns: Mapping[str, Mapping[str, Mapping[str, int]]], *, new: str = MULTIHOP_RAG
) -> dict[str, Any]:
    """D6's progression P10-A -> P10-B -> P10-C -> P14 in every column (`column -> system ->
    {supported, n_questions}`, in column order): each rung's Full Support @2,048 per column,
    each step's gain in points per column, its sign per column, whether all the signs are the
    same, and whether the `new` column's sign agrees with each other column's."""
    names = list(columns)
    rungs: list[dict[str, Any]] = []
    for system in config.PHASE_16_SYSTEMS:
        rung: dict[str, Any] = {"system": system, "label": config.PHASE_16_SYSTEM_LABELS[system]}
        for name in names:
            supported = int(columns[name][system]["supported"])
            n = int(columns[name][system]["n_questions"])
            rung[name] = {
                "supported": supported,
                "n_questions": n,
                "full_support_percent": phase15._percent(supported, n),
            }
        rungs.append(rung)
    steps: list[dict[str, Any]] = []
    for before, after in zip(rungs, rungs[1:], strict=False):
        gains = {
            name: after[name]["full_support_percent"] - before[name]["full_support_percent"]
            for name in names
        }
        signs = {name: _sign(gain) for name, gain in gains.items()}
        steps.append(
            {
                "from": before["label"],
                "to": after["label"],
                "gain_pp": gains,
                "signs": signs,
                "all_same_sign": len(set(signs.values())) == 1,
                "new_column_agrees_with": {
                    name: signs[new] == signs[name] for name in names if name != new
                },
            }
        )
    return {
        "metric": f"full_support@{BUDGET}_tokens",
        "columns": names,
        "new_column": new,
        "rungs": rungs,
        "steps": steps,
        "note": (
            "the columns differ in corpus, questions and size: they compare the direction "
            "and order of the gains, not their size (D6)"
        ),
    }


# --- D7: by gold count and by query type --------------------------------------------------


def _grouped(
    group_of: Mapping[str, Any],
    groups: Sequence[Any],
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], int]:
    body = {
        str(group): phase15.group_summary(
            {qid for qid, value in group_of.items() if value == group}, runs
        )
        for group in groups
    }
    return body, sum(1 for value in group_of.values() if value not in groups)


def by_gold_count(
    questions: Sequence[Question],
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    groups: Sequence[int] = tuple(sorted(config.PHASE_16_GOLD_COUNTS)),
) -> dict[str, Any]:
    """D7: `phase15.group_summary` per number of gold units after deduplication (the spec's
    1,079 / 780 / 396), sentinels included. Descriptive, deciding nothing."""
    body, other = _grouped({q.qid: len(q.gold_unit_ids) for q in questions}, groups, runs)
    return {
        "grouping": "the query's number of gold units after deduplication (len(gold_unit_ids))",
        "budget": int(BUDGET),
        "groups": body,
        "other": other,
        "note": "descriptive, deciding nothing (D7); every comparison is exact McNemar",
    }


def by_question_type(
    questions: Sequence[Question],
    types: Mapping[str, str],
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    groups: Sequence[str] = tuple(config.PHASE_16_TYPE_COUNTS),
) -> dict[str, Any]:
    """D7: `phase15.group_summary` per `question_type` (`types` maps qid to the source's
    own value). Descriptive, deciding nothing."""
    body, other = _grouped({q.qid: types[q.qid] for q in questions}, groups, runs)
    return {
        "grouping": "the query's question_type in MultiHopRAG.json",
        "budget": int(BUDGET),
        "groups": body,
        "other": other,
        "note": "descriptive, deciding nothing (D7); every comparison is exact McNemar",
    }


# --- D8: the same-article tally -----------------------------------------------------------


def _tally() -> dict[str, Any]:
    return {
        "queries": 0,
        "queries_with_hop_entered_gold": 0,
        "through_hop_p1_article": 0,
        "through_hop_other_article": 0,
        "without_hop": 0,
    }


def _finish(tally: dict[str, Any]) -> dict[str, Any]:
    through = tally["through_hop_p1_article"] + tally["through_hop_other_article"]
    tally["through_hop"] = through
    tally["p1_article_share"] = tally["through_hop_p1_article"] / through if through else None
    return tally


def same_article(
    questions: Sequence[Question],
    rankings: Sequence[Mapping[str, Any]],
    *,
    hop_component: str,
    candidate: Mapping[str, Collection[str]],
    control: Mapping[str, Collection[str]],
    articles: Mapping[str, Mapping[str, Any]],
    types: Mapping[str, str],
    type_groups: Sequence[str] = tuple(config.PHASE_16_TYPE_COUNTS),
) -> dict[str, Any]:
    """D8: which gold units a hop system brought into its 2,048-token context, and from where.

    `rankings` are the hop system's stored records (question order), `candidate` and
    `control` its and P10-B's checked contexts by qid, `articles` `unit_id -> article
    metadata` (`multihop_rag.load_articles`). A gold unit **entered through the hop** if it
    is in the candidate's context, not in the control's, and listed in the stored
    `hop_component` (depth 100); it is tallied as from P1's article (`record["hop"]["p1"]`) or
    another article. A gold unit that entered without the hop listing it (the reweighting of
    Dense and BM25) is counted apart, never attributed. Tallied over all queries and over the
    won ones (candidate Full Support, control not), overall and per `question_type`.
    """
    if [r["qid"] for r in rankings] != [q.qid for q in questions]:
        raise Phase16Error("the stored rankings are not in the question order")
    overall = {"all": _tally(), "won": _tally()}
    per_type = {group: {"all": _tally(), "won": _tally()} for group in type_groups}
    for question, record in zip(questions, rankings, strict=True):
        qid = question.qid
        gold = list(question.gold_unit_ids)
        inside, outside = set(candidate[qid]), set(control[qid])
        listed = {unit_id for unit_id, _score in record["components"][hop_component]}
        p1 = record["hop"]["p1"]
        p1_article = articles[p1]["index"] if p1 is not None else None
        entered = [g for g in gold if g in inside and g not in outside]
        same = sum(1 for g in entered if g in listed and articles[g]["index"] == p1_article)
        through = sum(1 for g in entered if g in listed)
        won = set(gold) <= inside and not set(gold) <= outside
        bins = [overall, *([per_type[types[qid]]] if types[qid] in per_type else [])]
        for tally in (b[key] for b in bins for key in ("all", "won") if key == "all" or won):
            tally["queries"] += 1
            tally["queries_with_hop_entered_gold"] += through > 0
            tally["through_hop_p1_article"] += same
            tally["through_hop_other_article"] += through - same
            tally["without_hop"] += len(entered) - through
    return {
        "hop_component": hop_component,
        "budget": int(BUDGET),
        "all": _finish(overall["all"]),
        "won": _finish(overall["won"]),
        "by_question_type": {
            group: {key: _finish(value) for key, value in tallies.items()}
            for group, tallies in per_type.items()
        },
        "rule": (
            "a gold unit entered through the hop if it is in the system's 2,048-token context, "
            "not in P10-B's, and listed in the system's stored hop component (depth 100); it is "
            "tallied by whether its article is P1's; gold entering without the hop listing it "
            "is counted apart (without_hop) and not attributed; 'won' = the system's Full "
            "Support and not P10-B's"
        ),
        "note": "descriptive, deciding nothing (D8)",
    }


def article_context(
    questions: Sequence[Question],
    dense: Sequence[Mapping[str, Any]],
    articles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """D8's context shares: queries whose gold units all lie in one article (a sentinel gold
    lies in none), and queries whose P1 (the first unit of `dense`'s stored Dense list, P10-A's
    run) comes from an article holding at least one of their gold units."""
    if [r["qid"] for r in dense] != [q.qid for q in questions]:
        raise Phase16Error("the stored Dense rankings are not in the question order")
    one_article = p1_holds = 0
    for question, record in zip(questions, dense, strict=True):
        gold = list(question.gold_unit_ids)
        gold_articles = {articles[g]["index"] for g in gold if g in articles}
        one_article += all(g in articles for g in gold) and len(gold_articles) == 1
        listed = record["components"]["dense"]
        p1_holds += bool(listed) and articles[listed[0][0]]["index"] in gold_articles
    n = len(questions)
    return {
        "n_queries": n,
        "gold_in_one_article": one_article,
        "gold_in_one_article_share": one_article / n if n else 0.0,
        "p1_article_holds_gold": p1_holds,
        "p1_article_holds_gold_share": p1_holds / n if n else 0.0,
        "p1_source": "the first unit of P10-A's stored Dense list",
    }


# --- Deviation 16.1: the other paragraph of a repeated fact -------------------------------


def repeated_other(
    repeated: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Collection[str]]],
) -> dict[str, Any]:
    """For the queries with a fact printed twice (`questions.json` `repeated_facts`), per
    system: the queries where one of the fact's other paragraphs (`other_unit_ids`) is in the
    2,048-token context while the fact's gold unit (`unit_id`) is not. Deciding nothing."""
    systems: dict[str, Any] = {}
    for system, by_qid in contexts.items():
        qids = sorted(
            {
                str(entry["qid"])
                for entry in repeated
                if entry["unit_id"] not in set(by_qid[entry["qid"]])
                and any(other in set(by_qid[entry["qid"]]) for other in entry["other_unit_ids"])
            }
        )
        systems[system] = {
            "label": config.PHASE_16_SYSTEM_LABELS.get(system, system),
            "queries": len(qids),
            "qids": qids,
        }
    return {
        "n_facts": len(repeated),
        "n_queries": len({str(entry["qid"]) for entry in repeated}),
        "budget": int(BUDGET),
        "systems": systems,
        "rule": (
            "a repeated fact's gold is its first occurrence; counted: queries where one of the "
            "fact's other paragraphs is in the system's 2,048-token context and its gold unit "
            "is not (deviation 16.1, option A)"
        ),
        "note": "reported, deciding nothing: bounds the possible false misses of option A",
    }


# --- D9: Hits@k over units ----------------------------------------------------------------

PAPER_HITS: dict[int, float] = {4: 0.6625, 10: 0.7467}
HITS_DIFFERENCES: tuple[str, ...] = (
    "unit: whole newline paragraphs of the article, not the paper's 256-token chunks",
    "corpus cut: every newline paragraph of the 609 articles, boilerplate kept and verbatim "
    "repeats within an article collapsed, not the paper's chunked corpus",
    "metric: gold units found in the first k of the stored fused list, micro-averaged over "
    "all gold units of the 2,255 answerable queries with unmapped gold in the denominator, "
    "not the paper's evaluation script",
)


def hits_at(
    rankings: Sequence[Mapping[str, Any]], questions: Sequence[Question], k: int
) -> dict[str, Any]:
    """Hits@k over units: the gold units found in the first `k` of each stored fused list,
    summed over the queries, over all their gold units (a sentinel gold, never retrieved,
    stays in the denominator)."""
    if [r["qid"] for r in rankings] != [q.qid for q in questions]:
        raise Phase16Error("the stored rankings are not in the question order")
    found = total = 0
    for record, question in zip(rankings, questions, strict=True):
        gold = set(question.gold_unit_ids)
        found += len(gold & {unit_id for unit_id, _score in record["fused"][:k]})
        total += len(gold)
    return {"k": k, "gold_found": found, "gold_units": total, "hits": found / total}


def hits_report(
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    questions: Sequence[Question],
    *,
    ks: Sequence[int] = config.PHASE_16_HITS_K,
) -> dict[str, Any]:
    """D9: Hits@k per system over its stored fused list (P10-A's is the Dense list), beside
    the paper's best reported pipeline with the three differences named."""
    return {
        "systems": {
            system: {
                "label": config.PHASE_16_SYSTEM_LABELS.get(system, system),
                "hits": {str(k): hits_at(records, questions, k) for k in ks},
            }
            for system, records in rankings.items()
        },
        "paper": {str(k): PAPER_HITS[k] for k in ks},
        "paper_source": "Tang & Yang 2024, MultiHop-RAG, Table 5: best reported pipeline",
        "differences": list(HITS_DIFFERENCES),
        "note": (
            "reported, not comparable with the paper's figures and deciding nothing (D9); "
            "never a comparison of systems with the paper"
        ),
    }
