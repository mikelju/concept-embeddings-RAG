"""Post-hoc, gold-informed, read-only: do MuSiQue's near-duplicate gold paragraphs bias the
Phase 15 comparisons?

Asked by the author after Phase 15 closed `TRANSFER_SUPPORTED` (2026-09-30). MuSiQue sometimes
carries two nearly equal versions of a passage and marks only one as supporting (author decision
5 of the Phase 15 plan): `corpus.json` counts 72 gold units with such a twin, in 172 validation
questions. Gold stayed the marked paragraph, so a system whose 2,048-token context holds the
twin instead of the marked one is counted as missing it. This script measures, from the
rankings the pass stored, how often that happens per system and what the paired comparisons
would be if a twin counted as the gold. It selects nothing, changes no artifact and decides
nothing: the label stands on the frozen rule.

    uv run python scripts/phase15_near_duplicate_diagnostic.py

Output: `data/phase15/diagnostics/near_duplicate_gold.json` (written once) and a console summary.
"""

import json
from collections import defaultdict

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus import fullwiki as fullwiki_corpus
from concept_embeddings_rag.corpus import musique
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation import phase15

BUDGET = config.PHASE_9_PRIMARY_BUDGET
DEPTH = config.PHASE_9_RANKING_DEPTH
OUT = config.PHASE_15_DIR / "diagnostics" / "near_duplicate_gold.json"


def main() -> None:
    target = config.PHASE_15_DIR
    if OUT.exists():
        raise SystemExit(f"[ERROR] {OUT} exists; the diagnostic is written once")
    units = fullwiki_corpus.load_corpus(target)
    corpus = json.loads((target / fullwiki_corpus.CORPUS_MANIFEST).read_text(encoding="utf-8"))
    questions, _body = musique.load_questions(
        target, corpus_unit_set_hash=str(corpus["unit_set_hash"])
    )
    unit_ids = [u.unit_id for u in units]
    token_counts = phase9.load_phase9_token_counts(
        target, unit_set_hash=str(corpus["unit_set_hash"]), unit_ids=unit_ids
    )

    # Twins: the rule of `musique.near_duplicate_gold`, same title and the same first 80 chars.
    prefix = int(corpus["near_duplicate_gold"]["prefix_chars"])
    by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for unit in units:
        by_key[(unit.title, unit.text[:prefix])].append(unit.unit_id)
    key_of = {unit.unit_id: (unit.title, unit.text[:prefix]) for unit in units}
    twins = {
        g: [t for t in by_key[key_of[g]] if t != g]
        for q in questions
        for g in q.gold_unit_ids
        if g in key_of and len(by_key[key_of[g]]) > 1
    }
    affected = [q for q in questions if any(g in twins for g in q.gold_unit_ids)]
    if len(twins) != corpus["near_duplicate_gold"]["gold_units_with_near_duplicate"]:
        raise SystemExit("[ERROR] the twin count does not match corpus.json")
    print(f"[INFO] {len(twins)} gold units with a twin, {len(affected)} questions affected")

    systems = list(config.PHASE_15_SYSTEMS)
    contexts: dict[str, dict[str, list[str]]] = {}
    strict: dict[str, dict[str, float]] = {}
    lenient: dict[str, dict[str, float]] = {}
    per_system: dict[str, dict[str, int]] = {}
    for system in systems:
        rankings = phase15.load_rankings(target, system)
        outcomes = phase9.load_outcomes(target, system)
        contexts[system] = phase15.recorded_contexts(rankings, outcomes, questions, token_counts)
        strict[system] = {}
        lenient[system] = {}
        tally = {"gold_in_context": 0, "twin_instead_of_gold": 0, "neither": 0, "flips": 0}
        for q in questions:
            context = set(contexts[system][q.qid])
            full = float(set(q.gold_unit_ids) <= context)
            lenient_full = float(
                all(
                    g in context or any(t in context for t in twins.get(g, ()))
                    for g in q.gold_unit_ids
                )
            )
            strict[system][q.qid] = full
            lenient[system][q.qid] = lenient_full
            if q in affected and lenient_full != full:
                tally["flips"] += 1
            for g in q.gold_unit_ids:
                if g not in twins:
                    continue
                if g in context:
                    tally["gold_in_context"] += 1
                elif any(t in context for t in twins[g]):
                    tally["twin_instead_of_gold"] += 1
                else:
                    tally["neither"] += 1
        per_system[system] = tally
        print(
            f"[INFO] {config.PHASE_15_SYSTEM_LABELS[system]}: twin pairs gold "
            f"{tally['gold_in_context']}, "
            f"twin instead {tally['twin_instead_of_gold']}, neither {tally['neither']}; "
            f"Full Support flips if a twin counted: {tally['flips']}"
        )

    def paired(control: str, candidate: str, table: dict[str, dict[str, float]]) -> dict:
        a, c = table[control], table[candidate]
        wins = sum(1 for q in a if c[q] == 1.0 and a[q] == 0.0)
        losses = sum(1 for q in a if a[q] == 1.0 and c[q] == 0.0)
        return {
            "control": config.PHASE_15_SYSTEM_LABELS[control],
            "candidate": config.PHASE_15_SYSTEM_LABELS[candidate],
            "control_supported": int(sum(a.values())),
            "candidate_supported": int(sum(c.values())),
            "wins": wins,
            "losses": losses,
            "exact_two_sided_p": phase9.exact_two_sided_p(wins, losses),
        }

    pairs = [(systems[1], systems[3]), (systems[2], systems[3]), (systems[1], systems[2])]
    comparisons = {
        "strict_marked_gold_only_as_recorded": [paired(a, b, strict) for a, b in pairs],
        "lenient_twin_counts_as_gold_exploratory": [paired(a, b, lenient) for a, b in pairs],
    }
    for name, rows in comparisons.items():
        print(f"[INFO] {name}")
        for row in rows:
            print(
                f"    {row['candidate']} vs {row['control']}: {row['candidate_supported']} vs "
                f"{row['control_supported']}, wins {row['wins']}, losses {row['losses']}, "
                f"p = {row['exact_two_sided_p']:.3g}"
            )
    recorded = json.loads((target / "outcome.json").read_text(encoding="utf-8"))
    primary = recorded["d5_primary_p14_vs_p10b"]
    check = comparisons["strict_marked_gold_only_as_recorded"][0]
    if (check["wins"], check["losses"]) != (primary["wins"], primary["losses"]):
        raise SystemExit("[ERROR] the strict recount does not reproduce outcome.json")

    body = {
        "phase": 15,
        "kind": "post-hoc, gold-informed diagnostic; exploratory; decides nothing",
        "rule": corpus["near_duplicate_gold"]["rule"],
        "budget": BUDGET,
        "depth": DEPTH,
        "gold_units_with_twin": len(twins),
        "questions_affected": len(affected),
        "twin_pairs": sum(1 for q in questions for g in q.gold_unit_ids if g in twins),
        "per_system": {config.PHASE_15_SYSTEM_LABELS[s]: t for s, t in per_system.items()},
        "comparisons": comparisons,
        "sources": {
            "outcome_digest": recorded.get("integrity", {}).get("outcome_digest"),
            "corpus_unit_set_hash": corpus["unit_set_hash"],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(OUT, json.dumps(body, indent=2, sort_keys=True))
    print(f"[OK] -> {OUT}")


if __name__ == "__main__":
    main()
