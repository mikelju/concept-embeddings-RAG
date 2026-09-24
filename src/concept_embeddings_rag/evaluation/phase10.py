"""Phase 10: Dense + BM25 + Entity Hop against Dense + BM25 at FullWiki scale.

`10.spec.md` (approved and frozen 2026-09-24) fixes every rule here before any Phase 10 number
exists. Dev is Phase 9's 7,405 validation questions; the held-out questions are three disjoint
samples of HotpotQA train "hard" questions, drawn at once: a contamination probe, this phase's
test and a set reserved for the next phase. The corpus, vectors, BM25, entity index and token
counts are Phase 9's, read-only.
"""

import gzip
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, write_text_atomic
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.retrieval.base import Hit

QUESTIONS_FILENAME = "questions.json"
DATA_STOP = "DATA_STOP"
DEV_STOP = "DEV_STOP"


class Phase10Error(Exception):
    """A Phase 10 input is missing, modified, or not the one this run may use."""


class DataStop(Phase10Error):
    """D2: the train release cannot supply the declared sets."""


# --- D2: the held-out draw --------------------------------------------------------------


def check_validation_level(validation_raws: Iterable[Mapping[str, Any]]) -> None:
    """The train level is chosen to match the validation questions; prove they are hard."""
    levels = {str(raw.get("level", "")) for raw in validation_raws}
    if levels != {config.PHASE_10_LEVEL}:
        raise DataStop(
            f"the validation questions are not all '{config.PHASE_10_LEVEL}': {sorted(levels)}"
        )


def hard_qids(raws: Iterable[Mapping[str, Any]]) -> list[str]:
    """The train qids at the validation level, in source order."""
    return [str(raw["_id"]) for raw in raws if raw.get("level") == config.PHASE_10_LEVEL]


def draw_sets(
    hard: Sequence[str],
    *,
    validation_qids: Iterable[str],
    seed: int = config.PHASE_10_SEED,
    sizes: Mapping[str, int] = config.PHASE_10_SET_SIZES,
) -> dict[str, list[str]]:
    """One seeded sample over the sorted qids, sliced in the declared order.

    Sorting first makes the draw a function of the qid set alone, not of the order the
    release happened to be paged in. One sample, sliced, makes the sets disjoint by
    construction rather than by a later check.
    """
    pool = sorted(set(hard))
    if len(pool) != len(hard):
        raise DataStop("the train release repeats a qid")
    overlap = set(pool) & set(validation_qids)
    if overlap:
        raise DataStop(f"{len(overlap)} train qids are also validation qids")
    needed = sum(sizes.values())
    if len(pool) < needed:
        raise DataStop(f"only {len(pool)} hard questions, the sets need {needed}")
    drawn = random.Random(seed).sample(pool, needed)  # noqa: S311 - a sample, not a secret
    sets: dict[str, list[str]] = {}
    start = 0
    for name, size in sizes.items():
        sets[name] = drawn[start : start + size]
        start += size
    return sets


def _set_questions(
    raws: Sequence[Mapping[str, Any]],
    split: str,
    resolutions: Mapping[str, phase9.TitleResolution],
) -> tuple[list[Question], list[dict[str, Any]], list[dict[str, Any]]]:
    questions: list[Question] = []
    entries: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for raw in raws:
        qid = str(raw["_id"])
        titles = phase9.supporting_titles(raw)
        gold: list[str] = []
        for position, title in enumerate(titles):
            resolution = resolutions[title]
            if resolution.unit_id is None:
                gold.append(f"{phase9.UNRESOLVED_PREFIX}{qid}:{position}")
                unresolved.append(
                    {
                        "qid": qid,
                        "title": title,
                        "status": resolution.status,
                        "candidates": resolution.candidates,
                    }
                )
            else:
                gold.append(resolution.unit_id)
        question = Question(
            qid=qid,
            question=str(raw["question"]),
            answer=str(raw["answer"]),
            gold_unit_ids=(gold[0], gold[1]),
            supporting_facts=tuple((str(t), int(i)) for t, i in raw["supporting_facts"]),
            split=split,
        )
        questions.append(question)
        entries.append(
            {
                "qid": qid,
                "set": split,
                "type": str(raw.get("type", "")),
                "question": question.question,
                "answer": question.answer,
                "supporting_titles": list(titles),
                "supporting_facts": [[t, i] for t, i in question.supporting_facts],
                "gold_unit_ids": list(question.gold_unit_ids),
                "resolution": [resolutions[title].status for title in titles],
            }
        )
    return questions, entries, unresolved


def build_sets(
    raws: Sequence[Mapping[str, Any]],
    sets: Mapping[str, Sequence[str]],
    units: Iterable[IndexingUnit],
    *,
    corpus: Mapping[str, Any],
    share: float = config.PHASE_10_UNRESOLVED_SHARE,
) -> tuple[dict[str, list[Question]], dict[str, Any]]:
    """Each set's questions with gold resolved under the Phase 9 D4 rule, and the body.

    A set with more than `share` of its questions carrying an unresolved gold title writes
    `DATA_STOP`; the unresolved title stays in the denominator as a sentinel no unit carries.
    """
    by_qid = {str(raw["_id"]): raw for raw in raws}
    missing = [qid for qids in sets.values() for qid in qids if qid not in by_qid]
    if missing:
        raise Phase10Error(f"{len(missing)} drawn qids are not in the train release")
    titles = {
        title
        for qids in sets.values()
        for qid in qids
        for title in phase9.supporting_titles(by_qid[qid])
    }
    resolutions = phase9.resolve_titles(titles, units)

    questions: dict[str, list[Question]] = {}
    set_bodies: dict[str, dict[str, Any]] = {}
    stop = False
    for name, qids in sets.items():
        chosen, entries, unresolved = _set_questions([by_qid[q] for q in qids], name, resolutions)
        questions[name] = chosen
        affected = len({entry["qid"] for entry in unresolved})
        over = affected > share * len(qids)
        stop = stop or over
        set_bodies[name] = {
            "n_questions": len(chosen),
            "question_digest": phase9._question_digest(chosen),
            "mapping_digest": phase9._mapping_digest(chosen, str(corpus["unit_set_hash"])),
            "questions_with_unresolved_gold": affected,
            "unresolved_ceiling_share": share,
            "over_ceiling": over,
            "unresolved": unresolved,
            "questions": entries,
        }
    body: dict[str, Any] = {
        "phase": 10,
        "corpus_unit_set_hash": corpus["unit_set_hash"],
        "corpus_ordered_unit_digest": corpus["ordered_unit_digest"],
        "distinct_titles": len(resolutions),
        "title_resolution": {
            status: sum(1 for r in resolutions.values() if r.status == status)
            for status in (phase9.EXACT, phase9.NORMALIZED, phase9.UNMATCHED, phase9.AMBIGUOUS)
        },
        "terminal_state": DATA_STOP if stop else None,
        "sets": set_bodies,
    }
    return questions, body


def write_questions(directory: Path | str, body: Mapping[str, Any]) -> Path:
    """Freeze the sets once. A second draw over an existing file is refused."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / QUESTIONS_FILENAME
    if target.exists():
        raise Phase10Error(f"{target} already freezes the Phase 10 sets")
    write_text_atomic(target, json.dumps(body, indent=2, sort_keys=True, ensure_ascii=True))
    return target


def load_set(directory: Path | str, name: str, *, corpus_unit_set_hash: str) -> list[Question]:
    """One frozen set's questions, their digests recomputed. `test-11` is never loaded here."""
    if name == config.PHASE_10_RESERVED:
        raise Phase10Error(f"{name} is reserved for the next phase; Phase 10 never loads it")
    body = json.loads((Path(directory) / QUESTIONS_FILENAME).read_text(encoding="utf-8"))
    if body["terminal_state"] is not None:
        raise DataStop(f"the Phase 10 sets recorded {body['terminal_state']}")
    if body["corpus_unit_set_hash"] != corpus_unit_set_hash:
        raise Phase10Error("the Phase 10 sets were mapped against another corpus")
    record = body["sets"][name]
    questions = [
        Question(
            qid=str(entry["qid"]),
            question=str(entry["question"]),
            answer=str(entry["answer"]),
            gold_unit_ids=(str(entry["gold_unit_ids"][0]), str(entry["gold_unit_ids"][1])),
            supporting_facts=tuple((str(t), int(i)) for t, i in entry["supporting_facts"]),
            split=name,
        )
        for entry in record["questions"]
    ]
    if (
        phase9._question_digest(questions) != record["question_digest"]
        or phase9._mapping_digest(questions, corpus_unit_set_hash) != record["mapping_digest"]
    ):
        raise Phase10Error(f"the {name} set does not match its recorded digests")
    return questions


# --- S4/S5: scoring rankings that were computed once --------------------------------------


class ReplayRetriever:
    """Serves precomputed rankings, in question order, to the unchanged Phase 9 harness.

    The dev grid re-fuses cached component lists instead of re-running retrieval 66 times;
    this hands each fused list to `measure_system` exactly as a live system would, and
    refuses a query that is not the one the list was computed for.
    """

    def __init__(self, name: str, rankings: Sequence[tuple[str, list[Hit]]]) -> None:
        self.name = name
        self._rankings = list(rankings)
        self._position = 0

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        expected, hits = self._rankings[self._position]
        if query != expected:
            raise Phase10Error(f"replay out of order at {self._position}: {query!r}")
        self._position += 1
        return list(hits[:top_k])


def supported(records: Sequence[Mapping[str, Any]], budget: int = 2048) -> int:
    """Full Support successes at `budget`, in whole questions."""
    return int(sum(r["budgets"][str(budget)]["full_support"] for r in records))


def gold_recall_sum(records: Sequence[Mapping[str, Any]], budget: int = 2048) -> float:
    return float(sum(r["budgets"][str(budget)]["gold_recall"] for r in records))


DEV_LISTS_FILENAME = "dev-lists.jsonl.gz"
DEV_LISTS_MANIFEST = "dev-lists.json"
LIST_NAMES: tuple[str, ...] = ("dense", "bm25", config.ENTITY_HOP_NAME)


def write_dev_lists(
    directory: Path | str, rows: Sequence[Mapping[str, Any]], *, provenance: Mapping[str, Any]
) -> Path:
    """The three component lists per dev question, written once with their digest."""
    directory = Path(directory)
    target = directory / DEV_LISTS_FILENAME
    if target.exists():
        raise Phase10Error(f"{target} already holds the dev lists")
    text = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(text.encode("utf-8"))
    temporary.replace(target)
    body = {"rows": len(rows), "digest": digest_of(text), "lists": list(LIST_NAMES), **provenance}
    write_text_atomic(directory / DEV_LISTS_MANIFEST, json.dumps(body, indent=2, sort_keys=True))
    return target


def load_dev_lists(directory: Path | str) -> list[dict[str, Any]]:
    directory = Path(directory)
    body = json.loads((directory / DEV_LISTS_MANIFEST).read_text(encoding="utf-8"))
    text = gzip.decompress((directory / DEV_LISTS_FILENAME).read_bytes()).decode("utf-8")
    if digest_of(text) != body["digest"]:
        raise Phase10Error(f"{DEV_LISTS_FILENAME} does not match its recorded digest")
    rows = [json.loads(line) for line in text.splitlines()]
    for row in rows:
        for name in LIST_NAMES:
            row[name] = [(str(u), float(s)) for u, s in row[name]]
    return rows


def reproduction_verdict(counts: Mapping[str, int]) -> dict[str, Any]:
    """D4: the laptop's dev counts against Phase 9's, exact."""
    reference = config.PHASE_10_DEV_REFERENCE_SUPPORTED
    checks = {
        name: {"observed": int(counts[name]), "recorded": reference[name]} for name in reference
    }
    return {
        "checks": checks,
        "passed": all(c["observed"] == c["recorded"] for c in checks.values()),
        "metric": "full_support@2048_tokens over the 7,405 dev questions",
    }


# --- D3: the contamination probe ----------------------------------------------------------


def probe_verdict(
    probe_supported: int, probe_n: int, dev_supported: int, dev_n: int
) -> dict[str, Any]:
    probe_fs = 100.0 * probe_supported / probe_n
    dev_fs = 100.0 * dev_supported / dev_n
    difference = probe_fs - dev_fs
    over = difference > config.PHASE_10_PROBE_MARGIN_PP
    return {
        "probe_supported": probe_supported,
        "probe_n": probe_n,
        "probe_fs_pct": probe_fs,
        "dev_supported": dev_supported,
        "dev_n": dev_n,
        "dev_fs_pct": dev_fs,
        "difference_pp": difference,
        "margin_pp": config.PHASE_10_PROBE_MARGIN_PP,
        "terminal_state": DATA_STOP if over else None,
    }


# --- D5: the dev grid and its gate ---------------------------------------------------------


def weight_grid(tenths: int = config.PHASE_10_GRID_TENTHS) -> list[tuple[float, float, float]]:
    """Every convex triple on a grid of `1 / tenths`, ordered by (dense, bm25) descending."""
    return [
        (d / tenths, b / tenths, (tenths - d - b) / tenths)
        for d in range(tenths, -1, -1)
        for b in range(tenths - d, -1, -1)
    ]


def choose_point(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Most Full Support; then gold recall; then larger dense weight; then larger BM25 weight."""
    return dict(
        max(
            curve,
            key=lambda p: (p["supported"], p["gold_recall_sum"], p["weights"][0], p["weights"][1]),
        )
    )


def dev_gate(chosen: Mapping[str, Any], control_supported: int) -> str | None:
    """`DEV_STOP` when the fit drops the Entity Hop or does not beat the control on dev."""
    if chosen["weights"][2] == 0.0 or chosen["supported"] <= control_supported:
        return DEV_STOP
    return None


# --- D6: the label -------------------------------------------------------------------------


def three_way_outcome(
    control: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """P10-C against P10-B, Full Support @2,048, exact McNemar, and the D6 label."""
    budget = "2048"
    b = {r["qid"]: r["budgets"][budget]["full_support"] for r in control}
    c = {r["qid"]: r["budgets"][budget]["full_support"] for r in candidate}
    if set(b) != set(c) or len(b) != config.PHASE_10_SET_SIZES[config.PHASE_10_TEST]:
        raise Phase10Error("the two runs do not hold the same test-10 questions")
    wins = sum(1 for q in b if c[q] == 1.0 and b[q] == 0.0)
    losses = sum(1 for q in b if b[q] == 1.0 and c[q] == 0.0)
    p = phase9.exact_two_sided_p(wins, losses)
    if wins > losses and p < config.PHASE_9_ALPHA:
        label = "THREE_WAY_SUPPORTED"
    elif losses > wins and p < config.PHASE_9_ALPHA:
        label = "THREE_WAY_REGRESSION"
    else:
        label = "THREE_WAY_NOT_SUPPORTED"
    n = len(b)
    control_successes, candidate_successes = int(sum(b.values())), int(sum(c.values()))
    return {
        "set": config.PHASE_10_TEST,
        "metric": "full_support@2048_tokens",
        "n_questions": n,
        "control_successes": control_successes,
        "candidate_successes": candidate_successes,
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
        "delta_percentage_points": 100.0 * (candidate_successes - control_successes) / n,
        "exact_two_sided_p": p,
        "alpha": config.PHASE_9_ALPHA,
        "test": "exact two-sided McNemar (binomial on discordant questions, p = 1/2)",
        "terminal_state": label,
    }
