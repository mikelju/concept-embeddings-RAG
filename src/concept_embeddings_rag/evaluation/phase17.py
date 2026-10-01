"""Phase 17: two zero-shot judges over the frozen systems. The pure core.

`17.spec.md` (approved and frozen 2026-09-30) fixes every rule here before any Phase 17
number exists; `17.0_judge_and_paper.md` (approved 2026-10-01) fixes how.

- **The pool (D1)**: per question, the union of the four systems' fused top-100, each unit
  with its 1-based rank in every system that lists it, ordered by best rank, then unit id.
- **The reorder (D1)**: a system under a judge is its own top-100 sorted by the judge's
  score, descending, ties broken by the system's original rank; nothing outside it enters.
  The union line sorts the whole pool by score, ties by best rank, then unit id; it is
  descriptive and enters no comparison.
- **D6**: gold recall and full support at k units, nDCG@10, and the judge ceiling; for
  k <= 20 they equal the Phase 9 harness's `fs_at_k` / `gpr_at_k` (checked).
- **D5**: eleven paired comparisons per set over `phase15.paired`, and the label from
  J(P14) against J(P10-B) through `phase14.label`.

Every file is written once, as gzip with `mtime = 0` and no embedded file name, so the same
rows always give the same bytes; its SHA-256 and the digest of its text are recorded. File
names and set names live here, never in `config` (the Phase 3 guard).
"""

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation import phase14, phase15
from concept_embeddings_rag.evaluation.metrics import full_support, ndcg_at_k, recall_at_k

P10A, P10B, P10C, P14 = config.PHASE_17_SYSTEMS
HOP_ADDS, HOP_HURTS, HOP_NEUTRAL = config.PHASE_17_LABELS
DATA_STOP = phase9.DATA_STOP
BUDGET = phase15.BUDGET

INTEGRITY_FILENAME = "integrity.json"
POOL_MANIFEST = "pool.json"

# A pool entry: a unit id and its 1-based rank in each system that lists it.
PoolEntry = tuple[str, dict[str, int]]
# A reordered entry: unit id, the judge's score, and the original (or best) rank.
Reordered = tuple[str, float, int]


class Phase17Error(Exception):
    """A Phase 17 input is missing, modified, or not the one this step may use."""


# --- File names ------------------------------------------------------------------------------


def pool_name(set_name: str) -> str:
    return f"pool-{set_name}.jsonl.gz"


def pairs_name(set_name: str) -> str:
    return f"pairs-{set_name}.jsonl.gz"


def scores_name(judge: str, set_name: str, *, tag: str | None = None) -> str:
    suffix = f"-{tag}" if tag else ""
    return f"scores-{judge}-{set_name}{suffix}.jsonl.gz"


def scoring_manifest_name(judge: str) -> str:
    return f"scoring-{judge}.json"


def shard_directory_name(judge: str, set_name: str, *, tag: str | None = None) -> str:
    """Where a (judge, set) run keeps its shards under `config.PHASE_17_SHARDS_DIR`."""
    suffix = f"-{tag}" if tag else ""
    return f"{judge}-{set_name}{suffix}"


def shard_name(index: int) -> str:
    return f"shard-{index:04d}.jsonl.gz"


def shard_meta_name(index: int) -> str:
    return f"shard-{index:04d}.json"


SHARD_RUNS_NAME = "runs.json"
# J-decision (deviation 17.1): the offline embedding passes of a (judge, set) run.
DECISION_OFFLINE_NAME = "offline.json"


def reordered_name(judge: str, set_name: str, line: str) -> str:
    return f"reordered-{judge}-{set_name}-{line}.jsonl.gz"


# --- The pool (D1) ---------------------------------------------------------------------------


def pool_of(
    lists: Mapping[str, Sequence[str]], *, depth: int = config.PHASE_17_DEPTH
) -> list[PoolEntry]:
    """The union of each system's first `depth` units, every unit with its rank per system,
    ordered by its best rank over the systems, then unit id."""
    ranks: dict[str, dict[str, int]] = {}
    for system, units in lists.items():
        top = list(units)[:depth]
        if len(set(top)) != len(top):
            raise Phase17Error(f"{system}'s list repeats a unit")
        for rank, unit_id in enumerate(top, start=1):
            ranks.setdefault(unit_id, {})[system] = rank
    return sorted(
        ((unit_id, dict(sorted(by.items()))) for unit_id, by in ranks.items()),
        key=lambda entry: (min(entry[1].values()), entry[0]),
    )


def best_rank(ranks: Mapping[str, int]) -> int:
    return min(ranks.values())


def pool_summary(pools: Sequence[tuple[str, Sequence[PoolEntry]]]) -> dict[str, Any]:
    """Questions, pairs, pool size (mean, median, p90, min, max) and how many pairs are
    listed by one, two, three or four systems."""
    sizes = np.asarray([len(pool) for _qid, pool in pools], dtype=np.float64)
    listed = {str(n): 0 for n in range(1, len(config.PHASE_17_SYSTEMS) + 1)}
    for _qid, pool in pools:
        for _unit, ranks in pool:
            listed[str(len(ranks))] += 1
    return {
        "questions": len(pools),
        "pairs": int(sizes.sum()),
        "size": {
            "mean": float(sizes.mean()),
            "median": float(np.percentile(sizes, 50)),
            "p90": float(np.percentile(sizes, 90)),
            "min": int(sizes.min()),
            "max": int(sizes.max()),
        },
        "units_listed_by": listed,
    }


def pair_rows(
    questions: Sequence[Question],
    pools: Sequence[tuple[str, Sequence[PoolEntry]]],
    texts: Mapping[str, str],
) -> list[dict[str, Any]]:
    """The pod's input: per question, its text and the pool's units with their texts, in
    pool order. A pool unit without a text refuses."""
    rows: list[dict[str, Any]] = []
    for question, (qid, pool) in zip(questions, pools, strict=True):
        if question.qid != qid:
            raise Phase17Error(f"the pools are not in the question order at {qid}")
        unit_ids = [unit_id for unit_id, _ranks in pool]
        missing = [unit_id for unit_id in unit_ids if unit_id not in texts]
        if missing:
            raise Phase17Error(f"{qid}: pool units with no text in the corpus: {missing[:5]}")
        rows.append(
            {
                "qid": qid,
                "question": question.question,
                "unit_ids": unit_ids,
                "texts": [texts[unit_id] for unit_id in unit_ids],
            }
        )
    return rows


# --- The reorder rule and the union line (D1) ------------------------------------------------


def _score(scores: Mapping[str, float], unit_id: str) -> float:
    if unit_id not in scores:
        raise Phase17Error(f"{unit_id} has no score")
    return float(scores[unit_id])


def reorder(
    top: Sequence[str], scores: Mapping[str, float], *, depth: int = config.PHASE_17_DEPTH
) -> list[Reordered]:
    """J(S): S's first `depth` units by score, descending, ties by S's original rank."""
    entries = [
        (unit_id, _score(scores, unit_id), rank)
        for rank, unit_id in enumerate(list(top)[:depth], start=1)
    ]
    return sorted(entries, key=lambda entry: (-entry[1], entry[2]))


def union_order(pool: Sequence[PoolEntry], scores: Mapping[str, float]) -> list[Reordered]:
    """The union line: the whole pool by score, descending, ties by best rank, then id."""
    entries = [(unit_id, _score(scores, unit_id), best_rank(ranks)) for unit_id, ranks in pool]
    return sorted(entries, key=lambda entry: (-entry[1], entry[2], entry[0]))


# --- D6: the comparable metrics and the ceiling ----------------------------------------------


def question_metrics(
    ranked: Sequence[str],
    gold: Sequence[str],
    *,
    ks: Sequence[int] = config.PHASE_17_KS,
    ndcg_k: int = config.PHASE_17_NDCG_K,
) -> dict[str, Any]:
    """One question's gold recall and full support at each k, and its nDCG@`ndcg_k`."""
    ranked = list(ranked)
    return {
        "gold_recall_at_k": {str(k): recall_at_k(ranked, gold, k) for k in ks},
        "full_support_at_k": {str(k): float(full_support(ranked[:k], gold)) for k in ks},
        f"ndcg_at_{ndcg_k}": ndcg_at_k(ranked, gold, ndcg_k),
    }


def full_support_at(ranked: Sequence[str], gold: Sequence[str], k: int) -> bool:
    """Whether the first `k` units hold every gold unit (D6's full support for one question)."""
    return bool(full_support(list(ranked)[:k], gold))


def line_metrics(
    rankings: Sequence[Sequence[str]],
    questions: Sequence[Question],
    *,
    ks: Sequence[int] = config.PHASE_17_KS,
    ndcg_k: int = config.PHASE_17_NDCG_K,
) -> dict[str, Any]:
    """D6 for one line: each metric averaged over the questions (a share, not a count)."""
    per = [
        question_metrics(ranked, question.gold_unit_ids, ks=ks, ndcg_k=ndcg_k)
        for ranked, question in zip(rankings, questions, strict=True)
    ]
    n = len(per)
    if n == 0:
        raise Phase17Error("a line with no questions has no metrics")
    ndcg = f"ndcg_at_{ndcg_k}"
    return {
        "n_questions": n,
        "gold_recall_at_k": {
            str(k): sum(p["gold_recall_at_k"][str(k)] for p in per) / n for k in ks
        },
        "full_support_at_k": {
            str(k): sum(p["full_support_at_k"][str(k)] for p in per) / n for k in ks
        },
        ndcg: sum(p[ndcg] for p in per) / n,
    }


def harness_mismatches(
    rankings: Sequence[Sequence[str]],
    questions: Sequence[Question],
    records: Sequence[Mapping[str, Any]],
    *,
    ks: Sequence[int] = config.PHASE_9_KS,
) -> list[str]:
    """The qids whose D6 values at k <= 20 differ from the harness's `fs_at_k` and
    `gpr_at_k` in the replay record of the same list."""
    mismatched: list[str] = []
    for ranked, question, record in zip(rankings, questions, records, strict=True):
        if record["qid"] != question.qid:
            raise Phase17Error("the records are not in the question order")
        mine = question_metrics(ranked, question.gold_unit_ids, ks=ks)
        if any(
            mine["full_support_at_k"][str(k)] != float(record["fs_at_k"][str(k)])
            or mine["gold_recall_at_k"][str(k)] != float(record["gpr_at_k"][str(k)])
            for k in ks
        ):
            mismatched.append(question.qid)
    return mismatched


def ceiling(
    questions: Sequence[Question],
    candidates: Sequence[Sequence[str]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """The judge ceiling of one line: questions with every gold unit among its candidates
    (a system's top-100, or the whole pool for the union line), its Full Support @2,048, and
    the failures at 2,048 among those questions."""
    inside = failures = supported = 0
    for question, units, record in zip(questions, candidates, records, strict=True):
        if record["qid"] != question.qid:
            raise Phase17Error("the records are not in the question order")
        success = float(record["budgets"][BUDGET]["full_support"]) == 1.0
        supported += int(success)
        if set(question.gold_unit_ids) <= set(units):
            inside += 1
            failures += int(not success)
    return {
        "n_questions": len(questions),
        "all_gold_inside": inside,
        "full_support_2048": supported,
        "failures_inside": failures,
    }


# --- D5: the eleven comparisons and the label ------------------------------------------------

PRIMARY = "j_p14_vs_j_p10b"
STRONG_VS_LIGHT = "strong_vs_light_p10b"
DECISION_VS_STRONG = "decision_vs_strong_p10b"
# Per judge, as (key, control, candidate), each side `(judged, system)`; the first decides
# the label. Then, once per set, J-strong(P10-B) against J-light(P10-B) and, by deviation
# 17.1, J-decision(P10-B) against J-strong(P10-B).
JUDGE_COMPARISONS: tuple[tuple[str, tuple[bool, str], tuple[bool, str]], ...] = (
    (PRIMARY, (True, P10B), (True, P14)),
    ("j_p10b_vs_p10b", (False, P10B), (True, P10B)),
    ("j_p14_vs_p14", (False, P14), (True, P14)),
    ("j_p10c_vs_p10c", (False, P10C), (True, P10C)),
    ("j_p10a_vs_p10a", (False, P10A), (True, P10A)),
)


def line_key(judge: str | None, line: str) -> str:
    """A line's key: the system (or union) alone, or `<judge>:<line>` under a judge."""
    return line if judge is None else f"{judge}:{line}"


def line_label(judge: str | None, line: str) -> str:
    name = config.PHASE_17_SYSTEM_LABELS.get(line, line)
    return name if judge is None else f"{config.PHASE_17_JUDGES[judge]['label']}({name})"


def _compared(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    control: tuple[str | None, str],
    candidate: tuple[str | None, str],
) -> dict[str, Any]:
    return {
        "control": line_label(*control),
        "candidate": line_label(*candidate),
        **phase15.paired(runs[line_key(*control)], runs[line_key(*candidate)]),
    }


def comparisons(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    judges: Sequence[str] = (
        config.PHASE_17_LIGHT,
        config.PHASE_17_STRONG,
        config.PHASE_17_DECISION,
    ),
) -> dict[str, Any]:
    """D5 on one set: five comparisons per judge, then the two cross-judge comparisons on
    P10-B, J-strong against J-light and J-decision against J-strong (deviation 17.1).
    `runs` maps every `line_key` to its replay records."""
    compared: dict[str, Any] = {}
    for judge in judges:
        compared[judge] = {
            key: _compared(
                runs,
                (judge if control[0] else None, control[1]),
                (judge if candidate[0] else None, candidate[1]),
            )
            for key, control, candidate in JUDGE_COMPARISONS
        }
    light, strong = config.PHASE_17_LIGHT, config.PHASE_17_STRONG
    decision = config.PHASE_17_DECISION
    compared[STRONG_VS_LIGHT] = _compared(runs, (light, P10B), (strong, P10B))
    compared[DECISION_VS_STRONG] = _compared(runs, (strong, P10B), (decision, P10B))
    return compared


def hop_label(result: Mapping[str, Any]) -> str:
    """D5's label from J(P14) against J(P10-B), by the three-way rule of Phases 11-16."""
    return phase14.label(
        int(result["wins"]),
        int(result["losses"]),
        float(result["exact_two_sided_p"]),
        supported=HOP_ADDS,
        regression=HOP_HURTS,
        not_supported=HOP_NEUTRAL,
    )


# --- D3: the reproduced counts and the verdict ----------------------------------------------


def supported_in(questions: Sequence[Question], contexts: Mapping[str, Sequence[str]]) -> int:
    """Questions whose packed context holds every gold unit (Full Support), from the contexts
    `phase15.recorded_contexts` packed and checked against the recorded outcomes."""
    return sum(
        1 for question in questions if set(question.gold_unit_ids) <= set(contexts[question.qid])
    )


def count_checks(
    observed: Mapping[str, Mapping[str, int]],
    expected: Mapping[str, Mapping[str, int]],
    n_questions: Mapping[str, int],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Each expected count beside the observed one, per set and system; a system with no
    observed count is a miss."""
    checks: dict[str, dict[str, dict[str, Any]]] = {}
    for set_name, systems in expected.items():
        checks[set_name] = {}
        for system, count in systems.items():
            seen = observed.get(set_name, {}).get(system)
            checks[set_name][system] = {
                "label": config.PHASE_17_SYSTEM_LABELS[system],
                "observed": None if seen is None else int(seen),
                "expected": int(count),
                "n_questions": int(n_questions[set_name]),
                "passed": seen is not None and int(seen) == int(count),
            }
    return checks


def d3_verdict(
    checks: Mapping[str, Mapping[str, Mapping[str, Any]]], *, extra_reasons: Sequence[str] = ()
) -> dict[str, Any]:
    """`DATA_STOP` on any miss, each reason naming the set, the system and both counts."""
    reasons = [
        f"{set_name} {check['label']}: {check['observed']} of {check['n_questions']}, "
        f"recorded {check['expected']}"
        for set_name, systems in checks.items()
        for check in systems.values()
        if not check["passed"]
    ]
    reasons.extend(extra_reasons)
    return {"terminal_state": DATA_STOP if reasons else None, "stop_reasons": reasons}


# --- Shards ----------------------------------------------------------------------------------


def shard_bounds(
    n_questions: int, size: int = config.PHASE_17_SHARD_QUESTIONS
) -> list[tuple[int, int]]:
    """`[start, stop)` of each shard of `size` questions, in order."""
    return [(start, min(start + size, n_questions)) for start in range(0, n_questions, size)]


# --- Write-once files ------------------------------------------------------------------------


def write_jsonl_once(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Rows as gzip JSON lines, keys sorted, written once and atomically.

    Streamed, so the largest pairs file never sits whole in memory. Returns the file name,
    its SHA-256, the digest of its text (`artifacts.digest_of` of the whole text), its bytes
    and its rows.
    """
    path = Path(path)
    if path.exists():
        raise Phase17Error(f"{path} already exists; a Phase 17 file is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text_hash = hashlib.sha256()
    count = 0
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as packed,
    ):
        for row in rows:
            line = (json.dumps(dict(row), sort_keys=True) + "\n").encode("utf-8")
            text_hash.update(line)
            packed.write(line)
            count += 1
    # `digest_of(text)` appends one separator byte after its single part.
    text_hash.update(b"\x00")
    temporary.replace(path)
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "digest": text_hash.hexdigest(),
        "bytes": path.stat().st_size,
        "rows": count,
    }


def sha256_file(path: Path | str) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_jsonl(
    path: Path | str, *, digest: str | None = None, sha256: str | None = None
) -> list[dict[str, Any]]:
    """Rows back from a file this module wrote, refused unless its digest (and SHA-256, when
    given) is the recorded one."""
    path = Path(path)
    if not path.exists():
        raise Phase17Error(f"{path} does not exist")
    if sha256 is not None and sha256_file(path) != sha256:
        raise Phase17Error(f"{path.name} does not match its recorded SHA-256")
    text_hash = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rb") as packed:
        for line in packed:
            text_hash.update(line)
            rows.append(json.loads(line))
    text_hash.update(b"\x00")
    if digest is not None and text_hash.hexdigest() != digest:
        raise Phase17Error(f"{path.name} does not match its recorded digest")
    return rows


def write_pool(
    directory: Path | str, set_name: str, pools: Sequence[tuple[str, Sequence[PoolEntry]]]
) -> dict[str, Any]:
    rows = (
        {"qid": qid, "units": [[unit_id, dict(ranks)] for unit_id, ranks in pool]}
        for qid, pool in pools
    )
    return write_jsonl_once(Path(directory) / pool_name(set_name), rows)


def read_pool(
    directory: Path | str, set_name: str, *, digest: str, sha256: str | None = None
) -> list[tuple[str, list[PoolEntry]]]:
    rows = read_jsonl(Path(directory) / pool_name(set_name), digest=digest, sha256=sha256)
    return [
        (
            str(row["qid"]),
            [
                (str(unit_id), {str(s): int(r) for s, r in ranks.items()})
                for unit_id, ranks in row["units"]
            ],
        )
        for row in rows
    ]


def write_pairs(
    directory: Path | str, set_name: str, rows: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    return write_jsonl_once(Path(directory) / pairs_name(set_name), rows)


def read_pairs(
    directory: Path | str, set_name: str, *, digest: str, sha256: str | None = None
) -> list[dict[str, Any]]:
    return read_jsonl(Path(directory) / pairs_name(set_name), digest=digest, sha256=sha256)


def write_scores(
    directory: Path | str,
    judge: str,
    set_name: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    tag: str | None = None,
) -> dict[str, Any]:
    """`{"qid", "unit_ids", "scores"}` per question, in the pairs file's order."""
    return write_jsonl_once(Path(directory) / scores_name(judge, set_name, tag=tag), rows)


def read_scores(
    directory: Path | str,
    judge: str,
    set_name: str,
    *,
    digest: str,
    sha256: str | None = None,
    tag: str | None = None,
) -> list[dict[str, Any]]:
    path = Path(directory) / scores_name(judge, set_name, tag=tag)
    return read_jsonl(path, digest=digest, sha256=sha256)


def write_reordered(
    directory: Path | str,
    judge: str,
    set_name: str,
    line: str,
    lists: Sequence[tuple[str, Sequence[Reordered]]],
) -> dict[str, Any]:
    """`{"qid", "ranked": [[unit_id, score, rank]]}`, the rank the original (or best) one."""
    rows = (
        {"qid": qid, "ranked": [[u, float(s), int(r)] for u, s, r in ranked]}
        for qid, ranked in lists
    )
    return write_jsonl_once(Path(directory) / reordered_name(judge, set_name, line), rows)


def read_reordered(
    directory: Path | str,
    judge: str,
    set_name: str,
    line: str,
    *,
    digest: str,
    sha256: str | None = None,
) -> list[tuple[str, list[Reordered]]]:
    path = Path(directory) / reordered_name(judge, set_name, line)
    rows = read_jsonl(path, digest=digest, sha256=sha256)
    return [
        (str(row["qid"]), [(str(u), float(s), int(r)) for u, s, r in row["ranked"]]) for row in rows
    ]
