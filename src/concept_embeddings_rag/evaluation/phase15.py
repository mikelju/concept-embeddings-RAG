"""Phase 15: the frozen systems on MuSiQue. S4: integrity before the pass (D4); S5: the
pass's marker, its rankings files and the hop lists at the five Phase 14 `alpha`.

`15.spec.md` (approved and frozen 2026-09-29) fixes every rule here before any Phase 15
number exists. Nothing is fitted on MuSiQue (D3): the four systems fuse with the weights and
`alpha` recorded in the HotpotQA fits, and the reader below refuses any other value.

D4 has three halves, each a verdict here, and any miss is `DATA_STOP`:

- **code identity**: the Phase 15 weight reader and `fuse_lists`, replayed over the recorded
  HotpotQA dev lists, give P10-C's 4,801 and P14's 5,224 of 7,405 exactly, and no question
  moves between the Phase 10 and Phase 14 dev-lists files;
- **extractor identity**: the pod's GLiNER configuration digest is the Phase 9 pod digest,
  and its library versions are Phase 9's key by key;
- **live path**: the four live systems complete on the 200 MuSiQue train questions of the
  live-path set, and their ranking records are well formed and survive serialization. No
  metric of those questions is computed: they carry no gold, and nothing here scores.

The ranking record is the prospective fix of deviation 14.1: the pass (S5) stores, per
question and system, the fused list and each component's list with scores to depth 100, and
the hop's P1, `|C(q)|` and, for P14, `sim_seconds`. The live path builds, checks and
round-trips the same records in memory, so S5 writes a format already exercised.

S6 reads the pass back: the paired comparisons and the label (D5, D6), the ladder beside
HotpotQA `test-11`, the split by supporting paragraphs (D7) and what the hop reaches (D8),
all from the recorded outcomes and rankings. S7 (D9, exploratory) checks that re-fusing the
stored lists at the frozen points reproduces the recorded outcomes before any refit is kept.
"""

import gzip
import json
import math
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation import phase11, phase14
from concept_embeddings_rag.evaluation.budget import fill_context
from concept_embeddings_rag.evaluation.entity_diagnostics import (
    RecordingHybrid,
    RecordingRetriever,
    RecordingStage,
)
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.entity_hop import (
    RELEVANCE_HOP_NAME,
    RelevanceExpansion,
    RelevanceHopStage,
)
from concept_embeddings_rag.retrieval.fusion import SEEDED_HOP_NAME, TRIPLE_COMPONENTS, fuse_lists

INTEGRITY_FILENAME = "integrity.json"
FIT_FILENAME = "fit.json"

P10A, P10B, P10C, P14 = config.PHASE_15_SYSTEMS
EXPECTED_DEV_SUPPORTED: dict[str, int] = {
    P10C: config.PHASE_15_P10C_DEV_SUPPORTED,
    P14: config.PHASE_15_P14_DEV_SUPPORTED,
}


class Phase15Error(Exception):
    """A Phase 15 input is missing, modified, or not the one this step may use."""


# --- The frozen weights (D3, R4) ------------------------------------------------------------


def fit_digest(fit: Mapping[str, Any]) -> str:
    """A recorded fit's digest, as the Phase 10 and 14 passes recorded it."""
    return digest_of(json.dumps(dict(fit), sort_keys=True))


def _agrees(recorded: Mapping[str, Any], names: Sequence[str], frozen: Sequence[float]) -> bool:
    return sorted(recorded) == sorted(names) and all(
        abs(float(recorded[name]) - weight) <= config.PHASE_9_WEIGHT_TOLERANCE
        for name, weight in zip(names, frozen, strict=True)
    )


def frozen_weights(
    *,
    is_committed: Callable[[Path], bool],
    phase10_fit: Path | str = config.PHASE_10_DIR / FIT_FILENAME,
    phase14_fit: Path | str = config.PHASE_14_DIR / FIT_FILENAME,
    bm25_file: Path | str = config.PHASE_9_BM25_WEIGHTS_FILE,
) -> dict[str, Any]:
    """The weights and `alpha` of the spec's table, read from the fits that chose them.

    P10-B from `phase9.read_frozen_weights`, P10-C from the Phase 10 fit, P14's `alpha` and
    weights from the Phase 14 fit. Each source must be tracked by git unmodified
    (`is_committed`) and hold the spec's values; anything else refuses, it is never adapted.
    As in Phase 9, the recorded floats are returned as recorded. The weights come keyed as each
    system fuses them: P14's hop is the stage named `seeded-hop`.
    """
    sources = {P10B: Path(bm25_file), P10C: Path(phase10_fit), P14: Path(phase14_fit)}
    for system, path in sources.items():
        if not is_committed(path):
            raise Phase15Error(
                f"{path} ({config.PHASE_15_SYSTEM_LABELS[system]}) is not committed unmodified"
            )
    p10b = phase9.read_frozen_weights(bm25_file)["hybrid-bm25"]
    if not _agrees(p10b, ("dense", "bm25"), config.PHASE_15_P10B_WEIGHTS):
        raise Phase15Error(f"P10-B's recorded weights are {p10b}, not the spec's 0.5 / 0.5")
    fits = {
        system: json.loads(sources[system].read_text(encoding="utf-8")) for system in (P10C, P14)
    }
    for system, fit in fits.items():
        if fit["terminal_state"] is not None:
            raise Phase15Error(f"the {system} fit recorded {fit['terminal_state']}")
    p10c, p14 = fits[P10C]["weights"], fits[P14]["weights"]
    if not _agrees(p10c, TRIPLE_COMPONENTS, config.PHASE_15_P10C_WEIGHTS):
        raise Phase15Error(f"P10-C's recorded weights are {p10c}, not the spec's 0.5 / 0.3 / 0.2")
    hop_names = ("dense", "bm25", RELEVANCE_HOP_NAME)
    if not _agrees(p14, hop_names, config.PHASE_15_P14_WEIGHTS):
        raise Phase15Error(f"P14's recorded weights are {p14}, not the spec's 0.5 / 0.3 / 0.2")
    alpha = float(fits[P14]["alpha"])
    if abs(alpha - config.PHASE_15_P14_ALPHA) > config.PHASE_9_WEIGHT_TOLERANCE:
        raise Phase15Error(f"P14's recorded alpha is {alpha}, not the spec's 0.75")
    return {
        "weights": {
            P10B: {name: float(p10b[name]) for name in ("dense", "bm25")},
            P10C: {name: float(p10c[name]) for name in TRIPLE_COMPONENTS},
            P14: {
                "dense": float(p14["dense"]),
                "bm25": float(p14["bm25"]),
                SEEDED_HOP_NAME: float(p14[RELEVANCE_HOP_NAME]),
            },
        },
        "alpha": alpha,
        "sources": {system: str(path) for system, path in sources.items()},
        "fit_digests": {system: fit_digest(fit) for system, fit in fits.items()},
    }


# --- D4: the three verdicts -----------------------------------------------------------------


def code_identity_verdict(
    observed: Mapping[str, Mapping[str, float]],
    reference: Mapping[str, Mapping[str, float]],
    *,
    expected: Mapping[str, int] = EXPECTED_DEV_SUPPORTED,
    n_questions: int = config.PHASE_15_DEV_QUESTIONS,
) -> dict[str, Any]:
    """P10-C and P14 on HotpotQA dev: the recorded count exactly, and no question moved.

    `observed` and `reference` map each system to its per-question Full Support @2,048 by
    qid: `observed` from the lists its own phase recorded, `reference` with the lists the other
    dev-lists file also holds (`cli._p15_code_identity`). A count off the recorded one, or any
    question whose outcome differs between the two, fails the system.
    """
    if set(observed) != set(expected) or set(reference) != set(expected):
        raise Phase15Error(f"code identity checks {sorted(expected)}, not {sorted(observed)}")
    checks: dict[str, Any] = {}
    for system, count in expected.items():
        outcome, other = observed[system], reference[system]
        if set(outcome) != set(other):
            raise Phase15Error(f"the two {system} routes did not read the same questions")
        supported = int(sum(outcome.values()))
        differing = sorted(qid for qid in outcome if outcome[qid] != other[qid])
        checks[system] = {
            "label": config.PHASE_15_SYSTEM_LABELS[system],
            "observed": supported,
            "expected": count,
            "n_questions": len(outcome),
            "differing_qids": differing,
            "passed": supported == count and not differing and len(outcome) == n_questions,
        }
    return {
        "metric": f"full_support@{config.PHASE_9_PRIMARY_BUDGET}_tokens over the "
        f"{n_questions} HotpotQA dev questions",
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def extractor_identity_verdict(
    extraction: Mapping[str, Any],
    phase9_extraction: Mapping[str, Any],
    *,
    pinned: str = config.PHASE_9_GLINER_CONFIGURATION_DIGEST,
) -> dict[str, Any]:
    """The Phase 15 GLiNER manifest against the pinned Phase 9 pod digest and manifest."""
    versions = dict(extraction["library_versions"])
    recorded = dict(phase9_extraction["library_versions"])
    differing = sorted(
        k for k in set(versions) | set(recorded) if versions.get(k) != recorded.get(k)
    )
    digest = str(extraction["configuration_digest"])
    phase9_digest = str(phase9_extraction["configuration_digest"])
    return {
        "configuration_digest": digest,
        "phase9_configuration_digest": phase9_digest,
        "pinned": pinned,
        "library_versions": versions,
        "library_versions_differing": differing,
        "passed": digest == pinned == phase9_digest and not differing,
    }


def d4_verdict(
    code_identity: Mapping[str, Any],
    extractor_identity: Mapping[str, Any],
    live_path: Mapping[str, Any],
) -> dict[str, Any]:
    """`DATA_STOP` on any miss of the three halves, each reason naming what missed."""
    reasons: list[str] = []
    for check in code_identity.get("checks", {}).values():
        if not check["passed"]:
            reasons.append(
                f"code identity: {check['label']} {check['observed']} of {check['n_questions']}, "
                f"recorded {check['expected']}; differing qids {check['differing_qids']}"
            )
    if not code_identity["passed"] and not reasons:
        reasons.append("code identity did not pass")
    if not extractor_identity["passed"]:
        reasons.append(
            f"extractor identity: digest {extractor_identity.get('configuration_digest')}, "
            f"library versions differing {extractor_identity.get('library_versions_differing')}"
        )
    if not live_path["passed"]:
        failed = [name for name, s in live_path["systems"].items() if not s["passed"]]
        reasons.append(f"live path: {failed or 'not passed'}")
    return {"terminal_state": phase9.DATA_STOP if reasons else None, "stop_reasons": reasons}


# --- The live systems and their ranking records ---------------------------------------------


class RecordingRelevanceStage:
    """P14's hop stage, keeping the `RelevanceExpansion` behind every list it proposes.

    Shaped like `RecordingStage`: it carries the inner stage's name (`seeded-hop`, the name
    `QueryTripleFusedRetriever` requires), delegates unchanged and returns the inner list.
    """

    def __init__(self, inner: RelevanceHopStage) -> None:
        self.inner = inner
        self.name = inner.name
        self.expansions: list[RelevanceExpansion] = []

    @property
    def sim_seconds(self) -> list[float]:
        return self.inner.sim_seconds

    def propose(self, query: str, first: Sequence[Hit], top_k: int) -> list[Hit]:
        expansion = self.inner.hop(query, first, top_k)
        self.expansions.append(expansion)
        return [(candidate.unit_id, candidate.score) for candidate in expansion.candidates]


@dataclass
class LiveSystem:
    """One live system and the recorders its ranking records are read from.

    `system` is what `retrieve` (and, in S5, `phase9.measure_system`) is handed; `components`
    are the recorders of the Dense and BM25 lists it fused, in fusion order; `hop` records the
    third component, named `hop_component` in the record; `fusion_weights` are the weights in
    that order, `None` for Dense alone.
    """

    name: str
    system: RecordingRetriever | RecordingHybrid
    components: dict[str, RecordingRetriever]
    hop: RecordingStage | RecordingRelevanceStage | None = None
    hop_component: str | None = None
    fusion_weights: tuple[float, ...] | None = None

    @property
    def fused(self) -> list[list[Hit]]:
        if isinstance(self.system, RecordingHybrid):
            return self.system.fused
        return self.system.calls


def _hits(hits: Sequence[Hit]) -> list[Hit]:
    return [(str(unit_id), float(score)) for unit_id, score in hits]


def ranking_records(live: LiveSystem, qids: Sequence[str]) -> list[dict[str, Any]]:
    """One record per question, in call order: the fused list and every component's list with
    scores, and the hop's P1 (Dense's first unit), `|C(q)|` and, for P14, `sim_seconds`."""
    lengths = {len(live.fused), *(len(r.calls) for r in live.components.values())}
    if live.hop is not None:
        lengths.add(len(live.hop.expansions))
    if lengths != {len(qids)}:
        raise Phase15Error(
            f"{live.name} recorded {sorted(lengths)} calls for {len(qids)} questions"
        )
    records: list[dict[str, Any]] = []
    for position, qid in enumerate(qids):
        components = {name: _hits(r.calls[position]) for name, r in live.components.items()}
        hop: dict[str, Any] | None = None
        if live.hop is not None and live.hop_component is not None:
            expansion = live.hop.expansions[position]
            components[live.hop_component] = [
                (c.unit_id, float(c.score)) for c in expansion.candidates
            ]
            dense = components["dense"]
            hop = {"p1": dense[0][0] if dense else None, "positives": int(expansion.positives)}
            if isinstance(live.hop, RecordingRelevanceStage):
                hop["sim_seconds"] = float(live.hop.sim_seconds[position])
        records.append(
            {
                "qid": qid,
                "system": live.name,
                "fused": _hits(live.fused[position]),
                "components": components,
                "hop": hop,
            }
        )
    return records


def ranking_problems(hits: Sequence[Hit], corpus_ids: Collection[str], *, depth: int) -> list[str]:
    """What makes one ranking malformed: past `depth`, a unit twice, a unit outside the corpus,
    a score that is not a finite number, or scores not in descending order."""
    problems: list[str] = []
    if len(hits) > depth:
        problems.append(f"{len(hits)} hits, past the depth of {depth}")
    ids = [unit_id for unit_id, _score in hits]
    repeated = sorted({unit_id for unit_id in ids if ids.count(unit_id) > 1})
    if repeated:
        problems.append(f"{repeated[:5]} listed twice")
    outside = [unit_id for unit_id in ids if unit_id not in corpus_ids]
    if outside:
        problems.append(f"{outside[:5]} not in the corpus")
    scores = [score for _unit_id, score in hits]
    if not all(math.isfinite(score) for score in scores):
        problems.append("a score is not a finite number")
    elif any(later > earlier for earlier, later in zip(scores, scores[1:], strict=False)):
        problems.append("scores not in descending order")
    return problems


def serialize_rankings(records: Sequence[Mapping[str, Any]]) -> str:
    """The rankings file's text: one JSON object per question, keys sorted."""
    return "".join(json.dumps(dict(record), sort_keys=True) + "\n" for record in records)


def parse_rankings(text: str) -> list[dict[str, Any]]:
    """The records back from their text, every list as `(unit_id, score)` pairs."""
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        record = json.loads(line)
        record["fused"] = _hits(record["fused"])
        record["components"] = {name: _hits(v) for name, v in record["components"].items()}
        if HOP_ALPHAS in record:
            record[HOP_ALPHAS] = {key: _hits(v) for key, v in record[HOP_ALPHAS].items()}
        records.append(record)
    return records


# --- S5: the pass's marker, its rankings file and the hop lists at the five alpha ----------

MARKER_FILENAME = "pass.json"
# P14's records carry the hop at each Phase 14 `alpha` under this key, apart from the
# components, so re-fusing `components` in order still gives the fused list.
HOP_ALPHAS = "hop_alphas"


def start_pass(directory: Path | str, body: Mapping[str, Any]) -> Path:
    """Mark the single pass as started. A second start is refused, whatever happened."""
    target = Path(directory) / MARKER_FILENAME
    if target.exists():
        raise Phase15Error(
            f"{target} exists: the pass has already started once and is never rerun; "
            "a failed pass is a decision for the author"
        )
    return write_text_atomic(target, json.dumps(dict(body), indent=2, sort_keys=True))


def rankings_name(system: str) -> str:
    return f"rankings-{system}.jsonl.gz"


def write_rankings(
    directory: Path | str, system: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    """One system's ranking records, written once as gzip with `mtime = 0`.

    Returns the file name and the digest of its text, which the run file records beside the
    outcomes digest (`load_rankings` reads it back only against that digest).
    """
    directory = Path(directory)
    archive = directory / rankings_name(system)
    if archive.exists():
        raise Phase15Error(f"{archive} already exists; a rankings file is written once")
    text = serialize_rankings(records)
    temporary = archive.with_name(archive.name + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(text.encode("utf-8"))
    temporary.replace(archive)
    return {"rankings_file": archive.name, "rankings_digest": digest_of(text)}


def load_rankings(directory: Path | str, system: str) -> list[dict[str, Any]]:
    """One system's ranking records, refused unless they match the digest its run records."""
    directory = Path(directory)
    run_name = f"run-{system}.json"
    body = json.loads((directory / run_name).read_text(encoding="utf-8"))
    text = gzip.decompress((directory / str(body["rankings_file"])).read_bytes()).decode("utf-8")
    if digest_of(text) != body["rankings_digest"]:
        raise Phase15Error(f"{body['rankings_file']} does not match the digest {run_name} records")
    return parse_rankings(text)


def hop_alpha_lists(
    stage: RelevanceHopStage,
    questions: Sequence[Question],
    records: Sequence[Mapping[str, Any]],
    *,
    depth: int = config.PHASE_9_RANKING_DEPTH,
    alphas: Sequence[float] = config.PHASE_14_ALPHAS,
) -> list[dict[str, list[Hit]]]:
    """The hop at every `alpha`, one `hop_all` per question over its stored Dense list.

    Run after the timed systems, outside any timing: D9 re-fuses these lists. Keyed as the
    Phase 14 dev lists are (`phase14.alpha_key`), in `alphas` order.
    """
    if [r["qid"] for r in records] != [q.qid for q in questions]:
        raise Phase15Error("the stored rankings are not in the question order")
    lists: list[dict[str, list[Hit]]] = []
    for question, record in zip(questions, records, strict=True):
        expansions = stage.hop_all(question.question, record["components"]["dense"], depth, alphas)
        lists.append(
            {
                phase14.alpha_key(alpha): [(c.unit_id, float(c.score)) for c in e.candidates]
                for alpha, e in expansions.items()
            }
        )
    return lists


def hop_alpha_mismatches(
    p14_records: Sequence[Mapping[str, Any]],
    p10c_records: Sequence[Mapping[str, Any]],
    lists: Sequence[Mapping[str, Sequence[Hit]]],
    *,
    alpha: float,
) -> list[str]:
    """Each question whose `alpha = 0` list is not P10-C's recorded hop list, or whose frozen
    `alpha` list is not P14's, as `<qid> alpha=<a>`. Any entry stops the pass."""
    zero, frozen = phase14.alpha_key(0.0), phase14.alpha_key(alpha)
    mismatches: list[str] = []
    for p14, p10c, row in zip(p14_records, p10c_records, lists, strict=True):
        if p14["qid"] != p10c["qid"]:
            raise Phase15Error("the P10-C and P14 rankings are not in the same question order")
        if list(row[zero]) != p10c["components"][config.ENTITY_HOP_NAME]:
            mismatches.append(f"{p14['qid']} alpha=0.00")
        if list(row[frozen]) != p14["components"][RELEVANCE_HOP_NAME]:
            mismatches.append(f"{p14['qid']} alpha={alpha:.2f}")
    return mismatches


def with_hop_alphas(
    records: Sequence[Mapping[str, Any]], lists: Sequence[Mapping[str, Sequence[Hit]]]
) -> list[dict[str, Any]]:
    """P14's records with each question's hop lists at the five `alpha` added."""
    return [
        {**record, HOP_ALPHAS: {key: _hits(hits) for key, hits in row.items()}}
        for record, row in zip(records, lists, strict=True)
    ]


def _seconds(values: Sequence[float]) -> dict[str, float]:
    return {
        "total": float(sum(values)),
        "mean": float(sum(values) / len(values)) if values else 0.0,
        "max": float(max(values, default=0.0)),
    }


def _check_system(
    live: LiveSystem, qids: Sequence[str], corpus_ids: Collection[str], depth: int
) -> dict[str, Any]:
    """The records of one completed system: well formed, re-fusable and serializable."""
    records = ranking_records(live, qids)
    problems: list[str] = []
    consistent = True
    for record in records:
        lists = {"fused": record["fused"], **record["components"]}
        for name, hits in lists.items():
            problems.extend(
                f"{record['qid']} {name}: {problem}"
                for problem in ranking_problems(hits, corpus_ids, depth=depth)
            )
        # The components in fusion order (Dense, BM25, the hop) must give back the fused list,
        # which is what S5's refit (D9) relies on.
        ordered = list(record["components"].values())
        again = (
            ordered[0]
            if live.fusion_weights is None
            else fuse_lists(ordered, live.fusion_weights, top_k=depth)
        )
        consistent = consistent and again == record["fused"]
    return {
        "records": len(records),
        "ranking_problems": problems,
        "fusion_consistent": consistent,
        "round_trip": parse_rankings(serialize_rankings(records)) == records,
    }


def run_live_path(
    systems: Mapping[str, LiveSystem],
    questions: Sequence[Question],
    corpus_ids: Collection[str],
    *,
    depth: int = config.PHASE_9_RANKING_DEPTH,
) -> dict[str, Any]:
    """D4's live path: every system's `retrieve` on every question, then its records checked.

    The questions must carry no gold: nothing here could score them, and nothing tries. Each
    call is timed with `phase9.TimedRetriever`; a call that raises stops that system and is
    recorded with its qid. The records are built, checked, re-fused from their components
    with `fuse_lists` and round-tripped through their serialization, in memory only.
    """
    if any(question.gold_unit_ids for question in questions):
        raise Phase15Error("a live-path question carries gold; the live path scores nothing")
    qids = [question.qid for question in questions]
    results: dict[str, Any] = {}
    for name, live in systems.items():
        timed = phase9.TimedRetriever(live.system)
        error: str | None = None
        failed_qid: str | None = None
        for question in questions:
            try:
                timed.retrieve(question.question, depth)
            except Exception as exception:  # noqa: BLE001 - any failure is a D4 miss, recorded
                error = f"{type(exception).__name__}: {exception}"
                failed_qid = question.qid
                break
        completed = error is None
        checked = (
            _check_system(live, qids, corpus_ids, depth)
            if completed
            else {
                "records": 0,
                "ranking_problems": [],
                "fusion_consistent": False,
                "round_trip": False,
            }
        )
        results[name] = {
            "label": config.PHASE_15_SYSTEM_LABELS[name],
            "completed": completed,
            "failed_qid": failed_qid,
            "error": error,
            "calls": len(timed.seconds),
            **checked,
            "seconds": _seconds(timed.seconds),
            "passed": completed
            and not checked["ranking_problems"]
            and bool(checked["fusion_consistent"])
            and bool(checked["round_trip"]),
        }
    return {
        "n_questions": len(questions),
        "qids": qids,
        "depth": depth,
        "order": list(systems),
        "systems": results,
        "no_metric": (
            "question text only, no gold: no metric of these questions is computed, printed "
            "or stored (D4); the ranking records were built and checked in memory only"
        ),
        "passed": all(result["passed"] for result in results.values()),
    }


# --- S6: the paired comparisons, the label and the descriptive splits (D5-D8) --------------

OUTCOME_FILENAME = "outcome.json"
REFIT_FILENAME = "refit.json"
TRANSFER_SUPPORTED, TRANSFER_NOT_SUPPORTED, TRANSFER_REGRESSION, _DATA_STOP = (
    config.PHASE_15_TERMINAL_STATES
)
BUDGET = str(config.PHASE_9_PRIMARY_BUDGET)

# D5 first, then D6's three, as (key, control, candidate). Author decision 4: D7 reports all
# four in each group. Only the first decides the label.
PRIMARY = "p14_vs_p10b"
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    (PRIMARY, P10B, P14),
    ("p14_vs_p10c", P10C, P14),
    ("p10c_vs_p10b", P10B, P10C),
    ("p10b_vs_p10a", P10A, P10B),
)
SUPPORTING_GROUPS: tuple[int, ...] = tuple(sorted(config.PHASE_15_SUPPORTING_COUNTS))

# D6: the four systems' Full Support @2,048 on HotpotQA `test-11`, measured in Phase 14 and
# read back from its run files, which must still hold them.
HOTPOTQA_TEST_11_QUESTIONS = 5000
HOTPOTQA_TEST_11_SUPPORTED: dict[str, int] = {P10A: 2852, P10B: 3079, P10C: 3285, P14: 3530}


def paired(
    control: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Full Support @2,048 of two runs on the same questions, exact McNemar, any `n`.

    `phase11.paired` without its `test-11` size check (the same code, `any_size=True`).
    """
    try:
        return phase11.paired(control, candidate, any_size=True)
    except phase11.Phase11Error as error:
        raise Phase15Error(str(error)) from error


def label(wins: int, losses: int, p: float) -> str:
    """D5: `TRANSFER_SUPPORTED`, `TRANSFER_REGRESSION` or `TRANSFER_NOT_SUPPORTED`, by the
    three-way rule of Phases 11-14 at `alpha = 0.05`."""
    return phase14.label(
        wins,
        losses,
        p,
        supported=TRANSFER_SUPPORTED,
        regression=TRANSFER_REGRESSION,
        not_supported=TRANSFER_NOT_SUPPORTED,
    )


def _labelled(control: str, candidate: str, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "control": config.PHASE_15_SYSTEM_LABELS[control],
        "candidate": config.PHASE_15_SYSTEM_LABELS[candidate],
        **result,
    }


def comparisons(runs: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """The four paired comparisons of `COMPARISONS`, each naming its control and candidate."""
    return {
        key: _labelled(control, candidate, paired(runs[control], runs[candidate]))
        for key, control, candidate in COMPARISONS
    }


def outcome_label(compared: Mapping[str, Mapping[str, Any]]) -> str:
    """The label, from P14 against P10-B only (D5); D6 cannot change it."""
    primary = compared[PRIMARY]
    return label(primary["wins"], primary["losses"], primary["exact_two_sided_p"])


def _supported(records: Sequence[Mapping[str, Any]]) -> int:
    return int(sum(r["budgets"][BUDGET]["full_support"] for r in records))


def _percent(count: int, n: int) -> float:
    return 100.0 * count / n


def ladder(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    hotpotqa: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """D6's progression P10-A -> P10-B -> P10-C -> P14: each rung's Full Support @2,048 on
    MuSiQue beside HotpotQA `test-11`, and each step's gain in points on both."""
    rungs: list[dict[str, Any]] = []
    for system in config.PHASE_15_SYSTEMS:
        records = runs[system]
        supported, n = _supported(records), len(records)
        other = hotpotqa[system]
        rungs.append(
            {
                "system": system,
                "label": config.PHASE_15_SYSTEM_LABELS[system],
                "musique": {
                    "supported": supported,
                    "n_questions": n,
                    "full_support_percent": _percent(supported, n),
                },
                "hotpotqa_test_11": {
                    "supported": int(other["supported"]),
                    "n_questions": int(other["n_questions"]),
                    "full_support_percent": _percent(
                        int(other["supported"]), int(other["n_questions"])
                    ),
                },
            }
        )
    steps: list[dict[str, Any]] = []
    for before, after in zip(rungs, rungs[1:], strict=False):
        musique_gain = (
            after["musique"]["full_support_percent"] - before["musique"]["full_support_percent"]
        )
        hotpotqa_gain = (
            after["hotpotqa_test_11"]["full_support_percent"]
            - before["hotpotqa_test_11"]["full_support_percent"]
        )
        steps.append(
            {
                "from": before["label"],
                "to": after["label"],
                "musique_gain_pp": musique_gain,
                "hotpotqa_gain_pp": hotpotqa_gain,
                "same_direction": bool(np.sign(musique_gain) == np.sign(hotpotqa_gain)),
            }
        )
    return {
        "metric": f"full_support@{BUDGET}_tokens",
        "rungs": rungs,
        "steps": steps,
        "note": (
            "the two columns differ in corpus and in questions: they compare the direction "
            "and order of the gains, not their size (D6)"
        ),
    }


def group_summary(
    members: Collection[str], runs: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    """One group of questions (`members`, by qid): each system's Full Support and mean gold
    recall @2,048 and the four paired comparisons over the group, `{"n_questions": 0}` for an
    empty group. The per-group body of `by_supporting` (D7) and of Phase 16's groupings."""
    if not members:
        return {"n_questions": 0}
    subset = {
        system: [r for r in records if r["qid"] in members] for system, records in runs.items()
    }
    systems: dict[str, Any] = {}
    for system in config.PHASE_15_SYSTEMS:
        records = subset[system]
        supported = _supported(records)
        systems[system] = {
            "label": config.PHASE_15_SYSTEM_LABELS[system],
            "full_support": supported,
            "full_support_share": supported / len(records),
            "mean_gold_recall": float(
                sum(r["budgets"][BUDGET]["gold_recall"] for r in records) / len(records)
            ),
        }
    return {"n_questions": len(members), "systems": systems, "comparisons": comparisons(subset)}


def by_supporting(
    questions: Sequence[Question],
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    groups: Sequence[int] = SUPPORTING_GROUPS,
) -> dict[str, Any]:
    """D7: per number of supporting paragraphs, each system's Full Support and mean gold
    recall @2,048 and the four paired comparisons. Descriptive, deciding nothing.

    The group is the question's count of `is_supporting` paragraphs (its `supporting_facts`,
    the count the spec's 1,252 / 760 / 405 refer to), not its gold count after two supporting
    paragraphs collapse into one unit; the gold counts inside each group are recorded.
    """
    size = {q.qid: len(q.supporting_facts) for q in questions}
    gold = {q.qid: len(q.gold_unit_ids) for q in questions}
    body: dict[str, Any] = {}
    for n in groups:
        members = {qid for qid, count in size.items() if count == n}
        body[str(n)] = group_summary(members, runs)
        if members:
            body[str(n)]["gold_counts"] = {
                str(k): v for k, v in sorted(Counter(gold[qid] for qid in members).items())
            }
    return {
        "grouping": "the question's number of supporting paragraphs (is_supporting)",
        "budget": int(BUDGET),
        "groups": body,
        "other": sum(1 for count in size.values() if count not in groups),
        "note": "descriptive, deciding nothing (D7); every comparison is exact McNemar",
    }


def refusion_mismatches(
    records: Sequence[Mapping[str, Any]],
    components: Sequence[str],
    weights: Sequence[float] | None,
    *,
    depth: int = config.PHASE_9_RANKING_DEPTH,
) -> list[str]:
    """The qids whose stored components, re-fused in the run file's `ranking_components`
    order, do not give the stored fused list. A parsed record's `components` come back in
    alphabetical key order, so the order is always taken from the run file."""
    mismatched: list[str] = []
    for record in records:
        if set(record["components"]) != set(components):
            mismatched.append(str(record["qid"]))
            continue
        ordered = [record["components"][name] for name in components]
        again = ordered[0] if weights is None else fuse_lists(ordered, weights, top_k=depth)
        if again != record["fused"]:
            mismatched.append(str(record["qid"]))
    return mismatched


def recorded_contexts(
    rankings: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    *,
    budget: int = config.PHASE_9_PRIMARY_BUDGET,
    depth: int = config.PHASE_9_RANKING_DEPTH,
) -> dict[str, list[str]]:
    """Each question's 2,048-token context, packed with `budget.fill_context` from the stored
    fused list, checked against the Full Support its outcome record reports."""
    contexts: dict[str, list[str]] = {}
    mismatched: list[str] = []
    for ranking, record, question in zip(rankings, outcomes, questions, strict=True):
        if not ranking["qid"] == record["qid"] == question.qid:
            raise Phase15Error("the rankings, outcomes and questions are not in one qid order")
        context = fill_context([u for u, _s in ranking["fused"][:depth]], token_counts, budget)
        full = float(set(question.gold_unit_ids) <= set(context))
        if full != float(record["budgets"][str(budget)]["full_support"]):
            mismatched.append(question.qid)
        contexts[question.qid] = context
    if mismatched:
        raise Phase15Error(
            f"the stored fused lists disagree with the recorded Full Support on "
            f"{len(mismatched)} questions: {mismatched[:20]}"
        )
    return contexts


def rank_split(
    questions: Sequence[Question],
    dense: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Sequence[str]],
    control: Mapping[str, Sequence[str]],
    *,
    depth: int = config.PHASE_9_RANKING_DEPTH,
) -> dict[str, Any]:
    """D8: `phase14.dense_rank_split` of two systems' checked contexts, by the Dense rank
    (in `dense`'s stored Dense component, depth 100) of each paragraph that changed."""
    entries = [
        {
            "qid": question.qid,
            "gold": list(question.gold_unit_ids),
            "candidate_context": list(candidate[question.qid]),
            "control_context": list(control[question.qid]),
            "dense_ids": [u for u, _s in record["components"]["dense"][:depth]],
        }
        for question, record in zip(questions, dense, strict=True)
    ]
    return phase14.dense_rank_split(entries)


def _size_summary(sizes: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(sizes, dtype=np.float64)
    if values.size == 0:
        return {"questions": 0}
    return {
        "questions": int(values.size),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": int(values.max()),
        "empty": int((values == 0).sum()),
        "empty_share": float((values == 0).mean()),
    }


def reach(
    questions: Sequence[Question],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    components: Mapping[str, Sequence[str]],
    *,
    dense_top: int = 10,
) -> dict[str, Any]:
    """D8: per system, how many supporting paragraphs outside Dense's first `dense_top` each
    component (and the fused list) lists at depth 100; for the systems with a hop, whether P1
    is a supporting paragraph and the hop's candidate-set sizes `|C(q)|`.

    Read from the stored rankings only; `components` is each system's `ranking_components`.
    """
    gold = {q.qid: set(q.gold_unit_ids) for q in questions}
    systems: dict[str, Any] = {}
    for system, records in rankings.items():
        names = [*components[system], "fused"]
        listed = {name: {"gold": 0, "questions": 0} for name in names}
        outside_total = 0
        p1_supporting = p1_absent = 0
        sizes: list[int] = []
        hop_system = any(record.get("hop") is not None for record in records)
        for record in records:
            dense_ids = [u for u, _s in record["components"]["dense"]]
            outside = gold[record["qid"]] - set(dense_ids[:dense_top])
            outside_total += len(outside)
            lists = {**record["components"], "fused": record["fused"]}
            for name in names:
                found = outside & {u for u, _s in lists[name]}
                listed[name]["gold"] += len(found)
                listed[name]["questions"] += bool(found)
            if hop_system:
                hop = record["hop"]
                p1 = hop["p1"]
                if p1 != (dense_ids[0] if dense_ids else None):
                    raise Phase15Error(f"{system} {record['qid']}: P1 is not Dense's first unit")
                p1_absent += p1 is None
                p1_supporting += p1 in gold[record["qid"]]
                sizes.append(int(hop["positives"]))
        entry: dict[str, Any] = {
            "label": config.PHASE_15_SYSTEM_LABELS.get(system, system),
            "gold_outside_dense_top_10": outside_total,
            "listed": listed,
        }
        if hop_system:
            entry["p1"] = {
                "questions": len(records),
                "p1_supporting": p1_supporting,
                "p1_supporting_share": p1_supporting / len(records) if records else 0.0,
                "p1_absent": p1_absent,
            }
            entry["candidate_set"] = _size_summary(sizes)
        systems[system] = entry
    return {
        "dense_top": dense_top,
        "depth": config.PHASE_9_RANKING_DEPTH,
        "systems": systems,
        "note": (
            "'gold' counts supporting paragraphs outside the system's own Dense first "
            f"{dense_top} that the list holds at depth 100; 'questions' the questions with at "
            "least one. Descriptive, deciding nothing (D8)."
        ),
    }


# --- S7: the refit's consistency (D9, exploratory) -----------------------------------------


def replay_consistency(
    observed: Sequence[Mapping[str, Any]], recorded: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """A replayed point against a system's recorded outcomes: the count, and every question
    whose Full Support @2,048 moved. Any moved question fails it."""
    if [r["qid"] for r in observed] != [r["qid"] for r in recorded]:
        raise Phase15Error("the replay and the recorded outcomes are not in one qid order")
    differing = [
        str(a["qid"])
        for a, b in zip(observed, recorded, strict=True)
        if a["budgets"][BUDGET]["full_support"] != b["budgets"][BUDGET]["full_support"]
    ]
    return {
        "supported": _supported(observed),
        "recorded": _supported(recorded),
        "differing_qids": differing,
        "passed": not differing,
    }
