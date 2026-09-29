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
"""

import gzip
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation import phase14
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
