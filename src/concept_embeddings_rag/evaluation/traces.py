"""The per-query traces of the expansion, written down so they can be read back.

A trace says which concept entered at which round and which unit it brought along.
It is a diagnostic and nothing else: no metric reads one, no decision is taken from
one, and the retriever produces them only when asked. What makes it worth keeping
on disk is that HU-8's reading of the recovered questions has to rest on the run
that actually produced the ranking rather than on a reconstruction of it.

Two rules shape this module.

**A trace is bound to its configuration.** The file is named by the seed arm and
the configuration digest, carries both inside as well, and the reader refuses a file
whose digest is not the one it was asked for. A trace read against another run
explains a ranking that was never produced, in a document that looks exactly as
authoritative as a correct one.

**A trace holds concept ids, never labels.** `retrieval/` may not import the
labelling artifact, and the one way a label could get in there is from inside the
data - so the writer refuses one and so does the reader, on a file whose hash
verifies perfectly.

Verified on load, written atomically: the rules the Phase 1 audit left behind.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, write_text_atomic
from concept_embeddings_rag.retrieval.diffusion import (
    STOP_REASONS,
    ExpansionTrace,
    IterationTrace,
)

TRACES_PREFIX: str = "traces"


class TraceArtifactError(Exception):
    """The trace artifact is missing, inconsistent, or not the one asked for."""


def traces_path(directory: Path | str, *, seed_arm: str, config_digest: str) -> Path:
    """One file per (seed arm, configuration), named so a human can tell them apart."""
    _check_names(seed_arm, config_digest)
    return Path(directory) / f"{TRACES_PREFIX}-{seed_arm}-{config_digest}.json"


def save_traces(
    traces: Sequence[ExpansionTrace],
    directory: Path | str,
    *,
    seed_arm: str,
    config_digest: str,
) -> Path:
    """Write one run's traces as a single visible step.

    Unlike a selection, this may be written again: a trace decided nothing, and a
    rerun of the same configuration produces the same walk. What it may not do is
    mix two runs, which is why every trace in the file is checked against the arm
    and the digest the file is filed under.
    """
    _check_names(seed_arm, config_digest)
    if not traces:
        raise TraceArtifactError(
            f"the trace file for {seed_arm}/{config_digest} would be empty; a file holding "
            "no walk records nothing and would still claim a configuration"
        )

    seen: set[str] = set()
    for trace in traces:
        if trace.seed_arm != seed_arm:
            raise TraceArtifactError(
                f"trace {trace.qid} was produced on the {trace.seed_arm!r} arm and is being "
                f"filed under {seed_arm!r}; one file is one arm"
            )
        if trace.config_digest != config_digest:
            raise TraceArtifactError(
                f"trace {trace.qid} names configuration digest {trace.config_digest} and is "
                f"being filed under {config_digest}; one file is one run"
            )
        if trace.stop_reason not in STOP_REASONS:
            raise TraceArtifactError(
                f"trace {trace.qid} stopped for {trace.stop_reason!r}, which is not one of "
                f"{sorted(STOP_REASONS)}"
            )
        if trace.qid in seen:
            raise TraceArtifactError(
                f"question {trace.qid} is traced twice in one file; one walk is one trace"
            )
        seen.add(trace.qid)

    payload = {
        "seed_arm": seed_arm,
        "config_digest": config_digest,
        "n_traces": len(traces),
        "traces": [_trace_payload(trace) for trace in traces],
    }
    payload["digest"] = _payload_digest(payload)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = traces_path(directory, seed_arm=seed_arm, config_digest=config_digest)
    write_text_atomic(path, _serialize(payload))
    return path


def load_traces(
    directory: Path | str, *, seed_arm: str, config_digest: str
) -> list[ExpansionTrace]:
    """Read the traces back, proving they describe the run being asked about.

    Three checks, in this order: the file exists for this arm and digest, the digest
    over its own contents verifies, and what it holds is what it claims - the same
    arm, the same configuration, concept ids rather than labels.
    """
    path = traces_path(directory, seed_arm=seed_arm, config_digest=config_digest)
    if not path.exists():
        raise TraceArtifactError(
            f"no trace file for {seed_arm}/{config_digest} in {directory}; run the expansion "
            "with tracing enabled before reading one"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    recorded = payload.get("digest")
    actual = _payload_digest(payload)
    if recorded != actual:
        raise TraceArtifactError(
            f"the traces at {path.name} hash to {actual[:16]} but record "
            f"{str(recorded)[:16]}; they have been modified"
        )

    if str(payload.get("seed_arm")) != seed_arm or str(payload.get("config_digest")) != (
        config_digest
    ):
        raise TraceArtifactError(
            f"{path.name} holds the {payload.get('seed_arm')!r} arm at configuration digest "
            f"{payload.get('config_digest')}, not {seed_arm!r} at {config_digest}"
        )

    traces = [_trace_from_payload(entry, path.name) for entry in payload["traces"]]
    if len(traces) != int(payload["n_traces"]):
        raise TraceArtifactError(
            f"{path.name} says it holds {payload['n_traces']} traces and holds {len(traces)}"
        )
    return traces


def _check_names(seed_arm: str, config_digest: str) -> None:
    if seed_arm not in config.SEED_ARMS:
        raise TraceArtifactError(
            f"unknown seed arm: {seed_arm!r}; expected one of {sorted(config.SEED_ARMS)}"
        )
    if not config_digest or not all(character in "0123456789abcdef" for character in config_digest):
        raise TraceArtifactError(
            f"a configuration digest is a hexadecimal name, and this one is {config_digest!r}"
        )


def _serialize(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _payload_digest(payload: Mapping[str, Any]) -> str:
    """The digest is taken over the serialization, minus the digest field itself."""
    return digest_of(_serialize({key: value for key, value in payload.items() if key != "digest"}))


def _trace_payload(trace: ExpansionTrace) -> dict[str, Any]:
    return {
        "qid": trace.qid,
        "seed_arm": trace.seed_arm,
        "config_digest": trace.config_digest,
        "stop_reason": trace.stop_reason,
        "iterations": [
            {
                "index": row.index,
                "concepts": [[concept, mass] for concept, mass in row.concepts],
                "units": [[unit_id, mass] for unit_id, mass in row.units],
                "new_mass": row.new_mass,
                "stopped": row.stopped,
            }
            for row in trace.iterations
        ],
    }


def _trace_from_payload(payload: Mapping[str, Any], filename: str) -> ExpansionTrace:
    rows: list[IterationTrace] = []
    for entry in payload["iterations"]:
        concepts = []
        for concept, mass in entry["concepts"]:
            if not isinstance(concept, int) or isinstance(concept, bool):
                raise TraceArtifactError(
                    f"{filename} names concept {concept!r} rather than a concept id; a label "
                    "belongs to the report that reads a trace, never to the trace"
                )
            concepts.append((int(concept), float(mass)))
        rows.append(
            IterationTrace(
                index=int(entry["index"]),
                concepts=tuple(concepts),
                units=tuple((str(unit_id), float(mass)) for unit_id, mass in entry["units"]),
                new_mass=float(entry["new_mass"]),
                stopped=bool(entry["stopped"]),
            )
        )
    return ExpansionTrace(
        qid=str(payload["qid"]),
        seed_arm=str(payload["seed_arm"]),
        config_digest=str(payload["config_digest"]),
        iterations=tuple(rows),
        stop_reason=str(payload["stop_reason"]),
    )
