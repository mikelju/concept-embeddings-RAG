"""The node index: which paragraph holds which entity and which concept (Phase 5, HU-3, D7).

Built from an extraction by normalization v1, and nothing else. Two namespaces - an entity
and a concept with the same form are two nodes - and a binary matrix of paragraphs x nodes
in pool order, so a row is a paragraph's node set with no duplicate in it.

Node ids are assigned by sorting (type, form), which makes the index a function of the
extraction alone: the same records, in any order, give the same ids and the same bytes.

The index is regenerable and git-ignored, like the concept matrices. It is written with a
digest over the matrix and its metadata and verified on load, because every navigation
figure is computed from it.
"""

import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, savez_compressed_atomic, write_text_atomic
from concept_embeddings_rag.nodes.extraction import OK, ExtractionRecord
from concept_embeddings_rag.nodes.normalization import VERSION, normalize

ENTITY, CONCEPT = config.NODE_TYPES


class NodeIndexError(Exception):
    """The node index is missing, modified, or not the one this run may use."""


@dataclass(frozen=True)
class Node:
    node_id: int
    type: str
    form: str


@dataclass(frozen=True, eq=False)
class NodeIndex:
    unit_ids: tuple[str, ...]
    nodes: tuple[Node, ...]
    incidence: sparse.csr_matrix
    extraction_digest: str
    failed_units: int
    dropped_empty: dict[str, int]
    normalization_version: str = VERSION
    digest: str = field(default="")

    def columns_of(self, types: Sequence[str]) -> np.ndarray:
        """The node ids whose type is in `types`, ascending."""
        wanted = set(types)
        return np.array(
            [node.node_id for node in self.nodes if node.type in wanted], dtype=np.int64
        )


def build_node_index(
    records: Mapping[str, ExtractionRecord],
    unit_ids: Sequence[str],
    *,
    extraction_digest: str,
) -> NodeIndex:
    """Normalize every record into typed nodes and lay them out in pool order."""
    missing = [unit_id for unit_id in unit_ids if unit_id not in records]
    if missing:
        raise NodeIndexError(
            f"the extraction does not cover {len(missing)} paragraph(s) of the pool; every unit "
            "needs a record, ok or failed, before its row can be built"
        )

    dropped: Counter[str] = Counter({node_type: 0 for node_type in config.NODE_TYPES})
    failed = 0
    rows: list[set[tuple[str, str]]] = []
    for unit_id in unit_ids:
        record = records[unit_id]
        keys: set[tuple[str, str]] = set()
        if record.status != OK:
            failed += 1
        else:
            for node_type, raw_forms in ((ENTITY, record.entities), (CONCEPT, record.concepts)):
                for raw in raw_forms:
                    form = normalize(raw)
                    if form:
                        keys.add((node_type, form))
                    else:
                        dropped[node_type] += 1
        rows.append(keys)

    order = {node_type: position for position, node_type in enumerate(config.NODE_TYPES)}
    vocabulary = sorted({key for keys in rows for key in keys}, key=lambda k: (order[k[0]], k[1]))
    node_id = {key: position for position, key in enumerate(vocabulary)}

    indptr = [0]
    indices: list[int] = []
    for keys in rows:
        indices.extend(sorted(node_id[key] for key in keys))
        indptr.append(len(indices))
    incidence = sparse.csr_matrix(
        (
            np.ones(len(indices)),
            np.array(indices, dtype=np.int64),
            np.array(indptr, dtype=np.int64),
        ),
        shape=(len(unit_ids), len(vocabulary)),
    )

    index = NodeIndex(
        unit_ids=tuple(unit_ids),
        nodes=tuple(Node(position, key[0], key[1]) for position, key in enumerate(vocabulary)),
        incidence=incidence,
        extraction_digest=extraction_digest,
        failed_units=failed,
        dropped_empty={node_type: int(dropped[node_type]) for node_type in config.NODE_TYPES},
    )
    return _with_digest(index)


def _metadata(index: NodeIndex) -> dict[str, Any]:
    return {
        "unit_ids": list(index.unit_ids),
        "nodes": [{"node_id": n.node_id, "type": n.type, "form": n.form} for n in index.nodes],
        "extraction_digest": index.extraction_digest,
        "failed_units": index.failed_units,
        "dropped_empty": index.dropped_empty,
        "normalization_version": index.normalization_version,
        "shape": list(index.incidence.shape),
    }


def _compute_digest(indptr: np.ndarray, indices: np.ndarray, metadata: Mapping[str, Any]) -> str:
    return digest_of(
        indptr.astype(np.int64),
        indices.astype(np.int64),
        json.dumps(metadata, sort_keys=True, ensure_ascii=True),
    )


def _with_digest(index: NodeIndex) -> NodeIndex:
    digest = _compute_digest(index.incidence.indptr, index.incidence.indices, _metadata(index))
    return NodeIndex(
        unit_ids=index.unit_ids,
        nodes=index.nodes,
        incidence=index.incidence,
        extraction_digest=index.extraction_digest,
        failed_units=index.failed_units,
        dropped_empty=index.dropped_empty,
        normalization_version=index.normalization_version,
        digest=digest,
    )


def _stem(extraction_digest: str) -> str:
    return f"nodes-{extraction_digest[:16]}"


def save_node_index(index: NodeIndex, directory: Path | str) -> Path:
    """Write the matrix and its sidecar. Regenerable, so an identical rewrite is harmless."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = _stem(index.extraction_digest)
    savez_compressed_atomic(
        directory / f"{stem}.npz",
        indptr=index.incidence.indptr.astype(np.int64),
        indices=index.incidence.indices.astype(np.int64),
    )
    sidecar = directory / f"{stem}.json"
    write_text_atomic(
        sidecar,
        json.dumps({**_metadata(index), "digest": index.digest}, indent=2, sort_keys=True),
    )
    return sidecar


def load_node_index(
    directory: Path | str,
    *,
    extraction_digest: str | None = None,
    expected_unit_ids: Sequence[str] | None = None,
) -> NodeIndex:
    """Read the index back, verified against its digest, its normalization and the pool."""
    directory = Path(directory)
    if extraction_digest is not None:
        sidecars = [directory / f"{_stem(extraction_digest)}.json"]
    else:
        sidecars = sorted(directory.glob("nodes-*.json")) if directory.exists() else []
    if len(sidecars) != 1 or not sidecars[0].exists():
        raise NodeIndexError(
            f"expected exactly one node index in {directory}, found {len(sidecars)}: "
            "run `cer nodes`"
        )
    sidecar = sidecars[0]
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    with np.load(sidecar.with_suffix(".npz"), allow_pickle=False) as arrays:
        indptr = arrays["indptr"]
        indices = arrays["indices"]
    recorded = payload.pop("digest", None)
    if recorded is None or _compute_digest(indptr, indices, payload) != recorded:
        raise NodeIndexError(f"{sidecar.name} does not match its digest; it has been modified")
    if payload["normalization_version"] != VERSION:
        raise NodeIndexError(
            f"the index was built with {payload['normalization_version']}, not {VERSION}; "
            "rebuild it with `cer nodes`"
        )
    unit_ids = tuple(payload["unit_ids"])
    if expected_unit_ids is not None and unit_ids != tuple(expected_unit_ids):
        raise NodeIndexError("the index was built over another pool; rebuild it with `cer nodes`")
    shape = tuple(payload["shape"])
    incidence = sparse.csr_matrix((np.ones(len(indices)), indices, indptr), shape=shape)
    return NodeIndex(
        unit_ids=unit_ids,
        nodes=tuple(
            Node(int(n["node_id"]), str(n["type"]), str(n["form"])) for n in payload["nodes"]
        ),
        incidence=incidence,
        extraction_digest=str(payload["extraction_digest"]),
        failed_units=int(payload["failed_units"]),
        dropped_empty={str(k): int(v) for k, v in payload["dropped_empty"].items()},
        normalization_version=str(payload["normalization_version"]),
        digest=recorded,
    )


def _summary(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": int(max(values)),
    }


def fragmentation(index: NodeIndex) -> dict[str, dict[str, Any]]:
    """HU-3's figures, per node type: numbers in place of an impression."""
    report: dict[str, dict[str, Any]] = {}
    for node_type in config.NODE_TYPES:
        columns = index.columns_of([node_type])
        sub = (
            index.incidence[:, columns]
            if columns.size
            else sparse.csr_matrix((len(index.unit_ids), 0))
        )
        paragraphs_per_node = np.asarray(sub.getnnz(axis=0)).ravel().tolist()
        nodes_per_paragraph = np.asarray(sub.getnnz(axis=1)).ravel().tolist()
        report[node_type] = {
            "distinct_nodes": int(columns.size),
            "singleton_share": (
                float(sum(1 for count in paragraphs_per_node if count == 1) / columns.size)
                if columns.size
                else 0.0
            ),
            "paragraphs_per_node": _summary(paragraphs_per_node),
            "nodes_per_paragraph": _summary(nodes_per_paragraph),
            "paragraphs_without_node": int(sum(1 for count in nodes_per_paragraph if count == 0)),
        }
    return report
