"""Phase 12: choosing with the question which of P1's entities the hop starts from.

`12.spec.md` (approved 2026-09-27) fixes every rule here before any Phase 12 number exists.
The Phase 9 hop is unchanged; what is new is *which* of P1's entity nodes seed it. D1 locates
each of P1's entity nodes in P1's own sentences by text, scores those sentences by cosine
similarity to the question (both encoded with the pinned Phase 9 Dense model), and D2 keeps
only the entities located in the top-ranked sentences, optionally excluding the ones the
question already names. This module holds those rules as pure functions over plain data, the
S2 sentence cache, the D4 reproduction and the D3/D5 grid, tie rule and gate. The runtime
stage and retriever live in `retrieval/entity_hop.py` and `retrieval/fusion.py`.
"""

import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, savez_compressed_atomic, write_text_atomic
from concept_embeddings_rag.embeddings.cache import unit_set_hash
from concept_embeddings_rag.nodes.normalization import normalize

DEV_STOP = "DEV_STOP"


class Phase12Error(Exception):
    """A Phase 12 input is missing, modified, or not the one this run may use."""


# --- S1 (D1, D2): location, exclusion and seed choice, over plain data ---------------------


def occurs(form: str, text: str) -> bool:
    """Whether the normalized `form` occurs in normalized `text`, bounded by non-word chars.

    Both sides are normalized here (`nodes.normalization.normalize`), so a caller may pass
    either an already-normalized node form or a raw span: normalization is idempotent.
    """
    needle = normalize(form)
    if not needle:
        return False
    haystack = normalize(text)
    pattern = re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)")
    return pattern.search(haystack) is not None


def locate(
    p1_nodes: Sequence[int], forms: Mapping[int, str], sentences: Sequence[str]
) -> dict[int, list[int]]:
    """Node id -> ascending sentence positions where it is located (D1).

    A node whose form is empty, or that occurs in none of `sentences`, is omitted: the title
    is not a sentence, so an entity mentioned only there is unlocated.
    """
    located: dict[int, list[int]] = {}
    for node_id in p1_nodes:
        form = forms.get(int(node_id), "")
        positions = [pos for pos, sentence in enumerate(sentences) if occurs(form, sentence)]
        if positions:
            located[int(node_id)] = positions
    return located


def named_in_question(p1_nodes: Sequence[int], forms: Mapping[int, str], question: str) -> set[int]:
    """The P1 entity nodes excluded when `x = yes` (D2): named in the question's own text."""
    return {int(node_id) for node_id in p1_nodes if occurs(forms.get(int(node_id), ""), question)}


def choose_seeds(
    p1_nodes: Sequence[int],
    located: Mapping[int, Sequence[int]],
    rel: Sequence[float],
    *,
    s: int | None,
    exclude: Collection[int] = (),
) -> list[int]:
    """The seed node ids, ascending (D2).

    Eligible entities are `p1_nodes` minus `exclude`. At `s = None` ("all") every eligible
    entity is a seed, unlocated ones included - this is what makes `s = None, exclude = False`
    exactly the Phase 9 hop. For `s` an integer, the sentences that hold at least one located
    eligible entity are ranked by `rel` descending, ties by position ascending; the seeds are
    the eligible entities located in the first `s` ranked sentences. Unlocated entities are
    never seeds when `s` is not `None`. An empty eligible set, or one whose entities are all
    unlocated at `s` not `None`, yields an empty seed list.
    """
    eligible = [int(node) for node in p1_nodes if int(node) not in exclude]
    if s is None:
        return sorted(set(eligible))
    holding_positions = sorted(
        {pos for node in eligible if node in located for pos in located[node]}
    )
    ranked = sorted(holding_positions, key=lambda pos: (-rel[pos], pos))
    top = set(ranked[:s])
    seeds = {
        node for node in eligible if node in located and any(pos in top for pos in located[node])
    }
    return sorted(seeds)


def seed_key(s: int | None, exclude: bool) -> str:
    """The dev-list / grid key for one seed configuration, e.g. `seeded-hop@s=all-x=no`."""
    return f"seeded-hop@s={'all' if s is None else s}-x={'yes' if exclude else 'no'}"


def seed_grid(
    sentences: Sequence[int | None] = config.PHASE_12_SENTENCES,
    exclude: Sequence[bool] = config.PHASE_12_EXCLUDE,
) -> list[tuple[int | None, bool]]:
    """The 6 seed configurations `(s, exclude)` (D2), in a fixed order."""
    return [(s, x) for s in sentences for x in exclude]


# --- S2: the sentence cache, one NPZ plus manifest per set, written once --------------------


def sentence_cache_key(model: str, revision: str, set_name: str, unit_ids: Sequence[str]) -> str:
    """A hash of the model, revision, text rule, set name and the sorted P1 unit ids."""
    payload = (
        f"sentence|{model}|{revision}|{config.PHASE_12_TEXT_RULE}|{set_name}|"
        f"{unit_set_hash(unit_ids)}"
    )
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def sentence_cache_filename(set_name: str, key: str) -> str:
    return f"sentences-{set_name}-{key}.npz"


def sentence_manifest_filename(set_name: str) -> str:
    return f"sentences-{set_name}.json"


def vectors_digest(vectors: np.ndarray) -> str:
    return digest_of(vectors.astype(np.float32))


def write_sentence_cache(
    directory: Path | str,
    set_name: str,
    key: str,
    unit_ids: Sequence[str],
    positions: Sequence[int],
    vectors: np.ndarray,
) -> Path:
    """The P1 sentence vectors of one set, one row per sentence, written once."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / sentence_cache_filename(set_name, key)
    if path.exists():
        raise Phase12Error(f"{path} already holds the {set_name} sentence cache")
    if vectors.shape[0] != len(unit_ids) or len(unit_ids) != len(positions):
        raise Phase12Error("unit_ids, positions and vectors must have one row each")
    savez_compressed_atomic(
        path,
        unit_ids=np.array(list(unit_ids), dtype=np.str_),
        positions=np.array(list(positions), dtype=np.int64),
        vectors=vectors.astype(np.float32),
    )
    return path


def write_sentence_manifest(directory: Path | str, set_name: str, body: Mapping[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / sentence_manifest_filename(set_name)
    if path.exists():
        raise Phase12Error(f"{path} already records the {set_name} sentence cache")
    write_text_atomic(path, json.dumps(dict(body), indent=2, sort_keys=True))
    return path


def load_sentence_manifest(directory: Path | str, set_name: str) -> dict[str, Any]:
    path = Path(directory) / sentence_manifest_filename(set_name)
    if not path.exists():
        raise Phase12Error(f"{path} does not exist: run 'p12-sentences' first")
    body: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return body


def load_sentence_cache(
    directory: Path | str, set_name: str, key: str
) -> tuple[list[str], list[int], np.ndarray]:
    path = Path(directory) / sentence_cache_filename(set_name, key)
    if not path.exists():
        raise Phase12Error(f"{path} does not exist: run 'p12-sentences' first")
    with np.load(path, allow_pickle=False) as payload:
        unit_ids = [str(u) for u in payload["unit_ids"]]
        positions = [int(p) for p in payload["positions"]]
        vectors = payload["vectors"]
    return unit_ids, positions, vectors


def read_sentence_cache(
    directory: Path | str, set_name: str
) -> tuple[list[str], list[int], np.ndarray, dict[str, Any]]:
    """The cache and its manifest for `set_name`, digest verified."""
    manifest = load_sentence_manifest(directory, set_name)
    unit_ids, positions, vectors = load_sentence_cache(directory, set_name, str(manifest["key"]))
    if vectors_digest(vectors) != manifest["vectors_digest"]:
        raise Phase12Error(f"the {set_name} sentence cache does not match its recorded digest")
    return unit_ids, positions, vectors, manifest


# --- S4 (D4, C2): the reproduction of P10-C on dev, and the location/seed coverage ----------


def reproduction_verdict(
    observed: int, differing_qids: Sequence[str], lists_differing_qids: Sequence[str]
) -> dict[str, Any]:
    """D4: exactly P10-C's dev count, no question moved, no seeded list moved."""
    recorded = config.PHASE_12_P10C_DEV_SUPPORTED
    s, exclude = config.PHASE_12_P10C_POINT
    weights = dict(zip(config.PHASE_12_COMPONENT_NAMES, config.PHASE_12_P10C_WEIGHTS, strict=True))
    return {
        "metric": "full_support@2048_tokens over the 7,405 dev questions",
        "point": {"s": s, "exclude": exclude, "weights": weights},
        "observed": int(observed),
        "recorded": recorded,
        "differing_qids": sorted(differing_qids),
        "seeded_list_differing_qids": sorted(lists_differing_qids),
        "passed": observed == recorded and not differing_qids and not lists_differing_qids,
    }


def _percentile(values: Sequence[float], share: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), share)) if values else 0.0


def coverage_summary(
    entity_totals: Sequence[tuple[int, int]],
    seed_counts_by_config: Mapping[tuple[int | None, bool], Sequence[int]],
) -> dict[str, Any]:
    """C2: the located share of P1 entity nodes, and per-`(s, exclude)` seed statistics.

    `entity_totals` is one `(p1_entity_nodes, located_entity_nodes)` pair per dev question,
    from any `exclude = False` configuration (location does not depend on `s` or `exclude`).
    `seed_counts_by_config` is the number of seeds chosen per dev question, for each of the 6
    configurations, in the same question order.
    """
    total_entities = sum(total for total, _located in entity_totals)
    total_located = sum(located for _total, located in entity_totals)
    by_config: dict[str, Any] = {}
    for (s, exclude), counts in seed_counts_by_config.items():
        counts = list(counts)
        empty = sum(1 for count in counts if count == 0)
        by_config[seed_key(s, exclude)] = {
            "empty_seed_share": empty / len(counts) if counts else 0.0,
            "seeds_median": float(np.median(counts)) if counts else 0.0,
            "seeds_p90": _percentile(counts, 90),
        }
    return {
        "p1_entity_nodes_total": total_entities,
        "p1_entity_nodes_located": total_located,
        "located_share": total_located / total_entities if total_entities else 0.0,
        "by_config": by_config,
    }


# --- S5 (D3, D5): the grid, the tie rule and the dev gate -----------------------------------


def weight_grid(tenths: int = config.PHASE_12_GRID_TENTHS) -> list[tuple[float, float, float]]:
    """Every convex triple (dense, bm25, seeded-hop) on a grid of `1 / tenths`: 66 for tenths.

    Numerically `phase10.weight_grid` again - not imported from there, because `phase10`
    reaches `evaluation.fullwiki`, which imports `retrieval.entity_hop`, and `entity_hop`
    imports this module; importing `phase10` here would cycle.
    """
    return [
        (d / tenths, b / tenths, (tenths - d - b) / tenths)
        for d in range(tenths, -1, -1)
        for b in range(tenths - d, -1, -1)
    ]


def grid_points(
    configs: Sequence[tuple[int | None, bool]] | None = None,
) -> list[tuple[tuple[int | None, bool], tuple[float, float, float]]]:
    """The 6 seed configurations x the 66 weight triples: 396 points (D3)."""
    configs_seq = list(configs) if configs is not None else seed_grid()
    return [(cfg, weights) for cfg in configs_seq for weights in weight_grid()]


def _s_rank(s: int | None) -> float:
    return float("inf") if s is None else float(s)


def selection_key(point: Mapping[str, Any]) -> tuple[Any, ...]:
    """D3, in order: Full Support; gold recall; `x = no`; larger `s`; larger `w_dense`, `w_bm25`."""
    w_dense, w_bm25, _w_hop = point["weights"]
    return (
        point["supported"],
        point["gold_recall_sum"],
        point["exclude"] is False,
        _s_rank(point["s"]),
        w_dense,
        w_bm25,
    )


def choose_point(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best: Mapping[str, Any] = max(curve, key=selection_key)
    return dict(best)


def dev_gate(chosen: Mapping[str, Any]) -> str | None:
    """D5: `DEV_STOP` below the bar (P10-C's 4,801 plus the margin, 4,835)."""
    return None if chosen["supported"] >= config.PHASE_12_DEV_BAR else DEV_STOP
