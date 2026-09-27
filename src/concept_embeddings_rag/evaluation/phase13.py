"""Phase 13: choosing P1's bridge entity by the words around its mention.

`13.spec.md` (approved 2026-09-27) fixes every rule here before any Phase 13 number exists.
The Phase 9 hop is unchanged in how it scores; what is new is a per-entity score (D1): each
of P1's entity nodes is located in P1's own normalized sentences, every match is a mention,
and the up-to-5 words each side of a mention - clipped at the sentence edges, the mention's
own words masked - are encoded with the pinned Phase 9 Dense model. An entity scores the
maximum cosine between the question and its windows. D2 seeds the hop with the `m` best-scored
entities; D3 screens the signal once against two trivial rules before any list or fit.

This module holds those rules as pure functions over plain data, plus the S2 window cache and
the screen tally. The runtime stage lives in `retrieval/entity_hop.py`.
"""

import hashlib
import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, savez_compressed_atomic, write_text_atomic
from concept_embeddings_rag.embeddings.cache import unit_set_hash
from concept_embeddings_rag.evaluation import phase12
from concept_embeddings_rag.nodes.normalization import normalize

SCREEN_STOP = "SCREEN_STOP"
DEV_STOP = "DEV_STOP"


class Phase13Error(Exception):
    """A Phase 13 input is missing, modified, or not the one this run may use."""


# --- S1 (D1): mentions and windows ----------------------------------------------------------


class Mention(NamedTuple):
    """One match of a node form in a normalized sentence.

    `offset` is the match's character offset in the normalized sentence; `first_word` and
    `last_word` are the indexes of the whitespace words the match overlaps (inclusive).
    """

    offset: int
    first_word: int
    last_word: int


_WORD = re.compile(r"\S+")


def _word_spans(normalized: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _WORD.finditer(normalized)]


def _mentions_in(needle: str, normalized: str, spans: Sequence[tuple[int, int]]) -> list[Mention]:
    if not needle:
        return []
    pattern = re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)")
    found: list[Mention] = []
    for match in pattern.finditer(normalized):
        start, end = match.start(), match.end()
        overlapped = [i for i, (ws, we) in enumerate(spans) if ws < end and start < we]
        found.append(Mention(start, overlapped[0], overlapped[-1]))
    return found


def mentions(form: str, sentence: str) -> list[Mention]:
    """Every match of the normalized `form` in the normalized `sentence` (D1).

    The pattern is `phase12.occurs`'s - bounded by non-word characters, the form escaped -
    applied with `finditer`, so every non-overlapping match is a mention. Both sides are
    normalized here; normalization is idempotent.
    """
    normalized = normalize(sentence)
    return _mentions_in(normalize(form), normalized, _word_spans(normalized))


def _window_of(words: Sequence[str], mention: Mention, width: int) -> str:
    left = words[max(0, mention.first_word - width) : mention.first_word]
    right = words[mention.last_word + 1 : mention.last_word + 1 + width]
    return " ".join([*left, *right])


def window(sentence: str, mention: Mention, width: int = config.PHASE_13_WINDOW_WORDS) -> str:
    """Up to `width` words before and after `mention`, its own words left out (D1).

    Cut from the normalized sentence, clipped at its edges, left context first, joined by one
    space. `""` when the mention is the whole sentence.
    """
    return _window_of(normalize(sentence).split(), mention, width)


@dataclass(frozen=True)
class Window:
    """One non-empty window: where its mention is, and the window text."""

    position: int
    first_word: int
    text: str


@dataclass(frozen=True)
class EntityMentions:
    """Every mention of one P1 entity node, and its non-empty windows.

    `spans` holds `(sentence position, match offset)` for every mention, empty-window ones
    included, in sentence then offset order; `windows` only the non-empty windows, in the
    same order; `empty` counts the mentions whose window is empty.
    """

    node_id: int
    spans: tuple[tuple[int, int], ...]
    windows: tuple[Window, ...]
    empty: int

    @property
    def first_mention(self) -> tuple[int, int]:
        return self.spans[0]

    @property
    def scored(self) -> bool:
        return bool(self.windows)


def entity_windows(
    p1_nodes: Sequence[int],
    forms: Mapping[int, str],
    sentences: Sequence[str],
    width: int = config.PHASE_13_WINDOW_WORDS,
) -> dict[int, EntityMentions]:
    """Node id -> its mentions and windows in `sentences`, for every node located at least once.

    A node with an empty form, or mentioned in no sentence, is omitted. A node whose every
    window is empty is kept, with no windows: it is located but unscored (D1).
    """
    prepared = []
    for sentence in sentences:
        normalized = normalize(sentence)
        prepared.append((normalized, normalized.split(), _word_spans(normalized)))
    found: dict[int, EntityMentions] = {}
    for node_id in p1_nodes:
        needle = normalize(forms.get(int(node_id), ""))
        if not needle:
            continue
        spans: list[tuple[int, int]] = []
        windows: list[Window] = []
        empty = 0
        for position, (normalized, words, word_spans) in enumerate(prepared):
            for mention in _mentions_in(needle, normalized, word_spans):
                spans.append((position, mention.offset))
                text = _window_of(words, mention, width)
                if text:
                    windows.append(Window(position, mention.first_word, text))
                else:
                    empty += 1
        if spans:
            found[int(node_id)] = EntityMentions(int(node_id), tuple(spans), tuple(windows), empty)
    return found


# --- S1 (D1, D2): scores, the ranking and the seeds -----------------------------------------


def entity_scores(
    node_ids: Sequence[int], vectors: np.ndarray, question_vector: np.ndarray
) -> dict[int, float]:
    """Node id -> the maximum cosine between the question and its windows (D1).

    `vectors` holds one L2-normalized row per window, `node_ids` the node of each row; the
    dot product is the cosine.
    """
    if not len(node_ids):
        return {}
    sims = np.asarray(vectors @ question_vector).ravel()
    scores: dict[int, float] = {}
    for node, sim in zip(node_ids, sims.tolist(), strict=True):
        node = int(node)
        if node not in scores or sim > scores[node]:
            scores[node] = float(sim)
    return scores


def _weight(weights: Mapping[int, float] | np.ndarray, node: int) -> float:
    return float(weights[node])


def rank_entities(
    scores: Mapping[int, float], weights: Mapping[int, float] | np.ndarray
) -> list[int]:
    """P1's scored entities: score descending, rarity weight descending, node id ascending."""
    return sorted(scores, key=lambda node: (-scores[node], -_weight(weights, node), node))


def seeds(ranking: Sequence[int], p1_nodes: Sequence[int], m: int | None) -> list[int]:
    """D2's seed set: the first `m` of the ranking, or every P1 entity node at `m = None`.

    For `m` an integer the seeds come in ranking order, fewer when fewer are scored, none
    when none is; at `m = None` they are every P1 node, ascending, unscored ones included.
    """
    if m is None:
        return sorted({int(node) for node in p1_nodes})
    return [int(node) for node in ranking[:m]]


def seed_key(m: int | None) -> str:
    """The dev-list / grid key for one seed count, e.g. `window-hop@m=all`."""
    return f"window-hop@m={'all' if m is None else m}"


# --- S1 (D3): the screen rules ---------------------------------------------------------------


def first_mention(scored: Collection[int], first: Mapping[int, tuple[int, int]]) -> int | None:
    """T1: the scored entity mentioned first - sentence position, match offset, node id."""
    if not scored:
        return None
    return min(scored, key=lambda node: (first[node][0], first[node][1], node))


def rarest(scored: Collection[int], weights: Mapping[int, float] | np.ndarray) -> int | None:
    """T2: the scored entity with the highest rarity weight, then the lowest node id."""
    if not scored:
        return None
    return min(scored, key=lambda node: (-_weight(weights, node), node))


def hit(choice: int | None, bridge: Collection[int]) -> bool:
    """A rule hits an item when the entity it puts first is one of the item's bridges."""
    return choice is not None and choice in bridge


def screen_verdict(n: int, hits_w: int, hits_t1: int, hits_t2: int) -> dict[str, Any]:
    """D3: W must clear the better trivial rule by 5.0 percentage points.

    The comparison is exact on counts: `100 * (W - ref) / n >= 5.0` is `20 * (W - ref) >= n`
    when the margin is 5.0, and in general `100 * (W - ref) >= margin * n`.
    """
    margin = config.PHASE_13_SCREEN_MARGIN_PP
    reference_rule, reference_hits = ("T1", hits_t1) if hits_t1 >= hits_t2 else ("T2", hits_t2)
    passed = n > 0 and 100 * (hits_w - reference_hits) >= margin * n

    def pp(hits: int) -> float:
        return 100.0 * hits / n if n else 0.0

    return {
        "population": n,
        "hits": {"W": hits_w, "T1": hits_t1, "T2": hits_t2},
        "rates_pp": {"W": pp(hits_w), "T1": pp(hits_t1), "T2": pp(hits_t2)},
        "reference_rule": reference_rule,
        "reference_pp": pp(reference_hits),
        "margin_pp": margin,
        "difference_pp": pp(hits_w) - pp(reference_hits),
        "passed": bool(passed),
        "terminal_state": None if passed else SCREEN_STOP,
    }


# --- S3 (D3): the screen population, the choices per question and the tally -----------------


def screen_population(
    *,
    dense_top10: Collection[str],
    gold: Sequence[str],
    p1_nodes: Collection[int],
    nodes_of: Callable[[str], Collection[int] | None],
) -> list[tuple[str, frozenset[int]]]:
    """One question's screen items: `(gold unit, bridge set)` (D3).

    A gold paragraph is an item when it is outside the Dense top 10, in the entity index
    (`nodes_of` returns its entity nodes, or `None` for a unit the index does not hold), and
    shares at least one entity node with P1. Its bridge set is that intersection. Each gold
    paragraph is its own item, so a question can contribute two. The definition is the
    Phase 12 closing diagnostic's.
    """
    top = set(dense_top10)
    p1 = {int(node) for node in p1_nodes}
    items: list[tuple[str, frozenset[int]]] = []
    for unit_id in gold:
        if unit_id in top:
            continue
        nodes = nodes_of(unit_id)
        if nodes is None:
            continue
        bridge = frozenset(p1 & {int(node) for node in nodes})
        if bridge:
            items.append((unit_id, bridge))
    return items


@dataclass(frozen=True)
class QuestionChoices:
    """What each rule puts first for one question, all among P1's scored entities.

    `ranking` is W's full D1 ranking; `sentence_rule` is the descriptive per-entity restatement
    of Phase 12's rule.
    """

    ranking: list[int]
    t1: int | None
    t2: int | None
    sentence_rule: int | None

    @property
    def w(self) -> int | None:
        return self.ranking[0] if self.ranking else None


def sentence_rule(
    entities: Mapping[int, EntityMentions], scored: Collection[int], rel: Sequence[float]
) -> int | None:
    """Descriptive: the first-mentioned scored entity of the P1 sentence most similar to q.

    Only sentences holding a mention of a scored entity compete, ranked by similarity
    descending, then position ascending (Phase 12's sentence order); within the first, the
    scored entity with the earliest match offset wins, then the lowest node id.
    """
    holding: dict[int, list[tuple[int, int]]] = {}
    for node in scored:
        for position, offset in entities[node].spans:
            holding.setdefault(position, []).append((offset, node))
    if not holding:
        return None
    top = min(holding, key=lambda position: (-rel[position], position))
    return min(holding[top])[1]


def question_choices(
    entities: Mapping[int, EntityMentions],
    scores: Mapping[int, float],
    weights: Mapping[int, float] | np.ndarray,
    rel: Sequence[float],
) -> QuestionChoices:
    """W, T1, T2 and the descriptive sentence rule for one question (D3)."""
    scored = list(scores)
    first = {node: entities[node].first_mention for node in scored}
    return QuestionChoices(
        ranking=rank_entities(scores, weights),
        t1=first_mention(scored, first),
        t2=rarest(scored, weights),
        sentence_rule=sentence_rule(entities, scored, rel),
    )


def screen_tally(
    items: Sequence[tuple[frozenset[int], QuestionChoices]],
    expected_population: int = config.PHASE_13_SCREEN_POPULATION,
) -> dict[str, Any]:
    """D3's hit counts, rates and verdict, plus the descriptive figures.

    Refuses, before counting any hit, a population other than the expected one.
    """
    n = len(items)
    if n != expected_population:
        raise Phase13Error(
            f"the screen population is {n}, not the {expected_population} the spec fixes"
        )
    hits_w = sum(hit(choices.w, bridge) for bridge, choices in items)
    hits_t1 = sum(hit(choices.t1, bridge) for bridge, choices in items)
    hits_t2 = sum(hit(choices.t2, bridge) for bridge, choices in items)
    body = screen_verdict(n, hits_w, hits_t1, hits_t2)

    def described(count: int) -> dict[str, Any]:
        return {"hits": count, "rate_pp": 100.0 * count / n if n else 0.0}

    body["descriptive"] = {
        "sentence_rule": described(
            sum(hit(choices.sentence_rule, bridge) for bridge, choices in items)
        ),
        "w_first_2": described(
            sum(bool(bridge & set(choices.ranking[:2])) for bridge, choices in items)
        ),
        "w_first_3": described(
            sum(bool(bridge & set(choices.ranking[:3])) for bridge, choices in items)
        ),
        "items_without_scored_entity": sum(not choices.ranking for _bridge, choices in items),
    }
    return body


# --- S2: the window cache, one NPZ plus manifest per set, written once -----------------------


def window_cache_key(model: str, revision: str, set_name: str, unit_ids: Sequence[str]) -> str:
    """A hash of the model, revision, window rule, set name and the sorted P1 unit ids."""
    payload = (
        f"window|{model}|{revision}|{config.PHASE_13_WINDOW_RULE}|{set_name}|"
        f"{unit_set_hash(unit_ids)}"
    )
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def window_cache_filename(set_name: str, key: str) -> str:
    return f"windows-{set_name}-{key}.npz"


def window_manifest_filename(set_name: str) -> str:
    return f"windows-{set_name}.json"


def vectors_digest(vectors: np.ndarray) -> str:
    return digest_of(vectors.astype(np.float32))


def rows_digest(
    unit_ids: Sequence[str],
    node_ids: Sequence[int],
    positions: Sequence[int],
    first_words: Sequence[int],
) -> str:
    """The digest of the cache's row metadata, as the NPZ stores it."""
    return digest_of(
        np.array(list(unit_ids), dtype=np.str_),
        np.array(list(node_ids), dtype=np.int64),
        np.array(list(positions), dtype=np.int64),
        np.array(list(first_words), dtype=np.int64),
    )


def write_window_cache(
    directory: Path | str,
    set_name: str,
    key: str,
    unit_ids: Sequence[str],
    node_ids: Sequence[int],
    positions: Sequence[int],
    first_words: Sequence[int],
    vectors: np.ndarray,
) -> Path:
    """The P1 window vectors of one set, one row per window, written once."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / window_cache_filename(set_name, key)
    if path.exists():
        raise Phase13Error(f"{path} already holds the {set_name} window cache")
    lengths = {len(unit_ids), len(node_ids), len(positions), len(first_words), vectors.shape[0]}
    if len(lengths) != 1:
        raise Phase13Error("every window column must have one row per window")
    savez_compressed_atomic(
        path,
        unit_ids=np.array(list(unit_ids), dtype=np.str_),
        node_ids=np.array(list(node_ids), dtype=np.int64),
        positions=np.array(list(positions), dtype=np.int64),
        first_words=np.array(list(first_words), dtype=np.int64),
        vectors=vectors.astype(np.float32),
    )
    return path


def write_window_manifest(directory: Path | str, set_name: str, body: Mapping[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / window_manifest_filename(set_name)
    if path.exists():
        raise Phase13Error(f"{path} already records the {set_name} window cache")
    write_text_atomic(path, json.dumps(dict(body), indent=2, sort_keys=True))
    return path


@dataclass(frozen=True, eq=False)
class WindowCache:
    """The window cache of one set, digest-verified, with its rows grouped by P1 unit."""

    unit_ids: list[str]
    node_ids: list[int]
    positions: list[int]
    first_words: list[int]
    vectors: np.ndarray
    manifest: dict[str, Any]
    _rows: dict[str, list[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rows: dict[str, list[int]] = {}
        for row, unit_id in enumerate(self.unit_ids):
            rows.setdefault(unit_id, []).append(row)
        object.__setattr__(self, "_rows", rows)

    def holds(self, unit_id: str) -> bool:
        return unit_id in self._rows

    def rows_of(self, unit_id: str) -> tuple[list[int], np.ndarray]:
        """The node id and vector of every window of `unit_id`, in cache row order."""
        if unit_id not in self._rows:
            raise Phase13Error(f"unit {unit_id!r} is not in the window cache")
        selected = self._rows[unit_id]
        return [self.node_ids[row] for row in selected], self.vectors[selected]


def read_window_cache(directory: Path | str, set_name: str) -> WindowCache:
    """The cache and its manifest for `set_name`, both digests verified."""
    directory = Path(directory)
    manifest_path = directory / window_manifest_filename(set_name)
    if not manifest_path.exists():
        raise Phase13Error(f"{manifest_path} does not exist: run 'p13-windows' first")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = directory / window_cache_filename(set_name, str(manifest["key"]))
    if not path.exists():
        raise Phase13Error(f"{path} does not exist: run 'p13-windows' first")
    with np.load(path, allow_pickle=False) as payload:
        unit_ids = [str(u) for u in payload["unit_ids"]]
        node_ids = [int(n) for n in payload["node_ids"]]
        positions = [int(p) for p in payload["positions"]]
        first_words = [int(w) for w in payload["first_words"]]
        vectors = payload["vectors"]
    if vectors_digest(vectors) != manifest["vectors_digest"]:
        raise Phase13Error(f"the {set_name} window vectors do not match its recorded digest")
    if rows_digest(unit_ids, node_ids, positions, first_words) != manifest["rows_digest"]:
        raise Phase13Error(f"the {set_name} window rows do not match its recorded digest")
    return WindowCache(unit_ids, node_ids, positions, first_words, vectors, manifest)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def window_coverage(
    question_p1: Sequence[str],
    p1_entity_nodes: Mapping[str, int],
    unit_entities: Mapping[str, Mapping[int, EntityMentions]],
) -> dict[str, Any]:
    """C2: how much of P1 the window score reaches, per question and per distinct P1.

    `question_p1` is each dev question's P1 unit, in question order; `p1_entity_nodes` the
    number of entity nodes of each P1; `unit_entities` the output of `entity_windows` per P1.
    Shares of entity nodes are summed over questions; mention and window counts are over the
    distinct P1 units.
    """
    total = located = scored = without = 0
    for unit_id in question_p1:
        entities = unit_entities[unit_id]
        n_scored = sum(1 for e in entities.values() if e.scored)
        total += p1_entity_nodes[unit_id]
        located += len(entities)
        scored += n_scored
        without += n_scored == 0
    per_entity = [len(e.spans) for ents in unit_entities.values() for e in ents.values()]
    windows = sum(len(e.windows) for ents in unit_entities.values() for e in ents.values())
    empty = sum(e.empty for ents in unit_entities.values() for e in ents.values())
    n_questions = len(question_p1)
    return {
        "questions": n_questions,
        "p1_entity_nodes_total": total,
        "p1_entity_nodes_located": located,
        "p1_entity_nodes_scored": scored,
        "located_share": located / total if total else 0.0,
        "scored_share": scored / total if total else 0.0,
        "questions_without_scored_entity": without,
        "questions_without_scored_entity_share": without / n_questions if n_questions else 0.0,
        "units": len(unit_entities),
        "mentions": sum(per_entity),
        "windows": windows,
        "empty_windows": empty,
        "mentions_per_located_entity": _distribution([float(v) for v in per_entity]),
    }


def window_rows(
    p1_ids: Sequence[str], unit_entities: Mapping[str, Mapping[int, EntityMentions]]
) -> tuple[list[str], list[int], list[int], list[int], list[str]]:
    """The cache's row order: unit id ascending as given, node id ascending, window order.

    Returns the unit id, node id, sentence position, first word and text columns. This is the
    order `cer p13-windows` writes; the screen rebuilds it and compares it row for row.
    """
    unit_ids: list[str] = []
    node_ids: list[int] = []
    positions: list[int] = []
    first_words: list[int] = []
    texts: list[str] = []
    for unit_id in p1_ids:
        entities = unit_entities[unit_id]
        for node_id in sorted(entities):
            for item in entities[node_id].windows:
                unit_ids.append(unit_id)
                node_ids.append(node_id)
                positions.append(item.position)
                first_words.append(item.first_word)
                texts.append(item.text)
    return unit_ids, node_ids, positions, first_words, texts


# --- S5 (D4): the reproduction of P10-C on dev -----------------------------------------------


def reproduction_verdict(
    observed: int, differing_qids: Sequence[str], lists_differing_qids: Sequence[str]
) -> dict[str, Any]:
    """D4: exactly P10-C's dev count, no question moved, no `m = all` hop list moved."""
    recorded = config.PHASE_13_P10C_DEV_SUPPORTED
    weights = dict(zip(config.PHASE_13_COMPONENT_NAMES, config.PHASE_13_P10C_WEIGHTS, strict=True))
    return {
        "metric": "full_support@2048_tokens over the 7,405 dev questions",
        "point": {"m": config.PHASE_13_P10C_M, "weights": weights},
        "observed": int(observed),
        "recorded": recorded,
        "differing_qids": sorted(differing_qids),
        "hop_list_differing_qids": sorted(lists_differing_qids),
        "passed": observed == recorded and not differing_qids and not lists_differing_qids,
    }


def list_summary(values: Sequence[int]) -> dict[str, float]:
    """Median, p90, max and the zero share of a per-question count (seeds, candidates)."""
    if not values:
        return {"median": 0.0, "p90": 0.0, "max": 0.0, "zero_share": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "zero_share": float((array == 0).mean()),
    }


# --- S6 (D5, D6): the grid, the tie rule and the dev gate ------------------------------------


def grid_points(
    seed_counts: Sequence[int | None] = config.PHASE_13_SEEDS,
) -> list[tuple[int | None, tuple[float, float, float]]]:
    """The 4 values of `m` x the 66 convex weight triples: 264 points (D5)."""
    triples = phase12.weight_grid(config.PHASE_13_GRID_TENTHS)
    return [(m, weights) for m in seed_counts for weights in triples]


def _m_rank(m: int | None) -> float:
    return float("inf") if m is None else float(m)


def selection_key(point: Mapping[str, Any]) -> tuple[Any, ...]:
    """D5, in order: Full Support; gold recall; larger `m` (all largest); larger w_dense, w_bm25."""
    w_dense, w_bm25, _w_hop = point["weights"]
    return (point["supported"], point["gold_recall_sum"], _m_rank(point["m"]), w_dense, w_bm25)


def choose_point(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best: Mapping[str, Any] = max(curve, key=selection_key)
    return dict(best)


def dev_gate(chosen: Mapping[str, Any]) -> str | None:
    """D6: `DEV_STOP` below the bar (P10-C's 4,801 plus 33, 4,834)."""
    return None if chosen["supported"] >= config.PHASE_13_DEV_BAR else DEV_STOP
