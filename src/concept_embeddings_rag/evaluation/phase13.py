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

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from concept_embeddings_rag import config
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
