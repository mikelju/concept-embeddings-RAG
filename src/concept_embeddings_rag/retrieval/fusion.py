"""Putting two rankings on one scale, in one place both hybrids read.

The conceptual score is a sum of products over `X`, the dense score is a cosine and
BM25 is neither. Nothing makes those three numbers comparable except a normalization
declared in advance, which is what decision D6 of the phase plan does. It lives here
rather than inside either hybrid so that System B and the HU-5 control are fused by
the same code, and a difference between them can only be a difference of signal.

Two schemes are measured. The weighted scheme buys one degree of freedom and has to
earn it against RRF, which has none: RRF reads ranks alone and cannot be moved by the
units a component happens to score in.

Three things D6 refused to leave implicit, and one it did not cover:

- A unit one component never returned is scored at `ABSENT_FLOOR` and stays in the
  union. "Ranked last by this component" and "never seen by this component" are
  different statements, and only the first is evidence.
- The floor is 0.0, which is also where min-max puts a component's own worst hit.
  That coincidence is declared rather than hidden: at the bottom of a returned list
  the two statements are worth the same, and the normalization says so.
- A component's internal ties are broken by unit id before its ranks are read, so an
  arbitrary array order inside one retriever cannot decide the fused ranking.
- The range D6 does not cover is the degenerate one. A component that scores every
  unit alike has expressed no preference, so min-max makes it abstain: every unit
  gets the floor. Putting them all at 1.0 instead would let the component's own
  tie-break choose the winner. RRF has no equivalent escape - it sees ranks and
  nothing else - so there the tie-break by unit id is the whole of the answer.

The primitives take two lists of `(unit_id, score)` and return one: no question
string, no corpus, no annotation, no split label. `FusedRetriever` is the thin
`Retriever` around them - it hands the query to its two components and fuses what
they hand back - and it reads nothing else either.

Phase 6 (decision D4 of its plan) adds one slot beside the second signals: a
**second stage**, which never reads the question. It is handed dense's own list for the
query and proposes candidates from it. The hybrid asks dense once and gives that list to
the stage, so the stage is conditioned on the very list that enters the fusion; for a
second signal the two calls are the ones Phase 3 made, in the same order, fused by the same
`fuse`. `SECOND_SIGNALS` is not extended: Phase 3's selection unpacks it into two names.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from concept_embeddings_rag import config
from concept_embeddings_rag.retrieval.base import Hit, Retriever

# What a unit absent from a component's list scores. Declared by D6, not inferred.
ABSENT_FLOOR: float = 0.0

# The two values the spec's `normalization` field can take.
MIN_MAX = "min_max"
NO_NORMALIZATION = "none"

# The same two names as `config.FUSION_SCHEMES`, which is what a sweep iterates over;
# these are what the code branches on. `fuse` checks membership against config, so a
# scheme that existed only here would be refused rather than silently accepted.
WEIGHTED = "weighted"
RRF = "rrf"

# Every hybrid of this phase is dense plus one second signal, in that order. D6 writes
# the weighted scheme as `w * dense_norm + (1 - w) * other_norm`, and the HU-5 control
# is the same object with `bm25` where System B has `conceptual`. Fixing the order here
# is what makes the two comparable: neither can be fused differently by accident.
DENSE = "dense"
SECOND_SIGNALS = ("conceptual", "bm25")
# Second-stage components (Phase 6, D4): conditioned on dense's list, not on the question.
SECOND_STAGES = (config.ENTITY_HOP_NAME,)

# The spec's `fitted_on`, carried on the object so a result says where its weights came
# from instead of leaving the reader to assume it.
FITTED_ON = "dev"


@runtime_checkable
class SecondStage(Protocol):
    """A component that proposes candidates from dense's list for a query, never the query."""

    name: str

    def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
        """Return up to `top_k` (unit_id, score) pairs, best first, derived from `first`."""
        ...


def union_of(components: Sequence[Sequence[Hit]]) -> list[str]:
    """Every unit any component returned, ordered so no tie depends on arrival order."""
    units: set[str] = set()
    for hits in components:
        units.update(unit_id for unit_id, _score in hits)
    return sorted(units)


def scored_once(hits: Sequence[Hit]) -> dict[str, float]:
    """A component's list as a mapping, refusing a unit it reported more than once.

    A duplicate would be counted twice by the weighted scheme and would occupy two
    ranks in RRF, so it is an error rather than something to deduplicate quietly.
    """
    scores: dict[str, float] = {}
    for unit_id, score in hits:
        if unit_id in scores:
            raise ValueError(f"a component returned {unit_id!r} twice in one query")
        scores[unit_id] = float(score)
    return scores


def min_max_normalize(hits: Sequence[Hit], units: Sequence[str]) -> dict[str, float]:
    """Map one component's scores onto [0, 1] over its own returned range.

    Computed per query: the range is the one this component produced for this
    question, never a corpus-wide range, so two questions whose raw scores differ by
    orders of magnitude still contribute on the same scale.

    Every unit of `units` comes back keyed. One this component did not return gets
    `ABSENT_FLOOR`; a component that returned nothing, or that scored everything
    alike, abstains and gets the floor throughout.
    """
    scores = scored_once(hits)
    normalized = dict.fromkeys(units, ABSENT_FLOOR)

    if not scores:
        return normalized

    lo = min(scores.values())
    hi = max(scores.values())
    if hi <= lo:
        return normalized

    span = hi - lo
    for unit_id in normalized:
        if unit_id in scores:
            normalized[unit_id] = (scores[unit_id] - lo) / span
    return normalized


def ranks_of(hits: Sequence[Hit]) -> list[str]:
    """The component's units in rank order, with its own ties broken by unit id."""
    scores = scored_once(hits)
    return sorted(scores, key=lambda unit_id: (-scores[unit_id], unit_id))


def rrf_scores(components: Sequence[Sequence[Hit]]) -> dict[str, float]:
    """`s = sum(1 / (RRF_K + rank))` over the components that returned the unit.

    The parameter-free reference. It reads ranks and nothing else, so no rescaling of
    a component's raw scores can move it. A component that did not return the unit
    contributes no term, which is not the same as contributing a zero.
    """
    fused = dict.fromkeys(union_of(components), 0.0)
    for hits in components:
        for rank, unit_id in enumerate(ranks_of(hits), start=1):
            fused[unit_id] += 1.0 / (config.RRF_K + rank)
    return fused


def _check_convex(weights: Sequence[float], n_components: int) -> None:
    """Refuse anything that is not a convex combination over `n_components` components.

    Shared by `weighted_scores` and by `FusedRetriever`, so a weight the fusion would
    refuse mid-sweep is refused when the hybrid is built instead.
    """
    if len(weights) != n_components:
        raise ValueError(
            f"one weight per component: {len(weights)} weights for {n_components} components"
        )
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"a negative weight is not a convex combination: {tuple(weights)}")
    total = sum(weights)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"the weights must sum to 1, not {total}")


def weighted_scores(
    components: Sequence[Sequence[Hit]], weights: Sequence[float]
) -> dict[str, float]:
    """`s = w * first_norm + (1 - w) * second_norm` over min-max normalized scores.

    D6 fixes this as a convex combination, so the weights are refused unless they are
    non-negative and sum to one: an endpoint of `config.FUSION_WEIGHT_GRID` has to be
    one component on its own for the sweep's curve to be readable at all.
    """
    if len(components) < 2:
        raise ValueError(f"a fusion needs two components, not {len(components)}")
    _check_convex(weights, len(components))

    units = union_of(components)
    fused = dict.fromkeys(units, 0.0)
    for hits, weight in zip(components, weights, strict=True):
        for unit_id, value in min_max_normalize(hits, units).items():
            fused[unit_id] += weight * value
    return fused


def normalization_for(scheme: str) -> str:
    """What the spec's `normalization` field records for a scheme. Rank-based is `none`.

    The single place a scheme name is checked against `config.FUSION_SCHEMES`, so a
    scheme nothing declared is refused here rather than defaulted somewhere later.
    """
    if scheme not in config.FUSION_SCHEMES:
        raise ValueError(
            f"unknown fusion scheme {scheme!r}, expected one of {config.FUSION_SCHEMES}"
        )
    return NO_NORMALIZATION if scheme == RRF else MIN_MAX


def fuse(
    components: Sequence[Sequence[Hit]],
    *,
    scheme: str,
    top_k: int,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    """The best `top_k` of the union, deterministic including ties.

    Ties are broken by unit id, the same criterion `dense.py` and `bm25.py` use: a
    ranking that drifts between runs for reasons unrelated to relevance is not a
    measurement.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, not {top_k}")
    # Refuses an unknown scheme, and says which normalization this fusion declares.
    normalization_for(scheme)

    if scheme == RRF:
        if weights is not None:
            raise ValueError("rrf takes no weight; it is the parameter-free reference")
        fused = rrf_scores(components)
    else:
        if weights is None:
            raise ValueError("the weighted scheme cannot run without weights")
        fused = weighted_scores(components, weights)

    ordered = sorted(fused.items(), key=lambda entry: (-entry[1], entry[0]))
    return ordered[:top_k]


def _is_retriever(component: object) -> bool:
    return callable(getattr(component, "retrieve", None))


def _is_stage(component: object) -> bool:
    return callable(getattr(component, "propose", None))


def _in_fixed_order(
    components: Sequence[Retriever | SecondStage],
) -> tuple[Retriever, Retriever | SecondStage, bool]:
    """The two components as `(dense, second, staged)`, whichever order they arrived in.

    The order is not the caller's to choose. System B and the HU-5 control differ only
    in their second signal, so if the order could vary between them, a difference in
    their numbers could be a difference of arrangement rather than of signal.

    The second component is a signal when its name is declared in `SECOND_SIGNALS` and it
    retrieves, or a stage when its name is declared in `SECOND_STAGES` and it proposes
    (Phase 6, D4). An object that is both, or neither, or whose capability does not match
    its name, is refused: a stage filed under a signal's name would be fused as the control.
    """
    if len(components) != 2:
        raise ValueError(f"a hybrid fuses two components, not {len(components)}")

    names = [component.name for component in components]
    if names[0] == names[1]:
        raise ValueError(f"the two components must be different retrievers, not {names[0]!r} twice")
    if DENSE not in names:
        raise ValueError(f"every hybrid of this phase is built on {DENSE!r}, not on {names}")

    dense, other = components if names[0] == DENSE else (components[1], components[0])
    if not isinstance(dense, Retriever) or _is_stage(dense):
        raise ValueError(f"the {DENSE!r} component must be a retriever")

    signal = other.name in SECOND_SIGNALS and _is_retriever(other) and not _is_stage(other)
    stage = other.name in SECOND_STAGES and _is_stage(other) and not _is_retriever(other)
    if not (signal or stage):
        raise ValueError(
            f"unknown second signal {other.name!r}, expected a retriever named one of "
            f"{SECOND_SIGNALS} or a second stage named one of {SECOND_STAGES}"
        )
    return dense, other, stage


def _checked_weights(
    scheme: str, components: Sequence[str], weights: Mapping[str, float] | None
) -> dict[str, float] | None:
    """The spec's `weights` field: component name -> weight, or `None` for RRF.

    Keyed by name rather than by position, so a weight cannot land on the component it
    was not meant for, and re-keyed into the fixed order before it reaches `fuse`.
    """
    if scheme == RRF:
        if weights is not None:
            raise ValueError("rrf takes no weight; it is the parameter-free reference")
        return None

    if weights is None:
        raise ValueError("the weighted scheme cannot run without weights")
    if sorted(weights) != sorted(components):
        raise ValueError(
            "the weights must name every component exactly once: "
            f"{sorted(weights)} against {sorted(components)}"
        )

    ordered = {name: float(weights[name]) for name in components}
    _check_convex(tuple(ordered.values()), len(components))
    return ordered


class FusedRetriever:
    """Two retrievers behind one, fused by the scheme D6 declared.

    The spec's contract, and the reason each field is derived rather than passed:

    - `name` is `hybrid-` plus the second component's name, so `hybrid-conceptual` and
      `hybrid-bm25` are what they say they are. A name the caller chose could label a
      result as System B when the control is what ran.
    - `components` is the fixed order, `dense` first.
    - `normalization` follows from the scheme, and is `none` for the rank-based one.
    - `weights` are keyed by component name and re-keyed into the fixed order here.
    - `fitted_on` is `dev`, always, because that is the only split this phase fits on.

    Everything it can refuse, it refuses at construction: a hybrid that only fails on
    its first query would fail in the middle of a sweep, after hours of measurement.

    Its dense component is the Phase 1 retriever, unmodified and reading the unchanged
    embedding cache. Nothing here re-embeds or re-ranks anything: it asks two objects
    for a ranking and combines the two lists.
    """

    def __init__(
        self,
        components: Sequence[Retriever | SecondStage],
        *,
        scheme: str,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self._dense, self._second, self._staged = _in_fixed_order(components)
        self.components = [self._dense.name, self._second.name]
        self.name = f"hybrid-{self.components[1]}"
        self.scheme = scheme
        self.normalization = normalization_for(scheme)
        self.weights = _checked_weights(scheme, self.components, weights)
        self.fitted_on = FITTED_ON

    def _as_retriever(self) -> Retriever:
        if not isinstance(self._second, Retriever):
            raise ValueError(f"{self._second.name!r} was admitted as a stage, not a retriever")
        return self._second

    def describe(self) -> dict:
        """The spec's data contract, to be recorded beside every metric."""
        return {
            "name": self.name,
            "components": list(self.components),
            "scheme": self.scheme,
            "normalization": self.normalization,
            "weights": None if self.weights is None else dict(self.weights),
            "fitted_on": self.fitted_on,
        }

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        """The best `top_k` of the union of both components' `top_k` lists.

        Each component is asked for the same `top_k` the hybrid was asked for, which at
        the phase's `top_k = 100` is D6's arithmetic exactly: two lists of 100, a union
        of at most 200, and the best 100 of it returned.

        Dense is asked once. A second signal is then asked the query, exactly as Phase 3
        did; a second stage is handed a copy of dense's list instead, so it is conditioned
        on the list that enters the fusion and cannot alter it.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, not {top_k}")

        first = self._dense.retrieve(query, top_k)
        second = (
            self._second.propose(list(first), top_k)
            if isinstance(self._second, SecondStage) and self._staged
            else self._as_retriever().retrieve(query, top_k)
        )
        hits = [first, second]
        ordered = (
            None if self.weights is None else tuple(self.weights[name] for name in self.components)
        )
        return fuse(hits, scheme=self.scheme, top_k=top_k, weights=ordered)


# --- Phase 10: three components -----------------------------------------------------------

BM25_NAME = "bm25"
TRIPLE_COMPONENTS: tuple[str, ...] = (DENSE, BM25_NAME, config.ENTITY_HOP_NAME)


def fuse_lists(
    lists: Sequence[Sequence[Hit]], weights: Sequence[float], *, top_k: int
) -> list[Hit]:
    """The weighted, min-max fusion of already-gathered component lists, in fixed order.

    Exactly the call a hybrid's `retrieve` makes once it holds its lists, so a dev grid
    that re-fuses cached lists ranks what the hybrid would have ranked.
    """
    return fuse(lists, scheme=WEIGHTED, top_k=top_k, weights=tuple(weights))


class TripleFusedRetriever:
    """Dense + BM25 + Entity Hop (Phase 10, P10-C), fused by the weighted scheme.

    Dense is asked once; BM25 reads the question; the Entity Hop stage is handed Dense's
    list, as in Phase 6. Each is asked for the same `top_k`, and the best `top_k` of the
    union is returned by the existing `fuse`. The components are fixed in role and order:
    a system whose second or third slot held something else would not be P10-C.
    """

    scheme = WEIGHTED
    normalization = MIN_MAX
    fitted_on = FITTED_ON

    def __init__(
        self,
        dense: Retriever,
        bm25: Retriever,
        stage: SecondStage,
        *,
        weights: Mapping[str, float],
    ) -> None:
        roles = ((dense, DENSE, "retrieve"), (bm25, BM25_NAME, "retrieve"))
        for component, name, method in roles:
            if getattr(component, "name", None) != name or not callable(
                getattr(component, method, None)
            ):
                raise ValueError(f"the {name!r} slot must hold a retriever named {name!r}")
        if getattr(stage, "name", None) != config.ENTITY_HOP_NAME or not _is_stage(stage):
            raise ValueError(f"the third slot must hold the {config.ENTITY_HOP_NAME!r} stage")
        self._dense, self._bm25, self._stage = dense, bm25, stage
        self.components = list(TRIPLE_COMPONENTS)
        self.name = f"hybrid-{BM25_NAME}-{config.ENTITY_HOP_NAME}"
        checked = _checked_weights(WEIGHTED, self.components, weights)
        if checked is None:  # unreachable for the weighted scheme; keeps the type narrow
            raise ValueError("the weighted scheme cannot run without weights")
        self.weights = checked

    def describe(self) -> dict:
        return {
            "name": self.name,
            "components": list(self.components),
            "scheme": self.scheme,
            "normalization": self.normalization,
            "weights": dict(self.weights),
            "fitted_on": self.fitted_on,
        }

    def gather(self, query: str, top_k: int) -> tuple[list[Hit], list[Hit], list[Hit]]:
        """The three component lists this hybrid fuses, in fixed order."""
        first = self._dense.retrieve(query, top_k)
        return first, self._bm25.retrieve(query, top_k), self._stage.propose(list(first), top_k)

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, not {top_k}")
        ordered = tuple(self.weights[name] for name in self.components)
        return fuse_lists(self.gather(query, top_k), ordered, top_k=top_k)
