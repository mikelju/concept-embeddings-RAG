"""Phase 12: choosing with the question which of P1's entities the hop starts from.

`12.spec.md` (approved 2026-09-27) fixes every rule here before any Phase 12 number exists.
The Phase 9 hop is unchanged; what is new is *which* of P1's entity nodes seed it. D1 locates
each of P1's entity nodes in P1's own sentences by text, scores those sentences by cosine
similarity to the question (both encoded with the pinned Phase 9 Dense model), and D2 keeps
only the entities located in the top-ranked sentences, optionally excluding the ones the
question already names. This module holds those rules as pure functions over plain data; the
sentence cache (S2), the D4 reproduction and the D3/D5 grid, tie rule and gate are added by
later steps. The runtime stage and retriever live in `retrieval/entity_hop.py` and
`retrieval/fusion.py`.
"""

import re
from collections.abc import Collection, Mapping, Sequence

from concept_embeddings_rag import config
from concept_embeddings_rag.nodes.normalization import normalize


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


def named_in_question(
    p1_nodes: Sequence[int], forms: Mapping[int, str], question: str
) -> set[int]:
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
