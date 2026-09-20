"""Deviation 8.1, S2: mapping the frozen C19 corpus onto the official FullWiki source.

This module answers one question - *which official paragraph corresponds to each frozen
C19 unit?* - and it answers it only to prove correspondence, to keep paragraphs already
represented by C19 out of the added-distractor pool, to verify that the benchmark
evidence exists in the official source, and to record provenance.

**C19 is never reconstructed.** A match, exact or normalized, does not replace a C19
title, sentence sequence, unit id or `indexable_text`. Retrieval and evaluation keep
using the existing frozen 19,366 units, which is what makes the later C19 reproduction
gate unambiguous: a mismatch there is an evaluation or alignment defect, never an allowed
side effect of reconciliation.

Why the gate is strict, recorded here so nobody relaxes it without knowing what it
protected: evaluation matches retrieved units against `Question.gold_unit_ids` **by unit
id**, and `unit_id_for` is a content hash. A byte-identical official duplicate of a C19
paragraph therefore collapses onto the same id and is harmless. The danger is the
**near**-duplicate - its bytes differ, it hashes to a different id, and it enters the
distractor pool as a separate unit. If such a twin is a gold paragraph, a model can
retrieve it, rank it first and receive zero credit; the penalty grows with corpus size,
applies to both encoders, and is indistinguishable in the results from genuine
degradation, so it could manufacture `CONVERGENCE` or `BOTH_DEGRADE` out of nothing.

Hence the remedy for a residue of unresolved units is **conservative title exclusion**:
every official paragraph sharing an unresolved unit's article title leaves the distractor
pool, matched or not, because a same-title twin cannot survive its title's removal. That
over-excludes a few legitimate distractors, which is bounded, recorded, and cannot bias
either encoder - both search the same corpus. Above the ceiling fixed in `config`, the
remedy stops being proportionate and the terminal state is `data_stop`.

One residual limitation the result record must state: a near-duplicate carrying a
*different* title is outside the remedy's reach. Title exclusion removes the twin the
deviation argues about; a disambiguation variant under another title is not removed by it.
"""

import gzip
import hashlib
import heapq
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import (
    digest_of,
    savez_compressed_atomic,
    write_text_atomic,
)
from concept_embeddings_rag.corpus.fullwiki import FullWikiRecord, plaintext_of
from concept_embeddings_rag.corpus.pool import IndexingUnit, unit_id_for

RECONCILIATION_FILENAME = "reconciliation.json"

DATA_STOP = "data_stop"

_WHITESPACE = re.compile(r"\s+")


class ScaleCorpusError(Exception):
    """A reconciliation or selection refusal that must not be worked around."""


def _collapsed(value: str) -> str:
    """NFC, then every whitespace run to one space, then stripped."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", value)).strip()


def normalized_title(value: str) -> str:
    """The declared title normalization: NFC, whitespace collapse, then casefold."""
    return _collapsed(value).casefold()


def normalized_text(value: str) -> str:
    """The declared paragraph normalization: NFC and whitespace collapse only.

    Deliberately **not** casefolded. Case carries meaning inside a paragraph, and the
    deviation declares the two normalizations separately; collapsing that distinction
    here would silently widen the matching rule.
    """
    return _collapsed(value)


@dataclass(frozen=True)
class Reconciliation:
    """The mapping and its counts, with the terminal state the gate assigned.

    `mapping` goes from C19 unit id to the official paragraph's own content id. For an
    exact match the two are equal - same title and same plaintext hash to the same id
    under `unit_id_for` - and that identity is precisely why a duplicate is harmless.
    """

    total_units: int
    exact: tuple[str, ...]
    normalized: tuple[str, ...]
    ambiguous: tuple[str, ...]
    unmatched: tuple[str, ...]
    mapping: Mapping[str, str]
    mapped_official_ids: frozenset[str]
    unresolved_titles: frozenset[str]
    excluded_official_counts: Mapping[str, int]
    excluded_official_paragraphs: int
    excluded_gold_units: tuple[str, ...]
    dev_gold_total: int
    dev_gold_matched: int
    dev_gold_unmatched: tuple[str, ...]
    terminal_state: str | None
    digest: str

    @property
    def unresolved(self) -> int:
        """Units the source did not resolve: what the ceiling is read against."""
        return len(self.ambiguous) + len(self.unmatched)


def excluded_titles(reconciliation: Reconciliation) -> frozenset[str]:
    """The normalized titles conservative exclusion removes from the distractor pool."""
    return reconciliation.unresolved_titles


def reconcile(
    units: Sequence[IndexingUnit],
    records: Iterable[FullWikiRecord],
    *,
    dev_gold_ids: frozenset[str],
) -> Reconciliation:
    """Map every frozen C19 unit onto the official source, exactly then normalized.

    One streaming pass over `records`, with only C19-sized indexes held in memory: the
    official release has millions of paragraphs and the laptop is the host. `units` is
    read, never modified.
    """
    exact_index: dict[tuple[str, str], str] = {}
    normalized_index: dict[tuple[str, str], list[str]] = {}
    c19_titles: set[str] = set()
    for unit in units:
        exact_index[(unit.title, unit.text)] = unit.unit_id
        key = (normalized_title(unit.title), normalized_text(unit.text))
        normalized_index.setdefault(key, []).append(unit.unit_id)
        c19_titles.add(normalized_title(unit.title))

    exact_hits: dict[str, str] = {}
    normalized_candidates: dict[str, set[str]] = {}
    # Counted only for titles C19 actually carries, so the counter stays C19-sized
    # rather than corpus-sized, and still answers "how many official paragraphs would
    # excluding this title remove?" without a second pass over the archive.
    official_per_title: Counter[str] = Counter()

    for record in records:
        text = plaintext_of(record)
        title_key = normalized_title(record.title)
        if title_key in c19_titles:
            official_per_title[title_key] += 1
        official_id = unit_id_for(record.title, record.sentences)
        exact = exact_index.get((record.title, text))
        if exact is not None:
            # Exact beats normalized, always, and an exact match never becomes a
            # normalized candidate for some other unit.
            exact_hits[exact] = official_id
            continue
        for unit_id in normalized_index.get((title_key, normalized_text(text)), ()):
            normalized_candidates.setdefault(unit_id, set()).add(official_id)

    matched_exact: list[str] = []
    matched_normalized: list[str] = []
    ambiguous: list[str] = []
    unmatched: list[str] = []
    mapping: dict[str, str] = {}
    for unit in units:
        if unit.unit_id in exact_hits:
            matched_exact.append(unit.unit_id)
            mapping[unit.unit_id] = exact_hits[unit.unit_id]
            continue
        candidates = normalized_candidates.get(unit.unit_id, set())
        if len(candidates) == 1:
            matched_normalized.append(unit.unit_id)
            mapping[unit.unit_id] = next(iter(candidates))
        elif candidates:
            # Ambiguity is counted, never resolved by preferring the first, the
            # shortest or the closest.
            ambiguous.append(unit.unit_id)
        else:
            unmatched.append(unit.unit_id)

    matched_normalized, ambiguous, mapping = _enforce_one_to_one(
        matched_normalized, ambiguous, mapping, matched_exact
    )

    unresolved_ids = frozenset(ambiguous) | frozenset(unmatched)
    titles_by_id = {unit.unit_id: normalized_title(unit.title) for unit in units}
    unresolved_titles = frozenset(titles_by_id[unit_id] for unit_id in unresolved_ids)
    excluded_counts = {title: official_per_title[title] for title in sorted(unresolved_titles)}

    gold_unmatched = tuple(
        unit.unit_id
        for unit in units
        if unit.unit_id in dev_gold_ids and unit.unit_id not in mapping
    )
    excluded_gold = tuple(
        unit.unit_id
        for unit in units
        if unit.unit_id in dev_gold_ids and unit.unit_id in unresolved_ids
    )

    terminal = DATA_STOP if len(unresolved_ids) > config.PHASE_8_1_UNMATCHED_CEILING else None

    return Reconciliation(
        total_units=len(units),
        exact=tuple(matched_exact),
        normalized=tuple(matched_normalized),
        ambiguous=tuple(ambiguous),
        unmatched=tuple(unmatched),
        mapping=mapping,
        mapped_official_ids=frozenset(mapping.values()),
        unresolved_titles=unresolved_titles,
        excluded_official_counts=excluded_counts,
        excluded_official_paragraphs=sum(excluded_counts.values()),
        excluded_gold_units=excluded_gold,
        dev_gold_total=len(dev_gold_ids),
        dev_gold_matched=sum(1 for unit_id in dev_gold_ids if unit_id in mapping),
        dev_gold_unmatched=gold_unmatched,
        terminal_state=terminal,
        digest=_mapping_digest(mapping),
    )


def _enforce_one_to_one(
    normalized: list[str],
    ambiguous: list[str],
    mapping: dict[str, str],
    exact: Sequence[str],
) -> tuple[list[str], list[str], dict[str, str]]:
    """Uniqueness runs both ways: one official paragraph resolves at most one unit.

    The declared rule is that a normalized match "must be unique". A single official
    paragraph standing as the unique candidate of two different C19 units satisfies that
    read only in one direction, and accepting it would map two distinct benchmark units
    onto one paragraph. Both units become ambiguous instead - the conservative reading,
    and the one that keeps the twin out of the pool.
    """
    claims: Counter[str] = Counter(mapping[unit_id] for unit_id in normalized)
    contested = {official for official, count in claims.items() if count > 1}
    if not contested:
        return normalized, ambiguous, mapping
    kept = [unit_id for unit_id in normalized if mapping[unit_id] not in contested]
    demoted = [unit_id for unit_id in normalized if mapping[unit_id] in contested]
    remaining = {
        unit_id: official
        for unit_id, official in mapping.items()
        if unit_id in kept or unit_id in exact
    }
    return kept, sorted([*ambiguous, *demoted]), remaining


def _mapping_digest(mapping: Mapping[str, str]) -> str:
    """A digest over the mapping itself, so a selection cannot claim another mapping."""
    parts: list[str] = []
    for unit_id in sorted(mapping):
        parts.append(f"{unit_id}:{mapping[unit_id]}")
    return digest_of(config.PHASE_8_1_ORDER_RULE, *parts)


def reconciliation_body(reconciliation: Reconciliation) -> dict[str, Any]:
    """Exactly the fields the deviation requires a later reader to recover."""
    return {
        "total_units": reconciliation.total_units,
        "exact": len(reconciliation.exact),
        "normalized_unique": len(reconciliation.normalized),
        "ambiguous": len(reconciliation.ambiguous),
        "unmatched": len(reconciliation.unmatched),
        "ambiguous_unit_ids": list(reconciliation.ambiguous),
        "unmatched_unit_ids": list(reconciliation.unmatched),
        "dev_gold_total": reconciliation.dev_gold_total,
        "dev_gold_matched": reconciliation.dev_gold_matched,
        "dev_gold_unmatched": list(reconciliation.dev_gold_unmatched),
        "excluded_titles": sorted(reconciliation.unresolved_titles),
        "excluded_gold_units": list(reconciliation.excluded_gold_units),
        "excluded_official_paragraphs": reconciliation.excluded_official_paragraphs,
        "excluded_official_counts": dict(reconciliation.excluded_official_counts),
        "unmatched_ceiling": config.PHASE_8_1_UNMATCHED_CEILING,
        # The mapping itself, not only its digest. S3 needs the official paragraphs already
        # represented by C19 in order to exclude them from `D`, and the ids of the
        # normalized matches are not derivable from the counts above: an artifact that
        # cannot be read back is one the next step has to recompute.
        "mapping": dict(reconciliation.mapping),
        "mapped_official_ids": sorted(reconciliation.mapped_official_ids),
        "mapping_digest": reconciliation.digest,
        "terminal_state": reconciliation.terminal_state,
        "title_exclusion_limitation": (
            "conservative title exclusion removes a same-title twin; a near-duplicate "
            "under a different title is outside its reach and must be stated as a "
            "residual limitation of the design"
        ),
    }


def write_reconciliation(directory: Path | str, reconciliation: Reconciliation) -> Path:
    """Persist the reconciliation, refusing to replace a different one in place.

    Re-writing the same mapping is idempotent and allowed; writing a *different* mapping
    over an artifact a later stage may already have trusted is not, because the selection
    and every scale figure downstream carry this digest.
    """
    target = Path(directory) / RECONCILIATION_FILENAME
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("mapping_digest") != reconciliation.digest:
            raise ScaleCorpusError(
                f"{target} already records a reconciliation with digest "
                f"{existing.get('mapping_digest')}, not {reconciliation.digest}: "
                "delete it deliberately or write to a new directory"
            )
    body = reconciliation_body(reconciliation)
    write_text_atomic(target, json.dumps(body, indent=2, ensure_ascii=False, sort_keys=True))
    return target


def load_reconciliation(directory: Path | str) -> dict[str, Any]:
    """The recorded reconciliation, or a refusal naming the stage that writes it."""
    target = Path(directory) / RECONCILIATION_FILENAME
    if not target.exists():
        raise ScaleCorpusError(f"{target} does not exist: run 'scale-corpus' first")
    body: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return body


# --- S3: the deterministic nested selection ---------------------------------
#
# `C19 < C100 < C250 < C500` holds because the three larger corpora are index prefixes of
# **one** frozen ordering, not three independent samples. The ordering is a function of
# paragraph content alone: `42` is a fixed salt inside a hash input string, so archive
# position, file system order and library version cannot reach the result, and there is no
# RNG anywhere on this path - a test asserts this module names none.

SELECTION_FILENAME = "selection.json"
DISTRACTORS_FILENAME = "c500-distractors.jsonl.gz"
# Named without a TOKEN_ prefix on purpose: ruff S105 reads that prefix as a hardcoded
# credential, and renaming a constant is cheaper than silencing a security rule.
COUNTS_FILENAME = "token-counts.npz"
COUNTS_SIDECAR = "token-counts.json"


def order_key(title: str, plaintext: str) -> str:
    """The declared stable ordering key: `sha256("42\\n" + title + "\\n" + plaintext)`.

    Content addressing over the official paragraph, salted so the ordering is specific to
    this experiment. Not integrity, and not randomness: the flag says so to hashlib and to
    the SAST gate alike.
    """
    payload = f"{config.PHASE_8_1_ORDER_SALT}\n{title}\n{plaintext}"
    return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


@dataclass(frozen=True)
class SelectedUnit:
    """One added distractor, with its ordering key and where it was read from."""

    order_key: str
    unit_id: str
    title: str
    sentences: tuple[str, ...]
    member: str
    line: int

    @property
    def unit(self) -> IndexingUnit:
        """The project's own indexing unit, so `indexable_text` matches C19's exactly."""
        return IndexingUnit(unit_id=self.unit_id, title=self.title, sentences=self.sentences)


@dataclass(frozen=True)
class ScaleCorpus:
    """The frozen C19 block followed by the frozen distractor ordering."""

    c19: tuple[IndexingUnit, ...]
    distractors: tuple[IndexingUnit, ...]
    selection_digest: str

    @property
    def total(self) -> int:
        return len(self.c19) + len(self.distractors)


def _eligible_keys(
    records: Iterable[FullWikiRecord],
    mapped_ids: frozenset[str],
    excluded_titles: frozenset[str],
    counted: list[int],
) -> Iterator[tuple[str, str]]:
    """`(order_key, unit_id)` for every official paragraph that may become a distractor."""
    for record in records:
        if normalized_title(record.title) in excluded_titles:
            continue
        unit_id = unit_id_for(record.title, record.sentences)
        if unit_id in mapped_ids:
            continue
        counted[0] += 1
        yield order_key(record.title, plaintext_of(record)), unit_id


def select_distractors(
    records: Callable[[], Iterable[FullWikiRecord]],
    *,
    mapped_ids: frozenset[str],
    excluded_titles: frozenset[str],
    limit: int,
) -> list[SelectedUnit]:
    """The first `limit` paragraphs of `D` in frozen order, in two streaming passes.

    `records` is a factory rather than an iterable because the archive is streamed twice:
    pass 1 keeps a bounded heap of the smallest `limit` keys - tens of megabytes, not the
    millions of records - and pass 2 re-streams to emit those paragraphs. Holding all of
    `D` in memory to sort it would cost gigabytes, and spilling it to disk would cost
    gigabytes of disk; two passes over a bz2 stream cost minutes of free laptop time.
    """
    counted = [0]
    smallest = heapq.nsmallest(
        limit, _eligible_keys(records(), mapped_ids, excluded_titles, counted)
    )
    # A byte-identical duplicate in the official source hashes to one `(key, unit_id)`
    # pair and collapses onto a single unit, exactly as `build_pool` already deduplicates.
    ranked = list(dict.fromkeys(smallest))
    if len(ranked) < limit:
        raise ScaleCorpusError(
            f"only {len(ranked)} distinct eligible paragraphs are available "
            f"({counted[0]} eligible before deduplication) but {limit} were asked for: "
            "the stage refuses rather than quietly selecting fewer"
        )

    wanted = {unit_id: position for position, (_key, unit_id) in enumerate(ranked)}
    found: dict[str, SelectedUnit] = {}
    for record in records():
        unit_id = unit_id_for(record.title, record.sentences)
        if unit_id not in wanted or unit_id in found:
            continue
        found[unit_id] = SelectedUnit(
            order_key=order_key(record.title, plaintext_of(record)),
            unit_id=unit_id,
            title=record.title,
            sentences=record.sentences,
            member=record.member,
            line=record.line,
        )
    if len(found) != len(wanted):
        raise ScaleCorpusError(
            f"pass 2 recovered {len(found)} of {len(wanted)} selected paragraphs: "
            "the two passes disagree, which means the archive changed between them"
        )
    return [found[unit_id] for _key, unit_id in ranked]


def selection_digest_of(unit_ids: Sequence[str]) -> str:
    """A digest over the ordered ids - and, because ids are content hashes, over content."""
    return digest_of(config.PHASE_8_1_ORDER_RULE, *unit_ids)


def _distractor_line(selected: SelectedUnit) -> str:
    return json.dumps(
        {
            "unit_id": selected.unit_id,
            "title": selected.title,
            "sentences": list(selected.sentences),
            "order_key": selected.order_key,
            "member": selected.member,
            "line": selected.line,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def freeze_selection(
    directory: Path | str,
    *,
    c19_unit_ids: Sequence[str],
    selected: Sequence[SelectedUnit],
    source_sha256: str,
    reconciliation_digest: str,
    expected_total: int = config.PHASE_8_1_DISTRACTOR_PREFIXES[-1],
) -> Path:
    """Persist the frozen ordering. No Dense result is read before this file exists."""
    if len(selected) != expected_total:
        raise ScaleCorpusError(
            f"the selection holds {len(selected)} distractors, not the {expected_total} "
            "the declared prefixes require"
        )
    frozen_ids = [item.unit_id for item in selected]
    overlap = frozenset(frozen_ids) & frozenset(c19_unit_ids)
    if overlap:
        raise ScaleCorpusError(
            f"{len(overlap)} selected distractors carry a C19 unit id: the two id blocks "
            "must be disjoint or a paragraph would be indexed twice"
        )
    if len(frozenset(frozen_ids)) != len(frozen_ids):
        raise ScaleCorpusError("the selection repeats a unit id")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = selection_digest_of(frozen_ids)

    target = directory / SELECTION_FILENAME
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("ordered_selection_digest") != digest:
            raise ScaleCorpusError(
                f"{target} already records selection digest "
                f"{existing.get('ordered_selection_digest')}, not {digest}: a frozen "
                "selection is not replaced in place"
            )

    text = "\n".join(_distractor_line(item) for item in selected) + "\n"
    archive = directory / DISTRACTORS_FILENAME
    temporary = archive.with_name(archive.name + ".tmp")
    # mtime=0 so the same selection always compresses to the same bytes.
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(text.encode("utf-8"))
    temporary.replace(archive)

    body = {
        "source_sha256": source_sha256,
        "reconciliation_digest": reconciliation_digest,
        "ordering_rule": config.PHASE_8_1_ORDER_RULE,
        "salt": config.PHASE_8_1_ORDER_SALT,
        "salt_basis": config.PHASE_8_1_ORDER_SALT_BASIS,
        "ordered_selection_digest": digest,
        "c19_digest": digest_of(*c19_unit_ids),
        "c19_units": len(c19_unit_ids),
        "selected": len(selected),
        "corpus_sizes": list(config.PHASE_8_1_CORPUS_SIZES),
        "corpus_labels": list(config.PHASE_8_1_CORPUS_LABELS),
        "distractor_prefixes": list(config.PHASE_8_1_DISTRACTOR_PREFIXES),
        "distractors_file": DISTRACTORS_FILENAME,
    }
    write_text_atomic(target, json.dumps(body, indent=2, ensure_ascii=False, sort_keys=True))
    return target


def load_selection(directory: Path | str, *, c19_units: Sequence[IndexingUnit]) -> ScaleCorpus:
    """Re-derive every id from its content and verify the frozen ordering.

    This is what lets the measurement host trust a file it received over `scp`: the ids are
    recomputed from the text rather than believed, and the ordered digest is recomputed
    from the ids. The C19 block must be the frozen pool, in its existing order.
    """
    directory = Path(directory)
    target = directory / SELECTION_FILENAME
    if not target.exists():
        raise ScaleCorpusError(f"{target} does not exist: run 'scale-corpus' first")
    body = json.loads(target.read_text(encoding="utf-8"))

    c19_ids = [unit.unit_id for unit in c19_units]
    if digest_of(*c19_ids) != body.get("c19_digest"):
        raise ScaleCorpusError(
            "the C19 block does not match the one this selection was frozen against: "
            "the frozen pool is never rebuilt, so this is an alignment defect"
        )

    archive = directory / body.get("distractors_file", DISTRACTORS_FILENAME)
    if not archive.exists():
        raise ScaleCorpusError(f"{archive} does not exist beside {target}")

    distractors: list[IndexingUnit] = []
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            title = str(record["title"])
            sentences = tuple(str(sentence) for sentence in record["sentences"])
            recomputed = unit_id_for(title, sentences)
            if recomputed != record["unit_id"]:
                raise ScaleCorpusError(
                    f"{archive} line {number}: unit id {record['unit_id']} does not match "
                    f"the content, which hashes to {recomputed}"
                )
            distractors.append(IndexingUnit(unit_id=recomputed, title=title, sentences=sentences))

    digest = selection_digest_of([unit.unit_id for unit in distractors])
    if digest != body.get("ordered_selection_digest"):
        raise ScaleCorpusError(
            f"{archive} has selection digest {digest}, not the frozen "
            f"{body.get('ordered_selection_digest')}: the ordering changed"
        )
    return ScaleCorpus(
        c19=tuple(c19_units), distractors=tuple(distractors), selection_digest=digest
    )


def corpus_prefix(corpus: ScaleCorpus, size: int) -> tuple[list[IndexingUnit], list[str]]:
    """The first `size` units of the frozen ordering: C19 first, then distractors.

    Every evaluated corpus is a prefix of one ordering, which is what makes the four
    readings nested by construction rather than by assertion after the fact.
    """
    if size < len(corpus.c19):
        raise ScaleCorpusError(
            f"a corpus of {size} is smaller than the frozen C19 block of {len(corpus.c19)}"
        )
    if size > corpus.total:
        raise ScaleCorpusError(
            f"a corpus of {size} was asked for but the frozen selection holds {corpus.total}"
        )
    units = [*corpus.c19, *corpus.distractors][:size]
    return units, [unit.unit_id for unit in units]


# --- S3: token counts for the added distractors -----------------------------


def token_counts_for(units: Sequence[IndexingUnit], counter: Any) -> dict[str, int]:
    """Token counts under the historical ruler, in bounded batches.

    `TokenCounter.count_units` tokenizes its whole input in a single call, which is correct
    at 19,366 texts and a memory fault at 480,634. Batching lives here rather than in
    `evaluation/budget.py`, because that module defines the ruler behind every inherited
    figure and 8.1 does not touch it.
    """
    counts: dict[str, int] = {}
    batch = config.PHASE_8_1_TOKENIZE_BATCH
    for start in range(0, len(units), batch):
        counts.update(counter.count_units(units[start : start + batch]))
    return counts


def write_token_counts(
    directory: Path | str, counts: Mapping[str, int], *, selection_digest: str
) -> Path:
    """The added distractors' token counts, as NPZ plus a provenance sidecar.

    NPZ because 480,634 entries are megabytes rather than tens of them, because
    `data/*.npz` is already ignored so a regenerable derivative cannot reach the
    repository, and because `savez_compressed_atomic` already exists. The sidecar carries
    what a reader needs to know the counts were measured with the historical ruler.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    unit_ids = sorted(counts)
    savez_compressed_atomic(
        directory / COUNTS_FILENAME,
        unit_ids=np.array(unit_ids, dtype=np.str_),
        counts=np.array([counts[unit_id] for unit_id in unit_ids], dtype=np.int32),
    )
    body = {
        "tokenizer_id": config.BUDGET_TOKENIZER_ID,
        "tokenizer_revision": config.BUDGET_TOKENIZER_REVISION,
        "selection_digest": selection_digest,
        "units": len(unit_ids),
        "counts_file": COUNTS_FILENAME,
        "historical_ruler": "data/token_counts.json is read-only and is not regenerated",
    }
    path = directory / COUNTS_SIDECAR
    write_text_atomic(path, json.dumps(body, indent=2, sort_keys=True))
    return path


def load_token_counts(directory: Path | str) -> dict[str, int]:
    """The added distractors' token counts, keyed by unit id."""
    path = Path(directory) / COUNTS_FILENAME
    if not path.exists():
        raise ScaleCorpusError(f"{path} does not exist: run 'scale-corpus' first")
    # No pickle: ids are fixed-width unicode, exactly as the embedding caches store them.
    with np.load(path) as payload:
        unit_ids = [str(unit_id) for unit_id in payload["unit_ids"]]
        counts = [int(count) for count in payload["counts"]]
    return dict(zip(unit_ids, counts, strict=True))


def check_token_coverage(
    *,
    historical: Mapping[str, int],
    new: Mapping[str, int],
    corpus_unit_ids: Sequence[str],
) -> None:
    """The two rulers must not overlap, and together must cover the corpus.

    An overlap would mean 8.1 recounted a C19 unit under its own artifact, which is the
    first step towards a second ruler; a gap would mean the budget silently skipped a
    paragraph, which would change what every scale figure measured.
    """
    overlap = frozenset(historical) & frozenset(new)
    if overlap:
        raise ScaleCorpusError(
            f"{len(overlap)} unit ids overlap between the historical token counts and the "
            "8.1 counts: the historical ruler is read-only and is never recounted"
        )
    missing = [
        unit_id for unit_id in corpus_unit_ids if unit_id not in historical and unit_id not in new
    ]
    if missing:
        raise ScaleCorpusError(
            f"{len(missing)} corpus units are missing a token count, first {missing[0]}: the "
            "context budget cannot be filled without one"
        )
