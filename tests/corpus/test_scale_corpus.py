"""Deviation 8.1, S2: reconciling the frozen C19 corpus against the official source.

What these tests protect is not bookkeeping. Evaluation matches retrieved units against
`Question.gold_unit_ids` **by unit id**, and `unit_id_for` is a content hash, so a
byte-identical FullWiki duplicate of a C19 paragraph collapses onto the same id and is
harmless. The danger is the **near**-duplicate: its bytes differ, it hashes to a
different id, and it enters the distractor pool as a separate unit. If such a twin is a
gold paragraph, a model can retrieve it, rank it first and score zero credit - a penalty
that grows with corpus size, applies to both encoders, and is indistinguishable in the
results from genuine degradation. It could manufacture `CONVERGENCE` or `BOTH_DEGRADE`
out of nothing.

So the assertions below are about refusal and about over-exclusion:

- exact matching wins over normalized matching, always;
- a normalized match that is not unique is counted ambiguous and **never accepted**;
- at most 50 unresolved units buy conservative title exclusion; 51 is `data_stop`;
- excluding a title removes **every** official paragraph carrying it, matched or not,
  because a same-title twin cannot survive its title's removal;
- the frozen C19 pool is never reconstructed, only mapped.

Every corpus here is a handful of units built in the test. Nothing reads the real
archive and nothing reaches the network.
"""

import gzip
import hashlib
import json
import unicodedata
from pathlib import Path

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus import scale_corpus
from concept_embeddings_rag.corpus.fullwiki import FullWikiRecord
from concept_embeddings_rag.corpus.pool import IndexingUnit, unit_id_for


def unit(title: str, *sentences: str) -> IndexingUnit:
    """A C19 unit built through the project's own identity function."""
    return IndexingUnit(
        unit_id=unit_id_for(title, tuple(sentences)),
        title=title,
        sentences=tuple(sentences),
    )


def record(
    title: str, *sentences: str, member: str = "AA/wiki_00", line: int = 1
) -> FullWikiRecord:
    """One official paragraph, as `iter_records` would hand it over."""
    return FullWikiRecord(
        title=title,
        sentences=tuple(sentences),
        member=member,
        line=line,
        page_id=None,
    )


# --- normalization ----------------------------------------------------------


def test_normalized_title_casefolds_and_collapses_whitespace() -> None:
    assert scale_corpus.normalized_title("  The   Great\tEscape ") == "the great escape"


def test_normalized_text_collapses_whitespace_but_keeps_case() -> None:
    assert scale_corpus.normalized_text("A  paragraph\nof text.") == "A paragraph of text."


def test_normalization_is_nfc() -> None:
    """An accent written two ways is one paragraph, not two.

    Built through `unicodedata` rather than typed as two literals: `ruff format` rewrote
    the escape sequences this test first used into visually identical characters, and an
    editor could normalize them next - which would leave the assertion comparing a string
    to itself and passing for no reason.
    """
    text = "Amelie works in a cafe."
    decomposed = unicodedata.normalize("NFD", text.replace("e ", "\u00e9 ", 1))
    composed = unicodedata.normalize("NFC", text.replace("e ", "\u00e9 ", 1))
    assert decomposed != composed

    assert scale_corpus.normalized_text(decomposed) == scale_corpus.normalized_text(composed)
    assert scale_corpus.normalized_title(decomposed) == scale_corpus.normalized_title(composed)


# --- matching --------------------------------------------------------------


def test_exact_match_beats_normalized() -> None:
    """A unit with both an exact and a merely-normalized candidate resolves exactly."""
    c19 = unit("Anarchism", "It is a political philosophy.")
    records = [
        record("anarchism", "It is  a political philosophy.", line=1),  # normalized only
        record("Anarchism", "It is a political philosophy.", line=2),  # exact
    ]

    result = scale_corpus.reconcile([c19], records, dev_gold_ids=frozenset())

    assert result.exact == (c19.unit_id,)
    assert result.normalized == ()
    assert result.unmatched == ()
    assert result.ambiguous == ()
    # An exact match is the same title and the same plaintext, so `unit_id_for` returns
    # the same id: the mapping is an identity, and that is the whole reason a duplicate
    # is harmless.
    assert result.mapping[c19.unit_id] == c19.unit_id
    assert result.terminal_state is None


def test_normalized_unique_match_is_accepted() -> None:
    c19 = unit("Amélie Poulain", "She works in a café.")
    official = record("amélie   poulain", "She works in a  café.")

    result = scale_corpus.reconcile([c19], [official], dev_gold_ids=frozenset())

    assert result.normalized == (c19.unit_id,)
    assert result.exact == ()
    assert result.unmatched == ()
    # It is a different paragraph by content, so it maps to a different id - which is
    # precisely why it must be kept out of the distractor pool.
    assert result.mapping[c19.unit_id] != c19.unit_id
    assert result.mapping[c19.unit_id] in result.mapped_official_ids


def test_normalized_match_that_is_not_unique_is_ambiguous_and_not_accepted() -> None:
    """Two distinct official paragraphs normalizing onto one unit resolve to neither."""
    c19 = unit("Albedo", "It is the measure of diffuse reflection.")
    records = [
        record("albedo", "It is the  measure of diffuse reflection.", line=1),
        record("ALBEDO", "It is the measure  of diffuse reflection.", line=2),
    ]

    result = scale_corpus.reconcile([c19], records, dev_gold_ids=frozenset())

    assert result.ambiguous == (c19.unit_id,)
    assert result.normalized == ()
    assert c19.unit_id not in result.mapping
    # Ambiguity is counted, never resolved by preferring the first or the shortest.
    assert result.mapped_official_ids == frozenset()


def test_unmatched_unit_is_counted_and_not_mapped() -> None:
    c19 = unit("Autism", "It is a developmental disorder.")
    other = record("Albedo", "Something else entirely.")

    result = scale_corpus.reconcile([c19], [other], dev_gold_ids=frozenset())

    assert result.unmatched == (c19.unit_id,)
    assert result.mapping == {}


# --- the gate and its remedy ------------------------------------------------


def unresolved_corpus(n: int) -> tuple[list[IndexingUnit], list[FullWikiRecord]]:
    """`n` C19 units the official source does not carry, plus one that it does."""
    units = [unit(f"Missing {i}", f"Paragraph {i}.") for i in range(n)]
    resolved = unit("Present", "This one is in the official source.")
    records = [record("Present", "This one is in the official source.")]
    return [*units, resolved], records


def test_fifty_unresolved_units_continue_with_title_exclusion() -> None:
    units, records = unresolved_corpus(config.PHASE_8_1_UNMATCHED_CEILING)

    result = scale_corpus.reconcile(units, records, dev_gold_ids=frozenset())

    assert len(result.unmatched) == config.PHASE_8_1_UNMATCHED_CEILING
    assert result.terminal_state is None
    assert len(scale_corpus.excluded_titles(result)) == config.PHASE_8_1_UNMATCHED_CEILING


def test_fifty_one_unresolved_units_are_a_data_stop() -> None:
    units, records = unresolved_corpus(config.PHASE_8_1_UNMATCHED_CEILING + 1)

    result = scale_corpus.reconcile(units, records, dev_gold_ids=frozenset())

    assert result.terminal_state == "data_stop"


def test_ambiguous_units_count_against_the_same_ceiling() -> None:
    """The ceiling is on unresolved units, and an ambiguous unit is unresolved."""
    units: list[IndexingUnit] = []
    records: list[FullWikiRecord] = []
    for i in range(config.PHASE_8_1_UNMATCHED_CEILING + 1):
        units.append(unit(f"Twin {i}", f"Paragraph {i}."))
        records.append(record(f"twin {i}", f"Paragraph  {i}.", line=1))
        records.append(record(f"TWIN {i}", f"Paragraph   {i}.", line=2))

    result = scale_corpus.reconcile(units, records, dev_gold_ids=frozenset())

    assert len(result.ambiguous) == config.PHASE_8_1_UNMATCHED_CEILING + 1
    assert result.terminal_state == "data_stop"


def test_title_exclusion_removes_every_paragraph_of_that_title() -> None:
    """Matched or not, every official paragraph under an unresolved title is excluded."""
    missing = unit("Shared Title", "The C19 paragraph.")
    records = [
        record("Shared Title", "A different paragraph under the same title.", line=1),
        record("shared title", "Yet another one, in another case.", line=2),
        record("Unrelated", "Nothing to do with it.", line=3),
    ]

    result = scale_corpus.reconcile([missing], records, dev_gold_ids=frozenset())

    excluded = scale_corpus.excluded_titles(result)
    assert scale_corpus.normalized_title("Shared Title") in excluded
    assert scale_corpus.normalized_title("Unrelated") not in excluded
    # Both same-title paragraphs are counted as removed, in either case form; the
    # unrelated one is not.
    assert result.excluded_official_paragraphs == 2


def test_pool_is_never_reconstructed() -> None:
    """Reconciliation maps C19; it does not touch a title, a sentence or an id."""
    c19 = unit("Amélie Poulain", "She works in a café.")
    before = (c19.unit_id, c19.title, c19.sentences, c19.indexable_text)
    units = [c19]

    scale_corpus.reconcile(
        units, [record("amélie poulain", "She works in a  café.")], dev_gold_ids=frozenset()
    )

    assert units == [c19]
    assert (c19.unit_id, c19.title, c19.sentences, c19.indexable_text) == before


# --- gold accounting -------------------------------------------------------


def test_dev_gold_units_are_accounted_separately() -> None:
    """A gold paragraph that does not resolve is the case the gate exists for."""
    resolved_gold = unit("Present", "In the source.")
    missing_gold = unit("Absent", "Not in the source.")
    filler = unit("Filler", "Neither gold nor missing.")
    records = [
        record("Present", "In the source."),
        record("Filler", "Neither gold nor missing."),
    ]

    result = scale_corpus.reconcile(
        [resolved_gold, missing_gold, filler],
        records,
        dev_gold_ids=frozenset({resolved_gold.unit_id, missing_gold.unit_id}),
    )

    assert result.dev_gold_total == 2
    assert result.dev_gold_matched == 1
    assert result.dev_gold_unmatched == (missing_gold.unit_id,)
    assert result.excluded_gold_units == (missing_gold.unit_id,)


# --- the artifact ----------------------------------------------------------


def test_digest_is_a_function_of_the_mapping(tmp_path: Path) -> None:
    c19 = unit("Anarchism", "It is a political philosophy.")
    matched = scale_corpus.reconcile(
        [c19], [record("Anarchism", "It is a political philosophy.")], dev_gold_ids=frozenset()
    )
    other = scale_corpus.reconcile(
        [c19], [record("anarchism", "It is  a political philosophy.")], dev_gold_ids=frozenset()
    )

    # Same unit, different mapping - the digest must move with the mapping, because it
    # is what a later reader checks a corpus selection against.
    assert matched.digest != other.digest
    # And it is reproducible from the same mapping.
    again = scale_corpus.reconcile(
        [c19], [record("Anarchism", "It is a political philosophy.")], dev_gold_ids=frozenset()
    )
    assert again.digest == matched.digest


def test_write_reconciliation_records_what_the_deviation_requires(tmp_path: Path) -> None:
    units = [
        unit("Present", "In the source."),
        unit("Absent", "Not in the source."),
    ]
    records = [
        record("Present", "In the source.", line=1),
        record("Absent", "A different paragraph under an unresolved title.", line=2),
    ]
    result = scale_corpus.reconcile(units, records, dev_gold_ids=frozenset({units[0].unit_id}))

    path = scale_corpus.write_reconciliation(tmp_path, result)
    body = json.loads(path.read_text(encoding="utf-8"))

    assert body["total_units"] == 2
    assert body["exact"] == 1
    assert body["normalized_unique"] == 1 or body["unmatched"] == 1
    for key in (
        "total_units",
        "exact",
        "normalized_unique",
        "ambiguous",
        "unmatched",
        "dev_gold_total",
        "dev_gold_matched",
        "dev_gold_unmatched",
        "excluded_titles",
        "excluded_gold_units",
        "excluded_official_paragraphs",
        "excluded_official_counts",
        "mapping",
        "mapped_official_ids",
        "unmatched_ceiling",
        "mapping_digest",
        "terminal_state",
    ):
        assert key in body, key
    assert body["unmatched_ceiling"] == config.PHASE_8_1_UNMATCHED_CEILING
    assert body["mapping_digest"] == result.digest
    # The artifact must be readable back into what S3 needs, without re-reconciling.
    assert body["mapping"] == dict(result.mapping)
    assert set(body["mapped_official_ids"]) == set(result.mapped_official_ids)


def test_write_reconciliation_refuses_to_overwrite_a_different_run(tmp_path: Path) -> None:
    """An artifact behind a decision is not silently replaced by a second run."""
    c19 = unit("Anarchism", "It is a political philosophy.")
    first = scale_corpus.reconcile(
        [c19], [record("Anarchism", "It is a political philosophy.")], dev_gold_ids=frozenset()
    )
    scale_corpus.write_reconciliation(tmp_path, first)

    second = scale_corpus.reconcile([c19], [], dev_gold_ids=frozenset())
    with pytest.raises(scale_corpus.ScaleCorpusError, match="reconciliation"):
        scale_corpus.write_reconciliation(tmp_path, second)


# --- no randomness on this path --------------------------------------------


def test_module_names_no_random_source() -> None:
    """`42` is a hash salt, not RNG state, and nothing here samples anything."""
    source = Path(scale_corpus.__file__).read_text(encoding="utf-8")
    for forbidden in ("default_rng", "RandomState", "random.", "shuffle", "sample("):
        assert forbidden not in source, forbidden


# --- S3: the deterministic nested selection ---------------------------------


def factory(records: list[FullWikiRecord]):
    """`select_distractors` takes a factory because it makes two streaming passes."""
    return lambda: iter(records)


def test_order_key_is_the_declared_salted_digest() -> None:
    """`42` is a salt inside a hash input, and the input is exactly as declared."""
    title, plaintext = "Albedo", "It is the measure of diffuse reflection."
    expected = hashlib.sha256(f"42\n{title}\n{plaintext}".encode()).hexdigest()

    assert scale_corpus.order_key(title, plaintext) == expected


def test_selection_excludes_mapped_paragraphs_and_excluded_titles() -> None:
    mapped = record("Mapped", "Already represented by C19.")
    excluded = record("Excluded Title", "A paragraph under an unresolved title.")
    kept = record("Kept", "An ordinary distractor.")

    selected = scale_corpus.select_distractors(
        factory([mapped, excluded, kept]),
        mapped_ids=frozenset({unit_id_for(mapped.title, mapped.sentences)}),
        excluded_titles=frozenset({scale_corpus.normalized_title("Excluded Title")}),
        limit=1,
    )

    assert [unit.title for unit in selected] == ["Kept"]


def test_selection_is_ordered_by_the_salted_digest() -> None:
    records = [record(f"Title {i}", f"Paragraph {i}.") for i in range(6)]

    selected = scale_corpus.select_distractors(
        factory(records), mapped_ids=frozenset(), excluded_titles=frozenset(), limit=4
    )

    keys = [unit.order_key for unit in selected]
    assert keys == sorted(keys)
    # And the four selected are the four smallest keys of the six, not the first four
    # the archive happened to hold.
    everything = sorted(scale_corpus.order_key(r.title, " ".join(r.sentences)) for r in records)
    assert keys == everything[:4]


def test_selection_is_reproducible() -> None:
    records = [record(f"Title {i}", f"Paragraph {i}.") for i in range(8)]
    kwargs = {"mapped_ids": frozenset(), "excluded_titles": frozenset(), "limit": 5}

    first = scale_corpus.select_distractors(factory(records), **kwargs)
    second = scale_corpus.select_distractors(factory(list(reversed(records))), **kwargs)

    # Archive order must not survive into the selection: the ordering is a function of
    # content alone.
    assert [u.unit_id for u in first] == [u.unit_id for u in second]


def test_selection_refuses_a_shortfall_with_both_numbers() -> None:
    records = [record(f"Title {i}", f"Paragraph {i}.") for i in range(3)]

    with pytest.raises(scale_corpus.ScaleCorpusError, match="3"):
        scale_corpus.select_distractors(
            factory(records), mapped_ids=frozenset(), excluded_titles=frozenset(), limit=10
        )


def test_declared_prefix_arithmetic_holds() -> None:
    """The three prefixes plus C19 are exactly the four declared corpus sizes."""
    c19 = config.EXPECTED_N_UNITS
    assert c19 == config.PHASE_8_1_CORPUS_SIZES[0]
    for prefix, size in zip(
        config.PHASE_8_1_DISTRACTOR_PREFIXES, config.PHASE_8_1_CORPUS_SIZES[1:], strict=True
    ):
        assert c19 + prefix == size


def small_corpus(tmp_path: Path) -> tuple[scale_corpus.ScaleCorpus, list[IndexingUnit]]:
    """A three-unit C19 block and five frozen distractors, round-tripped through disk."""
    c19_units = [unit(f"C19 {i}", f"Frozen paragraph {i}.") for i in range(3)]
    records = [record(f"Distractor {i}", f"Added paragraph {i}.") for i in range(5)]
    selected = scale_corpus.select_distractors(
        factory(records), mapped_ids=frozenset(), excluded_titles=frozenset(), limit=5
    )
    scale_corpus.freeze_selection(
        tmp_path,
        c19_unit_ids=[u.unit_id for u in c19_units],
        selected=selected,
        source_sha256="0" * 64,
        reconciliation_digest="1" * 64,
        expected_total=5,
    )
    return scale_corpus.load_selection(tmp_path, c19_units=c19_units), c19_units


def test_freeze_and_load_round_trip(tmp_path: Path) -> None:
    corpus, c19_units = small_corpus(tmp_path)

    assert [u.unit_id for u in corpus.c19] == [u.unit_id for u in c19_units]
    assert len(corpus.distractors) == 5
    # Every distractor is a project unit, so its indexable text is built exactly as a
    # C19 unit's is - the asymmetry that would bias every scale is impossible here.
    for distractor in corpus.distractors:
        assert distractor.indexable_text == f"{distractor.title}. {distractor.text}"
        assert distractor.unit_id == unit_id_for(distractor.title, distractor.sentences)


def test_corpus_prefixes_nest(tmp_path: Path) -> None:
    corpus, _ = small_corpus(tmp_path)

    small, small_ids = scale_corpus.corpus_prefix(corpus, 4)
    large, large_ids = scale_corpus.corpus_prefix(corpus, 6)

    assert len(small) == 4
    assert len(large) == 6
    assert small_ids == large_ids[:4]
    assert set(small_ids).issubset(set(large_ids))
    # The C19 block always comes first, in its frozen order.
    assert small_ids[:3] == [u.unit_id for u in corpus.c19]


def test_corpus_prefix_refuses_a_size_the_corpus_cannot_serve(tmp_path: Path) -> None:
    corpus, _ = small_corpus(tmp_path)

    with pytest.raises(scale_corpus.ScaleCorpusError, match="8"):
        scale_corpus.corpus_prefix(corpus, 9)


def test_load_selection_refuses_a_tampered_unit_line(tmp_path: Path) -> None:
    """A distractor whose content was edited no longer hashes to its recorded id."""
    small_corpus(tmp_path)
    archive = tmp_path / scale_corpus.DISTRACTORS_FILENAME
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    first = json.loads(lines[0])
    first["sentences"] = ["Tampered content."]
    lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
    with archive.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(("\n".join(lines) + "\n").encode("utf-8"))

    c19_units = [unit(f"C19 {i}", f"Frozen paragraph {i}.") for i in range(3)]
    with pytest.raises(scale_corpus.ScaleCorpusError, match="unit id"):
        scale_corpus.load_selection(tmp_path, c19_units=c19_units)


def test_load_selection_refuses_a_reordered_archive(tmp_path: Path) -> None:
    """The ordered-selection digest is what makes a 120 MB file trustworthy on the pod."""
    small_corpus(tmp_path)
    archive = tmp_path / scale_corpus.DISTRACTORS_FILENAME
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    lines.reverse()
    with archive.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(("\n".join(lines) + "\n").encode("utf-8"))

    c19_units = [unit(f"C19 {i}", f"Frozen paragraph {i}.") for i in range(3)]
    with pytest.raises(scale_corpus.ScaleCorpusError, match="digest"):
        scale_corpus.load_selection(tmp_path, c19_units=c19_units)


def test_load_selection_refuses_a_different_c19_block(tmp_path: Path) -> None:
    small_corpus(tmp_path)
    wrong = [unit(f"Other {i}", f"Another frozen paragraph {i}.") for i in range(3)]

    with pytest.raises(scale_corpus.ScaleCorpusError, match="C19"):
        scale_corpus.load_selection(tmp_path, c19_units=wrong)


def test_selection_manifest_records_its_provenance(tmp_path: Path) -> None:
    small_corpus(tmp_path)
    body = json.loads((tmp_path / scale_corpus.SELECTION_FILENAME).read_text(encoding="utf-8"))

    assert body["ordering_rule"] == config.PHASE_8_1_ORDER_RULE
    assert body["salt"] == config.PHASE_8_1_ORDER_SALT
    assert body["salt_basis"] == config.PHASE_8_1_ORDER_SALT_BASIS
    assert body["source_sha256"] == "0" * 64
    assert body["reconciliation_digest"] == "1" * 64
    assert body["corpus_sizes"] == list(config.PHASE_8_1_CORPUS_SIZES)
    assert body["selected"] == 5
    assert len(body["ordered_selection_digest"]) == 64


# --- S3: token counts for the added distractors -----------------------------


class RecordingCounter:
    """A `TokenCounter` stand-in that records the size of every batch it is handed."""

    def __init__(self) -> None:
        self.batches: list[int] = []
        self.tokenizer_id = config.BUDGET_TOKENIZER_ID
        self.revision = config.BUDGET_TOKENIZER_REVISION

    def count_units(self, units) -> dict[str, int]:
        self.batches.append(len(units))
        return {u.unit_id: len(u.indexable_text.split()) for u in units}


def test_token_counts_are_batched(tmp_path: Path) -> None:
    """`TokenCounter.count_units` tokenizes its whole input in one call, so 8.1 batches."""
    units = [
        unit(f"Unit {i}", f"Paragraph {i}.") for i in range(config.PHASE_8_1_TOKENIZE_BATCH + 7)
    ]
    counter = RecordingCounter()

    counts = scale_corpus.token_counts_for(units, counter)

    assert len(counts) == len(units)
    assert max(counter.batches) <= config.PHASE_8_1_TOKENIZE_BATCH
    assert sum(counter.batches) == len(units)


def test_token_counts_round_trip_with_their_provenance(tmp_path: Path) -> None:
    units = [unit(f"Unit {i}", f"Paragraph {i}.") for i in range(4)]
    counts = scale_corpus.token_counts_for(units, RecordingCounter())

    scale_corpus.write_token_counts(tmp_path, counts, selection_digest="2" * 64)

    assert scale_corpus.load_token_counts(tmp_path) == counts
    sidecar = json.loads((tmp_path / scale_corpus.COUNTS_SIDECAR).read_text(encoding="utf-8"))
    assert sidecar["tokenizer_id"] == config.BUDGET_TOKENIZER_ID
    assert sidecar["tokenizer_revision"] == config.BUDGET_TOKENIZER_REVISION
    assert sidecar["selection_digest"] == "2" * 64
    assert sidecar["units"] == 4


def test_token_coverage_refuses_an_overlap_or_a_gap() -> None:
    """The new counts must not touch the historical ruler, and together must cover the corpus."""
    historical = {"a": 10, "b": 20}
    new = {"c": 30}

    scale_corpus.check_token_coverage(
        historical=historical, new=new, corpus_unit_ids=["a", "b", "c"]
    )

    with pytest.raises(scale_corpus.ScaleCorpusError, match="overlap"):
        scale_corpus.check_token_coverage(
            historical=historical, new={"b": 1, "c": 30}, corpus_unit_ids=["a", "b", "c"]
        )
    with pytest.raises(scale_corpus.ScaleCorpusError, match="missing"):
        scale_corpus.check_token_coverage(
            historical=historical, new=new, corpus_unit_ids=["a", "b", "c", "d"]
        )
