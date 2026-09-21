"""Deviation 8.1, S1: the untrusted FullWiki archive.

This is the only module of 8.1 that reads external input it did not produce, so it is
the whole surface of the targeted security review. What the tests below assert is
therefore not "the parser works" but "the parser refuses":

- identity is checked **before** anything is parsed, so a wrong archive cannot spend a
  single line of parsing;
- a member name that could escape a directory is a refusal, not a skip, although nothing
  is ever written to disk in the first place;
- every decompression is bounded, so a member that expands past its ceiling raises
  instead of filling memory;
- the inner layout and the record schema are **observed and recorded**, never assumed,
  because nobody has yet verified that the official dump's representation matches the
  frozen C19 one byte for byte.

Every archive here is a tiny one built in `tmp_path`. No test opens the real 1.5 GB
release and no test reaches the network.
"""

import ast
import bz2
import io
import json
import re
import tarfile
from pathlib import Path

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus import fullwiki
from concept_embeddings_rag.corpus.pool import unit_id_for

# Two paragraphs in the shape the official abstracts dump is expected to hold: a
# `text` list whose first element repeats the title. Whether the real dump does that
# is exactly what `probe_layout` is for - these fixtures exercise both shapes.
LEADING = [
    {
        "id": "12",
        "url": "https://en.wikipedia.org/wiki?curid=12",
        "title": "Anarchism",
        "text": ["Anarchism", " is a political philosophy.", " It rejects hierarchy."],
    },
    {
        "id": "25",
        "url": "https://en.wikipedia.org/wiki?curid=25",
        "title": "Autism",
        "text": ["Autism", " is a developmental disorder."],
    },
]
PLAIN = [
    {
        "id": "31",
        "title": "Albedo",
        "text": ["Albedo", " is the measure of diffuse reflection.", " It is dimensionless."],
    },
]
# The other shape: no repeated title. Which of the two the real release uses is a fact
# S1 reads off the bytes, which is precisely why both fixtures exist.
BODY_ONLY = [
    {
        "id": "44",
        "title": "Aristotle",
        "text": ["Aristotle was a Greek philosopher.", " He studied under Plato."],
    },
]


def a_member_payload(records: list[dict], *, compress: bool) -> bytes:
    body = "\n".join(json.dumps(record) for record in records).encode("utf-8")
    return bz2.compress(body) if compress else body


def an_archive(
    path: Path,
    members: dict[str, bytes],
    *,
    directories: tuple[str, ...] = (),
    symlinks: tuple[str, ...] = (),
) -> Path:
    """A tiny `tar.bz2` in the shape of the real one: nested members inside an outer tar."""
    with tarfile.open(path, "w:bz2") as tar:
        for name in directories:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        for name in symlinks:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = "elsewhere"
            tar.addfile(info)
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


def a_standard_archive(tmp_path: Path, *, compress: bool = True) -> Path:
    return an_archive(
        tmp_path / "wiki.tar.bz2",
        {
            "AA/wiki_00.bz2": a_member_payload(LEADING, compress=compress),
            "AA/wiki_01.bz2": a_member_payload(PLAIN, compress=compress),
        },
        directories=("AA",),
    )


# --- Identity is checked before anything is parsed ----------------------------


def test_the_measured_identity_of_an_archive_is_its_bytes_its_md5_and_its_sha256(tmp_path):
    archive = a_standard_archive(tmp_path)

    verified = fullwiki.verify_archive(
        archive, published_bytes=archive.stat().st_size, published_md5=None
    )

    assert verified["measured_bytes"] == archive.stat().st_size
    assert verified["measured_md5"] == fullwiki.md5_of_file(archive)
    assert len(verified["measured_sha256"]) == 64
    assert verified["bytes_agree"] is True
    # Nothing was published to compare against, and nothing is invented either way.
    assert verified["published_md5"] is None
    assert verified["md5_agree"] is None
    assert "not published" in verified["md5_basis"]
    # The local record is the sha256; the md5 exists only to match a publisher's figure.
    assert "sha256" in verified["integrity_record"]


def test_a_size_mismatch_is_a_refusal(tmp_path):
    archive = a_standard_archive(tmp_path)

    with pytest.raises(fullwiki.FullWikiError, match="bytes"):
        fullwiki.verify_archive(
            archive, published_bytes=archive.stat().st_size + 1, published_md5=None
        )


def test_an_md5_mismatch_is_a_refusal(tmp_path):
    archive = a_standard_archive(tmp_path)

    with pytest.raises(fullwiki.FullWikiError, match="md5"):
        fullwiki.verify_archive(
            archive,
            published_bytes=archive.stat().st_size,
            published_md5="0" * 32,
        )


def test_a_wrong_identity_refuses_before_a_single_record_is_parsed(tmp_path, monkeypatch):
    """The ordering is the point: a wrong archive may not spend one line of parsing."""
    archive = a_standard_archive(tmp_path)
    probed: list[Path] = []
    monkeypatch.setattr(fullwiki, "probe_layout", lambda path: probed.append(path) or {})

    with pytest.raises(fullwiki.FullWikiError):
        fullwiki.describe_source(
            archive,
            url=config.PHASE_8_1_SOURCE_URL,
            published_bytes=archive.stat().st_size + 1,
            published_md5=None,
        )

    assert probed == [], "the layout was probed although identity verification failed"


def test_a_missing_archive_is_a_refusal_naming_the_path(tmp_path):
    with pytest.raises(fullwiki.FullWikiError, match="does not exist"):
        fullwiki.verify_archive(
            tmp_path / "absent.tar.bz2", published_bytes=None, published_md5=None
        )


# --- Member names, and what is skipped ----------------------------------------


@pytest.mark.parametrize("name", ["../escape.bz2", "/absolute.bz2", "AA/../../x", "C:\\x.bz2"])
def test_a_member_whose_name_could_escape_is_refused_rather_than_skipped(tmp_path, name):
    archive = an_archive(tmp_path / "bad.tar.bz2", {name: a_member_payload(PLAIN, compress=True)})

    with pytest.raises(fullwiki.FullWikiError, match="member name"):
        list(fullwiki.iter_members(archive))


def test_the_member_name_pattern_admits_the_real_layout_and_nothing_that_escapes():
    assert fullwiki.MEMBER_NAME.match("AA/wiki_00.bz2")
    assert fullwiki.MEMBER_NAME.match("wiki_00.bz2")
    for bad in ("../x", "/x", "AA/../x", "C:\\x", "AA\\x", "", "AA//x"):
        assert not fullwiki.MEMBER_NAME.match(bad), bad


def test_a_non_regular_member_is_skipped_and_counted(tmp_path):
    archive = an_archive(
        tmp_path / "mixed.tar.bz2",
        {"AA/wiki_00.bz2": a_member_payload(PLAIN, compress=True)},
        directories=("AA",),
        symlinks=("AA/link.bz2",),
    )
    counts = fullwiki.MemberCounts()

    names = [name for name, _payload in fullwiki.iter_members(archive, counts=counts)]

    assert names == ["AA/wiki_00.bz2"]
    assert counts.regular == 1
    assert counts.skipped == 2
    assert counts.by_type["directory"] == 1
    assert counts.by_type["symlink"] == 1


def test_a_member_payload_past_its_ceiling_raises_instead_of_filling_memory(tmp_path):
    """The ceiling is on the *decompressed* payload: 1 MB of zeros compresses to nothing."""
    archive = an_archive(
        tmp_path / "bomb.tar.bz2", {"AA/wiki_00.bz2": bz2.compress(b"0" * (1 << 20))}
    )

    with pytest.raises(fullwiki.FullWikiError, match="ceiling"):
        list(fullwiki.iter_members(archive, max_member_bytes=1024))


def test_a_raw_member_larger_than_its_ceiling_is_refused_before_being_read(tmp_path):
    archive = an_archive(tmp_path / "big.tar.bz2", {"AA/wiki_00.bz2": b"x" * 4096})

    with pytest.raises(fullwiki.FullWikiError, match="ceiling"):
        list(fullwiki.iter_members(archive, max_member_bytes=512))


# --- The layout is observed and recorded, never assumed -----------------------


def test_the_probe_records_the_layout_and_the_schema_it_observed(tmp_path):
    archive = a_standard_archive(tmp_path)

    probe = fullwiki.probe_layout(archive)

    assert probe["member_compression"] == "bz2"
    assert probe["title_field"] == "title"
    assert probe["sentences_field"] == "text"
    assert probe["page_id_field"] == "id"
    assert probe["probed_members"] == ["AA/wiki_00.bz2", "AA/wiki_01.bz2"]
    # The observed key set, exactly as read: this is the record a later reader checks.
    assert "url" in probe["observed_keys"]
    assert probe["observed_keys"] == sorted(probe["observed_keys"])
    assert probe["probed_records"] == 3


def test_the_probe_records_an_uncompressed_inner_member_as_such(tmp_path):
    archive = a_standard_archive(tmp_path, compress=False)

    probe = fullwiki.probe_layout(archive)

    assert probe["member_compression"] == "none"


def test_the_probe_records_whether_the_first_sentence_repeats_the_title(tmp_path):
    """The one schema fact reconciliation depends on, and it is measured, not believed."""
    leading = an_archive(
        tmp_path / "leading.tar.bz2", {"AA/wiki_00.bz2": a_member_payload(LEADING, compress=True)}
    )
    body_only = an_archive(
        tmp_path / "body.tar.bz2", {"AA/wiki_00.bz2": a_member_payload(BODY_ONLY, compress=True)}
    )

    observed = fullwiki.probe_layout(leading)
    assert observed["leading_title_sentence"] is True
    # The reading carries the evidence it was taken from, so a reader sees how many
    # records agreed rather than only the verdict.
    assert observed["leading_title_matches"] == observed["probed_records"]
    assert fullwiki.probe_layout(body_only)["leading_title_sentence"] is False


def test_a_mixed_archive_resolves_to_no_repeated_title_and_says_so(tmp_path):
    """Disagreement is recorded, never resolved by preference.

    A wrong reading here cannot corrupt a figure quietly: it makes the reconciliation
    gate miss wholesale and stop the experiment with `data_stop`.
    """
    archive = an_archive(
        tmp_path / "mixed-shape.tar.bz2",
        {
            "AA/wiki_00.bz2": a_member_payload(LEADING, compress=True),
            "AA/wiki_01.bz2": a_member_payload(BODY_ONLY, compress=True),
        },
    )

    probe = fullwiki.probe_layout(archive)

    assert probe["leading_title_sentence"] is False
    assert probe["leading_title_matches"] == 2
    assert probe["probed_records"] == 3


def test_a_record_schema_the_probe_cannot_resolve_is_a_refusal_naming_what_it_saw(tmp_path):
    archive = an_archive(
        tmp_path / "alien.tar.bz2",
        {"AA/wiki_00.bz2": bz2.compress(json.dumps({"heading": "X", "body": "y"}).encode())},
    )

    with pytest.raises(fullwiki.FullWikiError, match="heading"):
        fullwiki.probe_layout(archive)


def test_an_archive_with_no_usable_member_is_a_refusal(tmp_path):
    archive = an_archive(tmp_path / "empty.tar.bz2", {}, directories=("AA",))

    with pytest.raises(fullwiki.FullWikiError, match="no regular member"):
        fullwiki.probe_layout(archive)


# --- Records become units through the project's own constructors --------------


def test_the_parser_dispatches_on_the_recorded_layout_and_drops_a_repeated_title(tmp_path):
    archive = a_standard_archive(tmp_path)
    layout = fullwiki.layout_of(fullwiki.probe_layout(archive))

    records = list(fullwiki.iter_records(archive, layout))

    assert [record.title for record in records] == ["Anarchism", "Autism", "Albedo"]
    # The leading element repeated the title in this archive, so it is not a sentence.
    assert records[0].sentences == (" is a political philosophy.", " It rejects hierarchy.")
    assert records[0].member == "AA/wiki_00.bz2"
    assert records[0].line == 1
    assert records[1].line == 2
    assert records[2].member == "AA/wiki_01.bz2"
    assert records[2].page_id == "31"


def test_the_same_layout_read_twice_yields_the_identical_records(tmp_path):
    archive = a_standard_archive(tmp_path)
    layout = fullwiki.layout_of(fullwiki.probe_layout(archive))

    first = list(fullwiki.iter_records(archive, layout))
    second = list(fullwiki.iter_records(archive, layout))

    assert first == second


def test_a_records_unit_id_is_the_projects_own_content_hash(tmp_path):
    archive = a_standard_archive(tmp_path)
    layout = fullwiki.layout_of(fullwiki.probe_layout(archive))

    records = list(fullwiki.iter_records(archive, layout))
    units = [fullwiki.unit_of(record) for record in records]

    for record, unit in zip(records, units, strict=True):
        assert unit.unit_id == unit_id_for(record.title, record.sentences)
        # Identical indexed text to a C19 unit's: the title carries the bridge entity,
        # and encoding the two populations differently would bias every scale.
        assert unit.indexable_text == f"{record.title}. {fullwiki.plaintext_of(record)}"
        assert fullwiki.plaintext_of(record) == unit.text


def test_a_byte_identical_duplicate_collapses_onto_the_same_unit_id(tmp_path):
    archive = an_archive(
        tmp_path / "dup.tar.bz2",
        {
            "AA/wiki_00.bz2": a_member_payload(PLAIN, compress=True),
            "AA/wiki_01.bz2": a_member_payload(PLAIN, compress=True),
        },
    )
    layout = fullwiki.layout_of(fullwiki.probe_layout(archive))

    units = [fullwiki.unit_of(record) for record in fullwiki.iter_records(archive, layout)]

    assert len(units) == 2
    assert units[0].unit_id == units[1].unit_id


def test_the_record_ceiling_stops_the_stream_rather_than_reading_the_whole_archive(tmp_path):
    archive = a_standard_archive(tmp_path)
    layout = fullwiki.layout_of(fullwiki.probe_layout(archive))

    with pytest.raises(fullwiki.FullWikiError, match="ceiling"):
        list(fullwiki.iter_records(archive, layout, max_records=2))


def test_a_line_past_the_line_ceiling_raises(tmp_path):
    long_record = {"id": "9", "title": "X", "text": ["y" * 5000]}
    archive = an_archive(
        tmp_path / "long.tar.bz2",
        {"AA/wiki_00.bz2": a_member_payload([long_record], compress=True)},
    )
    layout = fullwiki.layout_of(fullwiki.probe_layout(archive))

    with pytest.raises(fullwiki.FullWikiError, match="ceiling"):
        list(fullwiki.iter_records(archive, layout, max_line_bytes=256))


def test_a_record_missing_its_declared_fields_is_a_refusal_naming_the_member_and_line(tmp_path):
    archive = an_archive(
        tmp_path / "ragged.tar.bz2",
        {
            "AA/wiki_00.bz2": a_member_payload(
                [PLAIN[0], {"id": "1", "title": "Only"}], compress=True
            )
        },
    )
    layout = fullwiki.layout_of(fullwiki.probe_layout(archive))

    with pytest.raises(fullwiki.FullWikiError, match="wiki_00.bz2 line 2"):
        list(fullwiki.iter_records(archive, layout))


# --- The source artifact --------------------------------------------------------


def test_describe_source_carries_identity_layout_and_member_counts(tmp_path):
    archive = a_standard_archive(tmp_path)

    described = fullwiki.describe_source(
        archive,
        url=config.PHASE_8_1_SOURCE_URL,
        published_bytes=archive.stat().st_size,
        published_md5=fullwiki.md5_of_file(archive),
    )

    assert described["url"] == config.PHASE_8_1_SOURCE_URL
    assert described["archive"] == archive.name
    assert described["license"] == config.PHASE_8_1_SOURCE_LICENSE
    assert described["page_checked"] is True
    assert described["md5_agree"] is True
    assert described["bytes_agree"] is True
    assert described["layout"]["sentences_field"] == "text"
    assert described["members"]["regular"] == 2
    # The deviation's recorded identity is kept beside the staged-byte verification,
    # without claiming that the live HotpotQA page was checked.
    assert described["declared_in_spec_bytes"] == config.PHASE_8_1_DECLARED_BYTES
    assert described["declared_in_spec_md5"] == config.PHASE_8_1_DECLARED_MD5
    basis = described["declared_in_spec_basis"].lower()
    assert "verified against the staged archive bytes" in basis
    assert "live hotpotqa page was not checked" in basis


# --- No unsafe deserialization, no network -------------------------------------


def code_names(source: str) -> set[str]:
    """Every identifier the code actually uses, docstrings and comments excluded.

    Checked over the syntax tree rather than the text, for the reason the contamination
    guard is: a docstring that explains the rule must not trip the rule it explains.
    """
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_module_never_extracts_and_never_deserializes_executable_data():
    source = Path(fullwiki.__file__).read_text(encoding="utf-8")
    used = code_names(source)

    for forbidden in ("extractall", "pickle", "marshal", "eval", "exec", "loads_pickle"):
        assert forbidden not in used, forbidden
    # JSON only, and never with a pickle escape hatch.
    assert "loads" in used
    assert "allow_pickle" not in used
    # No network on the untrusted-input path: the operator stages the archive with curl.
    assert not used & {"httpx", "urllib", "requests", "socket"}


def test_the_module_streams_the_archive_and_declares_its_md5_non_security():
    source = Path(fullwiki.__file__).read_text(encoding="utf-8")

    # Streaming access, so no member is ever written to disk and path traversal is
    # structurally impossible rather than defended against.
    assert '"r|bz2"' in source
    # The md5 is declared non-security, which is what the SAST rule asks for and what the
    # integrity record needs to say out loud.
    assert "usedforsecurity=False" in source
    assert "sha256_of_file" in code_names(source)


def test_the_module_reads_no_split_and_no_question():
    source = Path(fullwiki.__file__).read_text(encoding="utf-8")

    assert not re.search(r"gold_unit_ids|Question\b", source)
