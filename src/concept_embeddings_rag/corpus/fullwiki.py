"""Deviation 8.1, S1: the official HotpotQA FullWiki archive, as untrusted input.

This is the only module of the deviation that reads bytes the project did not produce,
which is why it is small, why it is separate, and why it is the whole reading list of the
targeted security review. Four properties are structural rather than defended:

1. **Nothing is ever extracted.** `tarfile.open(path, "r|bz2")` gives sequential access
   to members, each read into memory and parsed. No member is written to disk, so path
   traversal cannot happen; the member-name check below is a second, cheaper refusal that
   runs first.
2. **Every decompression is bounded.** A tar member larger than its ceiling is refused
   before it is read, and a nested bz2 payload is decompressed incrementally against a
   running total, so a member that expands past its ceiling raises rather than filling
   memory. Lines and total records have their own ceilings.
3. **Only JSON is parsed.** `json.loads`, and nothing else: no pickle, no `eval`, no
   `allow_pickle`, no `tarfile.extractall`.
4. **The layout is observed, not assumed.** Nobody has yet verified that the official
   dump's textual representation matches the frozen C19 one. So `probe_layout` reads the
   first few members, records their names, whether each payload carries the bz2 magic,
   the sorted key set it actually saw, which candidate field names resolved against it,
   and whether the first element of the sentence list repeats the article title - and all
   of that is written into `source.json`. `iter_records` then dispatches on the recorded
   fact. A recollection is a claim until the bytes are read.

The md5 here exists for exactly one purpose: matching a checksum the publisher printed.
The local integrity record is the sha256. `usedforsecurity=False` says so to hashlib and
to the SAST gate alike.

Records become `IndexingUnit`s through `unit_id_for` and `IndexingUnit` themselves, never
a local copy, so a byte-identical FullWiki duplicate of a C19 paragraph hashes to the
same id and collapses, and every distractor's `indexable_text` is `f"{title}. {text}"`
exactly as a C19 unit's is.
"""

import bz2
import gzip
import hashlib
import json
import re
import tarfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, write_text_atomic
from concept_embeddings_rag.corpus.download import sha256_of_file
from concept_embeddings_rag.corpus.pool import IndexingUnit, unit_id_for

__all__ = [
    "CORPUS_MANIFEST",
    "CORPUS_NAME",
    "MEMBER_NAME",
    "ArchiveLayout",
    "FullWikiError",
    "FullWikiRecord",
    "MemberCounts",
    "describe_source",
    "iter_members",
    "iter_records",
    "layout_of",
    "load_corpus",
    "md5_of_file",
    "plaintext_of",
    "probe_layout",
    "unit_of",
    "verify_archive",
    "write_corpus",
]

# No leading separator, no `..` segment, no drive letter and no backslash. The negative
# lookahead is what makes `..` impossible: `.` has to stay in the character class for
# ordinary names like `wiki_00.bz2`, so a `..` segment would otherwise match. A name that
# does not match is a refusal rather than a skip: an archive holding one is not the
# archive this deviation declared, whatever else it contains.
MEMBER_NAME = re.compile(r"^(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")

# The bz2 file magic. A nested member either starts with it or it does not, and which one
# is the case is recorded rather than believed.
BZ2_MAGIC = b"BZh"

INTEGRITY_RECORD = (
    "the local integrity record is measured_sha256; measured_md5 exists only to match "
    "the checksum the publisher printed beside the release"
)
MD5_NOT_PUBLISHED = "not published on the source page"
MD5_PUBLISHED = "compared against the checksum published beside the release"


class FullWikiError(Exception):
    """The archive in hand is not the one this deviation declared, or refuses to be read."""


@dataclass
class MemberCounts:
    """What the stream saw, by kind. Recorded in `source.json` rather than discarded."""

    regular: int = 0
    skipped: int = 0
    by_type: dict[str, int] = field(default_factory=dict)

    def note(self, kind: str) -> None:
        self.by_type[kind] = self.by_type.get(kind, 0) + 1
        self.skipped += 1

    def as_dict(self) -> dict[str, Any]:
        return {"regular": self.regular, "skipped": self.skipped, "by_type": dict(self.by_type)}


@dataclass(frozen=True)
class ArchiveLayout:
    """The layout and record schema **as observed at S1**, not as assumed.

    `leading_title_sentence` is the one field reconciliation depends on: when the first
    element of the sentence list repeats the article title, it is not a sentence, and
    including it would make every official plaintext differ from its C19 counterpart. A
    wrong reading here does not corrupt a result quietly - it makes the reconciliation
    gate fail wholesale and stop the experiment with `data_stop`, which is the designed
    safety net rather than a silent risk.
    """

    member_compression: str
    title_field: str
    sentences_field: str
    page_id_field: str | None
    leading_title_sentence: bool

    @property
    def compressed_members(self) -> bool:
        return self.member_compression == "bz2"


@dataclass(frozen=True)
class FullWikiRecord:
    """One official paragraph, with where it was read from."""

    title: str
    sentences: tuple[str, ...]
    member: str
    line: int
    page_id: str | None


def md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """The md5 of a file, for comparison against a published checksum and nothing else.

    Not an integrity mechanism: the record this project keeps is the sha256 beside it.
    `usedforsecurity=False` states that, and is what lets the SAST rule pass on a hash
    that would otherwise be a finding.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(
    path: Path | str,
    *,
    published_bytes: int | None,
    published_md5: str | None,
) -> dict[str, Any]:
    """Measure the archive's identity, refusing **before** anything is parsed.

    A size or checksum mismatch raises here, which is upstream of every parse in this
    module: a wrong archive may not spend a single line of parsing. Where the source page
    publishes no checksum, `published_md5` is `None` and the artifact records that fact
    rather than an invented figure.
    """
    path = Path(path)
    if not path.is_file():
        raise FullWikiError(f"{path} does not exist; stage the official archive there first")

    measured_bytes = path.stat().st_size
    if published_bytes is not None and measured_bytes != published_bytes:
        raise FullWikiError(
            f"{path.name} is {measured_bytes} bytes and the source page publishes "
            f"{published_bytes}; refusing to parse an archive that is not the declared one"
        )

    measured_md5 = md5_of_file(path)
    if published_md5 is not None and measured_md5.lower() != published_md5.lower():
        raise FullWikiError(
            f"{path.name} has md5 {measured_md5} and the source page publishes "
            f"{published_md5}; refusing to parse it"
        )

    return {
        "archive": path.name,
        "published_bytes": published_bytes,
        "published_md5": published_md5,
        "measured_bytes": measured_bytes,
        "measured_md5": measured_md5,
        "measured_sha256": sha256_of_file(path),
        "bytes_agree": None if published_bytes is None else measured_bytes == published_bytes,
        "md5_agree": None if published_md5 is None else True,
        "md5_basis": MD5_NOT_PUBLISHED if published_md5 is None else MD5_PUBLISHED,
        "integrity_record": INTEGRITY_RECORD,
    }


def _member_kind(member: tarfile.TarInfo) -> str:
    if member.isdir():
        return "directory"
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    if member.ischr() or member.isblk() or member.isfifo():
        return "device"
    return "other"


def _decompress_bounded(payload: bytes, name: str, ceiling: int) -> bytes:
    """Inflate a nested bz2 member against a running total.

    Decompressing first and checking the size afterwards would already have paid the
    memory, which is the whole failure this guards.
    """
    decompressor = bz2.BZ2Decompressor()
    chunks: list[bytes] = []
    total = 0
    pending = payload
    while not decompressor.eof:
        # Never ask for more than one byte past the ceiling: the bound has to hold on the
        # allocation, not only on the check that follows it.
        piece = decompressor.decompress(pending, max_length=min(1 << 20, ceiling + 1 - total))
        pending = b""
        if not piece:
            # Input exhausted before the stream ended, or nothing more to drain: either
            # way there is no further output to take, and looping would not produce any.
            break
        total += len(piece)
        if total > ceiling:
            raise FullWikiError(
                f"member {name} expands past the {ceiling} byte ceiling; refusing to inflate it"
            )
        chunks.append(piece)
    return b"".join(chunks)


def iter_members(
    path: Path | str,
    *,
    max_member_bytes: int = config.PHASE_8_1_MAX_MEMBER_BYTES,
    counts: MemberCounts | None = None,
) -> Iterator[tuple[str, bytes]]:
    """Stream the archive's regular members as `(name, decompressed payload)`.

    Sequential access, nothing written to disk, every member bounded, and anything that is
    not a regular file skipped and counted. A member name that could escape a directory is
    a refusal: nothing here would write it anywhere, and an archive containing one is not
    the declared release.
    """
    path = Path(path)
    counts = counts if counts is not None else MemberCounts()
    with tarfile.open(path, "r|bz2") as tar:
        for member in tar:
            if not member.isfile():
                counts.note(_member_kind(member))
                continue
            if not MEMBER_NAME.match(member.name):
                raise FullWikiError(
                    f"member name {member.name!r} is not a plain relative path; refusing to "
                    "read an archive whose layout is not the declared one"
                )
            if member.size > max_member_bytes:
                raise FullWikiError(
                    f"member {member.name} declares {member.size} bytes, past the "
                    f"{max_member_bytes} byte ceiling; refusing to read it"
                )
            handle = tar.extractfile(member)
            if handle is None:
                counts.note("unreadable")
                continue
            raw = handle.read(max_member_bytes + 1)
            if len(raw) > max_member_bytes:
                raise FullWikiError(
                    f"member {member.name} exceeds the {max_member_bytes} byte ceiling"
                )
            counts.regular += 1
            if raw.startswith(BZ2_MAGIC):
                yield member.name, _decompress_bounded(raw, member.name, max_member_bytes)
            else:
                yield member.name, raw


def _resolve_field(keys: Sequence[str], candidates: Sequence[str]) -> str | None:
    """The first declared candidate the archive actually holds, or `None`.

    A resolution against observed keys, not an assumption: the caller records which
    candidate won and refuses when none did.
    """
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def probe_layout(
    path: Path | str,
    *,
    members: int = config.PHASE_8_1_PROBE_MEMBERS,
    records: int = config.PHASE_8_1_PROBE_RECORDS,
    max_line_bytes: int = config.PHASE_8_1_MAX_LINE_BYTES,
) -> dict[str, Any]:
    """Read the first few members and **record** the layout and schema they hold.

    The return value goes verbatim into `source.json`, and `layout_of` turns it back into
    the `ArchiveLayout` the parser dispatches on. Nothing about the dump is hardcoded: the
    candidate field names in `config` are a resolution order, and an archive whose records
    resolve against none of them is a refusal naming the keys that were actually seen.
    """
    probed_members: list[str] = []
    observed_keys: set[str] = set()
    parsed: list[Mapping[str, Any]] = []
    counts = MemberCounts()

    for name, payload in iter_members(path, counts=counts):
        probed_members.append(name)
        for number, line in enumerate(payload.splitlines(), start=1):
            if not line.strip():
                continue
            # SEC-033: the same per-line ceiling `iter_records` enforces, applied before the
            # parser sees the line. The member ceiling alone let a single line of up to that
            # size reach `json.loads` here.
            if len(line) > max_line_bytes:
                raise FullWikiError(
                    f"{name} line {number} is {len(line)} bytes, past the {max_line_bytes} "
                    "byte ceiling"
                )
            record = json.loads(line)
            if not isinstance(record, dict):
                raise FullWikiError(
                    f"member {name} holds a JSON {type(record).__name__} where a record object "
                    "was expected"
                )
            observed_keys.update(str(key) for key in record)
            parsed.append(record)
            if len(parsed) >= records:
                break
        if len(probed_members) >= members or len(parsed) >= records:
            break

    if not probed_members:
        raise FullWikiError(
            f"{Path(path).name} holds no regular member to probe "
            f"({counts.skipped} non-regular entries skipped)"
        )
    if not parsed:
        raise FullWikiError(f"{Path(path).name} holds no JSON record in its first members")

    # Re-read the first member's raw bytes to decide the nested compression: the streamer
    # has already inflated what it yielded, so the fact has to come from the magic itself.
    compression = _probe_compression(path)

    keys = sorted(observed_keys)
    title_field = _resolve_field(keys, config.PHASE_8_1_TITLE_FIELDS)
    sentences_field = _resolve_field(keys, config.PHASE_8_1_SENTENCES_FIELDS)
    if title_field is None or sentences_field is None:
        raise FullWikiError(
            "the records in this archive resolve against none of the declared field names; "
            f"observed keys {keys}. Record the real schema in the deviation before parsing it"
        )
    page_id_field = _resolve_field(keys, config.PHASE_8_1_PAGE_ID_FIELDS)

    leading = sum(1 for record in parsed if _repeats_title(record, title_field, sentences_field))
    return {
        "member_compression": compression,
        "title_field": title_field,
        "sentences_field": sentences_field,
        "page_id_field": page_id_field,
        "leading_title_sentence": leading == len(parsed),
        "leading_title_matches": leading,
        "probed_members": probed_members,
        "probed_records": len(parsed),
        "observed_keys": keys,
        "candidates": {
            "title": list(config.PHASE_8_1_TITLE_FIELDS),
            "sentences": list(config.PHASE_8_1_SENTENCES_FIELDS),
            "page_id": list(config.PHASE_8_1_PAGE_ID_FIELDS),
        },
        "basis": (
            "layout and record schema read off the archive at S1; the parser dispatches on "
            "this record rather than on an assumption about the release"
        ),
    }


def _probe_compression(path: Path | str) -> str:
    """Whether the first regular member's own bytes carry the bz2 magic."""
    with tarfile.open(Path(path), "r|bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            return "bz2" if handle.read(len(BZ2_MAGIC)).startswith(BZ2_MAGIC) else "none"
    raise FullWikiError(f"{Path(path).name} holds no regular member to probe")


def _repeats_title(record: Mapping[str, Any], title_field: str, sentences_field: str) -> bool:
    title = record.get(title_field)
    sentences = record.get(sentences_field)
    if not isinstance(title, str) or not isinstance(sentences, list) or not sentences:
        return False
    first = sentences[0]
    return isinstance(first, str) and first.strip() == title.strip()


def layout_of(probe: Mapping[str, Any]) -> ArchiveLayout:
    """Turn a recorded probe back into the layout the parser reads."""
    try:
        return ArchiveLayout(
            member_compression=str(probe["member_compression"]),
            title_field=str(probe["title_field"]),
            sentences_field=str(probe["sentences_field"]),
            page_id_field=(
                None if probe.get("page_id_field") is None else str(probe["page_id_field"])
            ),
            leading_title_sentence=bool(probe["leading_title_sentence"]),
        )
    except KeyError as missing:
        raise FullWikiError(f"the recorded layout lacks {missing}; re-run the probe") from missing


def iter_records(
    path: Path | str,
    layout: ArchiveLayout,
    *,
    max_member_bytes: int = config.PHASE_8_1_MAX_MEMBER_BYTES,
    max_line_bytes: int = config.PHASE_8_1_MAX_LINE_BYTES,
    max_records: int = config.PHASE_8_1_MAX_TOTAL_RECORDS,
) -> Iterator[FullWikiRecord]:
    """Stream every official paragraph, in archive order, under the recorded layout.

    Deterministic and repeatable: the same archive and the same layout yield the same
    records in the same order, which is what lets the selection be frozen from two passes
    over the stream instead of from a sorted copy of five million records in memory.
    """
    seen = 0
    for name, payload in iter_members(path, max_member_bytes=max_member_bytes):
        line_number = 0
        for raw in payload.splitlines():
            if not raw.strip():
                continue
            line_number += 1
            if len(raw) > max_line_bytes:
                raise FullWikiError(
                    f"{name} line {line_number} is {len(raw)} bytes, past the {max_line_bytes} "
                    "byte ceiling"
                )
            seen += 1
            if seen > max_records:
                raise FullWikiError(
                    f"the archive holds more than the {max_records} record ceiling; refusing "
                    "to read further"
                )
            yield _record_of(json.loads(raw), layout, name, line_number)


def _record_of(
    payload: Mapping[str, Any], layout: ArchiveLayout, member: str, line: int
) -> FullWikiRecord:
    title = payload.get(layout.title_field)
    sentences = payload.get(layout.sentences_field)
    if not isinstance(title, str) or not isinstance(sentences, list):
        raise FullWikiError(
            f"{member} line {line} lacks a string {layout.title_field!r} and a list "
            f"{layout.sentences_field!r}; the recorded layout does not describe this record"
        )
    if any(not isinstance(sentence, str) for sentence in sentences):
        raise FullWikiError(f"{member} line {line} holds a non-string sentence")
    # Applying the recorded fact, not a belief about the release: when the first element
    # repeats the title it is not a sentence, and keeping it would make every official
    # plaintext differ from its C19 counterpart.
    if layout.leading_title_sentence and sentences and sentences[0].strip() == title.strip():
        sentences = sentences[1:]
    page_id = None
    if layout.page_id_field is not None:
        raw_id = payload.get(layout.page_id_field)
        page_id = None if raw_id is None else str(raw_id)
    return FullWikiRecord(
        title=title,
        sentences=tuple(sentences),
        member=member,
        line=line,
        page_id=page_id,
    )


def plaintext_of(record: FullWikiRecord) -> str:
    """The official plaintext, derived from the recorded schema and nothing else.

    Identical in construction to `IndexingUnit.text`, which is what makes an exact
    reconciliation match against a C19 unit meaningful.
    """
    return " ".join(record.sentences)


def unit_of(record: FullWikiRecord) -> IndexingUnit:
    """The record as an indexing unit, built through the project's own constructors."""
    return IndexingUnit(
        unit_id=unit_id_for(record.title, record.sentences),
        title=record.title,
        sentences=record.sentences,
    )


def describe_source(
    path: Path | str,
    *,
    url: str,
    published_bytes: int | None,
    published_md5: str | None,
) -> dict[str, Any]:
    """The body of `source.json`: identity first, then the observed layout.

    Identity is verified before the probe runs, so a wrong archive is refused without
    being parsed. The deviation's unverified recollection travels beside the measurement
    so a later reader sees both what was claimed and what was read.
    """
    verified = verify_archive(path, published_bytes=published_bytes, published_md5=published_md5)
    counts = MemberCounts()
    # Counted over the whole archive rather than over the probe, so `members` describes
    # the release and not the first three members of it.
    for _name, _payload in iter_members(path, counts=counts):
        pass
    probe = probe_layout(path)
    return {
        **verified,
        "url": url,
        "license": config.PHASE_8_1_SOURCE_LICENSE,
        "page_checked": published_bytes is not None or published_md5 is not None,
        "declared_in_spec_bytes": config.PHASE_8_1_DECLARED_BYTES,
        "declared_in_spec_md5": config.PHASE_8_1_DECLARED_MD5,
        "declared_in_spec_basis": config.PHASE_8_1_DECLARED_BASIS,
        "layout": probe,
        "members": counts.as_dict(),
    }


# --- Phase 9, S2: the whole archive is the corpus -----------------------------------------
#
# Deviation 8.1 selected a nested prefix of the archive; Phase 9 indexes all of it. Every
# record becomes one whole-paragraph unit through `unit_of`, in archive order, with nothing
# selected, trimmed or merged (D2). A byte-identical repeat hashes to the same content id
# and collapses onto the first occurrence, exactly as `build_pool` deduplicates, and the
# count is recorded rather than hidden. A record with no sentence is still a record of the
# release and stays, counted.

CORPUS_NAME = "corpus.jsonl.gz"
CORPUS_MANIFEST = "corpus.json"


def _corpus_line(unit: IndexingUnit) -> str:
    return json.dumps(
        {"unit_id": unit.unit_id, "title": unit.title, "sentences": list(unit.sentences)},
        ensure_ascii=False,
        sort_keys=True,
    )


def _unit_set_hash(unit_ids: Sequence[str]) -> str:
    """The project's corpus identity (`embeddings.cache.unit_set_hash`), without its imports."""
    joined = "\n".join(sorted(unit_ids))
    return hashlib.sha1(joined.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _pinned_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: manifest.get(key) for key in ("n_units", "unit_set_hash", "ordered_unit_digest")}


def write_corpus(
    archive: Path | str,
    directory: Path | str,
    *,
    expected_bytes: int,
    expected_md5: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Stream the verified archive into `corpus.jsonl.gz` and its `corpus.json` manifest.

    Identity comes first and refuses before a single record is parsed. An existing
    `corpus.json` is a pin: a stream that does not reproduce its unit count and digests is
    refused, and the recorded manifest is never replaced. That is what lets the measurement
    host rebuild the corpus from the downloaded archive and prove it is the laptop's.
    """
    archive, directory = Path(archive), Path(directory)
    verified = verify_archive(archive, published_bytes=expected_bytes, published_md5=expected_md5)
    if verified["measured_sha256"] != expected_sha256:
        raise FullWikiError(
            f"{archive.name} has sha256 {verified['measured_sha256']}, not the declared "
            f"{expected_sha256}; refusing to parse it"
        )
    target = directory / CORPUS_MANIFEST
    recorded = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None

    started = time.perf_counter()
    probe = probe_layout(archive)
    layout = layout_of(probe)
    directory.mkdir(parents=True, exist_ok=True)
    corpus_path = directory / CORPUS_NAME
    temporary = corpus_path.with_name(corpus_path.name + ".tmp")

    seen: set[str] = set()
    unit_ids: list[str] = []
    records_read = empty = text_bytes = 0
    # mtime=0 so the same stream always compresses to the same bytes.
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        for record in iter_records(archive, layout):
            records_read += 1
            unit = unit_of(record)
            if unit.unit_id in seen:
                continue
            seen.add(unit.unit_id)
            unit_ids.append(unit.unit_id)
            empty += 0 if unit.sentences else 1
            text_bytes += len(unit.indexable_text.encode("utf-8"))
            packed.write((_corpus_line(unit) + "\n").encode("utf-8"))

    manifest: dict[str, Any] = {
        "phase": 9,
        "archive": archive.name,
        "archive_bytes": verified["measured_bytes"],
        "archive_md5": verified["measured_md5"],
        "archive_sha256": verified["measured_sha256"],
        "layout": probe,
        "layout_digest": digest_of(json.dumps(probe, sort_keys=True, ensure_ascii=True)),
        "n_units": len(unit_ids),
        "records_read": records_read,
        "duplicate_records_collapsed": records_read - len(unit_ids),
        "empty_text_units": empty,
        "indexable_text_bytes": text_bytes,
        "unit_set_hash": _unit_set_hash(unit_ids),
        "ordered_unit_digest": digest_of(*unit_ids),
        "corpus_file": CORPUS_NAME,
        "corpus_file_bytes": temporary.stat().st_size,
        "unit_contract": (
            'indexable_text = f"{title}. {plaintext}", plaintext = " ".join(sentences)'
        ),
        "stream_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if recorded is not None and _pinned_fields(recorded) != _pinned_fields(manifest):
        temporary.unlink(missing_ok=True)
        raise FullWikiError(
            f"{target} already records {_pinned_fields(recorded)}, and this stream gives "
            f"{_pinned_fields(manifest)}; a recorded corpus is never replaced"
        )
    temporary.replace(corpus_path)
    if recorded is not None:
        return dict(recorded)
    write_text_atomic(target, json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return manifest


def load_corpus(directory: Path | str) -> list[IndexingUnit]:
    """The corpus in archive order, every id re-derived from its content and the order proved.

    A unit whose id does not hash to its text, or an order whose digest is not the recorded
    one, is a refusal: every downstream vector, token count and entity row is aligned to it.
    """
    directory = Path(directory)
    target = directory / CORPUS_MANIFEST
    if not target.exists():
        raise FullWikiError(f"{target} does not exist: run 'fullwiki-corpus' first")
    manifest = json.loads(target.read_text(encoding="utf-8"))
    units: list[IndexingUnit] = []
    with gzip.open(directory / str(manifest["corpus_file"]), "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            entry = json.loads(line)
            title = str(entry["title"])
            sentences = tuple(str(sentence) for sentence in entry["sentences"])
            unit_id = unit_id_for(title, sentences)
            if unit_id != entry["unit_id"]:
                raise FullWikiError(
                    f"{CORPUS_NAME} line {number}: {entry['unit_id']} does not hash to its "
                    f"content ({unit_id}); the corpus has been modified"
                )
            units.append(IndexingUnit(unit_id=unit_id, title=title, sentences=sentences))
    if digest_of(*(unit.unit_id for unit in units)) != manifest["ordered_unit_digest"]:
        raise FullWikiError(
            f"{CORPUS_NAME} is not in the recorded order: its ordered digest differs from "
            f"{target.name}"
        )
    return units
