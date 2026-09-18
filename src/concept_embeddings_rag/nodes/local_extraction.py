"""Reading entities out of every paragraph with a local extractor (Phase 7, S2).

Phase 5 read the pool with a frontier generative model and paid 25.98 USD for 19,366
paragraphs. This module does the same job with a model that runs on the machine, so
that Phase 7 can measure how much of the Entity Hop gain survives a cheap extractor.

What it is *not*: a second extraction framework. The record it produces is the Phase 5
`ExtractionRecord`, unchanged, so `build_node_index`, `EntityHopStage` and the whole
evaluation path downstream of it need no edit at all. The only genuinely new thing here
is the conversion from spans to a record, and the artifact that carries a local run's
identity, timing and throughput.

The conversion is decision D6 of the phase plan, and nothing about it is decided later:

1. The extractor sees **only** `unit.indexable_text` - title plus paragraph, what both
   retrievers index. A question, an answer, a gold annotation or a split cannot reach it
   by construction, exactly as in Phase 5.
2. Every span's **surface text** is taken verbatim, stripped of surrounding whitespace.
   No lemmatization, no casing change, no canonicalization: `normalization-v1` is this
   project's only normalizer and it runs later, inside `build_node_index`.
3. Forms are **deduplicated by exact surface**, first occurrence kept. A tagger emits one
   span per *mention*; Claude was asked for one item per *distinct name*, and this is
   what makes the two comparable.
4. **Bounds, then failure, never repair.** More than `MAX_NODES_PER_TYPE` forms, or one
   longer than `MAX_NODE_CHARS`, makes the record `failed` with reason `bounds` - the
   same code the eight Phase 5 failures carry. Nothing is truncated to fit.
5. A paragraph longer than the model's reading window is split at **its own sentence
   boundaries** into the fewest consecutive windows that each fit, and the forms of its
   windows are unioned. The *indexing* unit is untouched: only the extractor's reading
   window is split, and on this corpus the rule reaches 131 of 19,366 units.

`concepts` is always empty: Phase 5's gate returned NAMES ONLY, and Phase 7 changes the
extractor rather than the representation. Token counts are zero because no local run is
billed, and `prompt_digest` carries the candidate's **configuration digest** - that field
has always meant "which configuration produced this record", and a local extractor has no
prompt.

Nothing here reads a question, a split or a gold annotation, and no node form is ever
printed: entity forms are corpus text and have no place in a log.
"""

import gzip
import hashlib
import json
import os
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import IndexingUnit
from concept_embeddings_rag.nodes.extraction import FAILED, OK, ExtractionRecord

# One line per paragraph, `unit_id` sorted, in the Phase 5 line shape; and the manifest
# that says which extractor, on which hardware, in how long, produced them.
RECORDS_NAME = "records.jsonl.gz"
MANIFEST_NAME = "extraction.json"

# The one failure reason a local extractor can produce. A model that returns nothing for
# a paragraph has not failed - it has found no name, which is a measurement.
BOUNDS = "bounds"

DETERMINISM_NOTE = (
    "Both candidates are deterministic at inference and the seed is set anyway, but float "
    "non-associativity across batch sizes and devices can move a span that sits exactly on "
    "the threshold. As in Phase 5, this artifact is the record of what was measured; "
    "everything downstream of it is deterministic."
)

PROJECTION_METHOD = "linear from measured throughput over {n} paragraphs"

# Weight formats that execute code when they are read. None of them is ever downloaded
# (config.GLINER_ALLOW_PATTERNS), and the loader refuses a directory holding one anyway:
# an allow-list is a rule, and a check is the proof the rule held.
PICKLE_SUFFIXES: tuple[str, ...] = (".bin", ".pt", ".pth", ".pkl", ".ckpt", ".h5", ".ot")

# What a span extractor is: one list of surface forms per text handed to it, in order.
SpanExtractor = Callable[[Sequence[str]], list[list[str]]]


class LocalExtractionError(Exception):
    """A local extraction artifact, extractor or run is not what it claims to be."""


@dataclass(frozen=True, eq=False)
class LocalExtractor:
    """One candidate, fully identified, plus the callable that reads a batch of texts.

    A carrier, not a hierarchy: the two adapters below are functions that fill it in, and
    a third candidate would be a third function rather than a subclass. `configuration`
    is what the digest is taken over, so a record can always be traced to the exact model,
    revision, label set and parameters that produced it.
    """

    extractor_id: str
    model: str
    revision: str | None
    labels: tuple[str, ...]
    parameters: dict[str, Any]
    library_versions: dict[str, str]
    spans: SpanExtractor
    max_tokens: int | None = None
    count_tokens: Callable[[str], int] | None = None
    window_rule: str = "the whole paragraph in one pass; the model has no reading window"
    hardware_device: str = "cpu"
    _digest: list[str] = field(default_factory=list, repr=False, compare=False)

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "extractor_id": self.extractor_id,
            "model": self.model,
            "revision": self.revision,
            "labels": list(self.labels),
            "parameters": dict(self.parameters),
            "library_versions": dict(self.library_versions),
            "window": {"max_tokens": self.max_tokens, "rule": self.window_rule},
        }

    @property
    def configuration_digest(self) -> str:
        """sha256[:16] over the canonical configuration: the local analogue of a prompt."""
        if not self._digest:
            canonical = json.dumps(self.configuration, sort_keys=True, ensure_ascii=True)
            self._digest.append(hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16])
        return self._digest[0]


# --- The conversion: spans to a record (D6) --------------------------------------------


def windows_for(unit: IndexingUnit, extractor: LocalExtractor) -> list[str]:
    """The texts one paragraph is read as: one, or the fewest sentence windows that fit.

    Greedy packing over consecutive segments is optimal here, because the segments may not
    be reordered: the title and then the sentences, in the paragraph's own order, joined
    exactly as `indexable_text` joins them. A single sentence longer than the window still
    becomes its own window - the model truncates it internally rather than the rule
    splitting a sentence, which is what keeps the unit whole in the sense that matters.
    """
    if extractor.max_tokens is None or extractor.count_tokens is None:
        return [unit.indexable_text]
    if extractor.count_tokens(unit.indexable_text) <= extractor.max_tokens:
        return [unit.indexable_text]

    segments = [f"{unit.title}.", *unit.sentences]
    windows: list[str] = []
    current: list[str] = []
    for segment in segments:
        candidate = [*current, segment]
        if current and extractor.count_tokens(" ".join(candidate)) > extractor.max_tokens:
            windows.append(" ".join(current))
            current = [segment]
        else:
            current = candidate
    if current:
        windows.append(" ".join(current))
    return windows if len(windows) > 1 else [unit.indexable_text]


def record_from_forms(
    unit_id: str,
    forms: Sequence[str],
    *,
    model: str,
    configuration_digest: str,
) -> ExtractionRecord:
    """One paragraph's spans as an `ExtractionRecord`: dedupe, then bounds, never repair."""

    def record(status: str, entities: tuple[str, ...], failure: str | None) -> ExtractionRecord:
        return ExtractionRecord(
            unit_id=unit_id,
            model=model,
            prompt_digest=configuration_digest,
            status=status,
            entities=entities,
            concepts=(),
            failure=failure,
            input_tokens=0,
            output_tokens=0,
        )

    seen: set[str] = set()
    entities: list[str] = []
    for form in forms:
        surface = form.strip()
        if not surface or surface in seen:
            continue
        seen.add(surface)
        entities.append(surface)

    too_many = len(entities) > config.MAX_NODES_PER_TYPE
    too_long = any(len(entity) > config.MAX_NODE_CHARS for entity in entities)
    if too_many or too_long:
        return record(FAILED, (), BOUNDS)
    return record(OK, tuple(entities), None)


# --- The pass ---------------------------------------------------------------------------


def run_extraction(
    units: Sequence[IndexingUnit],
    extractor: LocalExtractor,
    *,
    chunk_size: int = 64,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[ExtractionRecord], float]:
    """Read every paragraph once, in `unit_id` order, and time the pass (D8).

    The clock covers the whole pass and nothing else: windowing, inference and the
    conversion to records. Model download and model load happen before it starts and are
    reported separately, because they are paid once whatever the corpus size and would
    flatter a throughput measured over 19,366 paragraphs.
    """
    if not units:
        raise LocalExtractionError("a local extraction needs paragraphs; none were given")
    ordered = sorted(units, key=lambda unit: unit.unit_id)

    started = time.perf_counter()
    texts: list[str] = []
    owners: list[int] = []
    for position, unit in enumerate(ordered):
        for window in windows_for(unit, extractor):
            texts.append(window)
            owners.append(position)

    forms: list[list[str]] = [[] for _ in ordered]
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        answered = extractor.spans(chunk)
        if len(answered) != len(chunk):
            raise LocalExtractionError(
                f"the extractor answered {len(answered)} of {len(chunk)} texts; a record may "
                "not be built from a batch whose answers cannot be matched to their paragraphs"
            )
        for offset, spans in enumerate(answered):
            forms[owners[start + offset]].extend(spans)
        if on_progress is not None:
            on_progress(min(start + chunk_size, len(texts)), len(texts))

    records = [
        record_from_forms(
            unit.unit_id,
            forms[position],
            model=extractor.model,
            configuration_digest=extractor.configuration_digest,
        )
        for position, unit in enumerate(ordered)
    ]
    return records, time.perf_counter() - started


# --- The artifact -----------------------------------------------------------------------


def record_line(record: ExtractionRecord) -> str:
    """The Phase 5 line shape, so one reader can read either extraction's archive."""
    return json.dumps(
        {
            "unit_id": record.unit_id,
            "status": record.status,
            "entities": list(record.entities),
            "concepts": list(record.concepts),
            "failure": record.failure,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def records_text(records: Sequence[ExtractionRecord]) -> str:
    return "".join(record_line(record) + "\n" for record in sorted(records, key=_by_unit))


def _by_unit(record: ExtractionRecord) -> str:
    return record.unit_id


def digest_of_records(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ram_bytes() -> int | None:
    """Total physical memory, best effort. A figure that cannot be read is declared, never
    guessed: `null` in the manifest says the machine did not report it."""
    try:
        if os.name == "nt":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return int(status.ullTotalPhys)
        sysconf = os.sysconf  # type: ignore[attr-defined]
        return int(sysconf("SC_PAGE_SIZE") * sysconf("SC_PHYS_PAGES"))
    except Exception:
        return None


def hardware_block(device: str = "cpu", gpu_name: str | None = None) -> dict[str, Any]:
    """The machine a throughput was measured on, as measured rather than as assumed."""
    block: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes(),
        "device": device,
        "gpu": gpu_name,
    }
    try:
        import torch

        block["torch_threads"] = int(torch.get_num_threads())
        block["torch_version"] = str(torch.__version__)
    except Exception:
        block["torch_threads"] = None
        block["torch_version"] = None
    return block


def projection(
    n_units: int, paragraphs_per_second: float, *, hourly_rate_usd: float = 0.0
) -> dict[str, Any]:
    """D9: linear from the measured throughput, with the arithmetic in the artifact.

    No efficiency factor, no assumed speed-up and no invented hardware. `usd` is zero on
    owned hardware and the real rate when compute is rented; either way the figure is
    labelled a projection and never reported as a measurement.
    """
    if paragraphs_per_second <= 0.0:
        raise LocalExtractionError(
            f"a projection needs a positive throughput, not {paragraphs_per_second}"
        )
    hours = config.PROJECTION_PARAGRAPHS / paragraphs_per_second / 3600.0
    return {
        "paragraphs": config.PROJECTION_PARAGRAPHS,
        "paragraphs_per_second": paragraphs_per_second,
        "hours": hours,
        "hourly_rate_usd": hourly_rate_usd,
        "usd": hours * hourly_rate_usd,
        "method": PROJECTION_METHOD.format(n=n_units),
    }


def build_manifest(
    records: Sequence[ExtractionRecord],
    extractor: LocalExtractor,
    *,
    seconds: float,
    hardware: Mapping[str, Any],
    usd: float = 0.0,
    hourly_rate_usd: float = 0.0,
    model_load_seconds: float | None = None,
) -> dict[str, Any]:
    """Everything a later reader needs to answer: which extractor produced this index?"""
    text = records_text(records)
    failures: dict[str, int] = {}
    for record in records:
        if record.status != OK and record.failure is not None:
            failures[record.failure] = failures.get(record.failure, 0) + 1
    n_units = len(records)
    per_second = n_units / seconds if seconds > 0 else 0.0
    return {
        "extractor_id": extractor.extractor_id,
        "model": extractor.model,
        "revision": extractor.revision,
        "labels": list(extractor.labels),
        "parameters": dict(extractor.parameters),
        "configuration": extractor.configuration,
        "configuration_digest": extractor.configuration_digest,
        "library_versions": dict(extractor.library_versions),
        "hardware": dict(hardware),
        "n_units": n_units,
        "entities": sum(len(record.entities) for record in records),
        "units_without_entity": sum(1 for record in records if not record.entities),
        "failures": dict(sorted(failures.items())),
        "failure_rate": sum(failures.values()) / n_units if n_units else 0.0,
        "seconds": seconds,
        "model_load_seconds": model_load_seconds,
        "paragraphs_per_second": per_second,
        "usd": usd,
        "projection_5m": projection(n_units, per_second, hourly_rate_usd=hourly_rate_usd),
        "records": RECORDS_NAME,
        "digest": digest_of_records(text),
        "normalization_note": (
            "forms are verbatim surfaces; normalization-v1 runs later, in build_node_index"
        ),
        "determinism_note": DETERMINISM_NOTE,
    }


def write_extraction(
    records: Sequence[ExtractionRecord], manifest: Mapping[str, Any], directory: Path | str
) -> Path:
    """Write the archive and its manifest. Regenerable, so an identical rewrite is harmless."""
    text = records_text(records)
    if manifest["digest"] != digest_of_records(text):
        raise LocalExtractionError(
            "the manifest's digest is not the digest of the records it is written beside"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    archive = directory / RECORDS_NAME
    temporary = archive.with_name(archive.name + ".tmp")
    # mtime=0 so the same records always compress to the same bytes.
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(text.encode("utf-8"))
    temporary.replace(archive)

    path = directory / MANIFEST_NAME
    write_text_atomic(path, json.dumps(dict(manifest), indent=2, sort_keys=True))
    return path


def load_manifest(directory: Path | str) -> dict[str, Any]:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise LocalExtractionError(
            f"no local extraction in {directory}: run `cer cheap-extract` for this candidate first"
        )
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def load_extraction(
    directory: Path | str, *, expected_unit_ids: Sequence[str] | None = None
) -> tuple[dict[str, ExtractionRecord], dict[str, Any]]:
    """Read the archive back, verified against its digest and, if asked, against the pool."""
    directory = Path(directory)
    manifest = load_manifest(directory)
    archive = directory / str(manifest.get("records", RECORDS_NAME))
    if not archive.exists():
        raise LocalExtractionError(f"{archive.name} is missing beside its manifest in {directory}")

    text = gzip.decompress(archive.read_bytes()).decode("utf-8")
    if digest_of_records(text) != manifest["digest"]:
        raise LocalExtractionError(
            f"{archive.name} does not match the digest its manifest records; it has been modified"
        )

    model = str(manifest["model"])
    configuration_digest = str(manifest["configuration_digest"])
    records: dict[str, ExtractionRecord] = {}
    for line in text.splitlines():
        entry = json.loads(line)
        records[str(entry["unit_id"])] = ExtractionRecord(
            unit_id=str(entry["unit_id"]),
            model=model,
            prompt_digest=configuration_digest,
            status=str(entry["status"]),
            entities=tuple(entry["entities"]),
            concepts=tuple(entry["concepts"]),
            failure=entry["failure"],
            input_tokens=int(entry["input_tokens"]),
            output_tokens=int(entry["output_tokens"]),
        )

    if expected_unit_ids is not None:
        missing = set(expected_unit_ids) - set(records)
        if missing:
            raise LocalExtractionError(
                f"the extraction does not cover {len(missing)} paragraph(s) of the pool; every "
                "unit needs a record, ok or failed, before nodes can be built from it"
            )
    return records, manifest


# --- The two adapters -------------------------------------------------------------------
# Each is one library call wrapped in the project's declared configuration. Neither is
# imported at module import time, so the test suite and every other stage stay free of
# both libraries.


def _versions(*names: str) -> dict[str, str]:
    import importlib.metadata

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _refuse_executable_weights(directory: Path) -> None:
    """The allow-list says no pickle archive is fetched; this proves none arrived."""
    found = sorted(
        path.name for path in directory.iterdir() if path.suffix.lower() in PICKLE_SUFFIXES
    )
    if found:
        raise LocalExtractionError(
            f"{directory} holds {found}, which deserialize executable content; this phase loads "
            "safetensors only and does not download or read those formats"
        )


def download_gliner(directory: Path | str) -> Path:
    """Fetch the pinned checkpoint and the pinned tokenizer, and nothing else (D14).

    Both repositories also ship `pytorch_model.bin`, and the tokenizer's ships `tf_model.h5`
    and `rust_model.ot`; none is in an allow-list, so no pickle archive reaches the disk and
    none can be deserialized. They land in one directory because that is where GLiNER looks
    for a tokenizer before falling back to the Hub, which keeps the whole load local and
    pinned rather than resolving `main` at run time.
    """
    from huggingface_hub import snapshot_download

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        config.GLINER_MODEL,
        revision=config.GLINER_REVISION,
        allow_patterns=list(config.GLINER_ALLOW_PATTERNS),
        local_dir=str(directory),
    )
    snapshot_download(
        config.GLINER_TOKENIZER_MODEL,
        revision=config.GLINER_TOKENIZER_REVISION,
        allow_patterns=list(config.GLINER_TOKENIZER_ALLOW_PATTERNS),
        local_dir=str(directory),
    )
    _refuse_executable_weights(directory)
    return directory


def gliner_extractor(directory: Path | str) -> tuple[LocalExtractor, float]:
    """Candidate B: the pinned GLiNER checkpoint, loaded from safetensors, on this machine."""
    import torch
    from gliner import GLiNER

    model_dir = download_gliner(directory)
    started = time.perf_counter()
    torch.manual_seed(config.DEFAULT_SEED)
    model = GLiNER.from_pretrained(str(model_dir))
    model.eval()
    load_seconds = time.perf_counter() - started

    labels = list(config.GLINER_LABELS)
    tokenizer = model.data_processor.transformer_tokenizer

    def spans(texts: Sequence[str]) -> list[list[str]]:
        with torch.inference_mode():
            answered = model.inference(
                list(texts),
                labels,
                flat_ner=config.GLINER_FLAT_NER,
                threshold=config.GLINER_THRESHOLD,
                multi_label=config.GLINER_MULTI_LABEL,
                batch_size=config.GLINER_BATCH_SIZE,
            )
        return [[str(entity["text"]) for entity in text_spans] for text_spans in answered]

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    extractor = LocalExtractor(
        extractor_id=config.GLINER_EXTRACTOR,
        model=config.GLINER_MODEL,
        revision=config.GLINER_REVISION,
        labels=config.GLINER_LABELS,
        parameters={
            "threshold": config.GLINER_THRESHOLD,
            "flat_ner": config.GLINER_FLAT_NER,
            "multi_label": config.GLINER_MULTI_LABEL,
            "batch_size": config.GLINER_BATCH_SIZE,
            "max_len": config.GLINER_MAX_LEN,
            "max_width": config.GLINER_MAX_WIDTH,
            "weights": "model.safetensors",
            "tokenizer_model": config.GLINER_TOKENIZER_MODEL,
            "tokenizer_revision": config.GLINER_TOKENIZER_REVISION,
            "seed": config.DEFAULT_SEED,
        },
        library_versions=_versions(
            "gliner", "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors"
        ),
        spans=spans,
        max_tokens=config.GLINER_MAX_LEN,
        count_tokens=count_tokens,
        window_rule=(
            "a paragraph over max_tokens sub-word tokens is split at its own sentence "
            "boundaries into the fewest consecutive windows that each fit; forms are unioned"
        ),
    )
    return extractor, load_seconds


def spacy_extractor() -> tuple[LocalExtractor, float]:
    """Candidate C: the pinned `en_core_web_sm` pipeline, NER components only.

    The pipeline arrives as an installed Python package rather than as a weight file, which
    is the surface the targeted security review records: installing it is executing its
    publisher's code, so it is installed from the official release wheel and pinned.
    """
    import spacy

    started = time.perf_counter()
    nlp = spacy.load(config.SPACY_MODEL, exclude=list(config.SPACY_EXCLUDE))
    load_seconds = time.perf_counter() - started
    kept = set(config.SPACY_LABELS)

    def spans(texts: Sequence[str]) -> list[list[str]]:
        documents = nlp.pipe(
            list(texts),
            batch_size=config.SPACY_BATCH_SIZE,
            n_process=config.SPACY_PROCESSES,
        )
        return [
            [str(entity.text) for entity in document.ents if entity.label_ in kept]
            for document in documents
        ]

    extractor = LocalExtractor(
        extractor_id=config.SPACY_EXTRACTOR,
        model=config.SPACY_MODEL,
        revision=config.SPACY_MODEL_VERSION,
        labels=config.SPACY_LABELS,
        parameters={
            "pipeline_version": config.SPACY_MODEL_VERSION,
            "exclude": list(config.SPACY_EXCLUDE),
            "batch_size": config.SPACY_BATCH_SIZE,
            "n_process": config.SPACY_PROCESSES,
            "dropped_labels": list(config.SPACY_DROPPED_LABELS),
            "wheel": config.SPACY_MODEL_WHEEL_URL,
        },
        library_versions=_versions("spacy", "thinc", "en-core-web-sm"),
        spans=spans,
    )
    return extractor, load_seconds


def build_extractor(extractor_id: str, directory: Path | str) -> tuple[LocalExtractor, float]:
    """The one place a candidate id becomes a loaded extractor."""
    if extractor_id == config.GLINER_EXTRACTOR:
        return gliner_extractor(directory)
    if extractor_id == config.SPACY_EXTRACTOR:
        return spacy_extractor()
    raise LocalExtractionError(
        f"unknown extractor {extractor_id!r}; this phase has exactly two candidates, "
        f"{list(config.PHASE_7_EXTRACTORS)}"
    )
