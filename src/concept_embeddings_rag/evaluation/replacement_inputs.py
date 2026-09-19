"""Every input Phase 6 inherits, loaded and verified before anything is measured (D2, D3).

HU-1: the run refuses to proceed if the pool, the split, the tokenizer, the embeddings, the node
index, the extraction, the Phase 5 reference artifacts or `selection.json` do not verify. Each
check below compares what is on disk with a pinned value or a recomputation, and raises
`ReplacementInputError` with both sides when they differ:

- **Pool**: `load_pool` re-derives every unit id; the set hash must equal the manifest's and
  the pin, and the unit count the manifest's and the expected one.
- **Split**: the Phase 1 rule re-applied to the pool's own qids must reproduce every recorded
  label and the manifest's sizes. Qids and labels only: nothing is evaluated.
- **Tokens**: `token_counts.json` carries no digest and no tokenizer name, so the only real
  check is a recount with the pinned tokenizer, loaded from the local cache only.
- **Embeddings**: the corpus cache aligned to the pool, and the **dev** question cache verified
  by its sidecar. The test question vectors are not touched here.
- **Selection, pilot, hop run, traces**: each loaded by its own verifying loader, then its digest
  compared with the pin.
- **Extraction and node index**: the summary's digest is compared with the pin **before** the
  archive is read, and the node index path is derived from the pin, never from the summary, so
  a summary digest shaped like a path reaches no file (SEC-022 is not reachable from here;
  SEC-020 to SEC-023 stay open).
- **Counts** the spec quotes, each refused with both numbers.

The four historical result files carry no digest. They are opened by the **identity reader**
alone: name, sha256 of the raw bytes, and from the parsed document `system`, `split`,
`created_at` and the HU-2 configuration keys. It returns nothing else and keeps no document.
The **dev figures reader** returns the `metrics` and `cost` of a dev file and refuses any other
split. This module holds no reader for the figures of a test file: that reader belongs to the
test stage (T18), called once, after the freeze.
"""

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config, decision_parameters
from concept_embeddings_rag.corpus.manifest import CorpusManifest, ManifestError
from concept_embeddings_rag.corpus.pool import (
    GoldResolutionError,
    IndexingUnit,
    PoolIntegrityError,
    Question,
    load_pool,
)
from concept_embeddings_rag.corpus.split import split_questions
from concept_embeddings_rag.embeddings.backend import EmbeddingBackend
from concept_embeddings_rag.embeddings.cache import (
    CacheAlignmentError,
    CachedQueryBackend,
    EmbeddingCache,
    build_query_backend,
    cache_key,
    question_cache_key,
    question_set_hash,
    unit_set_hash,
)
from concept_embeddings_rag.evaluation.budget import TokenCounter
from concept_embeddings_rag.evaluation.navigation import (
    NavigationError,
    load_hop_run,
    load_traces,
)
from concept_embeddings_rag.evaluation.pilot import (
    PilotError,
    PilotSet,
    check_pilot_against,
    load_pilot,
    pilot_digest,
)
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    SelectionError,
    SelectionReport,
    load_selection,
    selection_digest,
)
from concept_embeddings_rag.nodes.extraction import (
    ExtractionError,
    ExtractionRecord,
    load_extraction,
    load_extraction_summary,
)
from concept_embeddings_rag.nodes.index import NodeIndex, NodeIndexError, load_node_index

SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX16 = re.compile(r"^[0-9a-f]{16}$")
POOL_NAME = "pool.json"
MANIFEST_NAME = "manifest.json"
TOKENS_NAME = "token_counts.json"
SYSTEMS_WITH_HISTORY: tuple[str, ...] = ("dense", "hybrid-bm25")
HYBRID_WITH_HISTORY = "hybrid-bm25"

TokenCountFunction = Callable[[Sequence[IndexingUnit]], Mapping[str, int]]


class ReplacementInputError(Exception):
    """An inherited input is missing, modified, or not the one Phase 6 pinned."""


@dataclass(frozen=True)
class InputPaths:
    """Where every inherited input lives. The CLI passes `config`'s paths; tests pass a toy's."""

    data_dir: Path
    cache_dir: Path
    question_cache_dir: Path
    selection_dir: Path
    pilot_dir: Path
    navigation_dir: Path
    extraction_dir: Path
    nodes_dir: Path
    results_dir: Path

    @classmethod
    def from_config(cls) -> "InputPaths":
        return cls(
            data_dir=config.DATA_DIR,
            cache_dir=config.CACHE_DIR,
            question_cache_dir=config.QUESTION_CACHE_DIR,
            selection_dir=config.SELECTION_DIR,
            pilot_dir=config.PILOT_DIR,
            navigation_dir=config.NAVIGATION_DIR,
            extraction_dir=config.EXTRACTION_DIR,
            nodes_dir=config.NODES_DIR,
            results_dir=config.RESULTS_DIR,
        )


@dataclass(frozen=True)
class InputPins:
    """What every inherited input must be. The real run uses `from_config`, nothing else."""

    unit_set_hash: str
    model: str
    revision: str
    tokenizer: str
    seed: int
    top_k: int
    selection_digest: str
    pilot_digest: str
    hop_run_digest: str
    traces_digest: str
    extraction_digest: str
    extraction_prompt_digest: str
    extraction_model: str
    node_index_digest: str
    normalization_version: str
    historical_dev_files: Mapping[str, str]
    historical_test_files: Mapping[str, str]
    n_units: int
    entity_nodes: int
    failed_extractions: int
    units_without_entity_node: int
    pilot_questions: int
    entity_traces: int

    @classmethod
    def from_config(cls) -> "InputPins":
        return cls(
            unit_set_hash=config.PHASE_1_UNIT_SET_HASH,
            model=config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_REVISION,
            tokenizer=config.TOKENIZER_ID,
            seed=config.DEFAULT_SEED,
            top_k=config.EVALUATION_TOP_K,
            selection_digest=config.PINNED_SELECTION_DIGEST,
            pilot_digest=config.PINNED_PILOT_DIGEST,
            hop_run_digest=config.PINNED_HOP_RUN_DIGEST,
            traces_digest=config.PINNED_NAVIGATION_TRACES_DIGEST,
            extraction_digest=config.PINNED_EXTRACTION_DIGEST,
            extraction_prompt_digest=config.PINNED_EXTRACTION_PROMPT_DIGEST,
            extraction_model=config.EXTRACTION_MODEL,
            node_index_digest=config.PINNED_NODE_INDEX_DIGEST,
            normalization_version=config.PINNED_NORMALIZATION_VERSION,
            historical_dev_files=dict(config.HISTORICAL_DEV_RESULT_FILES),
            historical_test_files=dict(config.HISTORICAL_TEST_RESULT_FILES),
            n_units=config.EXPECTED_N_UNITS,
            entity_nodes=config.EXPECTED_ENTITY_NODES,
            failed_extractions=config.EXPECTED_FAILED_EXTRACTIONS,
            units_without_entity_node=config.EXPECTED_UNITS_WITHOUT_ENTITY_NODE,
            pilot_questions=config.EXPECTED_PILOT_QUESTIONS,
            entity_traces=config.EXPECTED_ENTITY_TRACES,
        )


@dataclass(frozen=True)
class HistoricalIdentity:
    """What identifying a historical result file yields. No metric, no cost, no document."""

    name: str
    sha256: str
    system: str
    split: str
    created_at: str
    config: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "system": self.system,
            "split": self.split,
            "created_at": self.created_at,
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class DevFigures:
    """The figures of a historical **dev** result, for the dev reproduction (D12)."""

    identity: HistoricalIdentity
    metrics: dict[str, Any]
    cost: dict[str, Any]


@dataclass(frozen=True)
class VerifiedInputs:
    """Everything Phase 6 inherits, each piece verified. Built only by `load_verified_inputs`."""

    paths: InputPaths
    pins: InputPins
    manifest: CorpusManifest
    units: list[IndexingUnit]
    questions: list[Question]
    unit_ids: list[str]
    pool_hash: str
    split_orders: dict[str, tuple[str, ...]]
    question_set_hashes: dict[str, str]
    token_counts: dict[str, int]
    token_counts_sha256: str
    vectors: np.ndarray
    corpus_cache_key: str
    dev_query_backend: CachedQueryBackend
    dev_question_cache_key: str
    selection: SelectionReport
    selection_digest: str
    pilot: PilotSet
    hop_run: dict[str, Any]
    navigation_traces: list[dict[str, Any]]
    extraction: dict[str, ExtractionRecord]
    extraction_summary: dict[str, Any]
    node_index: NodeIndex
    historical: dict[tuple[str, str], HistoricalIdentity]
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def dev_questions(self) -> list[Question]:
        return [question for question in self.questions if question.split == DEV_SPLIT]

    def identities(self) -> dict[str, Any]:
        """The identities a checks file and the freeze record, as JSON-ready values."""
        return {
            "pool": {
                "unit_set_hash": self.pool_hash,
                "n_units": len(self.unit_ids),
                "manifest_sha256": self.manifest.sha256,
                "manifest_seed": self.manifest.seed,
                "split_sizes": dict(self.manifest.split_sizes),
                "question_set_hashes": dict(self.question_set_hashes),
            },
            "token_counts_sha256": self.token_counts_sha256,
            "embeddings": {
                "model": self.pins.model,
                "revision": self.pins.revision,
                "corpus_cache_key": self.corpus_cache_key,
                "dev_question_cache_key": self.dev_question_cache_key,
            },
            "selection": {
                "digest": self.selection_digest,
                "frozen_at": self.selection.frozen_at,
            },
            "pilot_digest": self.pins.pilot_digest,
            "hop_run_digest": self.pins.hop_run_digest,
            "traces_digest": self.pins.traces_digest,
            "extraction": {
                "digest": self.pins.extraction_digest,
                "prompt_digest": self.pins.extraction_prompt_digest,
                "model": self.pins.extraction_model,
            },
            "node_index": {
                "digest": self.node_index.digest,
                "normalization_version": self.node_index.normalization_version,
            },
            "historical": {
                f"{system}/{split}": identity.as_payload()
                for (system, split), identity in sorted(self.historical.items())
            },
            "counts": dict(self.counts),
        }


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _refuse(what: str, expected: object, observed: object) -> None:
    if expected != observed:
        raise ReplacementInputError(f"{what}: expected {expected!r}, found {observed!r}")


def _check_pins(pins: InputPins) -> None:
    """Every pin that reaches a file path is refused unless it has the shape of a digest."""
    for name in (
        "selection_digest",
        "pilot_digest",
        "hop_run_digest",
        "traces_digest",
        "extraction_digest",
        "node_index_digest",
    ):
        if not SHA256.match(str(getattr(pins, name))):
            raise ReplacementInputError(f"the pinned {name} is not a sha256 digest")
    if not HEX16.match(str(pins.extraction_prompt_digest)):
        raise ReplacementInputError("the pinned extraction prompt digest is not 16 hex characters")


# --- Pool, split, tokens, embeddings ----------------------------------------------------------


def verify_pool(
    data_dir: Path, pins: InputPins
) -> tuple[CorpusManifest, list[IndexingUnit], list[Question], str]:
    pool_path, manifest_path = Path(data_dir) / POOL_NAME, Path(data_dir) / MANIFEST_NAME
    if not pool_path.exists() or not manifest_path.exists():
        raise ReplacementInputError(f"no pool or manifest in {data_dir}: run 'build' first")
    try:
        manifest = CorpusManifest.load(manifest_path)
        units, questions = load_pool(pool_path)
    except (PoolIntegrityError, GoldResolutionError, ManifestError) as error:
        raise ReplacementInputError(f"the pool does not verify: {error}") from error
    pool_hash = unit_set_hash([unit.unit_id for unit in units])
    _refuse("the pool's unit set hash against the manifest", manifest.unit_set_hash, pool_hash)
    _refuse("the pool's unit set hash against the pin", pins.unit_set_hash, pool_hash)
    _refuse("the pool's unit count against the manifest", manifest.n_units, len(units))
    _refuse("the pool's unit count (n_units)", pins.n_units, len(units))
    return manifest, units, questions, pool_hash


def verify_split(
    questions: Sequence[Question], manifest: CorpusManifest
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """The Phase 1 rule, re-applied to the pool's own qids, must reproduce every label."""
    qids = [question.qid for question in questions]
    if len(set(qids)) != len(qids):
        raise ReplacementInputError("the pool repeats a question id; the split cannot be checked")
    n_dev = manifest.split_sizes.get(DEV_SPLIT)
    if n_dev is None:
        raise ReplacementInputError("the manifest records no dev split size")
    rule = split_questions([{"_id": qid} for qid in qids], n_dev=n_dev, seed=manifest.seed)
    expected = {entry["_id"]: split for split, entries in rule.items() for entry in entries}
    wrong = [question.qid for question in questions if expected[question.qid] != question.split]
    if wrong:
        raise ReplacementInputError(
            f"{len(wrong)} question(s) sit on the wrong side of the Phase 1 split rule, e.g. "
            f"{wrong[:3]}; the split does not verify"
        )
    orders: dict[str, tuple[str, ...]] = {}
    for split in sorted({question.split for question in questions}):
        orders[split] = tuple(question.qid for question in questions if question.split == split)
    sizes = {split: len(order) for split, order in orders.items()}
    _refuse("the split sizes against the manifest", dict(manifest.split_sizes), sizes)
    hashes = {split: question_set_hash(list(order)) for split, order in orders.items()}
    return orders, hashes


class _OfflineTokenCounter(TokenCounter):
    """The Phase 1 counter, with the tokenizer loaded from the local cache, never the network."""

    _offline: Any = None

    def _load(self) -> Any:
        if self._offline is None:
            from transformers import AutoTokenizer

            print(f"[INFO] loading tokenizer {self.tokenizer_id} from the local cache only")
            self._offline = AutoTokenizer.from_pretrained(
                self.tokenizer_id, revision=self.revision, local_files_only=True
            )
        return self._offline


def offline_token_counter(
    tokenizer_id: str = config.TOKENIZER_ID, revision: str = config.EMBEDDING_REVISION
) -> TokenCountFunction:
    """`TokenCounter.count_units` with the pinned tokenizer; raises if it is not cached locally."""
    counter = _OfflineTokenCounter(tokenizer_id, revision)

    def count(units: Sequence[IndexingUnit]) -> dict[str, int]:
        counter._load()
        return counter.count_units(units) if units else {}

    return count


def verify_token_counts(
    data_dir: Path, units: Sequence[IndexingUnit], token_counter: TokenCountFunction
) -> tuple[dict[str, int], str]:
    path = Path(data_dir) / TOKENS_NAME
    if not path.exists():
        raise ReplacementInputError(f"no token counts in {data_dir}: run 'embed' first")
    recorded = {str(key): int(value) for key, value in json.loads(path.read_text("utf-8")).items()}
    recount = dict(token_counter(units))
    if recorded != recount:
        differing = sorted(
            unit_id
            for unit_id in set(recorded) | set(recount)
            if recorded.get(unit_id) != recount.get(unit_id)
        )
        raise ReplacementInputError(
            f"the token counts on disk differ from the recount with the pinned tokenizer for "
            f"{len(differing)} unit(s), e.g. {differing[:3]}"
        )
    return recorded, sha256_of(path)


def _sidecar(cache: EmbeddingCache, key: str) -> dict[str, Any]:
    path = cache.sidecar_for(key)
    if not path.exists():
        raise ReplacementInputError(f"the embedding cache {key} has no sidecar")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def verify_corpus_embeddings(
    cache_dir: Path, unit_ids: Sequence[str], pool_hash: str, pins: InputPins
) -> tuple[np.ndarray, str]:
    key = cache_key(pins.model, pins.revision, pool_hash, normalized=True)
    cache = EmbeddingCache(Path(cache_dir))
    try:
        loaded = cache.load(key, expected_unit_ids=unit_ids)
    except CacheAlignmentError as error:
        raise ReplacementInputError(
            f"the corpus embedding cache does not verify: {error}"
        ) from error
    if loaded is None:
        raise ReplacementInputError(f"no corpus embedding cache {key}: run 'embed' first")
    sidecar = _sidecar(cache, key)
    _refuse("the corpus embedding model", pins.model, sidecar.get("model"))
    _refuse(
        "the corpus embedding resolved revision", pins.revision, sidecar.get("resolved_revision")
    )
    vectors, _ids = loaded
    return vectors, key


def build_dev_query_backend(
    question_cache_dir: Path,
    dev: Sequence[Question],
    backend: EmbeddingBackend,
    pins: InputPins,
) -> tuple[CachedQueryBackend, str]:
    """The dev question vectors, from the cache only: a miss is refused, never embedded."""
    _refuse("the embedding backend's model", pins.model, backend.name)
    _refuse("the embedding backend's revision", pins.revision, backend.revision)
    if any(question.split != DEV_SPLIT for question in dev):
        raise ReplacementInputError("only dev questions are embedded for this stage")
    qids = [question.qid for question in dev]
    key = question_cache_key(pins.model, pins.revision, question_set_hash(qids), DEV_SPLIT, True)
    cache = EmbeddingCache(Path(question_cache_dir))
    try:
        cached = cache.load(key, expected_unit_ids=qids)
        if cached is None:
            raise ReplacementInputError(
                f"the dev question embeddings are not cached ({key}); refusing to embed them here"
            )
        query_backend = build_query_backend(dev, backend, cache)
    except CacheAlignmentError as error:
        raise ReplacementInputError(f"the dev question cache does not verify: {error}") from error
    sidecar = _sidecar(cache, key)
    _refuse("the dev question resolved revision", pins.revision, sidecar.get("resolved_revision"))
    return query_backend, key


# --- Selection and Phase 5 --------------------------------------------------------------------


def verify_selection(
    selection_dir: Path, pool_hash: str, pins: InputPins
) -> tuple[SelectionReport, str]:
    try:
        report = load_selection(selection_dir, expected_unit_set_hash=pool_hash)
    except SelectionError as error:
        raise ReplacementInputError(f"the selection does not verify: {error}") from error
    digest = selection_digest(report)
    _refuse("the selection digest against the pin", pins.selection_digest, digest)
    if report.control is None:
        raise ReplacementInputError("the selection carries no control fit")
    _refuse("the selection's control components", ("dense", "bm25"), report.control.components)
    return report, digest


def verify_pilot(
    pilot_dir: Path, pool_hash: str, questions: Sequence[Question], pins: InputPins
) -> PilotSet:
    try:
        pilot = load_pilot(pilot_dir, expected_unit_set_hash=pool_hash)
        check_pilot_against(pilot, questions)
    except PilotError as error:
        raise ReplacementInputError(f"the pilot does not verify: {error}") from error
    _refuse("the pilot digest against the pin", pins.pilot_digest, pilot_digest(pilot))
    _refuse("the pilot's read depth", config.PILOT_READ_DEPTH, pilot.read_depth)
    return pilot


def verify_hop_run(navigation_dir: Path, pins: InputPins) -> dict[str, Any]:
    try:
        hop_run = load_hop_run(navigation_dir)
    except NavigationError as error:
        raise ReplacementInputError(f"the hop run does not verify: {error}") from error
    _refuse("the hop run digest against the pin", pins.hop_run_digest, hop_run.get("digest"))
    provenance = hop_run.get("provenance", {})
    _refuse("the hop run's pilot", pins.pilot_digest, provenance.get("pilot_digest"))
    _refuse("the hop run's node index", pins.node_index_digest, provenance.get("node_index_digest"))
    _refuse("the hop run's extraction", pins.extraction_digest, provenance.get("extraction_digest"))
    _refuse("the hop run's depths", list(config.SECOND_HOP_DEPTHS), provenance.get("depths"))
    return hop_run


def verify_navigation_traces(navigation_dir: Path, pins: InputPins) -> list[dict[str, Any]]:
    try:
        traces = load_traces(navigation_dir, hop_run_digest=pins.hop_run_digest)
    except NavigationError as error:
        raise ReplacementInputError(f"the navigation traces do not verify: {error}") from error
    path = Path(navigation_dir) / "traces.json"
    recorded = json.loads(path.read_text(encoding="utf-8")).get("digest")
    _refuse("the navigation traces digest against the pin", pins.traces_digest, recorded)
    return traces


def verify_extraction(
    extraction_dir: Path, unit_ids: Sequence[str], pins: InputPins
) -> tuple[dict[str, ExtractionRecord], dict[str, Any]]:
    """The summary's digest is compared with the pin before the archive is opened."""
    try:
        summary = load_extraction_summary(
            extraction_dir, prompt_digest=pins.extraction_prompt_digest
        )
    except ExtractionError as error:
        raise ReplacementInputError(f"the extraction summary does not verify: {error}") from error
    _refuse("the extraction digest against the pin", pins.extraction_digest, summary.get("digest"))
    _refuse("the extraction prompt", pins.extraction_prompt_digest, summary.get("prompt_digest"))
    _refuse("the extraction model", pins.extraction_model, summary.get("model"))
    try:
        records = load_extraction(
            extraction_dir, expected_unit_ids=unit_ids, prompt_digest=pins.extraction_prompt_digest
        )
    except ExtractionError as error:
        raise ReplacementInputError(f"the extraction archive does not verify: {error}") from error
    identity = {
        "digest": pins.extraction_digest,
        "prompt_digest": pins.extraction_prompt_digest,
        "model": pins.extraction_model,
    }
    return records, identity


def verify_node_index(nodes_dir: Path, unit_ids: Sequence[str], pins: InputPins) -> NodeIndex:
    """The path is the pinned extraction digest's; the index must carry the pinned digest."""
    try:
        index = load_node_index(
            nodes_dir, extraction_digest=pins.extraction_digest, expected_unit_ids=unit_ids
        )
    except NodeIndexError as error:
        raise ReplacementInputError(f"the node index does not verify: {error}") from error
    _refuse("the node index digest against the pin", pins.node_index_digest, index.digest)
    _refuse("the node index's extraction", pins.extraction_digest, index.extraction_digest)
    _refuse(
        "the node index normalization version",
        pins.normalization_version,
        index.normalization_version,
    )
    return index


def count_inputs(
    unit_ids: Sequence[str],
    index: NodeIndex,
    pilot: PilotSet,
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    entity_columns = index.columns_of(config.ENTITY_HOP_TYPES)
    per_row = np.asarray(index.incidence[:, entity_columns].getnnz(axis=1)).ravel()
    return {
        "n_units": len(unit_ids),
        "entity_nodes": int(entity_columns.size),
        "failed_extractions": int(index.failed_units),
        "units_without_entity_node": int((per_row == 0).sum()),
        "pilot_questions": len(pilot.questions),
        "entity_traces": sum(1 for trace in traces if trace.get("arm") == "entity"),
    }


def check_counts(counts: Mapping[str, int], pins: InputPins) -> None:
    for name, observed in counts.items():
        expected = getattr(pins, name)
        if expected != observed:
            raise ReplacementInputError(
                f"count {name}: the spec expects {expected}, the inputs hold {observed}"
            )


# --- The historical result files ----------------------------------------------------------------


def identity_of(document: Mapping[str, Any], *, name: str, sha256: str) -> HistoricalIdentity:
    """Read only `system`, `split`, `created_at` and the HU-2 configuration keys."""
    recorded_config = document["config"]
    keys = list(config.HISTORICAL_CONFIG_KEYS)
    if document["system"] == HYBRID_WITH_HISTORY:
        keys.extend(config.HISTORICAL_HYBRID_CONFIG_KEYS)
    missing = [key for key in keys if key not in recorded_config]
    if missing:
        raise ReplacementInputError(f"historical file {name} lacks config keys {missing}")
    return HistoricalIdentity(
        name=name,
        sha256=sha256,
        system=str(document["system"]),
        split=str(document["split"]),
        created_at=str(document["created_at"]),
        config={key: recorded_config[key] for key in keys},
    )


def identify_result_file(
    path: Path, *, parse: Callable[[str], Mapping[str, Any]] = json.loads
) -> HistoricalIdentity:
    """The identity reader: the file's name and bytes, and the identity fields of its document."""
    path = Path(path)
    if not path.exists():
        raise ReplacementInputError(f"historical result file {path.name} is not in {path.parent}")
    raw = path.read_bytes()
    return identity_of(
        parse(raw.decode("utf-8")), name=path.name, sha256=hashlib.sha256(raw).hexdigest()
    )


def check_historical_identity(
    identity: HistoricalIdentity,
    *,
    system: str,
    split: str,
    pool_hash: str,
    pins: InputPins,
    selection: SelectionReport,
) -> None:
    """HU-2: every listed configuration value, read from the file, against its reference."""
    _refuse(f"historical file {identity.name}: system", system, identity.system)
    _refuse(f"historical file {identity.name}: split", split, identity.split)
    expected: dict[str, Any] = {
        "unit_set_hash": pool_hash,
        "tokenizer": pins.tokenizer,
        "seed": pins.seed,
        "top_k": pins.top_k,
        "model": pins.model,
        "revision": pins.revision,
    }
    if system == HYBRID_WITH_HISTORY:
        control = selection.control
        if control is None:
            raise ReplacementInputError("the selection carries no control fit")
        expected["fusion_scheme"] = control.winning_scheme
        expected["fusion_weights"] = control.weights
    for key, value in expected.items():
        _refuse(f"historical file {identity.name}: config {key}", value, identity.config.get(key))


def identify_historical_files(
    results_dir: Path,
    *,
    pool_hash: str,
    pins: InputPins,
    selection: SelectionReport,
    parse: Callable[[str], Mapping[str, Any]] = json.loads,
) -> dict[tuple[str, str], HistoricalIdentity]:
    identities: dict[tuple[str, str], HistoricalIdentity] = {}
    held_out = decision_parameters.DECISION_SPLIT
    for split, files in (
        (DEV_SPLIT, pins.historical_dev_files),
        (held_out, pins.historical_test_files),
    ):
        if sorted(files) != sorted(SYSTEMS_WITH_HISTORY):
            raise ReplacementInputError(
                f"the pinned historical files name {sorted(files)}, not {SYSTEMS_WITH_HISTORY}"
            )
        for system in SYSTEMS_WITH_HISTORY:
            identity = identify_result_file(Path(results_dir) / files[system], parse=parse)
            check_historical_identity(
                identity,
                system=system,
                split=split,
                pool_hash=pool_hash,
                pins=pins,
                selection=selection,
            )
            identities[(system, identity.split)] = identity
    return identities


def read_dev_figures(results_dir: Path, identity: HistoricalIdentity) -> DevFigures:
    """The metrics and cost of a historical **dev** file, re-checked against its identity."""
    if identity.split != DEV_SPLIT:
        raise ReplacementInputError(
            f"{identity.name} is a {identity.split!r} file; only dev figures are read here"
        )
    path = Path(results_dir) / identity.name
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != identity.sha256:
        raise ReplacementInputError(
            f"{identity.name} no longer has the sha256 it was identified by"
        )
    document = json.loads(raw.decode("utf-8"))
    if document.get("split") != DEV_SPLIT:
        raise ReplacementInputError(
            f"{identity.name} records split {document.get('split')!r}, not dev"
        )
    return DevFigures(
        identity=identity, metrics=dict(document["metrics"]), cost=dict(document["cost"])
    )


# --- Everything, in order -------------------------------------------------------------------------


def load_verified_inputs(
    paths: InputPaths,
    *,
    pins: InputPins,
    backend: EmbeddingBackend,
    token_counter: TokenCountFunction,
    parse: Callable[[str], Mapping[str, Any]] = json.loads,
) -> VerifiedInputs:
    """Load and verify every inherited input; the first failure raises and nothing is measured."""
    _check_pins(pins)
    manifest, units, questions, pool_hash = verify_pool(paths.data_dir, pins)
    unit_ids = [unit.unit_id for unit in units]
    split_orders, question_hashes = verify_split(questions, manifest)
    token_counts, tokens_sha = verify_token_counts(paths.data_dir, units, token_counter)
    vectors, corpus_key = verify_corpus_embeddings(paths.cache_dir, unit_ids, pool_hash, pins)
    dev = [question for question in questions if question.split == DEV_SPLIT]
    dev_backend, dev_key = build_dev_query_backend(paths.question_cache_dir, dev, backend, pins)
    selection, digest = verify_selection(paths.selection_dir, pool_hash, pins)
    historical = identify_historical_files(
        paths.results_dir, pool_hash=pool_hash, pins=pins, selection=selection, parse=parse
    )
    pilot = verify_pilot(paths.pilot_dir, pool_hash, questions, pins)
    hop_run = verify_hop_run(paths.navigation_dir, pins)
    traces = verify_navigation_traces(paths.navigation_dir, pins)
    extraction, summary = verify_extraction(paths.extraction_dir, unit_ids, pins)
    index = verify_node_index(paths.nodes_dir, unit_ids, pins)
    counts = count_inputs(unit_ids, index, pilot, traces)
    check_counts(counts, pins)
    return VerifiedInputs(
        paths=paths,
        pins=pins,
        manifest=manifest,
        units=units,
        questions=questions,
        unit_ids=unit_ids,
        pool_hash=pool_hash,
        split_orders=split_orders,
        question_set_hashes=question_hashes,
        token_counts=token_counts,
        token_counts_sha256=tokens_sha,
        vectors=vectors,
        corpus_cache_key=corpus_key,
        dev_query_backend=dev_backend,
        dev_question_cache_key=dev_key,
        selection=selection,
        selection_digest=digest,
        pilot=pilot,
        hop_run=hop_run,
        navigation_traces=traces,
        extraction=extraction,
        extraction_summary=summary,
        node_index=index,
        historical=historical,
        counts=counts,
    )
