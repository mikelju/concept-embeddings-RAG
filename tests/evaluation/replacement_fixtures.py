"""A whole toy Phase 1-5 pipeline on disk, for the Phase 6 tests (T7, T8, T16-T19).

Every inherited artifact Phase 6 verifies is written here by the project's own writers, over a
toy pool large enough for the Phase 5 hop depths (more than 100 units): the manifest, the pool,
token counts, the corpus and question embedding caches, a Phase 3 selection whose control curve
is a real fit of dense + BM25 on the toy dev questions, the four "historical" results of that
control measured by the unchanged harness, the pilot, the extraction archive, the node index,
and the navigation run with its traces. The pins are read back from what was written, so a toy
pipeline verifies exactly as the real one is meant to.

Nothing here touches `data/`: every path is under the directory handed in.
"""

import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.manifest import CorpusManifest
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question, save_pool
from concept_embeddings_rag.corpus.split import split_questions
from concept_embeddings_rag.embeddings.cache import (
    CachedQueryBackend,
    EmbeddingCache,
    cache_key,
    question_cache_key,
    question_set_hash,
    unit_set_hash,
)
from concept_embeddings_rag.evaluation.harness import evaluate_retriever
from concept_embeddings_rag.evaluation.navigation import run_navigation, save_navigation
from concept_embeddings_rag.evaluation.pilot import build_pilot, pilot_digest, save_pilot
from concept_embeddings_rag.evaluation.replacement_inputs import InputPaths, InputPins
from concept_embeddings_rag.evaluation.selection import (
    PerKEntry,
    SelectionReport,
    SweepCell,
    fit_fusion_weight,
    save_selection,
    selection_digest,
)
from concept_embeddings_rag.nodes.extraction import FAILED, OK, ExtractionRecord
from concept_embeddings_rag.nodes.index import build_node_index, save_node_index
from concept_embeddings_rag.retrieval.bm25 import BM25Retriever
from concept_embeddings_rag.retrieval.conceptual import concept_support, rarity_weights
from concept_embeddings_rag.retrieval.dense import DenseRetriever
from concept_embeddings_rag.retrieval.fusion import FusedRetriever

TOY_MODEL = "toy-model"
TOY_REVISION = "0123456789abcdef0123456789abcdef01234567"
TOY_TOKENIZER = "toy-tokenizer"
TOY_PROMPT = "0123456789abcdef"
TOY_EXTRACTOR = "toy-extractor"
N_UNITS = 120
N_QUESTIONS = 14
N_DEV = 6
DIM = 16
WORDS = ("river", "castle", "album", "film", "painter", "city", "league", "novel", "band")


class ToyBackend:
    """Names the toy model; never encodes, so every vector must come from a cache."""

    name = TOY_MODEL
    revision = TOY_REVISION
    dim = DIM
    normalize = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise AssertionError("the toy backend is never asked to encode: every vector is cached")


def toy_counter(units: Sequence[IndexingUnit]) -> dict[str, int]:
    """A deterministic stand-in tokenizer: one hundred tokens per word of indexable text."""
    return {unit.unit_id: 100 * len(unit.indexable_text.split()) for unit in units}


def _qid(index: int) -> str:
    # Content ids, not integrity: hex ids that do not sort in creation order, like HotpotQA's.
    return hashlib.sha1(f"toy-question-{index}".encode(), usedforsecurity=False).hexdigest()[:24]


def _normalized(rows: np.ndarray) -> np.ndarray:
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


@dataclass(frozen=True)
class ToyPipeline:
    root: Path
    paths: InputPaths
    pins: InputPins
    backend: ToyBackend
    units: list[IndexingUnit]
    questions: list[Question]
    replacement_dir: Path

    @property
    def dev(self) -> list[Question]:
        return [question for question in self.questions if question.split == "dev"]

    def base_config(self) -> dict:
        return {
            "model": TOY_MODEL,
            "revision": TOY_REVISION,
            "resolved_revision": TOY_REVISION,
            "unit_set_hash": self.pins.unit_set_hash,
            "seed": config.DEFAULT_SEED,
            "tokenizer": TOY_TOKENIZER,
            "code_version": "0.1.0",
            "top_k": config.EVALUATION_TOP_K,
            "n_units": N_UNITS,
        }


def _units() -> list[IndexingUnit]:
    from concept_embeddings_rag.corpus.pool import unit_id_for

    units = []
    for index in range(N_UNITS):
        words = [WORDS[(index + offset) % len(WORDS)] for offset in range(3 + index % 5)]
        title = f"Toy Title {index}"
        sentences = (" ".join(words) + f" number{index}.",)
        units.append(IndexingUnit(unit_id_for(title, sentences), title, sentences))
    return sorted(units, key=lambda unit: unit.unit_id)


def _questions(units: Sequence[IndexingUnit]) -> list[Question]:
    qids = [_qid(index) for index in range(N_QUESTIONS)]
    splits = split_questions([{"_id": qid} for qid in qids], n_dev=N_DEV, seed=config.DEFAULT_SEED)
    label = {entry["_id"]: split for split, entries in splits.items() for entry in entries}
    questions = []
    for index, qid in enumerate(qids):
        first = units[(7 * index + 3) % N_UNITS]
        second = units[(31 * index + 50) % N_UNITS]
        words = " ".join(first.sentences[0].split()[:2] + second.sentences[0].split()[:1])
        questions.append(
            Question(
                qid=qid,
                question=f"Which {words} is linked to question {index}?",
                answer="-",
                gold_unit_ids=(first.unit_id, second.unit_id),
                supporting_facts=((first.title, 0), (second.title, 0)),
                split=label[qid],
            )
        )
    return questions


def _record_line(record: ExtractionRecord) -> str:
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


def build_toy_pipeline(root: Path) -> ToyPipeline:
    """Write every inherited input under `root` and return where it is and what pins it."""
    paths = InputPaths(
        data_dir=root / "data",
        cache_dir=root / "data" / "cache",
        question_cache_dir=root / "data" / "cache" / "questions",
        selection_dir=root / "data" / "selection",
        pilot_dir=root / "data" / "pilot",
        navigation_dir=root / "data" / "navigation",
        extraction_dir=root / "data" / "extraction",
        nodes_dir=root / "data" / "nodes",
        results_dir=root / "data" / "results",
    )
    for directory in (
        paths.data_dir,
        paths.cache_dir,
        paths.question_cache_dir,
        paths.selection_dir,
        paths.pilot_dir,
        paths.navigation_dir,
        paths.extraction_dir,
        paths.nodes_dir,
        paths.results_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    units = _units()
    unit_ids = [unit.unit_id for unit in units]
    pool_hash = unit_set_hash(unit_ids)
    questions = _questions(units)
    save_pool(units, questions, paths.data_dir / "pool.json")
    CorpusManifest(
        dataset="toy",
        source_url="toy",
        sha256="0" * 64,
        downloaded_at="2026-09-01T00:00:00+00:00",
        seed=config.DEFAULT_SEED,
        n_questions=N_QUESTIONS,
        split_sizes={"dev": N_DEV, "test": N_QUESTIONS - N_DEV},
        n_units=N_UNITS,
        unit_set_hash=pool_hash,
    ).save(paths.data_dir / "manifest.json")
    counts = toy_counter(units)
    (paths.data_dir / "token_counts.json").write_text(
        json.dumps(counts, sort_keys=True), encoding="utf-8"
    )

    rng = np.random.default_rng(7)
    vectors = _normalized(rng.normal(size=(N_UNITS, DIM)))
    row = {unit_id: position for position, unit_id in enumerate(unit_ids)}
    metadata = {
        "model": TOY_MODEL,
        "revision": TOY_REVISION,
        "resolved_revision": TOY_REVISION,
        "dim": DIM,
        "normalized": True,
    }
    corpus = EmbeddingCache(paths.cache_dir)
    corpus.save(cache_key(TOY_MODEL, TOY_REVISION, pool_hash, True), vectors, unit_ids, metadata)

    question_cache = EmbeddingCache(paths.question_cache_dir)
    backends: dict[str, CachedQueryBackend] = {}
    for split in ("dev", "test"):
        subset = [question for question in questions if question.split == split]
        qids = [question.qid for question in subset]
        block = np.stack(
            [
                vectors[row[question.gold_unit_ids[0]]] + 0.35 * rng.normal(size=DIM)
                for question in subset
            ]
        )
        block = _normalized(block)
        key = question_cache_key(TOY_MODEL, TOY_REVISION, question_set_hash(qids), split, True)
        question_cache.save(key, block, qids, {**metadata, "split": split})
        backends[split] = CachedQueryBackend(
            [question.question for question in subset], block, TOY_MODEL, TOY_REVISION
        )

    dev = [question for question in questions if question.split == "dev"]
    base = {
        "model": TOY_MODEL,
        "revision": TOY_REVISION,
        "resolved_revision": TOY_REVISION,
        "unit_set_hash": pool_hash,
        "seed": config.DEFAULT_SEED,
        "tokenizer": TOY_TOKENIZER,
        "code_version": "0.1.0",
        "top_k": config.EVALUATION_TOP_K,
        "n_units": N_UNITS,
    }
    bm25 = BM25Retriever(units)
    dense_dev = DenseRetriever(vectors, unit_ids, backends["dev"])
    control = fit_fusion_weight(
        [dense_dev, bm25], questions=dev, token_counts=counts, base_config=base
    )
    cell = SweepCell(
        k=8,
        dictionary_key="toykey",
        view="raw",
        query_operator="projection_full",
        damping="none",
        n_questions=N_DEV,
        metrics={"budget_2048": {"gold_recall": 0.5, "full_support": 0.0, "precision": 0.1}},
    )
    report = SelectionReport(
        selected_k=8,
        selected_dictionary_key="toykey",
        selection_metric=config.SELECTION_METRIC,
        selection_budget=config.SELECTION_BUDGET,
        per_k={8: PerKEntry.from_cells([cell])},
        decisions=[],
        n_dev_decisions=0,
        tie_break=None,
        config=dict(base),
        seed=config.DEFAULT_SEED,
        code_version="0.1.0",
        frozen_at="2026-09-02T00:00:00+00:00",
        control=control,
    )
    save_selection(report, paths.selection_dir)

    historical: dict[str, dict[str, str]] = {"dev": {}, "test": {}}
    for split in ("dev", "test"):
        subset = [question for question in questions if question.split == split]
        dense = DenseRetriever(vectors, unit_ids, backends[split])
        hybrid = FusedRetriever(
            [dense, bm25], scheme=control.winning_scheme, weights=control.weights
        )
        extras = {
            "fusion_scheme": hybrid.scheme,
            "fusion_weights": hybrid.weights,
            "fitted_on": hybrid.fitted_on,
            "frozen_at": report.frozen_at,
        }
        for retriever, extra in ((dense, {}), (hybrid, extras)):
            result = evaluate_retriever(
                retriever,
                questions=subset,
                token_counts=counts,
                budgets=config.CONTEXT_BUDGETS,
                ks=config.RECALL_AT_K,
                top_k=config.EVALUATION_TOP_K,
                config={**base, **extra},
                split=split,
            )
            historical[split][retriever.name] = result.save(paths.results_dir).name

    pilot = build_pilot(
        dev,
        unit_ids=unit_ids,
        vectors=vectors,
        query_backend=backends["dev"],
        model=TOY_MODEL,
        revision=TOY_REVISION,
        unit_set_hash=pool_hash,
    )
    assert pilot.questions, "the toy embeddings must leave a gold paragraph outside the top-10"
    save_pilot(pilot, paths.pilot_dir)

    bridges: dict[str, list[str]] = {unit_id: [] for unit_id in unit_ids}
    protected: set[str] = set()
    for entry in pilot.questions:
        for unit_id in (entry.p1, *entry.missing):
            bridges[unit_id].append(f"Bridge {entry.qid[:6]}")
            protected.add(unit_id)
    unprotected = [unit_id for unit_id in unit_ids if unit_id not in protected]
    failed, lonely = unprotected[0], unprotected[1]
    records: dict[str, ExtractionRecord] = {}
    for position, unit_id in enumerate(unit_ids):
        status = FAILED if unit_id == failed else OK
        entities = (
            () if unit_id in (failed, lonely) else (f"Entity {position % 9}", *bridges[unit_id])
        )
        concepts = () if unit_id == failed else (f"concept {position % 4}",)
        records[unit_id] = ExtractionRecord(
            unit_id=unit_id,
            model=TOY_EXTRACTOR,
            prompt_digest=TOY_PROMPT,
            status=status,
            entities=tuple(entities),
            concepts=tuple(concepts),
            failure="invalid_json" if status == FAILED else None,
            input_tokens=10,
            output_tokens=5,
        )
    text = "".join(_record_line(records[unit_id]) + "\n" for unit_id in unit_ids)
    extraction_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (paths.extraction_dir / f"extraction-{TOY_PROMPT}.jsonl.gz").write_bytes(
        gzip.compress(text.encode("utf-8"), mtime=0)
    )
    (paths.extraction_dir / f"extraction-{TOY_PROMPT}.json").write_text(
        json.dumps(
            {
                "model": TOY_EXTRACTOR,
                "prompt_digest": TOY_PROMPT,
                "n_units": N_UNITS,
                "failures": {"invalid_json": 1},
                "failure_rate": 1 / N_UNITS,
                "finding": False,
                "digest": extraction_digest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    index = build_node_index(records, unit_ids, extraction_digest=extraction_digest)
    save_node_index(index, paths.nodes_dir)

    concept_rows = rng.random(size=(N_UNITS, 8))
    concept_rows[concept_rows < 0.6] = 0.0
    concept_matrix = sparse.csr_matrix(concept_rows)
    run = run_navigation(
        pilot,
        {question.qid: question for question in questions},
        unit_ids=unit_ids,
        texts=[unit.indexable_text for unit in units],
        vectors=vectors,
        query_backend=backends["dev"],
        bm25=bm25,
        index=index,
        concept_matrix=concept_matrix,
        concept_weights=rarity_weights(concept_support(concept_matrix), N_UNITS),
        provenance={
            "pilot_digest": pilot_digest(pilot),
            "node_index_digest": index.digest,
            "normalization_version": index.normalization_version,
            "extraction_digest": extraction_digest,
            "extraction_model": TOY_EXTRACTOR,
            "prompt_digest": TOY_PROMPT,
            "unit_set_hash": pool_hash,
            "model": TOY_MODEL,
            "revision": TOY_REVISION,
            "reference_dictionary_key": "toykey",
            "code_version": "0.1.0",
            "stochastic": False,
        },
    )
    run_path, traces_path = save_navigation(run, paths.navigation_dir)
    hop_run = json.loads(run_path.read_text(encoding="utf-8"))
    traces = json.loads(traces_path.read_text(encoding="utf-8"))

    entity_columns = index.columns_of(("entity",))
    entity_rows = index.incidence[:, entity_columns].getnnz(axis=1)
    pins = InputPins(
        unit_set_hash=pool_hash,
        model=TOY_MODEL,
        revision=TOY_REVISION,
        tokenizer=TOY_TOKENIZER,
        seed=config.DEFAULT_SEED,
        top_k=config.EVALUATION_TOP_K,
        selection_digest=selection_digest(report),
        pilot_digest=pilot_digest(pilot),
        hop_run_digest=str(hop_run["digest"]),
        traces_digest=str(traces["digest"]),
        extraction_digest=extraction_digest,
        extraction_prompt_digest=TOY_PROMPT,
        extraction_model=TOY_EXTRACTOR,
        node_index_digest=index.digest,
        normalization_version=index.normalization_version,
        historical_dev_files=dict(historical["dev"]),
        historical_test_files=dict(historical["test"]),
        n_units=N_UNITS,
        entity_nodes=int(entity_columns.size),
        failed_extractions=index.failed_units,
        units_without_entity_node=int((np.asarray(entity_rows).ravel() == 0).sum()),
        pilot_questions=len(pilot.questions),
        entity_traces=sum(1 for trace in traces["traces"] if trace["arm"] == "entity"),
    )
    return ToyPipeline(
        root=root,
        paths=paths,
        pins=pins,
        backend=ToyBackend(),
        units=units,
        questions=questions,
        replacement_dir=root / "data" / "replacement",
    )


def with_pins(pipeline: ToyPipeline, **changes: object) -> ToyPipeline:
    return replace(pipeline, pins=replace(pipeline.pins, **changes))


def rewrite_json(path: Path, change) -> None:
    """Edit a JSON file in place: what a tampered artifact looks like on disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def mapping_copy(mapping: Mapping) -> dict:
    return json.loads(json.dumps(mapping))
