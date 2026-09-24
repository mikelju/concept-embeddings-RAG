"""Command line interface: the stages of the pipeline, in the order they run.

    fetch     download and freeze the benchmark
    build     select the subset, split it, and build the unified pool
    embed     compute and cache embeddings for the pool
    evaluate  measure dense and BM25 at every context budget
    induce    build one concept space per dictionary size (Phase 2)
    label     name the concepts of one space, for the report only (Phase 2)
    select    choose the space on dev and freeze the configuration (Phase 3)
    expand    sweep the expansion grid on dev and freeze the cell (Phase 4)
    pilot     freeze the navigation pilot's dev questions by rule (Phase 5)
    extract   read entities and concepts out of every paragraph, offline (Phase 5)
    nodes     normalize the extraction into typed nodes over the pool (Phase 5)
    navigate  run every second hop over the pilot, once (Phase 5)
    replace-check   verify inputs, continuity and the control's reproduction on dev (Phase 6)
    replace-freeze  fit B on dev, check reproducibility, write the freeze (Phase 6)
    replace-test    the ordered test protocol, once, under the freeze: the state (Phase 6)
    cheap-extract   read entities out of every paragraph with a local extractor (Phase 7)
    cheap-eval      measure each local extractor on dev and select; --test reads test once
    strong-embed    encode the pool and the dev questions under the Phase 8 Dense model;
                    --test encodes the held-out questions, only after the dev gate passed
    strong-dense    fit both hybrids on dev and measure the three systems (Phase 8);
                    --test reads the held-out split once, refitting nothing
    fullwiki-corpus verify the FullWiki archive and write every paragraph as a unit (Phase 9)
    fullwiki-questions  freeze the three cohorts and the title-resolved gold (Phase 9)
    fullwiki-repro  the Phase 9 path over the historical pool must give the Phase 7 dev counts
    fullwiki-build  one build step over the whole corpus: tokens, embed, bm25, extract, index
    fullwiki-probe  time the three systems on 100 historical dev questions, rankings only
    fullwiki-eval   the single authorized evaluation pass over the 7,405 questions (Phase 9)
    fullwiki-outcome  the exact McNemar test and the mechanical terminal label (Phase 9)
    p10-questions   draw and freeze the Phase 10 held-out sets from HotpotQA train, level hard
    p10-embed       BGE question vectors for the probe and test-10 sets, on CPU (Phase 10)
    p10-repro       dev component lists; the laptop must reproduce the Phase 9 dev counts
    p10-probe       Dense on the contamination probe against its dev figure (Phase 10)
    p10-fit         the 66-point three-way weight grid on dev, the tie rule and the dev gate
    p10-eval        the single authorized held-out pass on test-10 (Phase 10)
    p10-outcome     the exact McNemar test and the mechanical terminal label (Phase 10)

Each stage is idempotent and refuses to run if its input is missing, saying which
stage to run first rather than failing somewhere deep inside numpy.
"""

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from scipy import sparse

from concept_embeddings_rag import __version__, config
from concept_embeddings_rag.artifacts import digest_of, write_text_atomic
from concept_embeddings_rag.concepts.coding import (
    ConceptMatrix,
    calibrate_coding_alpha,
    code_corpus,
    load_matrix,
    row_normalized,
    save_matrix,
)
from concept_embeddings_rag.concepts.dedup import (
    deduplicate_dictionary,
    load_merge_log,
    recode_after_merge,
    save_merge_log,
)
from concept_embeddings_rag.concepts.diagnostics import (
    SpaceDiagnostics,
    compute_diagnostics,
    load_diagnostics,
    save_diagnostics,
)
from concept_embeddings_rag.concepts.dictionary import (
    ConceptArtifactError,
    ConceptDictionary,
    dictionary_key,
    induce_dictionary,
    load_dictionary,
    save_dictionary,
)
from concept_embeddings_rag.concepts.labeling import (
    REQUEST_SHAPE,
    AnthropicLabelingClient,
    LabelingClient,
    LabelingError,
    MissingAPIKey,
    build_prompt,
    estimate_cost_usd,
    label_concepts,
    read_api_key,
    save_labels,
)
from concept_embeddings_rag.corpus import fullwiki, hf_source, scale_corpus
from concept_embeddings_rag.corpus.download import CorpusIntegrityError, sha256_of_file
from concept_embeddings_rag.corpus.manifest import CorpusManifest
from concept_embeddings_rag.corpus.pool import (
    IndexingUnit,
    Question,
    build_pool,
    load_pool,
    save_pool,
)
from concept_embeddings_rag.corpus.split import select_subset, split_questions
from concept_embeddings_rag.embeddings.backend import (
    BackendError,
    EmbeddingBackend,
    QueryPromptError,
    SentenceTransformerBackend,
    snapshot_directory,
    weights_sha256,
)
from concept_embeddings_rag.embeddings.cache import (
    CacheAlignmentError,
    CachedQueryBackend,
    EmbeddingCache,
    build_query_backend,
    cache_key,
    embed_questions,
    embed_units,
    query_prompt_of,
    question_cache_key,
    question_set_hash,
    resolved_revision,
    unit_set_hash,
)
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation import phase10, scale_sensitivity, strong_dense
from concept_embeddings_rag.evaluation.budget import TokenCounter
from concept_embeddings_rag.evaluation.cheap_extraction import (
    DEV_FILENAME,
    SELECTION_FILENAME,
    TEST_SPLIT,
    CheapEvaluationError,
    evaluate_candidate,
    measure_held_out,
    select_extractor,
)
from concept_embeddings_rag.evaluation.entity_diagnostics import RecordingHybrid, RecordingRetriever
from concept_embeddings_rag.evaluation.expansion_selection import (
    ExpansionSelection,
    ExpansionSelectionError,
    build_expansion_retrievers,
    cell_config,
    check_dictionary_agrees,
    choose_cell,
    expansion_path,
    expansion_selection_digest,
    freeze_expansion_selection,
    load_expansion_selection,
    save_expansion_selection,
    sweep_expansion,
)
from concept_embeddings_rag.evaluation.gate import compute_gate, save_gate
from concept_embeddings_rag.evaluation.harness import evaluate_retriever
from concept_embeddings_rag.evaluation.navigation import (
    HOP_RUN_FILENAME,
    diagnostic_mismatches,
    load_hop_run,
    run_navigation,
    save_navigation,
)
from concept_embeddings_rag.evaluation.pilot import (
    PilotError,
    build_pilot,
    check_pilot_against,
    load_pilot,
    pilot_digest,
    pilot_path,
    save_pilot,
)
from concept_embeddings_rag.evaluation.replacement_inputs import (
    InputPaths,
    InputPins,
    offline_token_counter,
)
from concept_embeddings_rag.evaluation.replacement_run import (
    ReplacementRunError,
    StageEnvironment,
    run_check,
    run_freeze,
    run_test,
)
from concept_embeddings_rag.evaluation.second_hop import node_weights
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    SelectionError,
    SelectionReport,
    SweepSpace,
    check_dev_only,
    check_freeze_precedes,
    choose_damping,
    choose_space,
    fit_fusion_weight,
    freeze_selection,
    load_selection,
    run_dev_sweep,
    save_selection,
    selection_digest,
    sweep_space,
)
from concept_embeddings_rag.nodes import local_extraction
from concept_embeddings_rag.nodes.extraction import (
    AnthropicBatchClient,
    AnthropicSyncClient,
    BatchClient,
    ExtractionError,
    SyncClient,
    load_extraction,
    load_extraction_summary,
    load_sample_report,
    run_full,
    run_sample,
)
from concept_embeddings_rag.nodes.index import (
    NodeIndexError,
    build_node_index,
    fragmentation,
    load_node_index,
    save_node_index,
)
from concept_embeddings_rag.nodes.local_extraction import LocalExtractionError, LocalExtractor
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.bm25 import BM25Retriever
from concept_embeddings_rag.retrieval.conceptual import (
    RARITY,
    UNDAMPED,
    ConceptualRetriever,
    concept_support,
    rarity_weights,
)
from concept_embeddings_rag.retrieval.dense import DenseRetriever
from concept_embeddings_rag.retrieval.diffusion import EXPANSION_NAMES, DiffusionRetriever
from concept_embeddings_rag.retrieval.entity_hop import (
    ColumnwiseEntityHopStage,
    EntityHopStage,
)
from concept_embeddings_rag.retrieval.fusion import (
    TRIPLE_COMPONENTS,
    WEIGHTED,
    FusedRetriever,
    TripleFusedRetriever,
    fuse_lists,
)

RAW_NAME = "hotpot_raw.json"
MANIFEST_NAME = "manifest.json"
POOL_NAME = "pool.json"
TOKENS_NAME = "token_counts.json"
SOURCE_NAME = "source.json"


def _die(message: str) -> NoReturn:
    raise SystemExit(f"[ERROR] {message}")


def cmd_fetch(data_dir: Path = config.DATA_DIR, seed: int = config.DEFAULT_SEED) -> Path:
    """Download the benchmark and record its hash in a manifest."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / RAW_NAME
    manifest_path = data_dir / MANIFEST_NAME

    # A manifest from an earlier run is what turns this into a verification instead of
    # a description: with nothing to compare against, hashing the file only ever
    # confirms itself. Absent a manifest, this run is the one that establishes the hash.
    expected = CorpusManifest.load(manifest_path).sha256 if manifest_path.exists() else None

    if raw_path.exists():
        actual = sha256_of_file(raw_path)
        if expected is not None and actual != expected:
            raise CorpusIntegrityError(
                f"{raw_path} has hash {actual}, manifest expects {expected}; "
                "delete the file to re-fetch it deliberately"
            )
        state = "verified" if expected is not None else "unverified, first run"
        print(f"[INFO] corpus already present at {raw_path} ({state})")
    else:
        print(f"[INFO] assembling {hf_source.DATASET} ({hf_source.CONFIG}/{hf_source.SPLIT})")
        rows = hf_source.fetch_split(pause=0.5)
        raw_path.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
        print(f"[OK] assembled {len(rows)} questions")

    digest = sha256_of_file(raw_path)
    # Both branches end here, and both must be checked. A corpus deleted while its
    # manifest survives comes back through the branch above, and accepting whatever
    # the API returns at that point would reopen exactly the hole this closes.
    if expected is not None and digest != expected:
        raise CorpusIntegrityError(
            f"assembled corpus has hash {digest}, manifest expects {expected}; "
            "the upstream dataset has changed - delete the manifest to refreeze it "
            "deliberately, and treat every recorded number as belonging to the old corpus"
        )

    manifest = CorpusManifest(
        dataset=config.DATASET_NAME,
        source_url=config.CORPUS_SOURCE,
        sha256=digest,
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        seed=seed,
        n_questions=config.N_QUESTIONS,
        split_sizes={"dev": config.N_DEV, "test": config.N_TEST},
    )
    manifest.save(manifest_path)
    print(f"[OK] corpus frozen at {raw_path} (sha256 {digest[:16]}...)")
    return raw_path


def cmd_build(
    data_dir: Path = config.DATA_DIR,
    n_questions: int = config.N_QUESTIONS,
    n_dev: int = config.N_DEV,
    seed: int = config.DEFAULT_SEED,
) -> Path:
    """Select the subset, split it, and build the unified pool."""
    data_dir = Path(data_dir)
    raw_path = data_dir / RAW_NAME
    manifest_path = data_dir / MANIFEST_NAME
    if not raw_path.exists() or not manifest_path.exists():
        _die("no frozen corpus found: run 'fetch' first")

    manifest = CorpusManifest.load(manifest_path)
    actual = sha256_of_file(raw_path)
    if actual != manifest.sha256:
        _die(
            f"corpus at {raw_path} has hash {actual} but the manifest expects "
            f"{manifest.sha256}; re-run 'fetch'"
        )
    raw_questions = json.loads(raw_path.read_text(encoding="utf-8"))
    print(f"[INFO] benchmark holds {len(raw_questions)} questions")

    subset = select_subset(raw_questions, n=n_questions, seed=seed)
    splits = split_questions(subset, n_dev=n_dev, seed=seed)
    units, questions = build_pool(splits)

    print(f"[INFO] pool holds {len(units)} indexing units for {len(questions)} questions")
    save_pool(units, questions, data_dir / POOL_NAME)

    updated = CorpusManifest(
        dataset=manifest.dataset,
        source_url=manifest.source_url,
        sha256=manifest.sha256,
        downloaded_at=manifest.downloaded_at,
        seed=seed,
        n_questions=n_questions,
        split_sizes={"dev": n_dev, "test": n_questions - n_dev},
        n_units=len(units),
        unit_set_hash=unit_set_hash([u.unit_id for u in units]),
    )
    updated.save(manifest_path)
    print(f"[OK] pool written to {data_dir / POOL_NAME}")
    return data_dir / POOL_NAME


def cmd_embed(data_dir: Path = config.DATA_DIR, cache_dir: Path = config.CACHE_DIR) -> None:
    """Compute and cache embeddings, and the token count of every unit."""
    data_dir = Path(data_dir)
    pool_path = data_dir / POOL_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")

    units, _ = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    backend = SentenceTransformerBackend()
    cache = EmbeddingCache(Path(cache_dir))
    vectors, _ = embed_units(units, backend, cache)
    print(f"[OK] embeddings ready: {vectors.shape[0]} x {vectors.shape[1]}")

    counter = TokenCounter()
    counts = counter.count_units(units)
    (data_dir / TOKENS_NAME).write_text(json.dumps(counts, sort_keys=True), encoding="utf-8")
    mean_tokens = sum(counts.values()) / len(counts)
    print(f"[OK] token counts written ({mean_tokens:.1f} tokens per unit on average)")


def cmd_evaluate(
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    results_dir: Path = config.RESULTS_DIR,
    concepts_dir: Path = config.CONCEPTS_DIR,
    selection_dir: Path = config.SELECTION_DIR,
    question_cache_dir: Path = config.QUESTION_CACHE_DIR,
    expansion_dir: Path = config.EXPANSION_DIR,
    top_k: int = config.EVALUATION_TOP_K,
    splits: Sequence[str] = ("dev", "test"),
    systems: Sequence[str] | None = None,
    backend: EmbeddingBackend | None = None,
) -> list[Path]:
    """Measure the requested systems on the requested splits, at every budget.

    Without `systems`, the two Phase 1 baselines and - whenever a frozen selection
    exists - the three Phase 3 systems, built entirely out of what that artifact
    names. That was the whole stage until Phase 4.

    `systems` is decision D11 of Phase 4. Running this stage again to measure the two
    expansion systems would otherwise produce a **second test-split result for every
    Phase 3 system**, which Phase 3 declared a deviation waiting to be written. So the
    Phase 4 test read names its two systems and measures nothing else - which is also
    what makes "the rivals' numbers are read from the Phase 3 files" the only
    possibility rather than a good intention.

    **The test split is not read without a freeze** (decision D10). Not because the
    baselines need one, but because after Phase 3 nothing may touch test on a
    configuration that was still open: test is read once, on a frozen configuration.
    The expansion systems additionally require the Phase 4 freeze and refuse to be
    built without it.
    """
    requested = _requested_systems(systems)
    data_dir = Path(data_dir)
    pool_path = data_dir / POOL_NAME
    tokens_path = data_dir / TOKENS_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")
    if not tokens_path.exists():
        _die("no embeddings or token counts found: run 'embed' first")

    units, questions = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    unit_ids_of_pool = [u.unit_id for u in units]
    pool_hash = unit_set_hash(unit_ids_of_pool)
    token_counts = json.loads(tokens_path.read_text(encoding="utf-8"))

    backend = backend if backend is not None else SentenceTransformerBackend()
    cache = EmbeddingCache(Path(cache_dir))
    cached = cache.load(_cache_key_for(backend, units), expected_unit_ids=unit_ids_of_pool)
    if cached is None:
        _die("embeddings are not cached for this pool: run 'embed' first")
    vectors, unit_ids = cached

    report = _frozen_selection(Path(selection_dir), Path(concepts_dir), pool_hash, splits)
    expansion = (
        _frozen_expansion(Path(expansion_dir), pool_hash, report)
        if _wants_expansion(requested)
        else None
    )
    run_config = {
        "model": backend.name,
        "revision": backend.revision,
        "resolved_revision": resolved_revision(backend),
        "unit_set_hash": pool_hash,
        "seed": config.DEFAULT_SEED,
        "tokenizer": config.TOKENIZER_ID,
        "code_version": __version__,
        "top_k": top_k,
        "n_units": len(units),
    }
    question_cache = EmbeddingCache(Path(question_cache_dir))

    written: list[Path] = []
    for split in splits:
        subset = [q for q in questions if q.split == split]
        if not subset:
            continue

        # One query backend per split, serving the vectors `select` cached: the
        # hybrids ask their dense component once per weight of the fitted scheme,
        # and re-encoding the same questions for each of those would cost minutes
        # and change nothing (decision D1).
        query_backend = build_query_backend(subset, backend, question_cache)
        dense = DenseRetriever(vectors=vectors, unit_ids=unit_ids, backend=query_backend)
        bm25 = BM25Retriever(units)

        built: list[tuple[Retriever, dict]] = [(dense, {}), (bm25, {})]
        if report is not None:
            built.extend(
                _phase_3_systems(
                    report, Path(concepts_dir), unit_ids_of_pool, query_backend, dense, bm25
                )
            )
        if report is not None and expansion is not None:
            built.extend(
                _phase_4_systems(
                    report,
                    expansion,
                    Path(concepts_dir),
                    unit_ids_of_pool,
                    query_backend,
                    dense,
                )
            )

        for retriever, extra in built:
            if requested is not None and retriever.name not in requested:
                continue
            print(f"[INFO] evaluating {retriever.name} on {split} ({len(subset)} questions)")
            if isinstance(retriever, DiffusionRetriever):
                # Decision D12: the counters are per run, so a caller that forgot to
                # reset them would report two runs as one.
                retriever.reset_stats()
            result = evaluate_retriever(
                retriever,
                questions=subset,
                token_counts=token_counts,
                budgets=config.CONTEXT_BUDGETS,
                ks=config.RECALL_AT_K,
                top_k=top_k,
                config={**run_config, **extra},
                split=split,
            )
            if isinstance(retriever, DiffusionRetriever):
                # The harness takes its configuration before the run exists, and it is
                # frozen for this phase, so a measurement of the run itself has no other
                # door into the result.
                result = replace(result, config={**result.config, **retriever.stats.describe()})
            if extra and report is not None:
                # The freeze has to precede the measurement, not merely exist.
                check_freeze_precedes(report, result)
                if expansion is not None and isinstance(retriever, DiffusionRetriever):
                    check_freeze_precedes(expansion, result)
            written.append(result.save(Path(results_dir)))

    print(f"[OK] wrote {len(written)} result files to {results_dir}")
    return written


def _frozen_selection(
    selection_dir: Path, concepts_dir: Path, pool_hash: str, splits: Sequence[str]
) -> SelectionReport | None:
    """The frozen configuration, or nothing - and nothing is fatal if test is asked for."""
    known = [path.stem.split("-", maxsplit=1)[1] for path in concepts_dir.glob("dictionary-*.npz")]
    try:
        report = load_selection(
            selection_dir, expected_unit_set_hash=pool_hash, known_dictionary_keys=known
        )
    except SelectionError as error:
        if DEV_SPLIT not in splits or len(splits) > 1:
            _die(
                f"the test split may not be read on a configuration that is still open "
                f"({error}): run 'select' to freeze one first"
            )
        print(f"[WARN] no frozen selection in {selection_dir}: measuring the baselines only")
        return None
    print(f"[INFO] using the selection frozen at {report.frozen_at} (k={report.selected_k})")
    return report


def _phase_3_systems(
    report: SelectionReport,
    concepts_dir: Path,
    unit_ids: Sequence[str],
    query_backend: EmbeddingBackend,
    dense: Retriever,
    bm25: Retriever,
) -> list[tuple[Retriever, dict]]:
    """The three systems this phase measures, built out of the frozen artifact alone.

    Every one of them carries the six keys the spec adds to a Phase 3 result plus the
    freeze they were built under, so a result file states on its own which of the four
    spaces produced it and under which decisions.

    System B and the control come out of the same constructor with the same scheme and
    the same weights-by-name (decision D9). Nothing here re-fits anything: the weights
    were fitted on dev by `select` and are read back from what it froze.
    """
    frozen = report.config
    key = str(frozen["dictionary_key"])
    view = str(frozen["view"])
    try:
        dictionary = load_dictionary(
            key, concepts_dir, expected_unit_set_hash=report.config.get("unit_set_hash")
        )
        matrix = load_matrix(key, concepts_dir, view=view, expected_unit_ids=unit_ids)
    except ConceptArtifactError as error:
        _die(f"the frozen selection names a space that is not on disk ({error}): run 'induce'")

    damping = str(frozen["damping"])
    weights = None
    if damping != UNDAMPED:
        raw = load_matrix(key, concepts_dir, view="raw", expected_unit_ids=unit_ids)
        weights = rarity_weights(concept_support(raw.X), raw.X.shape[0])

    conceptual = ConceptualRetriever(
        matrix,
        dictionary,
        query_backend,
        arm=str(frozen["query_operator"]),
        damping=damping,
        concept_weights=weights,
    )

    provenance = {
        "dictionary_key": key,
        "k": int(frozen["k"]),
        "view": view,
        "query_operator": str(frozen["query_operator"]),
        "damping": damping,
        "frozen_at": report.frozen_at,
    }
    fitted = [
        (fit, second)
        for fit, second in ((report.fusion, conceptual), (report.control, bm25))
        if fit is not None
    ]
    systems: list[tuple[Retriever, dict]] = [(conceptual, {**provenance, "fusion_scheme": None})]
    for fit, second in fitted:
        hybrid = FusedRetriever([dense, second], scheme=fit.winning_scheme, weights=fit.weights)
        systems.append(
            (
                hybrid,
                {
                    **provenance,
                    "fusion_scheme": hybrid.scheme,
                    "fusion_weights": hybrid.weights,
                    "fitted_on": hybrid.fitted_on,
                },
            )
        )
    return systems


# The names `--systems` accepts, and the two that need the Phase 4 freeze.
PHASE_1_AND_3_SYSTEMS: tuple[str, ...] = (
    "dense",
    "bm25",
    "conceptual",
    "hybrid-conceptual",
    "hybrid-bm25",
)
EXPANSION_SYSTEMS: tuple[str, ...] = tuple(EXPANSION_NAMES[arm] for arm in config.SEED_ARMS)
KNOWN_SYSTEMS: tuple[str, ...] = PHASE_1_AND_3_SYSTEMS + EXPANSION_SYSTEMS


def _requested_systems(systems: Sequence[str] | None) -> frozenset[str] | None:
    """What `--systems` asked for, or nothing - which means the pre-Phase-4 behaviour."""
    if systems is None:
        return None
    names = [name.strip() for name in systems if name.strip()]
    if not names:
        _die(f"--systems was given no name; expected some of {list(KNOWN_SYSTEMS)}")
    unknown = sorted({name for name in names if name not in KNOWN_SYSTEMS})
    if unknown:
        _die(f"unknown system(s) {unknown}; expected some of {list(KNOWN_SYSTEMS)}")
    return frozenset(names)


def _wants_expansion(requested: frozenset[str] | None) -> bool:
    """The expansion systems are built only when named.

    Never by default (decision D11): building them on a bare `evaluate` would measure
    them beside a second test-split result for every Phase 3 system, which is the
    duplication the selector exists to prevent.
    """
    return requested is not None and bool(requested & set(EXPANSION_SYSTEMS))


def _frozen_expansion(
    expansion_dir: Path, pool_hash: str, inherited: SelectionReport | None
) -> ExpansionSelection:
    """The Phase 4 freeze, or a refusal naming the stage that writes it."""
    if inherited is None:
        _die(
            "the expansion systems inherit a Phase 3 configuration and there is none frozen: "
            "run 'select' first"
        )
    try:
        return load_expansion_selection(
            expansion_dir,
            expected_unit_set_hash=pool_hash,
            inherits_selection_digest=selection_digest(inherited),
        )
    except ExpansionSelectionError as error:
        _die(f"the expansion systems may not be measured on an open configuration ({error})")


def _phase_4_systems(
    inherited: SelectionReport,
    expansion: ExpansionSelection,
    concepts_dir: Path,
    unit_ids: Sequence[str],
    query_backend: EmbeddingBackend,
    dense: Retriever,
) -> list[tuple[Retriever, dict]]:
    """The two arms of HU-3, built out of the two frozen artifacts alone.

    Nothing here chooses anything: the cell was chosen on dev by `expand` and is read
    back from what it froze, and the space, view, damping and query operator come from
    the Phase 3 artifact it inherits. The conceptual seed is Phase 3's own retriever at
    that configuration, so the isolating arm differs from System C in its seed and in
    nothing else.
    """
    dictionary, matrix, weights, inherits = _inherited_space(inherited, concepts_dir, unit_ids)
    try:
        check_dictionary_agrees(expansion, dictionary.key)
    except ExpansionSelectionError as error:
        _die(str(error))

    conceptual = ConceptualRetriever(
        matrix,
        dictionary,
        query_backend,
        arm=inherits["query_operator"],
        damping=inherits["damping"],
        concept_weights=weights,
    )
    retrievers = build_expansion_retrievers(
        matrix=matrix,
        dictionary=dictionary,
        seed_retrievers={"dense": dense, "conceptual": conceptual},
        restart=expansion.chosen_restart,
        normalization=expansion.chosen_normalization,
        inherits=inherits,
        concept_weights=weights,
    )

    digests = {
        "selection_digest": selection_digest(inherited),
        "expansion_selection_digest": expansion_selection_digest(expansion),
        "frozen_at": expansion.frozen_at,
        "inherits_frozen_at": inherited.frozen_at,
        "fusion_scheme": None,
    }
    return [(retriever, {**cell_config({}, retriever), **digests}) for retriever in retrievers]


def _diagnostics_for(
    dictionary: ConceptDictionary,
    matrix: ConceptMatrix,
    vectors: np.ndarray,
    concepts_dir: Path,
    seed: int,
) -> SpaceDiagnostics:
    """The diagnostics of one space, computed only if they are not already on disk.

    An artifact written before the quality block existed is recomputed rather than
    reused. It costs seconds - unlike the dictionary and `X` above it - and a
    diagnostics file without quality would read as a space nobody scored. The same
    goes for one written before it recorded the pool it was computed over: an
    artifact that cannot be checked against the pool is not reused as if it had
    been checked.
    """
    try:
        existing = load_diagnostics(
            dictionary.key, concepts_dir, expected_unit_set_hash=dictionary.unit_set_hash
        )
        if existing.quality is not None and existing.unit_set_hash:
            return existing
    except (ConceptArtifactError, OSError, ValueError, KeyError):
        pass

    diagnostics = compute_diagnostics(matrix, vectors=vectors, atoms=dictionary.atoms, seed=seed)
    save_diagnostics(diagnostics, concepts_dir)
    return diagnostics


def cmd_induce(
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    concepts_dir: Path = config.CONCEPTS_DIR,
    k_sweep: Sequence[int] = config.CONCEPT_K_SWEEP,
    seed: int = config.CONCEPT_SEED,
    induction_alpha: float = config.INDUCTION_ALPHA,
    target_band: tuple[int, int] = config.SPARSITY_TARGET_BAND,
    merge_threshold: float = config.MERGE_COSINE_THRESHOLD,
    max_merged_fraction: float = config.MAX_MERGED_FRACTION,
    budget_seconds: float = config.CONCEPT_SWEEP_BUDGET_SECONDS,
) -> list[str]:
    """Build one concept space per K: induce, calibrate, code, deduplicate, diagnose.

    Nothing is embedded here. The stage reads the vectors Phase 1 cached and
    refuses to run without them, because re-embedding the corpus is the expensive
    operation of this project and is not something a later stage may trigger by
    accident.

    Nothing already on disk is recomputed either. Every artifact is addressed by
    the configuration that produced it, so a re-run finds its own output and skips
    straight past it - which is what makes iterating on this phase affordable.

    Returns the key of each deduplicated dictionary, in sweep order. **No K is
    chosen here**: that decision belongs to Phase 3, made against dev recall.
    """
    data_dir = Path(data_dir)
    concepts_dir = Path(concepts_dir)
    pool_path = data_dir / POOL_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")

    units, _ = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    unit_ids = [unit.unit_id for unit in units]
    pool_hash = unit_set_hash(unit_ids)

    cached = EmbeddingCache(Path(cache_dir)).load(
        cache_key(
            config.EMBEDDING_MODEL,
            config.EMBEDDING_REVISION,
            pool_hash,
            normalized=config.NORMALIZE_EMBEDDINGS,
        ),
        expected_unit_ids=unit_ids,
    )
    if cached is None:
        _die("embeddings are not cached for this pool: run 'embed' first")
    vectors, _ = cached

    print(f"[INFO] inducing concept spaces over {len(unit_ids)} units, k in {tuple(k_sweep)}")
    concepts_dir.mkdir(parents=True, exist_ok=True)

    def derived_views_of(matrix: ConceptMatrix) -> None:
        """Write the views that are functions of `raw`, so a reader can verify them.

        `row_normalized` costs one pass over `X` to derive, which is why it was
        tempting to leave it to the reader. It is written instead because Phase 3
        measures on both views (decision D3) and this project verifies artifacts
        on load rather than trusting them: a view derived in memory would be the
        one matrix in the pipeline that no digest covers.
        """
        try:
            load_matrix(
                matrix.dictionary_key,
                concepts_dir,
                view="row_normalized",
                expected_unit_ids=unit_ids,
            )
            return
        except ConceptArtifactError:
            pass
        save_matrix(row_normalized(matrix), concepts_dir)

    def matrix_for(dictionary: ConceptDictionary) -> ConceptMatrix:
        """The raw `X` of a dictionary, coded only if it is not already on disk."""
        try:
            matrix = load_matrix(dictionary.key, concepts_dir, expected_unit_ids=unit_ids)
        except ConceptArtifactError:
            alpha, achieved = calibrate_coding_alpha(
                dictionary, vectors, target_band=target_band, seed=seed
            )
            print(f"[INFO]   coding alpha {alpha:.5f} -> {achieved:.2f} active concepts per unit")
            matrix = (
                recode_after_merge(dictionary, vectors, unit_ids, alpha=alpha)
                if dictionary.merge_threshold is not None
                else code_corpus(dictionary, vectors, unit_ids, alpha=alpha)
            )
            save_matrix(matrix, concepts_dir)
        derived_views_of(matrix)
        return matrix

    produced: list[str] = []
    elapsed_total = 0.0
    for k in k_sweep:
        started = time.perf_counter()
        key = dictionary_key(
            k=k,
            seed=seed,
            alpha=induction_alpha,
            unit_set_hash=pool_hash,
            model=config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_REVISION,
        )
        try:
            dictionary = load_dictionary(key, concepts_dir, expected_unit_set_hash=pool_hash)
            print(f"[INFO] k={k}: dictionary {key} already induced")
        except ConceptArtifactError:
            print(f"[INFO] k={k}: inducing {k} concepts (alpha {induction_alpha})")
            dictionary = induce_dictionary(
                vectors,
                k=k,
                seed=seed,
                alpha=induction_alpha,
                unit_set_hash=pool_hash,
                model=config.EMBEDDING_MODEL,
                revision=config.EMBEDDING_REVISION,
            )
            save_dictionary(dictionary, concepts_dir)

        matrix_for(dictionary)

        merged_key = dictionary_key(
            k=k,
            seed=seed,
            alpha=induction_alpha,
            unit_set_hash=pool_hash,
            model=config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_REVISION,
            merge_threshold=merge_threshold,
        )
        try:
            merged = load_dictionary(merged_key, concepts_dir, expected_unit_set_hash=pool_hash)
            log = load_merge_log(merged_key, concepts_dir)
        except ConceptArtifactError:
            merged, log = deduplicate_dictionary(
                dictionary,
                threshold=merge_threshold,
                max_merged_fraction=max_merged_fraction,
            )
            save_dictionary(merged, concepts_dir)
            save_merge_log(log, concepts_dir)
        print(f"[INFO] k={k}: {log.k_before} atoms -> {log.k_after} after deduplication")
        if log.finding is not None:
            print(f"[WARN] {log.finding}")

        merged_matrix = matrix_for(merged)

        quality = _diagnostics_for(merged, merged_matrix, vectors, concepts_dir, seed).quality
        if quality is not None:
            print(
                f"[INFO] k={k}: coherence {quality.coherence_stats['mean']:.3f} "
                f"against a null of {quality.null_mean:.3f}, "
                f"{len(quality.below_null_concepts)} concepts below it"
            )

        produced.append(merged.key)
        took = time.perf_counter() - started
        elapsed_total += took
        print(f"[OK] k={k}: space ready as {merged.key} in {took:.1f} s")

        if elapsed_total > budget_seconds:
            print(
                f"[WARN] the sweep has used {elapsed_total:.1f} s of its "
                f"{budget_seconds:.0f} s budget; stopping before the remaining k. "
                "Revisit the parameters rather than leaving it to run overnight"
            )
            break

    print(f"[OK] {len(produced)} concept spaces in {concepts_dir} ({elapsed_total:.1f} s total)")
    return produced


def _text_of(unit_id: str, text_by_id: dict[str, str], dictionary_key: str) -> str:
    """The text of an evidence unit, or an error that says which artifact is stale.

    The evidence ids come from a diagnostics artifact and the texts from the pool
    just loaded. When the two disagree the raw lookup raises a bare KeyError with
    an opaque id in it, which reads like a bug in the labelling stage rather than
    what it is: an artifact describing a pool that is no longer on disk.
    """
    try:
        return text_by_id[unit_id]
    except KeyError:
        _die(
            f"the diagnostics of {dictionary_key} cite unit {unit_id}, which is not in the "
            "current pool: the artifact is stale, re-run 'induce'"
        )


def cmd_label(
    data_dir: Path = config.DATA_DIR,
    concepts_dir: Path = config.CONCEPTS_DIR,
    k: int = config.LABELED_K,
    seed: int = config.CONCEPT_SEED,
    induction_alpha: float = config.INDUCTION_ALPHA,
    merge_threshold: float = config.MERGE_COSINE_THRESHOLD,
    n_units_shown: int = config.LABEL_EVIDENCE_UNITS,
    client: LabelingClient | None = None,
) -> Path:
    """Name the concepts of one dictionary, for the report and for nothing else.

    Only one space of the sweep is labelled - `LABELED_K` - because the other
    three are inspected through `top_units_per_concept`, which reads the same
    evidence and costs nothing. If Phase 3 selects a different K, labelling that
    dictionary is Phase 3's business, not a reason to relabel everything here.

    Nothing downstream depends on this stage: a concept's embedding is its atom.
    """
    data_dir = Path(data_dir)
    concepts_dir = Path(concepts_dir)
    pool_path = data_dir / POOL_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")

    units, _ = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    pool_hash = unit_set_hash([unit.unit_id for unit in units])

    merged_key = dictionary_key(
        k=k,
        seed=seed,
        alpha=induction_alpha,
        unit_set_hash=pool_hash,
        model=config.EMBEDDING_MODEL,
        revision=config.EMBEDDING_REVISION,
        merge_threshold=merge_threshold,
    )
    try:
        diagnostics = load_diagnostics(merged_key, concepts_dir, expected_unit_set_hash=pool_hash)
    except ConceptArtifactError:
        _die(f"no concept space for k={k} in {concepts_dir}: run 'induce' first")

    text_by_id = {unit.unit_id: unit.indexable_text for unit in units}
    evidence = {
        int(concept): [
            (unit_id, _text_of(unit_id, text_by_id, merged_key))
            for unit_id in unit_ids[:n_units_shown]
        ]
        for concept, unit_ids in diagnostics.top_units_per_concept.items()
    }
    prompts = [build_prompt(concept, shown) for concept, shown in sorted(evidence.items()) if shown]
    estimate = estimate_cost_usd(prompts, int(REQUEST_SHAPE["max_tokens"]))
    print(f"[INFO] labelling {len(prompts)} concepts of k={k} with {config.LABELING_MODEL}")
    print(f"[INFO] estimated cost {estimate:.2f} USD, before the on-disk cache is consulted")

    if client is None:
        try:
            # Reads the key and nothing else. No HTTP client is constructed until
            # this returns, so an absent key is a one-line message rather than a
            # failure deep inside a connection attempt.
            read_api_key()
        except MissingAPIKey as error:
            _die(str(error))
        client = AnthropicLabelingClient()

    labels = label_concepts(
        dictionary_key=merged_key,
        evidence=evidence,
        client=client,
        cache_dir=concepts_dir,
        n_units_shown=n_units_shown,
    )
    path = save_labels(labels, concepts_dir)
    print(
        f"[OK] {len(labels.labels)} concepts labelled "
        f"({labels.cost['calls']} calls, {labels.cost['cached']} from cache, "
        f"{labels.cost['actual_usd']:.2f} USD spent) -> {path}"
    )
    return path


def _spaces_for_selection(
    concepts_dir: Path,
    unit_ids: Sequence[str],
    pool_hash: str,
    *,
    k_sweep: Sequence[int],
    seed: int,
    induction_alpha: float,
    merge_threshold: float,
) -> list[SweepSpace]:
    """The Phase 2 spaces this phase chooses between, loaded and hash-verified.

    Both views of every space are required up front. Loading them lazily would let
    the sweep die twelve cells in, after the expensive part of the work, on a file
    that was already missing when the stage started.
    """
    spaces: list[SweepSpace] = []
    for k in k_sweep:
        key = dictionary_key(
            k=k,
            seed=seed,
            alpha=induction_alpha,
            unit_set_hash=pool_hash,
            model=config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_REVISION,
            merge_threshold=merge_threshold,
        )
        try:
            dictionary = load_dictionary(key, concepts_dir, expected_unit_set_hash=pool_hash)
            matrices = {
                view: load_matrix(key, concepts_dir, view=view, expected_unit_ids=unit_ids)
                for view in config.CONCEPT_VIEWS
            }
        except ConceptArtifactError as error:
            _die(f"no concept space for k={k} in {concepts_dir} ({error}): run 'induce' first")
        spaces.append(SweepSpace(k=dictionary.k, dictionary=dictionary, matrices=matrices))
    return spaces


def cmd_select(
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    concepts_dir: Path = config.CONCEPTS_DIR,
    selection_dir: Path = config.SELECTION_DIR,
    question_cache_dir: Path = config.QUESTION_CACHE_DIR,
    k_sweep: Sequence[int] = config.CONCEPT_K_SWEEP,
    seed: int = config.CONCEPT_SEED,
    induction_alpha: float = config.INDUCTION_ALPHA,
    merge_threshold: float = config.MERGE_COSINE_THRESHOLD,
    top_k: int = config.EVALUATION_TOP_K,
    backend: EmbeddingBackend | None = None,
) -> Path:
    """Choose the space on dev, fit the fusion on dev, and freeze the configuration.

    The four decisions of D8 in the order D8 fixed: the space, then the damping on
    the space that won, then the fusion scheme, then its weight. The HU-5 control is
    fitted by the same function immediately afterwards, which is what makes it a
    control rather than a second system.

    Nothing here reads the test split. Its questions are **embedded**, which is a
    vector and not a measurement: doing it now is what lets `evaluate` read test once
    later without loading the model, and the guard inside the sweep refuses a
    test-split question at every door regardless.
    """
    data_dir = Path(data_dir)
    concepts_dir = Path(concepts_dir)
    pool_path = data_dir / POOL_NAME
    tokens_path = data_dir / TOKENS_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")
    if not tokens_path.exists():
        _die("no token counts found: run 'embed' first")

    units, questions = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    unit_ids = [unit.unit_id for unit in units]
    pool_hash = unit_set_hash(unit_ids)
    token_counts = json.loads(tokens_path.read_text(encoding="utf-8"))

    backend = backend if backend is not None else SentenceTransformerBackend()
    corpus = EmbeddingCache(Path(cache_dir)).load(
        _cache_key_for(backend, units), expected_unit_ids=unit_ids
    )
    if corpus is None:
        _die("embeddings are not cached for this pool: run 'embed' first")
    vectors, cached_unit_ids = corpus

    spaces = _spaces_for_selection(
        concepts_dir,
        unit_ids,
        pool_hash,
        k_sweep=k_sweep,
        seed=seed,
        induction_alpha=induction_alpha,
        merge_threshold=merge_threshold,
    )

    dev = [question for question in questions if question.split == DEV_SPLIT]
    if not dev:
        _die(f"the pool holds no {DEV_SPLIT} questions: run 'build' first")

    question_cache = EmbeddingCache(Path(question_cache_dir))
    query_backend = build_query_backend(dev, backend, question_cache)
    for split in sorted({q.split for q in questions if q.split != DEV_SPLIT}):
        embed_questions([q for q in questions if q.split == split], backend, question_cache)

    base_config = {
        "model": backend.name,
        "revision": backend.revision,
        "resolved_revision": resolved_revision(backend),
        "unit_set_hash": pool_hash,
        "seed": seed,
        "tokenizer": config.TOKENIZER_ID,
        "code_version": __version__,
        "top_k": top_k,
        "n_units": len(units),
    }

    print(
        f"[INFO] sweeping {len(spaces)} spaces x {len(config.QUERY_OPERATORS)} arms x "
        f"{len(config.CONCEPT_VIEWS)} views over {len(dev)} {DEV_SPLIT} questions"
    )
    table = run_dev_sweep(
        spaces,
        questions=dev,
        token_counts=token_counts,
        backend=query_backend,
        base_config=base_config,
    )
    space = choose_space(table)
    print(
        f"[INFO] space: {space.label} at {config.SELECTION_METRIC} {space.primary:.4f} "
        f"(runner-up {space.runner_up}, margin {space.margin:.4f})"
    )
    if space.tie_break is not None:
        print(
            f"[INFO] the margin was inside what {table[space.k].n_questions} questions "
            f"resolve, so structure decided it -> {space.tie_break}"
        )

    selected = next(candidate for candidate in spaces if candidate.k == space.k)
    # The support of the raw view: `df_j` of decision D7, and the same sparsity
    # pattern the structural columns were read from, so the two cannot disagree.
    raw = selected.matrix_for("raw")
    weights = rarity_weights(concept_support(raw.X), raw.X.shape[0])
    damped = sweep_space(
        selected,
        questions=dev,
        token_counts=token_counts,
        backend=query_backend,
        base_config=base_config,
        arms=(space.best_arm["query_operator"],),
        views=(space.best_arm["view"],),
        damping=RARITY,
        concept_weights=weights,
    )
    undamped = next(cell for cell in table[space.k].cells if cell.arm == space.best_arm)
    damping = choose_damping(undamped, damped[0])
    print(f"[INFO] damping: {damping.chosen} over {damping.runner_up} by {damping.margin:.4f}")

    conceptual = ConceptualRetriever(
        selected.matrix_for(space.best_arm["view"]),
        selected.dictionary,
        query_backend,
        arm=space.best_arm["query_operator"],
        damping=damping.chosen,
        concept_weights=None if damping.chosen == UNDAMPED else weights,
    )
    dense = DenseRetriever(vectors=vectors, unit_ids=cached_unit_ids, backend=query_backend)

    print(f"[INFO] fitting the fusion over {len(config.FUSION_WEIGHT_GRID)} weights, twice")
    fusion = fit_fusion_weight(
        [dense, conceptual],
        questions=dev,
        token_counts=token_counts,
        base_config=base_config,
    )
    control = fit_fusion_weight(
        [dense, BM25Retriever(units)],
        questions=dev,
        token_counts=token_counts,
        base_config=base_config,
    )
    print(
        f"[INFO] fusion: {fusion.winning_scheme} at w={fusion.best_weight:.1f} "
        f"({fusion.best_weighted_score:.4f} weighted against {fusion.rrf_score:.4f} rrf)"
    )
    print(
        f"[INFO] control: {control.winning_scheme} at w={control.best_weight:.1f} "
        f"({control.best_weighted_score:.4f} weighted against {control.rrf_score:.4f} rrf)"
    )

    report = freeze_selection(
        table,
        space=space,
        damping=damping,
        fusion=fusion,
        base_config=base_config,
        control=control,
    )
    path = save_selection(report, selection_dir)
    print(
        f"[OK] {report.n_dev_decisions} decisions frozen at {report.frozen_at} "
        f"for k={report.selected_k} -> {path}"
    )
    return path


def _inherited_space(
    report: SelectionReport,
    concepts_dir: Path,
    unit_ids: Sequence[str],
) -> tuple[ConceptDictionary, ConceptMatrix, np.ndarray | None, dict]:
    """The Phase 3 configuration, loaded and hash-verified, never retyped.

    Returns the dictionary, the matrix in the view that was frozen, the damping
    weights if the frozen damping is not the identity, and the four decisions in the
    shape every Phase 4 artifact records them.
    """
    frozen = report.config
    key = str(frozen["dictionary_key"])
    view = str(frozen["view"])
    damping = str(frozen["damping"])
    try:
        dictionary = load_dictionary(
            key, concepts_dir, expected_unit_set_hash=frozen.get("unit_set_hash")
        )
        matrix = load_matrix(key, concepts_dir, view=view, expected_unit_ids=unit_ids)
        weights = None
        if damping != UNDAMPED:
            raw = load_matrix(key, concepts_dir, view="raw", expected_unit_ids=unit_ids)
            weights = rarity_weights(concept_support(raw.X), raw.X.shape[0])
    except ConceptArtifactError as error:
        _die(f"the frozen selection names a space that is not on disk ({error}): run 'induce'")

    inherits = {
        "k": int(frozen["k"]),
        "dictionary_key": key,
        "view": view,
        "damping": damping,
        "query_operator": str(frozen["query_operator"]),
    }
    return dictionary, matrix, weights, inherits


def cmd_expand(
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    concepts_dir: Path = config.CONCEPTS_DIR,
    selection_dir: Path = config.SELECTION_DIR,
    expansion_dir: Path = config.EXPANSION_DIR,
    question_cache_dir: Path = config.QUESTION_CACHE_DIR,
    top_k: int = config.EVALUATION_TOP_K,
    backend: EmbeddingBackend | None = None,
) -> Path:
    """Sweep the expansion grid on dev, choose one cell, and freeze it.

    Two free parameters and one decision (D7, D8). Everything else is either declared
    in `config.py` or **inherited from the frozen Phase 3 selection**, which is read
    from disk and verified rather than retyped here: if that artifact's digest does
    not check out, or it names a dictionary that is not the one on disk, this stage
    refuses to run.

    Nothing here reads the test split. The guard inside the grid refuses a question
    from any other split at the door, whatever this function passes it.
    """
    data_dir = Path(data_dir)
    concepts_dir = Path(concepts_dir)
    expansion_dir = Path(expansion_dir)
    pool_path = data_dir / POOL_NAME
    tokens_path = data_dir / TOKENS_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")
    if not tokens_path.exists():
        _die("no token counts found: run 'embed' first")
    # Checked before the sweep rather than at the write: a configuration freezes once,
    # and discovering that after twenty-four evaluations would waste the whole spend.
    if expansion_path(expansion_dir).exists():
        _die(
            f"a selection is already frozen at {expansion_path(expansion_dir)}; this phase "
            "writes one and never overwrites it. A second freeze is a deviation to be "
            "recorded, so move the existing artifact aside deliberately if that is meant"
        )

    units, questions = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    unit_ids = [unit.unit_id for unit in units]
    pool_hash = unit_set_hash(unit_ids)
    token_counts = json.loads(tokens_path.read_text(encoding="utf-8"))

    backend = backend if backend is not None else SentenceTransformerBackend()
    corpus = EmbeddingCache(Path(cache_dir)).load(
        _cache_key_for(backend, units), expected_unit_ids=unit_ids
    )
    if corpus is None:
        _die("embeddings are not cached for this pool: run 'embed' first")
    vectors, cached_unit_ids = corpus

    known = [path.stem.split("-", maxsplit=1)[1] for path in concepts_dir.glob("dictionary-*.npz")]
    try:
        inherited = load_selection(
            selection_dir, expected_unit_set_hash=pool_hash, known_dictionary_keys=known
        )
    except SelectionError as error:
        _die(
            f"this phase inherits its space, view, damping and query operator from a frozen "
            f"Phase 3 selection, and there is none to read ({error}): run 'select' first"
        )
    print(f"[INFO] inheriting the selection frozen at {inherited.frozen_at}")

    dictionary, matrix, weights, inherits = _inherited_space(inherited, concepts_dir, unit_ids)

    dev = [question for question in questions if question.split == DEV_SPLIT]
    if not dev:
        _die(f"the pool holds no {DEV_SPLIT} questions: run 'build' first")

    question_cache = EmbeddingCache(Path(question_cache_dir))
    query_backend = build_query_backend(dev, backend, question_cache)

    seed_retrievers: dict[str, Retriever] = {
        "dense": DenseRetriever(vectors=vectors, unit_ids=cached_unit_ids, backend=query_backend),
        "conceptual": ConceptualRetriever(
            matrix,
            dictionary,
            query_backend,
            arm=inherits["query_operator"],
            damping=inherits["damping"],
            concept_weights=weights,
        ),
    }

    base_config = {
        "model": backend.name,
        "revision": backend.revision,
        "resolved_revision": resolved_revision(backend),
        "unit_set_hash": pool_hash,
        "seed": config.DEFAULT_SEED,
        "tokenizer": config.TOKENIZER_ID,
        "code_version": __version__,
        "top_k": top_k,
        "n_units": len(units),
    }

    cells_total = len(config.RESTART_GRID) * len(config.NORMALIZATION_ARMS) * len(config.SEED_ARMS)
    print(
        f"[INFO] sweeping {cells_total} cells ({len(config.RESTART_GRID)} restarts x "
        f"{len(config.NORMALIZATION_ARMS)} normalizations x {len(config.SEED_ARMS)} arms) "
        f"over {len(dev)} {DEV_SPLIT} questions"
    )
    try:
        cells, spent = sweep_expansion(
            matrix=matrix,
            dictionary=dictionary,
            seed_retrievers=seed_retrievers,
            questions=dev,
            token_counts=token_counts,
            base_config=base_config,
            inherits=inherits,
            concept_weights=weights,
        )
        choice = choose_cell(cells)
        report = freeze_expansion_selection(
            cells,
            choice=choice,
            base_config=base_config,
            inherits=inherits,
            inherits_selection_digest=selection_digest(inherited),
            dev_evaluations_spent=spent,
        )
        path = save_expansion_selection(report, expansion_dir)
    except SelectionError as error:
        _die(str(error))

    print(
        f"[INFO] cell: {choice.label} at {config.EXPANSION_SELECTION_METRIC} "
        f"{choice.primary:.4f} (runner-up {choice.runner_up}, margin {choice.margin:.4f}, "
        f"resolution {choice.resolution:.4f})"
    )
    if choice.tie_break is not None:
        print(
            f"[INFO] the margin was inside what {len(dev)} questions resolve, so cost "
            f"decided it -> {choice.tie_break}"
        )
    print(
        f"[OK] frozen at {report.frozen_at} after {spent} of "
        f"{config.DEV_EVALUATION_CAP} dev evaluations -> {path}"
    )
    return path


def cmd_pilot(
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    question_cache_dir: Path = config.QUESTION_CACHE_DIR,
    pilot_dir: Path = config.PILOT_DIR,
    backend: EmbeddingBackend | None = None,
) -> Path:
    """Freeze the Phase 5 pilot: the dev questions dense leaves a gold paragraph outside top-10.

    HU-1 of the Phase 5 spec. Nothing here reads a concept, an extraction or a test
    question, and nothing is measured: the stage records where every hop will start and
    what it will have to find, once, before any text-derived node exists.
    """
    data_dir = Path(data_dir)
    pool_path = data_dir / POOL_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")
    # Checked before the work rather than at the write: the pilot freezes once.
    if pilot_path(pilot_dir).exists():
        _die(
            f"a pilot is already frozen at {pilot_path(pilot_dir)}; this phase writes one and "
            "never overwrites it. Move it aside deliberately if a second freeze is meant"
        )

    units, questions = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    unit_ids = [unit.unit_id for unit in units]
    pool_hash = unit_set_hash(unit_ids)

    backend = backend if backend is not None else SentenceTransformerBackend()
    corpus = EmbeddingCache(Path(cache_dir)).load(
        _cache_key_for(backend, units), expected_unit_ids=unit_ids
    )
    if corpus is None:
        _die("embeddings are not cached for this pool: run 'embed' first")
    vectors, cached_unit_ids = corpus

    dev = [question for question in questions if question.split == DEV_SPLIT]
    if not dev:
        _die(f"the pool holds no {DEV_SPLIT} questions: run 'build' first")
    query_backend = build_query_backend(dev, backend, EmbeddingCache(Path(question_cache_dir)))

    try:
        pilot = build_pilot(
            dev,
            unit_ids=cached_unit_ids,
            vectors=vectors,
            query_backend=query_backend,
            model=backend.name,
            revision=backend.revision,
            unit_set_hash=pool_hash,
        )
        if not pilot.questions:
            _die(
                f"no {DEV_SPLIT} question leaves a gold paragraph outside dense's top-"
                f"{pilot.read_depth}; there is nothing for a second hop to find"
            )
        check_pilot_against(pilot, questions)
        path = save_pilot(pilot, pilot_dir)
    except PilotError as error:
        _die(str(error))

    print(
        f"[OK] pilot frozen: {len(pilot.questions)} of {len(dev)} {DEV_SPLIT} questions miss "
        f"{pilot.n_missing} gold paragraphs outside dense top-{pilot.read_depth} -> {path}"
    )
    return path


def cmd_extract(
    sample: bool = False,
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    extraction_dir: Path = config.EXTRACTION_DIR,
    client: SyncClient | None = None,
    batch_client: BatchClient | None = None,
    wait: Callable[[], None] | None = None,
) -> None:
    """Read entities and concepts out of every paragraph, offline (Phase 5, HU-2).

    `--sample` extracts the 20 lowest unit ids synchronously and reports what they cost and
    how long they are beside the corpus. Without it, the whole pool goes through batches -
    only after a sample exists and only under the ceiling, both checked before anything is
    submitted. This is the second stage after `label` that spends money, and the key it
    needs is read from `.env` when a real client is built, never before.
    """
    data_dir = Path(data_dir)
    pool_path = data_dir / POOL_NAME
    tokens_path = data_dir / TOKENS_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")
    if not tokens_path.exists():
        _die("no token counts found: run 'embed' first")
    units, _ = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    token_counts = json.loads(tokens_path.read_text(encoding="utf-8"))

    try:
        if sample:
            if client is None:
                read_api_key()
                client = AnthropicSyncClient()
            report = run_sample(
                units,
                client,
                cache_dir=cache_dir,
                extraction_dir=extraction_dir,
                token_counts=token_counts,
            )
            failed = {unit: status for unit, status in report.statuses.items() if status != "ok"}
            print(
                f"[INFO] sample: {report.n_ok} of {len(report.unit_ids)} paragraphs extracted "
                f"within the schema and bounds"
                + (f"; failures {sorted(failed.values())}" if failed else "")
            )
            print(
                f"[INFO] tokens per request: {report.mean_input_tokens:.1f} in, "
                f"{report.mean_output_tokens:.1f} out; the sample cost {report.sample_usd:.4f} USD"
            )
            for name, lengths in (
                ("sample", report.sample_lengths),
                ("corpus", report.corpus_lengths),
            ):
                print(
                    f"[INFO] {name} paragraph length (tokens): mean {lengths['mean']:.1f}, "
                    f"median {lengths['median']:.1f}, min {lengths['min']:.0f}, "
                    f"max {lengths['max']:.0f}"
                )
            print(
                f"[INFO] estimate for the full run: {report.estimated_full_usd:.2f} USD at batch "
                f"rates, output margin {config.EXTRACTION_ESTIMATE_MARGIN}; ceiling "
                f"{report.ceiling_usd:.2f} USD"
            )
            print("[OK] sample written; the full run is `cer extract`, after the spend is approved")
            return

        try:
            load_sample_report(extraction_dir)
        except ExtractionError:
            _die("no sample report found: run 'extract --sample' first, and read its estimate")
        if batch_client is None:
            read_api_key()
            batch_client = AnthropicBatchClient()
        summary = run_full(
            units,
            batch_client,
            cache_dir=cache_dir,
            extraction_dir=extraction_dir,
            wait=wait if wait is not None else _poll_pause,
        )
    except MissingAPIKey as error:
        _die(str(error))
    except (ExtractionError, LabelingError) as error:
        _die(str(error))

    print(
        f"[INFO] {summary.n_units} paragraphs: {summary.calls} requested, {summary.cached} already "
        f"cached, {summary.resubmissions} resubmitted once"
    )
    print(
        f"[INFO] cost: {summary.actual_usd:.2f} USD actual against {summary.estimated_usd:.2f} USD "
        "estimated"
    )
    if summary.finding:
        print(
            f"[WARN] {summary.failure_rate:.2%} of the pool failed extraction "
            f"({summary.failures}), above the {config.EXTRACTION_FAILURE_FINDING:.0%} finding "
            "threshold: stop here, write the finding, and build no nodes"
        )
    elif summary.failures:
        print(f"[INFO] failures {summary.failures} ({summary.failure_rate:.2%} of the pool)")
    print(f"[OK] extraction written to {extraction_dir}")


def cmd_nodes(
    data_dir: Path = config.DATA_DIR,
    extraction_dir: Path = config.EXTRACTION_DIR,
    nodes_dir: Path = config.NODES_DIR,
) -> Path:
    """Normalize the extraction into typed nodes and index them over the pool (Phase 5, HU-3).

    Refuses an extraction whose failures are a finding (plan D5, task T13): above 1% of the
    pool, the finding is written and no node is built from it. Refuses one that does not
    cover every pool unit. Prints the fragmentation figures and never a node form, which is
    corpus text and has no place in a log.
    """
    data_dir = Path(data_dir)
    pool_path = data_dir / POOL_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")
    units, _ = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    unit_ids = [unit.unit_id for unit in units]

    try:
        summary = load_extraction_summary(extraction_dir)
        if summary["finding"]:
            _die(
                f"the extraction failed on {summary['failure_rate']:.2%} of the pool "
                f"({summary['failures']}), a finding above the "
                f"{config.EXTRACTION_FAILURE_FINDING:.0%} threshold: write the finding; no nodes "
                "are built from this extraction"
            )
        records = load_extraction(extraction_dir, expected_unit_ids=unit_ids)
        index = build_node_index(records, unit_ids, extraction_digest=str(summary["digest"]))
    except (ExtractionError, NodeIndexError) as error:
        _die(str(error))
    path = save_node_index(index, nodes_dir)

    print(
        f"[INFO] {len(unit_ids)} paragraphs, {index.failed_units} with a failed extraction, "
        f"{len(index.nodes)} nodes ({config.NORMALIZATION_VERSION})"
    )
    for node_type, figures in fragmentation(index).items():
        per_node = figures["paragraphs_per_node"]
        per_paragraph = figures["nodes_per_paragraph"]
        print(
            f"[INFO] {node_type}: {figures['distinct_nodes']} distinct, "
            f"{figures['singleton_share']:.1%} in a single paragraph; paragraphs per node mean "
            f"{per_node['mean']:.2f}, median {per_node['median']:.1f}, max {per_node['max']}; "
            f"nodes per paragraph mean {per_paragraph['mean']:.2f}; "
            f"{figures['paragraphs_without_node']} paragraphs without one; "
            f"{index.dropped_empty[node_type]} forms empty after normalization"
        )
    print(f"[OK] node index written -> {path}")
    return path


def cmd_navigate(
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    question_cache_dir: Path = config.QUESTION_CACHE_DIR,
    concepts_dir: Path = config.CONCEPTS_DIR,
    pilot_dir: Path = config.PILOT_DIR,
    extraction_dir: Path = config.EXTRACTION_DIR,
    nodes_dir: Path = config.NODES_DIR,
    navigation_dir: Path = config.NAVIGATION_DIR,
    backend: EmbeddingBackend | None = None,
) -> Path:
    """Run every second hop over the frozen pilot, once (Phase 5, HU-5 and HU-6).

    Everything it reads is frozen and verified first: the pilot, the extraction summary,
    the node index and the pooled-embedding space. It refuses to measure a second time
    before doing any work, because the gate may not be computed from a second measurement.
    """
    navigation_dir = Path(navigation_dir)
    if (navigation_dir / HOP_RUN_FILENAME).exists():
        _die(
            f"a navigation run is already written in {navigation_dir}; this phase measures once, "
            "and the gate may not be re-run on a second measurement"
        )
    data_dir = Path(data_dir)
    pool_path = data_dir / POOL_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")
    units, questions = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    unit_ids = [unit.unit_id for unit in units]
    pool_hash = unit_set_hash(unit_ids)

    try:
        pilot = load_pilot(pilot_dir, expected_unit_set_hash=pool_hash)
        check_pilot_against(pilot, questions)
    except PilotError as error:
        _die(f"{error}; run 'pilot' first if none is frozen")

    try:
        summary = load_extraction_summary(extraction_dir)
        if summary["finding"]:
            _die("the extraction's failures are a finding: no navigation is measured on it")
        index = load_node_index(
            nodes_dir, extraction_digest=str(summary["digest"]), expected_unit_ids=unit_ids
        )
    except ExtractionError as error:
        _die(f"{error}")
    except NodeIndexError as error:
        _die(f"{error}; run 'nodes' first")

    backend = backend if backend is not None else SentenceTransformerBackend()
    corpus = EmbeddingCache(Path(cache_dir)).load(
        _cache_key_for(backend, units), expected_unit_ids=unit_ids
    )
    if corpus is None:
        _die("embeddings are not cached for this pool: run 'embed' first")
    vectors, _ = corpus
    by_id = {question.qid: question for question in questions}
    pilot_questions = [by_id[entry.qid] for entry in pilot.questions]
    query_backend = build_query_backend(
        pilot_questions, backend, EmbeddingCache(Path(question_cache_dir))
    )

    reference_key = dictionary_key(
        k=config.REFERENCE_CONCEPT_K,
        seed=config.CONCEPT_SEED,
        alpha=config.INDUCTION_ALPHA,
        unit_set_hash=pool_hash,
        model=config.EMBEDDING_MODEL,
        revision=config.EMBEDDING_REVISION,
        merge_threshold=config.MERGE_COSINE_THRESHOLD,
    )
    try:
        reference = load_matrix(
            reference_key, Path(concepts_dir), view="raw", expected_unit_ids=unit_ids
        ).X
    except ConceptArtifactError as error:
        _die(f"the reference concept space is not on disk ({error}): run 'induce'")
    reference = sparse.csr_matrix(reference, dtype=np.float64)

    run = run_navigation(
        pilot,
        by_id,
        unit_ids=unit_ids,
        texts=[unit.indexable_text for unit in units],
        vectors=vectors,
        query_backend=query_backend,
        bm25=BM25Retriever(units),
        index=index,
        concept_matrix=reference,
        concept_weights=rarity_weights(concept_support(reference), reference.shape[0]),
        provenance={
            "pilot_digest": pilot_digest(pilot),
            "node_index_digest": index.digest,
            "normalization_version": index.normalization_version,
            "extraction_digest": str(summary["digest"]),
            "extraction_model": str(summary["model"]),
            "prompt_digest": str(summary["prompt_digest"]),
            "unit_set_hash": pool_hash,
            "model": backend.name,
            "revision": backend.revision,
            "reference_dictionary_key": reference_key,
            "code_version": __version__,
            "stochastic": False,
        },
    )
    run_path, traces_path = save_navigation(run, navigation_dir)

    print(f"[INFO] {len(pilot.questions)} pilot questions, {pilot.n_missing} missing paragraphs")
    print(f"[INFO] {'hop':<34} {'q@10':>6} {'q@100':>6} {'p@10':>6} {'p@100':>6}")
    for hop, figures in run.payload["aggregates"].items():
        q, p = figures["per_question"], figures["per_paragraph"]
        print(f"[INFO] {hop:<34} {q['10']:6.3f} {q['100']:6.3f} {p['10']:6.3f} {p['100']:6.3f}")

    diagnostic_path = data_dir / "diagnostics" / "second_hop-dev.json"
    if diagnostic_path.exists():
        mismatches = diagnostic_mismatches(
            load_hop_run(navigation_dir), json.loads(diagnostic_path.read_text(encoding="utf-8"))
        )
        if mismatches:
            for mismatch in mismatches:
                print(f"[WARN] continuity with the Phase 4 diagnostic: {mismatch}")
        else:
            print("[OK] the non-concept hops reproduce the Phase 4 diagnostic exactly")
    print(f"[OK] navigation run -> {run_path}; traces -> {traces_path}")

    # The gate reads the run as written and verified, never the object in memory.
    decision = compute_gate(load_hop_run(navigation_dir))
    gate_path = save_gate(decision, navigation_dir)
    for name, test in decision["tests"].items():
        print(
            f"[INFO] gate test '{name}': {test['wins']} wins, {test['losses']} losses, "
            f"{test['ties']} ties, p = {test['p_value']:.4g} -> "
            f"{'passed' if test['passed'] else 'not passed'}"
        )
    for name, test in decision["reported"].items():
        print(
            f"[INFO] reported only '{name}': {test['wins']} wins, {test['losses']} losses, "
            f"p = {test['p_value']:.4g}"
        )
    for anomaly in decision["anomalies"]:
        print(f"[WARN] anomaly: {anomaly}")
    print(f"[OK] gate outcome: {decision['outcome']} -> {gate_path}")
    return run_path


def replacement_environment() -> StageEnvironment:
    """The real inputs, pins and output directory of Phase 6, all read from `config`."""
    return StageEnvironment(
        paths=InputPaths.from_config(),
        pins=InputPins.from_config(),
        replacement_dir=config.REPLACEMENT_DIR,
        backend=SentenceTransformerBackend(),
        token_counter=offline_token_counter(),
    )


def cmd_replace_check(environment: StageEnvironment | None = None) -> int:
    """Phase 6, D11 steps 1-3 on dev: inputs, continuity, control reproduction. No B figure.

    Exits non-zero on any mismatch, after writing `checks-dev.json`; the deviation document is
    the author's to write before anything further runs.
    """
    try:
        result = run_check(environment or replacement_environment())
    except ReplacementRunError as error:
        print(f"[ERROR] {error}")
        return 1
    status = "OK" if result.passed else "ERROR"
    print(f"[{status}] {result.message} -> {result.path}")
    return 0 if result.passed else 1


def cmd_replace_freeze(
    environment: StageEnvironment | None = None,
    control_mode: str | None = None,
    deviation: str | None = None,
    supersede: bool = False,
) -> int:
    """Phase 6, D11 steps 4-8 on dev: B's fit and selected run, reproducibility, the freeze.

    Refuses without a passing `checks-dev.json`, or, after a failed one, without
    `--control-mode re-measured --deviation <existing document>`. The freeze is written once and
    must be committed before `replace-test` runs.
    """
    try:
        result = run_freeze(
            environment or replacement_environment(),
            control_mode=control_mode,
            deviation=deviation,
            supersede=supersede,
        )
    except ReplacementRunError as error:
        print(f"[ERROR] {error}")
        return 1
    print(f"[OK] {result.message} -> {result.path}")
    return 0


def cmd_replace_test(
    environment: StageEnvironment | None = None,
    control_mode: str | None = None,
    deviation: str | None = None,
) -> int:
    """Phase 6, D14 on test, once: dense, A, the reproduction check, B, the decision.

    Prints the state and the route, never the recorded texts (D21). Exits non-zero on any
    refusal or stop; a stop before B leaves no B figure.
    """
    try:
        result = run_test(
            environment or replacement_environment(),
            control_mode=control_mode,
            deviation=deviation,
        )
    except ReplacementRunError as error:
        print(f"[ERROR] {error}")
        return 1
    status = "OK" if result.passed else "ERROR"
    print(f"[{status}] {result.message} -> {result.path}")
    return 0 if result.passed else 1


ExtractorBuilder = Callable[[str, Path], tuple[LocalExtractor, float]]


def _progress(label: str, started: float) -> Callable[[int, int], None]:
    """Say how far the pass has got and what that projects to, in ASCII and no node form."""
    reported = [0]

    def report(done: int, total: int) -> None:
        if done - reported[0] < 1000 and done < total:
            return
        reported[0] = done
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate / 60.0 if rate > 0 else float("nan")
        print(
            f"[INFO] {label}: {done}/{total} texts in {elapsed / 60:.1f} min "
            f"({rate:.2f} texts/s, about {remaining:.1f} min left)"
        )

    return report


def cmd_cheap_extract(
    extractor_id: str,
    data_dir: Path = config.DATA_DIR,
    phase_7_dir: Path = config.PHASE_7_DIR,
    hourly_rate_usd: float = 0.0,
    actual_cost_usd: float | None = None,
    build: ExtractorBuilder = local_extraction.build_extractor,
    chunk_size: int = 64,
) -> Path:
    """Read entities out of every paragraph with one local extractor (Phase 7, S3).

    Writes `data/phase7/<extractor_id>/`: the records archive and the manifest that says
    which model, revision, label set, library versions and hardware produced them, how long
    the pass took, what it cost (0.00 USD on owned hardware) and what that projects to at
    five million paragraphs. Nothing is written to `data/extraction/` or `data/nodes/`,
    which are Phase 5 artifacts.

    It refuses to overwrite an existing extraction: the pass costs hours, and silently
    replacing the artifact behind a measured dev figure is exactly what the project's
    versioning rule forbids. Prints figures only - never a node form, which is corpus text.
    """
    if not math.isfinite(hourly_rate_usd) or hourly_rate_usd < 0:
        _die("hourly-rate-usd must be finite and non-negative")
    if actual_cost_usd is not None and (not math.isfinite(actual_cost_usd) or actual_cost_usd < 0):
        _die("actual-cost-usd must be finite and non-negative")
    if hourly_rate_usd > 0 and actual_cost_usd is None:
        _die(
            "rented compute requires --actual-cost-usd: record the actual charge, not a projection"
        )
    if extractor_id not in config.PHASE_7_EXTRACTORS:
        _die(
            f"unknown extractor {extractor_id!r}; this phase has exactly two candidates, "
            f"{','.join(config.PHASE_7_EXTRACTORS)}"
        )
    data_dir = Path(data_dir)
    pool_path = data_dir / POOL_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")
    units, _questions = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)

    directory = Path(phase_7_dir) / extractor_id
    if (directory / local_extraction.MANIFEST_NAME).exists():
        _die(
            f"{directory} already holds an extraction; this pass costs hours and does not "
            "overwrite the artifact a measured figure was read from - remove the directory "
            "deliberately to run it again"
        )

    try:
        extractor, load_seconds = build(extractor_id, Path(phase_7_dir) / "models" / extractor_id)
    except LocalExtractionError as error:
        _die(str(error))
    except ImportError as error:
        _die(
            f"the {extractor_id} library is not installed in this environment ({error}); "
            "install the phase7 dependency group first"
        )

    print(
        f"[INFO] {extractor_id}: {extractor.model} at revision {extractor.revision}, "
        f"{len(extractor.labels)} labels, configuration {extractor.configuration_digest}"
    )
    print(f"[INFO] model loaded in {load_seconds:.1f} s; reading {len(units)} paragraphs")

    started = time.perf_counter()
    try:
        records, seconds = local_extraction.run_extraction(
            units,
            extractor,
            chunk_size=chunk_size,
            on_progress=_progress(extractor_id, started),
        )
        manifest = local_extraction.build_manifest(
            records,
            extractor,
            seconds=seconds,
            hardware=local_extraction.hardware_block(device=extractor.hardware_device),
            usd=actual_cost_usd if actual_cost_usd is not None else 0.0,
            hourly_rate_usd=hourly_rate_usd,
            model_load_seconds=load_seconds,
        )
        path = local_extraction.write_extraction(records, manifest, directory)
    except LocalExtractionError as error:
        _die(str(error))

    projection = manifest["projection_5m"]
    print(
        f"[INFO] {manifest['n_units']} paragraphs in {seconds / 60:.1f} min "
        f"({manifest['paragraphs_per_second']:.3f} paragraphs/s)"
    )
    print(
        f"[INFO] {manifest['entities']} entity mentions kept, "
        f"{manifest['units_without_entity']} paragraphs with none, "
        f"{sum(manifest['failures'].values())} failed ({manifest['failure_rate']:.4%})"
    )
    print(
        f"[INFO] projection to {projection['paragraphs']} paragraphs: "
        f"{projection['hours']:.1f} h, {projection['usd']:.2f} USD "
        f"({projection['method']}); this run cost {manifest['usd']:.2f} USD"
    )
    print(f"[OK] extraction written -> {path}")
    return path


def _cheap_inputs(
    split: str,
    data_dir: Path,
    cache_dir: Path,
    question_cache_dir: Path,
    backend: EmbeddingBackend | None,
) -> tuple[list[str], list[Question], dict[str, int], DenseRetriever, dict[str, Any]]:
    """The one loader both Phase 7 measurement paths share: pool, cache, dense, provenance.

    Nothing is embedded here. Both splits were embedded in earlier phases and their caches
    are the record; a missing cache is a refusal, never a quiet re-encoding at a different
    library version.
    """
    if not (data_dir / POOL_NAME).exists():
        _die("no pool found: run 'build' first")
    if not (data_dir / TOKENS_NAME).exists():
        _die("no token counts found: run 'embed' first")
    units, questions = load_pool(data_dir / POOL_NAME)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    subset = [question for question in questions if question.split == split]
    if not subset:
        _die(f"the pool holds no {split} questions: run 'build' first")
    unit_ids = [unit.unit_id for unit in units]
    token_counts = json.loads((data_dir / TOKENS_NAME).read_text(encoding="utf-8"))
    if set(token_counts) != set(unit_ids):
        _die("token counts belong to another pool")
    backend = backend if backend is not None else SentenceTransformerBackend()
    corpus_key = _cache_key_for(backend, units)
    cached = EmbeddingCache(Path(cache_dir)).load(corpus_key, expected_unit_ids=unit_ids)
    if cached is None:
        _die("embeddings are not cached for this pool: run 'embed' first")
    question_cache = EmbeddingCache(Path(question_cache_dir))
    query_key = question_cache_key(
        backend.name,
        backend.revision,
        question_set_hash([q.qid for q in subset]),
        split,
        normalized=bool(getattr(backend, "normalize", True)),
    )
    if question_cache.load(query_key, expected_unit_ids=[q.qid for q in subset]) is None:
        _die(
            f"{split} question embeddings are missing; restore the existing {split} cache "
            "before cheap-eval"
        )
    query_backend = build_query_backend(subset, backend, question_cache)
    dense = DenseRetriever(cached[0], unit_ids, query_backend)
    run_config = {
        "model": backend.name,
        "revision": backend.revision,
        "unit_set_hash": unit_set_hash(unit_ids),
        "seed": config.DEFAULT_SEED,
        "tokenizer": config.TOKENIZER_ID,
        "code_version": __version__,
        "top_k": config.EVALUATION_TOP_K,
        "n_units": len(units),
        "corpus_cache_key": corpus_key,
        "question_cache_key": query_key,
    }
    return unit_ids, subset, token_counts, dense, run_config


def cmd_cheap_eval(
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    question_cache_dir: Path = config.QUESTION_CACHE_DIR,
    phase_7_dir: Path = config.PHASE_7_DIR,
    backend: EmbeddingBackend | None = None,
    test: bool = False,
) -> list[Path]:
    """Measure every extracted candidate on dev and apply the rule; with `test`, read test.

    Without `--test` this is the whole dev half of the phase: one node index, one fusion
    fit and one measurement per candidate, then `selection.json` written from those
    readings. With `--test` it measures the selected extractor on the held-out split,
    once, at the scheme and weight dev already chose. The two never run together, which
    is what keeps the selection strictly upstream of the only test figure the phase has.
    """
    data_dir, phase_7_dir = Path(data_dir), Path(phase_7_dir)
    if test:
        return [_cheap_test(data_dir, cache_dir, question_cache_dir, phase_7_dir, backend)]
    if (phase_7_dir / SELECTION_FILENAME).exists():
        _die(
            f"{phase_7_dir / SELECTION_FILENAME} already names a selection; the dev comparison "
            "is closed and a candidate measured after it would be chosen against a known result"
        )
    candidates = [
        name
        for name in config.PHASE_7_EXTRACTORS
        if (phase_7_dir / name / local_extraction.MANIFEST_NAME).exists()
    ]
    if not candidates:
        _die("no local extraction found: run 'cheap-extract' first")
    written: list[Path] = []
    pending: list[str] = []
    for name in candidates:
        path = phase_7_dir / name / DEV_FILENAME
        if path.exists():
            # Preserve earlier measurements when a second extraction arrives, or when a
            # previous invocation stopped between candidates. Selection verifies them.
            print(f"[INFO] preserving existing dev artifact for {name}: {path}")
            written.append(path)
        else:
            pending.append(name)
    if pending:
        unit_ids, dev, token_counts, dense, run_config = _cheap_inputs(
            DEV_SPLIT, data_dir, cache_dir, question_cache_dir, backend
        )
        check_dev_only(dev)
        for name in pending:
            print(f"[INFO] evaluating {name} on dev ({len(dev)} questions)")
            try:
                path = evaluate_candidate(
                    phase_7_dir / name,
                    extractor_id=name,
                    unit_ids=unit_ids,
                    dense=dense,
                    questions=dev,
                    token_counts=token_counts,
                    run_config=run_config,
                )
            except (
                CheapEvaluationError,
                LocalExtractionError,
                NodeIndexError,
                SelectionError,
            ) as error:
                _die(str(error))
            written.append(path)
            print(f"[OK] dev measurement written -> {path}")
    try:
        selection_path = select_extractor(phase_7_dir)
    except CheapEvaluationError as error:
        _die(str(error))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for row in selection["candidates"]:
        retention = row["retention"]
        print(
            f"[INFO] {row['extractor_id']}: dev full support @{retention['headline_budget']} "
            f"{retention['full_support_at_headline']:.4f} "
            f"({retention['supported_questions']}/{retention['n_questions']}, bar "
            f"{retention['bar_questions']}), retention "
            f"{'PASS' if retention['passed'] else 'FAIL'}, economic "
            f"{'PASS' if row['economics']['passed'] else 'FAIL'}"
        )
    if selection["candidates_without_dev_measurement"]:
        missing = ",".join(selection["candidates_without_dev_measurement"])
        print(f"[WARN] no dev measurement for {missing}; the rule was applied without them")
    print(f"[INFO] selected: {selection['selected']} - {selection['reason']}")
    print(f"[OK] selection written -> {selection_path}")
    written.append(selection_path)
    return written


def _cheap_test(
    data_dir: Path,
    cache_dir: Path,
    question_cache_dir: Path,
    phase_7_dir: Path,
    backend: EmbeddingBackend | None,
) -> Path:
    """The single held-out run of the selected extractor (D12)."""
    unit_ids, questions, token_counts, dense, run_config = _cheap_inputs(
        TEST_SPLIT, data_dir, cache_dir, question_cache_dir, backend
    )
    try:
        path = measure_held_out(
            phase_7_dir,
            unit_ids=unit_ids,
            dense=dense,
            questions=questions,
            token_counts=token_counts,
            run_config=run_config,
        )
    except (CheapEvaluationError, NodeIndexError) as error:
        _die(str(error))
    payload = json.loads(path.read_text(encoding="utf-8"))
    headline = payload["metrics"][f"budget_{config.SELECTION_BUDGET}"]["full_support"]
    print(
        f"[INFO] {payload['extractor_id']} on test ({payload['n_questions']} questions): "
        f"full support @{config.SELECTION_BUDGET} {headline:.4f}"
    )
    for name, value in sorted(payload["inherited_test_full_support"].items()):
        print(f"[INFO] inherited {name} test full support: {value:.4f}")
    print(f"[OK] held-out run written -> {path}")
    return path


# --- Phase 8: Strong Dense + Entity Hop -------------------------------------------------
#
# Two stages, two modes each, in the order the plan fixes: embed the corpus and the dev
# questions under the new Dense model, run the dev gate, and only then encode and measure
# the held-out split. Neither stage ever calls `TokenCounter`, so neither can rewrite
# `data/token_counts.json` - the ruler behind every inherited figure (R1).

SnapshotResolver = Callable[[str, str], Path]


def _embedding_device() -> str:
    """Where the encoding actually ran, as observed rather than as assumed."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _max_seq_length(backend: EmbeddingBackend) -> int | None:
    reader = getattr(backend, "max_seq_length", None)
    return reader() if callable(reader) else None


def _phase_8_question_key(questions: Sequence[Question], backend: EmbeddingBackend) -> str:
    """The question cache key, prompt included: an asymmetric model keys queries by it."""
    splits = sorted({question.split for question in questions})
    if len(splits) != 1:
        _die(f"the questions handed to this stage span more than one split: {splits}")
    return question_cache_key(
        backend.name,
        backend.revision,
        question_set_hash([question.qid for question in questions]),
        splits[0],
        normalized=bool(getattr(backend, "normalize", True)),
        query_prompt=query_prompt_of(backend),
    )


def _refuse_existing_cache(cache: EmbeddingCache, key: str, label: str) -> None:
    """A stage that measures an encoding may not record a wall clock it did not take."""
    if cache.path_for(key).exists():
        _die(
            f"the Phase 8 {label} embedding cache {key} already exists; this stage measures "
            "the encoding pass and refuses to write a timing for work it did not do - remove "
            f"{cache.path_for(key).name} deliberately to encode it again"
        )


def _phase_8_backend(
    settings: strong_dense.Phase8Pins, *, prompted: bool
) -> SentenceTransformerBackend:
    return SentenceTransformerBackend(
        name=settings.model,
        revision=settings.revision,
        dim=settings.dim,
        query_prompt=settings.query_prompt if prompted else "",
    )


def _scale_backend(
    pins: scale_sensitivity.ModelPins, *, prompted: bool
) -> SentenceTransformerBackend:
    """Reuse the Phase 8 backend shape with the model pins chosen by deviation 8.1."""
    return SentenceTransformerBackend(
        name=pins.model,
        revision=pins.revision,
        dim=pins.dim,
        query_prompt=pins.query_prompt if prompted else "",
    )


def cmd_strong_embed(
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    question_cache_dir: Path = config.QUESTION_CACHE_DIR,
    phase_8_dir: Path = config.PHASE_8_DIR,
    hourly_rate_usd: float = 0.0,
    test: bool = False,
    corpus_backend: EmbeddingBackend | None = None,
    query_backend: EmbeddingBackend | None = None,
    resolve_snapshot: SnapshotResolver = snapshot_directory,
    pins: strong_dense.Phase8Pins | None = None,
) -> Path:
    """Encode the pool and one split under the Phase 8 Dense model (Phase 8, S3 and S5).

    Without `--test` this is the initial pass: the provenance checks that need no
    measurement run **before the model is loaded**, the 19,366 paragraphs and the 600 dev
    questions are encoded under new cache keys, every block is checked for the pinned
    width before anything is cached, and `embedding.json` records which model, which
    weights and which caches produced the vectors.

    With `--test` it encodes the held-out questions, and only if the dev gate passed. The
    two modes never run together, which is what keeps the held-out split unencoded until
    Strong Dense has proved itself on dev.

    Existing caches are never overwritten and `data/token_counts.json` is never touched.
    """
    if not math.isfinite(hourly_rate_usd) or hourly_rate_usd < 0:
        _die("hourly-rate-usd must be finite and non-negative")
    settings = pins if pins is not None else strong_dense.Phase8Pins.from_config()
    data_dir, phase_8_dir = Path(data_dir), Path(phase_8_dir)
    pool_path = data_dir / POOL_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")
    units, questions = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    phase_8_dir.mkdir(parents=True, exist_ok=True)

    if test:
        return _strong_embed_test(
            questions,
            question_cache_dir=Path(question_cache_dir),
            phase_8_dir=phase_8_dir,
            hourly_rate_usd=hourly_rate_usd,
            query_backend=query_backend,
            settings=settings,
        )

    target = phase_8_dir / strong_dense.EMBEDDING_FILENAME
    if target.exists():
        _die(
            f"{target} already records this phase's encoding; it is written once and the dev "
            "fit is anchored on its digest"
        )
    dev = [question for question in questions if question.split == DEV_SPLIT]
    if not dev:
        _die("the pool holds no dev questions: run 'build' first")

    # D11: every check that needs no measurement runs here, before the first model load
    # and before a single vector exists. A check that only happens while writing the
    # report is a check that cannot stop a bad run.
    try:
        strong_dense.check_model_pin(settings.revision, settings.model)
        snapshot = Path(resolve_snapshot(settings.model, settings.revision))
        strong_dense.check_resolved_revision(snapshot.name, settings.revision)
        weights_digest = weights_sha256(snapshot)
    except (strong_dense.StrongDenseError, BackendError) as error:
        _die(str(error))
    print(
        f"[OK] model provenance verified: {settings.model} at {snapshot.name}, "
        f"{config.PHASE_8_WEIGHTS_FILE} sha256 {weights_digest[:16]}..."
    )

    corpus = (
        corpus_backend if corpus_backend is not None else _phase_8_backend(settings, prompted=False)
    )
    queries = (
        query_backend if query_backend is not None else _phase_8_backend(settings, prompted=True)
    )
    guarded_corpus = strong_dense.WidthCheckedBackend(corpus, settings.dim)
    guarded_queries = strong_dense.WidthCheckedBackend(queries, settings.dim)

    cache, question_cache = (
        EmbeddingCache(Path(cache_dir)),
        EmbeddingCache(Path(question_cache_dir)),
    )
    corpus_key = _cache_key_for(guarded_corpus, units)
    query_key = _phase_8_question_key(dev, guarded_queries)
    _refuse_existing_cache(cache, corpus_key, "corpus")
    _refuse_existing_cache(question_cache, query_key, "dev question")

    print(f"[INFO] encoding {len(units)} paragraphs and {len(dev)} dev questions")
    try:
        started = time.perf_counter()
        vectors, unit_ids = embed_units(units, guarded_corpus, cache)
        corpus_seconds = time.perf_counter() - started
        started = time.perf_counter()
        build_query_backend(dev, guarded_queries, question_cache)
        question_seconds = time.perf_counter() - started
    except (strong_dense.StrongDenseError, CacheAlignmentError, QueryPromptError) as error:
        _die(str(error))

    bytes_on_disk = sum(
        path.stat().st_size
        for path in (
            cache.path_for(corpus_key),
            cache.sidecar_for(corpus_key),
            question_cache.path_for(query_key),
            question_cache.sidecar_for(query_key),
        )
        if path.exists()
    )
    try:
        path = strong_dense.write_embedding(
            phase_8_dir,
            resolved_revision=snapshot.name,
            weights_sha256=weights_digest,
            max_seq_length=_max_seq_length(corpus),
            query_prompt=query_prompt_of(guarded_queries),
            corpus_cache_key=corpus_key,
            question_cache_key=query_key,
            unit_ids=unit_ids,
            n_questions=len(dev),
            dim=int(vectors.shape[1]),
            hardware=local_extraction.hardware_block(device=_embedding_device()),
            corpus_seconds=corpus_seconds,
            question_seconds=question_seconds,
            bytes_on_disk=bytes_on_disk,
            hourly_rate_usd=hourly_rate_usd,
            pins=settings,
        )
    except strong_dense.StrongDenseError as error:
        _die(str(error))
    payload = json.loads(path.read_text(encoding="utf-8"))
    projection_5m = payload["projection_5m"]
    print(
        f"[INFO] {payload['n_units']} paragraphs in {corpus_seconds / 60:.1f} min "
        f"({payload['paragraphs_per_second']:.3f} paragraphs/s), "
        f"{payload['dim']} dimensions, max_seq_length {payload['max_seq_length']}"
    )
    print(
        f"[INFO] projection to {projection_5m['paragraphs']} paragraphs: "
        f"{projection_5m['hours']:.1f} h, {projection_5m['usd']:.2f} USD "
        f"({projection_5m['method']}); this pass costs {payload['embedding_usd']:.2f} USD"
    )
    print(f"[OK] embedding artifact written -> {path}")
    return path


def _strong_embed_test(
    questions: Sequence[Question],
    *,
    question_cache_dir: Path,
    phase_8_dir: Path,
    hourly_rate_usd: float,
    query_backend: EmbeddingBackend | None,
    settings: strong_dense.Phase8Pins,
) -> Path:
    """The gated held-out encoding: it runs only after the dev stop rule said `pass`."""
    if (phase_8_dir / strong_dense.EMBEDDING_TEST_FILENAME).exists():
        _die(
            f"{phase_8_dir / strong_dense.EMBEDDING_TEST_FILENAME} already records the held-out "
            "encoding; it is written once"
        )
    try:
        strong_dense.check_gate(strong_dense.load_dev(phase_8_dir))
    except strong_dense.StrongDenseError as error:
        _die(str(error))
    held_out = [question for question in questions if question.split == TEST_SPLIT]
    if not held_out:
        _die(f"the pool holds no {TEST_SPLIT} questions: run 'build' first")

    queries = (
        query_backend if query_backend is not None else _phase_8_backend(settings, prompted=True)
    )
    guarded = strong_dense.WidthCheckedBackend(queries, settings.dim)
    question_cache = EmbeddingCache(Path(question_cache_dir))
    query_key = _phase_8_question_key(held_out, guarded)
    _refuse_existing_cache(question_cache, query_key, "held-out question")

    print(f"[INFO] encoding {len(held_out)} held-out questions")
    try:
        started = time.perf_counter()
        build_query_backend(held_out, guarded, question_cache)
        question_seconds = time.perf_counter() - started
        path = strong_dense.write_test_embedding(
            phase_8_dir,
            question_cache_key=query_key,
            n_questions=len(held_out),
            hardware=local_extraction.hardware_block(device=_embedding_device()),
            question_seconds=question_seconds,
            hourly_rate_usd=hourly_rate_usd,
            pins=settings,
        )
    except (strong_dense.StrongDenseError, CacheAlignmentError, QueryPromptError) as error:
        _die(str(error))
    print(f"[OK] held-out encoding written -> {path}")
    return path


def _strong_inputs(
    split: str,
    data_dir: Path,
    cache_dir: Path,
    question_cache_dir: Path,
    backend: EmbeddingBackend | None,
    settings: strong_dense.Phase8Pins,
) -> tuple[
    list[IndexingUnit], list[str], list[Question], dict[str, int], DenseRetriever, dict[str, Any]
]:
    """The loader both Phase 8 measurement paths share: pool, caches, dense, provenance.

    Nothing is embedded here and no model is loaded: both splits were encoded by
    `strong-embed`, their caches are the record, and a missing one is a refusal rather
    than a quiet re-encoding on whatever machine happens to be running the measurement.
    """
    if not (data_dir / POOL_NAME).exists():
        _die("no pool found: run 'build' first")
    if not (data_dir / TOKENS_NAME).exists():
        _die("no token counts found: run 'embed' first")
    units, questions = load_pool(data_dir / POOL_NAME)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    subset = [question for question in questions if question.split == split]
    if not subset:
        _die(f"the pool holds no {split} questions: run 'build' first")
    unit_ids = [unit.unit_id for unit in units]
    token_counts = json.loads((data_dir / TOKENS_NAME).read_text(encoding="utf-8"))
    if set(token_counts) != set(unit_ids):
        _die("token counts belong to another pool")

    reader = backend if backend is not None else _phase_8_backend(settings, prompted=True)
    corpus_key = _cache_key_for(reader, units)
    cached = EmbeddingCache(Path(cache_dir)).load(corpus_key, expected_unit_ids=unit_ids)
    if cached is None:
        _die("the Phase 8 corpus embeddings are not cached: run 'strong-embed' first")
    question_cache = EmbeddingCache(Path(question_cache_dir))
    query_key = _phase_8_question_key(subset, reader)
    if question_cache.load(query_key, expected_unit_ids=[q.qid for q in subset]) is None:
        mode = " --test" if split == TEST_SPLIT else ""
        _die(f"the Phase 8 {split} question embeddings are missing: run 'strong-embed{mode}' first")
    dense = DenseRetriever(cached[0], unit_ids, build_query_backend(subset, reader, question_cache))
    run_config = {
        "model": reader.name,
        "revision": reader.revision,
        "unit_set_hash": unit_set_hash(unit_ids),
        "seed": config.DEFAULT_SEED,
        "tokenizer": config.BUDGET_TOKENIZER_ID,
        "budget_tokenizer_revision": config.BUDGET_TOKENIZER_REVISION,
        "code_version": __version__,
        "top_k": config.EVALUATION_TOP_K,
        "n_units": len(units),
        "corpus_cache_key": corpus_key,
        "question_cache_key": query_key,
        "query_prompt": query_prompt_of(reader),
        "dim": int(cached[0].shape[1]),
    }
    return units, unit_ids, subset, token_counts, dense, run_config


def _baseline_dense(
    units: Sequence[IndexingUnit],
    questions: Sequence[Question],
    cache_dir: Path,
    question_cache_dir: Path,
    backend: EmbeddingBackend | None,
) -> DenseRetriever | None:
    """The BGE-small retriever, rebuilt from its existing caches for the `p1` diagnostic.

    No model is loaded and nothing is re-embedded: both caches are already on disk. When
    either is absent the diagnostic is declared unmeasured rather than invented.
    """
    reader = backend if backend is not None else SentenceTransformerBackend()
    unit_ids = [unit.unit_id for unit in units]
    cached = EmbeddingCache(Path(cache_dir)).load(
        _cache_key_for(reader, units), expected_unit_ids=unit_ids
    )
    if cached is None:
        return None
    question_cache = EmbeddingCache(Path(question_cache_dir))
    key = _phase_8_question_key(questions, reader)
    if question_cache.load(key, expected_unit_ids=[q.qid for q in questions]) is None:
        return None
    return DenseRetriever(
        cached[0], unit_ids, build_query_backend(questions, reader, question_cache)
    )


def cmd_strong_dense(
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    question_cache_dir: Path = config.QUESTION_CACHE_DIR,
    phase_8_dir: Path = config.PHASE_8_DIR,
    gliner_dir: Path = config.PHASE_8_GLINER_DIR,
    backend: EmbeddingBackend | None = None,
    baseline_backend: EmbeddingBackend | None = None,
    pins: strong_dense.Phase8Pins | None = None,
    test: bool = False,
) -> Path:
    """Fit on dev and measure the three systems; with `--test`, read the held-out split once.

    Both paths read cached vectors only, so neither loads a model and both run on the
    development machine. The Phase 7 GLiNER index is loaded and verified against its
    pinned digest; nothing is written into `data/phase7/`.
    """
    settings = pins if pins is not None else strong_dense.Phase8Pins.from_config()
    data_dir, phase_8_dir = Path(data_dir), Path(phase_8_dir)
    split = TEST_SPLIT if test else DEV_SPLIT
    units, unit_ids, questions, token_counts, dense, run_config = _strong_inputs(
        split, data_dir, Path(cache_dir), Path(question_cache_dir), backend, settings
    )
    try:
        index = load_node_index(
            Path(gliner_dir),
            extraction_digest=settings.extraction_digest,
            expected_unit_ids=unit_ids,
        )
    except NodeIndexError as error:
        _die(str(error))
    baseline = _baseline_dense(
        units, questions, Path(cache_dir), Path(question_cache_dir), baseline_backend
    )
    if baseline is None:
        print("[WARN] no BGE-small cache for this split; the p1 comparison stays unmeasured")

    arguments: dict[str, Any] = {
        "unit_ids": unit_ids,
        "dense": dense,
        "bm25": BM25Retriever(units),
        "index": index,
        "questions": questions,
        "token_counts": token_counts,
        "run_config": run_config,
        "baseline_dense": baseline,
        "pins": settings,
    }
    print(f"[INFO] measuring {len(config.PHASE_8_SYSTEMS)} systems on {split} ({len(questions)})")
    try:
        measure = strong_dense.measure_held_out if test else strong_dense.measure_dev
        path = measure(phase_8_dir, **arguments)
    except (strong_dense.StrongDenseError, SelectionError, NodeIndexError) as error:
        _die(str(error))
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name in config.PHASE_8_SYSTEMS:
        system = payload["systems"][name]
        print(
            f"[INFO] {name} on {split}: full support @{config.SELECTION_BUDGET} "
            f"{system['full_support']:.4f} "
            f"({system['supported_questions']}/{payload['n_questions']})"
        )
    if test:
        gain = payload["surviving_gain"]
        print(
            f"[INFO] entity gain over Strong Dense: {gain['entity_gain_questions']} questions, "
            f"{gain['entity_gain_points']:+.4f} points; inherited GLiNER gain "
            f"{gain['inherited_entity_gain_questions']} questions, "
            f"{gain['inherited_entity_gain_points']:+.4f} points"
        )
    else:
        rule = payload["stop_rule"]
        print(
            f"[INFO] stop rule: {rule['count']}/{rule['n_questions']} supported, "
            f"{rule['required_count']} required to continue, verdict {rule['verdict'].upper()}"
        )
    print(f"[OK] {split} measurement written -> {path}")
    return path


def _poll_pause() -> None:
    """How long the full run waits between two looks at a batch. Most end within an hour."""
    time.sleep(60)


def _check_pool_against_manifest(units: Sequence[IndexingUnit], manifest_path: Path) -> None:
    """Refuse to run on a pool the manifest does not recognise.

    `load_pool` already proves each unit hashes to its own content; this proves the
    set as a whole is the one the recorded numbers were produced from. The manifest
    has carried `unit_set_hash` since the corpus was frozen - it was simply never
    read back.
    """
    if not manifest_path.exists():
        return
    manifest = CorpusManifest.load(manifest_path)
    if manifest.unit_set_hash is None:
        return
    actual = unit_set_hash([u.unit_id for u in units])
    if actual != manifest.unit_set_hash:
        _die(
            f"pool hashes to {actual} but the manifest expects {manifest.unit_set_hash}; "
            "re-run 'build' so the corpus and the pool agree"
        )


# --- Deviation 8.1: Dense scale sensitivity ---------------------------------


def _source_path(archive: str | None) -> Path:
    """Where the operator staged the official archive. It is never downloaded here."""
    if archive is not None:
        return Path(archive)
    return config.PHASE_8_1_DIR / "source" / config.PHASE_8_1_SOURCE_ARCHIVE


def _scale_source(
    target_dir: Path,
    archive: Path,
    published_bytes: int | None,
    published_md5: str | None,
    page_checked: bool,
) -> dict[str, Any]:
    """S1: the archive's identity and observed layout, written once.

    Identity is verified before a single record is parsed. `page_checked` records whether
    the published figures came from the live source page or from the deviation's own
    unverified recollection - the artifact must never imply the former when only the latter
    happened.
    """
    path = target_dir / SOURCE_NAME
    if path.exists():
        print(f"[OK] {path} already exists; S1 is not re-run")
        return json.loads(path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    try:
        source = fullwiki.describe_source(
            archive,
            url=config.PHASE_8_1_SOURCE_URL,
            published_bytes=published_bytes,
            published_md5=published_md5,
        )
    except fullwiki.FullWikiError as error:
        _die(str(error))
    source["page_checked"] = page_checked
    source["published_basis"] = (
        "read from the live HotpotQA page by the operator"
        if page_checked
        else "the deviation's recorded values, verified against the staged bytes but not "
        "against the live page"
    )
    source["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_text_atomic(path, json.dumps(source, indent=2, sort_keys=True))

    print(f"[OK] archive {source['archive']}")
    print(f"[OK] bytes {source['measured_bytes']} (agree: {source['bytes_agree']})")
    print(f"[OK] md5 {source['measured_md5']} (agree: {source['md5_agree']})")
    print(f"[OK] sha256 {source['measured_sha256']}")
    print(f"[OK] members {source['members']}")
    print(f"[OK] layout {source['layout']}")
    print(f"[OK] S1 in {source['elapsed_seconds']} s -> {path}")
    return source


def _scale_reconciliation(
    target_dir: Path,
    archive: Path,
    source: dict[str, Any],
    units: Sequence[IndexingUnit],
    dev: Sequence[Question],
) -> dict[str, Any]:
    """S2: map every frozen C19 unit onto the official source. C19 is never rebuilt."""
    path = target_dir / scale_corpus.RECONCILIATION_FILENAME
    if path.exists():
        print(f"[OK] {path} already exists; S2 is not re-run")
        return json.loads(path.read_text(encoding="utf-8"))

    layout = fullwiki.layout_of(source["layout"])
    gold = frozenset(unit_id for question in dev for unit_id in question.gold_unit_ids)
    started = time.perf_counter()
    result = scale_corpus.reconcile(
        units, fullwiki.iter_records(archive, layout), dev_gold_ids=gold
    )
    elapsed = round(time.perf_counter() - started, 3)
    scale_corpus.write_reconciliation(target_dir, result)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["elapsed_seconds"] = elapsed
    write_text_atomic(path, json.dumps(body, indent=2, ensure_ascii=False, sort_keys=True))

    print(
        f"[OK] exact {body['exact']}, normalized {body['normalized_unique']}, "
        f"ambiguous {body['ambiguous']}, unmatched {body['unmatched']} of {body['total_units']}"
    )
    print(
        f"[OK] dev gold matched {body['dev_gold_matched']} of {body['dev_gold_total']}, "
        f"excluded gold units {len(body['excluded_gold_units'])}"
    )
    print(
        f"[OK] excluded titles {len(body['excluded_titles'])}, "
        f"official paragraphs removed {body['excluded_official_paragraphs']}"
    )
    print(f"[OK] mapping digest {body['mapping_digest']}")
    print(f"[OK] S2 in {elapsed} s -> {path}")
    if body["terminal_state"] is not None:
        _die(
            f"[ERROR] {body['terminal_state']}: "
            f"{body['ambiguous'] + body['unmatched']} C19 units did not resolve, past the "
            f"ceiling of {body['unmatched_ceiling']}. Recorded in {path}; no selection is "
            "frozen and no corpus is embedded"
        )
    return body


def _scale_selection(
    target_dir: Path,
    archive: Path,
    source: dict[str, Any],
    reconciliation: dict[str, Any],
    units: Sequence[IndexingUnit],
) -> None:
    """S3: freeze one ordering of the added distractors, and count their tokens."""
    path = target_dir / scale_corpus.SELECTION_FILENAME
    if path.exists():
        print(f"[OK] {path} already exists; S3 is not re-run")
        return

    layout = fullwiki.layout_of(source["layout"])
    mapped = frozenset(reconciliation["mapped_official_ids"])
    excluded = frozenset(reconciliation["excluded_titles"])
    limit = config.PHASE_8_1_DISTRACTOR_PREFIXES[-1]

    started = time.perf_counter()
    try:
        selected = scale_corpus.select_distractors(
            lambda: fullwiki.iter_records(archive, layout),
            mapped_ids=mapped,
            excluded_titles=excluded,
            limit=limit,
        )
    except scale_corpus.ScaleCorpusError as error:
        _die(str(error))
    selection_seconds = round(time.perf_counter() - started, 3)

    try:
        scale_corpus.freeze_selection(
            target_dir,
            c19_unit_ids=[unit.unit_id for unit in units],
            selected=selected,
            source_sha256=source["measured_sha256"],
            reconciliation_digest=reconciliation["mapping_digest"],
        )
    except scale_corpus.ScaleCorpusError as error:
        _die(str(error))
    body = json.loads(path.read_text(encoding="utf-8"))
    print(f"[OK] selected {body['selected']} distractors of the eligible pool")
    print(f"[OK] selection digest {body['ordered_selection_digest']}")
    print(f"[OK] S3 selection in {selection_seconds} s -> {path}")

    started = time.perf_counter()
    counter = TokenCounter()
    counts = scale_corpus.token_counts_for([item.unit for item in selected], counter)
    sidecar = scale_corpus.write_token_counts(
        target_dir, counts, selection_digest=body["ordered_selection_digest"]
    )
    token_seconds = round(time.perf_counter() - started, 3)

    historical = json.loads((config.DATA_DIR / TOKENS_NAME).read_text(encoding="utf-8"))
    corpus_ids = [unit.unit_id for unit in units] + [item.unit_id for item in selected]
    try:
        scale_corpus.check_token_coverage(
            historical=historical, new=counts, corpus_unit_ids=corpus_ids
        )
    except scale_corpus.ScaleCorpusError as error:
        _die(str(error))
    print(f"[OK] token counts for {len(counts)} added units in {token_seconds} s -> {sidecar}")
    print(f"[OK] C500 covered by {len(historical)} historical + {len(counts)} new counts")


def cmd_scale_corpus(
    archive: str | None = None,
    published_bytes: int | None = config.PHASE_8_1_DECLARED_BYTES,
    published_md5: str | None = config.PHASE_8_1_DECLARED_MD5,
    page_checked: bool = False,
    stop_after: str = "s3",
    data_dir: Path = config.DATA_DIR,
    target_dir: Path = config.PHASE_8_1_DIR,
) -> None:
    """Deviation 8.1, S1-S3: verify the archive, reconcile C19, freeze the nested corpora.

    Each step writes one artifact and is skipped when that artifact already exists, so the
    stage can be resumed without re-reading 1.5 GB for work already done. Nothing is
    downloaded and nothing is embedded here.
    """
    if not (data_dir / POOL_NAME).exists():
        _die("no pool found: run 'build' first")
    if not (data_dir / TOKENS_NAME).exists():
        _die("no token counts found: run 'embed' first")
    path = _source_path(archive)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    source = _scale_source(target_dir, path, published_bytes, published_md5, page_checked)
    if stop_after == "s1":
        print("[OK] stopping after S1 as asked")
        return

    units, questions = load_pool(data_dir / POOL_NAME)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    dev = [question for question in questions if question.split == DEV_SPLIT]
    if not dev:
        _die(f"the pool holds no {DEV_SPLIT} questions: run 'build' first")

    reconciliation = _scale_reconciliation(target_dir, path, source, units, dev)
    if stop_after == "s2":
        print("[OK] stopping after S2 as asked")
        return

    _scale_selection(target_dir, path, source, reconciliation, units)
    print("[OK] S1-S3 complete; the nested selection is frozen")


def _historical_dense(
    key: str,
    units: Sequence[IndexingUnit],
    questions: Sequence[Question],
    cache_dir: Path,
    question_cache_dir: Path,
) -> tuple[scale_sensitivity.ScaleDense, dict[str, Any]]:
    """One encoder's C19 retriever, rebuilt from the caches its own phase already wrote.

    Read-only, and no model is loaded anywhere on this path. Both cache keys are recomputed
    and checked against the ones deviation 8.1 recorded **before** either file is opened: a
    key that does not match is an alignment defect to diagnose, never a cache miss to fill
    in by re-encoding, because re-encoding would replace the very vectors the gate holds
    constant.
    """
    pins = scale_sensitivity.ModelPins.for_key(key)
    unit_ids = [unit.unit_id for unit in units]
    corpus_key = cache_key(
        pins.model,
        pins.revision,
        unit_set_hash(unit_ids),
        normalized=config.NORMALIZE_EMBEDDINGS,
    )
    expected_corpus = config.PHASE_8_1_HISTORICAL_CORPUS_KEYS[key]
    if corpus_key != expected_corpus:
        _die(
            f"the {key} corpus cache key resolves to {corpus_key}, not the recorded "
            f"{expected_corpus}: the pool or the model pin moved, and level 1 would no "
            "longer be reading the historical vectors"
        )
    corpus = EmbeddingCache(Path(cache_dir)).load(corpus_key, expected_unit_ids=unit_ids)
    if corpus is None:
        _die(
            f"the {key} C19 corpus cache embeddings-{corpus_key}.npz is not on this machine: "
            "level 1 reads the historical caches and does not re-encode them"
        )

    qids = [question.qid for question in questions]
    query_key = question_cache_key(
        pins.model,
        pins.revision,
        question_set_hash(qids),
        DEV_SPLIT,
        normalized=config.NORMALIZE_EMBEDDINGS,
        query_prompt=pins.query_prompt,
    )
    expected_query = config.PHASE_8_1_HISTORICAL_DEV_QUERY_KEYS[key]
    if query_key != expected_query:
        _die(
            f"the {key} dev question cache key resolves to {query_key}, not the recorded "
            f"{expected_query}: the question set, the pin or the query prompt moved"
        )
    queries = EmbeddingCache(Path(question_cache_dir)).load(query_key, expected_unit_ids=qids)
    if queries is None:
        _die(f"the {key} dev question cache embeddings-{query_key}.npz is not on this machine")

    retriever = scale_sensitivity.historical_retriever(
        corpus[0],
        unit_ids,
        questions=questions,
        query_vectors=queries[0],
        qids=queries[1],
        pins=pins,
        name=scale_sensitivity.run_name(key, config.EXPECTED_N_UNITS),
    )
    provenance = {
        "model": pins.model,
        "revision": pins.revision,
        "unit_set_hash": unit_set_hash(unit_ids),
        "seed": config.DEFAULT_SEED,
        "tokenizer": config.BUDGET_TOKENIZER_ID,
        "budget_tokenizer_revision": config.BUDGET_TOKENIZER_REVISION,
        "code_version": __version__,
        "top_k": config.EVALUATION_TOP_K,
        "n_units": len(unit_ids),
        "corpus_cache_key": corpus_key,
        "question_cache_key": query_key,
        "query_prompt": pins.query_prompt,
        "dim": int(corpus[0].shape[1]),
    }
    return retriever, provenance


def cmd_scale_repro(
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    question_cache_dir: Path = config.QUESTION_CACHE_DIR,
    target_dir: Path = config.PHASE_8_1_DIR,
) -> Path:
    """Deviation 8.1, S5: the C19 reproduction gate, level 1.

    Runs the new 8.1 evaluation path over the unchanged frozen C19 corpus using the existing
    historical caches, so the vectors are the recorded ones and the only variable is the
    code. Both counts must match exactly; anything else writes `reproduction_stop` and
    blocks every later stage. Nothing is embedded, no model is loaded, and no cache is
    written.
    """
    if not (data_dir / POOL_NAME).exists():
        _die("no pool found: run 'build' first")
    if not (data_dir / TOKENS_NAME).exists():
        _die("no token counts found: run 'embed' first")
    units, questions = load_pool(data_dir / POOL_NAME)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    dev = [question for question in questions if question.split == DEV_SPLIT]
    if not dev:
        _die(f"the pool holds no {DEV_SPLIT} questions: run 'build' first")
    token_counts = json.loads((data_dir / TOKENS_NAME).read_text(encoding="utf-8"))

    retrievers: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for key in config.PHASE_8_1_MODELS:
        retriever, model_provenance = _historical_dense(
            key, units, dev, Path(cache_dir), Path(question_cache_dir)
        )
        retrievers[key] = retriever
        provenance[key] = model_provenance
        print(f"[OK] {key}: {model_provenance['n_units']} units, dim {model_provenance['dim']}")

    host = {
        "platform": sys.platform,
        "device": "cpu",
        "basis": (
            "level 1 runs on the laptop because it gates the rental; level 2 runs on the "
            "measurement host inside S6"
        ),
    }
    try:
        path = scale_sensitivity.measure_level_1(
            Path(target_dir),
            retrievers=retrievers,
            questions=dev,
            token_counts=token_counts,
            provenance=provenance,
            host=host,
        )
    except scale_sensitivity.ScaleSensitivityError as error:
        _die(str(error))

    body = json.loads(path.read_text(encoding="utf-8"))
    for key in sorted(body["models"]):
        reading = body["models"][key]
        mark = "[OK]" if reading["matches"] else "[ERROR]"
        print(
            f"{mark} {key}: {reading['supported']} / {reading['n_questions']} supported, "
            f"required {reading['required']}"
        )
    if body["terminal_state"] is None:
        print(f"[OK] reproduction gate level 1 passed -> {path}")
    else:
        print(f"[ERROR] {body['terminal_state']} recorded in {path}")
    return path


def _scale_local_preflight(
    model: str,
    *,
    data_dir: Path,
    target_dir: Path,
    cache_dir: Path,
    phase_cache_dir: Path,
    phase_question_cache_dir: Path,
) -> dict[str, Any]:
    """CPU-only S6 checks. No model, CUDA probe or historical cache is required here."""
    try:
        pins = scale_sensitivity.ModelPins.for_key(model)
    except scale_sensitivity.ScaleSensitivityError as error:
        _die(str(error))

    data_dir = Path(data_dir)
    target_dir = Path(target_dir)
    cache_dir = Path(cache_dir)
    phase_cache_dir = Path(phase_cache_dir)
    phase_question_cache_dir = Path(phase_question_cache_dir)

    for path, stage in (
        (data_dir / POOL_NAME, "build"),
        (data_dir / MANIFEST_NAME, "build"),
        (data_dir / TOKENS_NAME, "embed"),
        (target_dir / SOURCE_NAME, "scale-corpus"),
        (target_dir / scale_corpus.RECONCILIATION_FILENAME, "scale-corpus"),
        (target_dir / scale_corpus.SELECTION_FILENAME, "scale-corpus"),
        (target_dir / scale_corpus.COUNTS_FILENAME, "scale-corpus"),
        (target_dir / scale_corpus.COUNTS_SIDECAR, "scale-corpus"),
        (target_dir / scale_sensitivity.REPRODUCTION_FILENAME, "scale-repro"),
    ):
        if not path.exists():
            _die(f"{path} does not exist: run '{stage}' first")

    if phase_cache_dir.resolve() == cache_dir.resolve():
        _die("the deviation 8.1 embedding cache must be separate from the historical cache")
    if phase_question_cache_dir.resolve().parent != phase_cache_dir.resolve():
        _die("the deviation 8.1 question cache must live under its own phase8_1 cache directory")

    units, questions = load_pool(data_dir / POOL_NAME)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    if len(units) != config.EXPECTED_N_UNITS:
        _die(f"C19 holds {len(units)} units, not the frozen {config.EXPECTED_N_UNITS}")

    dev = [question for question in questions if question.split == DEV_SPLIT]
    held_out = [question for question in questions if question.split == TEST_SPLIT]
    if (
        len(questions) != config.N_QUESTIONS
        or len(dev) != config.N_DEV
        or len(held_out) != config.N_TEST
    ):
        _die(
            "the frozen pool split sizes moved: "
            f"{len(questions)} total, {len(dev)} dev, {len(held_out)} test; expected "
            f"{config.N_QUESTIONS}/{config.N_DEV}/{config.N_TEST}"
        )

    source = json.loads((target_dir / SOURCE_NAME).read_text(encoding="utf-8"))
    if source.get("bytes_agree") is not True or source.get("md5_agree") is not True:
        _die("S1 source identity did not pass both the recorded byte-size and MD5 checks")
    source_sha256 = str(source.get("measured_sha256", ""))
    if len(source_sha256) != 64:
        _die("S1 records no valid measured SHA-256 for the staged FullWiki bytes")

    try:
        reconciliation = scale_corpus.load_reconciliation(target_dir)
        if reconciliation.get("terminal_state") is not None:
            _die(
                f"the reconciliation recorded {reconciliation['terminal_state']}: "
                "no scale run is permitted"
            )
        if int(reconciliation.get("total_units", -1)) != len(units):
            _die("the reconciliation belongs to another C19 pool")

        corpus = scale_corpus.load_selection(target_dir, c19_units=units)
        selection = json.loads(
            (target_dir / scale_corpus.SELECTION_FILENAME).read_text(encoding="utf-8")
        )
        if tuple(selection.get("corpus_sizes", ())) != tuple(config.PHASE_8_1_CORPUS_SIZES):
            _die("selection.json does not carry the four frozen 8.1 corpus sizes")
        if selection.get("reconciliation_digest") != reconciliation.get("mapping_digest"):
            _die("the frozen selection was built from another reconciliation")
        if selection.get("source_sha256") != source_sha256:
            _die("the frozen selection was built from another FullWiki source")
        expected_total = config.PHASE_8_1_CORPUS_SIZES[-1]
        if corpus.total != expected_total:
            _die(f"the frozen nested corpus holds {corpus.total} units, not C500={expected_total}")

        new_counts = scale_corpus.load_token_counts(target_dir)
        counts_sidecar = json.loads(
            (target_dir / scale_corpus.COUNTS_SIDECAR).read_text(encoding="utf-8")
        )
        if counts_sidecar.get("selection_digest") != corpus.selection_digest:
            _die("the 8.1 token counts belong to another frozen selection")
        if counts_sidecar.get("tokenizer_id") != config.BUDGET_TOKENIZER_ID:
            _die("the 8.1 token counts were measured with another budget tokenizer")
        if counts_sidecar.get("tokenizer_revision") != config.BUDGET_TOKENIZER_REVISION:
            _die("the 8.1 token counts were measured with another tokenizer revision")
        if int(counts_sidecar.get("units", -1)) != len(corpus.distractors):
            _die("the 8.1 token-count sidecar does not describe every added distractor")

        historical_counts = json.loads((data_dir / TOKENS_NAME).read_text(encoding="utf-8"))
        corpus_ids = [unit.unit_id for unit in (*corpus.c19, *corpus.distractors)]
        scale_corpus.check_token_coverage(
            historical=historical_counts,
            new=new_counts,
            corpus_unit_ids=corpus_ids,
        )
    except scale_corpus.ScaleCorpusError as error:
        _die(str(error))

    try:
        reproduction = scale_sensitivity.check_reproduction(target_dir)
    except scale_sensitivity.ScaleSensitivityError as error:
        _die(str(error))
    if int(reproduction.get("corpus_units", -1)) != config.EXPECTED_N_UNITS:
        _die("the level-1 reproduction artifact belongs to another C19 corpus")
    if int(reproduction.get("questions", -1)) != config.N_DEV:
        _die("the level-1 reproduction artifact does not contain the frozen 600 dev questions")
    if set(reproduction.get("models", {})) != set(config.PHASE_8_1_MODELS):
        _die("the level-1 reproduction artifact does not contain exactly BGE and Qwen")
    required = {key: int(value) for key, value in reproduction.get("required", {}).items()}
    if required != dict(config.PHASE_8_1_C19_DEV_SUPPORTED):
        _die("the level-1 reproduction artifact was checked against different frozen counts")

    scale_target = target_dir / scale_sensitivity.scale_filename(model)
    if scale_target.exists():
        _die(f"{scale_target} already records this model's S6 measurement; it is not overwritten")

    return {
        "model": model,
        "model_id": pins.model,
        "revision": pins.revision,
        "dim": pins.dim,
        "c19_units": len(corpus.c19),
        "c500_units": corpus.total,
        "dev_questions": len(dev),
        "selection_digest": corpus.selection_digest,
        "reproduction_matches": True,
        "phase_cache_dir": str(phase_cache_dir),
        "phase_question_cache_dir": str(phase_question_cache_dir),
    }


def _scale_cuda_available() -> bool:
    """CUDA is mandatory only on the real S6 measurement host, never on local preflight."""
    return _embedding_device() == "cuda"


def _scale_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _scale_peak_rss_mb() -> float:
    """Linux process RSS high-water mark, sampled after the C500 retrieval pass."""
    status = Path("/proc/self/status")
    if not status.exists():
        _die("measurement-host RSS probe requires Linux /proc/self/status")
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) / 1024.0
    _die("measurement-host RSS probe could not read VmHWM from /proc/self/status")


def _scale_required_free_bytes(pins: scale_sensitivity.ModelPins) -> int:
    """Operational disk floor: two raw vector blocks plus 1 GiB of working headroom."""
    rows = config.PHASE_8_1_CORPUS_SIZES[-1] + config.N_DEV
    raw_vectors = rows * pins.dim * np.dtype(np.float32).itemsize
    return 2 * raw_vectors + 1024**3


def _scale_measurement_host_preflight(
    model: str,
    *,
    units: Sequence[IndexingUnit],
    questions: Sequence[Question],
    historical_token_counts: dict[str, int],
    cache_dir: Path,
    question_cache_dir: Path,
    target_dir: Path,
) -> dict[str, Any]:
    """RunPod-only checks and D-L1, all before an embedding model is constructed."""
    pins = scale_sensitivity.ModelPins.for_key(model)
    if not _scale_cuda_available():
        _die("measurement-host preflight requires CUDA before any S6 model is loaded")

    required_free = _scale_required_free_bytes(pins)
    free = _scale_free_bytes(target_dir)
    if free < required_free:
        _die(
            f"measurement host has {free} free bytes but {model} requires at least "
            f"{required_free} bytes for the 8.1 vector cache and working headroom"
        )

    retriever, provenance = _historical_dense(
        model, units, questions, Path(cache_dir), Path(question_cache_dir)
    )
    host = local_extraction.hardware_block(device="cuda")
    try:
        d_l1 = scale_sensitivity.measure_level_one_on_measurement_host(
            model=model,
            retriever=retriever,
            questions=questions,
            token_counts=historical_token_counts,
            provenance=provenance,
            host=host,
            required=config.PHASE_8_1_C19_DEV_SUPPORTED[model],
            expected_questions=config.N_DEV,
        )
    except scale_sensitivity.ScaleSensitivityError as error:
        _die(str(error))

    print(
        f"[OK] measurement-host D-L1 {model}: {d_l1['supported']} / {d_l1['n_questions']} supported"
    )
    print(f"[OK] measurement-host disk: {free} bytes free, {required_free} required")
    return {
        "cuda": True,
        "free_bytes": free,
        "required_free_bytes": required_free,
        "host": host,
        "level_one_on_measurement_host": d_l1,
    }


def _cached_scale_query_backend(
    questions: Sequence[Question],
    *,
    vectors: np.ndarray,
    qids: Sequence[str],
    pins: scale_sensitivity.ModelPins,
) -> CachedQueryBackend:
    by_qid = {question.qid: question.question for question in questions}
    missing = [qid for qid in qids if qid not in by_qid]
    if missing:
        _die(
            f"the 8.1 question cache holds qid {missing[0]!r}, which is not in the frozen dev split"
        )
    return CachedQueryBackend(
        texts=[by_qid[qid] for qid in qids],
        vectors=vectors,
        name=pins.model,
        revision=pins.revision,
    )


def _scale_cache_bytes(
    corpus_cache: EmbeddingCache,
    corpus_key: str,
    question_cache: EmbeddingCache,
    question_key: str,
) -> dict[str, int]:
    corpus = sum(
        path.stat().st_size
        for path in (corpus_cache.path_for(corpus_key), corpus_cache.sidecar_for(corpus_key))
        if path.exists()
    )
    questions = sum(
        path.stat().st_size
        for path in (
            question_cache.path_for(question_key),
            question_cache.sidecar_for(question_key),
        )
        if path.exists()
    )
    return {"corpus": corpus, "questions": questions, "total": corpus + questions}


def cmd_scale_run(
    model: str,
    *,
    preflight_only: bool = False,
    hourly_rate_usd: float = 0.0,
    data_dir: Path = config.DATA_DIR,
    target_dir: Path = config.PHASE_8_1_DIR,
    cache_dir: Path = config.CACHE_DIR,
    historical_question_cache_dir: Path = config.QUESTION_CACHE_DIR,
    phase_cache_dir: Path = config.PHASE_8_1_CACHE_DIR,
    phase_question_cache_dir: Path = config.PHASE_8_1_QUESTION_CACHE_DIR,
    corpus_backend: EmbeddingBackend | None = None,
    query_backend: EmbeddingBackend | None = None,
    resolve_snapshot: SnapshotResolver = snapshot_directory,
) -> Path | dict[str, Any]:
    """Deviation 8.1 S6: one C500 encode, then exact nested-prefix measurements."""
    if not math.isfinite(hourly_rate_usd) or hourly_rate_usd < 0:
        _die("hourly-rate-usd must be finite and non-negative")

    data_dir = Path(data_dir)
    target_dir = Path(target_dir)
    phase_cache_dir = Path(phase_cache_dir)
    phase_question_cache_dir = Path(phase_question_cache_dir)
    summary = _scale_local_preflight(
        model,
        data_dir=data_dir,
        target_dir=target_dir,
        cache_dir=Path(cache_dir),
        phase_cache_dir=phase_cache_dir,
        phase_question_cache_dir=phase_question_cache_dir,
    )
    print(
        f"[OK] local preflight {model}: C19={summary['c19_units']}, "
        f"C500={summary['c500_units']}, dev={summary['dev_questions']}"
    )
    print(f"[OK] selection digest {summary['selection_digest']}")
    print(
        f"[OK] fresh caches isolated under {summary['phase_cache_dir']} "
        f"(questions: {summary['phase_question_cache_dir']})"
    )
    if preflight_only:
        print("[OK] local preflight complete; no model was loaded and no embedding was written")
        return summary

    pins = scale_sensitivity.ModelPins.for_key(model)
    units, questions = load_pool(data_dir / POOL_NAME)
    dev = [question for question in questions if question.split == DEV_SPLIT]
    try:
        corpus = scale_corpus.load_selection(target_dir, c19_units=units)
        new_counts = scale_corpus.load_token_counts(target_dir)
    except scale_corpus.ScaleCorpusError as error:
        _die(str(error))
    historical_counts = json.loads((data_dir / TOKENS_NAME).read_text(encoding="utf-8"))
    token_counts = {**historical_counts, **new_counts}
    c500_units = [*corpus.c19, *corpus.distractors]
    unit_ids = [unit.unit_id for unit in c500_units]

    measurement_host = _scale_measurement_host_preflight(
        model,
        units=units,
        questions=dev,
        historical_token_counts=historical_counts,
        cache_dir=Path(cache_dir),
        question_cache_dir=Path(historical_question_cache_dir),
        target_dir=target_dir,
    )

    try:
        strong_dense.check_model_pin(pins.revision, pins.model)
        snapshot = Path(resolve_snapshot(pins.model, pins.revision))
        strong_dense.check_resolved_revision(snapshot.name, pins.revision)
        weights_digest = weights_sha256(snapshot)
    except (strong_dense.StrongDenseError, BackendError) as error:
        _die(str(error))

    corpus_cache = EmbeddingCache(phase_cache_dir)
    question_cache = EmbeddingCache(phase_question_cache_dir)
    corpus_key = cache_key(
        pins.model,
        pins.revision,
        unit_set_hash(unit_ids),
        normalized=config.NORMALIZE_EMBEDDINGS,
    )
    # MetadataBackend never encodes. This key is the same key the real prompted backend
    # will use, while the separate 8.1 directory prevents collision with historical vectors.
    dev_question_hash = question_set_hash([question.qid for question in dev])
    query_key = question_cache_key(
        pins.model,
        pins.revision,
        dev_question_hash,
        DEV_SPLIT,
        normalized=config.NORMALIZE_EMBEDDINGS,
        query_prompt=pins.query_prompt,
    )

    embedding_path = target_dir / scale_sensitivity.embedding_filename(model)
    corpus_exists = corpus_cache.path_for(corpus_key).exists()
    questions_exist = question_cache.path_for(query_key).exists()
    if not embedding_path.exists() and (corpus_exists or questions_exist):
        _die(
            "an 8.1 cache exists without its embedding artifact; refusing to guess whether "
            "that partial encoding is reusable"
        )

    if embedding_path.exists():
        try:
            embedding = scale_sensitivity.load_embedding(target_dir, model)
        except scale_sensitivity.ScaleSensitivityError as error:
            _die(str(error))
        expected = {
            "corpus_cache_key": corpus_key,
            "question_cache_key": query_key,
            "selection_digest": corpus.selection_digest,
            "revision": pins.revision,
            "dim": pins.dim,
            "n_questions": len(dev),
            "corpus_units": len(c500_units),
        }
        for field, value in expected.items():
            if embedding.get(field) != value:
                _die(
                    f"{embedding_path} says {field}={embedding.get(field)!r}, "
                    f"but this run requires {value!r}"
                )
        loaded_corpus = corpus_cache.load(corpus_key, expected_unit_ids=unit_ids)
        qids = [question.qid for question in dev]
        loaded_questions = question_cache.load(query_key, expected_unit_ids=qids)
        if loaded_corpus is None or loaded_questions is None:
            _die(
                f"{embedding_path} exists but one of its recorded caches is missing; "
                "the expensive encoding is not silently repeated"
            )
        vectors, cached_unit_ids = loaded_corpus
        query_vectors, cached_qids = loaded_questions
        query_lookup = _cached_scale_query_backend(
            dev, vectors=query_vectors, qids=cached_qids, pins=pins
        )
        if int(vectors.shape[1]) != pins.dim or int(query_vectors.shape[1]) != pins.dim:
            _die(f"the reused {model} caches do not have the pinned width {pins.dim}")
        print(f"[OK] reusing {embedding_path.name} and its two 8.1 caches")
    else:
        corpus_model = (
            corpus_backend if corpus_backend is not None else _scale_backend(pins, prompted=False)
        )
        query_model = (
            query_backend if query_backend is not None else _scale_backend(pins, prompted=True)
        )
        guarded_corpus = strong_dense.WidthCheckedBackend(corpus_model, pins.dim)
        guarded_queries = strong_dense.WidthCheckedBackend(query_model, pins.dim)
        if _cache_key_for(guarded_corpus, c500_units) != corpus_key:
            _die("the real corpus backend does not resolve to the preflighted 8.1 cache key")
        if _phase_8_question_key(dev, guarded_queries) != query_key:
            _die("the real query backend does not resolve to the preflighted 8.1 question key")

        print(f"[INFO] encoding C500 once for {model}: {len(c500_units)} paragraphs")
        try:
            started = time.perf_counter()
            vectors, cached_unit_ids = embed_units(c500_units, guarded_corpus, corpus_cache)
            embedding_seconds = time.perf_counter() - started
            started = time.perf_counter()
            query_lookup = build_query_backend(dev, guarded_queries, question_cache)
            question_seconds = time.perf_counter() - started
        except (strong_dense.StrongDenseError, CacheAlignmentError, QueryPromptError) as error:
            _die(str(error))

        bytes_on_disk = _scale_cache_bytes(corpus_cache, corpus_key, question_cache, query_key)
        try:
            embedding_path = scale_sensitivity.write_embedding(
                target_dir,
                model=model,
                resolved_revision=snapshot.name,
                weights_sha256=weights_digest,
                query_prompt=query_prompt_of(guarded_queries),
                corpus_cache_key=corpus_key,
                question_cache_key=query_key,
                unit_ids=cached_unit_ids,
                n_questions=len(dev),
                dim=int(vectors.shape[1]),
                max_seq_length=_max_seq_length(corpus_model),
                selection_digest=corpus.selection_digest,
                hardware=measurement_host["host"],
                embedding_seconds=embedding_seconds,
                question_seconds=question_seconds,
                bytes_on_disk=bytes_on_disk,
                hourly_rate_usd=hourly_rate_usd,
            )
            embedding = scale_sensitivity.load_embedding(target_dir, model)
        except scale_sensitivity.ScaleSensitivityError as error:
            _die(str(error))
        print(f"[OK] one C500 encoding recorded -> {embedding_path}")

    def retriever_for_size(size: int):
        return scale_sensitivity.prefix_retriever(
            vectors,
            cached_unit_ids,
            size=size,
            backend=query_lookup,
            name=scale_sensitivity.run_name(model, size),
        )

    reproduction = scale_sensitivity.load_reproduction(target_dir)
    level_1_supported = int(reproduction["models"][model]["supported"])
    provenance = {
        "model": pins.model,
        "model_key": model,
        "revision": pins.revision,
        "resolved_revision": snapshot.name,
        "weights_sha256": weights_digest,
        "dim": pins.dim,
        "query_prompt": pins.query_prompt,
        "corpus_cache_key": corpus_key,
        "question_cache_key": query_key,
        "question_set_hash": dev_question_hash,
        "selection_digest": corpus.selection_digest,
        "embedding_digest": embedding["digest"],
        "unit_set_hash": unit_set_hash(unit_ids),
        "unit_set_hashes": {
            str(size): unit_set_hash(unit_ids[:size]) for size in config.PHASE_8_1_CORPUS_SIZES
        },
        "seed": config.DEFAULT_SEED,
        "tokenizer": config.BUDGET_TOKENIZER_ID,
        "code_version": __version__,
        "top_k": config.EVALUATION_TOP_K,
    }
    retrieval_cost = {
        "method": "exact DenseRetriever over aligned prefixes of one C500 vector block",
        "query_seconds_ceiling": config.PHASE_8_1_QUERY_SECONDS_CEILING,
        "fallback_used": False,
        "basis": (
            "per-scale mean latency is recorded by the evaluation harness; the C500 "
            "measurement is the feasibility probe"
        ),
    }
    try:
        path = scale_sensitivity.measure_scales(
            target_dir,
            model=model,
            retriever_for_size=retriever_for_size,
            questions=dev,
            token_counts=token_counts,
            provenance=provenance,
            level_1_supported=level_1_supported,
            retrieval_cost=retrieval_cost,
            host=measurement_host["host"],
            sizes=config.PHASE_8_1_CORPUS_SIZES,
            level_one_on_measurement_host=measurement_host["level_one_on_measurement_host"],
            expected_questions=config.N_DEV,
            peak_rss_mb=_scale_peak_rss_mb,
        )
    except scale_sensitivity.ScaleSensitivityError as error:
        _die(str(error))

    body = json.loads(path.read_text(encoding="utf-8"))
    probe = body["retrieval_cost"]
    if "mean_seconds_per_query" in probe:
        print(
            f"[INFO] C500 retrieval probe {model}: "
            f"{probe['mean_seconds_per_query']:.6f} s/query, "
            f"peak RSS {probe['peak_rss_mb']:.1f} MiB"
        )
        if probe["exceeds_query_seconds_ceiling"]:
            print(
                "[WARN] C500 exact-retrieval probe exceeds the frozen ceiling; "
                "preserve this measurement and implement the declared blockwise exact fallback "
                "before any further scale run"
            )

    level_2 = body["level_2"]
    mark = "[OK]" if level_2["within_tolerance"] else "[ERROR]"
    print(
        f"{mark} level 2 {model}: {level_2['supported']} vs "
        f"{level_2['level_1_supported']} ({level_2['difference']:+d})"
    )
    if body["terminal_state"] is None:
        print(f"[OK] measured C19/C100/C250/C500 -> {path}")
    else:
        print(f"[ERROR] {body['terminal_state']} recorded -> {path}")
    return path


def cmd_scale_outcome(target_dir: Path = config.PHASE_8_1_DIR) -> Path:
    """Deviation 8.1 S7: classify the two complete S6 measurements once."""
    target_dir = Path(target_dir)
    reconciliation_path = target_dir / scale_corpus.RECONCILIATION_FILENAME
    if not reconciliation_path.exists():
        _die(
            f"{reconciliation_path} does not exist: run 'scale-corpus' before classifying "
            "the scale measurements"
        )
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    try:
        bge = scale_sensitivity.load_scale(target_dir, "bge")
        qwen = scale_sensitivity.load_scale(target_dir, "qwen")
        path = scale_sensitivity.classify_outcome(
            target_dir,
            bge=bge,
            qwen=qwen,
            reconciliation=reconciliation,
        )
    except scale_sensitivity.ScaleSensitivityError as error:
        _die(str(error))

    body = json.loads(path.read_text(encoding="utf-8"))
    print(f"[OK] deviation 8.1 outcome: {body['terminal_state']} (rule {body['rule']})")
    for reading in body["headline"]:
        print(
            f"[OK] C{reading['corpus']}: BGE {reading['bge']} / {reading['n_questions']}, "
            f"Qwen {reading['qwen']} / {reading['n_questions']}, "
            f"difference {reading['deficit_questions']:+d}"
        )
    print(f"[OK] outcome recorded -> {path}")
    return path


# --- Phase 9: HotpotQA FullWiki ----------------------------------------------------------


def cmd_fullwiki_corpus(
    archive: str | None = None, target_dir: Path = config.PHASE_9_DIR
) -> dict[str, Any]:
    """S2: the verified official archive, whole, as `corpus.jsonl.gz` and `corpus.json`.

    Identity is D1's: the 8.1 bytes and md5, and the sha256 8.1 measured. A recorded
    `corpus.json` is a pin the stream must reproduce, which is how the measurement host
    proves its rebuilt corpus is this one.
    """
    path = _source_path(archive)
    print(f"[INFO] streaming {path.name} into {target_dir}")
    try:
        manifest = fullwiki.write_corpus(
            path,
            Path(target_dir),
            expected_bytes=config.PHASE_8_1_DECLARED_BYTES,
            expected_md5=config.PHASE_8_1_DECLARED_MD5,
            expected_sha256=config.PHASE_9_ARCHIVE_SHA256,
        )
    except fullwiki.FullWikiError as error:
        _die(str(error))
    print(
        f"[OK] corpus: {manifest['n_units']} units from {manifest['records_read']} records "
        f"({manifest['duplicate_records_collapsed']} duplicates collapsed, "
        f"{manifest['empty_text_units']} without a sentence), unit set "
        f"{manifest['unit_set_hash']}"
    )
    return manifest


def _phase_9_corpus(target_dir: Path) -> tuple[list[IndexingUnit], dict[str, Any]]:
    try:
        units = fullwiki.load_corpus(target_dir)
    except fullwiki.FullWikiError as error:
        _die(str(error))
    manifest = json.loads((target_dir / fullwiki.CORPUS_MANIFEST).read_text(encoding="utf-8"))
    return units, manifest


def cmd_fullwiki_questions(
    data_dir: Path = config.DATA_DIR, target_dir: Path = config.PHASE_9_DIR
) -> Path:
    """S3: the 7,405 / 2,000 / 5,405 cohorts by qid and the gold resolved by title (D4).

    Written before any Phase 9 retrieval exists. A DATA_STOP is written too, with the
    unresolved titles and qids, and the stage then exits non-zero: the stop is a result.
    """
    data_dir, target_dir = Path(data_dir), Path(target_dir)
    if not (data_dir / RAW_NAME).exists() or not (data_dir / POOL_NAME).exists():
        _die("the validation source or the Phase 1 pool is missing: run 'fetch' and 'build'")
    raws = json.loads((data_dir / RAW_NAME).read_text(encoding="utf-8"))
    _pool_units, historical = load_pool(data_dir / POOL_NAME)
    units, corpus = _phase_9_corpus(target_dir)
    try:
        _questions, body = phase9.build_questions(
            raws, [question.qid for question in historical], units, corpus=corpus
        )
        path = phase9.write_questions(target_dir, body)
    except phase9.FullWikiPhaseError as error:
        _die(str(error))
    print(
        f"[INFO] cohorts: {body['n_standard_dev']} standard, {body['n_historical_overlap']} "
        f"historical overlap, {body['n_retrieval_unseen']} retrieval-unseen"
    )
    print(
        f"[INFO] titles: {body['title_resolution']}; questions with unresolved gold "
        f"{body['questions_with_unresolved_gold']} (ceiling {body['unresolved_ceiling']})"
    )
    if body["terminal_state"] is not None:
        _die(f"{body['terminal_state']} recorded in {path}")
    print(f"[OK] questions frozen -> {path}")
    return path


REPRODUCTION_NAME = "reproduction.json"


def cmd_fullwiki_repro(
    data_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    question_cache_dir: Path = config.QUESTION_CACHE_DIR,
    target_dir: Path = config.PHASE_9_DIR,
    backend: EmbeddingBackend | None = None,
) -> Path:
    """S4: the Phase 9 systems and measurement path, over the historical 19,366 pool.

    The 600 historical dev questions, the historical caches and the Phase 7 GLiNER index:
    every input a Phase 7 dev figure was read from, and nothing new. The three Full Support
    @2,048 counts must equal the recorded ones exactly; a miss stops, and no code is changed
    to make a recorded figure come back.
    """
    data_dir, target_dir = Path(data_dir), Path(target_dir)
    unit_ids, dev, token_counts, dense, run_config = _cheap_inputs(
        DEV_SPLIT, data_dir, cache_dir, question_cache_dir, backend
    )
    check_dev_only(dev)
    units, _questions = load_pool(data_dir / POOL_NAME)
    try:
        index = load_node_index(
            config.PHASE_8_GLINER_DIR,
            extraction_digest=config.PHASE_8_GLINER_EXTRACTION_DIGEST,
            expected_unit_ids=unit_ids,
        )
        if index.digest != config.PHASE_8_GLINER_INDEX_DIGEST:
            _die(f"the Phase 7 GLiNER index has digest {index.digest}, not the pinned one")
        weights = phase9.read_frozen_weights()
    except (NodeIndexError, phase9.FullWikiPhaseError) as error:
        _die(str(error))
    stage = EntityHopStage(index, node_weights(index))
    systems = phase9.build_systems(dense, BM25Retriever(units), stage, weights)
    measured: dict[str, int] = {}
    latency: dict[str, dict[str, Any]] = {}
    primary = str(config.PHASE_9_PRIMARY_BUDGET)
    for name, system in systems.items():
        print(f"[INFO] measuring {name} over {len(dev)} historical dev questions")
        records = phase9.measure_system(name, system, dev, token_counts, run_config=run_config)
        measured[name] = int(sum(r["budgets"][primary]["full_support"] for r in records))
        latency[name] = phase9.latency_summary(records)
        print(f"[INFO] {name}: {measured[name]} / {len(dev)} supported @{primary}")
    verdict = phase9.reproduction_verdict(measured, config.PHASE_9_REPRODUCTION_DEV_SUPPORTED)
    body = {
        **verdict,
        "n_questions": len(dev),
        "metric": f"full_support@{primary}_tokens",
        "weights": weights,
        "weights_files": {
            "hybrid-bm25": str(config.PHASE_9_BM25_WEIGHTS_FILE.relative_to(config.DATA_DIR)),
            "hybrid-entity-hop": str(
                config.PHASE_9_ENTITY_WEIGHTS_FILE.relative_to(config.DATA_DIR)
            ),
        },
        "node_index_digest": index.digest,
        "extraction_digest": index.extraction_digest,
        "run_config": run_config,
        "latency": latency,
        "code_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    path = write_text_atomic(
        target_dir / REPRODUCTION_NAME, json.dumps(body, indent=2, sort_keys=True)
    )
    if not verdict["reproduced"]:
        _die(f"the reproduction missed the recorded counts {verdict['mismatches']}; see {path}")
    print(f"[OK] Phase 7 dev counts reproduced exactly -> {path}")
    return path


BUILD_STAGES: tuple[str, ...] = ("tokens", "embed", "bm25", "extract", "index")
SMOKE_DIRNAME = "smoke"


def _historical_dev_qids(data_dir: Path) -> set[str]:
    _pool_units, pool_questions = load_pool(Path(data_dir) / POOL_NAME)
    return {question.qid for question in pool_questions if question.split == DEV_SPLIT}


def _build_inputs(
    target_dir: Path, smoke: int | None
) -> tuple[list[IndexingUnit], dict[str, Any], Path]:
    """The corpus (or its first `smoke` units) and where this build writes.

    A smoke build reads the real corpus and writes under `smoke/`, never beside the frozen
    artifacts: it exists to exercise the code path, and its numbers are not Phase 9 figures.
    """
    units, corpus = _phase_9_corpus(target_dir)
    if smoke is None:
        return units, corpus, target_dir
    units = units[:smoke]
    corpus = {**corpus, "unit_set_hash": unit_set_hash([u.unit_id for u in units]), "smoke": smoke}
    return units, corpus, target_dir / SMOKE_DIRNAME


def cmd_fullwiki_build(
    stage: str,
    *,
    hourly_rate_usd: float = 0.0,
    smoke: int | None = None,
    target_dir: Path = config.PHASE_9_DIR,
) -> dict[str, Any]:
    """S5: one build step, run as its own process so it records its own peak memory."""
    target_dir = Path(target_dir)
    if stage not in BUILD_STAGES:
        _die(f"unknown build stage {stage!r}; expected one of {BUILD_STAGES}")
    units, corpus, out = _build_inputs(target_dir, smoke)
    corpus_hash = str(corpus["unit_set_hash"])
    unit_ids = [unit.unit_id for unit in units]
    device = _embedding_device()
    print(f"[INFO] build {stage} over {len(units)} units into {out} ({device})")
    try:
        if stage == "tokens":
            body = phase9.build_token_counts(units, TokenCounter(), out, unit_set_hash=corpus_hash)
        elif stage == "embed":
            questions, _body = phase9.load_questions(
                target_dir,
                corpus_unit_set_hash=str(
                    json.loads((target_dir / fullwiki.CORPUS_MANIFEST).read_text("utf-8"))[
                        "unit_set_hash"
                    ]
                ),
            )
            if smoke is not None:
                dev_qids = _historical_dev_qids(config.DATA_DIR)
                questions = [question for question in questions if question.qid in dev_qids]
            snapshot = snapshot_directory(config.EMBEDDING_MODEL, config.EMBEDDING_REVISION)
            body = phase9.build_embedding(
                units,
                questions,
                SentenceTransformerBackend(),
                out / "cache",
                out / "cache" / "questions",
                out,
                unit_set_hash=corpus_hash,
                weights_sha256=weights_sha256(snapshot, "model.safetensors"),
                resolved_revision=snapshot.name,
                hardware=local_extraction.hardware_block(device=device),
                hourly_rate_usd=hourly_rate_usd,
            )
        elif stage == "bm25":
            _retriever, body = phase9.build_bm25(units, unit_set_hash=corpus_hash)
            write_text_atomic(
                out / phase9.BM25_FILENAME, json.dumps(body, indent=2, sort_keys=True)
            )
        elif stage == "extract":
            body = _fullwiki_extract(units, out, corpus_hash, hourly_rate_usd, smoke)
        else:
            gliner_dir = out / phase9.GLINER_DIRNAME
            records, manifest = local_extraction.load_extraction(
                gliner_dir, expected_unit_ids=unit_ids
            )
            body = phase9.build_entity_index(
                records,
                unit_ids,
                gliner_dir,
                extraction_digest=str(manifest["digest"]),
                corpus_unit_set_hash=corpus_hash,
            )
    except (
        phase9.FullWikiPhaseError,
        LocalExtractionError,
        NodeIndexError,
        BackendError,
        CacheAlignmentError,
    ) as error:
        _die(str(error))
    print(f"[OK] build {stage}: {json.dumps(body.get('memory'))}")
    return body


def _fullwiki_extract(
    units: Sequence[IndexingUnit],
    out: Path,
    corpus_hash: str,
    hourly_rate_usd: float,
    smoke: int | None,
) -> dict[str, Any]:
    """GLiNER over every unit in resumable shards, at the frozen Phase 7 configuration (D6).

    The configuration digest covers the library versions, so it is checked before a single
    paragraph is read; a smoke build on another stack records the mismatch instead, and
    its records are never Phase 9 records.
    """
    gliner_dir = out / phase9.GLINER_DIRNAME
    model_dir = config.PHASE_9_DIR / "models" / config.GLINER_EXTRACTOR
    extractor, load_seconds = local_extraction.gliner_extractor(model_dir)
    digest = extractor.configuration_digest
    pinned = config.PHASE_9_GLINER_CONFIGURATION_DIGEST
    if digest != pinned and smoke is None:
        _die(
            f"the GLiNER configuration digest here is {digest}, not the frozen {pinned}: the "
            f"library stack differs from Phase 7's ({extractor.library_versions})"
        )
    weights = weights_sha256(model_dir, config.PHASE_9_GLINER_WEIGHTS_FILE)
    print(f"[INFO] GLiNER {digest} on {extractor.hardware_device}, weights {weights[:16]}")
    started = time.perf_counter()
    records, summary = local_extraction.run_sharded_extraction(
        units,
        extractor,
        gliner_dir,
        shard_units=config.PHASE_9_EXTRACTION_SHARD_UNITS if smoke is None else 100,
        on_progress=_progress("extract", started),
    )
    hardware = local_extraction.hardware_block(device=extractor.hardware_device)
    manifest = local_extraction.build_manifest(
        records,
        extractor,
        seconds=float(summary["seconds"]),
        hardware=hardware,
        usd=float(summary["seconds"]) / 3600.0 * hourly_rate_usd,
        hourly_rate_usd=hourly_rate_usd,
        model_load_seconds=load_seconds,
    )
    failures = manifest["failures"]
    manifest.update(
        {
            "phase": 9,
            "weights_sha256": weights,
            "configuration_digest_pinned": pinned,
            "configuration_digest_matches": digest == pinned,
            "corpus_unit_set_hash": corpus_hash,
            "attempted": summary["attempted"],
            "succeeded": summary["attempted"] - sum(failures.values()),
            "sharding": summary,
            "wall_seconds": summary["seconds"],
            "attributable_usd": manifest["usd"],
            "attributable_usd_basis": "derived from measured shard seconds x contracted rate",
            "memory": phase9.peak_memory(),
            "smoke": smoke,
        }
    )
    local_extraction.write_extraction(records, manifest, gliner_dir)
    return manifest


def _phase_9_inputs(target_dir: Path, smoke: int | None = None) -> dict[str, Any]:
    """Everything a probe or the pass reads, loaded once and checked against its manifest.

    Returns the verified components - Dense, BM25, the node index and its weights, the fusion
    weights, the questions and token counts - and the provenance chain every run records:
    corpus, questions, vectors, BM25, extraction and entity index. Systems are built from
    these by `_phase_9_systems`, so a second Entity Hop implementation never reloads or
    rebuilds anything. A smoke build is read from `smoke/` with only the historical dev
    questions: it feeds the probe's code path and nothing else.
    """
    full_hash = str(
        json.loads((target_dir / fullwiki.CORPUS_MANIFEST).read_text("utf-8"))["unit_set_hash"]
    )
    units, corpus, source_dir = _build_inputs(target_dir, smoke)
    corpus_hash = str(corpus["unit_set_hash"])
    unit_ids = [unit.unit_id for unit in units]
    embedding = json.loads((source_dir / phase9.EMBEDDING_FILENAME).read_text("utf-8"))
    bm25_record = json.loads((source_dir / phase9.BM25_FILENAME).read_text("utf-8"))
    gliner_dir = source_dir / phase9.GLINER_DIRNAME
    entity_record = json.loads((gliner_dir / phase9.ENTITY_INDEX_FILENAME).read_text("utf-8"))
    questions, question_body = phase9.load_questions(target_dir, corpus_unit_set_hash=full_hash)
    if smoke is not None:
        dev_qids = _historical_dev_qids(config.DATA_DIR)
        questions = [question for question in questions if question.qid in dev_qids]
    token_counts = phase9.load_phase9_token_counts(
        source_dir, unit_set_hash=corpus_hash, unit_ids=unit_ids
    )
    for name, record in (("embedding", embedding), ("entity index", entity_record)):
        if record.get("unit_set_hash", record.get("corpus_unit_set_hash")) != corpus_hash:
            _die(f"the {name} manifest names another corpus")
    loaded = EmbeddingCache(source_dir / "cache").load(
        str(embedding["corpus_cache_key"]), expected_unit_ids=unit_ids
    )
    if loaded is None:
        _die("the FullWiki vectors are missing: run 'fullwiki-build --stage embed'")
    vectors = loaded[0]
    if phase9.vectors_digest(vectors) != embedding["vectors_digest"]:
        _die("the FullWiki vectors do not match the digest embedding.json records")
    question_ids = [question.qid for question in questions]
    question_vectors = EmbeddingCache(source_dir / "cache" / "questions").load(
        str(embedding["question_cache_key"]), expected_unit_ids=question_ids
    )
    if question_vectors is None:
        _die("the question vectors are missing: run 'fullwiki-build --stage embed'")
    query_backend = CachedQueryBackend(
        texts=[question.question for question in questions],
        vectors=question_vectors[0],
        name=str(embedding["model"]),
        revision=str(embedding["revision"]),
    )
    dense = DenseRetriever(vectors, unit_ids, query_backend)
    bm25, rebuilt = phase9.build_bm25(units, unit_set_hash=corpus_hash)
    if rebuilt["index_digest"] != bm25_record["index_digest"]:
        _die("the rebuilt BM25 index does not match the digest bm25.json records")
    extraction = local_extraction.load_manifest(gliner_dir)
    index = load_node_index(
        gliner_dir, extraction_digest=str(extraction["digest"]), expected_unit_ids=unit_ids
    )
    if index.digest != entity_record["index_digest"]:
        _die("the entity index does not match the digest entity-index.json records")
    weights = phase9.read_frozen_weights()
    provenance = {
        "corpus_unit_set_hash": corpus_hash,
        "corpus_ordered_unit_digest": corpus["ordered_unit_digest"],
        "archive_sha256": corpus["archive_sha256"],
        "question_digest": question_body["question_digest"],
        "mapping_digest": question_body["mapping_digest"],
        "embedding": {
            key: embedding[key]
            for key in (
                "model",
                "revision",
                "weights_sha256",
                "vectors_digest",
                "corpus_cache_key",
                "question_cache_key",
            )
        },
        "bm25_index_digest": bm25_record["index_digest"],
        "extraction": {
            "model": extraction["model"],
            "revision": extraction["revision"],
            "configuration_digest": extraction["configuration_digest"],
            "weights_sha256": extraction.get("weights_sha256"),
            "records_digest": extraction["digest"],
        },
        "entity_index_digest": index.digest,
        "weights": weights,
    }
    return {
        "dense": dense,
        "bm25": bm25,
        "index": index,
        "node_weights": node_weights(index),
        "weights": weights,
        "questions": questions,
        "token_counts": token_counts,
        "provenance": provenance,
    }


def _phase_9_systems(components: dict[str, Any], hop: str) -> dict[str, Any]:
    """Fresh P9-A/B/C over already-loaded components; only the Entity Hop stage differs by hop."""
    stage_type = ColumnwiseEntityHopStage if hop == phase9.COLUMNWISE_HOP else EntityHopStage
    stage = stage_type(components["index"], components["node_weights"])
    return phase9.build_systems(
        components["dense"], components["bm25"], stage, components["weights"]
    )


def cmd_fullwiki_probe(
    data_dir: Path = config.DATA_DIR,
    target_dir: Path = config.PHASE_9_DIR,
    smoke: int | None = None,
) -> Path:
    """S6: mean query time per system on 100 seeded historical dev questions; no metric.

    The reference Entity Hop is timed first. If its mean reaches the plan's share of the
    ceiling, the exact column-wise hop - proven identical in candidates and scores - is
    timed too and becomes the one the pass uses. A system past the ceiling is an
    OPERATIONAL_STOP, written and reported.
    """
    data_dir, target_dir = Path(data_dir), Path(target_dir)
    out = target_dir if smoke is None else target_dir / SMOKE_DIRNAME
    target = out / phase9.PROBE_FILENAME
    if target.exists():
        _die(f"{target} already records the probe")
    dev_qids = _historical_dev_qids(data_dir)
    components = _phase_9_inputs(target_dir, smoke)
    provenance = components["provenance"]
    sample = phase9.probe_sample(components["questions"], dev_qids)
    systems = _phase_9_systems(components, phase9.REFERENCE_HOP)
    seconds = {name: phase9.time_system(system, sample) for name, system in systems.items()}
    reference_mean = statistics.fmean(seconds["hybrid-entity-hop"])
    hop = phase9.hop_implementation(reference_mean)
    timings: dict[str, Any] = {
        "reference_hybrid_entity_hop_mean_seconds": reference_mean,
        "hop_implementation": hop,
        "columnwise_rule": f"columnwise when the reference mean >= {phase9.COLUMNWISE_SHARE} x "
        f"{config.PHASE_9_QUERY_SECONDS_CEILING} s",
    }
    if hop == phase9.COLUMNWISE_HOP:
        # Built from the components already in memory: the reference systems are dropped
        # first, and nothing is reloaded or rebuilt for the second hop.
        del systems
        columnwise = _phase_9_systems(components, phase9.COLUMNWISE_HOP)
        seconds["hybrid-entity-hop"] = phase9.time_system(columnwise["hybrid-entity-hop"], sample)
    means = {name: statistics.fmean(values) for name, values in seconds.items()}
    body = {
        **phase9.probe_verdict(means),
        **timings,
        "n_questions": len(sample),
        "qids": [question.qid for question in sample],
        "sample_rule": f"random.Random({config.PHASE_9_PROBE_SEED}).sample over historical "
        f"dev qids sorted, n = {config.PHASE_9_PROBE_QUESTIONS}; rankings only",
        "seconds": seconds,
        "memory": phase9.peak_memory(),
        "host": local_extraction.hardware_block(device=_embedding_device()),
        "provenance": {**provenance, "hop_implementation": hop},
        "code_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "smoke": smoke,
    }
    body["digest"] = phase9.digest_of(json.dumps(body, sort_keys=True))
    path = write_text_atomic(target, json.dumps(body, indent=2, sort_keys=True))
    for name, mean in means.items():
        print(f"[INFO] {name}: {mean:.3f} s mean over {len(sample)} probe questions")
    if body["terminal_state"] is not None:
        _die(f"{body['terminal_state']}: {body['over_ceiling']} past the ceiling; see {path}")
    print(f"[OK] probe within the envelope ({hop} hop) -> {path}")
    return path


def cmd_fullwiki_eval(*, authorized: bool, target_dir: Path = config.PHASE_9_DIR) -> list[Path]:
    """S6: the single evaluation pass. Refuses without the author's explicit authorization.

    Metrics over every cohort are computed here and nowhere else, once: a second start is
    refused whatever the first one showed.
    """
    target_dir = Path(target_dir)
    if not authorized:
        _die(
            "the evaluation pass opens the 1,400 historical held-out and the 5,405 new "
            "questions; it runs only with --authorized-pass, after the author says so"
        )
    probe = json.loads((target_dir / phase9.PROBE_FILENAME).read_text("utf-8"))
    if probe["terminal_state"] is not None:
        _die(f"the probe recorded {probe['terminal_state']}; the pass does not run")
    if (target_dir / phase9.EVALUATION_MARKER).exists():
        _die(f"{target_dir / phase9.EVALUATION_MARKER} exists: the pass has already started once")
    commit = _git_commit()
    hop = str(probe["hop_implementation"])
    components = _phase_9_inputs(target_dir)
    provenance = {**components["provenance"], "hop_implementation": hop}
    systems = _phase_9_systems(components, hop)
    host = local_extraction.hardware_block(device=_embedding_device())
    # Marked here, after every input loaded and verified and before the first question is
    # evaluated: a load failure leaves the pass unstarted and retryable; anything later
    # leaves it started, and a second start is refused.
    try:
        phase9.start_pass(target_dir, probe_digest=str(probe["digest"]), code_commit=commit)
    except phase9.FullWikiPhaseError as error:
        _die(str(error))
    written: list[Path] = []
    questions, token_counts = components["questions"], components["token_counts"]
    for name, system in systems.items():
        print(f"[INFO] pass: {name} over {len(questions)} questions")
        records = phase9.measure_system(name, system, questions, token_counts)
        described = system.describe() if hasattr(system, "describe") else None
        body = {
            "phase": 9,
            "system": name,
            "label": config.PHASE_9_SYSTEM_LABELS[name],
            "corpus_unit_set_hash": provenance["corpus_unit_set_hash"],
            "question_digest": provenance["question_digest"],
            "ranking_depth": config.PHASE_9_RANKING_DEPTH,
            "fusion": (
                {"scheme": "none", "weights": {}}
                if described is None
                else {"scheme": described["scheme"], "weights": described["weights"]}
            ),
            "metrics": phase9.cohort_metrics(records),
            "latency": phase9.latency_summary(records),
            "entity_candidates": phase9.candidate_summary(records),
            "provenance": {
                **provenance,
                "code_commit": commit,
                "probe_digest": probe["digest"],
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "host": host,
                "memory": phase9.peak_memory(),
            },
        }
        try:
            path = phase9.write_run(target_dir, name, records, body=body)
        except phase9.FullWikiPhaseError as error:
            _die(str(error))
        written.append(path)
        print(f"[OK] {name} run written -> {path}")
    return written


OUTCOME_NAME = "outcome.json"


def cmd_fullwiki_outcome(target_dir: Path = config.PHASE_9_DIR) -> Path:
    """S7: P9-C against P9-A on the primary cohort, the exact test and the D10/D11 label."""
    target_dir = Path(target_dir)
    stop: str | None = None
    reasons: list[str] = []
    questions = json.loads((target_dir / phase9.QUESTIONS_FILENAME).read_text("utf-8"))
    probe_path = target_dir / phase9.PROBE_FILENAME
    probe = json.loads(probe_path.read_text("utf-8")) if probe_path.exists() else None
    if questions.get("terminal_state") is not None:
        stop = str(questions["terminal_state"])
        reasons.append(
            f"questions: {questions['questions_with_unresolved_gold']} with unresolved gold, "
            f"ceiling {questions['unresolved_ceiling']}"
        )
    elif probe is not None and probe["terminal_state"] is not None:
        stop = str(probe["terminal_state"])
        reasons.append(f"probe: {probe['over_ceiling']} past the query ceiling")
    if stop is not None:
        # D11: the pass never ran, so there is no run to read and no retrieval score to
        # publish. The label and its reason are the whole outcome.
        body_stop = {
            "primary_cohort": config.PHASE_9_RETRIEVAL_UNSEEN,
            "primary_metric": f"full_support@{config.PHASE_9_PRIMARY_BUDGET}_tokens",
            "terminal_state": stop,
            "stop_reasons": reasons,
            "code_commit": _git_commit(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path = write_text_atomic(
            target_dir / OUTCOME_NAME, json.dumps(body_stop, indent=2, sort_keys=True)
        )
        print(f"[OK] {stop}: {'; '.join(reasons)} -> {path}")
        return path
    runs = {
        name: json.loads((target_dir / f"run-{name}.json").read_text("utf-8"))
        for name in config.PHASE_9_SYSTEMS
    }
    slow = [
        name
        for name, run in runs.items()
        if run["latency"]["mean_ms"] / 1000.0 > config.PHASE_9_QUERY_SECONDS_CEILING
    ]
    if slow:
        stop = phase9.OPERATIONAL_STOP
        reasons.append(f"pass: {slow} past the mean query ceiling")
    dense = phase9.load_outcomes(target_dir, "dense")
    entity = phase9.load_outcomes(target_dir, "hybrid-entity-hop")
    bm25 = phase9.load_outcomes(target_dir, "hybrid-bm25")
    body = {
        **phase9.primary_outcome(dense, entity, stop=stop),
        "stop_reasons": reasons,
        "secondary_entity_vs_bm25": phase9.primary_outcome(bm25, entity),
        "secondary_note": "P9-C against P9-B, descriptive only; it cannot change the label",
        "runs": {name: run["outcomes_digest"] for name, run in runs.items()},
        "code_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    body["secondary_entity_vs_bm25"].pop("terminal_state")
    path = write_text_atomic(target_dir / OUTCOME_NAME, json.dumps(body, indent=2, sort_keys=True))
    print(
        f"[OK] {body['terminal_state']}: wins {body['wins']}, losses {body['losses']}, "
        f"p = {body['exact_two_sided_p']:.4g} -> {path}"
    )
    return path


def _git_commit() -> str:
    """The commit the code ran at, or `unknown` outside a checkout; recorded, never trusted."""
    import subprocess  # nosec B404: a fixed argv, no shell, no input

    try:
        completed = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=config.PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _cache_key_for(backend: EmbeddingBackend, units: Sequence[IndexingUnit]) -> str:
    return cache_key(
        backend.name,
        backend.revision,
        unit_set_hash([u.unit_id for u in units]),
        normalized=getattr(backend, "normalize", True),
    )


# The HotpotQA split the Phase 10 sets are drawn from (D2). It is named here, not in
# config.py, because the dev-selection guard forbids every other split name there.
P10_TRAIN_SPLIT = "train"
P10_TRAIN_NAME = "hotpot_train_hard_source.json"
P10_TRAIN_MANIFEST = "train-source.json"


def _p10_train(target_dir: Path) -> list[dict[str, Any]]:
    """HotpotQA train, fetched once and frozen: every row, without its distractor contexts.

    The contexts are not used by any Phase 10 step (the corpus is FullWiki), so only the
    fields the draw and the gold mapping read are kept. The first run records the file's
    SHA-256 and bytes; every later run verifies them.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_path = target_dir / P10_TRAIN_NAME
    manifest_path = target_dir / P10_TRAIN_MANIFEST
    if not raw_path.exists():
        print(f"[INFO] assembling {hf_source.DATASET} ({hf_source.CONFIG}/train)")
        fetched = hf_source.fetch_split(
            pause=0.5, split=P10_TRAIN_SPLIT, max_rows=config.PHASE_10_TRAIN_MAX_ROWS
        )
        keep = ("_id", "question", "answer", "type", "level", "supporting_facts")
        slim = [{key: row[key] for key in keep} for row in fetched]
        raw_path.write_text(json.dumps(slim, sort_keys=True), encoding="utf-8")
        manifest = {
            "dataset": hf_source.DATASET,
            "config": hf_source.CONFIG,
            "split": P10_TRAIN_SPLIT,
            "endpoint": hf_source.ROWS_ENDPOINT,
            "rows": len(slim),
            "fields_kept": list(keep),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_of_file(raw_path),
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        write_text_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
        print(f"[OK] train frozen: {len(slim)} rows, sha256 {str(manifest['sha256'])[:16]}...")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_of_file(raw_path) != manifest["sha256"]:
        _die(f"{raw_path} does not match the SHA-256 {manifest_path.name} records")
    rows: list[dict[str, Any]] = json.loads(raw_path.read_text(encoding="utf-8"))
    return rows


def cmd_p10_questions(
    data_dir: Path = config.DATA_DIR,
    source_dir: Path = config.PHASE_9_DIR,
    target_dir: Path = config.PHASE_10_DIR,
) -> Path:
    """S1: the three held-out sets, drawn at once and gold-mapped, before any retrieval (D2).

    A DATA_STOP is written with its evidence and the stage exits non-zero: the stop is a
    result, not an error to retry past.
    """
    data_dir, source_dir, target_dir = Path(data_dir), Path(source_dir), Path(target_dir)
    if (target_dir / phase10.QUESTIONS_FILENAME).exists():
        _die(f"{target_dir / phase10.QUESTIONS_FILENAME} already freezes the sets")
    validation = json.loads((data_dir / RAW_NAME).read_text(encoding="utf-8"))
    train = _p10_train(target_dir)
    try:
        phase10.check_validation_level(validation)
        hard = phase10.hard_qids(train)
        sets = phase10.draw_sets(hard, validation_qids=[str(raw["_id"]) for raw in validation])
    except phase10.DataStop as error:
        _die(f"DATA_STOP: {error}")
    sizes = dict(config.PHASE_10_SET_SIZES)
    print(f"[INFO] {len(train)} train rows, {len(hard)} hard; drawing {sizes}")
    units, corpus = _phase_9_corpus(source_dir)
    _questions, body = phase10.build_sets(train, sets, units, corpus=corpus)
    body["source"] = {
        **json.loads((target_dir / P10_TRAIN_MANIFEST).read_text(encoding="utf-8")),
        "hard_questions": len(hard),
        "seed": config.PHASE_10_SEED,
        "draw": "random.Random(seed).sample(sorted hard qids, sum of sizes), sliced in set order",
        "validation_level_checked": config.PHASE_10_LEVEL,
    }
    path = phase10.write_questions(target_dir, body)
    for name, record in body["sets"].items():
        print(
            f"[INFO] {name}: {record['n_questions']} questions, "
            f"{record['questions_with_unresolved_gold']} with unresolved gold"
        )
    if body["terminal_state"] is not None:
        _die(f"{body['terminal_state']} recorded in {path}")
    print(f"[OK] Phase 10 sets frozen -> {path}")
    return path


P10_EMBED_NAME = "embed.json"
P10_REPRODUCTION_NAME = "reproduction.json"
P10_PROBE_NAME = "probe.json"
P10_FIT_NAME = "fit.json"
P10_MARKER_NAME = "pass.json"
P10_OUTCOME_NAME = "outcome.json"
P10_EMBED_SETS: tuple[str, ...] = (config.PHASE_10_PROBE, config.PHASE_10_TEST)


def _p10_json(name: str, target_dir: Path = config.PHASE_10_DIR) -> dict[str, Any]:
    path = Path(target_dir) / name
    if not path.exists():
        _die(f"{path} does not exist: run the Phase 10 stage that writes it first")
    body: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return body


def _p10_write_once(name: str, body: Mapping[str, Any], target_dir: Path) -> Path:
    path = Path(target_dir) / name
    if path.exists():
        _die(f"{path} already exists; a Phase 10 artifact is written once")
    write_text_atomic(path, json.dumps(dict(body), indent=2, sort_keys=True))
    return path


def _p10_corpus_hash(source_dir: Path) -> str:
    corpus = json.loads((source_dir / fullwiki.CORPUS_MANIFEST).read_text(encoding="utf-8"))
    return str(corpus["unit_set_hash"])


def _p10_query_backend(
    questions: Sequence[Question], set_name: str, target_dir: Path
) -> CachedQueryBackend:
    """The set's cached BGE question vectors, under the key `p10-embed` recorded."""
    record = _p10_json(P10_EMBED_NAME, target_dir)["sets"][set_name]
    loaded = EmbeddingCache(Path(target_dir) / "cache" / "questions").load(
        str(record["key"]), expected_unit_ids=[q.qid for q in questions]
    )
    if loaded is None:
        _die(f"the {set_name} question vectors are missing: run 'p10-embed'")
    return CachedQueryBackend(
        texts=[q.question for q in questions],
        vectors=loaded[0],
        name=str(record["model"]),
        revision=str(record["revision"]),
    )


def cmd_p10_embed(
    source_dir: Path = config.PHASE_9_DIR, target_dir: Path = config.PHASE_10_DIR
) -> Path:
    """S2: BGE question vectors for `probe` and `test-10`, in Phase 10's own cache.

    `test-11` is not encoded here: the next phase encodes the set it is allowed to use.
    """
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    corpus_hash = _p10_corpus_hash(source_dir)
    backend = SentenceTransformerBackend()
    if (backend.name, backend.revision) != (config.EMBEDDING_MODEL, config.EMBEDDING_REVISION):
        _die(f"Phase 10 encodes with the pinned BGE-small only, not {backend.name}")
    cache = EmbeddingCache(target_dir / "cache" / "questions")
    sets: dict[str, Any] = {}
    for name in P10_EMBED_SETS:
        questions = phase10.load_set(target_dir, name, corpus_unit_set_hash=corpus_hash)
        started = time.perf_counter()
        vectors, _ = embed_questions(questions, backend, cache)
        sets[name] = {
            "key": question_cache_key(
                backend.name,
                backend.revision,
                question_set_hash([q.qid for q in questions]),
                name,
                normalized=bool(getattr(backend, "normalize", True)),
                query_prompt=query_prompt_of(backend),
            ),
            "n_questions": len(questions),
            "dim": int(vectors.shape[1]),
            "model": backend.name,
            "revision": backend.revision,
            "resolved_revision": resolved_revision(backend),
            "seconds": time.perf_counter() - started,
        }
        print(f"[INFO] {name}: {len(questions)} questions encoded")
    body = {"sets": sets, "device": _embedding_device()}
    path = _p10_write_once(P10_EMBED_NAME, body, target_dir)
    print(f"[OK] Phase 10 question vectors -> {path}")
    return path


def _p10_replay_records(
    name: str,
    rankings: Sequence[tuple[str, list[Any]]],
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    replay = phase10.ReplayRetriever(name, rankings)
    return phase9.measure_system(name, RecordingRetriever(replay), questions, token_counts)


def cmd_p10_repro(
    source_dir: Path = config.PHASE_9_DIR, target_dir: Path = config.PHASE_10_DIR
) -> Path:
    """S4 (D4): the three component lists per dev question, and the Phase 9 dev counts.

    Every Phase 9 artifact is loaded and verified by `_phase_9_inputs`. Dense, BM25 and the
    column-wise Entity Hop run once per dev question; P9-B and P9-C are rebuilt from those
    lists with the same `fuse` call their `FusedRetriever` makes (tested equal). A miss on
    any of the three counts stops the phase with the differing questions named.
    """
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    if (target_dir / P10_REPRODUCTION_NAME).exists():
        _die(f"{target_dir / P10_REPRODUCTION_NAME} already records the reproduction")
    components = _phase_9_inputs(source_dir)
    dense, bm25 = components["dense"], components["bm25"]
    stage = ColumnwiseEntityHopStage(components["index"], components["node_weights"])
    questions: list[Question] = components["questions"]
    depth = config.PHASE_9_RANKING_DEPTH
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, question in enumerate(questions, start=1):
        first = dense.retrieve(question.question, depth)
        rows.append(
            {
                "qid": question.qid,
                "question": question.question,
                "dense": first,
                "bm25": bm25.retrieve(question.question, depth),
                config.ENTITY_HOP_NAME: stage.propose(list(first), depth),
            }
        )
        if position % 500 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[INFO] dev lists {position}/{len(questions)} in {elapsed / 60:.1f} min",
                flush=True,
            )
    provenance = {
        **components["provenance"],
        "hop_implementation": phase9.COLUMNWISE_HOP,
        "code_commit": _git_commit(),
        "host": local_extraction.hardware_block(device="cpu"),
        "seconds": time.perf_counter() - started,
    }
    phase10.write_dev_lists(target_dir, rows, provenance=provenance)
    weights = components["weights"]
    wb, we = weights["hybrid-bm25"], weights["hybrid-entity-hop"]
    rankings = {
        "dense": [(r["question"], list(r["dense"])) for r in rows],
        "hybrid-bm25": [
            (
                r["question"],
                fuse_lists((r["dense"], r["bm25"]), (wb["dense"], wb["bm25"]), top_k=depth),
            )
            for r in rows
        ],
        "hybrid-entity-hop": [
            (
                r["question"],
                fuse_lists(
                    (r["dense"], r[config.ENTITY_HOP_NAME]),
                    (we["dense"], we[config.ENTITY_HOP_NAME]),
                    top_k=depth,
                ),
            )
            for r in rows
        ],
    }
    token_counts = components["token_counts"]
    counts: dict[str, int] = {}
    differing: dict[str, list[str]] = {}
    for name, ranking in rankings.items():
        records = _p10_replay_records(name, ranking, questions, token_counts)
        counts[name] = phase10.supported(records)
        recorded = {
            r["qid"]: r["budgets"]["2048"]["full_support"]
            for r in phase9.load_outcomes(source_dir, name)
        }
        differing[name] = sorted(
            r["qid"] for r in records if r["budgets"]["2048"]["full_support"] != recorded[r["qid"]]
        )
    body = {
        **phase10.reproduction_verdict(counts),
        "differing_qids": differing,
        "dev_lists_digest": _p10_json(phase10.DEV_LISTS_MANIFEST, target_dir)["digest"],
        "code_commit": _git_commit(),
    }
    path = _p10_write_once(P10_REPRODUCTION_NAME, body, target_dir)
    for name, check in body["checks"].items():
        print(f"[INFO] {name}: {check['observed']} (recorded {check['recorded']})")
    if not body["passed"]:
        _die(f"the laptop does not reproduce the Phase 9 dev counts; see {path}")
    print(f"[OK] Phase 9 dev counts reproduced on this host -> {path}")
    return path


def _p10_dense_only(source_dir: Path) -> tuple[Any, list[str], dict[str, int]]:
    """Corpus ids, verified vectors and token counts: what Dense alone needs (the probe)."""
    units, corpus = _phase_9_corpus(source_dir)
    unit_ids = [unit.unit_id for unit in units]
    del units
    embedding = json.loads((source_dir / phase9.EMBEDDING_FILENAME).read_text(encoding="utf-8"))
    loaded = EmbeddingCache(source_dir / "cache").load(
        str(embedding["corpus_cache_key"]), expected_unit_ids=unit_ids
    )
    if loaded is None:
        _die("the FullWiki vectors are missing")
    vectors = loaded[0]
    if phase9.vectors_digest(vectors) != embedding["vectors_digest"]:
        _die("the FullWiki vectors do not match the digest embedding.json records")
    token_counts = phase9.load_phase9_token_counts(
        source_dir, unit_set_hash=str(corpus["unit_set_hash"]), unit_ids=unit_ids
    )
    return vectors, unit_ids, token_counts


def cmd_p10_probe(
    source_dir: Path = config.PHASE_9_DIR, target_dir: Path = config.PHASE_10_DIR
) -> Path:
    """S4 (D3): Dense alone on `probe`, against its reproduced dev figure, before `test-10`."""
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    reproduction = _p10_json(P10_REPRODUCTION_NAME, target_dir)
    if not reproduction["passed"]:
        _die("the reproduction did not pass; the probe does not run")
    if (target_dir / P10_PROBE_NAME).exists():
        _die(f"{target_dir / P10_PROBE_NAME} already records the probe")
    corpus_hash = _p10_corpus_hash(source_dir)
    questions = phase10.load_set(
        target_dir, config.PHASE_10_PROBE, corpus_unit_set_hash=corpus_hash
    )
    vectors, unit_ids, token_counts = _p10_dense_only(source_dir)
    dense = DenseRetriever(
        vectors, unit_ids, _p10_query_backend(questions, config.PHASE_10_PROBE, target_dir)
    )
    records = phase9.measure_system("dense", RecordingRetriever(dense), questions, token_counts)
    verdict = phase10.probe_verdict(
        phase10.supported(records),
        len(questions),
        int(reproduction["checks"]["dense"]["observed"]),
        config.PHASE_9_COHORT_SIZES[config.PHASE_9_STANDARD],
    )
    body = {**verdict, "latency": phase9.latency_summary(records), "code_commit": _git_commit()}
    path = _p10_write_once(P10_PROBE_NAME, body, target_dir)
    print(
        f"[INFO] Dense Full Support @2048: probe {verdict['probe_fs_pct']:.2f} %, dev "
        f"{verdict['dev_fs_pct']:.2f} %, difference {verdict['difference_pp']:+.2f} pp"
    )
    if verdict["terminal_state"] is not None:
        _die(f"{verdict['terminal_state']}: train questions too contaminated; see {path}")
    print(f"[OK] probe within the margin -> {path}")
    return path


def cmd_p10_fit(
    source_dir: Path = config.PHASE_9_DIR, target_dir: Path = config.PHASE_10_DIR
) -> Path:
    """S5 (D5): the 66-point three-way grid on the cached dev lists, the tie rule, the gate."""
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    reproduction = _p10_json(P10_REPRODUCTION_NAME, target_dir)
    probe = _p10_json(P10_PROBE_NAME, target_dir)
    if not reproduction["passed"] or probe["terminal_state"] is not None:
        _die("the reproduction or the probe stopped the phase; nothing is fitted")
    if (target_dir / P10_FIT_NAME).exists():
        _die(f"{target_dir / P10_FIT_NAME} already records the fit")
    units, corpus = _phase_9_corpus(source_dir)
    unit_ids = [unit.unit_id for unit in units]
    del units
    token_counts = phase9.load_phase9_token_counts(
        source_dir, unit_set_hash=str(corpus["unit_set_hash"]), unit_ids=unit_ids
    )
    questions, _body = phase9.load_questions(
        source_dir, corpus_unit_set_hash=str(corpus["unit_set_hash"])
    )
    rows = phase10.load_dev_lists(target_dir)
    if [r["qid"] for r in rows] != [q.qid for q in questions]:
        _die("the dev lists are not in the dev question order")
    name = config.PHASE_10_SYSTEMS[2]
    depth = config.PHASE_9_RANKING_DEPTH
    curve: list[dict[str, Any]] = []
    for weights in phase10.weight_grid():
        rankings = [
            (
                r["question"],
                fuse_lists(
                    (r["dense"], r["bm25"], r[config.ENTITY_HOP_NAME]), weights, top_k=depth
                ),
            )
            for r in rows
        ]
        records = _p10_replay_records(name, rankings, questions, token_counts)
        curve.append(
            {
                "weights": weights,
                "supported": phase10.supported(records),
                "gold_recall_sum": phase10.gold_recall_sum(records),
            }
        )
        print(f"[INFO] {weights}: {curve[-1]['supported']}", flush=True)
    chosen = phase10.choose_point(curve)
    control = int(reproduction["checks"]["hybrid-bm25"]["observed"])
    body = {
        "grid": "convex triples (dense, bm25, entity-hop) on tenths, 66 points",
        "objective": "full_support@2048 over the 7,405 dev questions",
        "tie_rule": "gold_recall@2048 sum, then larger dense weight, then larger bm25 weight",
        "curve": curve,
        "chosen": chosen,
        "weights": dict(zip(TRIPLE_COMPONENTS, chosen["weights"], strict=True)),
        "control_supported": control,
        "terminal_state": phase10.dev_gate(chosen, control),
        "dev_lists_digest": reproduction["dev_lists_digest"],
        "code_commit": _git_commit(),
    }
    path = _p10_write_once(P10_FIT_NAME, body, target_dir)
    print(
        f"[INFO] chosen {chosen['weights']}: {chosen['supported']} of 7,405 against the "
        f"control's {control}"
    )
    if body["terminal_state"] is not None:
        _die(f"{body['terminal_state']} recorded in {path}")
    print(f"[OK] fit written -> {path}; commit it before the held-out pass")
    return path


def _p10_fit_is_committed(path: Path) -> bool:
    """The held-out pass runs only on a fit that git tracks, unmodified."""
    import subprocess  # nosec B404: a fixed argv, no shell, no input

    relative = str(path.resolve().relative_to(config.PROJECT_ROOT)).replace("\\", "/")
    try:
        subprocess.run(  # noqa: S603
            ["git", "ls-files", "--error-unmatch", relative],  # noqa: S607
            capture_output=True,
            check=True,
            cwd=config.PROJECT_ROOT,
        )
        status = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain", "--", relative],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=config.PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return status.stdout.strip() == ""


def _p10_pass_provenance(
    components: Mapping[str, Any], fit: Mapping[str, Any], target_dir: Path
) -> dict[str, Any]:
    """The Phase 9 chain with the fields the pass changes: test-10, its vectors, P10 weights.

    `_phase_9_inputs` describes the dev questions, their vector cache and the Phase 9 weights.
    The first pass (2026-09-24) recorded those unchanged; its runs keep them as written.
    """
    base = components["provenance"]
    test_record = json.loads((target_dir / phase10.QUESTIONS_FILENAME).read_text(encoding="utf-8"))[
        "sets"
    ][config.PHASE_10_TEST]
    return {
        **base,
        "question_digest": test_record["question_digest"],
        "mapping_digest": test_record["mapping_digest"],
        "embedding": {
            **base["embedding"],
            "question_cache_key": _p10_json(P10_EMBED_NAME, target_dir)["sets"][
                config.PHASE_10_TEST
            ]["key"],
        },
        "weights": {
            config.PHASE_10_SYSTEMS[1]: components["weights"]["hybrid-bm25"],
            config.PHASE_10_SYSTEMS[2]: dict(fit["weights"]),
        },
    }


def cmd_p10_eval(
    *,
    authorized: bool,
    source_dir: Path = config.PHASE_9_DIR,
    target_dir: Path = config.PHASE_10_DIR,
) -> list[Path]:
    """S6: P10-A, P10-B and P10-C on `test-10`, once, after the author's authorization."""
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    if not authorized:
        _die("the held-out pass opens test-10: it needs --authorized-pass from the author")
    fit = _p10_json(P10_FIT_NAME, target_dir)
    if fit["terminal_state"] is not None:
        _die(f"the fit recorded {fit['terminal_state']}; the held-out pass does not run")
    if not _p10_fit_is_committed(target_dir / P10_FIT_NAME):
        _die("fit.json is not committed unmodified; commit the fit before the held-out pass")
    if (target_dir / P10_MARKER_NAME).exists():
        _die(f"{target_dir / P10_MARKER_NAME} exists: the held-out pass has already started once")
    components = _phase_9_inputs(source_dir)
    corpus_hash = str(components["provenance"]["corpus_unit_set_hash"])
    questions = phase10.load_set(target_dir, config.PHASE_10_TEST, corpus_unit_set_hash=corpus_hash)
    base = components["dense"]
    dense = DenseRetriever(
        base.vectors, base.unit_ids, _p10_query_backend(questions, config.PHASE_10_TEST, target_dir)
    )
    bm25 = components["bm25"]
    stage = ColumnwiseEntityHopStage(components["index"], components["node_weights"])
    systems: dict[str, Any] = {
        "dense": RecordingRetriever(dense),
        "hybrid-bm25": RecordingHybrid(
            dense, bm25, scheme=WEIGHTED, weights=components["weights"]["hybrid-bm25"]
        ),
        "hybrid-bm25-entity-hop": RecordingRetriever(
            TripleFusedRetriever(dense, bm25, stage, weights=fit["weights"])
        ),
    }
    commit = _git_commit()
    host = local_extraction.hardware_block(device="cpu")
    fit_digest = digest_of(json.dumps(fit, sort_keys=True))
    provenance = _p10_pass_provenance(components, fit, target_dir)
    _p10_write_once(
        P10_MARKER_NAME,
        {
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "code_commit": commit,
            "fit_digest": fit_digest,
            "set": config.PHASE_10_TEST,
        },
        target_dir,
    )
    written: list[Path] = []
    for name, system in systems.items():
        print(f"[INFO] pass: {name} over {len(questions)} questions", flush=True)
        records = phase9.measure_system(name, system, questions, components["token_counts"])
        inner = system.inner if isinstance(system, RecordingRetriever) else system
        body = {
            "phase": 10,
            "system": name,
            "label": config.PHASE_10_SYSTEM_LABELS[name],
            "set": config.PHASE_10_TEST,
            "ranking_depth": config.PHASE_9_RANKING_DEPTH,
            "fusion": inner.describe() if hasattr(inner, "describe") else None,
            "metrics": phase9.cohort_metrics(records)[config.PHASE_9_STANDARD],
            "latency": phase9.latency_summary(records),
            "provenance": {
                **provenance,
                "hop_implementation": phase9.COLUMNWISE_HOP,
                "question_set": config.PHASE_10_TEST,
                "fit_digest": fit_digest,
                "code_commit": commit,
                "host": host,
            },
        }
        written.append(phase9.write_run(target_dir, name, records, body=body))
        print(f"[OK] {name} run written -> {written[-1]}", flush=True)
    return written


def cmd_p10_outcome(target_dir: Path = config.PHASE_10_DIR) -> Path:
    """S7 (D6): exact McNemar of P10-C against P10-B and the mechanical label."""
    target_dir = Path(target_dir)
    control, candidate = config.PHASE_10_SYSTEMS[1], config.PHASE_10_SYSTEMS[2]
    outcome = phase10.three_way_outcome(
        phase9.load_outcomes(target_dir, control), phase9.load_outcomes(target_dir, candidate)
    )
    latency = {
        name: _p10_json(f"run-{name}.json", target_dir)["latency"]
        for name in config.PHASE_10_SYSTEMS
    }
    body = {
        **outcome,
        "latency": latency,
        "latency_ratio_candidate_over_control": (
            latency[candidate]["mean_ms"] / latency[control]["mean_ms"]
        ),
        "code_commit": _git_commit(),
    }
    path = _p10_write_once(P10_OUTCOME_NAME, body, target_dir)
    print(
        f"[OK] {outcome['terminal_state']}: wins {outcome['wins']}, losses {outcome['losses']}, "
        f"p = {outcome['exact_two_sided_p']:.4g} -> {path}"
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cer",
        description="Phase 1 pipeline: frozen corpus, baselines and evaluation harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fetch", help="download and freeze the benchmark into data/")
    subparsers.add_parser("build", help="select the subset, split it, build the unified pool")
    subparsers.add_parser("embed", help="compute and cache embeddings and token counts")
    evaluation = subparsers.add_parser(
        "evaluate", help="measure the systems of the experiment at every context budget"
    )
    # Decision D11 of Phase 4: without this, measuring the two expansion systems would
    # write a second test-split result for every Phase 3 system as a side effect.
    evaluation.add_argument(
        "--systems",
        default=None,
        help=(
            "comma-separated systems to measure (default: every Phase 1 and Phase 3 system); "
            f"one or more of {','.join(KNOWN_SYSTEMS)}"
        ),
    )
    subparsers.add_parser("induce", help="build one concept space per dictionary size")
    labelling = subparsers.add_parser(
        "label", help="name the concepts of one space, for the report only"
    )
    # Phase 2 labelled `LABELED_K` because that was the space it expected to matter.
    # Phase 3 selects its own K against dev recall, and when that is a different space
    # it is the one the report has to be able to read - so the stage takes the K rather
    # than assuming it (decision D11).
    labelling.add_argument(
        "--k",
        type=int,
        default=config.LABELED_K,
        help=f"dictionary size to label (default {config.LABELED_K})",
    )
    subparsers.add_parser("select", help="choose the concept space on dev and freeze it")
    subparsers.add_parser(
        "expand", help="sweep the expansion grid on dev and freeze the cell it chose"
    )
    subparsers.add_parser(
        "pilot", help="freeze the Phase 5 navigation pilot's dev questions by rule"
    )
    subparsers.add_parser(
        "nodes", help="normalize the extraction into typed nodes and index them over the pool"
    )
    subparsers.add_parser("navigate", help="run every second hop over the frozen pilot, once")
    subparsers.add_parser(
        "replace-check",
        help="Phase 6 on dev: verify inputs and continuity, reproduce the control; no B figure",
    )
    freezing = subparsers.add_parser(
        "replace-freeze", help="Phase 6 on dev: fit B, check reproducibility, write the freeze"
    )
    # D12: re-measuring the control is admissible only after a failed check, and only with the
    # deviation document that records the mismatch.
    freezing.add_argument(
        "--control-mode", choices=("reused", "re-measured"), default=None, dest="control_mode"
    )
    freezing.add_argument("--deviation", default=None, help="path of the 6.Y deviation document")
    # OI-4: the one way back to dev after test, and only with a deviation document.
    freezing.add_argument(
        "--supersede", action="store_true", help="write a freeze superseding the valid one"
    )
    testing = subparsers.add_parser(
        "replace-test", help="Phase 6 on test, once, under the freeze: the ordered protocol"
    )
    testing.add_argument(
        "--control-mode",
        choices=("reused", "re-measured"),
        default=None,
        dest="control_mode",
        help=(
            "re-measured only on the deviation branch of D14: after a reused test run stopped on "
            "a failed reproduction and a superseding freeze was written"
        ),
    )
    testing.add_argument("--deviation", default=None, help="path of the 6.Y deviation document")
    cheap = subparsers.add_parser(
        "cheap-extract", help="read entities out of every paragraph with a local extractor"
    )
    cheap.add_argument(
        "--extractor",
        choices=config.PHASE_7_EXTRACTORS,
        required=True,
        help="which Phase 7 candidate to run; each writes its own directory",
    )
    # D9: zero on owned hardware, the real rate when the pass runs on rented compute, so
    # the projection carries an arithmetic a reader can check rather than an assumption.
    cheap.add_argument(
        "--hourly-rate-usd",
        type=float,
        default=0.0,
        dest="hourly_rate_usd",
        help="machine rate for the 5M projection; 0.0 on owned hardware (default)",
    )
    cheap.add_argument(
        "--actual-cost-usd",
        type=float,
        default=None,
        help="actual infrastructure charge for this run; required with a nonzero hourly rate",
    )
    evaluation = subparsers.add_parser(
        "cheap-eval", help="fit and measure available local extractors on dev, then select"
    )
    evaluation.add_argument(
        "--test",
        action="store_true",
        help="measure the selected extractor on the held-out split, once",
    )
    # Phase 8, D5 and D9: the encoding runs on a rented GPU and the measurement on the
    # laptop, so they are two stages rather than one; `--test` is the gated mode of each.
    embedding = subparsers.add_parser(
        "strong-embed", help="encode the pool and one split under the Phase 8 Dense model"
    )
    embedding.add_argument(
        "--test",
        action="store_true",
        help="encode the held-out questions, only after the dev stop rule passed",
    )
    embedding.add_argument(
        "--hourly-rate-usd",
        type=float,
        default=0.0,
        dest="hourly_rate_usd",
        help=(
            "machine rate behind the attributable cost and the 5M projection; 0.0 on owned "
            "hardware (default). The total rented-session charge is a different quantity and "
            "is reported in 8.results.md"
        ),
    )
    measurement = subparsers.add_parser(
        "strong-dense", help="fit both Phase 8 hybrids on dev and measure the three systems"
    )
    measurement.add_argument(
        "--test",
        action="store_true",
        help="measure the three systems on the held-out split, once, refitting nothing",
    )
    scale = subparsers.add_parser(
        "scale-corpus",
        help="deviation 8.1: verify the FullWiki archive, reconcile C19, freeze C100/C250/C500",
    )
    scale.add_argument(
        "--archive",
        default=None,
        help="path to the staged official archive (default: data/phase8_1/source/<name>)",
    )
    scale.add_argument(
        "--published-bytes",
        type=int,
        default=config.PHASE_8_1_DECLARED_BYTES,
        help="the byte size the source page publishes; 0 means the page publishes none",
    )
    scale.add_argument(
        "--published-md5",
        default=config.PHASE_8_1_DECLARED_MD5,
        help="the md5 the source page publishes, or 'none' when it publishes none",
    )
    scale.add_argument(
        "--page-checked",
        action="store_true",
        help="the published figures above were read off the live source page, not recollected",
    )
    scale.add_argument(
        "--stop-after",
        choices=("s1", "s2", "s3"),
        default="s3",
        help="stop after this step instead of running S1-S3",
    )
    subparsers.add_parser(
        "scale-repro",
        help="deviation 8.1: reproduce the frozen C19 dev counts from the historical caches",
    )
    scale_run = subparsers.add_parser(
        "scale-run",
        help="deviation 8.1: preflight or measure one Dense encoder over C19-C500",
    )
    scale_run.add_argument(
        "--model",
        choices=config.PHASE_8_1_MODELS,
        required=True,
        help="which of the two frozen Dense encoders to prepare",
    )
    scale_run.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate CPU-side artifacts and gates only; never load a model or require CUDA",
    )
    scale_run.add_argument(
        "--hourly-rate-usd",
        type=float,
        default=0.0,
        dest="hourly_rate_usd",
        help="rented-machine hourly rate used for attributable C500 encoding cost",
    )
    subparsers.add_parser(
        "scale-outcome",
        help="deviation 8.1: classify the complete BGE/Qwen C19-C500 measurements",
    )
    fullwiki_corpus = subparsers.add_parser(
        "fullwiki-corpus",
        help="Phase 9: verify the FullWiki archive and write every paragraph as a unit",
    )
    fullwiki_corpus.add_argument(
        "--archive",
        default=None,
        help="path to the staged official archive (default: data/phase8_1/source/<name>)",
    )
    subparsers.add_parser(
        "fullwiki-questions",
        help="Phase 9: freeze the three cohorts and the title-resolved gold",
    )
    subparsers.add_parser(
        "fullwiki-repro",
        help="Phase 9: reproduce the Phase 7 dev counts through the Phase 9 path, historical pool",
    )
    build = subparsers.add_parser(
        "fullwiki-build",
        help="Phase 9: one build step over the whole FullWiki corpus (measurement host)",
    )
    build.add_argument("--stage", choices=BUILD_STAGES, required=True)
    build.add_argument(
        "--hourly-rate-usd",
        type=float,
        default=0.0,
        dest="hourly_rate_usd",
        help="contracted hourly rate, for the attributable cost of GPU steps",
    )
    probe = subparsers.add_parser(
        "fullwiki-probe",
        help="Phase 9: time the three systems on 100 historical dev questions (rankings only)",
    )
    probe.add_argument(
        "--smoke",
        type=int,
        default=None,
        help="probe the smoke build of the first N units (a code check, not a figure)",
    )
    evaluation_pass = subparsers.add_parser(
        "fullwiki-eval",
        help="Phase 9: the single evaluation pass over the 7,405 questions (author-authorized)",
    )
    evaluation_pass.add_argument(
        "--authorized-pass",
        action="store_true",
        dest="authorized",
        help="the author has explicitly authorized the single evaluation pass",
    )
    subparsers.add_parser(
        "fullwiki-outcome",
        help="Phase 9: the exact McNemar test and the mechanical terminal label",
    )
    subparsers.add_parser(
        "p10-questions",
        help="Phase 10: draw and freeze the held-out sets from HotpotQA train (hard)",
    )
    for command, text in (
        ("p10-embed", "Phase 10: BGE question vectors for probe and test-10 (CPU)"),
        ("p10-repro", "Phase 10: dev component lists and the exact Phase 9 dev counts"),
        ("p10-probe", "Phase 10: Dense on the contamination probe against its dev figure"),
        ("p10-fit", "Phase 10: the 66-point three-way grid on dev, the tie rule and the gate"),
        ("p10-outcome", "Phase 10: exact McNemar of P10-C against P10-B and the label"),
    ):
        subparsers.add_parser(command, help=text)
    p10_eval = subparsers.add_parser(
        "p10-eval", help="Phase 10: the single authorized held-out pass on test-10"
    )
    p10_eval.add_argument("--authorized-pass", action="store_true", dest="authorized")
    build.add_argument(
        "--smoke",
        type=int,
        default=None,
        help="build over the first N units into data/phase9/smoke/ (a code check, not a figure)",
    )
    extraction = subparsers.add_parser(
        "extract", help="read entities and concepts out of every paragraph (spends money)"
    )
    # Decision D4 of Phase 5: the full run refuses without the sample, whose measured
    # estimate is what the author approves before anything else is spent.
    extraction.add_argument(
        "--sample",
        action="store_true",
        help=(
            f"extract only the {config.EXTRACTION_SAMPLE_SIZE} lowest unit ids and report "
            "the estimate"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config.ensure_directories()

    if args.command == "fetch":
        cmd_fetch()
    elif args.command == "build":
        cmd_build()
    elif args.command == "embed":
        cmd_embed()
    elif args.command == "evaluate":
        cmd_evaluate(systems=None if args.systems is None else args.systems.split(","))
    elif args.command == "induce":
        cmd_induce()
    elif args.command == "label":
        cmd_label(k=args.k)
    elif args.command == "select":
        cmd_select()
    elif args.command == "expand":
        cmd_expand()
    elif args.command == "pilot":
        cmd_pilot()
    elif args.command == "extract":
        cmd_extract(sample=args.sample)
    elif args.command == "nodes":
        cmd_nodes()
    elif args.command == "navigate":
        cmd_navigate()
    elif args.command == "replace-check":
        return cmd_replace_check()
    elif args.command == "replace-freeze":
        return cmd_replace_freeze(
            control_mode=args.control_mode, deviation=args.deviation, supersede=args.supersede
        )
    elif args.command == "replace-test":
        return cmd_replace_test(control_mode=args.control_mode, deviation=args.deviation)
    elif args.command == "cheap-extract":
        cmd_cheap_extract(
            extractor_id=args.extractor,
            hourly_rate_usd=args.hourly_rate_usd,
            actual_cost_usd=args.actual_cost_usd,
        )
    elif args.command == "cheap-eval":
        cmd_cheap_eval(test=args.test)
    elif args.command == "strong-embed":
        cmd_strong_embed(hourly_rate_usd=args.hourly_rate_usd, test=args.test)
    elif args.command == "strong-dense":
        cmd_strong_dense(test=args.test)
    elif args.command == "scale-corpus":
        cmd_scale_corpus(
            archive=args.archive,
            published_bytes=args.published_bytes or None,
            published_md5=None if args.published_md5 == "none" else args.published_md5,
            page_checked=args.page_checked,
            stop_after=args.stop_after,
        )
    elif args.command == "scale-repro":
        cmd_scale_repro()
    elif args.command == "scale-run":
        cmd_scale_run(
            model=args.model,
            preflight_only=args.preflight_only,
            hourly_rate_usd=args.hourly_rate_usd,
        )
    elif args.command == "scale-outcome":
        cmd_scale_outcome()
    elif args.command == "fullwiki-corpus":
        cmd_fullwiki_corpus(archive=args.archive)
    elif args.command == "fullwiki-questions":
        cmd_fullwiki_questions()
    elif args.command == "fullwiki-repro":
        cmd_fullwiki_repro()
    elif args.command == "fullwiki-build":
        cmd_fullwiki_build(args.stage, hourly_rate_usd=args.hourly_rate_usd, smoke=args.smoke)
    elif args.command == "fullwiki-probe":
        cmd_fullwiki_probe(smoke=args.smoke)
    elif args.command == "fullwiki-eval":
        cmd_fullwiki_eval(authorized=args.authorized)
    elif args.command == "fullwiki-outcome":
        cmd_fullwiki_outcome()
    elif args.command == "p10-questions":
        cmd_p10_questions()
    elif args.command == "p10-embed":
        cmd_p10_embed()
    elif args.command == "p10-repro":
        cmd_p10_repro()
    elif args.command == "p10-probe":
        cmd_p10_probe()
    elif args.command == "p10-fit":
        cmd_p10_fit()
    elif args.command == "p10-eval":
        cmd_p10_eval(authorized=args.authorized)
    elif args.command == "p10-outcome":
        cmd_p10_outcome()
    return 0


if __name__ == "__main__":
    sys.exit(main())
