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

Each stage is idempotent and refuses to run if its input is missing, saying which
stage to run first rather than failing somewhere deep inside numpy.
"""

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from scipy import sparse

from concept_embeddings_rag import __version__, config
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
from concept_embeddings_rag.corpus import hf_source
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
    EmbeddingBackend,
    SentenceTransformerBackend,
)
from concept_embeddings_rag.embeddings.cache import (
    EmbeddingCache,
    build_query_backend,
    cache_key,
    embed_questions,
    embed_units,
    question_cache_key,
    question_set_hash,
    resolved_revision,
    unit_set_hash,
)
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
from concept_embeddings_rag.retrieval.fusion import FusedRetriever

RAW_NAME = "hotpot_raw.json"
MANIFEST_NAME = "manifest.json"
POOL_NAME = "pool.json"
TOKENS_NAME = "token_counts.json"


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


def _cache_key_for(backend: EmbeddingBackend, units: Sequence[IndexingUnit]) -> str:
    return cache_key(
        backend.name,
        backend.revision,
        unit_set_hash([u.unit_id for u in units]),
        normalized=getattr(backend, "normalize", True),
    )


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
