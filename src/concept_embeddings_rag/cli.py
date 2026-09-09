"""Command line interface: the four stages of the Phase 1 pipeline.

    fetch     download and freeze the benchmark
    build     select the subset, split it, and build the unified pool
    embed     compute and cache embeddings for the pool
    evaluate  measure dense and BM25 at every context budget
    induce    build one concept space per dictionary size (Phase 2)
    label     name the concepts of one space, for the report only (Phase 2)

Each stage is idempotent and refuses to run if its input is missing, saying which
stage to run first rather than failing somewhere deep inside numpy.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import numpy as np

from concept_embeddings_rag import __version__, config
from concept_embeddings_rag.concepts.coding import (
    ConceptMatrix,
    calibrate_coding_alpha,
    code_corpus,
    load_matrix,
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
    cache_key,
    embed_units,
    unit_set_hash,
)
from concept_embeddings_rag.evaluation.budget import TokenCounter
from concept_embeddings_rag.evaluation.harness import evaluate_retriever
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.bm25 import BM25Retriever
from concept_embeddings_rag.retrieval.dense import DenseRetriever

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
    top_k: int = 100,
) -> list[Path]:
    """Measure dense and BM25 on both splits, at every budget."""
    data_dir = Path(data_dir)
    pool_path = data_dir / POOL_NAME
    tokens_path = data_dir / TOKENS_NAME
    if not pool_path.exists():
        _die("no pool found: run 'build' first")
    if not tokens_path.exists():
        _die("no embeddings or token counts found: run 'embed' first")

    units, questions = load_pool(pool_path)
    _check_pool_against_manifest(units, data_dir / MANIFEST_NAME)
    token_counts = json.loads(tokens_path.read_text(encoding="utf-8"))

    backend = SentenceTransformerBackend()
    cache = EmbeddingCache(Path(cache_dir))
    cached = cache.load(
        _cache_key_for(backend, units), expected_unit_ids=[u.unit_id for u in units]
    )
    if cached is None:
        _die("embeddings are not cached for this pool: run 'embed' first")
    vectors, unit_ids = cached

    retrievers: list[Retriever] = [
        DenseRetriever(vectors=vectors, unit_ids=unit_ids, backend=backend),
        BM25Retriever(units),
    ]
    run_config = {
        "model": backend.name,
        "revision": backend.revision,
        "resolved_revision": backend.resolved_revision(),
        "unit_set_hash": unit_set_hash([u.unit_id for u in units]),
        "seed": config.DEFAULT_SEED,
        "tokenizer": config.TOKENIZER_ID,
        "code_version": __version__,
        "top_k": top_k,
        "n_units": len(units),
    }

    written: list[Path] = []
    for retriever in retrievers:
        for split in ("dev", "test"):
            subset = [q for q in questions if q.split == split]
            if not subset:
                continue
            print(f"[INFO] evaluating {retriever.name} on {split} ({len(subset)} questions)")
            result = evaluate_retriever(
                retriever,
                questions=subset,
                token_counts=token_counts,
                budgets=config.CONTEXT_BUDGETS,
                ks=config.RECALL_AT_K,
                top_k=top_k,
                config=run_config,
                split=split,
            )
            written.append(result.save(Path(results_dir)))

    print(f"[OK] wrote {len(written)} result files to {results_dir}")
    return written


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

    def matrix_for(dictionary: ConceptDictionary) -> ConceptMatrix:
        """The raw `X` of a dictionary, coded only if it is not already on disk."""
        try:
            return load_matrix(dictionary.key, concepts_dir, expected_unit_ids=unit_ids)
        except ConceptArtifactError:
            pass
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
    subparsers.add_parser("evaluate", help="measure dense and BM25 at every context budget")
    subparsers.add_parser("induce", help="build one concept space per dictionary size")
    subparsers.add_parser("label", help="name the concepts of one space, for the report only")
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
        cmd_evaluate()
    elif args.command == "induce":
        cmd_induce()
    elif args.command == "label":
        cmd_label()
    return 0


if __name__ == "__main__":
    sys.exit(main())
