"""Command line interface: the four stages of the Phase 1 pipeline.

    fetch     download and freeze the benchmark
    build     select the subset, split it, and build the unified pool
    embed     compute and cache embeddings for the pool
    evaluate  measure dense and BM25 at every context budget

Each stage is idempotent and refuses to run if its input is missing, saying which
stage to run first rather than failing somewhere deep inside numpy.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from concept_embeddings_rag import __version__, config
from concept_embeddings_rag.corpus import hf_source
from concept_embeddings_rag.corpus.download import sha256_of_file
from concept_embeddings_rag.corpus.manifest import CorpusManifest
from concept_embeddings_rag.corpus.pool import build_pool, load_pool, save_pool
from concept_embeddings_rag.corpus.split import select_subset, split_questions
from concept_embeddings_rag.embeddings.backend import SentenceTransformerBackend
from concept_embeddings_rag.embeddings.cache import EmbeddingCache, embed_units, unit_set_hash
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

    if raw_path.exists():
        print(f"[INFO] corpus already present at {raw_path}")
    else:
        print(f"[INFO] assembling {hf_source.DATASET} ({hf_source.CONFIG}/{hf_source.SPLIT})")
        rows = hf_source.fetch_split(pause=0.5)
        raw_path.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
        print(f"[OK] assembled {len(rows)} questions")
    digest = sha256_of_file(raw_path)

    manifest = CorpusManifest(
        dataset=config.DATASET_NAME,
        source_url=config.CORPUS_SOURCE,
        sha256=digest,
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        seed=seed,
        n_questions=config.N_QUESTIONS,
        split_sizes={"dev": config.N_DEV, "test": config.N_TEST},
    )
    manifest.save(data_dir / MANIFEST_NAME)
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


def _cache_key_for(backend, units) -> str:
    from concept_embeddings_rag.embeddings.cache import cache_key

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
