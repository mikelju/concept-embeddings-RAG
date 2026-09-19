"""Phase 8: does Entity Hop still add signal when Dense itself is much stronger?

One variable changes - the Dense retriever - and everything else is inherited: the
frozen pool, the splits, the budget ruler, the Phase 7 GLiNER entity index and the
one-hop, query-blind Entity Hop. Three systems are measured on dev, the fusion weight
of each hybrid is fitted there, and the held-out split is read once, afterwards, and
only if Strong Dense actually beat the inherited Dense reading.

The module holds four things and no more:

1. **The pins** (`Phase8Pins`) - the model, its commit, its width, its query prompt and
   the digests of the inherited entity index, read from `config` rather than typed.
2. **The provenance artifacts** - `embedding.json` for the corpus and dev encoding,
   `embedding-test.json` for the gated test encoding, each written once and never
   rewritten, so the digest chain that authorizes a held-out figure cannot be relinked
   after the fact.
3. **The stop rule** - counted in whole questions, because 487 of 600 is a tie and a tie
   is not "substantially stronger". Its verdict is what gates the test encoding and the
   held-out run; a `stop` closes the phase on its dev figures.
4. **The two measurement passes** - dev (two fits, three systems, diagnostics) and the
   single held-out pass, which refits nothing and reads the schemes and weights dev
   already chose.

No new retriever, no new metric, no new fusion scheme, no freeze/supersede protocol and
no statistical test: the harness, the fusion fitting and the entity hop are the Phase 3,
6 and 7 ones, unmodified. `evaluation/cheap_extraction.py` is the closest precedent and
is deliberately not edited - it verifies Phase 7 invariants and belongs to a closed phase.
"""

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.embeddings.backend import EmbeddingBackend
from concept_embeddings_rag.embeddings.cache import (
    query_prompt_of,
    resolved_revision,
    unit_set_hash,
)
from concept_embeddings_rag.evaluation.budget import fill_context
from concept_embeddings_rag.evaluation.entity_diagnostics import (
    QuestionRecord,
    RecordingHybrid,
    records_of,
)
from concept_embeddings_rag.evaluation.harness import RunResult, evaluate_retriever
from concept_embeddings_rag.evaluation.second_hop import node_weights
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    FusionFit,
    check_dev_only,
    digest_of_payload,
    fit_fusion_weight,
)
from concept_embeddings_rag.nodes.index import NodeIndex, fragmentation

# Phase 7's throughput projection, reused rather than rewritten: the arithmetic and the
# `method` label the spec demands are already there, and already tested.
from concept_embeddings_rag.nodes.local_extraction import projection
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.entity_hop import EntityHopStage
from concept_embeddings_rag.retrieval.fusion import FusedRetriever

EMBEDDING_FILENAME = "embedding.json"
EMBEDDING_TEST_FILENAME = "embedding-test.json"
DEV_FILENAME = "dev.json"
TEST_FILENAME = "test.json"
HELD_OUT_SPLIT = "test"
HEADLINE_BUDGET = config.SELECTION_BUDGET
# The two verdicts the stop rule can return. Named for what each authorizes rather
# than spelled `PASS_*`, which the SAST lint reads as a hardcoded password.
VERDICT_CONTINUE = "pass"
VERDICT_STOP = "stop"

DENSE_SYSTEM, BM25_SYSTEM, ENTITY_SYSTEM = config.PHASE_8_SYSTEMS

__all__ = [
    "DEV_FILENAME",
    "EMBEDDING_FILENAME",
    "EMBEDDING_TEST_FILENAME",
    "HELD_OUT_SPLIT",
    "VERDICT_CONTINUE",
    "VERDICT_STOP",
    "TEST_FILENAME",
    "Phase8Pins",
    "StrongDenseError",
    "WidthCheckedBackend",
    "check_model_pin",
    "check_resolved_revision",
    "load_dev",
    "load_embedding",
    "load_test_embedding",
    "measure_dev",
    "measure_held_out",
    "stop_rule",
    "stop_rule_bar",
    "write_embedding",
    "write_test_embedding",
]


class StrongDenseError(ValueError):
    """A Phase 8 pass was handed inputs it may not measure, or an artifact it may not write."""


@dataclass(frozen=True)
class Phase8Pins:
    """Everything this phase fixed before its first number existed.

    A dataclass rather than direct `config` reads, for the reason `InputPins` is one in
    Phase 6: the checks are then exercisable against a toy world without monkeypatching
    the module every real run depends on, and the real run uses `from_config` and
    nothing else.
    """

    model: str
    revision: str
    dim: int
    query_prompt: str
    unit_set_hash: str
    index_digest: str
    extraction_digest: str
    dense_dev_share: float

    @classmethod
    def from_config(cls) -> "Phase8Pins":
        return cls(
            model=config.PHASE_8_DENSE_MODEL,
            revision=config.PHASE_8_DENSE_REVISION,
            dim=config.PHASE_8_DENSE_DIM,
            query_prompt=config.PHASE_8_QUERY_PROMPT,
            unit_set_hash=config.PHASE_1_UNIT_SET_HASH,
            index_digest=config.PHASE_8_GLINER_INDEX_DIGEST,
            extraction_digest=config.PHASE_8_GLINER_EXTRACTION_DIGEST,
            dense_dev_share=config.PHASE_8_INHERITED_DEV_FULL_SUPPORT["dense"],
        )


def _pins(pins: Phase8Pins | None) -> Phase8Pins:
    return pins if pins is not None else Phase8Pins.from_config()


# --- Provenance checks that need no measurement (D11) ------------------------------------
#
# They run inside the stage, before the first model load and before a single vector
# exists. A provenance check that only happens while writing the report is a check that
# cannot stop a bad run.


def check_model_pin(revision: str, model: str = "") -> None:
    """Refuse anything but a 40-character commit sha: a branch is not a pin."""
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise StrongDenseError(
            f"the Dense model {model or 'revision'} is pinned to {revision!r}, which is not a "
            "40-character commit sha; a moving revision silently redefines every vector"
        )


def check_resolved_revision(resolved: str, pinned: str) -> None:
    """The pin is only worth the ability to check that it was honoured."""
    if resolved != pinned:
        raise StrongDenseError(
            f"the Hub served revision {resolved!r} but this phase is pinned to {pinned!r}; "
            "no vector produced under another commit may be recorded as this one"
        )


class WidthCheckedBackend:
    """A backend that refuses a vector block of any width but the declared one.

    Nothing in this project checks embedding width: `EMBEDDING_DIM` is only a default and
    `DenseRetriever` takes whatever vectors it is given, which is exactly why 1024-dim
    vectors need no change to retrieval or fusion. But Qwen3-Embedding supports MRL
    truncation and the spec pins full width with none, so a silently narrower block is a
    real failure mode here. The guard sits between the model and the cache because
    `embed_units` writes the artifact itself: a check downstream of it would find the
    wrong vectors already on disk.
    """

    def __init__(self, inner: EmbeddingBackend, dim: int) -> None:
        self.inner = inner
        self.name = inner.name
        self.revision = inner.revision
        self.dim = dim
        self.normalize = bool(getattr(inner, "normalize", True))

    @property
    def resolved_query_prompt(self) -> str:
        return query_prompt_of(self.inner)

    def resolved_revision(self) -> str:
        return resolved_revision(self.inner)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.inner.encode(texts)
        if vectors.ndim != 2 or int(vectors.shape[1]) != self.dim:
            raise StrongDenseError(
                f"{self.name} produced vectors of shape {tuple(vectors.shape)}, and this phase "
                f"is pinned to {self.dim} dimensions with no MRL truncation; nothing is cached"
            )
        return vectors


# --- The embedding artifacts (D5, D8) -----------------------------------------------------


def _verified(path: Path, stage: str) -> dict[str, Any]:
    """Read a Phase 8 artifact back, refusing one that does not match its own digest."""
    if not path.exists():
        raise StrongDenseError(f"{path} does not exist: run '{stage}' first")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("digest") != digest_of_payload(payload):
        raise StrongDenseError(f"{path} does not match its digest; it was edited")
    return payload


def load_embedding(directory: Path | str) -> dict[str, Any]:
    return _verified(Path(directory) / EMBEDDING_FILENAME, "strong-embed")


def load_test_embedding(directory: Path | str) -> dict[str, Any]:
    return _verified(Path(directory) / EMBEDDING_TEST_FILENAME, "strong-embed --test")


def load_dev(directory: Path | str) -> dict[str, Any]:
    return _verified(Path(directory) / DEV_FILENAME, "strong-dense")


def _write(target: Path, payload: dict[str, Any]) -> Path:
    payload["digest"] = digest_of_payload(payload)
    return write_text_atomic(target, json.dumps(payload, indent=2, sort_keys=True))


def write_embedding(
    directory: Path | str,
    *,
    resolved_revision: str,
    weights_sha256: str,
    max_seq_length: int | None,
    query_prompt: str,
    corpus_cache_key: str,
    question_cache_key: str,
    unit_ids: Sequence[str],
    n_questions: int,
    dim: int,
    hardware: Mapping[str, Any],
    corpus_seconds: float,
    question_seconds: float,
    bytes_on_disk: int,
    hourly_rate_usd: float,
    pins: Phase8Pins | None = None,
) -> Path:
    """Write `embedding.json`: which model, which weights and which caches, as measured.

    Written once and never rewritten. `dev.json` anchors the digest chain on this file,
    so appending the test-question key to it later would break that link, and preserving
    a stale digest inside a rewritten file is the quiet provenance drift this project
    refuses. The gated test encoding gets its own small artifact instead (D5).
    """
    settings = _pins(pins)
    target = Path(directory) / EMBEDDING_FILENAME
    if target.exists():
        raise StrongDenseError(
            f"{target} already records this phase's encoding; it is written once and the "
            "dev fit is anchored on its digest"
        )
    check_model_pin(settings.revision, settings.model)
    check_resolved_revision(resolved_revision, settings.revision)
    if query_prompt != settings.query_prompt:
        raise StrongDenseError(
            "the resolved query prompt is not the declared one; the run should have aborted "
            "before encoding anything"
        )
    if dim != settings.dim:
        raise StrongDenseError(f"vectors are {dim}-dimensional and this phase pins {settings.dim}")
    corpus_hash = unit_set_hash(unit_ids)
    if corpus_hash != settings.unit_set_hash:
        raise StrongDenseError(
            f"the pool hashes to {corpus_hash} and this phase measures {settings.unit_set_hash}"
        )
    if corpus_cache_key == question_cache_key:
        raise StrongDenseError("the corpus and question caches cannot share one key")
    if corpus_seconds <= 0.0:
        raise StrongDenseError("a throughput needs a positive wall clock, and this one has none")

    per_second = len(unit_ids) / corpus_seconds
    payload: dict[str, Any] = {
        "phase": 8,
        "split": DEV_SPLIT,
        "model": settings.model,
        "revision": settings.revision,
        "resolved_revision": resolved_revision,
        "weights_file": config.PHASE_8_WEIGHTS_FILE,
        "weights_sha256": weights_sha256,
        "dim": dim,
        "normalized": bool(config.NORMALIZE_EMBEDDINGS),
        "max_seq_length": max_seq_length,
        "query_prompt": query_prompt,
        "corpus_cache_key": corpus_cache_key,
        "question_cache_key": question_cache_key,
        "unit_set_hash": corpus_hash,
        "n_units": len(unit_ids),
        "n_questions": n_questions,
        "budget_tokenizer": config.BUDGET_TOKENIZER_ID,
        "budget_tokenizer_revision": config.BUDGET_TOKENIZER_REVISION,
        "hardware": dict(hardware),
        "embedding_seconds": corpus_seconds,
        "question_seconds": question_seconds,
        "paragraphs_per_second": per_second,
        "hourly_rate_usd": hourly_rate_usd,
        "embedding_usd": hourly_rate_usd * corpus_seconds / 3600.0,
        "bytes": bytes_on_disk,
        "projection_5m": projection(len(unit_ids), per_second, hourly_rate_usd=hourly_rate_usd),
        "cost_basis": (
            "embedding_usd is the charge attributable to this encoding pass, "
            "hourly_rate_usd * embedding_seconds / 3600. The total rented-session charge "
            "covers image pull, model download, upload and idle time as well; it is known "
            "from billing afterwards and is reported in 8.results.md, never written here"
        ),
        "trust_remote_code": False,
    }
    return _write(target, payload)


def write_test_embedding(
    directory: Path | str,
    *,
    question_cache_key: str,
    n_questions: int,
    hardware: Mapping[str, Any],
    question_seconds: float,
    hourly_rate_usd: float,
    pins: Phase8Pins | None = None,
) -> Path:
    """Write `embedding-test.json`, the encoding the dev gate authorized.

    It carries both parent digests, so the artifact itself records that the held-out
    questions were encoded only after Strong Dense had proved itself on dev.
    """
    settings = _pins(pins)
    directory = Path(directory)
    target = directory / EMBEDDING_TEST_FILENAME
    if target.exists():
        raise StrongDenseError(
            f"{target} already records the held-out encoding; it is written once"
        )
    embedding = load_embedding(directory)
    dev = load_dev(directory)
    check_gate(dev)
    if question_cache_key == embedding["question_cache_key"]:
        raise StrongDenseError(
            "the held-out questions hash to the dev question cache key; dev and test must "
            "never be able to land in the same artifact"
        )
    payload: dict[str, Any] = {
        "phase": 8,
        "split": HELD_OUT_SPLIT,
        "model": settings.model,
        "revision": settings.revision,
        "resolved_revision": embedding["resolved_revision"],
        "weights_sha256": embedding["weights_sha256"],
        "dim": embedding["dim"],
        "query_prompt": embedding["query_prompt"],
        "question_cache_key": question_cache_key,
        "n_questions": n_questions,
        "hardware": dict(hardware),
        "question_seconds": question_seconds,
        "hourly_rate_usd": hourly_rate_usd,
        "question_usd": hourly_rate_usd * question_seconds / 3600.0,
        "embedding_digest": embedding["digest"],
        "dev_digest": dev["digest"],
        "authorized_by": (
            "the dev stop rule returned 'pass', so the held-out questions were encoded; a "
            "'stop' verdict leaves them neither encoded nor measured"
        ),
    }
    return _write(target, payload)


# --- The stop rule (Decision 4, D10) ------------------------------------------------------


def stop_rule_bar(n_questions: int, baseline_share: float) -> tuple[int, int]:
    """The inherited Dense dev reading in whole questions, and the strict improvement on it.

    Over the 600 dev questions the inherited share is 487 of them. A tie is not
    "substantially stronger", so the rule passes only at 488 or more. Comparing rounded
    shares instead would make the verdict depend on float formatting - the same reasoning
    that made Phase 7 apply its retention bar as a count rather than as `0.8617`.
    """
    baseline = round(baseline_share * n_questions)
    return baseline, baseline + 1


def stop_rule(full_support: float, n_questions: int, baseline_share: float) -> dict[str, Any]:
    """The verdict, with every number it was read from beside it."""
    baseline_count, required = stop_rule_bar(n_questions, baseline_share)
    count = round(full_support * n_questions)
    return {
        "metric": "full_support",
        "budget": HEADLINE_BUDGET,
        "count": count,
        "n_questions": n_questions,
        "share": full_support,
        "baseline_count": baseline_count,
        "baseline_share": baseline_share,
        "required_count": required,
        "verdict": VERDICT_CONTINUE if count >= required else VERDICT_STOP,
        "rule": (
            "Strong Dense alone must support strictly more dev questions than the inherited "
            "Dense reading; a tie is not substantially stronger"
        ),
    }


def check_gate(dev: Mapping[str, Any]) -> None:
    """The held-out split is neither encoded nor measured unless dev said `pass`."""
    verdict = dev.get("stop_rule", {}).get("verdict")
    if verdict != VERDICT_CONTINUE:
        raise StrongDenseError(
            f"the dev stop rule returned {verdict!r}: Strong Dense is not stronger than the "
            "inherited Dense reading on this corpus, so the phase closes on its dev figures "
            "and the held-out split stays unopened"
        )


# --- Diagnostics, all from machinery that already exists (D7) -----------------------------


def hop_behaviour(
    recorded: Sequence[QuestionRecord],
    token_counts: Mapping[str, int],
    index: NodeIndex,
) -> dict[str, Any]:
    """What the hop actually did under the new Dense: `P(q)`, empty `p1`s, new context."""
    expansions = [row.expansion for row in recorded if row.expansion is not None]
    if not expansions:
        raise StrongDenseError("the recording holds no entity expansion to diagnose")
    positives = [expansion.positives for expansion in expansions]
    introduced = []
    for row in recorded:
        context = fill_context(
            [unit_id for unit_id, _score in row.fused], token_counts, config.SELECTION_BUDGET
        )
        supplied = {unit_id for unit_id, _score in row.dense}
        introduced.append(len(set(context) - supplied))
    return {
        "positive_candidates_mean": statistics.fmean(positives),
        "positive_candidates_median": statistics.median(positives),
        "zero_candidate_share": sum(value == 0 for value in positives) / len(positives),
        "p1s_without_entity": sum(value.p1_entity_nodes == 0 for value in expansions),
        "introduced_context_units": sum(introduced),
        "introduced_context_units_mean": statistics.fmean(introduced),
        "introduced_context_budget": config.SELECTION_BUDGET,
        "introduced_context_definition": "fused context units outside Dense top_k",
        "entity_index": fragmentation(index)["entity"],
    }


def p1_changes(
    recorded: Sequence[QuestionRecord],
    questions: Sequence[Question],
    baseline: Retriever | None,
) -> dict[str, Any]:
    """How often `p1` moved when the Dense model changed - the cheapest available diagnosis.

    The hop is conditioned on Dense's first paragraph, so this is the mechanism by which a
    stronger retriever changes what the hop contributes. Descriptive: it fits nothing, and
    the Strong Dense side is read off the lists the measurement already recorded rather
    than re-retrieved.
    """
    if baseline is None:
        return {
            "measured": False,
            "reason": "no BGE-small dense retriever was supplied, so nothing is compared",
        }
    changed = 0
    compared = 0
    for row, question in zip(recorded, questions, strict=True):
        if not row.dense:
            continue
        compared += 1
        first = baseline.retrieve(question.question, 1)
        if not first or first[0][0] != row.dense[0][0]:
            changed += 1
    return {
        "measured": True,
        "baseline": baseline.name,
        "compared": compared,
        "changed": changed,
        "share": changed / compared if compared else 0.0,
        "definition": "questions whose top-1 Dense paragraph differs between the two models",
    }


# --- The two measurement passes (D6) ------------------------------------------------------


def _check_index(index: NodeIndex, unit_ids: Sequence[str], pins: Phase8Pins) -> None:
    """The inherited entity index, and no other: Phase 8 changes the Dense retriever only."""
    if index.digest != pins.index_digest:
        raise StrongDenseError(
            f"the node index digest is {index.digest}, and this phase is pinned to "
            f"{pins.index_digest}; the Phase 7 GLiNER index is inherited, never rebuilt"
        )
    if index.extraction_digest != pins.extraction_digest:
        raise StrongDenseError(
            "the node index was built from another extraction than the pinned GLiNER one"
        )
    if tuple(index.unit_ids) != tuple(unit_ids):
        raise StrongDenseError("the node index was built over another pool")


def _check_inputs(
    unit_ids: Sequence[str],
    run_config: Mapping[str, Any],
    embedding: Mapping[str, Any],
    split: str,
) -> None:
    if run_config.get("unit_set_hash") != unit_set_hash(unit_ids):
        raise StrongDenseError("the run configuration belongs to another pool")
    if embedding.get("corpus_cache_key") != run_config.get("corpus_cache_key"):
        raise StrongDenseError(
            "these vectors are not the ones embedding.json recorded; a figure measured from "
            "another cache cannot be attributed to this phase's encoding"
        )
    key = "question_cache_key"
    if run_config.get(key) is None:
        raise StrongDenseError("the run configuration names no question cache")
    if split == DEV_SPLIT and embedding.get(key) != run_config.get(key):
        raise StrongDenseError("these dev questions are not the ones embedding.json recorded")


def _provenance(
    run_config: Mapping[str, Any], index: NodeIndex, pins: Phase8Pins, embedding_digest: str
) -> dict[str, Any]:
    return {
        **run_config,
        "phase": 8,
        "embedding_digest": embedding_digest,
        "node_index_digest": index.digest,
        "extraction_digest": index.extraction_digest,
        "normalization_version": index.normalization_version,
        "read_depth": config.PILOT_READ_DEPTH,
        "entity_types": list(config.ENTITY_HOP_TYPES),
        "dense_model": pins.model,
        "dense_revision": pins.revision,
    }


def _headline(result: RunResult) -> float:
    return float(result.metrics[f"budget_{HEADLINE_BUDGET}"]["full_support"])


def _system_block(
    result: RunResult, run_path: Path, n_questions: int, fit: FusionFit | None = None
) -> dict[str, Any]:
    """One system's reading, individually traceable to the run file that produced it."""
    share = _headline(result)
    block: dict[str, Any] = {
        "system": result.system,
        "config": result.config,
        "metrics": result.metrics,
        "cost": result.cost,
        "full_support": share,
        "supported_questions": round(share * n_questions),
        "run_file": run_path.name,
        "run_digest": digest_of_payload(asdict(result)),
    }
    if fit is not None:
        block["fit"] = {**asdict(fit), "scheme": fit.winning_scheme, "weights": fit.weights}
    return block


def _measure(
    retriever: Retriever,
    *,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    provenance: Mapping[str, Any],
    split: str,
    directory: Path,
    n_questions: int,
    fit: FusionFit | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = evaluate_retriever(
        retriever,
        questions=questions,
        token_counts=token_counts,
        budgets=config.CONTEXT_BUDGETS,
        ks=config.RECALL_AT_K,
        top_k=int(provenance["top_k"]),
        config={**provenance, **(extra or {})},
        split=split,
    )
    return _system_block(result, result.save(directory), n_questions, fit)


def measure_dev(
    directory: Path | str,
    *,
    unit_ids: Sequence[str],
    dense: Retriever,
    bm25: Retriever,
    index: NodeIndex,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    run_config: Mapping[str, Any],
    baseline_dense: Retriever | None = None,
    pins: Phase8Pins | None = None,
) -> Path:
    """Fit both hybrids on dev, measure all three systems there, and apply the stop rule."""
    check_dev_only(questions)
    settings = _pins(pins)
    directory = Path(directory)
    target = directory / DEV_FILENAME
    if target.exists():
        raise StrongDenseError(f"{target} already holds a dev measurement; not overwriting it")
    if (directory / TEST_FILENAME).exists():
        raise StrongDenseError("a held-out run already exists; the dev fit is closed")
    embedding = load_embedding(directory)
    _check_inputs(unit_ids, run_config, embedding, DEV_SPLIT)
    _check_index(index, unit_ids, settings)

    provenance = _provenance(run_config, index, settings, embedding["digest"])
    weights = node_weights(index)
    n_questions = len(questions)

    bm25_fit = fit_fusion_weight(
        [dense, bm25],
        questions=questions,
        token_counts=token_counts,
        base_config=provenance,
    )
    # The two fits are independent and run over the same grid with the same objective:
    # giving either a shorter grid or a weaker reference is the easiest way to
    # manufacture a headline (Phase 3, decision D9).
    entity_fit = fit_fusion_weight(
        [dense, EntityHopStage(index, weights)],
        questions=questions,
        token_counts=token_counts,
        base_config=provenance,
    )

    dense_block = _measure(
        dense,
        questions=questions,
        token_counts=token_counts,
        provenance=provenance,
        split=DEV_SPLIT,
        directory=directory,
        n_questions=n_questions,
    )
    bm25_hybrid = FusedRetriever(
        [dense, bm25], scheme=bm25_fit.winning_scheme, weights=bm25_fit.weights
    )
    bm25_block = _measure(
        bm25_hybrid,
        questions=questions,
        token_counts=token_counts,
        provenance=provenance,
        split=DEV_SPLIT,
        directory=directory,
        n_questions=n_questions,
        fit=bm25_fit,
        extra=bm25_hybrid.describe(),
    )
    # A fresh stage prevents the fit's memo from flattering measured query latency.
    entity_hybrid = RecordingHybrid(
        dense,
        EntityHopStage(index, weights),
        scheme=entity_fit.winning_scheme,
        weights=entity_fit.weights,
    )
    entity_block = _measure(
        entity_hybrid,
        questions=questions,
        token_counts=token_counts,
        provenance=provenance,
        split=DEV_SPLIT,
        directory=directory,
        n_questions=n_questions,
        fit=entity_fit,
        extra=entity_hybrid.describe(),
    )

    recorded = records_of(entity_hybrid, questions)
    payload: dict[str, Any] = {
        "phase": 8,
        "split": DEV_SPLIT,
        "n_questions": n_questions,
        "model": settings.model,
        "revision": settings.revision,
        "embedding_digest": embedding["digest"],
        "node_index_digest": index.digest,
        "extraction_digest": index.extraction_digest,
        "systems": {
            DENSE_SYSTEM: dense_block,
            BM25_SYSTEM: bm25_block,
            ENTITY_SYSTEM: entity_block,
        },
        "stop_rule": stop_rule(
            dense_block["full_support"], n_questions, settings.dense_dev_share
        ),
        "hop": hop_behaviour(recorded, token_counts, index),
        "p1_change": p1_changes(recorded, questions, baseline_dense),
        "inherited_dev_full_support": dict(config.PHASE_8_INHERITED_DEV_FULL_SUPPORT),
        "inherited_dev_files": dict(config.PHASE_8_INHERITED_DEV_FILES),
    }
    return _write(target, payload)


def surviving_gain(
    systems: Mapping[str, Mapping[str, Any]], n_questions: int
) -> dict[str, Any]:
    """How much of the inherited Entity Hop gain survived, in whole questions and in points.

    Both readings are reported the way Phase 7 reported its own: `1,231 - 1,155 = 76 of 91`
    is easier to argue about than `+0.0543`. The inherited counts are over the 1,400
    held-out questions and are labelled with that denominator, because a measured pass over
    a different number of questions must not be silently rescaled.
    """
    inherited = config.PHASE_8_INHERITED_HELD_OUT_FULL_SUPPORT
    measured_gain = (
        systems[ENTITY_SYSTEM]["full_support"] - systems[DENSE_SYSTEM]["full_support"]
    )
    inherited_gain = inherited["hybrid-entity-hop-gliner"] - inherited["dense"]
    return {
        "budget": HEADLINE_BUDGET,
        "measured": {
            name: {
                "full_support": systems[name]["full_support"],
                "supported_questions": systems[name]["supported_questions"],
            }
            for name in config.PHASE_8_SYSTEMS
        },
        "measured_n_questions": n_questions,
        "inherited": {
            name: {
                "full_support": share,
                "supported_questions": round(share * config.N_TEST),
            }
            for name, share in inherited.items()
        },
        "inherited_n_questions": config.N_TEST,
        "inherited_files": dict(config.PHASE_8_INHERITED_HELD_OUT_FILES),
        "entity_gain_points": measured_gain,
        "entity_gain_questions": (
            systems[ENTITY_SYSTEM]["supported_questions"]
            - systems[DENSE_SYSTEM]["supported_questions"]
        ),
        "bm25_gain_points": (
            systems[BM25_SYSTEM]["full_support"] - systems[DENSE_SYSTEM]["full_support"]
        ),
        "bm25_gain_questions": (
            systems[BM25_SYSTEM]["supported_questions"]
            - systems[DENSE_SYSTEM]["supported_questions"]
        ),
        "inherited_entity_gain_points": inherited_gain,
        "inherited_entity_gain_questions": (
            round(inherited["hybrid-entity-hop-gliner"] * config.N_TEST)
            - round(inherited["dense"] * config.N_TEST)
        ),
        "surviving_share": (measured_gain / inherited_gain) if inherited_gain else None,
        "comparison_note": (
            "the inherited row is the GLiNER Entity Hop, because this phase inherits the "
            "GLiNER index; the Claude row is historical context only"
        ),
    }


def measure_held_out(
    directory: Path | str,
    *,
    unit_ids: Sequence[str],
    dense: Retriever,
    bm25: Retriever,
    index: NodeIndex,
    questions: Sequence[Question],
    token_counts: Mapping[str, int],
    run_config: Mapping[str, Any],
    baseline_dense: Retriever | None = None,
    pins: Phase8Pins | None = None,
) -> Path:
    """Measure the three systems on the held-out split once, at the dev-chosen schemes.

    Nothing is refitted and nothing is chosen here: the schemes and weights come out of
    `dev.json`, and the artifact records the digests of the encoding, the entity index,
    the dev fit and the gated test encoding, so a test figure cannot be detached from
    what authorized it.
    """
    settings = _pins(pins)
    directory = Path(directory)
    target = directory / TEST_FILENAME
    if target.exists():
        raise StrongDenseError(f"{target} already holds the held-out run; it is read once")
    intruders = sorted(
        {question.split for question in questions if question.split != HELD_OUT_SPLIT}
    )
    if intruders or not questions:
        raise StrongDenseError(
            f"the held-out run was handed questions of split(s) {intruders or ['none']}; it "
            "measures the held-out split and nothing else"
        )
    embedding = load_embedding(directory)
    dev = load_dev(directory)
    check_gate(dev)
    test_embedding = load_test_embedding(directory)
    if test_embedding["dev_digest"] != dev["digest"]:
        raise StrongDenseError("the held-out encoding was authorized by another dev reading")
    if test_embedding["embedding_digest"] != embedding["digest"]:
        raise StrongDenseError("the held-out encoding belongs to another corpus encoding")
    if test_embedding["question_cache_key"] != run_config.get("question_cache_key"):
        raise StrongDenseError(
            "these held-out questions are not the ones embedding-test.json recorded"
        )
    _check_inputs(unit_ids, run_config, embedding, HELD_OUT_SPLIT)
    _check_index(index, unit_ids, settings)
    if dev["node_index_digest"] != index.digest:
        raise StrongDenseError("the node index on disk is not the one dev was measured on")

    provenance = {
        **_provenance(run_config, index, settings, embedding["digest"]),
        "dev_digest": dev["digest"],
        "embedding_test_digest": test_embedding["digest"],
        "fitted_on": DEV_SPLIT,
    }
    weights = node_weights(index)
    n_questions = len(questions)
    systems = {
        DENSE_SYSTEM: _measure(
            dense,
            questions=questions,
            token_counts=token_counts,
            provenance=provenance,
            split=HELD_OUT_SPLIT,
            directory=directory,
            n_questions=n_questions,
        )
    }
    entity_recording: RecordingHybrid | None = None
    for name, second in ((BM25_SYSTEM, bm25), (ENTITY_SYSTEM, EntityHopStage(index, weights))):
        fitted = dev["systems"][name]["fit"]
        scheme, fitted_weights = fitted["scheme"], fitted["weights"]
        if name == ENTITY_SYSTEM:
            hybrid: Any = RecordingHybrid(dense, second, scheme=scheme, weights=fitted_weights)
            entity_recording = hybrid
        else:
            hybrid = FusedRetriever([dense, second], scheme=scheme, weights=fitted_weights)
        systems[name] = _measure(
            hybrid,
            questions=questions,
            token_counts=token_counts,
            provenance=provenance,
            split=HELD_OUT_SPLIT,
            directory=directory,
            n_questions=n_questions,
            extra=hybrid.describe(),
        )
        systems[name]["fit"] = {"scheme": scheme, "weights": fitted_weights, "fitted_on": DEV_SPLIT}

    recorded = records_of(entity_recording, questions) if entity_recording is not None else []
    payload: dict[str, Any] = {
        "phase": 8,
        "split": HELD_OUT_SPLIT,
        "n_questions": n_questions,
        "model": settings.model,
        "revision": settings.revision,
        "embedding_digest": embedding["digest"],
        "embedding_test_digest": test_embedding["digest"],
        "dev_digest": dev["digest"],
        "node_index_digest": index.digest,
        "systems": systems,
        "surviving_gain": surviving_gain(systems, n_questions),
        "hop": hop_behaviour(recorded, token_counts, index),
        "p1_change": p1_changes(recorded, questions, baseline_dense),
        "inherited_test_full_support": dict(config.PHASE_8_INHERITED_HELD_OUT_FULL_SUPPORT),
        "inherited_test_files": dict(config.PHASE_8_INHERITED_HELD_OUT_FILES),
    }
    return _write(target, payload)
