"""Central configuration: paths, corpus scale, evaluation budgets and model ids.

Every constant that decides what an experiment measures lives here, so that a
result can always be traced back to the configuration that produced it. Nothing
in this project may hardcode a budget, a seed or a model name inline.
"""

from pathlib import Path
from typing import Final

# --- Paths -----------------------------------------------------------------
# Resolved from this file, never from the working directory: the CLI must behave
# identically wherever it is invoked from.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
CACHE_DIR: Final[Path] = PROJECT_ROOT / "data" / "cache"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "data" / "results"

# --- Corpus ----------------------------------------------------------------
DATASET_NAME: Final[str] = "hotpotqa-distractor-dev"
# The authors' original host stopped responding (see deviation 1.1). Kept for
# provenance: it is where this dataset was originally published.
HOTPOTQA_DEV_DISTRACTOR_URL: Final[str] = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)
# Source actually in use: the canonical HuggingFace dataset, paged as JSON.
CORPUS_SOURCE: Final[str] = "huggingface:hotpotqa/hotpot_qa[distractor/validation]"

N_QUESTIONS: Final[int] = 2000
N_DEV: Final[int] = 600
N_TEST: Final[int] = 1400

# --- Reproducibility -------------------------------------------------------
# Subset selection uses this seed. The dev/test split does NOT: it derives from a
# hash of the question id, because RNG streams can change across library versions
# while a hash cannot.
DEFAULT_SEED: Final[int] = 42

# --- Embeddings ------------------------------------------------------------
EMBEDDING_MODEL: Final[str] = "BAAI/bge-small-en-v1.5"
# Pinned to a commit, never to a branch. A moving revision silently redefines every
# vector in the project and every number derived from them, while every artifact on
# disk keeps claiming the same configuration. The resolved commit is recorded again
# at embedding time, so a run can be checked against the pin rather than trusting it.
EMBEDDING_REVISION: Final[str] = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EMBEDDING_DIM: Final[int] = 384
NORMALIZE_EMBEDDINGS: Final[bool] = True

# Tokenizer used to measure context budgets. Its absolute values do not matter;
# what matters is that every system is measured with the same ruler.
#
# Phase 8, restriction R1: this used to be `TOKENIZER_ID = EMBEDDING_MODEL`, which made
# the ruler follow the Dense model. Repointing the embedding model would then have
# re-tokenized the corpus with a different tokenizer and rewritten
# `data/token_counts.json`, the single file that defines the fixed context budget behind
# every Phase 1-7 result - so no new number would have been comparable to any inherited
# one, and the inherited ones would have stopped being reproducible from the repository.
# The budget tokenizer is therefore its own configuration item, declared at exactly the
# values the alias resolved to in Phases 1-7: a change of meaning, not of number.
BUDGET_TOKENIZER_ID: Final[str] = "BAAI/bge-small-en-v1.5"
BUDGET_TOKENIZER_REVISION: Final[str] = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
# The name four provenance sites already record. Kept so every artifact written since
# Phase 1 keeps meaning what it meant, and now defined by the ruler rather than by the
# embedding model.
TOKENIZER_ID: Final[str] = BUDGET_TOKENIZER_ID

# --- Evaluation ------------------------------------------------------------
CONTEXT_BUDGETS: Final[tuple[int, ...]] = (512, 1024, 2048, 4096)
RECALL_AT_K: Final[tuple[int, ...]] = (2, 5, 10)

# --- Concept space (Phase 2) -----------------------------------------------
CONCEPTS_DIR: Final[Path] = DATA_DIR / "concepts"

# The sweep. Phase 2 produces all four spaces and deliberately chooses none of
# them: the choice belongs to Phase 3, made against dev recall.
CONCEPT_K_SWEEP: Final[tuple[int, ...]] = (512, 1024, 2048, 4096)
CONCEPT_SEED: Final[int] = DEFAULT_SEED

# Sparsity is an outcome of the L1 penalty, not a setting, so the target is a band
# and `alpha` is searched until the achieved mean lands inside it.
SPARSITY_TARGET_BAND: Final[tuple[int, int]] = (8, 16)
# Below this, the fit is one-hot and is not shipped: that regime destroys
# co-activation, unit-to-unit similarity in concept space, and expansion itself.
SPARSITY_FLOOR: Final[int] = 3

# The L1 penalty the induction itself runs under. It is not the penalty the corpus
# is finally coded with: that one is calibrated per K against the target band above,
# because achieved sparsity moves with K. This value shapes which atoms are learned,
# so it travels in the dictionary key - see decision D10 of the phase plan.
INDUCTION_ALPHA: Final[float] = 0.05

# Atoms are unit-norm, so cosine between them is a dot product and this threshold
# means the same thing for every pair.
MERGE_COSINE_THRESHOLD: Final[float] = 0.90
# Merging more than this share of a dictionary is a finding about K, not a detail.
MAX_MERGED_FRACTION: Final[float] = 0.20

# Coordinate descent works against a K x K Gram matrix, and `transform` returns a
# dense block, so both stages are batched.
CONCEPT_BATCH_SIZE: Final[int] = 512
CONCEPT_MAX_ITER: Final[int] = 30
CODING_BATCH_SIZE: Final[int] = 2048

# The whole sweep is meant to fit in an afternoon. A fit that costs a night is a
# design error, exactly as embedding was in Phase 1, so the sweep stops instead.
CONCEPT_SWEEP_BUDGET_SECONDS: Final[int] = 7200

# --- Space diagnostics (Phase 2, HU-5) --------------------------------------
# A concept activated by fewer units than this is dead: it is a dimension the
# corpus never uses, and it is reported rather than quietly carried.
DEAD_ATOM_MIN_UNITS: Final[int] = 1
# How much evidence a diagnostic shows per concept. Reporting only - these decide
# what a human reads, never what the space contains.
TOP_UNITS_PER_CONCEPT: Final[int] = 10
TOP_COACTIVATIONS: Final[int] = 10

# --- Dictionary quality (Phase 2, HU-7) -------------------------------------
# How many activating units a concept is scored over. Deliberately larger than the
# ten shown as evidence above: those are what a human reads, these are a sample a
# mean is taken from, and ten pairs rank concepts by noise.
COHERENCE_TOP_UNITS: Final[int] = 25
# Below this many active units there is no sample, and no score is invented.
COHERENCE_MIN_UNITS: Final[int] = 3
# Random sets of units drawn from this same pool, to measure what an incoherent
# concept actually scores here. Without this baseline the cosine is unreadable:
# the embedding space is anisotropic and two unrelated paragraphs score far from 0.
COHERENCE_NULL_SAMPLES: Final[int] = 200

# --- Concept labelling (Phase 2, interpretability only) ---------------------
# Nothing in retrieval, scoring or expansion ever reads a label: a concept's
# embedding is its dictionary atom. Only one dictionary of the sweep is labelled.
LABELING_MODEL: Final[str] = "claude-sonnet-5"
LABELED_K: Final[int] = 2048
LABEL_EVIDENCE_UNITS: Final[int] = 10
# Published rates for the model above, per million tokens. Recorded with the run
# so a later relabelling can be compared against this one on cost as well.
LABELING_INPUT_USD_PER_MTOK: Final[float] = 2.00
LABELING_OUTPUT_USD_PER_MTOK: Final[float] = 10.00

# --- Hybrid conceptual retrieval (Phase 3) ----------------------------------
# Everything this phase decides is declared here before a number is measured.
# Nothing below may be repeated inline in a retriever, a sweep or the CLI: a
# constant that appears twice is a constant that will disagree with itself.

SELECTION_DIR: Final[Path] = DATA_DIR / "selection"
# Dev and test question embeddings, cached once. A dense query costs ~30.7 ms on
# this CPU, so re-encoding the same 600 dev strings on every pass of the sweep
# would cost hours and buy nothing - see decision D1 of the phase plan.
QUESTION_CACHE_DIR: Final[Path] = CACHE_DIR / "questions"

# How the question becomes a concept vector. Truncation is part of the operator,
# not a knob swept beside it, so the three arms are one decision with three
# alternatives (decision D4). `projection_*` is `atoms @ q` clipped at zero;
# `sparse_coding` runs the corpus coding penalty over the query.
QUERY_OPERATORS: Final[tuple[str, ...]] = (
    "projection_top16",
    "projection_full",
    "sparse_coding",
)
# The dense end of SPARSITY_TARGET_BAND: the corpus is coded at 8-16 active
# concepts per unit, so a top-16 query describes the question at the same
# resolution the units are described at.
QUERY_TOP_M: Final[int] = 16

# Both views of X are measured and neither is assumed. `row_normalized` is a pure
# function of `raw`, and both exist on disk from Phase 2: the sweep loads the one
# it measures rather than deriving it, so what is measured is what a later phase
# would load.
CONCEPT_VIEWS: Final[tuple[str, ...]] = ("raw", "row_normalized")

# Hub damping: a rarity weight over the query vector, leaving X and the atoms
# untouched. `none` is the identity and must be provably so. See decision D7.
DAMPING_MODES: Final[tuple[str, ...]] = ("none", "idf")

# Fusion. The weighted scheme buys a degree of freedom; RRF is the parameter-free
# reference it has to earn that freedom against.
FUSION_SCHEMES: Final[tuple[str, ...]] = ("weighted", "rrf")
# `s = w * dense + (1 - w) * other`. The endpoints are in the grid on purpose: an
# optimum at w = 1.0 says the second signal contributes nothing, and that is a
# result to be read off the curve rather than inferred from its absence.
FUSION_WEIGHT_GRID: Final[tuple[float, ...]] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)
# `s = sum(1 / (RRF_K + rank))`. The standard constant, fixed rather than fitted:
# fitting it would make the parameter-free reference no longer parameter-free.
RRF_K: Final[int] = 60

# What chooses the space, and at which budget. Kept as constants because the
# selection artifact has to name them and the sweep has to obey them.
SELECTION_METRIC: Final[str] = "gold_recall"
SELECTION_BUDGET: Final[int] = 2048

# A concept supported by fewer units than this is thin: reported beside recall,
# because a space can win on recall while being made of dimensions the corpus
# barely uses. Read from the Phase 2 diagnostics, never recomputed here.
THIN_CONCEPT_MAX_UNITS: Final[int] = 30


# --- Query-aware iterative expansion (Phase 4) ------------------------------
# Everything this phase decides is declared here before a number is measured,
# and the two grids below are the *only* free parameters it has.

EXPANSION_DIR: Final[Path] = DATA_DIR / "expansion"
# Per-query traces of what the expansion did, for the sample declared in D10.
# Reporting and diagnosis only: nothing in the metrics path reads one.
TRACES_DIR: Final[Path] = DATA_DIR / "traces"

# How deep every system is asked before the budget filling starts. It was a
# default argument in the CLI until Phase 4 needed to assert something about it:
# the diffusion asks its seed for exactly this, so the two cannot drift apart.
EVALUATION_TOP_K: Final[int] = 100
# Decision D5: the seed is asked for the depth the harness reads, and this is not
# a third free parameter. Asking for more would make "restart = 1.0 reproduces the
# seed" a property that holds at one depth and fails at another.
SEED_TOP_K: Final[int] = EVALUATION_TOP_K

# Which retriever the walk starts from. `dense` is System C proper; `conceptual`
# is the isolating variant of HU-3, which exists so that a gain can be told apart
# from "the dense retriever already brought almost everything".
SEED_ARMS: Final[tuple[str, ...]] = ("dense", "conceptual")

# First free parameter: how much of the mass returns to the seed's own signal at
# every round. The endpoints are deliberately absent - 1.0 is the seed itself and
# is asserted by a test rather than paid for with a dev evaluation, and 0.0 is the
# declared failure mode of sections 52-53, measured once after the choice is made.
RESTART_GRID: Final[tuple[float, ...]] = (0.2, 0.4, 0.6, 0.8)

# Second free parameter: how the propagation operator is normalized (decision D3).
# `none` is the null hypothesis - maybe on this space it does not matter - and the
# other two demote well-connected units and concepts in different ways.
NORMALIZATION_ARMS: Final[tuple[str, ...]] = ("none", "symmetric", "stochastic")

# Declared, not fitted (HU-4). The walk stops when a round moves less than this
# fraction of the total mass, and never runs more rounds than the cap. HotpotQA is
# a two-hop benchmark and one round is one hop, so five is more than twice what the
# corpus's structure needs: a run that mostly reaches the cap is a finding about
# the threshold, not a reason to raise the cap.
STOP_THRESHOLD: Final[float] = 1e-3
MAX_ITERATIONS: Final[int] = 5

# What the whole phase may spend on dev: one evaluation is one pass of one
# retriever over the 600 dev questions. The grid costs 24, leaving slack that is
# counted and written into the selection artifact rather than assumed.
DEV_EVALUATION_CAP: Final[int] = 40

# What chooses the cell, and where. Phase 3 selected on `gold_recall` because it was
# choosing a space and half the evidence still tells you something about a space.
# This phase selects a retrieval system, and a HotpotQA question needs *both* of its
# paragraphs - so the figure that decides is Full Support, at the same budget.
EXPANSION_SELECTION_METRIC: Final[str] = "full_support"
EXPANSION_SELECTION_BUDGET: Final[int] = SELECTION_BUDGET

# How wide a round of the trace is. Reporting only, exactly like the Phase 2
# diagnostics above: these decide how much of a round a human reads, never what
# the walk computed. A round touches every concept the mass reaches, and writing
# all 512 of them per round per question would produce an artifact nobody opens.
TRACE_TOP_CONCEPTS: Final[int] = 10
TRACE_TOP_UNITS: Final[int] = 10


# --- Text-derived concepts: navigation pilot (Phase 5) ----------------------
# Declared before any extraction runs or any hop is measured. The spec settles the
# choices; `5.0_text_derived_concepts.md` gives the reason for each value below.

PILOT_DIR: Final[Path] = DATA_DIR / "pilot"
EXTRACTION_DIR: Final[Path] = DATA_DIR / "extraction"
NODES_DIR: Final[Path] = DATA_DIR / "nodes"
NAVIGATION_DIR: Final[Path] = DATA_DIR / "navigation"

# The second-hop protocol of the Phase 4 diagnostic, reused unchanged (HU-5): a reader
# has read dense's top-10, and a hop is scored at these two depths.
PILOT_READ_DEPTH: Final[int] = 10
SECOND_HOP_DEPTHS: Final[tuple[int, ...]] = (10, 100)

# The extractor. Sonnet 5 was chosen in the specification conversation: the task is
# structured extraction, and vocabulary consistency is the normalization's job rather
# than a larger model's. Published per-token rates, and the Batches API's discount.
EXTRACTION_MODEL: Final[str] = "claude-sonnet-5"
EXTRACTION_INPUT_USD_PER_MTOK: Final[float] = 2.00
EXTRACTION_OUTPUT_USD_PER_MTOK: Final[float] = 10.00
BATCH_PRICE_FACTOR: Final[float] = 0.5
# A ceiling, not an expectation: the estimate is taken from the sample (D4).
EXTRACTION_MAX_TOKENS: Final[int] = 1024

# Bounds on what an answer may hold, enforced client-side because structured outputs
# cannot express them. An answer outside them is a failure, never truncated (D3).
MAX_NODES_PER_TYPE: Final[int] = 50
MAX_NODE_CHARS: Final[int] = 120

# Cost guards (D4, D5). The margin exists because the Phase 2 labelling ran 45% over
# its forecast. The ceiling was 25 USD in the approved spec; deviation 5.1 raised it to
# 35 USD after the sample measured 779.4 input tokens per request and an estimate of 33.17
# USD. Nothing else in these guards changed with it.
EXTRACTION_SAMPLE_SIZE: Final[int] = 20
EXTRACTION_ESTIMATE_MARGIN: Final[float] = 1.5
EXTRACTION_COST_CEILING_USD: Final[float] = 35.0
EXTRACTION_BATCH_SIZE: Final[int] = 5000
# Above this share of the pool, failed extractions are a finding that stops the phase.
EXTRACTION_FAILURE_FINDING: Final[float] = 0.01

# Two node types and three arms, so that navigating by names can be told apart from
# navigating by concepts (HU-4).
NODE_TYPES: Final[tuple[str, ...]] = ("entity", "concept")
NAVIGATION_ARMS: Final[tuple[str, ...]] = ("entity", "concept", "entity+concept")
NORMALIZATION_VERSION: Final[str] = "normalization-v1"

# The gate of HU-7, exactly as the spec writes it: hit@10 per question, a one-sided
# paired sign test at this level, against a comparator frozen before measuring.
GATE_METRIC_DEPTH: Final[int] = 10
GATE_ALPHA: Final[float] = 0.05
GATE_COMPARATOR: Final[str] = "bm25-question"

# Reporting only: how many traces the report reads per arm, and which pooled-embedding
# space the reference concept hop uses (the one Phase 3 selected).
TRACES_READ_PER_ARM: Final[int] = 5
REFERENCE_CONCEPT_K: Final[int] = 512


# --- Dense + entity navigation, end to end (Phase 6) -------------------------
# Declared before any Phase 6 number exists. The spec (`6.spec.md`) fixes every value;
# `6.0_dense_entity_navigation.md` gives the reason for each (decisions D1, D2, D20). The
# decision parameters of the statistical convention live beside this module, in
# `decision_parameters.py`: they name the test split, and this module is imported by the
# Phase 3 selection, whose guard forbids naming any split but dev.

# D1: everything the phase writes lives here, flat, and nothing under RESULTS_DIR. A
# Phase 6 `run-hybrid-bm25-test-*.json` there would become the Phase 3 control for the
# comparison reader, which takes the newest file of each system.
REPLACEMENT_DIR: Final[Path] = DATA_DIR / "replacement"

# The second-stage component and the three systems, by the hybrid naming rule:
# `hybrid-` plus the second component's name.
ENTITY_HOP_NAME: Final[str] = "entity-hop"
ENTITY_HOP_TYPES: Final[tuple[str, ...]] = ("entity",)
PHASE_6_SYSTEMS: Final[tuple[str, ...]] = ("dense", "hybrid-bm25", "hybrid-entity-hop")
# The entity list is cut at the depth every fusion component is asked for.
ENTITY_HOP_MAX_DEPTH: Final[int] = EVALUATION_TOP_K

# D2: the inherited artifacts, pinned by their full digests. The spec quotes each by
# prefix; these were copied once from the artifacts and a data-dependent test holds them
# equal. Loading derives file paths from these constants, never from a summary on disk.
PHASE_1_UNIT_SET_HASH: Final[str] = "101f564fdcca620c"
PINNED_SELECTION_DIGEST: Final[str] = (
    "91daa10ef0a75b6eba377d18a55ac868467b01b09fb7284c7835a84d4e4e602d"  # pragma: allowlist secret
)
PINNED_SELECTION_FROZEN_AT: Final[str] = "2026-09-11T13:30:36+00:00"
PINNED_PILOT_DIGEST: Final[str] = "e0f0af8f468dbf3332d9311948f416b75e4e5bf0d9971a0bd332b04acff2c364"
PINNED_HOP_RUN_DIGEST: Final[str] = (
    "49847f3cedb635406416bbb15dfa1913f4fa10782ed07a6461ccd56d5ded246a"  # pragma: allowlist secret
)
PINNED_NAVIGATION_TRACES_DIGEST: Final[str] = (
    "163777a044e1d6fcded400495a6e8f9917411e614978aa1aa43427a6cc8d5925"  # pragma: allowlist secret
)
PINNED_EXTRACTION_DIGEST: Final[str] = (
    "8e59854118ae5f4d83d88ce967098801f9f922309d7beb6b3e3deaf06fb1f12f"  # pragma: allowlist secret
)
PINNED_EXTRACTION_PROMPT_DIGEST: Final[str] = "0107de3ae9b4e4a3"
PINNED_NODE_INDEX_DIGEST: Final[str] = (
    "b498f99418389b2f7849c680cda1749ea58700a9997fbfed7ce474d3b0a0559a"  # pragma: allowlist secret
)
PINNED_NORMALIZATION_VERSION: Final[str] = NORMALIZATION_VERSION

# The four historical result files of the Phase 3 control. They carry no digest, so they
# are pinned by name, their configuration is checked key by key (HU-2), and the sha256 of
# their bytes is recorded at the dev stage. Before the test stage, the two test files are
# identified and never read for their figures.
# Keyed by system and grouped by role rather than by split name: this module is on the
# Phase 3 selection path, whose guard forbids any string naming a split other than dev.
HISTORICAL_DEV_RESULT_FILES: Final[dict[str, str]] = {
    "dense": "run-dense-dev-20260911T133154+0000.json",
    "hybrid-bm25": "run-hybrid-bm25-dev-20260911T133156+0000.json",
}
HISTORICAL_TEST_RESULT_FILES: Final[dict[str, str]] = {
    "dense": "run-dense-test-20260911T133159+0000.json",
    "hybrid-bm25": "run-hybrid-bm25-test-20260911T133204+0000.json",
}
HISTORICAL_CONFIG_KEYS: Final[tuple[str, ...]] = (
    "unit_set_hash",
    "tokenizer",
    "seed",
    "top_k",
    "model",
    "revision",
)
HISTORICAL_HYBRID_CONFIG_KEYS: Final[tuple[str, ...]] = ("fusion_scheme", "fusion_weights")

# Counts the spec quotes. Checked on load, and a mismatch stops with both numbers.
EXPECTED_N_UNITS: Final[int] = 19366
EXPECTED_ENTITY_NODES: Final[int] = 96416
EXPECTED_FAILED_EXTRACTIONS: Final[int] = 8
EXPECTED_UNITS_WITHOUT_ENTITY_NODE: Final[int] = 163
EXPECTED_PILOT_QUESTIONS: Final[int] = 152
EXPECTED_ENTITY_TRACES: Final[int] = 116

# OI-2, decided: the node hop's own test tolerance, a relative tolerance applied to a
# `math.fsum` of the shared nodes' weights against the recorded score.
TRACE_WEIGHT_REL_TOL: Final[float] = 1e-12


# --- Cheap entity extraction (Phase 7) ---------------------------------------
# Declared before the first Phase 7 number exists, which is the whole point: the
# spec (`7.spec.md`, decision 10) forbids choosing a model, a revision, a label set
# or a bar after a figure has been seen. Every value below comes from
# `7.0_cheap_entity_extraction.md`, decisions D1-D5, D9 and D11. A different
# checkpoint, pipeline or threshold is a **separate declared candidate**, never a
# retune of one of these.

PHASE_7_DIR: Final[Path] = DATA_DIR / "phase7"

# The two local candidates. The Claude extraction of Phase 5 is the reference arm and
# is quoted from its existing artifact, never re-run, so it is not an id here.
GLINER_EXTRACTOR: Final[str] = "gliner"
SPACY_EXTRACTOR: Final[str] = "spacy"
PHASE_7_EXTRACTORS: Final[tuple[str, ...]] = (GLINER_EXTRACTOR, SPACY_EXTRACTOR)

# D1-D3: candidate B, local zero-shot NER. Pinned to a commit, never to `main`, for
# the same reason the embedding model is: a moving revision silently redefines every
# node the phase extracts while the artifact keeps claiming the same configuration.
GLINER_MODEL: Final[str] = "urchade/gliner_medium-v2.1"
GLINER_REVISION: Final[str] = "40ec419335d09393f298636f471328b722c6da9e"
GLINER_TOKENIZER_MODEL: Final[str] = "microsoft/deberta-v3-base"
GLINER_TOKENIZER_REVISION: Final[str] = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
# D14: what is fetched, and therefore what cannot be deserialized. Both repositories
# also ship `pytorch_model.bin`, and the DeBERTa one `tf_model.h5` and `rust_model.ot`;
# none of them is in an allow-list, so no pickle archive ever reaches the disk.
GLINER_ALLOW_PATTERNS: Final[tuple[str, ...]] = ("gliner_config.json", "model.safetensors")
GLINER_TOKENIZER_ALLOW_PATTERNS: Final[tuple[str, ...]] = (
    "config.json",
    "spm.model",
    "tokenizer_config.json",
)
# One label per category the frozen Phase 5 prompt already names: people,
# organizations, places, works, events. The comparison is between extractors, so the
# ontology has to be the same question asked of a different reader.
GLINER_LABELS: Final[tuple[str, ...]] = (
    "person",
    "organization",
    "location",
    "work of art",
    "event",
)
GLINER_THRESHOLD: Final[float] = 0.5
GLINER_FLAT_NER: Final[bool] = True
GLINER_MULTI_LABEL: Final[bool] = False
GLINER_BATCH_SIZE: Final[int] = 8
# The checkpoint's own reading window and span width. A paragraph longer than this is
# split at its own sentence boundaries (D6.5); the *indexing* unit is untouched.
GLINER_MAX_LEN: Final[int] = 384
GLINER_MAX_WIDTH: Final[int] = 12

# D4-D5: candidate C, conventional local NER. `sm` and not `trf`: this is the
# conservative, CPU-friendly arm, and a transformer pipeline would be a different
# candidate with a different cost profile.
SPACY_MODEL: Final[str] = "en_core_web_sm"
SPACY_MODEL_VERSION: Final[str] = "3.8.0"
SPACY_MODEL_WHEEL_URL: Final[str] = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
# Only `tok2vec` and `ner` run; everything else is loaded for nothing.
SPACY_EXCLUDE: Final[tuple[str, ...]] = (
    "tagger",
    "parser",
    "attribute_ruler",
    "lemmatizer",
    "senter",
)
SPACY_BATCH_SIZE: Final[int] = 64
SPACY_PROCESSES: Final[int] = 1
# Keep every OntoNotes label that names a *thing*; drop the seven that name a quantity
# or a time. Dates and cardinals are not named things in the Phase 5 sense and would
# manufacture hub nodes across unrelated paragraphs.
SPACY_LABELS: Final[tuple[str, ...]] = (
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "NORP",
    "ORG",
    "PERSON",
    "PRODUCT",
    "WORK_OF_ART",
)
SPACY_DROPPED_LABELS: Final[tuple[str, ...]] = (
    "CARDINAL",
    "DATE",
    "MONEY",
    "ORDINAL",
    "PERCENT",
    "QUANTITY",
    "TIME",
)

# The inherited Phase 6 figures the phase is read against, Full Support @2,048. Copied
# once from the run artifacts named beside them and held equal by a data-dependent
# test, so a candidate is never compared against a number typed from memory.
PHASE_7_REFERENCE_DEV_FULL_SUPPORT: Final[dict[str, float]] = {
    "dense": 0.8116666666666666,
    "hybrid-bm25": 0.8633333333333333,
    "hybrid-entity-hop": 0.8783333333333333,
}
PHASE_7_REFERENCE_HELD_OUT_FULL_SUPPORT: Final[dict[str, float]] = {
    "dense": 0.825,
    "hybrid-bm25": 0.8642857142857143,
    "hybrid-entity-hop": 0.89,
}
PHASE_7_REFERENCE_DEV_FILES: Final[dict[str, str]] = {
    "dense": "run-dense-dev-20260917T174901+0000.json",
    "hybrid-bm25": "run-hybrid-bm25-dev-20260917T174929+0000.json",
    "hybrid-entity-hop": "run-hybrid-entity-hop-dev-20260917T212300+0000.json",
}
PHASE_7_REFERENCE_HELD_OUT_FILES: Final[dict[str, str]] = {
    "dense": "run-dense-test-20260917T215214+0000.json",
    "hybrid-bm25": "run-hybrid-bm25-test-20260917T215216+0000.json",
    "hybrid-entity-hop": "run-hybrid-entity-hop-test-20260917T215231+0000.json",
}

# D11, the retention bar: a candidate is materially equivalent when it keeps at least
# this share of the Claude Entity Hop's dev gain over Dense. The share applied to the
# measured dev figures lands exactly on 517 of the 600 dev questions
# (0.8116667 + 0.75 * 0.0666667 = 0.8616667 = 517/600); the spec writes that bar as
# `0.8617`, which is the same number rounded for reading. The count is what the rule
# is applied on, because a float comparison against the rounded form would reject a
# candidate sitting exactly on the bar.
PHASE_7_RETENTION_SHARE: Final[float] = 0.75
PHASE_7_RETENTION_BAR: Final[float] = 0.8617
PHASE_7_RETENTION_BAR_QUESTIONS: Final[int] = 517

# D11, the economic bar. The first two are absolute: no per-paragraph third-party API
# call, and nothing paid for the extraction software itself. The last two are the
# ceilings the spec sets on a FullWiki projection and on this phase's own compute.
PHASE_7_SOFTWARE_COST_CEILING_USD: Final[float] = 0.0
PHASE_7_FULLWIKI_HOURS_CEILING: Final[float] = 48.0
PHASE_7_INFRASTRUCTURE_CEILING_USD: Final[float] = 30.0

# D9: the scale the throughput is projected to, linearly and with the arithmetic shown.
PROJECTION_PARAGRAPHS: Final[int] = 5_000_000


# --- Strong Dense + Entity Hop (Phase 8) -------------------------------------
# Declared before the first Phase 8 number exists. The spec (`8.spec.md`) settles every
# value below and closes all ten of its decisions; `8.0_strong_dense.md` gives the
# implementation reason for each (D1-D3, D10). Exactly one variable changes in this
# phase - the Dense retriever - so `EMBEDDING_MODEL` is deliberately *not* repointed:
# every inherited stage builds `SentenceTransformerBackend()` from it and would look for
# a corpus cache that does not exist. The Phase 8 stages construct their backend from the
# constants below instead, and nothing else in the project changes model by accident.

PHASE_8_DIR: Final[Path] = DATA_DIR / "phase8"

# Decision 1, approved in the spec: one modern strong open Dense model, chosen a priori
# from published evidence rather than by comparing dev scores across encoders, which is
# the leaderboard the master plan forbids. Pinned to a commit and never to a branch, for
# the same reason `EMBEDDING_REVISION` is. Full width, no MRL truncation.
PHASE_8_DENSE_MODEL: Final[str] = "Qwen/Qwen3-Embedding-0.6B"
PHASE_8_DENSE_REVISION: Final[str] = (
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"  # pragma: allowlist secret
)
PHASE_8_DENSE_DIM: Final[int] = 1024
# What is fetched, and therefore what cannot be deserialized: the pinned revision ships
# `model.safetensors` as its only weight file, and its sha256 is recorded over the bytes
# actually loaded (R7, the prospective answer to Phase 7's SEC-031).
PHASE_8_WEIGHTS_FILE: Final[str] = "model.safetensors"

# Decision 3 / R2: the model is asymmetric, so the query instruction is a declared
# configuration item rather than a free parameter. The run reads `model.prompts["query"]`
# and aborts on any inexact match before producing a single embedding; documents stay
# unprefixed. It is never customized or tuned for HotpotQA - a prompt fitted to this
# benchmark would be a hyperparameter chosen on our own data.
PHASE_8_QUERY_PROMPT: Final[str] = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

# The inherited entity representation, pinned by digest and read-only for this phase.
# Phase 8 loads the Phase 7 GLiNER index and writes nothing into `data/phase7/`.
PHASE_8_GLINER_DIR: Final[Path] = PHASE_7_DIR / "gliner"
PHASE_8_GLINER_EXTRACTION_DIGEST: Final[str] = (
    "840b4f78325d5c3393588e82312b29e168bc265e42127896c588e75a422a0aba"  # pragma: allowlist secret
)
PHASE_8_GLINER_INDEX_DIGEST: Final[str] = (
    "0aec5c32440ff600c1abbc7d866ad143ddf5c5263bf8252f1dbacdffc1496be5"  # pragma: allowlist secret
)

# The three systems the phase measures, and nothing else.
PHASE_8_SYSTEMS: Final[tuple[str, ...]] = ("dense", "hybrid-bm25", "hybrid-entity-hop")

# The inherited BGE-small figures this phase is read against, Full Support @2,048. The
# Phase 6 rows are taken from the constants Phase 7 already pinned rather than typed a
# second time; the GLiNER rows come from `data/phase7/` and are held equal to those
# artifacts by a data-dependent test. The GLiNER row is the comparison baseline, because
# Phase 8 inherits the GLiNER index; the Claude row is historical context only.
PHASE_8_INHERITED_DEV_FULL_SUPPORT: Final[dict[str, float]] = {
    "dense": PHASE_7_REFERENCE_DEV_FULL_SUPPORT["dense"],
    "hybrid-bm25": PHASE_7_REFERENCE_DEV_FULL_SUPPORT["hybrid-bm25"],
    "hybrid-entity-hop-claude": PHASE_7_REFERENCE_DEV_FULL_SUPPORT["hybrid-entity-hop"],
    "hybrid-entity-hop-gliner": 0.8633333333333333,
}
PHASE_8_INHERITED_HELD_OUT_FULL_SUPPORT: Final[dict[str, float]] = {
    "dense": PHASE_7_REFERENCE_HELD_OUT_FULL_SUPPORT["dense"],
    "hybrid-bm25": PHASE_7_REFERENCE_HELD_OUT_FULL_SUPPORT["hybrid-bm25"],
    "hybrid-entity-hop-claude": PHASE_7_REFERENCE_HELD_OUT_FULL_SUPPORT["hybrid-entity-hop"],
    "hybrid-entity-hop-gliner": 0.8792857142857143,
}
# Where each inherited figure is read from, relative to `DATA_DIR`, so a report can name
# the file behind every number it quotes.
PHASE_8_INHERITED_DEV_FILES: Final[dict[str, str]] = {
    "dense": f"{REPLACEMENT_DIR.name}/{PHASE_7_REFERENCE_DEV_FILES['dense']}",
    "hybrid-bm25": f"{REPLACEMENT_DIR.name}/{PHASE_7_REFERENCE_DEV_FILES['hybrid-bm25']}",
    "hybrid-entity-hop-claude": (
        f"{REPLACEMENT_DIR.name}/{PHASE_7_REFERENCE_DEV_FILES['hybrid-entity-hop']}"
    ),
    "hybrid-entity-hop-gliner": (
        f"{PHASE_7_DIR.name}/{PHASE_8_GLINER_DIR.name}/"
        "run-hybrid-entity-hop-dev-20260919T070042+0000.json"
    ),
}
PHASE_8_INHERITED_HELD_OUT_FILES: Final[dict[str, str]] = {
    "dense": f"{REPLACEMENT_DIR.name}/{PHASE_7_REFERENCE_HELD_OUT_FILES['dense']}",
    "hybrid-bm25": f"{REPLACEMENT_DIR.name}/{PHASE_7_REFERENCE_HELD_OUT_FILES['hybrid-bm25']}",
    "hybrid-entity-hop-claude": (
        f"{REPLACEMENT_DIR.name}/{PHASE_7_REFERENCE_HELD_OUT_FILES['hybrid-entity-hop']}"
    ),
    "hybrid-entity-hop-gliner": (
        f"{PHASE_7_DIR.name}/{PHASE_8_GLINER_DIR.name}/"
        "run-hybrid-entity-hop-test-20260919T070735+0000.json"
    ),
}

# Decision 4 / D10, the stop rule: if Strong Dense alone does not beat the inherited
# Dense dev reading, the phase's premise has failed and the held-out split is neither
# measured nor encoded. The comparison is made in whole questions, never on floats: 487
# of 600 is a tie, and a tie is not "substantially stronger", so the rule passes only on
# at least one more supported question. The count below is that reading, and a test holds
# it equal to the arithmetic rather than to a number typed beside it.
PHASE_8_STOP_RULE_BASELINE_QUESTIONS: Final[int] = 487


# --- Deviation 8.1: Dense model ranking under larger retrieval corpora --------
# Every value below is fixed by `8.1_dense_scale_sensitivity.md` before the first 8.1
# number exists, and `8.1_implementation_plan.md` gives the implementation reason for
# each. Phase 8 stays closed with its STOP: this block adds a bounded, dev-only
# diagnostic and repoints nothing the earlier phases measured.
#
# The dictionaries below are keyed by **model** and by **outcome**, never by split name.
# This module is on the import closure of `evaluation/selection.py`, whose static guard
# forbids any string literal naming a split other than dev.

PHASE_8_1_DIR: Final[Path] = DATA_DIR / "phase8_1"
# 8.1's own cache directories, and the reason they exist at all: `question_cache_key`
# does not depend on the corpus, so the 600 dev questions under the same model, revision
# and prompt hash to the **same key** as the Phase 1-8 question caches. Encoding them
# into the shared directory would overwrite the artifact behind 487/600 with vectors
# produced on other hardware. Separate directories make that impossible, not unlikely.
PHASE_8_1_CACHE_DIR: Final[Path] = PHASE_8_1_DIR / "cache"
PHASE_8_1_QUESTION_CACHE_DIR: Final[Path] = PHASE_8_1_CACHE_DIR / "questions"

# The official HotpotQA FullWiki release of introductory paragraphs, English Wikipedia
# 2017-10-01. A modern snapshot or an independently reconstructed corpus cannot replace
# it inside 8.1 without another written deviation.
PHASE_8_1_SOURCE_URL: Final[str] = "https://nlp.stanford.edu/projects/hotpotqa/"
PHASE_8_1_SOURCE_ARCHIVE: Final[str] = (
    "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
)
PHASE_8_1_SOURCE_LICENSE: Final[str] = "CC BY-SA 4.0"
# The size and MD5 were carried into deviation 8.1 from its recorded release metadata.
# S1 verified both against the staged archive bytes. The live HotpotQA page was not
# checked, so neither this constant nor source.json claims that it was.
PHASE_8_1_DECLARED_BYTES: Final[int] = 1_553_565_403
PHASE_8_1_DECLARED_MD5: Final[str] = "01edf64cd120ecc03a2745352779514c"  # pragma: allowlist secret
PHASE_8_1_DECLARED_BASIS: Final[str] = (
    "size and MD5 recorded in deviation 8.1 and verified against the staged archive bytes; "
    "the live HotpotQA page was not checked"
)

# The four nested corpora and the distractor prefixes that build them. C19 is always the
# frozen Phase 1 pool in its existing order, and the three larger corpora are index
# prefixes of one single frozen ordering, so `C19 < C100 < C250 < C500` as subsets.
PHASE_8_1_CORPUS_SIZES: Final[tuple[int, ...]] = (19366, 100000, 250000, 500000)
PHASE_8_1_CORPUS_LABELS: Final[tuple[str, ...]] = ("c19", "c100", "c250", "c500")
PHASE_8_1_DISTRACTOR_PREFIXES: Final[tuple[int, ...]] = (80634, 230634, 480634)

# The ordering rule. `42` is a fixed salt inside a hash input string: there is no RNG on
# this path, no `default_rng`, no `shuffle` and no sampling call, and a test asserts the
# module that applies it names none of them.
PHASE_8_1_ORDER_SALT: Final[str] = "42"
PHASE_8_1_ORDER_RULE: Final[str] = "sha256-salted-title-plaintext-v1"
PHASE_8_1_ORDER_SALT_BASIS: Final[str] = "fixed hash salt, not RNG state"

# The reconciliation gate. At or below this many unresolved C19 units, conservative
# title exclusion is the remedy and the experiment continues; above it, the terminal
# state is `data_stop`. 50 is 0.26% of 19,366, and the ceiling is never raised after
# seeing how many units actually missed.
PHASE_8_1_UNMATCHED_CEILING: Final[int] = 50

# The C19 reproduction gate. Level 1 must reproduce both counts exactly from the
# historical caches; level 2 - C19 as a prefix of the new C500 encode - may differ by at
# most this many questions in either model before it becomes `reproduction_stop`.
PHASE_8_1_C19_DEV_SUPPORTED: Final[dict[str, int]] = {"bge": 487, "qwen": 446}
PHASE_8_1_REENCODE_TOLERANCE: Final[int] = 1

# The pre-declared outcome thresholds. 20 questions is roughly half the 41-question C19
# deficit; 244 is half BGE's own 487, rounded up, and is the floor that separates
# `convergence` from `both_degrade`. Neither is adjusted after a scale result is seen.
PHASE_8_1_CONVERGENCE_QUESTIONS: Final[int] = 20
PHASE_8_1_BGE_RETENTION_FLOOR: Final[int] = 244
# Evaluated in this order by `classify_outcome`, and no other label is invented.
PHASE_8_1_OUTCOMES: Final[tuple[str, ...]] = (
    "crossover",
    "convergence",
    "both_degrade",
    "stable_ranking",
)
PHASE_8_1_STOPS: Final[tuple[str, ...]] = ("data_stop", "reproduction_stop")

# The two encoders, by short name. Their model ids, revisions, widths and query handling
# are **reused** from the constants above rather than re-declared: two declarations of
# one pin are two places for it to disagree with itself.
PHASE_8_1_MODELS: Final[tuple[str, ...]] = ("bge", "qwen")

# The historical caches level 1 reads, read-only. The BGE corpus key is the
# revision-pinned one: `a28365b7ed90ed10` holds the same vectors under revision "main"
# and is a migration artifact, not the file to read.
PHASE_8_1_HISTORICAL_CORPUS_KEYS: Final[dict[str, str]] = {
    "bge": "bacd74e7f468f0d4",
    "qwen": "66ef5c19a95f6cbe",
}
PHASE_8_1_HISTORICAL_DEV_QUERY_KEYS: Final[dict[str, str]] = {
    "bge": "35329d9e75da1d4f",
    "qwen": "c9c589c29c226fb3",
}

# Ceilings on the untrusted archive. Every decompression is bounded, so a member that
# expands past its ceiling raises instead of filling memory - the `MAX_CORPUS_BYTES`
# precedent of `corpus/download.py`, applied to a nested archive.
PHASE_8_1_MAX_MEMBER_BYTES: Final[int] = 256 * 1024 * 1024
PHASE_8_1_MAX_LINE_BYTES: Final[int] = 4 * 1024 * 1024
PHASE_8_1_MAX_TOTAL_RECORDS: Final[int] = 8_000_000
# How many members the layout probe reads before the schema is recorded.
PHASE_8_1_PROBE_MEMBERS: Final[int] = 3
# How many records the probe parses per probed member.
PHASE_8_1_PROBE_RECORDS: Final[int] = 20

# Candidate field names, in preference order, for the layout probe to **resolve** the
# observed record schema against. The probe records which candidate the archive actually
# holds and refuses when none of them is there; the parser then dispatches on the
# recorded fact. This is a resolution mechanism, not an assumption about the dump.
PHASE_8_1_TITLE_FIELDS: Final[tuple[str, ...]] = ("title",)
PHASE_8_1_SENTENCES_FIELDS: Final[tuple[str, ...]] = ("text", "sentences")
PHASE_8_1_PAGE_ID_FIELDS: Final[tuple[str, ...]] = ("id", "page_id", "pageid")

# Texts per tokenizer call. `TokenCounter.count_units` tokenizes its whole input in one
# call, which is correct at 19,366 texts and a memory fault at 480,634; the batching
# wrapper lives in `corpus/scale_corpus.py` and `evaluation/budget.py` is not edited.
PHASE_8_1_TOKENIZE_BATCH: Final[int] = 2048

# Above this mean seconds per query at C500, the blockwise exact fallback of the plan's
# D3 is written. Below it, `DenseRetriever` is reused unchanged and no second retriever
# exists - a branch for a failure that has not occurred is not written.
PHASE_8_1_QUERY_SECONDS_CEILING: Final[float] = 1.0


# --- Phase 9: HotpotQA FullWiki ------------------------------------------------
# Every value below is fixed by `9.spec.md` (approved and frozen 2026-09-23) before the
# first Phase 9 number exists. One variable changes - corpus scale - so the models, the
# extractor, the Entity Hop and both fusion weights are the inherited ones, read from the
# constants and artifacts that already pin them. Keyed by cohort and system, never by a
# split name: this module is on the import closure of `evaluation/selection.py`.

PHASE_9_DIR: Final[Path] = DATA_DIR / "phase9"
# Own caches, never the historical ones: `question_cache_key` does not include the corpus.
PHASE_9_CACHE_DIR: Final[Path] = PHASE_9_DIR / "cache"
PHASE_9_QUESTION_CACHE_DIR: Final[Path] = PHASE_9_CACHE_DIR / "questions"

# D1: the archive 8.1 verified. Bytes and md5 are the 8.1 constants; the sha256 is the
# local integrity record 8.1 measured.
PHASE_9_ARCHIVE_SHA256: Final[str] = (
    "1acca1c5cc93c4890ea51091d2bad7c3ef6987aead127ab88728dc9e26555729"  # pragma: allowlist secret
)

# D3: the three cohorts, by qid, and their sizes as acceptance conditions.
PHASE_9_RETRIEVAL_UNSEEN: Final[str] = "retrieval-unseen"
PHASE_9_STANDARD: Final[str] = "standard-dev"
PHASE_9_HISTORICAL_OVERLAP: Final[str] = "historical-overlap"
PHASE_9_COHORTS: Final[tuple[str, ...]] = (
    PHASE_9_RETRIEVAL_UNSEEN,
    PHASE_9_STANDARD,
    PHASE_9_HISTORICAL_OVERLAP,
)
PHASE_9_COHORT_SIZES: Final[dict[str, int]] = {
    PHASE_9_RETRIEVAL_UNSEEN: 5405,
    PHASE_9_STANDARD: 7405,
    PHASE_9_HISTORICAL_OVERLAP: N_QUESTIONS,
}

# D4: more questions than this with an unresolved gold title is DATA_STOP (1% of 7,405).
PHASE_9_UNRESOLVED_CEILING: Final[int] = 74

# D5: the only Dense model measured, and D6: the only extractor, at the frozen Phase 7
# configuration digest. The digest covers the library versions, so a host whose stack
# differs from Phase 7's cannot pass as the same extractor.
PHASE_9_GLINER_CONFIGURATION_DIGEST: Final[str] = "2f7864661b8ce7ff"
PHASE_9_GLINER_WEIGHTS_FILE: Final[str] = "model.safetensors"
# Units per resumable extraction shard: a finished shard is never re-extracted.
PHASE_9_EXTRACTION_SHARD_UNITS: Final[int] = 100_000

# D8: the three systems, by their inherited names, and their frozen weights. The weights
# are read from the recorded fits named below and must equal these, or nothing runs.
PHASE_9_SYSTEMS: Final[tuple[str, ...]] = ("dense", "hybrid-bm25", "hybrid-entity-hop")
PHASE_9_SYSTEM_LABELS: Final[dict[str, str]] = {
    "dense": "P9-A",
    "hybrid-bm25": "P9-B",
    "hybrid-entity-hop": "P9-C",
}
PHASE_9_FROZEN_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    "hybrid-bm25": {"dense": 0.5, "bm25": 0.5},
    "hybrid-entity-hop": {"dense": 0.7, ENTITY_HOP_NAME: 0.3},
}
PHASE_9_WEIGHT_TOLERANCE: Final[float] = 1e-9
# Phase 3 fitted the BM25 control; Phase 6 recorded it in its dev run's configuration.
PHASE_9_BM25_WEIGHTS_FILE: Final[Path] = (
    REPLACEMENT_DIR / PHASE_7_REFERENCE_DEV_FILES["hybrid-bm25"]
)
# Phase 7 fitted the GLiNER Entity Hop weight and recorded it in its dev artifact.
PHASE_9_ENTITY_WEIGHTS_FILE: Final[Path] = PHASE_8_GLINER_DIR / "dev.json"

# S4: the Phase 7 dev Full Support @2,048 counts the Phase 9 path must reproduce on the
# historical pool, in whole questions. Held equal to the artifacts by a data-dependent test.
PHASE_9_REPRODUCTION_DEV_SUPPORTED: Final[dict[str, int]] = {
    "dense": 487,
    "hybrid-bm25": 518,
    "hybrid-entity-hop": 518,
}

# D9/D10: the primary endpoint and its one test.
PHASE_9_BUDGETS: Final[tuple[int, ...]] = CONTEXT_BUDGETS
PHASE_9_PRIMARY_BUDGET: Final[int] = 2048
PHASE_9_KS: Final[tuple[int, ...]] = (2, 5, 10, 20)
PHASE_9_RANKING_DEPTH: Final[int] = EVALUATION_TOP_K
PHASE_9_ALPHA: Final[float] = 0.05
PHASE_9_TERMINAL_STATES: Final[tuple[str, ...]] = (
    "SCALE_SUPPORTED",
    "SCALE_NOT_SUPPORTED",
    "SCALE_REGRESSION",
    "DATA_STOP",
    "OPERATIONAL_STOP",
)

# D11: the frozen envelope.
PHASE_9_QUERY_SECONDS_CEILING: Final[float] = 1.0
PHASE_9_SPEND_CEILING_USD: Final[float] = 25.0
# The operational probe: a seeded sample of already-used historical dev questions,
# timed for rankings only; no metric is computed on it.
PHASE_9_PROBE_QUESTIONS: Final[int] = 100
PHASE_9_PROBE_SEED: Final[int] = DEFAULT_SEED


def ensure_directories() -> None:
    """Create the data directories if they do not exist yet."""
    for path in (
        DATA_DIR,
        CACHE_DIR,
        RESULTS_DIR,
        CONCEPTS_DIR,
        SELECTION_DIR,
        QUESTION_CACHE_DIR,
        EXPANSION_DIR,
        TRACES_DIR,
        PILOT_DIR,
        EXTRACTION_DIR,
        NODES_DIR,
        NAVIGATION_DIR,
        REPLACEMENT_DIR,
        PHASE_7_DIR,
        PHASE_8_DIR,
        PHASE_8_1_DIR,
        PHASE_8_1_CACHE_DIR,
        PHASE_8_1_QUESTION_CACHE_DIR,
        PHASE_9_DIR,
        PHASE_9_CACHE_DIR,
        PHASE_9_QUESTION_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
