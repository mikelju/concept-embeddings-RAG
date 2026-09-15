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
TOKENIZER_ID: Final[str] = EMBEDDING_MODEL

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
# declared failure mode of §52-§53, measured once after the choice is made.
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
# its forecast; the ceiling is the one the spec approved.
EXTRACTION_SAMPLE_SIZE: Final[int] = 20
EXTRACTION_ESTIMATE_MARGIN: Final[float] = 1.5
EXTRACTION_COST_CEILING_USD: Final[float] = 25.0
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
    ):
        path.mkdir(parents=True, exist_ok=True)
