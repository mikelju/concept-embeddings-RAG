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


def ensure_directories() -> None:
    """Create the data directories if they do not exist yet."""
    for path in (DATA_DIR, CACHE_DIR, RESULTS_DIR, CONCEPTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
