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


def ensure_directories() -> None:
    """Create the data directories if they do not exist yet."""
    for path in (DATA_DIR, CACHE_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
