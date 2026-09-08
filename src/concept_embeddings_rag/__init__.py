"""Retrieval over a corpus-induced concept space.

Phase 1 builds the measuring instrument: a frozen corpus, dense and BM25
baselines, and an evaluation harness at a fixed context budget.
"""

__version__ = "0.1.0"


def main() -> None:
    print("[INFO] Use the command line interface: python -m concept_embeddings_rag.cli --help")
