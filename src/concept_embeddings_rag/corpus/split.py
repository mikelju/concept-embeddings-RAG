"""Subset selection and dev/test split, both derived from hashes.

An RNG stream is not a stable contract: `random` and numpy may change how a seed
maps to a sequence across versions, which would silently redefine the test split.
A hash of the question id cannot drift, so the same seed always selects the same
questions and assigns them to the same side, forever.
"""

import hashlib


def _sort_key(seed: int, qid: str) -> str:
    return hashlib.sha1(f"{seed}:{qid}".encode()).hexdigest()


def select_subset(questions: list[dict], n: int, seed: int) -> list[dict]:
    """Pick `n` questions deterministically, independent of input order."""
    if n > len(questions):
        raise ValueError(f"asked for {n} questions but only {len(questions)} are available")
    ordered = sorted(questions, key=lambda q: _sort_key(seed, q["_id"]))
    return ordered[:n]


def split_questions(questions: list[dict], n_dev: int, seed: int) -> dict[str, list[dict]]:
    """Split into development and test, deterministically and disjointly.

    The development split is what tuning is allowed to look at. The test split is
    measured once, at the end, and never used to make a decision.
    """
    if n_dev > len(questions):
        raise ValueError(f"n_dev={n_dev} exceeds the {len(questions)} questions provided")
    ordered = sorted(questions, key=lambda q: _sort_key(seed + 1, q["_id"]))
    return {"dev": ordered[:n_dev], "test": ordered[n_dev:]}
