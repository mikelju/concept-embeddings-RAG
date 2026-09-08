"""T3: subset selection and dev/test split must be reproducible forever.

Both derive from a hash rather than from an RNG stream: numpy and random may
change their streams across versions, and the test split has to still be the same
set of questions in six months.
"""

from concept_embeddings_rag.corpus.split import select_subset, split_questions


def questions(n: int) -> list[dict]:
    return [{"_id": f"q{i:05d}", "question": f"question {i}"} for i in range(n)]


def test_selection_is_deterministic_for_the_same_seed():
    pool = questions(500)
    first = select_subset(pool, n=100, seed=42)
    second = select_subset(pool, n=100, seed=42)
    assert [q["_id"] for q in first] == [q["_id"] for q in second]


def test_a_different_seed_selects_a_different_subset():
    pool = questions(500)
    first = {q["_id"] for q in select_subset(pool, n=100, seed=42)}
    second = {q["_id"] for q in select_subset(pool, n=100, seed=7)}
    assert first != second


def test_selection_ignores_input_order():
    pool = questions(200)
    shuffled = list(reversed(pool))
    assert select_subset(pool, n=50, seed=42) == select_subset(shuffled, n=50, seed=42)


def test_split_sizes_are_exact_and_disjoint():
    subset = questions(2000)
    split = split_questions(subset, n_dev=600, seed=42)

    assert len(split["dev"]) == 600
    assert len(split["test"]) == 1400
    dev_ids = {q["_id"] for q in split["dev"]}
    test_ids = {q["_id"] for q in split["test"]}
    assert dev_ids.isdisjoint(test_ids)
    assert len(dev_ids | test_ids) == 2000


def test_split_is_stable_across_runs():
    subset = questions(2000)
    first = split_questions(subset, n_dev=600, seed=42)
    second = split_questions(subset, n_dev=600, seed=42)
    assert [q["_id"] for q in first["dev"]] == [q["_id"] for q in second["dev"]]


def test_asking_for_more_questions_than_available_raises():
    import pytest

    with pytest.raises(ValueError):
        select_subset(questions(10), n=50, seed=42)
