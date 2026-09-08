"""Deviation 1.1: assembling the corpus from the HuggingFace rows API.

The API returns a different shape from the original HotpotQA JSON, so the
normalization is what keeps every downstream stage unaware that the source moved.
Getting it wrong would silently mis-assign supporting facts.
"""

import pytest

from concept_embeddings_rag.corpus.hf_source import (
    RateLimitError,
    fetch_split,
    normalize_row,
    retry_call,
)

HF_ROW = {
    "id": "5a8b57f25542995d1e6f1371",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 1]},
    "context": {
        "title": ["Scott Derrickson", "Ed Wood"],
        "sentences": [
            ["Scott Derrickson is an American director.", "He was born in 1966."],
            ["Ed Wood was an American filmmaker.", "He directed Plan 9."],
        ],
    },
}


def test_normalization_restores_the_original_hotpotqa_shape():
    row = normalize_row(HF_ROW)

    assert row["_id"] == HF_ROW["id"]
    assert row["question"] == HF_ROW["question"]
    assert row["answer"] == "yes"
    assert row["context"] == [
        ["Scott Derrickson", ["Scott Derrickson is an American director.", "He was born in 1966."]],
        ["Ed Wood", ["Ed Wood was an American filmmaker.", "He directed Plan 9."]],
    ]
    assert row["supporting_facts"] == [["Scott Derrickson", 0], ["Ed Wood", 1]]


def test_normalized_rows_are_consumable_by_the_pool_builder():
    from concept_embeddings_rag.corpus.pool import build_pool

    units, questions = build_pool({"dev": [normalize_row(HF_ROW)]})

    assert len(questions) == 1
    assert len(questions[0].gold_unit_ids) == 2
    assert {u.title for u in units} == {"Scott Derrickson", "Ed Wood"}


def test_fetch_split_pages_until_every_row_is_collected():
    pages = []

    def fake_get(offset: int, length: int) -> dict:
        pages.append((offset, length))
        total = 250
        rows = [
            {"row": dict(HF_ROW, id=f"q{i}")} for i in range(offset, min(offset + length, total))
        ]
        return {"num_rows_total": total, "rows": rows}

    rows = fetch_split(page_fetcher=fake_get, page_size=100)

    assert len(rows) == 250
    assert pages == [(0, 100), (100, 100), (200, 100)]
    assert rows[0]["_id"] == "q0"
    assert rows[-1]["_id"] == "q249"


def test_fetch_split_stops_when_a_page_comes_back_empty():
    def fake_get(offset: int, length: int) -> dict:
        rows = [{"row": HF_ROW}] if offset == 0 else []
        return {"num_rows_total": 999, "rows": rows}

    rows = fetch_split(page_fetcher=fake_get, page_size=100)
    assert len(rows) == 1


def test_retry_call_retries_a_rate_limited_request_and_then_succeeds():
    """The API rate-limits bursts of requests; a 429 is a wait, not a failure."""
    attempts = []
    slept = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RateLimitError("429")
        return "ok"

    assert retry_call(flaky, attempts=5, sleep_fn=slept.append, base_delay=1.0) == "ok"
    assert len(attempts) == 3
    assert slept == [1.0, 2.0]  # exponential backoff


def test_retry_call_gives_up_after_the_last_attempt():
    def always_limited():
        raise RateLimitError("429")

    with pytest.raises(RateLimitError):
        retry_call(always_limited, attempts=3, sleep_fn=lambda _: None, base_delay=0.0)


def test_fetch_split_pauses_between_pages_to_stay_under_the_rate_limit():
    slept = []

    def fake_get(offset: int, length: int) -> dict:
        total = 250
        rows = [{"row": HF_ROW} for _ in range(offset, min(offset + length, total))]
        return {"num_rows_total": total, "rows": rows}

    fetch_split(page_fetcher=fake_get, page_size=100, pause=0.4, sleep_fn=slept.append)
    # Three pages, two pauses: there is nothing to be polite about after the last one.
    assert slept == [0.4, 0.4]


def test_paging_stops_at_the_row_ceiling_instead_of_looping_forever():
    """SEC-004: when the API stops reporting num_rows_total, an empty page was the
    only thing that ended the loop. A peer that never sends one exhausted memory."""
    import pytest

    from concept_embeddings_rag.corpus.hf_source import fetch_split

    def endless_pages(offset: int, length: int) -> dict:
        # No num_rows_total, and never an empty page: the shape that used to hang.
        return {"rows": [{"row": dict(HF_ROW, id=f"q{offset + i}")} for i in range(length)]}

    with pytest.raises(RuntimeError, match="ceiling"):
        fetch_split(page_fetcher=endless_pages, page_size=100, max_rows=500)
