"""Assemble HotpotQA from the HuggingFace rows API (see deviation 1.1).

The authors' original host stopped responding, so the corpus is pulled from the
canonical HuggingFace dataset instead. Rows come back in a columnar shape, and
normalizing them back to the original HotpotQA layout keeps every downstream
stage unaware that the source ever moved.

The API rate-limits bursts, so pages are requested with a small pause and a 429
is treated as "wait and try again" rather than as a failure. Assembling the split
is a one-off cost: after this, the corpus is frozen on disk and hashed.
"""

import time
from collections.abc import Callable
from functools import partial

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"

PageFetcher = Callable[[int, int], dict]

# The distractor validation split holds 7,405 questions. This ceiling is well
# clear of that and, unlike the loop below, finite: if the API stops reporting
# num_rows_total, an empty page is otherwise the only thing that ends paging.
MAX_ROWS = 20_000


class RateLimitError(Exception):
    """The API asked us to slow down (HTTP 429) or hiccuped server-side."""


def normalize_row(row: dict) -> dict:
    """Turn one API row into the original HotpotQA question shape."""
    context = row["context"]
    facts = row["supporting_facts"]
    return {
        "_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row.get("type", ""),
        "level": row.get("level", ""),
        "context": [
            [title, list(sentences)]
            for title, sentences in zip(context["title"], context["sentences"], strict=True)
        ],
        "supporting_facts": [
            [title, int(sent_id)]
            for title, sent_id in zip(facts["title"], facts["sent_id"], strict=True)
        ],
    }


def retry_call(
    call: Callable[[], dict],
    attempts: int = 6,
    sleep_fn: Callable[[float], None] = time.sleep,
    base_delay: float = 2.0,
):
    """Run `call`, backing off exponentially while it reports rate limiting."""
    for attempt in range(attempts):
        try:
            return call()
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            print(f"[WARN] rate limited, waiting {delay:.0f}s before retrying")
            sleep_fn(delay)
    raise RateLimitError("exhausted retries")


def _http_page_fetcher(offset: int, length: int, split: str = SPLIT) -> dict:
    import httpx

    response = httpx.get(
        ROWS_ENDPOINT,
        params={
            "dataset": DATASET,
            "config": CONFIG,
            "split": split,
            "offset": offset,
            "length": length,
        },
        timeout=120.0,
    )
    if response.status_code == 429 or response.status_code >= 500:
        raise RateLimitError(f"HTTP {response.status_code}")
    response.raise_for_status()
    return response.json()


def fetch_split(
    page_fetcher: PageFetcher | None = None,
    page_size: int = 100,
    pause: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_rows: int = MAX_ROWS,
    split: str = SPLIT,
) -> list[dict]:
    """Page through `split` and return every question, normalized.

    Phase 10 reads `train` through the same path; the default stays the validation split
    every earlier phase assembled.
    """
    page_fetcher = page_fetcher or partial(_http_page_fetcher, split=split)

    rows: list[dict] = []
    offset = 0
    total: int | None = None

    while True:
        payload = retry_call(partial(page_fetcher, offset, page_size), sleep_fn=sleep_fn)
        if total is None:
            total = int(payload.get("num_rows_total", 0))
            print(f"[INFO] split holds {total} questions")

        page = payload.get("rows", [])
        if not page:
            break

        rows.extend(normalize_row(entry["row"]) for entry in page)
        offset += len(page)
        if len(rows) > max_rows:
            raise RuntimeError(
                f"fetched {len(rows)} rows, past the {max_rows} ceiling; "
                "the API shape has probably changed"
            )
        if offset % 1000 == 0:
            print(f"[INFO] fetched {offset}/{total}")

        if total and offset >= total:
            break
        if pause:
            sleep_fn(pause)

    return rows
