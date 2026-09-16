"""Reading entities and concepts out of each paragraph, offline (Phase 5, HU-2).

The representation Phase 5 tests is the one the proposal describes: what a paragraph is
about, read from its text. A model reads every paragraph of the pool once, before any hop
runs, and returns two lists - entities and concepts - in a fixed structure. Nothing here
is ever called from a retrieval or an evaluation.

Three rules, each a decision of the phase plan:

- **A request is a pure function of one paragraph** (D3). `build_request` takes an
  `IndexingUnit` and the model name and nothing else, so a question, an answer, a gold
  annotation or a split cannot reach the extractor by construction.
- **An answer outside its bounds is a failure, never a repair** (D3). Structured outputs
  cannot express length limits, so they are checked here; unlike `label`, which truncates,
  nothing is cut to fit, because the spec forbids silent repair.
- **The cache is the record** (D5, spec restrictions). Current models reject `temperature`,
  so the same paragraph can come back with different nodes on another day. Every answer is
  cached per paragraph, bound to the model and the prompt digest it came from, and an entry
  that does not say so is refused rather than trusted.

The key is never touched by the request, the record or the cache. The two real clients at
the bottom of the module read it through `labeling.read_api_key`, and only the CLI builds
them: every test hands in a fake.
"""

import gzip
import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import IndexingUnit

OK = "ok"
FAILED = "failed"

# Why a record failed. `refusal` and `max_tokens` come from the stop reason; `not_json`,
# `schema` and `bounds` from reading the answer; `invalid_request` from the batch service
# rejecting the request; `transient` from a server-side failure that recurred on its one
# resubmission (D5). Every one is cached like a success, so none is paid for twice: a
# deliberate retry is deleting the entry, a decision a person makes.
FAILURE_REASONS: tuple[str, ...] = (
    "refusal",
    "max_tokens",
    "not_json",
    "schema",
    "bounds",
    "invalid_request",
    "transient",
)

# What the batch service can say about one request (D5).
SUCCEEDED = "succeeded"
INVALID = "invalid_request"
TRANSIENT = "transient"
# How the API names a rejected request inside an errored batch result: the documented
# shorthand and the wire name are both accepted as permanent.
INVALID_ERROR_TYPES: tuple[str, ...] = ("invalid_request", "invalid_request_error")

REPRODUCIBILITY_NOTE = (
    "Current models reject `temperature`, so extracting the same paragraph again may return "
    "different entities and concepts. This artifact, not a re-run, is the record of what the "
    "pilot measured; everything built on it is deterministic."
)

PROMPT_TEMPLATE = """Read the paragraph below and list what it is about, in two lists.

"entities": the specific named things the paragraph mentions or is about - people, \
organizations, places, works such as books, films, songs, albums or television series, \
events, and anything else with a proper name. Write each as the name itself, in its most \
complete common form, never a pronoun or a description.

"concepts": the general topics and notions the paragraph is about - kinds of things, \
fields, activities, roles and properties - that paragraphs about other subjects could \
share. A concept is never a proper name. Write each as a short noun phrase of one to three \
words, in its most common general wording and in the singular.

Use only what the paragraph itself says, without outside knowledge. Do not repeat an item \
within a list. List at most 50 items in each list, and return a list empty when nothing \
fits it.

Paragraph:
{paragraph}"""

PROMPT_DIGEST = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:16]

NODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific named things: people, organizations, places, works, events",
        },
        "concepts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "General topics and notions, never proper names",
        },
    },
    "required": ["entities", "concepts"],
    "additionalProperties": False,
}

# Everything a request carries besides the model and the paragraph. Thinking is disabled
# as in `label`: extraction needs no reasoning budget, and thinking is billed as output.
REQUEST_SHAPE: dict[str, Any] = {
    "thinking": {"type": "disabled"},
    "output_config": {"format": {"type": "json_schema", "schema": NODE_SCHEMA}},
    "max_tokens": config.EXTRACTION_MAX_TOKENS,
}

_CACHE_FIELDS = (
    "unit_id",
    "model",
    "prompt_digest",
    "status",
    "entities",
    "concepts",
    "failure",
    "input_tokens",
    "output_tokens",
)


class ExtractionError(Exception):
    """An extraction artifact or cache entry is not what it claims to be."""


@dataclass(frozen=True)
class RawResponse:
    """What a client hands back, stripped of the SDK: enough to judge the answer and its cost."""

    stop_reason: str
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ExtractionRecord:
    unit_id: str
    model: str
    prompt_digest: str
    status: str
    entities: tuple[str, ...]
    concepts: tuple[str, ...]
    failure: str | None
    input_tokens: int
    output_tokens: int


def render_prompt(unit: IndexingUnit) -> str:
    """The frozen template, filled with what both retrievers index: title and paragraph."""
    return PROMPT_TEMPLATE.format(paragraph=unit.indexable_text)


def build_request(unit: IndexingUnit, *, model: str = config.EXTRACTION_MODEL) -> dict[str, Any]:
    """One paragraph, one request. The signature is the guarantee HU-8 relies on."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": render_prompt(unit)}],
        **REQUEST_SHAPE,
    }


class _Rejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _validated_lists(answer: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The two lists, exactly as returned, or the reason they cannot be accepted."""
    if not isinstance(answer, dict) or set(answer) != {"entities", "concepts"}:
        raise _Rejected("schema")
    lists = []
    for field in ("entities", "concepts"):
        items = answer[field]
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise _Rejected("schema")
        if len(items) > config.MAX_NODES_PER_TYPE:
            raise _Rejected("bounds")
        if any(len(item) > config.MAX_NODE_CHARS for item in items):
            raise _Rejected("bounds")
        lists.append(tuple(items))
    return lists[0], lists[1]


def record_from_response(unit_id: str, response: RawResponse, *, model: str) -> ExtractionRecord:
    """Judge one answer. It is external input: nothing reaches a record unchecked."""

    def failed(reason: str) -> ExtractionRecord:
        return ExtractionRecord(
            unit_id=unit_id,
            model=model,
            prompt_digest=PROMPT_DIGEST,
            status=FAILED,
            entities=(),
            concepts=(),
            failure=reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    if response.stop_reason == "refusal":
        return failed("refusal")
    if response.stop_reason == "max_tokens":
        return failed("max_tokens")
    if response.stop_reason != "end_turn":
        # A request with no tools and no stop sequences has no other legitimate way to end;
        # whatever it returned is not the structured answer the schema asked for.
        return failed("schema")
    try:
        decoded = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return failed("not_json")
    try:
        entities, concepts = _validated_lists(decoded)
    except _Rejected as rejection:
        return failed(rejection.reason)

    return ExtractionRecord(
        unit_id=unit_id,
        model=model,
        prompt_digest=PROMPT_DIGEST,
        status=OK,
        entities=entities,
        concepts=concepts,
        failure=None,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def _cache_path(cache_dir: Path | str, unit_id: str, model: str, prompt_digest: str) -> Path:
    """One file per (model, prompt, paragraph). No key material ever enters the name."""
    key = hashlib.sha256(f"{model}\n{prompt_digest}\n{unit_id}".encode()).hexdigest()[:32]
    return Path(cache_dir) / "extraction-cache" / f"{key}.json"


def write_cached(cache_dir: Path | str, record: ExtractionRecord) -> Path:
    path = _cache_path(cache_dir, record.unit_id, record.model, record.prompt_digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit_id": record.unit_id,
        "model": record.model,
        "prompt_digest": record.prompt_digest,
        "status": record.status,
        "entities": list(record.entities),
        "concepts": list(record.concepts),
        "failure": record.failure,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
    }
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))
    return path


def read_cached(
    cache_dir: Path | str,
    unit_id: str,
    *,
    model: str,
    prompt_digest: str = PROMPT_DIGEST,
) -> ExtractionRecord | None:
    """The cached record for this paragraph under this model and prompt, or nothing.

    A lookup under another model or prompt finds another file, so an answer to a different
    question is never reused. An entry that exists but does not bind itself to the key it is
    filed under is refused: this cache fixes a non-reproducible result, and an edited entry
    would silently rewrite what the pilot measured.
    """
    path = _cache_path(cache_dir, unit_id, model, prompt_digest)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ExtractionError(
            f"the cached extraction in {path} is not readable JSON; delete that file to "
            "re-request it"
        ) from error
    if not isinstance(payload, dict) or set(payload) != set(_CACHE_FIELDS):
        raise ExtractionError(
            f"the cached extraction in {path} is not an extraction entry; delete that file to "
            "re-request it"
        )
    if (payload["unit_id"], payload["model"], payload["prompt_digest"]) != (
        unit_id,
        model,
        prompt_digest,
    ):
        raise ExtractionError(
            f"the cached extraction in {path} claims another paragraph, model or prompt than "
            "the one it is filed under; delete that file to re-request it"
        )
    status = payload["status"]
    failure = payload["failure"]
    if status not in (OK, FAILED) or (status == FAILED) != (failure in FAILURE_REASONS):
        raise ExtractionError(
            f"the cached extraction in {path} has an inconsistent status; delete that file to "
            "re-request it"
        )
    try:
        entities, concepts = _validated_lists(
            {"entities": payload["entities"], "concepts": payload["concepts"]}
        )
    except _Rejected as rejection:
        raise ExtractionError(
            f"the cached extraction in {path} fails its own bounds ({rejection.reason}); delete "
            "that file to re-request it"
        ) from rejection
    return ExtractionRecord(
        unit_id=unit_id,
        model=model,
        prompt_digest=prompt_digest,
        status=status,
        entities=entities,
        concepts=concepts,
        failure=failure,
        input_tokens=int(payload["input_tokens"]),
        output_tokens=int(payload["output_tokens"]),
    )


# --- Running it: the sample, the estimate, the batches (T6, decisions D4 and D5) --------


class SyncClient(Protocol):
    """Sends one request and waits for the answer. Used for the sample only."""

    model: str

    def extract(self, request: dict[str, Any]) -> RawResponse: ...


@dataclass(frozen=True)
class BatchOutcome:
    """The batch service's verdict on one request, stripped of the SDK."""

    unit_id: str
    kind: str
    response: RawResponse | None


class BatchClient(Protocol):
    """Submits requests as a batch, reports its status, and yields its outcomes."""

    model: str

    def submit(self, requests: list[tuple[str, dict[str, Any]]]) -> str: ...

    def status(self, batch_id: str) -> str: ...

    def results(self, batch_id: str) -> Iterable[BatchOutcome]: ...


SAMPLE_REPORT_FILENAME = "sample-report.json"
BATCH_ENDED = "ended"


def _extraction_stem(prompt_digest: str = PROMPT_DIGEST) -> str:
    return f"extraction-{prompt_digest}"


def sample_units(units: Sequence[IndexingUnit], size: int) -> list[IndexingUnit]:
    """D4's rule: the `size` paragraphs with the lowest unit ids. It knows no question."""
    return sorted(units, key=lambda unit: unit.unit_id)[:size]


def _length_stats(values: Sequence[int]) -> dict[str, float]:
    if not values:
        raise ExtractionError("no paragraph length to summarise")
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def estimate_full_cost(
    *,
    n_paragraphs: int,
    mean_input_tokens: float,
    mean_output_tokens: float,
    margin: float = config.EXTRACTION_ESTIMATE_MARGIN,
) -> float:
    """D4: input as the sample measured it, output with the margin, at batch rates."""
    input_usd = n_paragraphs * mean_input_tokens / 1e6 * config.EXTRACTION_INPUT_USD_PER_MTOK
    output_usd = (
        n_paragraphs * mean_output_tokens * margin / 1e6 * config.EXTRACTION_OUTPUT_USD_PER_MTOK
    )
    return (input_usd + output_usd) * config.BATCH_PRICE_FACTOR


def _standard_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1e6 * config.EXTRACTION_INPUT_USD_PER_MTOK
        + output_tokens / 1e6 * config.EXTRACTION_OUTPUT_USD_PER_MTOK
    )


@dataclass(frozen=True)
class SampleReport:
    model: str
    prompt_digest: str
    unit_ids: tuple[str, ...]
    statuses: dict[str, str]
    n_ok: int
    mean_input_tokens: float
    mean_output_tokens: float
    sample_lengths: dict[str, float]
    corpus_lengths: dict[str, float]
    n_pool: int
    estimated_full_usd: float
    ceiling_usd: float
    sample_usd: float


def run_sample(
    units: Sequence[IndexingUnit],
    client: SyncClient,
    *,
    cache_dir: Path | str,
    extraction_dir: Path | str,
    token_counts: Mapping[str, int],
    size: int = config.EXTRACTION_SAMPLE_SIZE,
    ceiling_usd: float = config.EXTRACTION_COST_CEILING_USD,
) -> SampleReport:
    """Extract the sample synchronously, with the full run's exact request, and report.

    The records are cached under the full run's keys, so the batches never pay for them
    again. The report is what the author reads before approving the spend (D4, D14).
    """
    chosen = sample_units(units, size)
    records: list[ExtractionRecord] = []
    for unit in chosen:
        record = read_cached(cache_dir, unit.unit_id, model=client.model)
        if record is None:
            response = client.extract(build_request(unit, model=client.model))
            record = record_from_response(unit.unit_id, response, model=client.model)
            write_cached(cache_dir, record)
        records.append(record)

    n_uncached = sum(
        1 for unit in units if read_cached(cache_dir, unit.unit_id, model=client.model) is None
    )
    mean_input = statistics.fmean(record.input_tokens for record in records)
    mean_output = statistics.fmean(record.output_tokens for record in records)
    report = SampleReport(
        model=client.model,
        prompt_digest=PROMPT_DIGEST,
        unit_ids=tuple(unit.unit_id for unit in chosen),
        statuses={
            record.unit_id: record.status if record.failure is None else record.failure
            for record in records
        },
        n_ok=sum(1 for record in records if record.status == OK),
        mean_input_tokens=float(mean_input),
        mean_output_tokens=float(mean_output),
        sample_lengths=_length_stats([token_counts[unit.unit_id] for unit in chosen]),
        corpus_lengths=_length_stats([token_counts[unit.unit_id] for unit in units]),
        n_pool=len(units),
        estimated_full_usd=estimate_full_cost(
            n_paragraphs=n_uncached, mean_input_tokens=mean_input, mean_output_tokens=mean_output
        ),
        ceiling_usd=ceiling_usd,
        sample_usd=sum(_standard_cost(r.input_tokens, r.output_tokens) for r in records),
    )
    _save_sample_report(report, extraction_dir)
    return report


def _save_sample_report(report: SampleReport, extraction_dir: Path | str) -> Path:
    path = Path(extraction_dir) / SAMPLE_REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": report.model,
        "prompt_digest": report.prompt_digest,
        "unit_ids": list(report.unit_ids),
        "statuses": report.statuses,
        "n_ok": report.n_ok,
        "mean_input_tokens": report.mean_input_tokens,
        "mean_output_tokens": report.mean_output_tokens,
        "sample_lengths": report.sample_lengths,
        "corpus_lengths": report.corpus_lengths,
        "n_pool": report.n_pool,
        "estimated_full_usd": report.estimated_full_usd,
        "ceiling_usd": report.ceiling_usd,
        "sample_usd": report.sample_usd,
        "length_unit": "tokens of the project tokenizer (token_counts.json)",
    }
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_sample_report(extraction_dir: Path | str) -> SampleReport:
    path = Path(extraction_dir) / SAMPLE_REPORT_FILENAME
    if not path.exists():
        raise ExtractionError(
            f"no sample report in {extraction_dir}: the full extraction needs the sample first "
            "(`cer extract --sample`), because its estimate is taken from what the sample measured"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SampleReport(
        model=str(payload["model"]),
        prompt_digest=str(payload["prompt_digest"]),
        unit_ids=tuple(payload["unit_ids"]),
        statuses=dict(payload["statuses"]),
        n_ok=int(payload["n_ok"]),
        mean_input_tokens=float(payload["mean_input_tokens"]),
        mean_output_tokens=float(payload["mean_output_tokens"]),
        sample_lengths=dict(payload["sample_lengths"]),
        corpus_lengths=dict(payload["corpus_lengths"]),
        n_pool=int(payload["n_pool"]),
        estimated_full_usd=float(payload["estimated_full_usd"]),
        ceiling_usd=float(payload["ceiling_usd"]),
        sample_usd=float(payload["sample_usd"]),
    )


@dataclass(frozen=True)
class ExtractionSummary:
    model: str
    prompt_digest: str
    n_units: int
    calls: int
    cached: int
    resubmissions: int
    failures: dict[str, int]
    failure_rate: float
    finding: bool
    input_tokens: int
    output_tokens: int
    estimated_usd: float
    actual_usd: float
    digest: str


def _state_path(extraction_dir: Path | str) -> Path:
    return Path(extraction_dir) / f"batches-{PROMPT_DIGEST}.json"


def _load_state(extraction_dir: Path | str, model: str) -> list[dict[str, Any]]:
    path = _state_path(extraction_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != model or payload.get("prompt_digest") != PROMPT_DIGEST:
        raise ExtractionError(
            f"the batch state in {path} belongs to another model or prompt; move it aside "
            "deliberately before starting a different extraction"
        )
    return list(payload["batches"])


def _save_state(extraction_dir: Path | str, model: str, batches: list[dict[str, Any]]) -> None:
    path = _state_path(extraction_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "prompt_digest": PROMPT_DIGEST, "batches": batches}
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def run_full(
    units: Sequence[IndexingUnit],
    client: BatchClient,
    *,
    cache_dir: Path | str,
    extraction_dir: Path | str,
    ceiling_usd: float = config.EXTRACTION_COST_CEILING_USD,
    batch_size: int = config.EXTRACTION_BATCH_SIZE,
    wait: Callable[[], None],
) -> ExtractionSummary:
    """Extract every paragraph not yet cached, through batches, and write the aggregate.

    In order, and each step before the next can spend anything: the sample must exist; the
    estimate for what is not yet committed must be under the ceiling; every batch id is
    written to disk before it is polled, so an interrupted run resumes instead of paying
    twice; a server-side failure is resubmitted once and recorded if it recurs.
    """
    sample = load_sample_report(extraction_dir)
    if sample.model != client.model or sample.prompt_digest != PROMPT_DIGEST:
        raise ExtractionError(
            "the sample was taken with another model or prompt; take a new sample first"
        )
    unit_by_id = {unit.unit_id: unit for unit in units}
    state = _load_state(extraction_dir, client.model)

    def cached(unit_id: str) -> ExtractionRecord | None:
        return read_cached(cache_dir, unit_id, model=client.model)

    initially_cached = sum(1 for unit_id in unit_by_id if cached(unit_id) is not None)

    def passes_of(unit_id: str) -> int:
        return max((entry["pass"] for entry in state if unit_id in entry["unit_ids"]), default=0)

    uncommitted = [
        unit_id for unit_id in unit_by_id if cached(unit_id) is None and passes_of(unit_id) == 0
    ]
    estimate = estimate_full_cost(
        n_paragraphs=len(uncommitted),
        mean_input_tokens=sample.mean_input_tokens,
        mean_output_tokens=sample.mean_output_tokens,
    )
    if estimate > ceiling_usd:
        raise ExtractionError(
            f"the estimate for {len(uncommitted)} paragraphs is {estimate:.2f} USD, above the "
            f"ceiling of {ceiling_usd:.2f} USD; nothing was submitted"
        )

    calls = 0
    resubmissions = 0

    def process(entry: dict[str, Any]) -> None:
        while client.status(entry["batch_id"]) != BATCH_ENDED:
            wait()
        seen: set[str] = set()
        for outcome in client.results(entry["batch_id"]):
            seen.add(outcome.unit_id)
            if outcome.kind == SUCCEEDED and outcome.response is not None:
                record = record_from_response(outcome.unit_id, outcome.response, model=client.model)
            elif outcome.kind == INVALID:
                record = _failure(outcome.unit_id, "invalid_request", client.model)
            elif entry["pass"] >= 2:
                record = _failure(outcome.unit_id, "transient", client.model)
            else:
                continue
            write_cached(cache_dir, record)
        for unit_id in entry["unit_ids"]:
            if unit_id not in seen and entry["pass"] >= 2 and cached(unit_id) is None:
                write_cached(cache_dir, _failure(unit_id, "transient", client.model))

    def submit(unit_ids: list[str], pass_number: int) -> None:
        nonlocal calls
        for start in range(0, len(unit_ids), batch_size):
            chunk = unit_ids[start : start + batch_size]
            requests = [
                (unit_id, build_request(unit_by_id[unit_id], model=client.model))
                for unit_id in chunk
            ]
            batch_id = client.submit(requests)
            calls += len(chunk)
            entry = {"batch_id": batch_id, "pass": pass_number, "unit_ids": chunk}
            state.append(entry)
            _save_state(extraction_dir, client.model, state)
            process(entry)

    for entry in list(state):
        if any(cached(unit_id) is None for unit_id in entry["unit_ids"]):
            process(entry)

    submit([unit_id for unit_id in uncommitted if cached(unit_id) is None], 1)

    retry = [
        unit_id for unit_id in unit_by_id if cached(unit_id) is None and passes_of(unit_id) == 1
    ]
    resubmissions += len(retry)
    submit(retry, 2)

    return _write_aggregate(
        units,
        cache_dir=cache_dir,
        extraction_dir=extraction_dir,
        model=client.model,
        sample=sample,
        calls=calls,
        cached=initially_cached,
        resubmissions=resubmissions,
        estimated_usd=estimate,
    )


def _failure(unit_id: str, reason: str, model: str) -> ExtractionRecord:
    return ExtractionRecord(
        unit_id=unit_id,
        model=model,
        prompt_digest=PROMPT_DIGEST,
        status=FAILED,
        entities=(),
        concepts=(),
        failure=reason,
        input_tokens=0,
        output_tokens=0,
    )


def _record_line(record: ExtractionRecord) -> str:
    return json.dumps(
        {
            "unit_id": record.unit_id,
            "status": record.status,
            "entities": list(record.entities),
            "concepts": list(record.concepts),
            "failure": record.failure,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def _write_aggregate(
    units: Sequence[IndexingUnit],
    *,
    cache_dir: Path | str,
    extraction_dir: Path | str,
    model: str,
    sample: SampleReport,
    calls: int,
    cached: int,
    resubmissions: int,
    estimated_usd: float,
) -> ExtractionSummary:
    records = []
    for unit in sorted(units, key=lambda u: u.unit_id):
        record = read_cached(cache_dir, unit.unit_id, model=model)
        if record is None:
            raise ExtractionError(
                f"paragraph {unit.unit_id} has no extraction record after the run; the batches "
                "did not return it, and it is not silently left out"
            )
        records.append(record)

    text = "".join(_record_line(record) + "\n" for record in records)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    directory = Path(extraction_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = _extraction_stem()
    archive = directory / f"{stem}.jsonl.gz"
    temporary = archive.with_name(archive.name + ".tmp")
    # mtime=0 so that the same records always compress to the same bytes.
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(text.encode("utf-8"))
    temporary.replace(archive)

    failures = Counter(record.failure for record in records if record.status == FAILED)
    sample_ids = set(sample.unit_ids)
    batch_input = sum(r.input_tokens for r in records if r.unit_id not in sample_ids)
    batch_output = sum(r.output_tokens for r in records if r.unit_id not in sample_ids)
    actual_usd = sample.sample_usd + _standard_cost(batch_input, batch_output) * (
        config.BATCH_PRICE_FACTOR
    )
    failure_rate = sum(failures.values()) / len(records)
    summary = ExtractionSummary(
        model=model,
        prompt_digest=PROMPT_DIGEST,
        n_units=len(records),
        calls=calls,
        cached=cached,
        resubmissions=resubmissions,
        failures={str(reason): count for reason, count in sorted(failures.items())},
        failure_rate=failure_rate,
        finding=failure_rate > config.EXTRACTION_FAILURE_FINDING,
        input_tokens=sum(r.input_tokens for r in records),
        output_tokens=sum(r.output_tokens for r in records),
        estimated_usd=estimated_usd,
        actual_usd=actual_usd,
        digest=digest,
    )
    write_text_atomic(
        directory / f"{stem}.json",
        json.dumps(
            {
                **{field: getattr(summary, field) for field in summary.__dataclass_fields__},
                "archive": archive.name,
                "reproducibility_note": REPRODUCIBILITY_NOTE,
                "rates_usd_per_mtok": {
                    "input": config.EXTRACTION_INPUT_USD_PER_MTOK,
                    "output": config.EXTRACTION_OUTPUT_USD_PER_MTOK,
                    "batch_factor": config.BATCH_PRICE_FACTOR,
                },
            },
            indent=2,
            sort_keys=True,
        ),
    )
    return summary


def load_extraction_summary(
    extraction_dir: Path | str, *, prompt_digest: str = PROMPT_DIGEST
) -> dict[str, Any]:
    """The summary of the aggregate: counts, costs, failures, and the digest of the archive."""
    path = Path(extraction_dir) / f"{_extraction_stem(prompt_digest)}.json"
    if not path.exists():
        raise ExtractionError(f"no extraction in {extraction_dir}: run `cer extract` first")
    summary: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return summary


def load_extraction(
    extraction_dir: Path | str,
    *,
    expected_unit_ids: Sequence[str] | None = None,
    prompt_digest: str = PROMPT_DIGEST,
) -> dict[str, ExtractionRecord]:
    """Read the aggregate back, verified against its digest and, if asked, the whole pool."""
    directory = Path(extraction_dir)
    stem = _extraction_stem(prompt_digest)
    archive, summary_path = directory / f"{stem}.jsonl.gz", directory / f"{stem}.json"
    if not archive.exists() or not summary_path.exists():
        raise ExtractionError(f"no extraction in {directory}: run `cer extract` first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    text = gzip.decompress(archive.read_bytes()).decode("utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != summary["digest"]:
        raise ExtractionError(
            f"{archive.name} does not match the digest its summary records; it has been modified"
        )
    model = str(summary["model"])
    records: dict[str, ExtractionRecord] = {}
    for line in text.splitlines():
        entry = json.loads(line)
        records[entry["unit_id"]] = ExtractionRecord(
            unit_id=str(entry["unit_id"]),
            model=model,
            prompt_digest=prompt_digest,
            status=str(entry["status"]),
            entities=tuple(entry["entities"]),
            concepts=tuple(entry["concepts"]),
            failure=entry["failure"],
            input_tokens=int(entry["input_tokens"]),
            output_tokens=int(entry["output_tokens"]),
        )
    if expected_unit_ids is not None:
        missing = set(expected_unit_ids) - set(records)
        if missing:
            raise ExtractionError(
                f"the extraction does not cover {len(missing)} paragraph(s) of the pool; every "
                "unit needs a record, ok or failed, before nodes can be built"
            )
    return records


# --- The real clients: constructed only by the CLI, never by a test ---------------------


def _sdk() -> Any:
    try:
        import anthropic
    except ImportError as error:
        raise ExtractionError(
            "the extraction stage needs the optional dependency group: "
            "run 'uv sync --group labeling'"
        ) from error
    return anthropic


def _raw_response(message: Any) -> RawResponse:
    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
    return RawResponse(
        stop_reason=str(message.stop_reason),
        text=text,
        input_tokens=int(message.usage.input_tokens),
        output_tokens=int(message.usage.output_tokens),
    )


class AnthropicSyncClient:
    """The sample's client. The key is read by `labeling.read_api_key` and never kept in view."""

    def __init__(self, *, model: str = config.EXTRACTION_MODEL) -> None:
        from concept_embeddings_rag.concepts.labeling import MAX_RETRIES, read_api_key

        anthropic = _sdk()
        self.model = model
        self._client = anthropic.Anthropic(api_key=read_api_key(), max_retries=MAX_RETRIES)

    def extract(self, request: dict[str, Any]) -> RawResponse:
        return _raw_response(self._client.messages.create(**request))


class AnthropicBatchClient:
    """The full run's client, over the Message Batches API at half price."""

    def __init__(self, *, model: str = config.EXTRACTION_MODEL) -> None:
        from concept_embeddings_rag.concepts.labeling import MAX_RETRIES, read_api_key

        anthropic = _sdk()
        self.model = model
        self._client = anthropic.Anthropic(api_key=read_api_key(), max_retries=MAX_RETRIES)

    def submit(self, requests: list[tuple[str, dict[str, Any]]]) -> str:
        batch = self._client.messages.batches.create(
            requests=[{"custom_id": unit_id, "params": params} for unit_id, params in requests]
        )
        return str(batch.id)

    def status(self, batch_id: str) -> str:
        return str(self._client.messages.batches.retrieve(batch_id).processing_status)

    def results(self, batch_id: str) -> Iterable[BatchOutcome]:
        for result in self._client.messages.batches.results(batch_id):
            kind = getattr(result.result, "type", None)
            if kind == "succeeded":
                yield BatchOutcome(
                    unit_id=str(result.custom_id),
                    kind=SUCCEEDED,
                    response=_raw_response(result.result.message),
                )
                continue
            # An errored result wraps the API error, whose own `type` names the kind; the
            # documented shorthand and the wire name of an invalid request are both accepted.
            # Everything else - server errors, expired and cancelled requests - is transient.
            error = getattr(result.result, "error", None)
            error_type = getattr(error, "type", None)
            if error_type == "error":
                error_type = getattr(getattr(error, "error", None), "type", None)
            invalid = kind == "errored" and error_type in INVALID_ERROR_TYPES
            yield BatchOutcome(
                unit_id=str(result.custom_id),
                kind=INVALID if invalid else TRANSIENT,
                response=None,
            )
