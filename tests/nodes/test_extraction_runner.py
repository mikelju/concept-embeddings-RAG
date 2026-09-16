"""T6 of Phase 5: the sample, the batches, the ceiling and the resume (HU-2, D4, D5).

Every client here is a fake. No test in this module can reach the network: the real
clients are only constructed by the CLI when no client is handed in, and every CLI test
below hands one in.
"""

import json
from pathlib import Path

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question, save_pool, unit_id_for
from concept_embeddings_rag.nodes import extraction
from concept_embeddings_rag.nodes.extraction import (
    FAILED,
    INVALID,
    OK,
    SUCCEEDED,
    TRANSIENT,
    BatchOutcome,
    ExtractionError,
    RawResponse,
    estimate_full_cost,
    load_extraction,
    load_sample_report,
    read_cached,
    run_full,
    run_sample,
    sample_units,
)

MODEL = config.EXTRACTION_MODEL


def the_units(n: int = 30) -> list[IndexingUnit]:
    units = []
    for index in range(n):
        title = f"Article {index}"
        sentences = (f"Paragraph number {index} says something.",)
        units.append(
            IndexingUnit(unit_id=unit_id_for(title, sentences), title=title, sentences=sentences)
        )
    return units


def token_counts(units: list[IndexingUnit]) -> dict[str, int]:
    return {unit.unit_id: 50 + index for index, unit in enumerate(units)}


def a_good_answer(input_tokens: int = 150, output_tokens: int = 30) -> RawResponse:
    text = json.dumps({"entities": ["Some Person"], "concepts": ["river"]})
    return RawResponse("end_turn", text, input_tokens, output_tokens)


class FakeSyncClient:
    model = MODEL

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def extract(self, request: dict) -> RawResponse:
        self.requests.append(request)
        return a_good_answer()


class FakeBatchClient:
    """A batch service in memory. `script[unit_id]` lists the outcome of each submission."""

    model = MODEL

    def __init__(self, script: dict[str, list[str]] | None = None, store: dict | None = None):
        self.script = {unit: list(kinds) for unit, kinds in (script or {}).items()}
        self.batches: dict[str, list[str]] = store if store is not None else {}
        self.submissions: list[list[str]] = []
        self.polls = 0
        self.fail_on_status = False

    def submit(self, requests: list[tuple[str, dict]]) -> str:
        unit_ids = [unit_id for unit_id, _ in requests]
        batch_id = f"batch-{len(self.batches)}"
        self.batches[batch_id] = unit_ids
        self.submissions.append(unit_ids)
        return batch_id

    def status(self, batch_id: str) -> str:
        self.polls += 1
        if self.fail_on_status:
            raise ConnectionError("the process died while polling")
        return "ended"

    def results(self, batch_id: str):
        for unit_id in self.batches[batch_id]:
            kinds = self.script.get(unit_id, [])
            kind = kinds.pop(0) if kinds else SUCCEEDED
            response = a_good_answer(200, 40) if kind == SUCCEEDED else None
            yield BatchOutcome(unit_id=unit_id, kind=kind, response=response)


def no_wait() -> None:
    return None


def sampled(tmp_path: Path, units: list[IndexingUnit], size: int = 5):
    return run_sample(
        units,
        FakeSyncClient(),
        cache_dir=tmp_path / "cache",
        extraction_dir=tmp_path / "extraction",
        token_counts=token_counts(units),
        size=size,
    )


def full(tmp_path: Path, units: list[IndexingUnit], client: FakeBatchClient, **overrides):
    arguments = {
        "cache_dir": tmp_path / "cache",
        "extraction_dir": tmp_path / "extraction",
        "ceiling_usd": 100.0,
        "batch_size": 7,
        "wait": no_wait,
    }
    arguments.update(overrides)
    return run_full(units, client, **arguments)


# --- The sample ------------------------------------------------------------------------


def test_the_sample_is_the_lowest_unit_ids_whatever_the_pool_order():
    units = the_units()
    lowest = sorted(unit.unit_id for unit in units)[:5]

    assert [unit.unit_id for unit in sample_units(list(reversed(units)), 5)] == lowest


def test_the_sample_caches_its_paragraphs_under_the_full_runs_keys(tmp_path: Path):
    units = the_units()
    client = FakeSyncClient()

    run_sample(
        units,
        client,
        cache_dir=tmp_path / "cache",
        extraction_dir=tmp_path / "extraction",
        token_counts=token_counts(units),
        size=5,
    )

    sample_ids = sorted(unit.unit_id for unit in units)[:5]
    assert [request["messages"][0]["content"] for request in client.requests] == [
        extraction.render_prompt(unit) for unit in sample_units(units, 5)
    ]
    for unit_id in sample_ids:
        assert read_cached(tmp_path / "cache", unit_id, model=MODEL) is not None

    batch = FakeBatchClient()
    full(tmp_path, units, batch)
    submitted = {unit_id for submission in batch.submissions for unit_id in submission}
    assert submitted.isdisjoint(sample_ids)
    assert len(submitted) == len(units) - 5


def test_the_sample_report_compares_its_lengths_with_the_corpus(tmp_path: Path):
    units = the_units()
    report = sampled(tmp_path, units)

    loaded = load_sample_report(tmp_path / "extraction")

    assert loaded == report
    assert report.n_ok == 5
    assert report.mean_output_tokens == 30
    counts = token_counts(units)
    assert report.corpus_lengths["max"] == max(counts.values())
    sample_counts = [counts[unit.unit_id] for unit in sample_units(units, 5)]
    assert report.sample_lengths["mean"] == pytest.approx(sum(sample_counts) / 5)
    assert set(report.sample_lengths) == {"mean", "median", "min", "max"}


# --- Money -----------------------------------------------------------------------------


def test_the_estimate_pays_input_as_measured_and_output_with_the_margin_at_batch_rates():
    usd = estimate_full_cost(n_paragraphs=1000, mean_input_tokens=400, mean_output_tokens=100)

    input_usd = 1000 * 400 / 1e6 * config.EXTRACTION_INPUT_USD_PER_MTOK
    output_usd = 1000 * 100 * config.EXTRACTION_ESTIMATE_MARGIN / 1e6
    output_usd *= config.EXTRACTION_OUTPUT_USD_PER_MTOK
    assert usd == pytest.approx((input_usd + output_usd) * config.BATCH_PRICE_FACTOR)


def test_the_full_run_refuses_without_a_sample_before_any_request(tmp_path: Path):
    batch = FakeBatchClient()

    with pytest.raises(ExtractionError, match="sample"):
        full(tmp_path, the_units(), batch)

    assert batch.submissions == [] and batch.polls == 0


def test_the_full_run_refuses_above_the_ceiling_before_any_request(tmp_path: Path):
    units = the_units()
    sampled(tmp_path, units)
    batch = FakeBatchClient()

    with pytest.raises(ExtractionError, match="ceiling"):
        full(tmp_path, units, batch, ceiling_usd=0.0001)

    assert batch.submissions == [] and batch.polls == 0


# --- Batches ---------------------------------------------------------------------------


def test_paragraphs_already_cached_are_never_resubmitted(tmp_path: Path):
    units = the_units()
    sampled(tmp_path, units)
    full(tmp_path, units, FakeBatchClient())

    again = FakeBatchClient()
    summary = full(tmp_path, units, again)

    assert again.submissions == []
    assert summary.calls == 0
    assert summary.cached == len(units)


def test_batches_are_submitted_in_declared_chunks(tmp_path: Path):
    units = the_units()
    sampled(tmp_path, units)
    batch = FakeBatchClient()

    full(tmp_path, units, batch, batch_size=7)

    assert [len(submission) for submission in batch.submissions] == [7, 7, 7, 4]


def test_persisted_batch_ids_resume_without_resubmitting(tmp_path: Path):
    units = the_units()
    sampled(tmp_path, units)
    store: dict[str, list[str]] = {}
    dying = FakeBatchClient(store=store)
    dying.fail_on_status = True

    with pytest.raises(ConnectionError):
        full(tmp_path, units, dying, batch_size=100)
    assert len(dying.submissions) == 1

    resumed = FakeBatchClient(store=store)
    summary = full(tmp_path, units, resumed, batch_size=100)

    assert resumed.submissions == []
    assert summary.failures == {}
    for unit in units:
        assert read_cached(tmp_path / "cache", unit.unit_id, model=MODEL).status == OK


def test_a_transient_failure_is_resubmitted_exactly_once(tmp_path: Path):
    units = the_units()
    sampled(tmp_path, units)
    sample_ids = {unit.unit_id for unit in sample_units(units, 5)}
    rest = [unit.unit_id for unit in units if unit.unit_id not in sample_ids]
    recovers, stays_down = rest[0], rest[1]
    batch = FakeBatchClient({recovers: [TRANSIENT, SUCCEEDED], stays_down: [TRANSIENT, TRANSIENT]})

    summary = full(tmp_path, units, batch)

    submitted = [unit_id for submission in batch.submissions for unit_id in submission]
    assert submitted.count(recovers) == 2
    assert submitted.count(stays_down) == 2
    assert read_cached(tmp_path / "cache", recovers, model=MODEL).status == OK
    down = read_cached(tmp_path / "cache", stays_down, model=MODEL)
    assert (down.status, down.failure) == (FAILED, "transient")
    assert summary.resubmissions == 2
    assert summary.failures == {"transient": 1}


def test_an_invalid_request_is_a_failure_and_is_not_resubmitted(tmp_path: Path):
    units = the_units()
    sampled(tmp_path, units)
    sample_ids = {unit.unit_id for unit in sample_units(units, 5)}
    target = next(unit.unit_id for unit in units if unit.unit_id not in sample_ids)
    batch = FakeBatchClient({target: [INVALID]})

    summary = full(tmp_path, units, batch)

    submitted = [unit_id for submission in batch.submissions for unit_id in submission]
    assert submitted.count(target) == 1
    assert read_cached(tmp_path / "cache", target, model=MODEL).failure == "invalid_request"
    assert summary.failures == {"invalid_request": 1}


def test_failures_above_one_percent_are_a_finding(tmp_path: Path):
    units = the_units(n=30)
    sampled(tmp_path, units)
    sample_ids = {unit.unit_id for unit in sample_units(units, 5)}
    target = next(unit.unit_id for unit in units if unit.unit_id not in sample_ids)

    summary = full(tmp_path, units, FakeBatchClient({target: [INVALID]}))

    assert summary.failure_rate == pytest.approx(1 / 30)
    assert summary.finding is True


def test_no_failure_is_no_finding(tmp_path: Path):
    units = the_units()
    sampled(tmp_path, units)

    summary = full(tmp_path, units, FakeBatchClient())

    assert summary.failure_rate == 0.0
    assert summary.finding is False


# --- The aggregate artifact ------------------------------------------------------------


def test_the_extraction_artifact_round_trips_and_covers_the_pool(tmp_path: Path):
    units = the_units()
    sampled(tmp_path, units)
    summary = full(tmp_path, units, FakeBatchClient())

    records = load_extraction(
        tmp_path / "extraction", expected_unit_ids=[unit.unit_id for unit in units]
    )

    assert set(records) == {unit.unit_id for unit in units}
    assert all(record.status == OK for record in records.values())
    assert summary.n_units == len(units)
    assert summary.actual_usd > 0


def test_an_extraction_that_misses_a_pool_unit_is_refused(tmp_path: Path):
    units = the_units()
    sampled(tmp_path, units)
    full(tmp_path, units, FakeBatchClient())

    with pytest.raises(ExtractionError, match="cover"):
        load_extraction(
            tmp_path / "extraction",
            expected_unit_ids=[unit.unit_id for unit in units] + ["not-extracted"],
        )


def test_a_modified_extraction_artifact_is_refused(tmp_path: Path):
    import gzip

    units = the_units()
    sampled(tmp_path, units)
    full(tmp_path, units, FakeBatchClient())
    archive = next((tmp_path / "extraction").glob("extraction-*.jsonl.gz"))
    lines = gzip.decompress(archive.read_bytes()).decode("utf-8").splitlines()
    lines[0] = lines[0].replace("river", "lake")
    archive.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))

    with pytest.raises(ExtractionError, match="digest"):
        load_extraction(tmp_path / "extraction")


# --- The CLI ---------------------------------------------------------------------------


def a_workspace(tmp_path: Path, units: list[IndexingUnit]) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    question = Question(
        qid="q0",
        question="Unused here?",
        answer="-",
        gold_unit_ids=(units[0].unit_id,),
        supporting_facts=((units[0].title, 0),),
        split="dev",
    )
    save_pool(units, [question], data_dir / "pool.json")
    (data_dir / "token_counts.json").write_text(json.dumps(token_counts(units)), encoding="utf-8")
    return data_dir


def extract(tmp_path: Path, data_dir: Path, **overrides):
    from concept_embeddings_rag.cli import cmd_extract

    arguments = {
        "data_dir": data_dir,
        "cache_dir": tmp_path / "cache",
        "extraction_dir": tmp_path / "extraction",
        "wait": no_wait,
    }
    arguments.update(overrides)
    return cmd_extract(**arguments)


def test_the_sample_stage_prints_its_estimate_and_lengths_in_ascii(tmp_path: Path, capsys):
    units = the_units()
    data_dir = a_workspace(tmp_path, units)

    extract(tmp_path, data_dir, sample=True, client=FakeSyncClient())

    out = capsys.readouterr().out
    assert out.isascii()
    assert "estimate" in out
    assert "ceiling" in out
    assert "corpus" in out


def test_the_full_stage_without_a_sample_names_the_sample_stage(tmp_path: Path):
    units = the_units()
    data_dir = a_workspace(tmp_path, units)

    with pytest.raises(SystemExit) as excinfo:
        extract(tmp_path, data_dir, batch_client=FakeBatchClient())

    assert "--sample" in str(excinfo.value)


def test_the_full_stage_warns_when_failures_exceed_one_percent(tmp_path: Path, capsys):
    units = the_units()
    data_dir = a_workspace(tmp_path, units)
    extract(tmp_path, data_dir, sample=True, client=FakeSyncClient())
    sample_ids = {unit.unit_id for unit in sample_units(units, config.EXTRACTION_SAMPLE_SIZE)}
    target = next(unit.unit_id for unit in units if unit.unit_id not in sample_ids)

    extract(tmp_path, data_dir, batch_client=FakeBatchClient({target: [INVALID]}))

    out = capsys.readouterr().out
    assert "[WARN]" in out
    assert "invalid_request" in out
    assert out.isascii()
