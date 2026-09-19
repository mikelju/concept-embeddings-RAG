"""Phase 8: the embedding artifact, the dev pass and its gate, and the held-out run.

The dev and held-out passes run real fusion fitting, a real entity hop and the real
harness over a small bridge corpus; the provenance checks and the stop rule are driven
with values chosen so a bar can be landed on exactly rather than approached by luck.

No model is loaded and no network is reached: the width guard is exercised against a
stub backend, and every pin is overridden through `Phase8Pins`, which is what that
dataclass exists for.
"""

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question, load_pool
from concept_embeddings_rag.embeddings.cache import (
    EmbeddingCache,
    cache_key,
    embed_units,
    question_cache_key,
    question_set_hash,
    unit_set_hash,
)
from concept_embeddings_rag.evaluation.selection import SelectionError, digest_of_payload
from concept_embeddings_rag.evaluation.strong_dense import (
    DEV_FILENAME,
    EMBEDDING_FILENAME,
    EMBEDDING_TEST_FILENAME,
    TEST_FILENAME,
    VERDICT_CONTINUE,
    VERDICT_STOP,
    Phase8Pins,
    StrongDenseError,
    WidthCheckedBackend,
    check_model_pin,
    load_dev,
    measure_dev,
    measure_held_out,
    stop_rule,
    stop_rule_bar,
    write_embedding,
    write_test_embedding,
)
from concept_embeddings_rag.evaluation.strong_dense import (
    HELD_OUT_SPLIT as TEST_SPLIT,
)
from concept_embeddings_rag.nodes.index import build_node_index
from concept_embeddings_rag.nodes.local_extraction import record_from_forms

SOURCE = Path(__file__).resolve().parents[2] / "src" / "concept_embeddings_rag"
IDS = [f"u{index:03d}" for index in range(120)]
EXTRACTION_DIGEST = "d" * 64
CORPUS_KEY = "c" * 16
DEV_KEY = "q" * 16
TEST_KEY = "t" * 16
RESOLVED = config.PHASE_8_DENSE_REVISION


class DenseFixture:
    """Dense returns the pool in id order, and swaps the first two for one question."""

    name = "dense"

    def retrieve(self, query, top_k):
        ids = list(IDS)
        if query == "no bridge":
            ids[0], ids[1] = ids[1], ids[0]
        return [(uid, 1.0 - position / 120) for position, uid in enumerate(ids[:top_k])]


class StubRetriever:
    def __init__(self, name: str, hits: list[tuple[str, float]]) -> None:
        self.name = name
        self.hits = hits

    def retrieve(self, query, top_k):
        return self.hits[:top_k]


class StubBackend:
    """Encodes every text as one row of a declared width, and counts its calls."""

    name = config.PHASE_8_DENSE_MODEL
    revision = config.PHASE_8_DENSE_REVISION
    normalize = True

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.ones((len(texts), self.dim), dtype=np.float32)


def a_node_index():
    """Three paragraphs share entity names; `u000` bridges to `u119`."""
    forms = {"u000": ["Bridge", "Common"], "u118": ["Common"], "u119": ["Bridge", "Common"]}
    records = {
        uid: record_from_forms(
            uid, forms.get(uid, []), model="fixture", configuration_digest="f" * 16
        )
        for uid in IDS
    }
    return build_node_index(records, IDS, extraction_digest=EXTRACTION_DIGEST)


def a_pool() -> list[IndexingUnit]:
    return [IndexingUnit(unit_id=uid, title=uid, sentences=(f"text of {uid}.",)) for uid in IDS]


def questions_on(split: str) -> list[Question]:
    return [
        Question(f"{split}1", "bridge", "", ("u000", "u119"), (), split),
        Question(f"{split}2", "no bridge", "", ("u001",), (), split),
    ]


def embedding_arguments(**overrides) -> dict:
    arguments = {
        "resolved_revision": RESOLVED,
        "weights_sha256": "a" * 64,
        "max_seq_length": 32768,
        "query_prompt": config.PHASE_8_QUERY_PROMPT,
        "corpus_cache_key": CORPUS_KEY,
        "question_cache_key": DEV_KEY,
        "unit_ids": IDS,
        "n_questions": 2,
        "dim": config.PHASE_8_DENSE_DIM,
        "hardware": {"device": "cuda", "gpu": "RTX 4090"},
        "corpus_seconds": 120.0,
        "question_seconds": 1.5,
        "bytes_on_disk": 4096,
        "hourly_rate_usd": 0.74,
    }
    arguments.update(overrides)
    return arguments


@pytest.fixture
def phase_8(tmp_path):
    """A written embedding artifact, a toy entity index, and the dev pass's arguments."""
    directory = tmp_path / "phase8"
    directory.mkdir(parents=True)
    index = a_node_index()
    pins = replace(
        Phase8Pins.from_config(),
        unit_set_hash=unit_set_hash(IDS),
        index_digest=index.digest,
        extraction_digest=EXTRACTION_DIGEST,
        # The real bar is exercised on its own below; a toy pool of two questions cannot
        # land on 487 of 600, so the fixture's baseline is an empty one.
        dense_dev_share=0.0,
    )
    write_embedding(directory, pins=pins, **embedding_arguments())
    arguments = {
        "unit_ids": IDS,
        "dense": DenseFixture(),
        "bm25": StubRetriever("bm25", [("u002", 3.0), ("u119", 2.0), ("u001", 1.0)]),
        "index": index,
        "questions": questions_on("dev"),
        "token_counts": dict.fromkeys(IDS, 1024),
        "run_config": {
            "model": config.PHASE_8_DENSE_MODEL,
            "revision": config.PHASE_8_DENSE_REVISION,
            "unit_set_hash": unit_set_hash(IDS),
            "seed": config.DEFAULT_SEED,
            "tokenizer": config.BUDGET_TOKENIZER_ID,
            "code_version": "test",
            "top_k": config.EVALUATION_TOP_K,
            "corpus_cache_key": CORPUS_KEY,
            "question_cache_key": DEV_KEY,
        },
        "pins": pins,
    }
    return directory, pins, arguments


# --- S3: the embedding artifact and the width guard ---------------------------------------


def test_the_embedding_artifact_records_the_model_the_weights_and_both_caches(tmp_path):
    pins = replace(Phase8Pins.from_config(), unit_set_hash=unit_set_hash(IDS))

    path = write_embedding(tmp_path, pins=pins, **embedding_arguments())

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model"] == config.PHASE_8_DENSE_MODEL
    assert payload["revision"] == payload["resolved_revision"] == config.PHASE_8_DENSE_REVISION
    assert payload["weights_file"] == config.PHASE_8_WEIGHTS_FILE
    assert payload["weights_sha256"] == "a" * 64
    assert payload["dim"] == 1024
    assert payload["max_seq_length"] == 32768
    assert payload["query_prompt"] == config.PHASE_8_QUERY_PROMPT
    assert payload["corpus_cache_key"] != payload["question_cache_key"]
    assert payload["unit_set_hash"] == unit_set_hash(IDS)
    assert payload["trust_remote_code"] is False
    assert payload["budget_tokenizer"] == config.BUDGET_TOKENIZER_ID
    assert payload["split"] == "dev"
    assert payload["digest"] == digest_of_payload(payload)


def test_the_cost_is_attributable_and_the_projection_is_labelled_a_projection(tmp_path):
    pins = replace(Phase8Pins.from_config(), unit_set_hash=unit_set_hash(IDS))

    payload = json.loads(
        write_embedding(tmp_path, pins=pins, **embedding_arguments()).read_text(encoding="utf-8")
    )

    assert payload["paragraphs_per_second"] == pytest.approx(len(IDS) / 120.0)
    assert payload["embedding_usd"] == pytest.approx(0.74 * 120.0 / 3600.0)
    projection = payload["projection_5m"]
    assert projection["paragraphs"] == config.PROJECTION_PARAGRAPHS
    assert projection["method"].startswith("linear from measured throughput")
    assert projection["usd"] == pytest.approx(projection["hours"] * 0.74)
    assert "total rented-session charge" in payload["cost_basis"]


def test_a_revision_the_hub_did_not_serve_leaves_no_artifact_behind(tmp_path):
    pins = replace(Phase8Pins.from_config(), unit_set_hash=unit_set_hash(IDS))

    with pytest.raises(StrongDenseError, match="pinned to"):
        write_embedding(tmp_path, pins=pins, **embedding_arguments(resolved_revision="f" * 40))

    assert not (tmp_path / EMBEDDING_FILENAME).exists()


def test_a_branch_name_is_not_a_pin():
    check_model_pin(config.PHASE_8_DENSE_REVISION, config.PHASE_8_DENSE_MODEL)
    for revision in ("main", "v2", config.PHASE_8_DENSE_REVISION[:39], "z" * 40):
        with pytest.raises(StrongDenseError, match="commit sha"):
            check_model_pin(revision)


def test_a_pool_that_is_not_the_frozen_one_is_refused(tmp_path):
    """The default pins name the Phase 1 pool, and a toy one is not it."""
    with pytest.raises(StrongDenseError, match="hashes to"):
        write_embedding(tmp_path, **embedding_arguments())


def test_the_two_caches_may_not_share_one_key(tmp_path):
    pins = replace(Phase8Pins.from_config(), unit_set_hash=unit_set_hash(IDS))

    with pytest.raises(StrongDenseError, match="one key"):
        write_embedding(tmp_path, pins=pins, **embedding_arguments(question_cache_key=CORPUS_KEY))


def test_a_prompt_other_than_the_declared_one_cannot_be_recorded(tmp_path):
    pins = replace(Phase8Pins.from_config(), unit_set_hash=unit_set_hash(IDS))

    with pytest.raises(StrongDenseError, match="query prompt"):
        write_embedding(tmp_path, pins=pins, **embedding_arguments(query_prompt="Query:"))


def test_the_embedding_artifact_is_written_once(phase_8):
    directory, pins, _ = phase_8

    with pytest.raises(StrongDenseError, match="written once"):
        write_embedding(directory, pins=pins, **embedding_arguments())


def test_a_block_of_another_width_aborts_before_anything_is_cached(tmp_path):
    """Qwen3-Embedding supports MRL truncation; this phase pins the full width."""
    inner = StubBackend(dim=512)
    guarded = WidthCheckedBackend(inner, config.PHASE_8_DENSE_DIM)
    cache = EmbeddingCache(tmp_path)

    with pytest.raises(StrongDenseError, match="1024 dimensions"):
        embed_units(a_pool(), guarded, cache)

    assert inner.calls == 1
    assert list(tmp_path.glob("*.npz")) == []


def test_a_block_of_the_declared_width_reaches_the_cache(tmp_path):
    guarded = WidthCheckedBackend(StubBackend(dim=config.PHASE_8_DENSE_DIM), 1024)

    vectors, _ = embed_units(a_pool(), guarded, EmbeddingCache(tmp_path))

    assert vectors.shape == (len(IDS), 1024)
    assert len(list(tmp_path.glob("*.npz"))) == 1


# --- S4: the dev pass, its diagnostics and the stop rule ----------------------------------


def test_the_dev_pass_fits_both_hybrids_and_measures_the_three_systems(phase_8):
    directory, _, arguments = phase_8

    path = measure_dev(directory, **arguments)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split"] == "dev"
    assert sorted(payload["systems"]) == sorted(config.PHASE_8_SYSTEMS)
    assert payload["digest"] == digest_of_payload(payload)
    assert payload["embedding_digest"] == json.loads(
        (directory / EMBEDDING_FILENAME).read_text(encoding="utf-8")
    )["digest"]
    assert payload["node_index_digest"] == arguments["index"].digest
    for name in config.PHASE_8_SYSTEMS:
        system = payload["systems"][name]
        for budget in config.CONTEXT_BUDGETS:
            assert "full_support" in system["metrics"][f"budget_{budget}"]
        for depth in config.RECALL_AT_K:
            assert f"recall_at_{depth}" in system["metrics"]
        run = json.loads((directory / system["run_file"]).read_text(encoding="utf-8"))
        assert run["split"] == "dev"
        assert run["metrics"] == system["metrics"]
        assert system["run_digest"] == digest_of_payload(run)
    # Both hybrids are fitted over the same declared grid, with the same objective.
    for name in ("hybrid-bm25", "hybrid-entity-hop"):
        fit = payload["systems"][name]["fit"]
        assert fit["metric"] == config.SELECTION_METRIC
        assert fit["budget"] == config.SELECTION_BUDGET
        assert fit["grid"] == list(config.FUSION_WEIGHT_GRID)
        assert len(fit["curve"]) == len(config.FUSION_WEIGHT_GRID)
        assert "rrf_score" in fit
    assert "fit" not in payload["systems"]["dense"]


def test_the_dev_pass_records_what_the_hop_did_and_whether_p1_moved(phase_8):
    directory, _, arguments = phase_8
    arguments["baseline_dense"] = StubRetriever("bge-small", [("u005", 1.0)])

    payload = json.loads(measure_dev(directory, **arguments).read_text(encoding="utf-8"))

    hop = payload["hop"]
    assert hop["positive_candidates_mean"] == 1.0
    assert hop["zero_candidate_share"] == 0.5
    assert hop["p1s_without_entity"] == 1
    assert hop["introduced_context_units"] >= 1
    assert hop["entity_index"]["distinct_nodes"] == 2
    change = payload["p1_change"]
    assert change["measured"] is True
    assert change["compared"] == 2
    assert change["changed"] == 2


def test_an_unsupplied_baseline_is_declared_rather_than_invented(phase_8):
    directory, _, arguments = phase_8

    payload = json.loads(measure_dev(directory, **arguments).read_text(encoding="utf-8"))

    assert payload["p1_change"] == {
        "measured": False,
        "reason": "no BGE-small dense retriever was supplied, so nothing is compared",
    }


def test_the_stop_rule_is_counted_in_whole_questions_at_the_inherited_reading():
    """487 of 600 is a tie, and a tie is not substantially stronger."""
    share = config.PHASE_8_INHERITED_DEV_FULL_SUPPORT["dense"]

    baseline, required = stop_rule_bar(config.N_DEV, share)

    assert (baseline, required) == (config.PHASE_8_STOP_RULE_BASELINE_QUESTIONS, 488)
    assert stop_rule(488 / config.N_DEV, config.N_DEV, share)["verdict"] == VERDICT_CONTINUE
    assert stop_rule(487 / config.N_DEV, config.N_DEV, share)["verdict"] == VERDICT_STOP
    stopped = stop_rule(487 / config.N_DEV, config.N_DEV, share)
    assert stopped["count"] == 487
    assert stopped["baseline_count"] == 487
    assert stopped["required_count"] == 488
    assert stopped["budget"] == config.SELECTION_BUDGET


def test_the_dev_pass_writes_the_verdict_whatever_it_says(phase_8):
    directory, pins, arguments = phase_8
    arguments["pins"] = replace(pins, dense_dev_share=1.0)

    payload = json.loads(measure_dev(directory, **arguments).read_text(encoding="utf-8"))

    assert payload["stop_rule"]["verdict"] == VERDICT_STOP
    assert payload["stop_rule"]["required_count"] == 3


@pytest.mark.parametrize("hidden", [False, True])
def test_held_out_questions_are_refused_before_any_measurement(phase_8, hidden):
    directory, _, arguments = phase_8
    intruder = replace(arguments["questions"][0], split="test")
    arguments["questions"] = [arguments["questions"][1], intruder] if hidden else [intruder]

    with pytest.raises(SelectionError, match="test"):
        measure_dev(directory, **arguments)

    assert not (directory / DEV_FILENAME).exists()
    assert not list(directory.glob("run-*.json"))


def test_an_index_that_is_not_the_pinned_one_is_refused(phase_8):
    directory, pins, arguments = phase_8
    arguments["pins"] = replace(pins, index_digest="0" * 64)

    with pytest.raises(StrongDenseError, match="inherited, never rebuilt"):
        measure_dev(directory, **arguments)


def test_an_index_over_another_pool_is_refused(phase_8):
    directory, pins, arguments = phase_8
    forms = {"u000": ["Bridge"]}
    records = {
        uid: record_from_forms(
            uid, forms.get(uid, []), model="fixture", configuration_digest="f" * 16
        )
        for uid in IDS
    }
    other = build_node_index(records, list(reversed(IDS)), extraction_digest=EXTRACTION_DIGEST)
    arguments["index"] = other
    arguments["pins"] = replace(pins, index_digest=other.digest)

    with pytest.raises(StrongDenseError, match="another pool"):
        measure_dev(directory, **arguments)


def test_vectors_the_embedding_artifact_did_not_record_are_refused(phase_8):
    directory, _, arguments = phase_8
    arguments["run_config"] = {**arguments["run_config"], "corpus_cache_key": "0" * 16}

    with pytest.raises(StrongDenseError, match="embedding.json recorded"):
        measure_dev(directory, **arguments)


def test_the_dev_pass_refuses_without_an_embedding_artifact(phase_8):
    directory, _, arguments = phase_8
    (directory / EMBEDDING_FILENAME).unlink()

    with pytest.raises(StrongDenseError, match="strong-embed"):
        measure_dev(directory, **arguments)


def test_an_edited_embedding_artifact_is_refused_rather_than_measured_against(phase_8):
    directory, _, arguments = phase_8
    payload = json.loads((directory / EMBEDDING_FILENAME).read_text(encoding="utf-8"))
    payload["weights_sha256"] = "b" * 64
    (directory / EMBEDDING_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StrongDenseError, match="digest"):
        measure_dev(directory, **arguments)


def test_a_measured_dev_artifact_is_never_overwritten(phase_8):
    directory, _, arguments = phase_8
    measure_dev(directory, **arguments)

    with pytest.raises(StrongDenseError, match="not overwriting"):
        measure_dev(directory, **arguments)


# --- S5: the gated held-out encoding and the single held-out run --------------------------


def held_out_arguments(arguments: dict) -> dict:
    return {
        **arguments,
        "questions": questions_on("test"),
        "run_config": {**arguments["run_config"], "question_cache_key": TEST_KEY},
    }


def gate_the_test_encoding(directory: Path, pins: Phase8Pins) -> Path:
    return write_test_embedding(
        directory,
        question_cache_key=TEST_KEY,
        n_questions=2,
        hardware={"device": "cuda"},
        question_seconds=2.0,
        hourly_rate_usd=0.74,
        pins=pins,
    )


def test_the_gated_encoding_records_both_parents_that_authorized_it(phase_8):
    directory, pins, arguments = phase_8
    measure_dev(directory, **arguments)

    payload = json.loads(gate_the_test_encoding(directory, pins).read_text(encoding="utf-8"))

    assert payload["split"] == "test"
    assert payload["question_cache_key"] == TEST_KEY
    assert payload["dev_digest"] == load_dev(directory)["digest"]
    assert payload["embedding_digest"] == json.loads(
        (directory / EMBEDDING_FILENAME).read_text(encoding="utf-8")
    )["digest"]
    assert payload["question_usd"] == pytest.approx(0.74 * 2.0 / 3600.0)
    assert payload["digest"] == digest_of_payload(payload)


def test_a_stop_verdict_leaves_the_held_out_questions_unencoded(phase_8):
    directory, pins, arguments = phase_8
    arguments["pins"] = replace(pins, dense_dev_share=1.0)
    measure_dev(directory, **arguments)

    with pytest.raises(StrongDenseError, match="closes on its dev figures"):
        gate_the_test_encoding(directory, pins)

    assert not (directory / EMBEDDING_TEST_FILENAME).exists()


def test_the_gated_encoding_refuses_a_dev_reading_that_fails_its_digest(phase_8):
    directory, pins, arguments = phase_8
    measure_dev(directory, **arguments)
    payload = json.loads((directory / DEV_FILENAME).read_text(encoding="utf-8"))
    payload["stop_rule"]["verdict"] = VERDICT_CONTINUE
    payload["systems"]["dense"]["full_support"] = 1.0
    (directory / DEV_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StrongDenseError, match="digest"):
        gate_the_test_encoding(directory, pins)


def test_the_gated_encoding_refuses_the_dev_question_key(phase_8):
    directory, pins, arguments = phase_8
    measure_dev(directory, **arguments)

    with pytest.raises(StrongDenseError, match="never be able to land"):
        write_test_embedding(
            directory,
            question_cache_key=DEV_KEY,
            n_questions=2,
            hardware={},
            question_seconds=1.0,
            hourly_rate_usd=0.0,
            pins=pins,
        )


def test_the_held_out_run_measures_three_systems_once_and_states_the_surviving_gain(phase_8):
    directory, pins, arguments = phase_8
    measure_dev(directory, **arguments)
    gate_the_test_encoding(directory, pins)

    path = measure_held_out(directory, **held_out_arguments(arguments))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split"] == "test"
    assert sorted(payload["systems"]) == sorted(config.PHASE_8_SYSTEMS)
    assert payload["dev_digest"] == load_dev(directory)["digest"]
    assert payload["embedding_test_digest"] == json.loads(
        (directory / EMBEDDING_TEST_FILENAME).read_text(encoding="utf-8")
    )["digest"]
    assert payload["digest"] == digest_of_payload(payload)
    # Nothing is refitted: the schemes and weights are the ones dev chose.
    dev = load_dev(directory)
    for name in ("hybrid-bm25", "hybrid-entity-hop"):
        assert payload["systems"][name]["fit"] == {
            "scheme": dev["systems"][name]["fit"]["scheme"],
            "weights": dev["systems"][name]["fit"]["weights"],
            "fitted_on": "dev",
        }
        run = json.loads((directory / payload["systems"][name]["run_file"]).read_text("utf-8"))
        assert run["split"] == "test"
    gain = payload["surviving_gain"]
    assert gain["budget"] == config.SELECTION_BUDGET
    assert gain["inherited_n_questions"] == config.N_TEST
    assert gain["inherited"]["dense"]["supported_questions"] == 1155
    assert gain["inherited"]["hybrid-entity-hop-gliner"]["supported_questions"] == 1231
    assert gain["inherited_entity_gain_questions"] == 76
    assert payload["inherited_test_full_support"] == config.PHASE_8_INHERITED_HELD_OUT_FULL_SUPPORT

    with pytest.raises(StrongDenseError, match="read once"):
        measure_held_out(directory, **held_out_arguments(arguments))


def test_the_held_out_run_refuses_without_the_gated_encoding(phase_8):
    directory, _, arguments = phase_8
    measure_dev(directory, **arguments)

    with pytest.raises(StrongDenseError, match="strong-embed --test"):
        measure_held_out(directory, **held_out_arguments(arguments))


def test_the_held_out_run_refuses_without_a_dev_reading(phase_8):
    directory, _, arguments = phase_8

    with pytest.raises(StrongDenseError, match="strong-dense"):
        measure_held_out(directory, **held_out_arguments(arguments))


@pytest.mark.parametrize("hidden", [False, True])
def test_the_held_out_run_refuses_dev_questions(phase_8, hidden):
    directory, pins, arguments = phase_8
    measure_dev(directory, **arguments)
    gate_the_test_encoding(directory, pins)
    held_out = held_out_arguments(arguments)
    intruder = arguments["questions"][0]
    held_out["questions"] = [*questions_on("test"), intruder] if hidden else [intruder]

    with pytest.raises(StrongDenseError, match="measures the held-out split"):
        measure_held_out(directory, **held_out)

    assert not (directory / TEST_FILENAME).exists()


def test_the_held_out_run_refuses_questions_the_gated_encoding_did_not_cover(phase_8):
    directory, pins, arguments = phase_8
    measure_dev(directory, **arguments)
    gate_the_test_encoding(directory, pins)
    held_out = held_out_arguments(arguments)
    held_out["run_config"] = {**held_out["run_config"], "question_cache_key": "z" * 16}

    with pytest.raises(StrongDenseError, match="embedding-test.json recorded"):
        measure_held_out(directory, **held_out)


def test_the_phase_8_cache_keys_resolve_to_no_existing_artifact():
    """R3, read against the real cache directory: a new model writes new files.

    `cache_key` and `question_cache_key` already guarantee this by construction, but the
    property the restriction actually cares about is that nothing under `data/cache/` is
    about to be overwritten - which is a fact about this checkout, not about the hash.
    """
    pool = config.DATA_DIR / "pool.json"
    if not pool.exists():
        pytest.skip("the frozen pool is not on this checkout")
    units, questions = load_pool(pool)
    unit_ids = [unit.unit_id for unit in units]
    corpus_hash = unit_set_hash(unit_ids)

    phase_8 = cache_key(
        config.PHASE_8_DENSE_MODEL,
        config.PHASE_8_DENSE_REVISION,
        corpus_hash,
        normalized=config.NORMALIZE_EMBEDDINGS,
    )
    inherited = cache_key(
        config.EMBEDDING_MODEL,
        config.EMBEDDING_REVISION,
        corpus_hash,
        normalized=config.NORMALIZE_EMBEDDINGS,
    )
    assert phase_8 != inherited
    assert not (config.CACHE_DIR / f"embeddings-{phase_8}.npz").exists()
    for split in ("dev", TEST_SPLIT):
        subset = [question for question in questions if question.split == split]
        if not subset:
            continue
        key = question_cache_key(
            config.PHASE_8_DENSE_MODEL,
            config.PHASE_8_DENSE_REVISION,
            question_set_hash([question.qid for question in subset]),
            split,
            config.NORMALIZE_EMBEDDINGS,
            config.PHASE_8_QUERY_PROMPT,
        )
        assert not (config.QUESTION_CACHE_DIR / f"embeddings-{key}.npz").exists()


# --- The security surface this phase declares, asserted on the code that runs -------------


@pytest.mark.parametrize(
    "module", ["evaluation/strong_dense.py", "embeddings/backend.py", "cli.py"]
)
def test_no_phase_8_module_ever_asks_for_remote_code(module):
    """The approved model needs no `trust_remote_code`, and enabling it is a different surface."""
    source = (SOURCE / module).read_text(encoding="utf-8")
    flagged = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.keyword) and node.arg == "trust_remote_code"
    ]
    assert flagged == [], f"{module} passes trust_remote_code at line(s) {flagged}"
