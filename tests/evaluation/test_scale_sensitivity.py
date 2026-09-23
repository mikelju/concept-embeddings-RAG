"""Deviation 8.1, S5: the C19 reproduction gate, level 1.

Level 1 is the only place in 8.1 where new code meets old vectors, and that is its entire
point. The naive single gate would compare a number produced by new evaluation code **and**
new vectors against one produced by old code and old vectors, and when it failed it could
not say which moved. So level 1 runs the new path over the **unchanged frozen C19 corpus**
using the **existing historical caches**: the vectors are bit-identical to the recorded
runs, the only variable is the code, and both counts must match exactly.

```text
BGE-small   487 / 600
Qwen        446 / 600
```

Anything else is a defect in the new path, recorded before the implementation is touched,
and the terminal state is `reproduction_stop`. The tests below use small synthetic corpora
and an injected required-count map, because what is under test is the gate's arithmetic and
its refusals - not a re-measurement of the historical figures, which no test may perform.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import scale_sensitivity
from concept_embeddings_rag.evaluation.selection import SelectionError


class ScriptedRetriever:
    """Returns a fixed ranking per question. No vectors, no model, no network."""

    def __init__(self, name: str, rankings: dict[str, list[str]]) -> None:
        self.name = name
        self.rankings = rankings

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        ranked = self.rankings[query]
        return [(unit_id, 1.0 - 0.01 * i) for i, unit_id in enumerate(ranked[:top_k])]


def dev_questions(n: int) -> list[Question]:
    return [
        Question(
            qid=f"q{i}",
            question=f"question {i}",
            answer="answer",
            gold_unit_ids=(f"gold{i}a", f"gold{i}b"),
            supporting_facts=((f"Title {i}", 0),),
            split="dev",
        )
        for i in range(n)
    ]


def token_counts_for(questions: list[Question]) -> dict[str, int]:
    counts = {"filler": 10}
    for question in questions:
        for unit_id in question.gold_unit_ids:
            counts[unit_id] = 10
    return counts


def scripted(name: str, questions: list[Question], hits: int) -> ScriptedRetriever:
    """A retriever that fully supports the first `hits` questions and misses the rest."""
    rankings: dict[str, list[str]] = {}
    for index, question in enumerate(questions):
        if index < hits:
            rankings[question.question] = list(question.gold_unit_ids) + ["filler"]
        else:
            rankings[question.question] = ["filler"]
    return ScriptedRetriever(name, rankings)


def provenance_for(model: str) -> dict[str, object]:
    return {
        "model": model,
        "revision": "0" * 40,
        "unit_set_hash": "abc123",
        "seed": config.DEFAULT_SEED,
        "tokenizer": config.BUDGET_TOKENIZER_ID,
        "code_version": "test",
        "top_k": config.EVALUATION_TOP_K,
        "corpus_cache_key": "cachekey",
        "question_cache_key": "questionkey",
        "dim": 384,
    }


def measure(tmp_path: Path, *, bge_hits: int, qwen_hits: int, required: dict[str, int]) -> Path:
    questions = dev_questions(3)
    return scale_sensitivity.measure_level_1(
        tmp_path,
        retrievers={
            "bge": scripted("dense-bge-c19", questions, bge_hits),
            "qwen": scripted("dense-qwen-c19", questions, qwen_hits),
        },
        questions=questions,
        token_counts=token_counts_for(questions),
        provenance={"bge": provenance_for("bge-model"), "qwen": provenance_for("qwen-model")},
        host={"host": "laptop", "device": "cpu"},
        required=required,
        expected_questions=3,
    )


def test_matching_both_counts_passes_the_gate(tmp_path: Path) -> None:
    path = measure(tmp_path, bge_hits=2, qwen_hits=1, required={"bge": 2, "qwen": 1})
    body = json.loads(path.read_text(encoding="utf-8"))

    assert body["terminal_state"] is None
    assert body["matches"] is True
    for model, expected in (("bge", 2), ("qwen", 1)):
        reading = body["models"][model]
        assert reading["required"] == expected
        assert reading["supported"] == expected
        assert reading["matches"] is True
        # The headline budget is the deviation's, and every budget is persisted beside it.
        assert reading["full_support"][str(config.SELECTION_BUDGET)] == pytest.approx(expected / 3)
        assert set(reading["recall_at_k"]) == {str(k) for k in config.RECALL_AT_K}
        assert reading["corpus_cache_key"] == "cachekey"
        assert reading["question_cache_key"] == "questionkey"
    assert body["host"]["device"] == "cpu"


def test_one_question_off_is_a_reproduction_stop(tmp_path: Path) -> None:
    """Level 1 has no tolerance: the vectors are identical, so the code is the variable."""
    path = measure(tmp_path, bge_hits=1, qwen_hits=1, required={"bge": 2, "qwen": 1})
    body = json.loads(path.read_text(encoding="utf-8"))

    assert body["terminal_state"] == "reproduction_stop"
    assert body["matches"] is False
    assert body["models"]["bge"]["matches"] is False
    assert body["models"]["qwen"]["matches"] is True


def test_check_reproduction_raises_on_a_stop(tmp_path: Path) -> None:
    measure(tmp_path, bge_hits=0, qwen_hits=1, required={"bge": 2, "qwen": 1})

    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="reproduction_stop"):
        scale_sensitivity.check_reproduction(tmp_path)


def test_check_reproduction_raises_when_the_artifact_is_absent(tmp_path: Path) -> None:
    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="scale-repro"):
        scale_sensitivity.check_reproduction(tmp_path)


def test_check_reproduction_returns_the_reading_when_it_passed(tmp_path: Path) -> None:
    measure(tmp_path, bge_hits=2, qwen_hits=1, required={"bge": 2, "qwen": 1})

    body = scale_sensitivity.check_reproduction(tmp_path)

    assert body["matches"] is True
    assert set(body["models"]) == {"bge", "qwen"}


def test_a_second_run_does_not_overwrite_the_gate(tmp_path: Path) -> None:
    measure(tmp_path, bge_hits=2, qwen_hits=1, required={"bge": 2, "qwen": 1})

    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="reproduction"):
        measure(tmp_path, bge_hits=2, qwen_hits=1, required={"bge": 2, "qwen": 1})


def test_a_held_out_question_is_refused(tmp_path: Path) -> None:
    questions = dev_questions(3)
    contaminated = [*questions[:2], Question(**{**questions[2].__dict__, "split": "test"})]

    # The refusal is the project's own `check_dev_only`, not a local copy of it: one
    # guard at every door is what keeps a held-out question from ever being measured.
    with pytest.raises(SelectionError, match="split"):
        scale_sensitivity.measure_level_1(
            tmp_path,
            retrievers={"bge": scripted("dense-bge-c19", contaminated, 2)},
            questions=contaminated,
            token_counts=token_counts_for(questions),
            provenance={"bge": provenance_for("bge-model")},
            host={"host": "laptop", "device": "cpu"},
            required={"bge": 2},
            expected_questions=3,
        )


def test_a_question_set_that_is_not_the_dev_split_size_is_refused(tmp_path: Path) -> None:
    questions = dev_questions(3)

    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="600"):
        scale_sensitivity.measure_level_1(
            tmp_path,
            retrievers={"bge": scripted("dense-bge-c19", questions, 2)},
            questions=questions,
            token_counts=token_counts_for(questions),
            provenance={"bge": provenance_for("bge-model")},
            host={"host": "laptop", "device": "cpu"},
            required={"bge": 2},
        )


def test_the_metadata_backend_never_encodes() -> None:
    """Level 1 reads caches. If anything tried to embed, that is the bug, not a fallback."""
    backend = scale_sensitivity.MetadataBackend(
        name=config.EMBEDDING_MODEL, revision=config.EMBEDDING_REVISION, query_prompt=""
    )

    assert backend.name == config.EMBEDDING_MODEL
    assert backend.normalize is config.NORMALIZE_EMBEDDINGS
    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="cache"):
        backend.encode(["a question"])


def test_model_pins_come_from_config_and_not_from_literals() -> None:
    bge = scale_sensitivity.ModelPins.for_key("bge")
    qwen = scale_sensitivity.ModelPins.for_key("qwen")

    assert (bge.model, bge.revision, bge.dim) == (
        config.EMBEDDING_MODEL,
        config.EMBEDDING_REVISION,
        config.EMBEDDING_DIM,
    )
    assert bge.query_prompt == ""
    assert (qwen.model, qwen.revision, qwen.dim) == (
        config.PHASE_8_DENSE_MODEL,
        config.PHASE_8_DENSE_REVISION,
        config.PHASE_8_DENSE_DIM,
    )
    assert qwen.query_prompt == config.PHASE_8_QUERY_PROMPT
    with pytest.raises(scale_sensitivity.ScaleSensitivityError):
        scale_sensitivity.ModelPins.for_key("spacy")


def test_the_required_counts_default_to_the_recorded_ones() -> None:
    assert scale_sensitivity.required_counts() == config.PHASE_8_1_C19_DEV_SUPPORTED
    assert config.PHASE_8_1_C19_DEV_SUPPORTED["bge"] == config.PHASE_8_STOP_RULE_BASELINE_QUESTIONS


def test_the_required_bge_count_matches_the_artifact_it_came_from() -> None:
    """487 is not a literal in a report: it is the inherited dev reading, times 600."""
    inherited = config.PHASE_8_INHERITED_DEV_FULL_SUPPORT["dense"]

    assert round(inherited * config.N_DEV) == config.PHASE_8_1_C19_DEV_SUPPORTED["bge"]


def test_the_required_qwen_count_matches_the_phase_8_dev_artifact() -> None:
    """446 is checked against `data/phase8/dev.json` when that untracked file is present."""
    artifact = config.PHASE_8_DIR / "dev.json"
    if not artifact.exists():
        pytest.skip("data/phase8/dev.json is not on this checkout")
    body = json.loads(artifact.read_text(encoding="utf-8"))

    dense = body["systems"]["dense"]
    supported = round(float(dense["full_support"]) * config.N_DEV)
    assert supported == config.PHASE_8_1_C19_DEV_SUPPORTED["qwen"]


def test_historical_caches_are_only_ever_read(tmp_path: Path) -> None:
    """The module must not name a writing door into the historical cache directories."""
    source = Path(scale_sensitivity.__file__).read_text(encoding="utf-8")

    for forbidden in ("savez", "write_text(", "CACHE_DIR /", "QUESTION_CACHE_DIR /"):
        assert forbidden not in source, forbidden
    assert "write_text_atomic" in source


def test_level_1_reads_vectors_it_does_not_own(tmp_path: Path) -> None:
    """`historical_retriever` builds a retriever from a cache and refuses to encode."""
    vectors = np.eye(3, dtype=np.float32)
    unit_ids = ["a", "b", "c"]
    questions = dev_questions(1)
    query_vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    retriever = scale_sensitivity.historical_retriever(
        vectors,
        unit_ids,
        questions=questions,
        query_vectors=query_vectors,
        qids=[questions[0].qid],
        pins=scale_sensitivity.ModelPins.for_key("bge"),
        name="dense-bge-c19",
    )

    hits = retriever.retrieve(questions[0].question, top_k=3)
    assert hits[0][0] == "a"
    assert hits[0][1] == pytest.approx(1.0)


def test_measurement_host_level_1_is_exact_and_does_not_write_an_artifact(
    tmp_path: Path,
) -> None:
    questions = dev_questions(3)
    reading = scale_sensitivity.measure_level_one_on_measurement_host(
        model="bge",
        retriever=scripted("dense-bge-c19", questions, 2),
        questions=questions,
        token_counts=token_counts_for(questions),
        provenance=provenance_for("bge-model"),
        host={"host": "pod", "device": "cuda"},
        required=2,
        expected_questions=3,
    )

    assert reading["supported"] == 2
    assert reading["matches"] is True
    assert reading["host"]["device"] == "cuda"
    assert len(reading["run_digest"]) == 64
    assert not (tmp_path / scale_sensitivity.REPRODUCTION_FILENAME).exists()

    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="reproduction_stop"):
        scale_sensitivity.measure_level_one_on_measurement_host(
            model="bge",
            retriever=scripted("dense-bge-c19", questions, 1),
            questions=questions,
            token_counts=token_counts_for(questions),
            provenance=provenance_for("bge-model"),
            host={"host": "pod", "device": "cuda"},
            required=2,
            expected_questions=3,
        )


# --- S6: the scale readings and the level-2 gate -----------------------------


def scale_provenance(model: str) -> dict[str, object]:
    return {**provenance_for(model), "embedding_digest": "e" * 16}


def scaled(name: str, questions: list[Question], hits_by_size: dict[int, int]) -> dict:
    """One scripted retriever per corpus size, supporting `hits` questions at that size."""
    return {size: scripted(name, questions, hits) for size, hits in hits_by_size.items()}


def measure_scales(
    tmp_path: Path,
    *,
    model: str = "bge",
    hits_by_size: dict[int, int],
    level_1_supported: int,
    sizes: tuple[int, ...] = (3, 4, 5, 6),
) -> Path:
    questions = dev_questions(3)
    retrievers = scaled(f"dense-{model}-c19", questions, hits_by_size)
    return scale_sensitivity.measure_scales(
        tmp_path,
        model=model,
        retriever_for_size=lambda size: retrievers[size],
        questions=questions,
        token_counts=token_counts_for(questions),
        provenance=scale_provenance(model),
        level_1_supported=level_1_supported,
        retrieval_cost={"mean_seconds_per_query": 0.01, "peak_rss_mb": 100},
        host={"host": "pod", "device": "cuda"},
        sizes=sizes,
        expected_questions=3,
    )


def test_scale_readings_are_persisted_at_every_size(tmp_path: Path) -> None:
    path = measure_scales(tmp_path, hits_by_size={3: 3, 4: 2, 5: 2, 6: 1}, level_1_supported=3)
    body = json.loads(path.read_text(encoding="utf-8"))

    assert body["model"] == "bge"
    assert sorted(body["scales"]) == ["3", "4", "5", "6"]
    for size, expected in (("3", 3), ("4", 2), ("5", 2), ("6", 1)):
        reading = body["scales"][size]
        assert reading["supported"] == expected
        assert reading["units"] == int(size)
        assert set(reading["recall_at_k"]) == {str(k) for k in config.RECALL_AT_K}
        assert reading["run_file"].startswith("run-dense-bge-")
        assert len(reading["run_digest"]) == 64
    # Each model's own drop from its C19 reading travels beside every deficit.
    assert body["own_drop_c19_to_c500"] == 2
    assert body["retrieval_cost"]["peak_rss_mb"] == 100
    assert body["terminal_state"] is None


def test_level_2_agreement_is_recorded_without_a_caveat(tmp_path: Path) -> None:
    path = measure_scales(tmp_path, hits_by_size={3: 3, 4: 3, 5: 3, 6: 3}, level_1_supported=3)
    body = json.loads(path.read_text(encoding="utf-8"))

    level_2 = body["level_2"]
    assert level_2["supported"] == 3
    assert level_2["level_1_supported"] == 3
    assert level_2["difference"] == 0
    assert level_2["within_tolerance"] is True
    assert level_2["caveat"] is None


def test_a_one_question_level_2_divergence_is_a_recorded_caveat(tmp_path: Path) -> None:
    """One question is tolerated and reported without inventing a causal attribution."""
    path = measure_scales(tmp_path, hits_by_size={3: 2, 4: 2, 5: 2, 6: 1}, level_1_supported=3)
    body = json.loads(path.read_text(encoding="utf-8"))

    assert body["level_2"]["difference"] == -1
    assert body["level_2"]["within_tolerance"] is True
    assert body["level_2"]["caveat"] is not None
    assert "residual numerical/host/re-encoding variation" in body["level_2"]["caveat"]
    assert "Float non-determinism" not in body["level_2"]["caveat"]
    assert body["terminal_state"] is None


def test_d_l1_can_localize_a_tolerated_level_2_difference_to_host_evaluation(
    tmp_path: Path,
) -> None:
    questions = dev_questions(3)
    retrievers = scaled("dense-bge-c19", questions, {3: 2, 4: 2, 5: 2, 6: 1})
    path = scale_sensitivity.measure_scales(
        tmp_path,
        model="bge",
        retriever_for_size=lambda size: retrievers[size],
        questions=questions,
        token_counts=token_counts_for(questions),
        provenance=scale_provenance("bge"),
        level_1_supported=3,
        retrieval_cost={},
        host={"host": "pod", "device": "cuda"},
        sizes=(3, 4, 5, 6),
        level_one_on_measurement_host={"supported": 2, "matches": True},
        expected_questions=3,
    )
    caveat = json.loads(path.read_text(encoding="utf-8"))["level_2"]["caveat"]

    assert "localizing the difference to host-side evaluation" in caveat
    assert "Float non-determinism" not in caveat


def test_a_two_question_level_2_divergence_stops_before_larger_scales(tmp_path: Path) -> None:
    questions = dev_questions(3)
    retrievers = scaled("dense-bge-c19", questions, {3: 1, 4: 1, 5: 1, 6: 1})
    requested: list[int] = []

    def retriever_for_size(size: int) -> ScriptedRetriever:
        requested.append(size)
        return retrievers[size]

    path = scale_sensitivity.measure_scales(
        tmp_path,
        model="bge",
        retriever_for_size=retriever_for_size,
        questions=questions,
        token_counts=token_counts_for(questions),
        provenance=scale_provenance("bge"),
        level_1_supported=3,
        retrieval_cost={"mean_seconds_per_query": 0.01},
        host={"host": "pod", "device": "cuda"},
        sizes=(3, 4, 5, 6),
        expected_questions=3,
    )
    body = json.loads(path.read_text(encoding="utf-8"))

    assert body["level_2"]["difference"] == -2
    assert body["level_2"]["within_tolerance"] is False
    assert body["terminal_state"] == "reproduction_stop"
    assert requested == [3]
    assert body["sizes"] == [3]
    assert set(body["scales"]) == {"3"}
    assert body["own_drop_c19_to_c500"] is None


def test_a_second_scale_run_is_refused(tmp_path: Path) -> None:
    measure_scales(tmp_path, hits_by_size={3: 3, 4: 2, 5: 2, 6: 1}, level_1_supported=3)

    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="scale-bge.json"):
        measure_scales(tmp_path, hits_by_size={3: 3, 4: 2, 5: 2, 6: 1}, level_1_supported=3)


def test_the_scale_run_refuses_a_held_out_question(tmp_path: Path) -> None:
    questions = dev_questions(3)
    contaminated = [*questions[:2], Question(**{**questions[2].__dict__, "split": "test"})]
    retriever = scripted("dense-bge-c19", contaminated, 2)

    with pytest.raises(SelectionError, match="split"):
        scale_sensitivity.measure_scales(
            tmp_path,
            model="bge",
            retriever_for_size=lambda size: retriever,
            questions=contaminated,
            token_counts=token_counts_for(questions),
            provenance=scale_provenance("bge"),
            level_1_supported=2,
            retrieval_cost={},
            host={},
            sizes=(3,),
            expected_questions=3,
        )


# --- S7: the pre-declared outcome -------------------------------------------


def scale_body(model: str, supported: dict[int, int], *, terminal: str | None = None) -> dict:
    """A minimal scale-<model>.json body, as `measure_scales` writes one."""
    sizes = sorted(supported)
    return {
        "model": model,
        "questions": 600,
        "scales": {str(size): {"supported": supported[size], "units": size} for size in sizes},
        "own_drop_c19_to_c500": supported[sizes[0]] - supported[sizes[-1]],
        "level_2": {"difference": 0, "within_tolerance": True, "caveat": None},
        "provenance": {"question_set_hash": "f" * 64},
        "terminal_state": terminal,
    }


SIZES = (19366, 100000, 250000, 500000)


def outcome_for(tmp_path: Path, bge: dict[int, int], qwen: dict[int, int], **kwargs) -> dict:
    path = scale_sensitivity.classify_outcome(
        tmp_path,
        bge=scale_body("bge", bge),
        qwen=scale_body("qwen", qwen),
        reconciliation={"terminal_state": None},
        **kwargs,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_crossover_fires_when_qwen_reaches_bge_at_any_scale(tmp_path: Path) -> None:
    body = outcome_for(
        tmp_path,
        bge={19366: 487, 100000: 400, 250000: 350, 500000: 300},
        qwen={19366: 446, 100000: 400, 250000: 340, 500000: 290},
    )

    assert body["terminal_state"] == "crossover"
    assert body["rule"] == "C"
    assert body["crossover_at"] == [100000]


def test_c19_equality_alone_does_not_count_as_a_crossover(tmp_path: Path) -> None:
    body = outcome_for(
        tmp_path,
        bge={19366: 487, 100000: 450, 250000: 420, 500000: 400},
        qwen={19366: 487, 100000: 420, 250000: 390, 500000: 370},
    )

    assert body["terminal_state"] == "stable_ranking"
    assert body["crossover_at"] == []


def test_convergence_needs_the_floor_guard(tmp_path: Path) -> None:
    """Deficit within 20 and BGE above the floor: the models converged."""
    body = outcome_for(
        tmp_path,
        bge={19366: 487, 100000: 420, 250000: 380, 500000: 350},
        qwen={19366: 446, 100000: 405, 250000: 365, 500000: 335},
    )

    assert body["terminal_state"] == "convergence"
    assert body["rule"] == "B"
    assert body["deficit_c500_questions"] == -15
    assert body["floor_guard"]["bge_c500"] == 350
    assert body["floor_guard"]["floor"] == config.PHASE_8_1_BGE_RETENTION_FLOOR
    assert body["floor_guard"]["above_floor"] is True
    # Every outcome reports each model's own drop beside the deficit.
    assert body["own_drops"] == {"bge": 137, "qwen": 111}


def test_both_degrade_when_the_deficit_narrows_below_the_floor(tmp_path: Path) -> None:
    """The worked example the deviation gives: 487 -> 300 and 446 -> 285 is not convergence."""
    body = outcome_for(
        tmp_path,
        bge={19366: 487, 100000: 400, 250000: 350, 500000: 230},
        qwen={19366: 446, 100000: 380, 250000: 330, 500000: 215},
    )

    assert body["terminal_state"] == "both_degrade"
    assert body["rule"] == "BD"
    assert body["floor_guard"]["above_floor"] is False
    assert "converged" not in body["permitted_reading"]


def test_stable_ranking_when_the_deficit_survives(tmp_path: Path) -> None:
    body = outcome_for(
        tmp_path,
        bge={19366: 487, 100000: 450, 250000: 420, 500000: 400},
        qwen={19366: 446, 100000: 400, 250000: 370, 500000: 350},
    )

    assert body["terminal_state"] == "stable_ranking"
    assert body["rule"] == "A"
    assert body["deficit_c500_questions"] == -50


def test_the_boundary_of_twenty_questions_is_inclusive(tmp_path: Path) -> None:
    """20 questions is convergence; 21 is a surviving deficit. The threshold is frozen."""
    twenty = outcome_for(
        tmp_path / "twenty",
        bge={19366: 487, 100000: 450, 250000: 420, 500000: 400},
        qwen={19366: 446, 100000: 420, 250000: 390, 500000: 380},
    )
    twenty_one = outcome_for(
        tmp_path / "twentyone",
        bge={19366: 487, 100000: 450, 250000: 420, 500000: 400},
        qwen={19366: 446, 100000: 420, 250000: 390, 500000: 379},
    )

    assert twenty["deficit_c500_questions"] == -config.PHASE_8_1_CONVERGENCE_QUESTIONS
    assert twenty["terminal_state"] == "convergence"
    assert twenty_one["terminal_state"] == "stable_ranking"


def test_classification_refuses_after_a_reproduction_stop(tmp_path: Path) -> None:
    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="reproduction_stop"):
        scale_sensitivity.classify_outcome(
            tmp_path,
            bge=scale_body("bge", {19366: 487, 500000: 300}, terminal="reproduction_stop"),
            qwen=scale_body("qwen", {19366: 446, 500000: 290}),
            reconciliation={"terminal_state": None},
        )


def test_classification_refuses_after_a_data_stop(tmp_path: Path) -> None:
    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="data_stop"):
        scale_sensitivity.classify_outcome(
            tmp_path,
            bge=scale_body("bge", {19366: 487, 500000: 300}),
            qwen=scale_body("qwen", {19366: 446, 500000: 290}),
            reconciliation={"terminal_state": "data_stop"},
        )


def test_classification_refuses_incomplete_scale_sets(tmp_path: Path) -> None:
    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="complete measurements"):
        scale_sensitivity.classify_outcome(
            tmp_path,
            bge=scale_body("bge", {19366: 487, 500000: 300}),
            qwen=scale_body("qwen", {19366: 446, 500000: 290}),
            reconciliation={"terminal_state": None},
        )


def test_classification_refuses_any_question_count_but_600(tmp_path: Path) -> None:
    bge = scale_body("bge", {19366: 487, 100000: 450, 250000: 420, 500000: 400})
    qwen = scale_body("qwen", {19366: 446, 100000: 400, 250000: 370, 500000: 350})
    qwen["questions"] = 599

    with pytest.raises(scale_sensitivity.ScaleSensitivityError, match="exactly 600 dev questions"):
        scale_sensitivity.classify_outcome(
            tmp_path,
            bge=bge,
            qwen=qwen,
            reconciliation={"terminal_state": None},
        )


def test_classification_refuses_different_dev_question_sets(tmp_path: Path) -> None:
    bge = scale_body("bge", {19366: 487, 100000: 450, 250000: 420, 500000: 400})
    qwen = scale_body("qwen", {19366: 446, 100000: 400, 250000: 370, 500000: 350})
    qwen["provenance"]["question_set_hash"] = "e" * 64

    with pytest.raises(
        scale_sensitivity.ScaleSensitivityError, match="exactly the same frozen dev"
    ):
        scale_sensitivity.classify_outcome(
            tmp_path,
            bge=bge,
            qwen=qwen,
            reconciliation={"terminal_state": None},
        )


def test_the_level_2_caveat_travels_into_the_outcome(tmp_path: Path) -> None:
    bge = scale_body("bge", {19366: 487, 100000: 450, 250000: 420, 500000: 400})
    bge["level_2"] = {"difference": -1, "within_tolerance": True, "caveat": "one question"}
    path = scale_sensitivity.classify_outcome(
        tmp_path,
        bge=bge,
        qwen=scale_body("qwen", {19366: 446, 100000: 400, 250000: 370, 500000: 350}),
        reconciliation={"terminal_state": None},
    )
    body = json.loads(path.read_text(encoding="utf-8"))

    assert body["measurement_caveats"] == {"bge": "one question"}


def test_the_headline_table_is_in_the_artifact(tmp_path: Path) -> None:
    body = outcome_for(
        tmp_path,
        bge={19366: 487, 100000: 450, 250000: 420, 500000: 400},
        qwen={19366: 446, 100000: 400, 250000: 370, 500000: 350},
    )

    table = {row["corpus"]: row for row in body["headline"]}
    assert set(table) == set(SIZES)
    assert table[19366]["bge"] == 487
    assert table[19366]["qwen"] == 446
    assert table[19366]["deficit_questions"] == -41
    assert table[19366]["deficit_points"] == pytest.approx(-41 / 600 * 100, abs=1e-6)


def test_no_label_outside_the_four_can_be_produced() -> None:
    assert set(config.PHASE_8_1_OUTCOMES) == {
        "crossover",
        "convergence",
        "both_degrade",
        "stable_ranking",
    }
    assert set(config.PHASE_8_1_STOPS) == {"data_stop", "reproduction_stop"}
