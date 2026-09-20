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
