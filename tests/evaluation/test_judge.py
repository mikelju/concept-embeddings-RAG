"""Phase 17, S1: the judge's scores stay aligned with their texts, truncations are counted
with the judge's own tokenizer, and a non-finite score refuses. A stub stands in for the
cross-encoder: it sorts its inputs by length and batches them, as `CrossEncoder.predict`
does, so a misaligned split would show."""

import math
from pathlib import Path

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import judge, phase17


def text_score(question: str, text: str) -> float:
    """A score that names its pair: any misalignment changes the value read back."""
    return float(len(text)) + 0.001 * float(sum(map(ord, text)) % 997) + 0.1 * len(question)


class StubModel:
    def __init__(self, nan_on: str | None = None) -> None:
        self.calls: list[int] = []
        self.nan_on = nan_on

    def predict(self, inputs, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        self.calls.append(len(inputs))
        order = np.argsort([-len(q) - len(t) for q, t in inputs], kind="stable")
        out = np.empty(len(inputs), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            for index in order[start : start + batch_size]:
                question, text = inputs[index]
                value = math.nan if text == self.nan_on else text_score(question, text)
                out[index] = value
        return out


class StubTokenizer:
    """Whitespace tokens plus three special tokens ([CLS] q [SEP] t [SEP])."""

    def __call__(self, questions, texts, truncation=False, add_special_tokens=True):
        assert truncation is False and add_special_tokens is True
        ids = [
            [0] * (len(q.split()) + len(t.split()) + 3)
            for q, t in zip(questions, texts, strict=True)
        ]
        return {"input_ids": ids}


def rows() -> list[dict]:
    return [
        {
            "qid": f"q{i}",
            "question": f"question number {i}?",
            "unit_ids": [f"u{i}-{j}" for j in range(n)],
            "texts": [f"text {i} {j} " + "w " * ((i * 7 + j * 13) % 11) for j in range(n)],
        }
        for i, n in enumerate([3, 1, 5, 2, 4, 6, 2])
    ]


def test_scores_stay_aligned_with_their_texts_through_length_sorting_and_batches():
    model = StubModel()

    scored = judge.score_rows(model, rows(), batch_size=3)

    assert model.calls == [sum(len(r["texts"]) for r in rows())]
    for row, scores in zip(rows(), scored, strict=True):
        assert scores.dtype == np.float32
        expected = np.asarray(
            [text_score(row["question"], t) for t in row["texts"]], dtype=np.float32
        )
        np.testing.assert_array_equal(scores, expected)


def test_shards_give_the_same_scores_as_one_call():
    model = StubModel()
    whole = judge.score_rows(model, rows(), batch_size=4)
    sharded = []
    for start, stop in phase17.shard_bounds(len(rows()), 3):
        sharded.extend(judge.score_rows(model, rows()[start:stop], batch_size=4))

    assert len(model.calls) == 1 + 3
    for a, b in zip(whole, sharded, strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_non_finite_score_refuses():
    model = StubModel(nan_on=rows()[2]["texts"][1])
    with pytest.raises(judge.JudgeError, match="non-finite"):
        judge.score_rows(model, rows(), batch_size=8)


def test_a_row_whose_texts_and_ids_differ_in_length_refuses():
    broken = rows()
    broken[0]["texts"] = broken[0]["texts"][:-1]
    with pytest.raises(judge.JudgeError, match="q0"):
        judge.score_rows(StubModel(), broken, batch_size=8)


def test_truncations_count_pairs_longer_than_the_maximum_with_special_tokens():
    texts = ["a b c", "a b c d e f", "a"]
    # Question: 2 tokens. Lengths with 3 specials: 8, 11, 6.
    counted = judge.truncations(StubTokenizer(), "who is", texts, max_length=8)

    assert counted == {"pairs": 3, "truncated": 1, "longest": 11}
    assert judge.truncations(StubTokenizer(), "who is", texts, max_length=11)["truncated"] == 0


def test_the_pins_are_full_revisions_and_weight_digests():
    for key, pin in config.PHASE_17_JUDGES.items():
        assert key in (config.PHASE_17_LIGHT, config.PHASE_17_STRONG)
        assert len(pin["revision"]) == 40 and int(pin["revision"], 16) >= 0
        assert len(pin["weights_sha256"]) == 64 and int(pin["weights_sha256"], 16) >= 0
    light = config.PHASE_17_JUDGES[config.PHASE_17_LIGHT]
    strong = config.PHASE_17_JUDGES[config.PHASE_17_STRONG]
    # The plan's prefixes and suffixes, measured on 2026-10-01.
    assert light["revision"].startswith("233902d2") and light["revision"].endswith("ee0a")
    assert light["weights_sha256"].startswith("821d1aa6")
    assert light["weights_sha256"].endswith("b0ae")
    assert strong["revision"].startswith("953dc6f6") and strong["revision"].endswith("d41e")
    assert strong["weights_sha256"].startswith("d9e3e081")
    assert strong["weights_sha256"].endswith("5286")
    assert (light["max_length"], light["batch_size"]) == (512, 256)
    assert (strong["max_length"], strong["batch_size"]) == (8192, 32)


def test_a_served_snapshot_other_than_the_pin_refuses_before_any_model_loads(tmp_path: Path):
    pin = config.PHASE_17_JUDGES[config.PHASE_17_LIGHT]
    wrong = tmp_path / ("0" * 40)
    wrong.mkdir()
    (wrong / config.PHASE_17_WEIGHTS_FILE).write_bytes(b"weights")
    with pytest.raises(judge.JudgeError, match="revision"):
        judge.check_snapshot(pin, wrong)

    right = tmp_path / pin["revision"]
    right.mkdir()
    (right / config.PHASE_17_WEIGHTS_FILE).write_bytes(b"other weights")
    with pytest.raises(judge.JudgeError, match="SHA-256"):
        judge.check_snapshot(pin, right)
