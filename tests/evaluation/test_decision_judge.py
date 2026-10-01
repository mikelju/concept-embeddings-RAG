"""Phase 17, S3 (deviation 17.1): J-decision's parts against stubs. The logit is the scale
times the dot product of the projected vectors, each unit's vector stays with its own id
through the sort, truncation is counted with the encoder's tokenizer, a non-finite value
refuses, and a heads file whose SHA-256 is not the pin's is refused before `torch.load`."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import decision_judge as dj
from concept_embeddings_rag.evaluation.judge import JudgeError

HIDDEN, PROJ = 8, 4


def vector_of(text: str) -> np.ndarray:
    """A deterministic unit vector named by the text: a misaligned unit changes the logit."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    v = np.random.default_rng(seed).normal(size=HIDDEN).astype(np.float32)
    return v / np.linalg.norm(v)


class StubEmbedder:
    def __init__(self, *, nan_on: str | None = None) -> None:
        self.batch = 32
        self.sent: list[list[str]] = []
        self.nan_on = nan_on

    def embed(self, texts):
        self.sent.append(list(texts))
        vectors = np.stack([vector_of(t) for t in texts])
        for i, t in enumerate(texts):
            if t == self.nan_on:
                vectors[i] = np.nan
        return vectors, 7 * len(texts)


def stub_heads(device: str = "cpu") -> dj.LoadedHeads:
    torch.manual_seed(0)
    state, action = torch.nn.Linear(HIDDEN, PROJ), torch.nn.Linear(HIDDEN, PROJ)
    return dj.LoadedHeads(
        state_head=state.eval(),
        action_head=action.eval(),
        scale=12.5,
        cfg={"projection_dim": PROJ},
        parameters=sum(p.numel() for m in (state, action) for p in m.parameters()),
        path=Path("snapshots") / config.PHASE_17_JUDGES["decision"]["revision"] / "heads.pt",
        sha256="s" * 64,
        bytes=1,
        device=device,
    )


def a_judge(embedder: StubEmbedder | None = None) -> dj.DecisionJudge:
    return dj.DecisionJudge(
        judge_pin=config.PHASE_17_JUDGES["decision"],
        heads=stub_heads(),
        embedder=embedder or StubEmbedder(),
        encoder={"revision": "e" * 40},
        served_revision="r" * 40,
        serving={},
    )


def reference_logit(judge: dj.DecisionJudge, question: str, text: str) -> float:
    def normalised(head, x):
        with torch.no_grad():
            y = head(torch.from_numpy(x))
        return torch.nn.functional.normalize(y, dim=-1).numpy()

    zs = normalised(judge.heads.state_head, vector_of(question.strip()))
    zc = normalised(judge.heads.action_head, vector_of(text))
    return float(judge.scale * float(zs @ zc))


def test_the_score_is_the_scale_times_the_dot_product_of_the_projected_vectors():
    judge = a_judge()
    states, _ = judge.embed_states(["  a question?  "])
    units, _ = judge.embed_units(["alpha", "beta"])

    got = judge.logits(states[0], units)

    assert got.dtype == np.float32 and got.shape == (2,)
    assert np.allclose(units @ states[0] * judge.scale, got, atol=1e-5)
    for value, text in zip(got, ["alpha", "beta"], strict=True):
        assert value == pytest.approx(reference_logit(judge, "a question?", text), abs=1e-4)
    assert np.allclose(np.linalg.norm(units, axis=1), 1.0, atol=1e-5)  # L2-normalised, 4-d here


def test_each_units_vector_stays_with_its_id_through_the_sort():
    judge = a_judge()
    rows = [
        {
            "qid": "q0",
            "question": "first?",
            "unit_ids": ["u9", "u1", "u5"],
            "texts": ["T9", "T1", "T5"],
        },
        {"qid": "q1", "question": "second?", "unit_ids": ["u5", "u2"], "texts": ["T5", "T2"]},
    ]
    ids = sorted({"u9": 1, "u1": 1, "u5": 1, "u2": 1})
    texts = {"u9": "T9", "u1": "T1", "u5": "T5", "u2": "T2"}
    vectors, _ = judge.embed_units([texts[i] for i in ids])
    position = {unit_id: k for k, unit_id in enumerate(ids)}

    scored, tokens = judge.score_rows(rows, vectors, position)

    assert tokens == 7 * 2  # the stub's reported tokens for the two states
    for row, scores in zip(rows, scored, strict=True):
        assert len(scores) == len(row["unit_ids"])
        for unit_id, value in zip(row["unit_ids"], scores, strict=True):
            assert value == pytest.approx(
                reference_logit(judge, row["question"], texts[unit_id]), abs=1e-4
            )


def test_units_are_sent_verbatim_in_chunks_and_the_request_size_is_restorable():
    embedder = StubEmbedder()
    judge = a_judge(embedder)
    texts = [f"  text {i}\n" for i in range(dj.EMBED_CHUNK + 3)]

    judge.embed_units(texts)

    assert [len(s) for s in embedder.sent] == [dj.EMBED_CHUNK, 3]
    assert embedder.sent[0][0] == "  text 0\n"  # verbatim: no strip on a unit
    with judge.request_size(1):
        assert embedder.batch == 1
    assert embedder.batch == 32


def test_a_non_finite_embedding_refuses():
    judge = a_judge(StubEmbedder(nan_on="bad"))
    with pytest.raises(JudgeError, match="non-finite"):
        judge.embed_units(["ok", "bad"])


def test_a_non_finite_logit_refuses():
    judge = a_judge()
    state = np.ones(PROJ, dtype=np.float32)
    units = np.full((2, PROJ), np.inf, dtype=np.float32)
    with pytest.raises(JudgeError, match="non-finite logits"):
        judge.logits(state, units)


class WordTokenizer:
    """One token per whitespace word, three special tokens."""

    def __call__(self, texts, add_special_tokens=True, truncation=False):
        assert add_special_tokens is True and truncation is False
        return {"input_ids": [[0] * (len(t.split()) + 3) for t in texts]}


def test_the_truncation_count_uses_the_tokenizer_with_special_tokens():
    long_text = "w " * 2046  # 2,046 words + 3 = 2,049 tokens: one over
    edge = "w " * 2045  # 2,048 tokens: kept
    rows = [
        {"qid": "q0", "question": "short?", "unit_ids": ["a", "b"], "texts": [long_text, edge]},
        {"qid": "q1", "question": " " + "q " * 2046, "unit_ids": ["b", "c"], "texts": [edge, "x"]},
    ]

    scan = dj.scan_lengths(rows, WordTokenizer(), batch=2)
    summary = dj.truncation_summary(scan, 2048)

    assert scan.units == {"a": 2049, "b": 2048, "c": 4}  # distinct: b counted once
    assert scan.states == [4, 2049]
    assert summary["units"] == {"texts": 3, "truncated": 1, "share": 1 / 3, "longest": 2049}
    assert summary["states"]["truncated"] == 1
    assert summary["count"] == 2 and summary["share"] == pytest.approx(2 / 5)
    assert summary["longest"] == 2049


def test_one_unit_id_with_two_texts_refuses():
    rows = [
        {"qid": "q0", "question": "a?", "unit_ids": ["a"], "texts": ["one"]},
        {"qid": "q1", "question": "b?", "unit_ids": ["a"], "texts": ["two"]},
    ]
    with pytest.raises(JudgeError, match="two different texts"):
        dj.scan_lengths(rows, WordTokenizer())


# --- The heads file: digest before torch.load ----------------------------------------------


def a_pin(path: Path, *, sha256: str | None = None) -> dict:
    return {
        **config.PHASE_17_JUDGES["decision"],
        "weights_sha256": sha256 or hashlib.sha256(path.read_bytes()).hexdigest(),
        "weights_bytes": path.stat().st_size,
    }


def stub_make_head(width, depth, proj, activation, layernorm, residual, hidden):
    return torch.nn.Linear(hidden, proj)


def write_checkpoint(path: Path) -> None:
    head = torch.nn.Linear(HIDDEN, PROJ)
    torch.save(
        {
            "state_head": head.state_dict(),
            "action_head": head.state_dict(),
            "logit_scale": torch.tensor(5.0),  # exp = 148 -> clamped at 100
            "cfg": {"width": 6, "depth": 2, "hidden_size": HIDDEN, "projection_dim": PROJ},
        },
        path,
    )


def test_a_wrong_heads_digest_refuses_before_torch_load_is_reached(tmp_path, monkeypatch):
    path = tmp_path / "heads.pt"
    write_checkpoint(path)
    reached: list[object] = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: reached.append(a))

    with pytest.raises(JudgeError, match="SHA-256"):
        dj.load_heads(a_pin(path, sha256="0" * 64), path=path, make_head=stub_make_head)

    assert reached == []


def test_a_wrong_heads_size_refuses_before_torch_load_is_reached(tmp_path, monkeypatch):
    path = tmp_path / "heads.pt"
    write_checkpoint(path)
    pin = {**a_pin(path), "weights_bytes": path.stat().st_size + 1}
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("torch.load was reached"))

    with pytest.raises(JudgeError, match="bytes"):
        dj.load_heads(pin, path=path, make_head=stub_make_head)


def test_the_pinned_heads_load_weights_only_float32_eval_with_the_clamped_scale(
    tmp_path, monkeypatch
):
    path = tmp_path / "heads.pt"
    write_checkpoint(path)
    real_load = torch.load
    kwargs_seen: list[dict] = []

    def spy(*args, **kwargs):
        kwargs_seen.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)

    heads = dj.load_heads(a_pin(path), path=path, make_head=stub_make_head)

    assert kwargs_seen == [{"map_location": "cpu", "weights_only": True}]
    assert heads.scale == 100.0
    assert heads.cfg["projection_dim"] == PROJ
    assert heads.parameters == 2 * (HIDDEN * PROJ + PROJ)
    assert not heads.state_head.training and not heads.action_head.training
    assert all(p.dtype == torch.float32 for p in heads.state_head.parameters())
    assert heads.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_downloaded_heads_file_from_another_revision_refuses(tmp_path):
    path = tmp_path / "snapshots" / ("f" * 40) / "heads.pt"
    path.parent.mkdir(parents=True)
    write_checkpoint(path)
    with pytest.raises(JudgeError, match="revision"):
        dj.load_heads(a_pin(path), download=lambda _pin: path, make_head=stub_make_head)


def test_the_encoder_shards_are_checked_against_their_pins(tmp_path):
    files = {f"model-0000{i}-of-00002.safetensors": f"shard {i}".encode() for i in (1, 2)}
    for name, body in files.items():
        (tmp_path / name).write_bytes(body)
    pin = {
        "encoder": {
            "name": "Qwen/Qwen3-8B",
            "revision": "b" * 40,
            "shards": {
                n: {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
                for n, b in files.items()
            },
        }
    }

    checked = dj.check_encoder(pin, tmp_path)
    assert set(checked["shards_verified"]) == set(files) and checked["total_bytes"] == 14

    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"shard X")  # same size
    with pytest.raises(JudgeError, match="SHA-256"):
        dj.check_encoder(pin, tmp_path)
