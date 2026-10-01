"""Phase 17, S2: `cer p17-pool` on synthetic sets. The two loaders that read the real Phase
9-16 artifacts are replaced by fixtures; everything after them (D3, the replays, the pools,
the pairs join over a corpus read by `fullwiki.load_corpus`, the manifest) is the stage's
own code."""

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from concept_embeddings_rag import cli, config
from concept_embeddings_rag.artifacts import digest_of
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question, unit_id_for
from concept_embeddings_rag.evaluation import phase17

P10A, P10B, P10C, P14 = config.PHASE_17_SYSTEMS
SETS = tuple(config.PHASE_17_SETS)


def write_corpus(directory: Path, units: list[IndexingUnit]) -> None:
    """A corpus in the format `fullwiki.load_corpus` reads (and proves) for all three sets."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        json.dumps({"unit_id": u.unit_id, "title": u.title, "sentences": list(u.sentences)}) + "\n"
        for u in units
    )
    (directory / "corpus.jsonl.gz").write_bytes(gzip.compress(lines.encode("utf-8")))
    manifest = {
        "corpus_file": "corpus.jsonl.gz",
        "ordered_unit_digest": digest_of(*(u.unit_id for u in units)),
    }
    (directory / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")


def a_unit(set_name: str, number: int) -> IndexingUnit:
    title, sentences = f"{set_name} title {number}", (f"Sentence {number} of {set_name}.",)
    return IndexingUnit(unit_id_for(title, sentences), title, sentences)


def fake_set(set_name: str, units: list[IndexingUnit], *, miss: bool = False) -> dict[str, Any]:
    """Questions, lists and counts shaped as the two real loaders return them."""
    n = config.PHASE_17_SETS[set_name]
    ids = [u.unit_id for u in units]
    questions = [
        Question(f"{set_name}-q{i}", f"question {i} of {set_name}?", "", (ids[i % 5],), (), "x")
        for i in range(n)
    ]
    lists = {
        P10A: [ids[0:3]] * n,
        P10B: [ids[2:5]] * n,
        P10C: [[ids[5], ids[0]]] * n,
        P14: [[ids[6]]] * n,
    }
    observed = {
        **config.PHASE_17_D3_SUPPORTED[set_name],
        **config.PHASE_17_REPLAYS_SUPPORTED[set_name],
    }
    if miss:
        observed[P14] -= 1
    return {
        "questions": questions,
        "token_counts": {u: 10 + k for k, u in enumerate(ids)},
        "lists": lists,
        "observed": observed,
        "extra_reasons": [],
        "evidence": {"fixture": set_name},
        "sources": {"fixture": {"path": set_name, "sha256": "0" * 64, "recorded_digest": None}},
    }


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """Synthetic corpora for the three sets and the loaders replaced; returns a runner."""
    units = {s: [a_unit(s, k) for k in range(8)] for s in SETS}
    dirs = {s: tmp_path / f"corpus-{s}" for s in SETS}
    for s in SETS:
        write_corpus(dirs[s], units[s])
    misses: set[str] = set()
    monkeypatch.setattr(
        cli,
        "_p17_hotpotqa",
        lambda *_a: fake_set(config.PHASE_17_HOTPOTQA, units[config.PHASE_17_HOTPOTQA]),
    )
    monkeypatch.setattr(
        cli,
        "_p17_recorded",
        lambda set_name, _d, _s: fake_set(set_name, units[set_name], miss=set_name in misses),
    )
    target = tmp_path / "phase17"

    def run() -> Path:
        return cli.cmd_p17_pool(
            target,
            phase9_dir=dirs[config.PHASE_17_HOTPOTQA],
            phase15_dir=dirs[config.PHASE_17_MUSIQUE],
            phase16_dir=dirs[config.PHASE_17_MULTIHOP_RAG],
        )

    return {"run": run, "target": target, "units": units, "misses": misses}


def test_the_stage_writes_integrity_pools_pairs_and_the_manifest_once(stage):
    path = stage["run"]()
    target: Path = stage["target"]

    integrity = json.loads((target / "integrity.json").read_text(encoding="utf-8"))
    assert integrity["terminal_state"] is None
    assert integrity["recorded_replays"]["passed"] is True
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["integrity_digest"] == digest_of(
        (target / "integrity.json").read_text(encoding="utf-8")
    )
    for set_name in SETS:
        block = manifest["sets"][set_name]
        # Pool per question: ids 0-6, seven units, every question the same.
        assert block["size"]["max"] == block["size"]["min"] == 7
        assert block["pairs"] == 7 * config.PHASE_17_SETS[set_name]
        assert block["longest_unit"]["budget_tokens"] == 16
        pool = phase17.read_pool(target, set_name, digest=block["pool_file"]["digest"])
        assert len(pool) == config.PHASE_17_SETS[set_name]
        assert (
            phase17.sha256_file(target / phase17.pairs_name(set_name))
            == (block["pairs_file"]["sha256"])
        )
    assert manifest["judges"] == config.PHASE_17_JUDGES

    with pytest.raises(SystemExit, match="written once"):
        stage["run"]()


def test_the_pairs_join_puts_each_units_indexable_text_beside_its_own_id(stage):
    stage["run"]()
    target: Path = stage["target"]
    manifest = json.loads((target / "pool.json").read_text(encoding="utf-8"))
    for set_name in SETS:
        by_id = {u.unit_id: u.indexable_text for u in stage["units"][set_name]}
        rows = phase17.read_pairs(
            target, set_name, digest=manifest["sets"][set_name]["pairs_file"]["digest"]
        )
        assert rows[0]["question"] == f"question 0 of {set_name}?"
        for row in rows:
            assert len(row["unit_ids"]) == len(row["texts"]) == 7
            for unit_id, text in zip(row["unit_ids"], row["texts"], strict=True):
                assert text == by_id[unit_id]
        # The pool order: best rank, then id.
        ids = [u.unit_id for u in stage["units"][set_name]]
        firsts = sorted([ids[0], ids[2], ids[5], ids[6]])
        assert rows[0]["unit_ids"][:4] == firsts


def test_a_d3_miss_writes_data_stop_exits_non_zero_and_writes_no_pool(stage):
    stage["misses"].add(config.PHASE_17_MULTIHOP_RAG)

    with pytest.raises(SystemExit, match="DATA_STOP"):
        stage["run"]()

    target: Path = stage["target"]
    integrity = json.loads((target / "integrity.json").read_text(encoding="utf-8"))
    assert integrity["terminal_state"] == "DATA_STOP"
    assert integrity["stop_reasons"] == ["multihop-rag P14: 512 of 2255, recorded 513"]
    assert sorted(p.name for p in target.iterdir()) == ["integrity.json"]


def test_a_replay_miss_outside_d3_stops_with_nothing_written(stage, monkeypatch):
    original = cli._p17_recorded

    def missing_replay(set_name, directory, source):
        data = original(set_name, directory, source)
        if set_name == config.PHASE_17_MUSIQUE:
            data["observed"][P10A] = 435
        return data

    monkeypatch.setattr(cli, "_p17_recorded", missing_replay)

    with pytest.raises(SystemExit, match="for the author"):
        stage["run"]()

    assert list(stage["target"].iterdir()) == []


# --- S3: cer p17-score, the cross-encoders' path ----------------------------------------------


class ScoreStubModel:
    """A deterministic hash-based scorer; `predict` counts its calls."""

    max_length = 512

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.tokenizer = self

    def __call__(self, questions, texts, truncation=False, add_special_tokens=True):
        return {
            "input_ids": [
                [0] * (len(t.split()) + len(q.split()) + 3)
                for q, t in zip(questions, texts, strict=True)
            ]
        }

    def predict(self, inputs, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        import hashlib

        import numpy as np

        self.calls.append(len(inputs))
        return np.asarray(
            [
                int.from_bytes(hashlib.sha256((q + "|" + t).encode()).digest()[:4], "big") / 1e8
                for q, t in inputs
            ],
            dtype=np.float32,
        )


@pytest.fixture
def scoring(stage, monkeypatch):
    """The p17-pool chain built once, a stub judge, and a runner for `cmd_p17_score`."""
    from concept_embeddings_rag.evaluation import judge as judge_module

    stage["run"]()
    model = ScoreStubModel()

    def fake_load(key, **_kw):
        return judge_module.LoadedJudge(
            key=key,
            pin=config.PHASE_17_JUDGES[key],
            model=model,
            snapshot=Path("snap"),
            served_revision=config.PHASE_17_JUDGES[key]["revision"],
            weights_sha256=config.PHASE_17_JUDGES[key]["weights_sha256"],
        )

    monkeypatch.setattr(judge_module, "load", fake_load)
    monkeypatch.setattr(
        judge_module,
        "identity",
        lambda loaded: {
            "key": loaded.key,
            "kind": loaded.pin["kind"],
            "max_length": 512,
            "revision_served": loaded.served_revision,
            "weights_sha256": loaded.weights_sha256,
        },
    )
    set_name = config.PHASE_17_MULTIHOP_RAG

    def run(
        *, judge_key="light", tag=None, smoke=None, committed=True, shard_questions=300, **extra
    ):
        return cli.cmd_p17_score(
            judge_key,
            set_name,
            hourly_rate_usd=1.0,
            smoke=smoke,
            tag=tag,
            target_dir=stage["target"],
            shard_questions=shard_questions,
            is_committed=lambda _p: committed,
            **extra,
        )

    return {**stage, "model": model, "score": run, "set": set_name}


def test_a_pairs_digest_mismatch_refuses_before_any_pair_is_scored(scoring):
    pool_path = scoring["target"] / "pool.json"
    manifest = json.loads(pool_path.read_text(encoding="utf-8"))
    manifest["sets"][scoring["set"]]["pairs_file"]["sha256"] = "0" * 64
    pool_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(phase17.Phase17Error, match="SHA-256"):
        scoring["score"]()

    assert scoring["model"].calls == []
    assert not (scoring["target"] / "shards").exists()


def test_an_uncommitted_chain_refuses(scoring):
    with pytest.raises(phase17.Phase17Error, match="committed"):
        scoring["score"](committed=False)
    assert scoring["model"].calls == []


def test_a_shard_left_by_an_interrupted_run_is_skipped_and_the_file_is_the_same(scoring):
    target: Path = scoring["target"]
    scoring["score"]()
    one_pass = (target / phase17.scores_name("light", scoring["set"])).read_bytes()
    first_calls = list(scoring["model"].calls)
    n_shards = len(phase17.shard_bounds(config.PHASE_17_SETS[scoring["set"]], 300))
    assert first_calls[:n_shards] == [
        7 * (stop - start)
        for start, stop in phase17.shard_bounds(config.PHASE_17_SETS[scoring["set"]], 300)
    ]

    # Interrupt: the assembled file, the manifest and the last shard are gone.
    (target / phase17.scores_name("light", scoring["set"])).unlink()
    (target / "scoring-light.json").unlink()
    shard_dir = target / "shards" / f"light-{scoring['set']}"
    (shard_dir / phase17.shard_name(n_shards - 1)).unlink()
    scoring["model"].calls.clear()

    scoring["score"]()

    assert scoring["model"].calls[0] == 7 * (
        config.PHASE_17_SETS[scoring["set"]] - 300 * (n_shards - 1)
    )
    assert (target / phase17.scores_name("light", scoring["set"])).read_bytes() == one_pass
    block = json.loads((target / "scoring-light.json").read_text(encoding="utf-8"))["sets"][
        scoring["set"]
    ]
    assert block["restarts"] == 1
    assert block["shards"]["scored_now"] == 1


def test_an_existing_scores_file_refuses_and_a_tag_writes_distinct_files_and_block(scoring):
    target: Path = scoring["target"]
    scoring["score"]()
    with pytest.raises(phase17.Phase17Error, match="exists"):
        scoring["score"]()

    scoring["score"](tag="rerun")

    plain = phase17.scores_name("light", scoring["set"])
    tagged = phase17.scores_name("light", scoring["set"], tag="rerun")
    assert (target / tagged).exists() and tagged != plain
    sets = json.loads((target / "scoring-light.json").read_text(encoding="utf-8"))["sets"]
    assert set(sets) == {scoring["set"], f"{scoring['set']}-rerun"}


def test_the_manifest_block_carries_pairs_shards_truncations_and_determinism(scoring):
    target: Path = scoring["target"]
    scoring["score"]()
    manifest = json.loads((target / "scoring-light.json").read_text(encoding="utf-8"))
    block = manifest["sets"][scoring["set"]]
    n = config.PHASE_17_SETS[scoring["set"]]

    assert manifest["judge"]["revision_served"] == config.PHASE_17_JUDGES["light"]["revision"]
    assert manifest["score"] == "logit" and "allocation" in manifest and "host" in manifest
    assert block["pairs"] == 7 * n and block["questions"] == n
    assert block["shards"]["count"] == len(phase17.shard_bounds(n, 300))
    assert block["truncations"]["count"] == 0 and block["truncations"]["longest"] > 0
    assert block["determinism"]["questions"] == config.PHASE_17_DETERMINISM_QUESTIONS
    assert block["determinism"]["max_abs_difference"] == 0.0
    assert block["determinism"]["questions_order_changed"] == 0
    assert (
        block["pairs_sha256_read"]
        == json.loads((target / "pool.json").read_text(encoding="utf-8"))["sets"][scoring["set"]][
            "pairs_file"
        ]["sha256"]
    )
    rows = phase17.read_jsonl(
        target / phase17.scores_name("light", scoring["set"]),
        digest=block["scores_file"]["digest"],
        sha256=block["scores_file"]["sha256"],
    )
    assert len(rows) == n and rows[0]["qid"] == f"{scoring['set']}-q0"
    assert all(isinstance(x, float) for x in rows[0]["scores"])


def test_a_smoke_writes_into_the_smoke_directory_only(scoring):
    path = scoring["score"](smoke=3, committed=False)

    assert path.parent.name == "smoke"
    assert len(phase17.read_jsonl(path)) == 3
    assert not (scoring["target"] / "scoring-light.json").exists()
    assert not (scoring["target"] / "shards").exists()


# --- S3, deviation 17.1: the decision judge's path ---------------------------------------------

DECISION = config.PHASE_17_DECISION


def vector_of(text: str):
    """A deterministic unit vector named by the text: a misaligned unit changes the logit."""
    import hashlib

    import numpy as np

    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    v = np.random.default_rng(seed).normal(size=8).astype(np.float32)
    return v / np.linalg.norm(v)


def a_judge(embedder):
    """The decision judge over small linear heads and a stub embedder (no vLLM, no clm)."""
    import torch

    from concept_embeddings_rag.evaluation import decision_judge as dj_module

    torch.manual_seed(0)
    state, action = torch.nn.Linear(8, 4), torch.nn.Linear(8, 4)
    heads = dj_module.LoadedHeads(
        state_head=state.eval(),
        action_head=action.eval(),
        scale=12.5,
        cfg={"projection_dim": 4},
        parameters=2 * (8 * 4 + 4),
        path=Path("snapshots") / config.PHASE_17_JUDGES[DECISION]["revision"] / "heads.pt",
        sha256="s" * 64,
        bytes=1,
        device="cpu",
    )
    return dj_module.DecisionJudge(
        judge_pin=config.PHASE_17_JUDGES[DECISION],
        heads=heads,
        embedder=embedder,
        encoder={"revision": "e" * 40},
        served_revision=heads.path.parent.name,
        serving={},
    )


def reference_logit(judge, question: str, text: str) -> float:
    import torch

    def z(head, x):
        with torch.no_grad():
            return torch.nn.functional.normalize(head(torch.from_numpy(x)), dim=-1).numpy()

    zs, zc = (
        z(judge.heads.state_head, vector_of(question.strip())),
        z(judge.heads.action_head, vector_of(text)),
    )
    return float(judge.scale * float(zs @ zc))


class WordTokenizer:
    """One token per whitespace word plus three special tokens."""

    def __call__(self, texts, add_special_tokens=True, truncation=False):
        return {"input_ids": [[0] * (len(t.split()) + 3) for t in texts]}


class DecisionEmbedder:
    batch = 32

    def __init__(self) -> None:
        self.sent: list[list[str]] = []

    def embed(self, texts):
        import numpy as np

        self.sent.append(list(texts))
        return np.stack([vector_of(t) for t in texts]), 5 * len(texts)


@pytest.fixture
def decision(scoring, monkeypatch):
    from concept_embeddings_rag.evaluation import decision_judge as dj_module

    embedder = DecisionEmbedder()
    loads: list[dict] = []

    def fake_load(**kwargs):
        loads.append(kwargs)
        return a_judge(embedder)

    monkeypatch.setattr(dj_module, "load", fake_load)
    monkeypatch.setattr(dj_module, "load_tokenizer", lambda _pin: WordTokenizer())

    def run(**kwargs):
        kwargs.setdefault("vllm_command", "vllm serve Qwen/Qwen3-8B --runner pooling")
        kwargs.setdefault("pooler", "LAST")
        return scoring["score"](judge_key=DECISION, **kwargs)

    return {**scoring, "embedder": embedder, "loads": loads, "run": run}


def test_a_decision_record_run_needs_the_serve_command_and_the_pooler(decision):
    with pytest.raises(phase17.Phase17Error, match="--vllm-command"):
        decision["run"](vllm_command=None)
    with pytest.raises(phase17.Phase17Error, match="--pooler"):
        decision["run"](pooler=None)
    assert decision["loads"] == [] and decision["embedder"].sent == []


def test_a_decision_pairs_digest_mismatch_refuses_before_anything_is_embedded(decision):
    pool_path = decision["target"] / "pool.json"
    manifest = json.loads(pool_path.read_text(encoding="utf-8"))
    manifest["sets"][decision["set"]]["pairs_file"]["sha256"] = "0" * 64
    pool_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(phase17.Phase17Error, match="SHA-256"):
        decision["run"]()

    assert decision["loads"] == [] and decision["embedder"].sent == []


def test_the_decision_logits_are_the_scale_times_the_dot_of_the_projected_vectors(decision):
    target: Path = decision["target"]
    decision["run"]()
    block = json.loads((target / "scoring-decision.json").read_text(encoding="utf-8"))["sets"][
        decision["set"]
    ]
    rows = phase17.read_jsonl(target / phase17.scores_name(DECISION, decision["set"]))
    pairs = phase17.read_jsonl(target / phase17.pairs_name(decision["set"]))
    judge = a_judge(DecisionEmbedder())

    assert len(rows) == config.PHASE_17_SETS[decision["set"]] == block["questions"]
    for row, pair in zip(rows[:3], pairs[:3], strict=True):
        assert row["unit_ids"] == pair["unit_ids"]
        for value, text in zip(row["scores"], pair["texts"], strict=True):
            assert value == pytest.approx(reference_logit(judge, pair["question"], text), abs=1e-4)


def test_a_decision_shard_left_by_an_interrupted_run_is_skipped_and_the_file_is_the_same(
    decision,
):
    target: Path = decision["target"]
    decision["run"]()
    name = phase17.scores_name(DECISION, decision["set"])
    one_pass = (target / name).read_bytes()
    n = config.PHASE_17_SETS[decision["set"]]
    n_shards = len(phase17.shard_bounds(n, 300))
    (target / name).unlink()
    (target / "scoring-decision.json").unlink()
    shard_dir = target / "shards" / f"decision-{decision['set']}"
    (shard_dir / phase17.shard_name(n_shards - 1)).unlink()
    decision["embedder"].sent.clear()

    decision["run"]()

    assert (target / name).read_bytes() == one_pass
    # The restart's offline pass embeds the distinct units of the one missing shard (7).
    first = decision["embedder"].sent[0]
    assert len(first) == 7 and len(set(first)) == 7
    block = json.loads((target / "scoring-decision.json").read_text(encoding="utf-8"))["sets"][
        decision["set"]
    ]
    assert block["restarts"] == 1 and block["shards"]["scored_now"] == 1
    assert block["offline"]["passes"] == 2 and block["offline"]["units_embedded"] == 14
    assert block["offline"]["distinct_units"] == 7


def test_the_decision_manifest_carries_the_judge_the_serving_and_the_block(decision):
    target: Path = decision["target"]
    decision["run"](emb_url="http://pod:8090/v1/embeddings", emb_model="qwen3-8b")
    manifest = json.loads((target / "scoring-decision.json").read_text(encoding="utf-8"))
    block = manifest["sets"][decision["set"]]
    n = config.PHASE_17_SETS[decision["set"]]

    assert manifest["kind"] == "bi-encoder" and manifest["score"] == "logit"
    assert manifest["scale"] == 12.5 and manifest["judge"]["heads"]["scale"] == 12.5
    assert manifest["serving"]["vllm_command"] == "vllm serve Qwen/Qwen3-8B --runner pooling"
    assert manifest["serving"]["pooler"] == "LAST"
    assert manifest["serving"]["request_size"] == 32 == manifest["judge"]["batch_size"]
    assert manifest["serving"]["max_tokens"] == 2048 and manifest["serving"]["cache_size"] == 0
    assert manifest["heads"]["sha256"] == manifest["judge"]["weights_sha256"]
    assert "library_versions" in manifest and "host" in manifest and "allocation" in manifest
    assert decision["loads"][0]["emb_url"] == "http://pod:8090/v1/embeddings"
    assert block["pairs"] == 7 * n and block["questions"] == n
    assert block["offline"]["units_embedded"] == 7 == block["offline"]["distinct_units"]
    assert block["offline"]["tokens_reported"] == 35 and block["offline"]["tokens_expected"] > 0
    assert block["query"]["questions"] == n and block["query"]["state_tokens_reported"] == 5 * n
    assert block["truncations"]["count"] == 0 and block["truncations"]["longest"] > 0
    assert block["determinism"]["questions"] == config.PHASE_17_DETERMINISM_QUESTIONS
    assert block["determinism"]["request_size"] == 1
    assert block["determinism"]["max_abs_difference"] < 1e-4
    assert block["determinism"]["questions_order_changed"] == 0
    assert block["shards"]["count"] == len(phase17.shard_bounds(n, 300))
    rows = phase17.read_jsonl(
        target / phase17.scores_name(DECISION, decision["set"]),
        digest=block["scores_file"]["digest"],
        sha256=block["scores_file"]["sha256"],
    )
    assert rows[0]["qid"] == f"{decision['set']}-q0"
    with pytest.raises(phase17.Phase17Error, match="exists"):
        decision["run"]()


def test_a_decision_smoke_writes_into_the_smoke_directory_only(decision):
    path = decision["run"](smoke=3, committed=False, vllm_command=None, pooler=None)

    assert path.parent.name == "smoke" and len(phase17.read_jsonl(path)) == 3
    assert (path.parent / f"scoring-decision-{decision['set']}.json").exists()
    assert not (decision["target"] / "scoring-decision.json").exists()
    assert not (decision["target"] / "shards").exists()


def test_truncation_only_counts_with_the_tokenizer_alone_and_loads_no_judge(decision):
    target: Path = decision["target"]
    path = decision["score"](judge_key=DECISION, committed=False, truncation_only=True)

    body = json.loads(path.read_text(encoding="utf-8"))
    n = config.PHASE_17_SETS[decision["set"]]
    assert path == target / f"truncation-decision-{decision['set']}.json"
    assert body["judge"] == DECISION and body["questions"] == n
    assert body["units"]["texts"] == 7 and body["states"]["texts"] == n
    assert body["count"] == 0 and body["max_length"] == 2048 and body["longest"] > 0
    assert body["tokenizer"]["revision"] == config.PHASE_17_JUDGES[DECISION]["encoder"]["revision"]
    assert decision["loads"] == [] and decision["embedder"].sent == []
    with pytest.raises(phase17.Phase17Error, match="exists"):
        decision["score"](judge_key=DECISION, truncation_only=True)


def test_truncation_only_is_the_decision_judges_alone(scoring):
    with pytest.raises(phase17.Phase17Error, match="decision"):
        scoring["score"](judge_key="light", truncation_only=True)


# --- S3: cer p17-outcome, written before any score exists ---------------------------------------

JUDGES = (config.PHASE_17_LIGHT, config.PHASE_17_STRONG, config.PHASE_17_DECISION)
SMALL_SETS = {
    config.PHASE_17_HOTPOTQA: 40,
    config.PHASE_17_MUSIQUE: 33,
    config.PHASE_17_MULTIHOP_RAG: 27,
}
# One unit fits a 2,048-token context (1,500 each), so Full Support is "the list's first unit
# is the gold unit": the order, and so the judge, decides it.
TOKENS = 1500


def hand_lists(ids: list[str]) -> dict[str, list[str]]:
    return {
        P10A: ids[0:3],
        P10B: ids[2:5],
        P10C: [ids[5], ids[0]],
        P14: [ids[1], ids[6], ids[3]],
    }


def hand_counts(n: int) -> dict[str, int]:
    """Unjudged Full Support: questions whose gold `ids[i % 5]` heads the system's list."""
    first = {P10A: 0, P10B: 2, P10C: 5, P14: 1}
    return {s: sum(1 for i in range(n) if i % 5 == k) for s, k in first.items()}


@pytest.fixture
def shrunk(monkeypatch):
    """Tiny sets, a binding token budget and counts that match what the lists replay to."""
    original = fake_set

    def small_fake_set(set_name, units, *, miss=False):
        data = original(set_name, units, miss=miss)
        ids = [u.unit_id for u in units]
        n = len(data["questions"])
        data["lists"] = {**data["lists"], **{s: [lst] * n for s, lst in hand_lists(ids).items()}}
        data["token_counts"] = {u: TOKENS + k for k, u in enumerate(ids)}
        return data

    monkeypatch.setitem(globals(), "fake_set", small_fake_set)
    for set_name, n in SMALL_SETS.items():
        monkeypatch.setitem(config.PHASE_17_SETS, set_name, n)
        counts = hand_counts(n)
        for table in (config.PHASE_17_D3_SUPPORTED, config.PHASE_17_REPLAYS_SUPPORTED):
            monkeypatch.setitem(table, set_name, {s: counts[s] for s in table[set_name]})
    return small_fake_set


@pytest.fixture
def chain(shrunk, decision, monkeypatch):
    """The pool, nine scorings (stub judges, all three sets) and a runner for the outcome."""
    target: Path = decision["target"]
    for judge_key in JUDGES:
        for set_name in SETS:
            cli.cmd_p17_score(
                judge_key,
                set_name,
                hourly_rate_usd=1.0,
                target_dir=target,
                shard_questions=300,
                is_committed=lambda _p: True,
                vllm_command="vllm serve stub",
                pooler="LAST",
            )

    def replay_inputs(set_name, _directory):
        data = shrunk(set_name, decision["units"][set_name])
        return data["questions"], data["token_counts"]

    monkeypatch.setattr(cli, "_p17_replay_inputs", replay_inputs)

    def run(*, committed=True):
        return cli.cmd_p17_outcome(target, phase9_dir=target, is_committed=lambda _p: committed)

    return {**decision, "outcome": run}


def hand_head(units: list[str], scores: dict[str, float]) -> str:
    """The unit a judge puts first: highest score, ties by the system's own order."""
    return sorted(units, key=lambda unit: (-scores[unit], units.index(unit)))[0]


def exact_p(wins: int, losses: int) -> float:
    from math import comb

    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)


def hand_label(wins: int, losses: int) -> str:
    adds, hurts, neutral = config.PHASE_17_LABELS
    p = exact_p(wins, losses)
    if wins > losses and p < 0.05:
        return adds
    return hurts if losses > wins and p < 0.05 else neutral


def nothing_written(target: Path) -> bool:
    return not (
        list(target.glob("reordered-*"))
        or (target / "outcome.json").exists()
        or (target / "metrics.json").exists()
    )


def test_the_outcome_writes_fifteen_reordered_files_per_set_and_two_json_files(chain):
    target: Path = chain["target"]
    path = chain["outcome"]()

    assert path == target / "outcome.json" and (target / "metrics.json").exists()
    for set_name in SETS:
        assert len(list(target.glob(f"reordered-*-{set_name}-*.jsonl.gz"))) == 15
    assert len(list(target.glob("reordered-*.jsonl.gz"))) == 45
    outcome = json.loads(path.read_text(encoding="utf-8"))
    assert outcome["terminal_state"] is None and outcome["gate"] is None
    for set_name, block in outcome["sets"].items():
        assert block["in_sample_weights"] is (set_name == config.PHASE_17_HOTPOTQA)
        assert set(block["per_judge"]) == set(JUDGES)
        assert all(len(by) == 5 for by in block["per_judge"].values())
        assert set(block["cross_judge"]) == {"strong_vs_light_p10b", "decision_vs_strong_p10b"}
        assert block["replayed_lines"] == 19
        record = outcome["provenance"]["reordered_files"][set_name][f"light:{P14}"]
        assert record["sha256"] == phase17.sha256_file(
            target / phase17.reordered_name("light", set_name, P14)
        )
        assert record["rows"] == SMALL_SETS[set_name]
    metrics = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
    union = f"strong:{config.PHASE_17_UNION}"
    assert metrics["sets"][config.PHASE_17_MULTIHOP_RAG]["lines"][union]["descriptive"] is True
    for block in metrics["sets"].values():
        assert len(block["lines"]) == 19
        assert all(line["harness_mismatches"] == [] for line in block["lines"].values())


def test_the_primary_comparison_and_label_equal_a_hand_count_from_the_scores(chain):
    target: Path = chain["target"]
    outcome = json.loads(chain["outcome"]().read_text(encoding="utf-8"))
    for set_name in SETS:
        ids = [u.unit_id for u in chain["units"][set_name]]
        lists = hand_lists(ids)
        for judge_key in JUDGES:
            rows = phase17.read_jsonl(target / phase17.scores_name(judge_key, set_name))
            by_q = {r["qid"]: dict(zip(r["unit_ids"], r["scores"], strict=True)) for r in rows}

            def head(system, i, by_q=by_q, set_name=set_name, lists=lists):
                return hand_head(lists[system], by_q[f"{set_name}-q{i}"])

            wins = losses = 0
            for i in range(SMALL_SETS[set_name]):
                gold = ids[i % 5]
                control, candidate = head(P10B, i) == gold, head(P14, i) == gold
                wins += int(candidate and not control)
                losses += int(control and not candidate)
            result = outcome["sets"][set_name]["per_judge"][judge_key]["j_p14_vs_j_p10b"]
            assert (result["wins"], result["losses"]) == (wins, losses)
            assert result["exact_two_sided_p"] == pytest.approx(exact_p(wins, losses))
            label = outcome["sets"][set_name]["hop_label"][judge_key]
            assert label == hand_label(wins, losses)
            # J(P10-B) against P10-B, by hand as well.
            plain = sum(1 for i in range(SMALL_SETS[set_name]) if lists[P10B][0] == ids[i % 5])
            judged = sum(1 for i in range(SMALL_SETS[set_name]) if head(P10B, i) == ids[i % 5])
            counts = outcome["sets"][set_name]["full_support_2048"]
            assert counts[P10B]["supported"] == plain
            assert counts[f"{judge_key}:{P10B}"]["supported"] == judged


def test_the_unjudged_replay_equals_the_integrity_counts_and_a_judge_permutes_the_list(chain):
    target: Path = chain["target"]
    outcome = json.loads(chain["outcome"]().read_text(encoding="utf-8"))
    metrics = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
    integrity = json.loads((target / "integrity.json").read_text(encoding="utf-8"))
    for set_name in SETS:
        recorded = {
            **integrity["d3"][set_name],
            **integrity["recorded_replays"]["checks"][set_name],
        }
        counts = outcome["sets"][set_name]["full_support_2048"]
        lines = metrics["sets"][set_name]["lines"]
        for system in config.PHASE_17_SYSTEMS:
            assert counts[system]["supported"] == recorded[system]["observed"]
            assert counts[system]["supported"] == hand_counts(SMALL_SETS[set_name])[system]
            for judge_key in JUDGES:
                judged, plain = lines[f"{judge_key}:{system}"], lines[system]
                assert judged["full_support_at_k"]["100"] == plain["full_support_at_k"]["100"]
                assert judged["gold_recall_at_k"]["100"] == plain["gold_recall_at_k"]["100"]
                assert judged["ceiling"]["all_gold_inside"] == plain["ceiling"]["all_gold_inside"]
    set_name = config.PHASE_17_MULTIHOP_RAG
    record = outcome["provenance"]["reordered_files"][set_name][f"light:{P14}"]
    judged_lists = phase17.read_reordered(
        target, "light", set_name, P14, digest=record["digest"], sha256=record["sha256"]
    )
    ids = [u.unit_id for u in chain["units"][set_name]]
    for _qid, ranked in judged_lists:
        assert sorted(u for u, _s, _r in ranked) == sorted([ids[1], ids[6], ids[3]])
        assert [s for _u, s, _r in ranked] == sorted((s for _u, s, _r in ranked), reverse=True)


def test_a_tampered_scores_file_refuses_and_nothing_is_written(chain):
    target: Path = chain["target"]
    path = target / phase17.scores_name("strong", config.PHASE_17_MUSIQUE)
    path.write_bytes(gzip.compress(b'{"qid": "x", "scores": [], "unit_ids": []}\n'))

    with pytest.raises(phase17.Phase17Error, match="SHA-256"):
        chain["outcome"]()

    assert nothing_written(target)


def test_a_missing_judge_manifest_or_an_uncommitted_chain_refuses(chain):
    target: Path = chain["target"]
    with pytest.raises(phase17.Phase17Error, match="committed"):
        chain["outcome"](committed=False)
    (target / "scoring-decision.json").rename(target / "elsewhere.json")
    with pytest.raises(phase17.Phase17Error, match="every judge is reported"):
        chain["outcome"]()
    assert nothing_written(target)


def test_a_pool_that_does_not_replay_to_the_integrity_counts_refuses(chain):
    target: Path = chain["target"]
    path = target / "integrity.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["d3"][config.PHASE_17_MUSIQUE][P14]["observed"] += 1
    text = json.dumps(body)
    path.write_text(text, encoding="utf-8")
    pool_path = target / "pool.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    pool["integrity_digest"] = digest_of(text)
    pool_path.write_text(json.dumps(pool), encoding="utf-8")

    with pytest.raises(phase17.Phase17Error, match="integrity.json records"):
        chain["outcome"]()

    assert nothing_written(target)


def test_a_second_run_refuses(chain):
    chain["outcome"]()
    with pytest.raises(phase17.Phase17Error, match="written once"):
        chain["outcome"]()
