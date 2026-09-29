"""Phase 14 CLI: `p14-lists` refuses a second run and stops on a bad input before any list;
`p14-fit` refuses without a passing reproduction or over an existing fit, and splits the
won and lost questions by the Dense rank of the gold paragraphs whose coverage changed.

Mirrors `tests/test_cli_phase13.py`. The input checks run over a small fixture: three units,
their vectors saved in a real `EmbeddingCache`, an `embedding.json` naming them, one dev
question, and the loaders of the real inputs replaced by that fixture.
"""

import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from concept_embeddings_rag import cli, config
from concept_embeddings_rag.artifacts import digest_of
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.embeddings.cache import CachedQueryBackend, EmbeddingCache
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation import phase14

UNITS = ["u1", "u2", "u3"]


def write(directory, name, body):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(body), encoding="utf-8")


def fixture(tmp_path, monkeypatch, *, digest_ok=True, order_ok=True, p1_shift=0.0):
    """The inputs of one dev question; returns `(source_dir, target_dir)`."""
    rng = np.random.default_rng(14)
    vectors = rng.normal(size=(3, 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    question = rng.normal(size=4).astype(np.float32)
    question /= np.linalg.norm(question)
    source_dir, target_dir = tmp_path / "phase9", tmp_path / "phase14"
    saved_ids = UNITS if order_ok else UNITS[::-1]
    EmbeddingCache(source_dir / "cache").save("passages", vectors, saved_ids, {})
    write(
        source_dir,
        phase9.EMBEDDING_FILENAME,
        {
            "corpus_cache_key": "passages",
            "vectors_digest": phase9.vectors_digest(vectors) if digest_ok else "not-it",
            "question_cache_key": config.PHASE_14_DEV_QUESTION_VECTORS_KEY,
        },
    )
    p1_score = float(vectors[0] @ question) + p1_shift
    rows10 = [
        {
            "qid": "q1",
            "question": "which one?",
            "dense": [("u1", p1_score), ("u2", 0.1)],
            "bm25": [("u2", 3.0)],
            config.ENTITY_HOP_NAME: [],
        }
    ]
    monkeypatch.setattr(cli, "_p13_phase10_dev_lists", lambda *a: (rows10, "p10-lists"))
    monkeypatch.setattr(cli, "_p11_dev_inputs", lambda *a: ([SimpleNamespace(qid="q1")], {}))
    monkeypatch.setattr(
        cli,
        "_p11_entity_index",
        lambda *a: (SimpleNamespace(unit_ids=list(UNITS)), {"entity_index_digest": "e"}),
    )
    monkeypatch.setattr(
        cli,
        "_p12_dev_question_vector",
        lambda *a: CachedQueryBackend(["which one?"], question[None, :], "bge", "rev"),
    )
    return source_dir, target_dir


def no_hop(*args, **kwargs):
    raise AssertionError("the hop must not start")


def written(target_dir):
    return sorted(p.name for p in target_dir.iterdir()) if target_dir.exists() else []


def test_lists_refuse_a_second_run(tmp_path, monkeypatch):
    write(tmp_path, cli.P14_REPRODUCTION_NAME, {"passed": True})
    monkeypatch.setattr(
        cli, "_p10_json", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p14_lists(target_dir=tmp_path)


def test_lists_refuse_when_the_lists_already_exist(tmp_path, monkeypatch):
    write(tmp_path, cli.P14_DEV_LISTS_MANIFEST, {"digest": "recorded"})
    monkeypatch.setattr(
        cli, "_p10_json", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p14_lists(target_dir=tmp_path)


def test_a_vector_digest_mismatch_stops_the_stage(tmp_path, monkeypatch):
    source_dir, target_dir = fixture(tmp_path, monkeypatch, digest_ok=False)
    monkeypatch.setattr(cli, "node_weights", no_hop)
    with pytest.raises(SystemExit, match="vectors_digest"):
        cli.cmd_p14_lists(source_dir=source_dir, target_dir=target_dir)
    assert written(target_dir) == []


def test_a_unit_order_mismatch_stops_the_stage(tmp_path, monkeypatch):
    source_dir, target_dir = fixture(tmp_path, monkeypatch, order_ok=False)
    monkeypatch.setattr(cli, "node_weights", no_hop)
    with pytest.raises(SystemExit, match="order"):
        cli.cmd_p14_lists(source_dir=source_dir, target_dir=target_dir)
    assert written(target_dir) == []


def test_a_d1_miss_stops_the_stage_before_any_list(tmp_path, monkeypatch):
    source_dir, target_dir = fixture(tmp_path, monkeypatch, p1_shift=2e-5)
    monkeypatch.setattr(cli, "node_weights", no_hop)
    with pytest.raises(SystemExit, match=r"D1 .*q1"):
        cli.cmd_p14_lists(source_dir=source_dir, target_dir=target_dir)
    assert written(target_dir) == []


def test_d1_passes_on_the_recorded_score_and_the_hop_starts(tmp_path, monkeypatch):
    source_dir, target_dir = fixture(tmp_path, monkeypatch)

    class HopStarted(Exception):
        pass

    def started(*args, **kwargs):
        raise HopStarted

    monkeypatch.setattr(cli, "node_weights", started)
    with pytest.raises(HopStarted):
        cli.cmd_p14_lists(source_dir=source_dir, target_dir=target_dir)
    assert written(target_dir) == []


def test_the_p14_lists_stage_is_registered():
    assert cli.build_parser().parse_args(["p14-lists"]).command == "p14-lists"


# --- S4: p14-fit ------------------------------------------------------------------------------


def test_fit_refuses_without_a_reproduction(tmp_path):
    with pytest.raises(SystemExit, match="reproduction.json does not exist"):
        cli.cmd_p14_fit(target_dir=tmp_path)


def test_fit_refuses_after_a_failed_reproduction(tmp_path, monkeypatch):
    write(tmp_path, cli.P14_REPRODUCTION_NAME, {"passed": False, "dev_lists_digest": "d"})
    monkeypatch.setattr(
        cli, "_p11_dev_inputs", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="did not pass"):
        cli.cmd_p14_fit(target_dir=tmp_path)


def test_a_refit_refuses_to_overwrite(tmp_path, monkeypatch):
    write(tmp_path, cli.P14_REPRODUCTION_NAME, {"passed": True, "dev_lists_digest": "d"})
    write(tmp_path, cli.P14_FIT_NAME, {"terminal_state": None})
    monkeypatch.setattr(
        cli, "_p11_dev_inputs", lambda *a, **k: pytest.fail("no input read after the refusal")
    )
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p14_fit(target_dir=tmp_path)


ZERO, ONE = phase14.alpha_key(0.0), phase14.alpha_key(1.0)


def dense_list(gold_rank):
    """Dense's list: `a` first, then 15 fillers, with `g` at `gold_rank` when one is given."""
    ids = ["a", *(f"d{i}" for i in range(1, 16))]
    if gold_rank is not None:
        ids[gold_rank - 1] = "g"
    return [(u, 1.0 if u == "a" else 0.1 - 0.005 * i) for i, u in enumerate(ids)]


def fit_fixture(tmp_path, monkeypatch, *, digest_ok=True):
    """Three dev questions, gold `a` and `g`, every unit 1,024 tokens: two fit in 2,048.

    At P10-C's weights the context is `a` plus the hop's first unit (the Dense fillers and
    BM25's second unit stay below the hop's top score), so which of `z` and `g` leads the hop
    list decides Full Support. At `alpha = 1`: q1 is won (`g` at Dense rank 12), q2 is lost
    (`g` at Dense rank 3), q3 ties (the same list at both points).
    """
    z_first = [("z", 3.0), ("g", 2.0), ("w", 1.0)]
    g_first = [("g", 3.0), ("z", 2.0), ("w", 1.0)]
    plan = {
        "q1": (12, z_first, g_first),
        "q2": (3, g_first, z_first),
        "q3": (None, z_first, z_first),
    }
    rows, questions = [], []
    for qid, (rank, control, candidate) in plan.items():
        row = {
            "qid": qid,
            "question": f"{qid}?",
            "dense": dense_list(rank),
            "bm25": [("a", 1.0), ("b0", 0.0)],
            "positives": {},
        }
        for alpha in config.PHASE_14_ALPHAS:
            row[phase14.alpha_key(alpha)] = candidate if alpha == 1.0 else control
        rows.append(row)
        questions.append(Question(qid, f"{qid}?", "", ("a", "g"), (), config.PHASE_9_STANDARD))
    units = {"a", "b0", "g", "z", "w", *(f"d{i}" for i in range(1, 16))}
    token_counts = dict.fromkeys(units, 1024)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / cli.P14_DEV_LISTS).write_bytes(gzip.compress(text.encode("utf-8")))
    names = ["dense", "bm25", *(phase14.alpha_key(a) for a in config.PHASE_14_ALPHAS)]
    bridge = {"population": 2651, "first_5": {ZERO: 590}, "in_list": {ZERO: 1665}}
    write(
        tmp_path,
        cli.P14_DEV_LISTS_MANIFEST,
        {"digest": digest_of(text), "lists": names, "bridge_items": bridge},
    )
    write(
        tmp_path,
        cli.P14_REPRODUCTION_NAME,
        {"passed": True, "dev_lists_digest": digest_of(text) if digest_ok else "other"},
    )
    monkeypatch.setattr(cli, "_p11_dev_inputs", lambda *a: (questions, token_counts))
    return rows, questions, token_counts


def test_fit_refuses_lists_the_reproduction_did_not_check(tmp_path, monkeypatch):
    fit_fixture(tmp_path, monkeypatch, digest_ok=False)
    with pytest.raises(SystemExit, match="not the ones the reproduction checked"):
        cli.cmd_p14_fit(target_dir=tmp_path)
    assert not (tmp_path / cli.P14_FIT_NAME).exists()


def test_the_dense_rank_split_of_won_and_lost_questions(tmp_path, monkeypatch):
    fit_fixture(tmp_path, monkeypatch)
    rows = cli._p14_load_dev_lists(tmp_path)
    questions, token_counts = cli._p11_dev_inputs()
    split = cli._p14_dense_rank_split(
        rows, questions, token_counts, alpha=1.0, weights=config.PHASE_14_P10C_WEIGHTS
    )
    assert split["won"] == {
        "questions": 1,
        "gold_changed": {"1-10": 0, "11-100": 1, "beyond-100": 0},
        "gold_changed_total": 1,
    }
    assert split["lost"] == {
        "questions": 1,
        "gold_changed": {"1-10": 1, "11-100": 0, "beyond-100": 0},
        "gold_changed_total": 1,
    }
    assert split["total"]["questions"] == 2
    assert [(q["qid"], q["outcome"]) for q in split["per_question"]] == [
        ("q1", "won"),
        ("q2", "lost"),
    ]
    assert split["per_question"][0]["changes"] == [
        {"unit_id": "g", "change": "gained", "dense_rank": 12, "dense_class": "11-100"}
    ]


def test_the_fit_writes_the_curve_and_the_gate_verdict_once(tmp_path, monkeypatch):
    fit_fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="DEV_STOP"):
        cli.cmd_p14_fit(target_dir=tmp_path)
    fit = json.loads((tmp_path / cli.P14_FIT_NAME).read_text(encoding="utf-8"))
    assert fit["n_points"] == len(fit["curve"]) == 330
    assert fit["p10c_point_in_grid"]["alpha"] == 0.0
    assert fit["p10c_point_in_grid"]["weights"] == list(config.PHASE_14_P10C_WEIGHTS)
    assert fit["p10c_point_in_grid"]["supported"] == 1  # q2 only
    assert fit["dev_bar"] == 4835 and fit["terminal_state"] == "DEV_STOP"
    assert set(fit["best_per_alpha"]) == {phase14.alpha_key(a) for a in config.PHASE_14_ALPHAS}
    assert fit["d8"]["bridge_items"]["first_5"] == {ZERO: 590}
    assert fit["d8"]["dense_rank_split"]["total"]["questions"] >= 0
    assert fit["chosen"] == phase14.choose_point(fit["curve"])
    with pytest.raises(SystemExit, match="already records"):
        cli.cmd_p14_fit(target_dir=tmp_path)


def test_the_p14_fit_stage_is_registered():
    assert cli.build_parser().parse_args(["p14-fit"]).command == "p14-fit"


# --- S5: p14-eval, the single held-out pass on test-11 ---------------------------------------


def test_the_pass_refuses_without_the_authorization_flag(tmp_path):
    with pytest.raises(SystemExit, match="authorized-pass"):
        cli.cmd_p14_eval(authorized=False, target_dir=tmp_path)


def test_the_pass_refuses_without_a_fit(tmp_path):
    with pytest.raises(SystemExit, match="fit.json does not exist"):
        cli.cmd_p14_eval(authorized=True, target_dir=tmp_path)


def test_the_pass_refuses_after_a_dev_stop(tmp_path):
    write(tmp_path, cli.P14_FIT_NAME, {"terminal_state": "DEV_STOP"})
    with pytest.raises(SystemExit, match="DEV_STOP"):
        cli.cmd_p14_eval(authorized=True, target_dir=tmp_path)


def test_the_pass_refuses_an_uncommitted_fit(tmp_path, monkeypatch):
    write(tmp_path, cli.P14_FIT_NAME, {"terminal_state": None})
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: False)
    with pytest.raises(SystemExit, match="not committed"):
        cli.cmd_p14_eval(authorized=True, target_dir=tmp_path)


def test_the_pass_refuses_a_second_start(tmp_path, monkeypatch):
    write(tmp_path, cli.P14_FIT_NAME, {"terminal_state": None})
    write(tmp_path, cli.P14_MARKER_NAME, {"started_at": "earlier"})
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: True)
    monkeypatch.setattr(
        cli, "_phase_9_inputs", lambda *a, **k: pytest.fail("inputs loaded after a start")
    )
    with pytest.raises(SystemExit, match="already started once"):
        cli.cmd_p14_eval(authorized=True, target_dir=tmp_path)


def test_test_10_is_not_a_phase_14_set():
    assert config.PHASE_10_TEST not in config.PHASE_14_QUESTION_SETS
    assert config.PHASE_14_TEST == config.PHASE_11_TEST != config.PHASE_10_TEST


PASS_ALPHA = 0.5  # not the real selection, so a hard-coded alpha would fail
PASS_WEIGHTS = {"dense": 0.5, "bm25": 0.3, "relevance-hop": 0.2}


def pass_fixture(tmp_path, monkeypatch, *, key=config.PHASE_14_TEST_QUESTION_VECTORS_KEY):
    """A 40-unit corpus, three test-11 questions, and the loaders replaced by it.

    Returns `(dirs, questions)`. A `test-10` load fails the test.
    """
    from concept_embeddings_rag.corpus.pool import IndexingUnit
    from concept_embeddings_rag.evaluation import phase10, phase11
    from concept_embeddings_rag.evaluation.second_hop import node_weights
    from concept_embeddings_rag.nodes.index import build_node_index
    from concept_embeddings_rag.nodes.local_extraction import record_from_forms
    from concept_embeddings_rag.retrieval.bm25 import BM25Retriever
    from concept_embeddings_rag.retrieval.dense import DenseRetriever

    rng = np.random.default_rng(5)
    unit_ids = [f"u{i:02d}" for i in range(40)]
    records = {
        u: record_from_forms(
            u,
            [f"Form {int(f)}" for f in rng.integers(0, 6, size=int(rng.integers(1, 4)))],
            model="fixture",
            configuration_digest="f",
        )
        for u in unit_ids
    }
    index = build_node_index(records, unit_ids, extraction_digest="0" * 64)
    vectors = rng.normal(size=(40, 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    units = [
        IndexingUnit(u, f"Title {u}", (f"text about {u} and form {i % 6}",))
        for i, u in enumerate(unit_ids)
    ]
    questions = [
        Question(
            f"t{i}",
            f"which form {i}?",
            "",
            (unit_ids[i], unit_ids[i + 20]),
            (),
            config.PHASE_14_TEST,
        )
        for i in range(3)
    ]
    question_vectors = rng.normal(size=(3, 4)).astype(np.float32)
    question_vectors /= np.linalg.norm(question_vectors, axis=1, keepdims=True)

    source_dir, phase10_dir = tmp_path / "phase9", tmp_path / "phase10"
    phase11_dir, target_dir = tmp_path / "phase11", tmp_path / "phase14"
    write(
        phase10_dir,
        phase10.QUESTIONS_FILENAME,
        {"sets": {config.PHASE_11_TEST: {"question_digest": "q11", "mapping_digest": "m11"}}},
    )
    write(
        phase10_dir,
        cli.P10_FIT_NAME,
        {"terminal_state": None, "weights": {"dense": 0.5, "bm25": 0.3, "entity-hop": 0.2}},
    )
    EmbeddingCache(phase11_dir / "cache" / "questions").save(
        key, question_vectors, [q.qid for q in questions], {}
    )
    write(
        phase11_dir,
        cli.P11_EMBED_NAME,
        {"sets": {config.PHASE_11_TEST: {"key": key, "model": "bge", "revision": "rev"}}},
    )
    write(
        target_dir,
        cli.P14_FIT_NAME,
        {"terminal_state": None, "alpha": PASS_ALPHA, "weights": PASS_WEIGHTS},
    )
    components = {
        "dense": DenseRetriever(
            vectors, unit_ids, CachedQueryBackend(["unused"], vectors[:1], "bge", "rev")
        ),
        "bm25": BM25Retriever(units),
        "index": index,
        "node_weights": node_weights(index),
        "weights": {"hybrid-bm25": {"dense": 0.5, "bm25": 0.5}},
        "questions": [],  # the dev questions: the pass never reads them
        "token_counts": dict.fromkeys(unit_ids, 300),
        "provenance": {
            "corpus_unit_set_hash": "corpus",
            "question_digest": "q9",
            "mapping_digest": "m9",
            "embedding": {"model": "bge", "revision": "rev", "question_cache_key": "k9"},
            "weights": {},
        },
    }
    monkeypatch.setattr(cli, "_phase_9_inputs", lambda *a, **k: components)
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: True)
    monkeypatch.setattr(phase11, "load_test_11", lambda *a, **k: list(questions))
    monkeypatch.setattr(phase10, "load_set", lambda *a, **k: pytest.fail("phase10.load_set"))
    real_read_set = phase10._read_set

    def read_set(directory, name, **kwargs):
        assert name != config.PHASE_10_TEST, "test-10 was loaded"
        return real_read_set(directory, name, **kwargs)

    monkeypatch.setattr(phase10, "_read_set", read_set)
    dirs = {
        "source_dir": source_dir,
        "phase10_dir": phase10_dir,
        "phase11_dir": phase11_dir,
        "target_dir": target_dir,
    }
    return dirs, questions


def test_the_pass_refuses_question_vectors_under_another_key(tmp_path, monkeypatch):
    dirs, _questions = pass_fixture(tmp_path, monkeypatch, key="not-the-key")
    with pytest.raises(SystemExit, match="3ab80134c058052d"):
        cli.cmd_p14_eval(authorized=True, **dirs)
    assert not (dirs["target_dir"] / cli.P14_MARKER_NAME).exists()


def test_the_pass_builds_p14_at_the_fitted_alpha_and_writes_every_run(tmp_path, monkeypatch):
    dirs, questions = pass_fixture(tmp_path, monkeypatch)
    built = []
    real_stage = cli.RelevanceHopStage

    def spy(*args, **kwargs):
        stage = real_stage(*args, **kwargs)
        built.append(stage)
        return stage

    monkeypatch.setattr(cli, "RelevanceHopStage", spy)
    written_paths = cli.cmd_p14_eval(authorized=True, **dirs)
    target_dir = dirs["target_dir"]

    assert [stage.alpha for stage in built] == [PASS_ALPHA]
    assert len(built[0].sim_seconds) == len(questions)
    marker = json.loads((target_dir / cli.P14_MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["set"] == config.PHASE_14_TEST
    assert marker["alpha"] == PASS_ALPHA
    assert [p.name for p in written_paths] == [
        f"run-{name}.json" for name in config.PHASE_14_SYSTEMS
    ]
    p14 = json.loads((target_dir / "run-hybrid-bm25-seeded-hop.json").read_text("utf-8"))
    assert p14["label"] == "P14"
    assert p14["set"] == config.PHASE_14_TEST
    assert p14["metrics"]["n_questions"] == len(questions)
    assert p14["fusion"]["weights"] == {"dense": 0.5, "bm25": 0.3, "seeded-hop": 0.2}
    assert p14["relevance_hop"]["alpha"] == PASS_ALPHA
    assert p14["relevance_hop"]["component"] == "relevance-hop"
    provenance = p14["provenance"]
    assert provenance["question_set"] == config.PHASE_14_TEST
    assert provenance["question_digest"] == "q11"
    key = provenance["embedding"]["question_cache_key"]
    assert key == config.PHASE_14_TEST_QUESTION_VECTORS_KEY
    assert provenance["fit_digest"] == marker["fit_digest"]
    assert provenance["weights"]["hybrid-bm25-entity-hop"] == {
        "dense": 0.5,
        "bm25": 0.3,
        "entity-hop": 0.2,
    }
    outcomes = phase9.load_outcomes(target_dir, "hybrid-bm25-seeded-hop")
    assert [r["qid"] for r in outcomes] == [q.qid for q in questions]
    assert [r["sim_seconds"] for r in outcomes] == built[0].sim_seconds
    assert all(r["sim_seconds"] * 1000.0 <= r["latency_ms"] for r in outcomes)
    for name in config.PHASE_14_SYSTEMS[:3]:
        assert "sim_seconds" not in phase9.load_outcomes(target_dir, name)[0]
    with pytest.raises(SystemExit, match="already started once"):
        cli.cmd_p14_eval(authorized=True, **dirs)


def test_the_p14_eval_stage_is_registered():
    args = cli.build_parser().parse_args(["p14-eval", "--authorized-pass"])
    assert (args.command, args.authorized) == ("p14-eval", True)


# --- S6: p14-outcome, the paired tests, the label, D8 on test-11 and D9 ---------------------

N_TEST = config.PHASE_10_SET_SIZES[config.PHASE_14_TEST]
OUTCOME_FIT = {"terminal_state": None, "alpha": 0.75, "weights": {"dense": 0.5}}


def outcome_record(qid, supported, *, sim=None, latency=100.0):
    row = {
        "qid": qid,
        "cohort": config.PHASE_14_TEST,
        "budgets": {
            "2048": {
                "full_support": float(supported),
                "gold_recall": 1.0 if supported else 0.5,
                "context_precision": (2 if supported else 1) / 30,
                "units_included": 30,
            }
        },
        "gpr_at_k": {"10": 0.5},
        "latency_ms": latency,
    }
    if sim is not None:
        row["sim_seconds"] = sim
    return row


def outcome_fixture(tmp_path, monkeypatch, *, p14_wins, p14_losses, p10b_extra=0):
    """Four constructed `test-11` runs; returns the target directory.

    P14 against P10-C: `p14_wins` questions only P14 supports, `p14_losses` only P10-C does.
    P10-B supports those questions too plus `p10b_extra` that no other system supports, so a
    strong P10-B makes the secondary comparison a regression without touching the primary.
    """
    fit_digest = digest_of(json.dumps(OUTCOME_FIT, sort_keys=True))
    write(tmp_path, cli.P14_FIT_NAME, OUTCOME_FIT)
    write(tmp_path, cli.P14_MARKER_NAME, {"fit_digest": fit_digest, "set": config.PHASE_14_TEST})
    monkeypatch.setattr(cli, "_p10_fit_is_committed", lambda path: True)
    qids = [f"q{i:04d}" for i in range(N_TEST)]
    wins = set(qids[:p14_wins])
    losses = set(qids[p14_wins : p14_wins + p14_losses])
    extra = set(qids[p14_wins + p14_losses : p14_wins + p14_losses + p10b_extra])
    supported = {
        "dense": set(),
        "hybrid-bm25": wins | losses | extra,
        "hybrid-bm25-entity-hop": losses,
        "hybrid-bm25-seeded-hop": wins,
    }
    for name in config.PHASE_14_SYSTEMS:
        is_p14 = name == "hybrid-bm25-seeded-hop"
        records = [
            outcome_record(
                q,
                q in supported[name],
                sim=0.02 if is_p14 else None,
                latency=300.0 if is_p14 else 100.0,
            )
            for q in qids
        ]
        body = {
            "system": name,
            "set": config.PHASE_14_TEST,
            "latency": phase9.latency_summary(records),
            "provenance": {"question_set": config.PHASE_14_TEST, "fit_digest": fit_digest},
        }
        if is_p14:
            body["relevance_hop"] = {"alpha": OUTCOME_FIT["alpha"]}
        phase9.write_run(tmp_path, name, records, body=body)
    return tmp_path


def read_outcome(target_dir):
    return json.loads((target_dir / cli.P14_OUTCOME_NAME).read_text(encoding="utf-8"))


def test_the_outcome_labels_a_significant_gain_supported(tmp_path, monkeypatch):
    target_dir = outcome_fixture(tmp_path, monkeypatch, p14_wins=40, p14_losses=10)
    cli.cmd_p14_outcome(target_dir=target_dir)
    body = read_outcome(target_dir)
    primary = body["primary_p14_vs_p10c"]
    assert (primary["wins"], primary["losses"], primary["ties"]) == (40, 10, N_TEST - 50)
    assert primary["delta_percentage_points"] == pytest.approx(100.0 * 30 / N_TEST)
    assert primary["exact_two_sided_p"] == phase9.exact_two_sided_p(40, 10)
    assert body["terminal_state"] == "CANDIDATE_RELEVANCE_SUPPORTED"
    assert body["stop_reasons"] == []


def test_the_outcome_labels_a_significant_loss_a_regression(tmp_path, monkeypatch):
    target_dir = outcome_fixture(tmp_path, monkeypatch, p14_wins=10, p14_losses=40)
    cli.cmd_p14_outcome(target_dir=target_dir)
    assert read_outcome(target_dir)["terminal_state"] == "CANDIDATE_RELEVANCE_REGRESSION"


def test_the_outcome_labels_an_insignificant_gain_not_supported(tmp_path, monkeypatch):
    target_dir = outcome_fixture(tmp_path, monkeypatch, p14_wins=12, p14_losses=10)
    cli.cmd_p14_outcome(target_dir=target_dir)
    assert read_outcome(target_dir)["terminal_state"] == "CANDIDATE_RELEVANCE_NOT_SUPPORTED"


def test_the_secondary_comparison_never_changes_the_label(tmp_path, monkeypatch):
    target_dir = outcome_fixture(tmp_path, monkeypatch, p14_wins=40, p14_losses=10, p10b_extra=200)
    cli.cmd_p14_outcome(target_dir=target_dir)
    body = read_outcome(target_dir)
    secondary = body["secondary_p14_vs_p10b"]
    assert secondary["losses"] > secondary["wins"]
    assert secondary["exact_two_sided_p"] < 0.05
    assert body["terminal_state"] == "CANDIDATE_RELEVANCE_SUPPORTED"
    descriptive = body["descriptive_p10c_vs_p10b"]
    assert (descriptive["wins"], descriptive["losses"]) == (0, 240)


def test_the_outcome_records_d8_from_the_recorded_runs_and_d9(tmp_path, monkeypatch):
    target_dir = outcome_fixture(tmp_path, monkeypatch, p14_wins=40, p14_losses=10)
    cli.cmd_p14_outcome(target_dir=target_dir)
    body = read_outcome(target_dir)
    d8 = body["d8_test_11"]
    assert d8["won"]["questions"] == 40
    assert d8["lost"]["questions"] == 10
    assert d8["won"]["gold_changed"] == {"1-10": 0, "beyond-10": 0, "undetermined": 40}
    assert d8["computed_from_rankings"] is False
    p14 = body["latency"]["hybrid-bm25-seeded-hop"]
    assert p14["sim"]["mean_ms"] == pytest.approx(20.0)
    assert p14["retrieve_without_sim"]["mean_ms"] == pytest.approx(280.0)
    assert p14["retrieve_with_sim"]["mean_ms"] == pytest.approx(300.0)
    assert body["latency"]["hybrid-bm25-entity-hop"]["retrieve"]["mean_ms"] == pytest.approx(100.0)
    ratios = body["latency_ratios"]
    assert ratios["p14_with_sim_over_p10c"] == pytest.approx(3.0)
    assert ratios["p14_without_sim_over_p10b"] == pytest.approx(2.8)


def test_the_outcome_refuses_to_overwrite(tmp_path, monkeypatch):
    target_dir = outcome_fixture(tmp_path, monkeypatch, p14_wins=40, p14_losses=10)
    write(target_dir, cli.P14_OUTCOME_NAME, {"terminal_state": "earlier"})
    monkeypatch.setattr(
        phase9, "load_outcomes", lambda *a: pytest.fail("runs read over an existing outcome")
    )
    with pytest.raises(SystemExit, match="written once"):
        cli.cmd_p14_outcome(target_dir=target_dir)
    assert read_outcome(target_dir) == {"terminal_state": "earlier"}


def test_the_outcome_refuses_an_outcomes_file_its_digest_does_not_match(tmp_path, monkeypatch):
    target_dir = outcome_fixture(tmp_path, monkeypatch, p14_wins=40, p14_losses=10)
    archive = target_dir / "outcomes-hybrid-bm25-entity-hop.jsonl.gz"
    text = gzip.decompress(archive.read_bytes()).decode("utf-8")
    archive.write_bytes(
        gzip.compress(text.replace('"latency_ms": 100.0', '"latency_ms": 1.0').encode())
    )
    with pytest.raises(SystemExit, match="does not match the digest"):
        cli.cmd_p14_outcome(target_dir=target_dir)
    assert not (target_dir / cli.P14_OUTCOME_NAME).exists()


def test_the_outcome_refuses_a_run_the_committed_fit_did_not_authorize(tmp_path, monkeypatch):
    target_dir = outcome_fixture(tmp_path, monkeypatch, p14_wins=40, p14_losses=10)
    write(target_dir, cli.P14_FIT_NAME, {**OUTCOME_FIT, "alpha": 0.5})
    with pytest.raises(SystemExit, match="fit"):
        cli.cmd_p14_outcome(target_dir=target_dir)
    assert not (target_dir / cli.P14_OUTCOME_NAME).exists()


def test_the_p14_outcome_stage_is_registered():
    assert cli.build_parser().parse_args(["p14-outcome"]).command == "p14-outcome"
