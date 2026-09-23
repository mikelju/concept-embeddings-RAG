"""Phase 9: the FullWiki cohorts, the gold mapping, the systems and the outcome rule.

What can change a Phase 9 figure is tested here: which questions are in which cohort, which
FullWiki unit a supporting title resolves to, what an unresolved title does to the
denominator, the frozen weights the systems are built with, the per-cohort metrics, and the
mechanical terminal label. Every corpus here is a handful of units built in memory.
"""

import json

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import IndexingUnit, unit_id_for
from concept_embeddings_rag.evaluation import fullwiki as p9
from concept_embeddings_rag.evaluation.budget import fill_context
from concept_embeddings_rag.evaluation.metrics import full_support, gold_recall


def a_unit(title: str, text: str = "Some text.") -> IndexingUnit:
    sentences = (text,)
    return IndexingUnit(unit_id=unit_id_for(title, sentences), title=title, sentences=sentences)


def a_raw(qid: str, titles: tuple[str, ...], question: str = "q?") -> dict:
    return {
        "_id": qid,
        "question": f"{question} {qid}",
        "answer": "a",
        "supporting_facts": [[title, index] for index, title in enumerate(titles)],
    }


SIZES = {
    config.PHASE_9_RETRIEVAL_UNSEEN: 2,
    config.PHASE_9_STANDARD: 3,
    config.PHASE_9_HISTORICAL_OVERLAP: 1,
}


# --- Cohorts ---------------------------------------------------------------------------


def test_the_cohorts_are_the_qid_arithmetic_and_nothing_else():
    cohorts = p9.cohorts_of(["a", "b", "c"], ["b"], sizes=SIZES)

    assert cohorts[config.PHASE_9_STANDARD] == ["a", "b", "c"]
    assert cohorts[config.PHASE_9_HISTORICAL_OVERLAP] == ["b"]
    assert cohorts[config.PHASE_9_RETRIEVAL_UNSEEN] == ["a", "c"]


@pytest.mark.parametrize(
    ("raw", "historical"),
    [
        (["a", "b", "b"], ["b"]),  # a repeated qid
        (["a", "b", "c"], ["z"]),  # a historical qid outside the source
        (["a", "b", "c", "d"], ["b"]),  # the source is not the declared size
        (["a", "b", "c"], ["a", "b"]),  # the historical subset is not the declared size
    ],
)
def test_qids_that_do_not_reconcile_exactly_are_a_data_stop(raw, historical):
    with pytest.raises(p9.DataStop, match="DATA_STOP"):
        p9.cohorts_of(raw, historical, sizes=SIZES)


def test_the_real_cohort_sizes_are_the_spec_ones():
    assert config.PHASE_9_COHORT_SIZES == {
        "retrieval-unseen": 5405,
        "standard-dev": 7405,
        "historical-overlap": 2000,
    }


def test_supporting_titles_collapse_to_distinct_titles_in_first_appearance_order():
    raw = {"_id": "q", "supporting_facts": [["B", 0], ["A", 2], ["B", 1]]}

    assert p9.supporting_titles(raw) == ("B", "A")


def test_a_question_without_exactly_two_supporting_titles_is_a_data_stop():
    raw = {"_id": "q", "supporting_facts": [["A", 0], ["A", 1]]}

    with pytest.raises(p9.DataStop, match="two"):
        p9.supporting_titles(raw)


# --- Gold mapping (D4) -----------------------------------------------------------------


def test_an_exact_title_resolves_to_its_unit():
    units = [a_unit("Ed Wood"), a_unit("ed wood")]

    resolved = p9.resolve_titles({"Ed Wood"}, units)["Ed Wood"]

    assert resolved.status == p9.EXACT
    assert resolved.unit_id == units[0].unit_id


def test_a_title_with_no_exact_match_resolves_on_a_unique_normalized_title():
    units = [a_unit("Scott  Derrickson")]

    resolved = p9.resolve_titles({"scott derrickson"}, units)["scott derrickson"]

    assert resolved.status == p9.NORMALIZED
    assert resolved.unit_id == units[0].unit_id


def test_a_normalized_title_naming_two_units_is_ambiguous_and_never_resolved_to_one():
    units = [a_unit("Mercury"), a_unit("MERCURY")]

    resolved = p9.resolve_titles({"mercury"}, units)["mercury"]

    assert resolved.status == p9.AMBIGUOUS
    assert resolved.unit_id is None
    assert resolved.candidates == 2


def test_an_exact_title_carried_by_two_units_is_ambiguous():
    units = [a_unit("Paris", "One."), a_unit("Paris", "Two.")]

    resolved = p9.resolve_titles({"Paris"}, units)["Paris"]

    assert resolved.status == p9.AMBIGUOUS
    assert resolved.candidates == 2


def test_a_title_with_no_match_is_unmatched():
    resolved = p9.resolve_titles({"Nowhere"}, [a_unit("Somewhere")])["Nowhere"]

    assert resolved.status == p9.UNMATCHED
    assert resolved.unit_id is None


def build(units, raws, historical, *, ceiling=config.PHASE_9_UNRESOLVED_CEILING):
    return p9.build_questions(
        raws,
        historical,
        units,
        sizes={
            config.PHASE_9_RETRIEVAL_UNSEEN: len(raws) - len(historical),
            config.PHASE_9_STANDARD: len(raws),
            config.PHASE_9_HISTORICAL_OVERLAP: len(historical),
        },
        ceiling=ceiling,
        corpus={"unit_set_hash": "h", "ordered_unit_digest": "o"},
    )


def test_an_unresolved_gold_stays_in_the_denominator_and_can_never_be_retrieved():
    units = [a_unit("A"), a_unit("B"), a_unit("C")]
    raws = [a_raw("q1", ("A", "B")), a_raw("q2", ("C", "Missing"))]

    questions, body = build(units, raws, ["q1"])

    assert [question.qid for question in questions] == ["q1", "q2"]
    q2 = questions[1]
    assert q2.gold_unit_ids[0] == units[2].unit_id
    sentinel = q2.gold_unit_ids[1]
    assert sentinel.startswith(p9.UNRESOLVED_PREFIX)
    assert sentinel not in {unit.unit_id for unit in units}
    # Even a ranking holding every unit of the corpus reaches one gold of two, never both.
    ranked = [unit.unit_id for unit in units]
    context = fill_context(ranked, dict.fromkeys(ranked, 10), 4096)
    assert full_support(context, q2.gold_unit_ids) == 0
    assert gold_recall(context, q2.gold_unit_ids) == 0.5
    assert body["questions_with_unresolved_gold"] == 1
    assert body["unmatched_titles"] == 1
    assert body["terminal_state"] is None


def test_every_question_carries_its_cohort():
    units = [a_unit("A"), a_unit("B")]
    raws = [a_raw("q1", ("A", "B")), a_raw("q2", ("A", "B"))]

    questions, body = build(units, raws, ["q2"])

    assert {q.qid: q.split for q in questions} == {
        "q1": config.PHASE_9_RETRIEVAL_UNSEEN,
        "q2": config.PHASE_9_HISTORICAL_OVERLAP,
    }
    assert body["n_retrieval_unseen"] == 1
    assert body["n_historical_overlap"] == 1
    assert body["n_standard_dev"] == 2


def test_more_questions_than_the_ceiling_with_unresolved_gold_is_a_data_stop():
    units = [a_unit("A")]
    raws = [a_raw("q1", ("A", "X")), a_raw("q2", ("A", "Y"))]

    _questions, at_ceiling = build(units, raws, ["q1"], ceiling=2)
    _questions, past = build(units, raws, ["q1"], ceiling=1)

    assert at_ceiling["terminal_state"] is None
    assert past["terminal_state"] == p9.DATA_STOP


def test_the_mapping_digest_moves_with_the_gold_and_the_question_digest_does_not():
    raws = [a_raw("q1", ("A", "B"))]
    _q, first = build([a_unit("A"), a_unit("B")], raws, [])
    _q, second = build([a_unit("A"), a_unit("B", "Other text.")], raws, [])

    assert first["question_digest"] == second["question_digest"]
    assert first["mapping_digest"] != second["mapping_digest"]


def test_the_questions_artifact_round_trips_and_refuses_another_mapping(tmp_path):
    units = [a_unit("A"), a_unit("B")]
    raws = [a_raw("q1", ("A", "B"))]
    questions, body = build(units, raws, [])

    p9.write_questions(tmp_path, body)
    loaded, recorded = p9.load_questions(tmp_path, corpus_unit_set_hash="h")

    assert loaded == questions
    assert recorded["mapping_digest"] == body["mapping_digest"]
    _q, other = build([a_unit("A"), a_unit("B", "Else.")], raws, [])
    with pytest.raises(p9.FullWikiPhaseError, match="already records"):
        p9.write_questions(tmp_path, other)
    with pytest.raises(p9.FullWikiPhaseError, match="corpus"):
        p9.load_questions(tmp_path, corpus_unit_set_hash="another")


def test_a_questions_artifact_edited_on_disk_is_refused(tmp_path):
    units = [a_unit("A"), a_unit("B")]
    _questions, body = build(units, [a_raw("q1", ("A", "B"))], [])
    path = p9.write_questions(tmp_path, body)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["questions"][0]["gold_unit_ids"].reverse()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(p9.FullWikiPhaseError, match="digest"):
        p9.load_questions(tmp_path, corpus_unit_set_hash="h")


def test_a_data_stop_artifact_cannot_be_loaded_for_retrieval(tmp_path):
    units = [a_unit("A")]
    _questions, body = build(units, [a_raw("q1", ("A", "X"))], [], ceiling=0)
    p9.write_questions(tmp_path, body)

    with pytest.raises(p9.DataStop, match="DATA_STOP"):
        p9.load_questions(tmp_path, corpus_unit_set_hash="h")


# --- Systems and the shared measurement path (S4, S6) ------------------------------------

from concept_embeddings_rag.corpus.pool import Question  # noqa: E402
from concept_embeddings_rag.evaluation.second_hop import node_weights  # noqa: E402
from concept_embeddings_rag.nodes.index import build_node_index  # noqa: E402
from concept_embeddings_rag.nodes.local_extraction import record_from_forms  # noqa: E402
from concept_embeddings_rag.retrieval.entity_hop import EntityHopStage  # noqa: E402

IDS = [f"u{index:03d}" for index in range(120)]
# Every paragraph is too long to fit a budget except the four the questions are about, so a
# context holds exactly the short paragraphs a ranking reaches, whatever their rank.
TOKENS = {uid: (100 if uid in {"u000", "u001", "u118", "u119"} else 5000) for uid in IDS}


class DenseFixture:
    """Dense returns the pool in id order, and swaps the first two for one question."""

    name = "dense"

    def retrieve(self, query, top_k):
        ids = list(IDS)
        if query == "no bridge":
            ids[0], ids[1] = ids[1], ids[0]
        return [(uid, 1.0 - position / 120) for position, uid in enumerate(ids[:top_k])]


class StubBM25:
    name = "bm25"

    def retrieve(self, query, top_k):
        return [(uid, float(120 - position)) for position, uid in enumerate(IDS[:top_k])]


def a_stage() -> EntityHopStage:
    """`u000` bridges to `u119`, far outside the dense top 100; `u118` shares less with it."""
    forms = {"u000": ["Bridge", "Common"], "u119": ["Bridge", "Common"], "u118": ["Common"]}
    records = {
        uid: record_from_forms(uid, forms.get(uid, []), model="fixture", configuration_digest="f")
        for uid in IDS
    }
    index = build_node_index(records, IDS, extraction_digest="e" * 64)
    return EntityHopStage(index, node_weights(index))


def weight_files(tmp_path, bm25=None, entity=None, scheme="weighted"):
    bm25_file = tmp_path / "bm25.json"
    entity_file = tmp_path / "entity.json"
    bm25_file.write_text(
        json.dumps(
            {
                "config": {
                    "fusion_scheme": scheme,
                    "fusion_weights": bm25 or {"dense": 0.5, "bm25": 0.5},
                }
            }
        ),
        encoding="utf-8",
    )
    entity_file.write_text(
        json.dumps(
            {
                "fit": {
                    "scheme": scheme,
                    "weights": entity or {"dense": 0.7, "entity-hop": 0.30000000000000004},
                }
            }
        ),
        encoding="utf-8",
    )
    return bm25_file, entity_file


def test_the_weights_are_read_from_the_recorded_fits_exactly_as_recorded(tmp_path):
    bm25_file, entity_file = weight_files(tmp_path)

    weights = p9.read_frozen_weights(bm25_file, entity_file)

    assert weights["hybrid-bm25"] == {"dense": 0.5, "bm25": 0.5}
    # The recorded float, not a retyped 0.3: that is what reproduces Phase 7 to the question.
    assert weights["hybrid-entity-hop"]["entity-hop"] == 0.30000000000000004


@pytest.mark.parametrize(
    "overrides",
    [
        {"bm25": {"dense": 0.6, "bm25": 0.4}},
        {"entity": {"dense": 0.8, "entity-hop": 0.2}},
        {"scheme": "rrf"},
    ],
)
def test_a_recorded_fit_that_is_not_the_frozen_one_is_refused(tmp_path, overrides):
    with pytest.raises(p9.FullWikiPhaseError, match="frozen"):
        p9.read_frozen_weights(*weight_files(tmp_path, **overrides))


def test_the_three_systems_are_built_with_the_frozen_weights_and_nothing_else(tmp_path):
    weights = p9.read_frozen_weights(*weight_files(tmp_path))

    systems = p9.build_systems(DenseFixture(), StubBM25(), a_stage(), weights)

    assert list(systems) == list(config.PHASE_9_SYSTEMS)
    assert systems["dense"].name == "dense"
    assert systems["hybrid-bm25"].describe()["weights"] == {"dense": 0.5, "bm25": 0.5}
    described = systems["hybrid-entity-hop"].describe()
    assert described["scheme"] == "weighted"
    assert described["weights"] == {"dense": 0.7, "entity-hop": 0.30000000000000004}


def toy_questions() -> list[Question]:
    return [
        Question("q1", "bridge", "", ("u000", "u119"), (), config.PHASE_9_RETRIEVAL_UNSEEN),
        Question("q2", "no bridge", "", ("u001", "u000"), (), config.PHASE_9_HISTORICAL_OVERLAP),
    ]


def measured(tmp_path):
    weights = p9.read_frozen_weights(*weight_files(tmp_path))
    systems = p9.build_systems(DenseFixture(), StubBM25(), a_stage(), weights)
    return {
        name: p9.measure_system(name, system, toy_questions(), TOKENS)
        for name, system in systems.items()
    }


def test_the_measurement_records_every_question_with_budgets_ranks_and_latency(tmp_path):
    records = measured(tmp_path)

    dense = {record["qid"]: record for record in records["dense"]}
    # u119 is rank 120: outside dense's top 100, so dense can never support q1.
    assert dense["q1"]["budgets"]["2048"]["full_support"] == 0.0
    assert dense["q2"]["fs_at_k"]["2"] == 1.0
    assert dense["q2"]["gpr_at_k"]["2"] == 1.0
    assert dense["q1"]["gpr_at_k"]["20"] == 0.5
    assert dense["q1"]["latency_ms"] >= 0.0
    assert "entity" not in dense["q1"]


def test_the_entity_hop_brings_the_bridge_and_records_its_candidates(tmp_path):
    records = measured(tmp_path)

    entity = {record["qid"]: record for record in records["hybrid-entity-hop"]}
    assert entity["q1"]["budgets"]["2048"]["full_support"] == 1.0
    assert entity["q1"]["entity"]["positives"] == 2
    assert entity["q1"]["entity"]["gold_introduced_top100"] == 1
    assert entity["q1"]["entity"]["gold_introduced_context_2048"] == 1
    # q2's p1 is u001, which holds no entity: an empty expansion, recorded as such.
    assert entity["q2"]["entity"]["positives"] == 0


def test_cohort_metrics_are_means_over_exactly_the_cohort_questions(tmp_path):
    records = measured(tmp_path)["dense"]

    metrics = p9.cohort_metrics(records)

    unseen = metrics[config.PHASE_9_RETRIEVAL_UNSEEN]
    standard = metrics[config.PHASE_9_STANDARD]
    overlap = metrics[config.PHASE_9_HISTORICAL_OVERLAP]
    assert unseen["n_questions"] == 1 and overlap["n_questions"] == 1
    assert standard["n_questions"] == 2
    assert unseen["at_budget"]["2048"]["full_support"] == 0.0
    assert overlap["at_budget"]["2048"]["full_support"] == 1.0
    assert standard["at_budget"]["2048"]["full_support"] == 0.5
    assert standard["full_support_at_k"]["2"] == 0.5
    assert unseen["supported_at_primary_budget"] == 0


def test_the_reproduction_passes_only_on_the_exact_recorded_counts():
    counts = {"dense": 487, "hybrid-bm25": 518, "hybrid-entity-hop": 518}

    assert p9.reproduction_verdict(counts, counts)["reproduced"] is True
    missed = p9.reproduction_verdict({**counts, "hybrid-entity-hop": 517}, counts)
    assert missed["reproduced"] is False
    assert missed["mismatches"] == {"hybrid-entity-hop": {"measured": 517, "recorded": 518}}


# --- The build (S5) ------------------------------------------------------------------------

import numpy as np  # noqa: E402

from concept_embeddings_rag.embeddings.cache import unit_set_hash as corpus_hash  # noqa: E402


class WordCounter:
    def count_units(self, units):
        return {unit.unit_id: len(unit.indexable_text.split()) for unit in units}


class StubBGE:
    name = config.EMBEDDING_MODEL
    revision = config.EMBEDDING_REVISION
    normalize = True
    dim = 4

    def encode(self, texts):
        vectors = np.ones((len(texts), self.dim), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def few_units() -> list[IndexingUnit]:
    return [a_unit("A", "Alpha beta."), a_unit("B", "Gamma delta epsilon."), a_unit("C")]


def test_the_token_counts_cover_the_corpus_and_refuse_another_one(tmp_path):
    units = few_units()

    body = p9.build_token_counts(units, WordCounter(), tmp_path, unit_set_hash="h")
    counts = p9.load_phase9_token_counts(tmp_path, unit_set_hash="h")

    assert counts == WordCounter().count_units(units)
    assert body["n_units"] == 3
    with pytest.raises(p9.FullWikiPhaseError, match="another corpus"):
        p9.load_phase9_token_counts(tmp_path, unit_set_hash="other")


def test_the_bm25_digest_is_a_function_of_the_corpus_alone():
    first, record = p9.build_bm25(few_units(), unit_set_hash="h")
    second, _ = p9.build_bm25(few_units(), unit_set_hash="h")
    other, _ = p9.build_bm25(few_units()[:2], unit_set_hash="h2")

    assert p9.bm25_digest(first) == p9.bm25_digest(second)
    assert p9.bm25_digest(first)[0] != p9.bm25_digest(other)[0]
    assert record["index_digest"] == p9.bm25_digest(first)[0]
    assert record["n_units"] == 3


def test_the_entity_statistics_count_incidences_empty_rows_and_hubs():
    stats = p9.entity_index_statistics(a_stage()._index)

    # Bridge in two rows, Common in three; 117 of 120 rows hold no entity.
    assert stats["n_entity_nodes"] == 2
    assert stats["n_incidences"] == 5
    assert stats["zero_entity_units"] == 117
    assert stats["df_percentiles"]["max"] == 3
    assert stats["largest_hubs"][0] == {"form": "common", "df": 3}


def build_embedding(tmp_path, questions, backend=None):
    return p9.build_embedding(
        few_units(),
        questions,
        backend or StubBGE(),
        tmp_path / "cache",
        tmp_path / "cache" / "questions",
        tmp_path,
        unit_set_hash=corpus_hash([unit.unit_id for unit in few_units()]),
        weights_sha256="a" * 64,
        resolved_revision=config.EMBEDDING_REVISION,
        hardware={"device": "cpu"},
        hourly_rate_usd=0.5,
    )


def test_the_embedding_manifest_records_identity_size_and_cost_once(tmp_path):
    body = build_embedding(tmp_path, toy_questions())

    assert body["n_units"] == 3 and body["n_questions"] == 2
    assert body["dim"] == 4 and body["normalized"] is True
    assert body["vector_bytes"] == 3 * 4 * 4
    assert body["weights_sha256"] == "a" * 64
    assert body["question_cache_key"] == p9.question_cache_key_for(toy_questions(), StubBGE())
    with pytest.raises(p9.FullWikiPhaseError, match="already records"):
        build_embedding(tmp_path, toy_questions())


def test_the_embedding_refuses_any_dense_model_but_bge_small(tmp_path):
    class Other(StubBGE):
        name = "Qwen/Qwen3-Embedding-0.6B"

    with pytest.raises(p9.FullWikiPhaseError, match="BGE-small"):
        build_embedding(tmp_path, toy_questions(), backend=Other())


# --- The probe, the single pass (S6) and the outcome (S7) --------------------------------


def test_the_probe_sample_is_seeded_and_drawn_only_from_historical_dev_questions():
    questions = [
        Question(f"q{i:03d}", f"text {i}", "", ("a", "b"), (), config.PHASE_9_HISTORICAL_OVERLAP)
        for i in range(300)
    ]
    dev = {f"q{i:03d}" for i in range(0, 300, 2)}

    first = p9.probe_sample(questions, dev, n=100, seed=42)
    second = p9.probe_sample(list(reversed(questions)), dev, n=100, seed=42)

    assert [q.qid for q in first] == [q.qid for q in second]
    assert len(first) == 100
    assert all(q.qid in dev for q in first)


def test_the_probe_verdict_stops_on_a_system_past_the_query_ceiling():
    fast = {"dense": 0.4, "hybrid-bm25": 0.5, "hybrid-entity-hop": 0.99}
    slow = {**fast, "hybrid-bm25": 1.01}

    assert p9.probe_verdict(fast)["terminal_state"] is None
    stopped = p9.probe_verdict(slow)
    assert stopped["terminal_state"] == p9.OPERATIONAL_STOP
    assert stopped["over_ceiling"] == ["hybrid-bm25"]


def test_the_columnwise_hop_is_chosen_only_when_the_reference_nears_the_ceiling():
    assert p9.hop_implementation(0.5) == p9.REFERENCE_HOP
    assert p9.hop_implementation(0.8) == p9.COLUMNWISE_HOP
    assert p9.hop_implementation(3.0) == p9.COLUMNWISE_HOP


def test_the_pass_refuses_to_start_twice(tmp_path):
    p9.start_pass(tmp_path, probe_digest="p", code_commit="c")

    with pytest.raises(p9.FullWikiPhaseError, match="once"):
        p9.start_pass(tmp_path, probe_digest="p", code_commit="c")


def test_a_run_is_written_once_with_its_outcomes_and_read_back(tmp_path):
    records = measured(tmp_path)["dense"]

    path = p9.write_run(tmp_path, "dense", records, body={"system": "dense"})
    loaded = p9.load_outcomes(tmp_path, "dense")

    assert path.name == "run-dense.json"
    assert loaded == records
    with pytest.raises(p9.FullWikiPhaseError, match="already"):
        p9.write_run(tmp_path, "dense", records, body={"system": "dense"})


def test_the_exact_two_sided_p_is_the_binomial_on_discordant_questions():
    assert p9.exact_two_sided_p(10, 0) == pytest.approx(2 / 1024)
    assert p9.exact_two_sided_p(0, 0) == 1.0
    assert p9.exact_two_sided_p(5, 5) == 1.0
    assert p9.exact_two_sided_p(3, 1) == pytest.approx(0.625)


def outcome_records(pairs):
    """`pairs` of (dense success, entity success) on retrieval-unseen questions."""
    dense, entity = [], []
    for index, (a, c) in enumerate(pairs):
        for rows, value in ((dense, a), (entity, c)):
            rows.append(
                {
                    "qid": f"q{index}",
                    "cohort": config.PHASE_9_RETRIEVAL_UNSEEN,
                    "budgets": {"2048": {"full_support": float(value)}},
                }
            )
    # A historical-overlap question never enters the primary test, whatever it scores.
    dense.append(
        {
            "qid": "h",
            "cohort": config.PHASE_9_HISTORICAL_OVERLAP,
            "budgets": {"2048": {"full_support": 0.0}},
        }
    )
    entity.append(
        {
            "qid": "h",
            "cohort": config.PHASE_9_HISTORICAL_OVERLAP,
            "budgets": {"2048": {"full_support": 1.0}},
        }
    )
    return dense, entity


@pytest.mark.parametrize(
    ("pairs", "label"),
    [
        ([(0, 1)] * 10 + [(1, 1)] * 5, "SCALE_SUPPORTED"),
        ([(1, 0)] * 10, "SCALE_REGRESSION"),
        ([(0, 1)] * 3 + [(1, 0)] * 1, "SCALE_NOT_SUPPORTED"),
        ([(1, 1)] * 4, "SCALE_NOT_SUPPORTED"),
    ],
)
def test_the_terminal_label_is_derived_mechanically_from_the_paired_counts(pairs, label):
    dense, entity = outcome_records(pairs)

    outcome = p9.primary_outcome(dense, entity)

    assert outcome["terminal_state"] == label
    assert outcome["primary_cohort"] == "retrieval-unseen"
    assert outcome["wins"] + outcome["losses"] + outcome["ties"] == len(pairs)


def test_the_counts_and_delta_are_read_off_the_primary_cohort_only():
    dense, entity = outcome_records([(0, 1)] * 3 + [(1, 1)])

    outcome = p9.primary_outcome(dense, entity)

    assert outcome["dense_successes"] == 1
    assert outcome["entity_successes"] == 4
    assert outcome["delta_percentage_points"] == pytest.approx(75.0)
    assert outcome["n_questions"] == 4


def test_a_stop_overrides_the_scientific_label():
    dense, entity = outcome_records([(0, 1)] * 10)

    assert p9.primary_outcome(dense, entity, stop=p9.OPERATIONAL_STOP)["terminal_state"] == (
        p9.OPERATIONAL_STOP
    )


def test_the_vectors_digest_is_the_sha256_of_the_float32_bytes():
    import hashlib

    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)

    assert p9.vectors_digest(vectors) == hashlib.sha256(vectors.tobytes()).hexdigest()
    assert p9.vectors_digest(vectors) != p9.vectors_digest(vectors[::-1])


def test_blockwise_encoding_fills_every_row_in_corpus_order():
    class Positional(StubBGE):
        def encode(self, texts):
            return np.array([[float(len(t)), 1.0, 0.0, 0.0] for t in texts], dtype=np.float32)

    units = few_units()

    blocked = p9.encode_blockwise(units, Positional(), block=2)

    assert np.array_equal(blocked, Positional().encode([u.indexable_text for u in units]))
