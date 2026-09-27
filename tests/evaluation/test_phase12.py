"""Phase 12: location, exclusion and seed choice (S1) - what decides a fused ranking.

`choose_seeds` is the one function whose output can change a measured result: which of P1's
entities become seeds. Everything here is fixture-based and fast; the real GLiNER index and
corpus are read only by the CLI stages.
"""

import json

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import phase12 as p12


def test_occurs_respects_word_boundaries():
    assert p12.occurs("cameron", "james cameron directed it") is True
    assert p12.occurs("cameron", "camerons directed it") is False
    assert p12.occurs("cameron", "it was cameron.") is True


def test_occurs_normalizes_both_sides():
    assert p12.occurs("James Cameron", "It was directed by james cameron.") is True
    assert p12.occurs("café", "the Café is nice") is True
    assert p12.occurs("", "anything") is False


def test_occurs_escapes_regex_metacharacters_in_the_form():
    assert p12.occurs("st. louis", "born in st. louis, missouri") is True
    assert p12.occurs("st. louis", "born in st louis, missouri") is False  # the dot is literal
    assert p12.occurs("a(b)", "it mentions a(b) once") is True


TITANIC_FORMS = {
    1: "james cameron",
    2: "leonardo dicaprio",
    3: "kate winslet",
    4: "unlocated entity",
}
TITANIC_SENTENCES = [
    "Titanic is a 1997 American film.",
    "It was directed by James Cameron.",
    "The film stars Leonardo DiCaprio and Kate Winslet.",
]
TITANIC_NODES = [1, 2, 3, 4]


def test_locate_finds_only_the_nodes_mentioned_in_a_sentence():
    located = p12.locate(TITANIC_NODES, TITANIC_FORMS, TITANIC_SENTENCES)
    assert located == {1: [1], 2: [2], 3: [2]}  # 4 is unlocated: only the title names it


def test_named_in_question_matches_by_the_same_rule_as_locate():
    assert p12.named_in_question(TITANIC_NODES, TITANIC_FORMS, "Who directed Titanic?") == set()
    assert p12.named_in_question(
        TITANIC_NODES, TITANIC_FORMS, "In which year was James Cameron born?"
    ) == {1}


def test_choose_seeds_s1_picks_the_entities_of_the_single_best_sentence():
    located = p12.locate(TITANIC_NODES, TITANIC_FORMS, TITANIC_SENTENCES)
    rel = [0.1, 0.9, 0.5]  # sentence 1 ("directed by James Cameron") is most relevant
    assert p12.choose_seeds(TITANIC_NODES, located, rel, s=1, exclude=set()) == [1]


def test_choose_seeds_s2_adds_the_second_best_sentences_entities():
    located = p12.locate(TITANIC_NODES, TITANIC_FORMS, TITANIC_SENTENCES)
    rel = [0.1, 0.9, 0.5]
    assert p12.choose_seeds(TITANIC_NODES, located, rel, s=2, exclude=set()) == [1, 2, 3]


def test_choose_seeds_none_keeps_unlocated_nodes_too():
    located = p12.locate(TITANIC_NODES, TITANIC_FORMS, TITANIC_SENTENCES)
    assert p12.choose_seeds(TITANIC_NODES, located, [], s=None, exclude=set()) == [1, 2, 3, 4]


def test_choose_seeds_never_seeds_an_unlocated_node_when_s_is_not_none():
    located = p12.locate(TITANIC_NODES, TITANIC_FORMS, TITANIC_SENTENCES)
    rel = [0.1, 0.9, 0.5]
    seeds = p12.choose_seeds(TITANIC_NODES, located, rel, s=3, exclude=set())
    assert 4 not in seeds  # located in no sentence, so never a seed below s = None


def test_choose_seeds_ties_break_by_sentence_position():
    located = {1: [2], 2: [0]}
    rel = [0.5, 0.5, 0.5]  # every sentence ties: position 0 must win the tie
    assert p12.choose_seeds([1, 2], located, rel, s=1, exclude=set()) == [2]


def test_choose_seeds_applies_exclusion_before_ranking():
    located = p12.locate(TITANIC_NODES, TITANIC_FORMS, TITANIC_SENTENCES)
    rel = [0.1, 0.9, 0.5]
    excluded = p12.named_in_question(
        TITANIC_NODES, TITANIC_FORMS, "In which year was James Cameron born?"
    )
    # Node 1's sentence no longer holds an eligible entity once excluded, so it drops out of
    # the ranking entirely rather than merely losing a tie.
    assert p12.choose_seeds(TITANIC_NODES, located, rel, s=1, exclude=excluded) == [2, 3]


def test_choose_seeds_empty_when_every_entity_is_excluded():
    located = p12.locate(TITANIC_NODES, TITANIC_FORMS, TITANIC_SENTENCES)
    assert p12.choose_seeds(TITANIC_NODES, located, [], s=None, exclude=set(TITANIC_NODES)) == []


def test_choose_seeds_empty_when_nothing_is_located_and_s_is_not_none():
    assert p12.choose_seeds([4], {}, [], s=1, exclude=set()) == []


def test_seed_key_names_all_and_yes_no():
    assert p12.seed_key(None, False) == "seeded-hop@s=all-x=no"
    assert p12.seed_key(1, True) == "seeded-hop@s=1-x=yes"


def test_seed_grid_has_the_six_declared_configurations():
    grid = p12.seed_grid()
    assert len(grid) == 6 and len(set(grid)) == 6
    assert (None, False) in grid and (None, True) in grid


# --- S2: the sentence cache ------------------------------------------------------------------


def a_cache(tmp_path, set_name="dev", unit_ids=("u1", "u1", "u2"), positions=(0, 1, 0)):
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    key = p12.sentence_cache_key("model", "rev", set_name, sorted(set(unit_ids)))
    p12.write_sentence_cache(tmp_path, set_name, key, unit_ids, positions, vectors)
    manifest = {"key": key, "vectors_digest": p12.vectors_digest(vectors)}
    p12.write_sentence_manifest(tmp_path, set_name, manifest)
    return key, vectors


def test_the_cache_key_changes_with_the_set_and_the_text_rule(monkeypatch):
    key_dev = p12.sentence_cache_key("m", "r", "dev", ["u1", "u2"])
    key_test = p12.sentence_cache_key("m", "r", "test-11", ["u1", "u2"])
    assert key_dev != key_test
    monkeypatch.setattr(p12.config, "PHASE_12_TEXT_RULE", "sentence-alone-v2")
    assert p12.sentence_cache_key("m", "r", "dev", ["u1", "u2"]) != key_dev


def test_the_sentence_cache_is_written_once(tmp_path):
    key, vectors = a_cache(tmp_path)
    loaded = p12.load_sentence_cache(tmp_path, "dev", key)
    assert loaded[0] == ["u1", "u1", "u2"]
    assert loaded[1] == [0, 1, 0]
    assert np.array_equal(loaded[2], vectors)
    with pytest.raises(p12.Phase12Error, match="already holds"):
        p12.write_sentence_cache(tmp_path, "dev", key, ["u1"], [0], vectors[:1])
    with pytest.raises(p12.Phase12Error, match="already records"):
        p12.write_sentence_manifest(tmp_path, "dev", {"key": key})


def test_reading_the_cache_checks_the_manifest_digest(tmp_path):
    key, vectors = a_cache(tmp_path)
    unit_ids, positions, loaded_vectors, manifest = p12.read_sentence_cache(tmp_path, "dev")
    assert manifest["key"] == key
    assert np.array_equal(loaded_vectors, vectors)
    bad_manifest = {**manifest, "vectors_digest": "not-the-real-digest"}
    (tmp_path / p12.sentence_manifest_filename("dev")).write_text(json.dumps(bad_manifest))
    with pytest.raises(p12.Phase12Error, match="does not match its recorded digest"):
        p12.read_sentence_cache(tmp_path, "dev")


def test_missing_cache_files_are_refused(tmp_path):
    with pytest.raises(p12.Phase12Error, match="does not exist"):
        p12.load_sentence_manifest(tmp_path, "dev")
    with pytest.raises(p12.Phase12Error, match="does not exist"):
        p12.load_sentence_cache(tmp_path, "dev", "deadbeef")


# --- S4: the D4 reproduction verdict and the coverage summary --------------------------------


def test_the_reproduction_passes_only_on_the_exact_count_with_nothing_moved():
    assert p12.reproduction_verdict(4801, [], [])["passed"]
    assert not p12.reproduction_verdict(4800, [], [])["passed"]
    assert not p12.reproduction_verdict(4801, ["q1"], [])["passed"]
    assert not p12.reproduction_verdict(4801, [], ["q2"])["passed"]
    point = p12.reproduction_verdict(4801, [], [])["point"]
    assert point == {
        "s": None,
        "exclude": False,
        "weights": {"dense": 0.5, "bm25": 0.3, "seeded-hop": 0.2},
    }


def test_coverage_summary_on_a_small_fixture():
    entity_totals = [(3, 2), (2, 2), (0, 0)]  # 5 of 5 located overall
    seed_counts = {
        (None, False): [3, 2, 0],
        (1, False): [1, 1, 0],
    }
    summary = p12.coverage_summary(entity_totals, seed_counts)
    assert summary["p1_entity_nodes_total"] == 5
    assert summary["p1_entity_nodes_located"] == 4
    assert summary["located_share"] == pytest.approx(0.8)
    assert summary["by_config"]["seeded-hop@s=all-x=no"]["empty_seed_share"] == pytest.approx(1 / 3)
    assert summary["by_config"]["seeded-hop@s=all-x=no"]["seeds_median"] == 2.0
    assert summary["by_config"]["seeded-hop@s=1-x=no"]["empty_seed_share"] == pytest.approx(1 / 3)


def test_coverage_summary_handles_no_entities_at_all():
    summary = p12.coverage_summary([(0, 0)], {(None, False): [0]})
    assert summary["located_share"] == 0.0
    assert summary["by_config"]["seeded-hop@s=all-x=no"]["empty_seed_share"] == 1.0


# --- S5: the grid, the tie rule and the dev gate ----------------------------------------------


def test_weight_grid_has_66_triples_summing_to_one():
    grid = p12.weight_grid()
    assert len(grid) == 66 and len(set(grid)) == 66
    assert all(abs(sum(w) - 1.0) < 1e-9 and min(w) >= 0.0 for w in grid)
    assert (0.5, 0.3, 0.2) in grid


def test_grid_points_has_396_points_and_contains_p10c():
    points = p12.grid_points()
    assert len(points) == 396
    assert (config.PHASE_12_P10C_POINT, config.PHASE_12_P10C_WEIGHTS) in points


def a_point(supported, recall=0.0, s=None, exclude=False, weights=(0.5, 0.3, 0.2)):
    return {
        "supported": supported,
        "gold_recall_sum": recall,
        "s": s,
        "exclude": exclude,
        "weights": weights,
    }


def test_the_tie_rule_applies_in_the_spec_order():
    assert p12.choose_point([a_point(10), a_point(11)])["supported"] == 11
    assert p12.choose_point([a_point(10, 1.0), a_point(10, 2.0)])["gold_recall_sum"] == 2.0
    # x = no wins a tie on the first two keys.
    no = a_point(10, 1.0, s=1, exclude=False)
    yes = a_point(10, 1.0, s=None, exclude=True)
    assert p12.choose_point([no, yes]) == no
    # Then the larger s, all (None) largest.
    s1, s2, s_all = a_point(10, 1.0, s=1), a_point(10, 1.0, s=2), a_point(10, 1.0, s=None)
    assert p12.choose_point([s1, s_all]) == s_all
    assert p12.choose_point([s1, s2]) == s2
    # Then the larger w_dense, then w_bm25.
    a = a_point(10, 1.0, s=None, weights=(0.6, 0.2, 0.2))
    b = a_point(10, 1.0, s=None, weights=(0.5, 0.4, 0.1))
    assert p12.choose_point([b, a]) == a
    c = a_point(10, 1.0, s=None, weights=(0.5, 0.2, 0.3))
    assert p12.choose_point([c, b]) == b


def test_the_dev_gate_falls_at_4835():
    assert p12.dev_gate(a_point(4835)) is None
    assert p12.dev_gate(a_point(4834)) == p12.DEV_STOP
