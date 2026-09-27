"""Phase 13: mentions, windows, the entity ranking, seeds and the screen rules.

These are the functions whose output can change a measured result: which words score an
entity, which entity ranks first, which entities seed the hop, and whether the screen passes.
Everything is fixture-based and fast; the real corpus, index and caches are read only by the
CLI stages.
"""

import json

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import phase13 as p13

# --- mentions --------------------------------------------------------------------------------


def test_mentions_respect_word_boundaries():
    assert p13.mentions("cameron", "james cameron directed it") == [p13.Mention(6, 1, 1)]
    assert p13.mentions("cameron", "camerons directed it") == []
    assert p13.mentions("cameron", "it was cameron.") == [p13.Mention(7, 2, 2)]


def test_mentions_escape_regex_metacharacters():
    assert p13.mentions("st. louis", "born in st. louis, missouri") == [p13.Mention(8, 2, 3)]
    assert p13.mentions("st. louis", "born in st louis, missouri") == []
    assert p13.mentions("a(b)", "it mentions a(b) once") == [p13.Mention(12, 2, 2)]


def test_mentions_finds_every_match_in_one_sentence():
    found = p13.mentions("paris", "from paris to london and back to paris")
    assert [(m.first_word, m.last_word) for m in found] == [(1, 1), (7, 7)]
    assert [m.offset for m in found] == [5, 33]


def test_mentions_normalize_both_sides():
    # Diacritics and case fold on both sides; the offset lives in the normalized sentence.
    assert p13.mentions("Café Rouge", "We met at the CAFÉ ROUGE today") == [p13.Mention(14, 4, 5)]
    assert p13.mentions("", "anything") == []


def test_a_match_inside_a_word_with_an_apostrophe_overlaps_the_whole_word():
    # "cameron's": the apostrophe is a non-word character, so the match is bounded; the
    # whitespace word it overlaps is "cameron's", which is masked whole.
    found = p13.mentions("cameron", "it is cameron's best film")
    assert found == [p13.Mention(6, 2, 2)]
    assert p13.window("it is cameron's best film", found[0], 5) == "it is best film"


# --- windows ---------------------------------------------------------------------------------


def test_window_clips_at_both_sentence_edges():
    sentence = "one two three x four five"
    (mention,) = p13.mentions("x", sentence)
    assert p13.window(sentence, mention, 5) == "one two three four five"
    long = "a1 a2 a3 a4 a5 a6 a7 x b1 b2 b3 b4 b5 b6 b7"
    (mention,) = p13.mentions("x", long)
    assert p13.window(long, mention, 5) == "a3 a4 a5 a6 a7 b1 b2 b3 b4 b5"


def test_window_masks_a_multi_word_mention():
    sentence = "it was directed by james cameron and produced by him"
    (mention,) = p13.mentions("james cameron", sentence)
    assert (mention.first_word, mention.last_word) == (4, 5)
    assert p13.window(sentence, mention, 5) == "it was directed by and produced by him"


def test_a_mention_that_is_the_whole_sentence_has_an_empty_window():
    (mention,) = p13.mentions("titanic", "Titanic.")
    assert p13.window("Titanic.", mention, 5) == ""


def test_entity_windows_keep_only_non_empty_windows_in_sentence_then_word_order():
    sentences = ["Paris.", "From Paris to Lyon and back to Paris.", "Nothing here."]
    found = p13.entity_windows([7, 8], {7: "paris", 8: "berlin"}, sentences)
    assert set(found) == {7}  # 8 is never mentioned
    paris = found[7]
    assert paris.spans == ((0, 0), (1, 5), (1, 31))  # every mention, the empty one included
    assert [(w.position, w.first_word) for w in paris.windows] == [(1, 1), (1, 7)]
    assert paris.windows[0].text == "from to lyon and back to"
    assert paris.windows[1].text == "to lyon and back to"
    assert paris.empty == 1
    assert paris.first_mention == (0, 0)


def test_the_normalized_sentence_drops_a_leading_article_before_words_are_counted():
    # Normalization v1 applied to the sentence: windows are cut from that normalized text.
    found = p13.entity_windows([1], {1: "film"}, ["The film stars him."])
    assert found[1].windows[0].text == "stars him"
    assert found[1].windows[0].first_word == 0


# --- the entity ranking ----------------------------------------------------------------------

TITANIC_FORMS = {1: "james cameron", 2: "titanic", 3: "leonardo dicaprio", 4: "unlocated"}
TITANIC_SENTENCES = [
    "Titanic is a 1997 American epic romance film.",
    "Titanic was directed by James Cameron.",
    "Titanic stars Leonardo DiCaprio.",
]


def test_titanic_the_masked_window_ranks_the_director_first():
    windows = p13.entity_windows([1, 2, 3, 4], TITANIC_FORMS, TITANIC_SENTENCES)
    assert set(windows) == {1, 2, 3}
    # A toy bag-of-words encoder: the question's words against the window's words.
    vocabulary = ["directed", "by", "stars", "film", "epic", "titanic", "james", "cameron"]

    def encode(text):
        words = text.split()
        vector = np.array([float(words.count(v)) for v in vocabulary], dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    rows = [(node, encode(w.text)) for node, e in windows.items() for w in e.windows]
    node_ids = [node for node, _ in rows]
    vectors = np.stack([vector for _, vector in rows])
    question = encode("who directed titanic")
    scores = p13.entity_scores(node_ids, vectors, question)
    weights = {1: 1.0, 2: 5.0, 3: 1.0}  # titanic is the rarest: it must not win by name
    ranking = p13.rank_entities(scores, weights)
    assert ranking[0] == 1
    assert "titanic" not in windows[2].windows[0].text  # masked


def test_entity_scores_take_the_maximum_over_an_entitys_windows():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    scores = p13.entity_scores([5, 5, 6], vectors, np.array([0.0, 1.0], dtype=np.float32))
    assert scores == {5: pytest.approx(1.0), 6: pytest.approx(0.8)}
    assert p13.entity_scores([], np.zeros((0, 2), dtype=np.float32), np.ones(2)) == {}


def test_rank_entities_ties_by_rarity_then_node_id():
    scores = {3: 0.5, 1: 0.5, 2: 0.5, 9: 0.9}
    weights = {3: 2.0, 1: 1.0, 2: 1.0, 9: 0.1}
    assert p13.rank_entities(scores, weights) == [9, 3, 1, 2]


def test_rank_entities_reads_weights_from_an_array():
    weights = np.array([0.0, 1.0, 3.0])
    assert p13.rank_entities({1: 0.2, 2: 0.2}, weights) == [2, 1]


# --- seeds -----------------------------------------------------------------------------------


def test_seeds_take_the_first_m_of_the_ranking():
    assert p13.seeds([9, 3, 1], [1, 3, 9, 12], 1) == [9]
    assert p13.seeds([9, 3, 1], [1, 3, 9, 12], 2) == [9, 3]


def test_seeds_with_m_larger_than_the_scored_set_take_every_scored_entity():
    assert p13.seeds([9, 3], [1, 3, 9, 12], 3) == [9, 3]


def test_seeds_at_all_are_every_p1_node_unscored_included():
    assert p13.seeds([9, 3], [12, 1, 3, 9], None) == [1, 3, 9, 12]


def test_no_scored_entity_gives_no_seed_below_all():
    assert p13.seeds([], [1, 2], 1) == []
    assert p13.seeds([], [1, 2], None) == [1, 2]


def test_seed_key_names_all():
    assert p13.seed_key(None) == "window-hop@m=all"
    assert p13.seed_key(2) == "window-hop@m=2"


# --- the screen rules ------------------------------------------------------------------------


def test_first_mention_ties_by_sentence_then_offset_then_node_id():
    first = {1: (1, 10), 2: (0, 30), 3: (0, 5), 4: (0, 5)}
    assert p13.first_mention([1, 2, 3, 4], first) == 3
    assert p13.first_mention([1, 2], first) == 2
    assert p13.first_mention([], first) is None


def test_rarest_ties_by_node_id():
    weights = {1: 2.0, 2: 3.0, 3: 3.0}
    assert p13.rarest([1, 2, 3], weights) == 2
    assert p13.rarest([1], weights) == 1
    assert p13.rarest([], weights) is None


def test_hit_needs_a_choice_inside_the_bridge():
    assert p13.hit(2, {1, 2}) is True
    assert p13.hit(3, {1, 2}) is False
    assert p13.hit(None, {1, 2}) is False


def test_the_screen_passes_at_exactly_five_points_and_not_just_below():
    # 200 items: 5.0 pp is 10 items.
    at = p13.screen_verdict(200, hits_w=60, hits_t1=50, hits_t2=40)
    assert at["terminal_state"] is None and at["passed"] is True
    assert at["reference_rule"] == "T1"
    assert at["rates_pp"]["W"] == pytest.approx(30.0)
    assert at["reference_pp"] == pytest.approx(25.0)
    below = p13.screen_verdict(200, hits_w=59, hits_t1=40, hits_t2=50)
    assert below["terminal_state"] == p13.SCREEN_STOP and below["passed"] is False
    assert below["reference_rule"] == "T2"
    # A margin that is not a whole number of items: 5.0 pp of 2,651 is 132.55 items.
    assert p13.screen_verdict(2651, hits_w=1133, hits_t1=1000, hits_t2=0)["passed"] is True
    assert p13.screen_verdict(2651, hits_w=1132, hits_t1=1000, hits_t2=0)["passed"] is False


def test_the_margin_is_the_frozen_constant():
    assert config.PHASE_13_SCREEN_MARGIN_PP == 5.0
    assert p13.screen_verdict(100, 10, 5, 5)["margin_pp"] == 5.0


# --- S2: the window cache and the coverage ---------------------------------------------------


def a_window_cache(tmp_path, set_name="dev"):
    unit_ids = ["u1", "u1", "u2"]
    node_ids = [3, 4, 3]
    positions = [0, 1, 0]
    first_words = [2, 0, 5]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    key = p13.window_cache_key("model", "rev", set_name, ["u1", "u2"])
    p13.write_window_cache(
        tmp_path, set_name, key, unit_ids, node_ids, positions, first_words, vectors
    )
    manifest = {
        "key": key,
        "vectors_digest": p13.vectors_digest(vectors),
        "rows_digest": p13.rows_digest(unit_ids, node_ids, positions, first_words),
    }
    p13.write_window_manifest(tmp_path, set_name, manifest)
    return key, vectors


def test_the_window_cache_key_changes_with_the_set_and_the_window_rule(monkeypatch):
    key_dev = p13.window_cache_key("m", "r", "dev", ["u1", "u2"])
    assert key_dev == p13.window_cache_key("m", "r", "dev", ["u2", "u1"])
    assert p13.window_cache_key("m", "r", "test-11", ["u1", "u2"]) != key_dev
    assert p13.window_cache_key("m2", "r", "dev", ["u1", "u2"]) != key_dev
    monkeypatch.setattr(p13.config, "PHASE_13_WINDOW_RULE", "window-5w-other-v2")
    assert p13.window_cache_key("m", "r", "dev", ["u1", "u2"]) != key_dev


def test_the_window_cache_is_written_once(tmp_path):
    key, vectors = a_window_cache(tmp_path)
    cache = p13.read_window_cache(tmp_path, "dev")
    assert cache.unit_ids == ["u1", "u1", "u2"]
    assert cache.node_ids == [3, 4, 3]
    assert cache.positions == [0, 1, 0]
    assert cache.first_words == [2, 0, 5]
    assert np.array_equal(cache.vectors, vectors)
    assert cache.manifest["key"] == key
    with pytest.raises(p13.Phase13Error, match="already holds"):
        p13.write_window_cache(tmp_path, "dev", key, ["u1"], [3], [0], [0], vectors[:1])
    with pytest.raises(p13.Phase13Error, match="already records"):
        p13.write_window_manifest(tmp_path, "dev", {"key": key})


def test_a_manifest_whose_digest_does_not_match_is_refused(tmp_path):
    a_window_cache(tmp_path)
    path = tmp_path / p13.window_manifest_filename("dev")
    body = json.loads(path.read_text())
    path.write_text(json.dumps({**body, "vectors_digest": "not-the-digest"}))
    with pytest.raises(p13.Phase13Error, match="not match its recorded digest"):
        p13.read_window_cache(tmp_path, "dev")
    path.write_text(json.dumps({**body, "rows_digest": "not-the-digest"}))
    with pytest.raises(p13.Phase13Error, match="not match its recorded digest"):
        p13.read_window_cache(tmp_path, "dev")


def test_missing_window_cache_files_are_refused(tmp_path):
    with pytest.raises(p13.Phase13Error, match="does not exist"):
        p13.read_window_cache(tmp_path, "dev")


def test_the_cached_windows_serve_one_unit_in_row_order(tmp_path):
    a_window_cache(tmp_path)
    cache = p13.read_window_cache(tmp_path, "dev")
    node_ids, vectors = cache.rows_of("u1")
    assert node_ids == [3, 4]
    assert np.array_equal(vectors, cache.vectors[:2])
    with pytest.raises(p13.Phase13Error, match="not in the window cache"):
        cache.rows_of("u9")


def test_window_coverage_on_a_small_fixture():
    sentences = ["Paris.", "From Paris to Lyon.", "Titanic."]
    unit_entities = {
        "u1": p13.entity_windows([1, 2, 3], {1: "paris", 2: "lyon", 3: "titanic"}, sentences),
        "u2": {},
    }
    # q1 and q2 share P1 u1 (3 entity nodes, all located, 2 scored); q3's P1 u2 has 2 nodes
    # located nowhere.
    p1_nodes = {"u1": 3, "u2": 2}
    coverage = p13.window_coverage(["u1", "u1", "u2"], p1_nodes, unit_entities)
    assert coverage["p1_entity_nodes_total"] == 8
    assert coverage["p1_entity_nodes_located"] == 6
    assert coverage["p1_entity_nodes_scored"] == 4
    assert coverage["scored_share"] == pytest.approx(0.5)
    assert coverage["questions_without_scored_entity"] == 1
    assert coverage["questions_without_scored_entity_share"] == pytest.approx(1 / 3)
    # Distinct units: paris 2 mentions (1 empty), lyon 1, titanic 1 (empty).
    assert coverage["units"] == 2
    assert coverage["mentions"] == 4
    assert coverage["windows"] == 2
    assert coverage["empty_windows"] == 2
    assert coverage["mentions_per_located_entity"]["mean"] == pytest.approx(4 / 3)


# --- S3: the screen population, the choices per question, the tally --------------------------

NODES_OF = {
    "p1": {1, 2, 3},
    "g_top": {1},  # inside the Dense top 10: out
    "g_share": {2, 9},  # shares node 2 with P1: in, bridge {2}
    "g_none": {8, 9},  # shares nothing: out
    "g_two": {1, 3, 7},  # bridge {1, 3}
}


def test_the_population_keeps_gold_outside_the_top10_in_the_index_sharing_an_entity():
    items = p13.screen_population(
        dense_top10=["p1", "g_top", "x"],
        gold=["g_top", "g_share", "g_none", "g_missing", "g_two"],
        p1_nodes=[1, 2, 3],
        nodes_of=NODES_OF.get,
    )
    assert items == [("g_share", frozenset({2})), ("g_two", frozenset({1, 3}))]


def a_question_entities():
    sentences = [
        "Titanic is a 1997 American epic romance film.",
        "Titanic was directed by James Cameron.",
        "Titanic stars Leonardo DiCaprio.",
    ]
    return p13.entity_windows([1, 2, 3], TITANIC_FORMS, sentences)


def test_question_choices_apply_each_rule_to_the_same_scored_set():
    entities = a_question_entities()  # 2 (titanic) is mentioned first, in sentence 0
    scores = {1: 0.9, 2: 0.4, 3: 0.7}
    weights = {1: 1.0, 2: 1.0, 3: 4.0}
    rel = [0.1, 0.2, 0.8]  # sentence 2 is the most similar: its first mention is titanic
    choices = p13.question_choices(entities, scores, weights, rel)
    assert choices.ranking == [1, 3, 2]
    assert choices.w == 1
    assert choices.t1 == 2
    assert choices.t2 == 3
    assert choices.sentence_rule == 2


def test_the_sentence_rule_skips_sentences_without_a_scored_entity():
    entities = a_question_entities()
    scores = {1: 0.9, 3: 0.7}  # titanic unscored: sentence 0 holds no scored entity
    rel = [0.9, 0.2, 0.8]
    choices = p13.question_choices(entities, scores, {1: 1.0, 3: 1.0}, rel)
    assert choices.sentence_rule == 3
    assert choices.t1 == 1  # the earliest scored mention: james cameron in sentence 1


def test_no_scored_entity_misses_every_rule():
    choices = p13.question_choices(a_question_entities(), {}, {}, [0.1, 0.2, 0.3])
    assert choices.ranking == [] and choices.w is None and choices.t1 is None
    assert choices.t2 is None and choices.sentence_rule is None


def test_the_tally_counts_hits_per_rule():
    hit_all = p13.QuestionChoices(ranking=[1, 2, 3], t1=1, t2=1, sentence_rule=1)
    w_only = p13.QuestionChoices(ranking=[2, 5, 1], t1=5, t2=6, sentence_rule=5)
    nothing = p13.QuestionChoices(ranking=[], t1=None, t2=None, sentence_rule=None)
    items = [
        (frozenset({1}), hit_all),
        (frozenset({2}), w_only),
        (frozenset({1}), w_only),  # W's first misses; its first 3 hold 1
        (frozenset({1}), nothing),
    ]
    body = p13.screen_tally(items, expected_population=4)
    assert body["hits"] == {"W": 2, "T1": 1, "T2": 1}
    assert body["descriptive"]["sentence_rule"]["hits"] == 1
    assert body["descriptive"]["w_first_2"]["hits"] == 2
    assert body["descriptive"]["w_first_3"]["hits"] == 3
    assert body["descriptive"]["items_without_scored_entity"] == 1
    assert body["population"] == 4


def test_the_tally_refuses_a_population_other_than_the_expected_one():
    item = (frozenset({1}), p13.QuestionChoices(ranking=[1], t1=1, t2=1, sentence_rule=1))
    with pytest.raises(p13.Phase13Error, match="population"):
        p13.screen_tally([item, item], expected_population=3)
    assert config.PHASE_13_SCREEN_POPULATION == 2651


def test_window_rows_follow_unit_then_node_then_window_order():
    sentences = ["From Paris to Lyon.", "Lyon is near Paris."]
    entities = p13.entity_windows([2, 1], {1: "paris", 2: "lyon"}, sentences)
    rows = p13.window_rows(["u1", "u2"], {"u1": entities, "u2": {}})
    assert rows[0] == ["u1"] * 4
    assert rows[1] == [1, 1, 2, 2]
    assert rows[2] == [0, 1, 0, 1]
    assert rows[3] == [1, 3, 3, 0]
    assert rows[4] == ["from to lyon", "lyon is near", "from paris to", "is near paris"]


# --- S5: the D4 reproduction verdict ---------------------------------------------------------


def test_the_reproduction_passes_only_on_the_exact_count_with_nothing_moved():
    assert p13.reproduction_verdict(4801, [], [])["passed"]
    assert not p13.reproduction_verdict(4800, [], [])["passed"]
    assert not p13.reproduction_verdict(4801, ["q1"], [])["passed"]
    assert not p13.reproduction_verdict(4801, [], ["q2"])["passed"]
    point = p13.reproduction_verdict(4801, [], [])["point"]
    assert point == {"m": None, "weights": {"dense": 0.5, "bm25": 0.3, "window-hop": 0.2}}


# --- S6 (D5, D6): the grid, the tie rule and the dev gate ------------------------------------


def test_the_grid_has_66_triples_per_m_and_264_points_containing_p10c():
    points = p13.grid_points()
    assert len(points) == 264 and len(set(points)) == 264
    assert len({weights for _m, weights in points}) == 66
    assert [m for m, _w in points[::66]] == [1, 2, 3, None]
    assert (config.PHASE_13_P10C_M, config.PHASE_13_P10C_WEIGHTS) in points
    assert all(abs(sum(w) - 1.0) < 1e-9 and min(w) >= 0.0 for _m, w in points)


def a_fit_point(supported, recall=0.0, m=None, weights=(0.5, 0.3, 0.2)):
    return {"supported": supported, "gold_recall_sum": recall, "m": m, "weights": weights}


def test_the_fit_tie_rule_applies_in_the_spec_order():
    assert p13.choose_point([a_fit_point(10), a_fit_point(11)])["supported"] == 11
    assert p13.choose_point([a_fit_point(10, 1.0), a_fit_point(10, 2.0)])["gold_recall_sum"] == 2.0
    m1, m3, m_all = a_fit_point(10, 1.0, m=1), a_fit_point(10, 1.0, m=3), a_fit_point(10, 1.0)
    assert p13.choose_point([m1, m_all, m3]) == m_all
    assert p13.choose_point([m1, m3]) == m3
    a = a_fit_point(10, 1.0, m=2, weights=(0.6, 0.2, 0.2))
    b = a_fit_point(10, 1.0, m=2, weights=(0.5, 0.4, 0.1))
    assert p13.choose_point([b, a]) == a
    c = a_fit_point(10, 1.0, m=2, weights=(0.5, 0.2, 0.3))
    assert p13.choose_point([c, b]) == b
    # Full Support dominates everything below it.
    assert p13.choose_point([a_fit_point(11, 0.0, m=1), a_fit_point(10, 9.0)])["m"] == 1


def test_the_dev_gate_falls_at_4834():
    assert config.PHASE_13_DEV_BAR == 4834
    assert p13.dev_gate(a_fit_point(4834)) is None
    assert p13.dev_gate(a_fit_point(4833)) == p13.DEV_STOP
