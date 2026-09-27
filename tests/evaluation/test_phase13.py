"""Phase 13: mentions, windows, the entity ranking, seeds and the screen rules.

These are the functions whose output can change a measured result: which words score an
entity, which entity ranks first, which entities seed the hop, and whether the screen passes.
Everything is fixture-based and fast; the real corpus, index and caches are read only by the
CLI stages.
"""

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
