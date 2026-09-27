"""Phase 12: location, exclusion and seed choice (S1) - what decides a fused ranking.

`choose_seeds` is the one function whose output can change a measured result: which of P1's
entities become seeds. Everything here is fixture-based and fast; the real GLiNER index and
corpus are read only by the CLI stages.
"""

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
