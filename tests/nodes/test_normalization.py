"""T7 of Phase 5: normalization v1, one case per rule of decision D6 (HU-3)."""

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.nodes import normalization
from concept_embeddings_rag.nodes.normalization import normalize


def test_the_version_is_the_declared_one():
    assert normalization.VERSION == config.NORMALIZATION_VERSION


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # NFKC: compatibility forms fold to their plain equivalents.
        ("\ufb01le format", "file format"),
        ("\uff21\uff22\uff23 Network", "abc network"),
        # Diacritics are stripped.
        ("Beyonc\xe9", "beyonce"),
        ("S\xe3o Paulo", "sao paulo"),
        # Casefold, not merely lower().
        ("Stra\xdfe", "strasse"),
        ("TELEVISION Series", "television series"),
        # Typographic quotes and apostrophes become ASCII.
        ("Rock \u2019n\u2019 Roll", "rock 'n' roll"),
        ("\u201cThe Wire\u201d", "wire"),
        # Hyphens, dashes, underscores and slashes become spaces.
        ("Coca-Cola", "coca cola"),
        ("singer\u2013songwriter", "singer songwriter"),
        ("AC/DC", "ac dc"),
        ("snake_case", "snake case"),
        # Surrounding punctuation is stripped; inner punctuation is kept.
        ("(album)", "album"),
        ("Inc.", "inc"),
        ("St. Louis", "st. louis"),
        # One leading article is dropped.
        ("The Beatles", "beatles"),
        ("a television series", "television series"),
        ("An Officer", "officer"),
        # Whitespace collapses, including around a dropped article.
        ("  the   New   York  Times ", "new york times"),
    ],
)
def test_each_rule_of_d6(raw: str, expected: str):
    assert normalize(raw) == expected


def test_an_article_is_only_dropped_as_a_whole_leading_word():
    assert normalize("Theatre") == "theatre"
    assert normalize("Anaconda") == "anaconda"
    assert normalize("A") == "a"
    assert normalize("the") == "the"


@pytest.mark.parametrize("raw", ["", "   ", "---", "\u201c\u201d", "()", "_/_"])
def test_a_form_with_nothing_left_normalizes_to_empty(raw: str):
    assert normalize(raw) == ""


@pytest.mark.parametrize(
    ("plural", "singular"),
    [("rivers", "river"), ("series", "sery"), ("physics", "physic"), ("Paris", "Pari")],
)
def test_plural_forms_are_deliberately_not_folded(plural: str, singular: str):
    """D6: a rule-based singularizer breaks words, and exceptions would be a curated list."""
    assert normalize(plural) == plural.casefold()
    assert normalize(plural) != normalize(singular)


def test_normalization_is_idempotent_and_pure():
    forms = ["The Coca-Cola Company", "Beyonc\xe9", "  (singer\u2013songwriter) ", "AC/DC"]
    once = [normalize(form) for form in forms]
    assert [normalize(form) for form in once] == once
    assert [normalize(form) for form in forms] == once


def test_the_module_source_is_ascii():
    """The project rule for Python strings: every non-ASCII character is written as an escape."""
    source = normalization.__file__
    with open(source, encoding="utf-8") as handle:
        assert handle.read().isascii()
