"""Normalization v1: raw extracted strings into node forms (Phase 5, HU-3, decision D6).

A pure function of the string. No model, no embedding, no vocabulary and no list of
exceptions: the spec rules all of them out, because each would be a second experimental
variable hidden inside the representation under test. What it cannot resolve - synonyms
such as two wordings of the same genre - is measured by the fragmentation figures and
declared as a limitation that works against the concept arm.

The steps, in order:

1. Unicode NFKC, so compatibility forms (ligatures, full-width letters) fold to plain ones.
2. Diacritics stripped: NFKD, then every combining mark dropped.
3. Casefold, which is lower-casing that also handles forms `lower()` leaves apart.
4. Typographic quotes and apostrophes to their ASCII forms.
5. Hyphens, dashes, underscores and slashes to spaces.
6. Punctuation stripped from both ends; punctuation inside the form is kept.
7. Whitespace collapsed, then one leading `the`, `a` or `an` dropped, then collapsed again.

Plural folding is deliberately absent: a rule-based singularizer turns `physics`, `series`
and `Paris` into wrong forms, and the prompt already asks for singular wording.

Every non-ASCII character in this file is written as an escape, per the project rule for
Python strings.
"""

import re
import unicodedata

from concept_embeddings_rag import config

VERSION = config.NORMALIZATION_VERSION

_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\xab": '"',
        "\xbb": '"',
    }
)
_SEPARATORS = re.compile("[-_/\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]")
_EDGE_PUNCTUATION = re.compile(r"^[\W_]+|[\W_]+$")
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an) (?=\S)")


def _collapse(text: str) -> str:
    return " ".join(text.split())


def normalize(form: str) -> str:
    """The node form of one extracted string; empty when nothing survives."""
    text = unicodedata.normalize("NFKC", form)
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
    text = text.casefold()
    text = text.translate(_QUOTES)
    text = _SEPARATORS.sub(" ", text)
    text = _EDGE_PUNCTUATION.sub("", text)
    text = _collapse(text)
    text = _LEADING_ARTICLE.sub("", text, count=1)
    return _collapse(text)
