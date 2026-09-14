"""T3: a seed's hit list turned into the mass the walk starts from.

Decision D1 makes this the restart target as well as the starting point, so two
properties of it carry the whole attribution argument of the phase: it must rank
the units exactly as the seed did, and it must not quietly repair a seed that
returned something the walk cannot use.
"""

import numpy as np
import pytest

from concept_embeddings_rag.retrieval.diffusion import DiffusionError, seed_mass

UNIT_IDS = ["u0", "u1", "u2", "u3"]
INDEX = {unit_id: row for row, unit_id in enumerate(UNIT_IDS)}


def test_the_mass_lands_on_the_units_the_seed_returned_and_nowhere_else():
    mass, clipped = seed_mass([("u2", 0.8), ("u0", 0.2)], INDEX)

    assert clipped == 0
    assert mass[INDEX["u2"]] > 0.0
    assert mass[INDEX["u0"]] > 0.0
    assert mass[INDEX["u1"]] == 0.0
    assert mass[INDEX["u3"]] == 0.0


def test_the_mass_is_a_distribution():
    mass, _ = seed_mass([("u0", 0.9), ("u1", 0.6), ("u2", 0.3)], INDEX)

    assert mass.sum() == pytest.approx(1.0)
    assert (mass >= 0.0).all()
    assert mass.shape == (len(UNIT_IDS),)


def test_the_mass_ranks_the_units_exactly_as_the_seed_did():
    """The property decision D1 exists for: L1 normalization is a positive scalar."""
    hits = [("u3", 0.91), ("u1", 0.90), ("u0", 0.42), ("u2", 0.41)]

    mass, _ = seed_mass(hits, INDEX)

    by_mass = sorted(UNIT_IDS, key=lambda unit_id: (-mass[INDEX[unit_id]], unit_id))
    assert by_mass == [unit_id for unit_id, _score in hits]


def test_a_negative_score_is_clipped_and_the_count_reaches_the_caller():
    """A dense cosine can be negative. Declared rather than assumed, and counted."""
    mass, clipped = seed_mass([("u0", 0.8), ("u1", -0.1), ("u2", -0.4)], INDEX)

    assert clipped == 2
    assert mass[INDEX["u1"]] == 0.0
    assert mass[INDEX["u2"]] == 0.0
    assert mass.sum() == pytest.approx(1.0)


def test_an_empty_seed_is_refused_rather_than_spread_uniformly():
    with pytest.raises(DiffusionError, match="returned no hits"):
        seed_mass([], INDEX)


def test_a_seed_whose_every_score_is_zero_is_refused():
    """Nothing to start from, and a uniform prior would be a different system."""
    with pytest.raises(DiffusionError, match="no positive score"):
        seed_mass([("u0", 0.0), ("u1", -0.5)], INDEX)


def test_a_hit_outside_the_pool_is_named_rather_than_dropped():
    with pytest.raises(DiffusionError, match="u9"):
        seed_mass([("u0", 0.5), ("u9", 0.4)], INDEX)


def test_a_repeated_unit_is_refused():
    """One unit is one row of the mass vector; adding a score twice invents mass."""
    with pytest.raises(DiffusionError, match="twice"):
        seed_mass([("u0", 0.5), ("u0", 0.4)], INDEX)


def test_the_mass_is_deterministic_to_the_bit():
    hits = [("u0", 0.9), ("u1", 0.6), ("u2", 0.3)]

    first, _ = seed_mass(hits, INDEX)
    second, _ = seed_mass(hits, INDEX)

    assert np.array_equal(first, second)
