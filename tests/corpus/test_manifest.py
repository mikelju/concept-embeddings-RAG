"""T2: the corpus manifest is the only door to the data.

If a manifest can be loaded while missing a field, some later result will not be
reproducible and nobody will notice until it matters.
"""

import json

import pytest

from concept_embeddings_rag.corpus.manifest import CorpusManifest, ManifestError


def a_manifest(**overrides) -> CorpusManifest:
    fields = {
        "dataset": "hotpotqa-distractor-dev",
        "source_url": "http://example.invalid/hotpot.json",
        "sha256": "a" * 64,
        "downloaded_at": "2026-09-07T10:00:00",
        "seed": 42,
        "n_questions": 2000,
        "split_sizes": {"dev": 600, "test": 1400},
    }
    fields.update(overrides)
    return CorpusManifest(**fields)


def test_round_trip_preserves_every_field(tmp_path):
    manifest = a_manifest(n_units=18000, unit_set_hash="b" * 16)
    path = tmp_path / "manifest.json"
    manifest.save(path)
    assert CorpusManifest.load(path) == manifest


def test_loading_a_manifest_with_a_missing_field_raises(tmp_path):
    payload = json.loads(a_manifest().to_json())
    del payload["sha256"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        CorpusManifest.load(path)


def test_split_sizes_must_add_up_to_n_questions():
    with pytest.raises(ManifestError):
        a_manifest(split_sizes={"dev": 600, "test": 1000})


def test_manifest_records_the_tool_version():
    assert a_manifest().tool_version


def test_a_manifest_whose_sha256_is_not_a_digest_is_refused():
    """SEC-008: the field is about to be compared against a real hash, so a value
    that cannot be one makes the comparison meaningless."""
    with pytest.raises(ManifestError):
        CorpusManifest(
            dataset="d",
            source_url="u",
            sha256="not-a-hash",
            downloaded_at="2026-09-08T00:00:00",
            seed=42,
            n_questions=2,
            split_sizes={"dev": 1, "test": 1},
        )


def test_a_manifest_with_a_non_integer_seed_is_refused():
    """SEC-008: dataclass annotations are not runtime validation, and this one
    arrives from JSON on disk."""
    with pytest.raises(ManifestError):
        CorpusManifest(
            dataset="d",
            source_url="u",
            sha256="a" * 64,
            downloaded_at="2026-09-08T00:00:00",
            seed="forty-two",
            n_questions=2,
            split_sizes={"dev": 1, "test": 1},
        )
