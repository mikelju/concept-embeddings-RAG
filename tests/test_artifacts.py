"""OBS-006: an artifact is written whole or not at all, and hashed over its bytes.

`Path.write_text` and `np.savez_compressed` truncate the target before they start
writing it, so an interrupted run leaves a zero-length or truncated file where a
valid artifact used to be. The next run then reads corruption instead of absence -
and in this project absence is recoverable (recompute) while corruption is a
verifier failing on numbers already published.

The digest helper is tested here too, because every "verify on load" check added by
the audit is only as good as the guarantee that both sides hash the same bytes.
"""

import numpy as np
import pytest

from concept_embeddings_rag import artifacts
from concept_embeddings_rag.artifacts import (
    _temporary_neighbour,
    digest_of,
    savez_compressed_atomic,
    write_text_atomic,
)


def test_a_text_artifact_round_trips_and_leaves_nothing_beside_it(tmp_path):
    path = write_text_atomic(tmp_path / "sidecar.json", '{"k": 512}')

    assert path.read_text(encoding="utf-8") == '{"k": 512}'
    assert [p.name for p in tmp_path.iterdir()] == ["sidecar.json"]


def test_an_npz_artifact_is_renamed_into_place_rather_than_left_as_a_temporary(tmp_path):
    """numpy appends `.npz` to any name lacking it, which would strand the temporary."""
    path = savez_compressed_atomic(tmp_path / "matrix.npz", data=np.arange(4, dtype=np.float32))

    with np.load(path, allow_pickle=False) as payload:
        assert np.array_equal(payload["data"], np.arange(4, dtype=np.float32))
    assert [p.name for p in tmp_path.iterdir()] == ["matrix.npz"]


def test_a_text_write_that_fails_leaves_the_previous_artifact_intact(tmp_path, monkeypatch):
    path = tmp_path / "sidecar.json"
    write_text_atomic(path, '{"k": 512}')

    def fail(source, target):
        raise OSError("no space left on device")

    monkeypatch.setattr(artifacts.os, "replace", fail)

    with pytest.raises(OSError):
        write_text_atomic(path, '{"k": 1024}')

    assert path.read_text(encoding="utf-8") == '{"k": 512}'
    assert [p.name for p in tmp_path.iterdir()] == ["sidecar.json"]


def test_a_write_that_dies_mid_serialize_leaves_the_previous_artifact_intact(tmp_path, monkeypatch):
    """The failure this is about: a crash after the target was already truncated."""
    path = tmp_path / "matrix.npz"
    savez_compressed_atomic(path, data=np.arange(4, dtype=np.float32))
    before = path.read_bytes()

    def die_halfway(target, **arrays):
        with open(target, "wb") as handle:
            handle.write(b"PK\x03\x04 truncated")
        raise OSError("no space left on device")

    monkeypatch.setattr(artifacts.np, "savez_compressed", die_halfway)

    with pytest.raises(OSError):
        savez_compressed_atomic(path, data=np.arange(8, dtype=np.float32))

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["matrix.npz"]


def test_two_writers_of_the_same_artifact_do_not_share_a_temporary(tmp_path):
    """Two runs of the same stage interleaving into one temporary is the same bug."""
    path = tmp_path / "matrix.npz"

    names = {_temporary_neighbour(path).name for _ in range(100)}

    assert len(names) == 100


def test_the_digest_changes_with_the_content_it_covers():
    assert digest_of(np.arange(4)) != digest_of(np.arange(5))
    assert digest_of(np.arange(4)) == digest_of(np.arange(4))


def test_two_parts_cannot_be_re_split_into_the_same_digest():
    """Without a separator, ("ab", "c") and ("a", "bc") would hash identically -
    which would let a rearranged artifact keep the digest of the original.
    """
    assert digest_of("ab", "c") != digest_of("a", "bc")


def test_the_digest_is_taken_over_bytes_not_over_the_object_they_came_from():
    """A digest recorded from float64 and recomputed from float32 would never match."""
    assert digest_of(np.arange(4, dtype=np.float32)) != digest_of(np.arange(4, dtype=np.float64))
