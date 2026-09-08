"""T2: download with hash verification, and no network in the test suite.

The fetcher is injected, so these tests never touch the network. A corpus whose
hash does not match is an error: every number in the project rests on this file.
"""

import pytest

from concept_embeddings_rag.corpus.download import (
    CorpusIntegrityError,
    ensure_corpus,
    sha256_of_bytes,
    sha256_of_file,
)

PAYLOAD = b'[{"_id": "q1"}]'
PAYLOAD_SHA = sha256_of_bytes(PAYLOAD)


def test_downloads_when_the_file_is_absent(tmp_path):
    calls = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        return PAYLOAD

    destination = tmp_path / "hotpot.json"
    path = ensure_corpus("http://example.invalid/x.json", destination, fetcher=fetcher)

    assert path.read_bytes() == PAYLOAD
    assert len(calls) == 1


def test_second_run_does_not_download_again(tmp_path):
    calls = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        return PAYLOAD

    destination = tmp_path / "hotpot.json"
    ensure_corpus("http://example.invalid/x.json", destination, fetcher=fetcher)
    ensure_corpus(
        "http://example.invalid/x.json",
        destination,
        expected_sha256=PAYLOAD_SHA,
        fetcher=fetcher,
    )

    assert len(calls) == 1


def test_a_corrupted_file_raises_instead_of_being_used(tmp_path):
    destination = tmp_path / "hotpot.json"
    destination.write_bytes(b"truncated garbage")

    with pytest.raises(CorpusIntegrityError):
        ensure_corpus(
            "http://example.invalid/x.json",
            destination,
            expected_sha256=PAYLOAD_SHA,
            fetcher=lambda url: PAYLOAD,
            redownload_on_mismatch=False,
        )


def test_hash_of_file_matches_hash_of_bytes(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(PAYLOAD)
    assert sha256_of_file(path) == PAYLOAD_SHA


def test_a_mismatch_now_fails_instead_of_silently_redownloading(tmp_path):
    """SEC-001 / DEF-004: overwriting a file that failed verification destroys the
    only evidence of what went wrong. Failing closed is the default."""
    destination = tmp_path / "hotpot.json"
    destination.write_bytes(b"truncated garbage")

    with pytest.raises(CorpusIntegrityError):
        ensure_corpus(
            "http://example.invalid/x.json",
            destination,
            expected_sha256=PAYLOAD_SHA,
            fetcher=lambda url: PAYLOAD,
        )
    assert destination.read_bytes() == b"truncated garbage"


def test_a_download_past_the_byte_ceiling_is_refused(tmp_path):
    """SEC-004: a timeout bounds how long a response may take, not how large it is."""
    from concept_embeddings_rag.corpus import download

    class _Response:
        def raise_for_status(self):
            return None

        def iter_bytes(self):
            for _ in range(4):
                yield b"x" * 1024

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Client:
        def stream(self, method, url):
            return _Response()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import sys
    import types

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = lambda **kwargs: _Client()
    sys.modules["httpx"] = fake_httpx
    try:
        with pytest.raises(CorpusIntegrityError):
            download._http_fetcher("http://example.invalid/big.json", max_bytes=2048)
    finally:
        del sys.modules["httpx"]
