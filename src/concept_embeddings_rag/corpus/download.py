"""Download the benchmark once and verify it by hash.

The downloaded JSON is external input, so it is verified before anything trusts
it. The fetcher is injected, which keeps the test suite off the network.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

Fetcher = Callable[[str], bytes]

# A timeout bounds how long a download may take, not how large it may be.
# The frozen HotpotQA split is ~50 MB; this ceiling is far above it and finite.
MAX_CORPUS_BYTES = 512 * 1024 * 1024


class CorpusIntegrityError(Exception):
    """The file on disk does not match the expected hash."""


def sha256_of_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http_fetcher(url: str, max_bytes: int = MAX_CORPUS_BYTES) -> bytes:
    import httpx

    with (
        httpx.Client(follow_redirects=True, timeout=300.0) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > max_bytes:
                raise CorpusIntegrityError(
                    f"{url} exceeds the {max_bytes} byte ceiling; refusing to buffer it"
                )
            chunks.append(chunk)
        return b"".join(chunks)


def ensure_corpus(
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
    fetcher: Fetcher | None = None,
    redownload_on_mismatch: bool = False,
) -> Path:
    """Return a local, verified copy of the corpus, downloading it only if needed.

    With no expected hash, an existing file is accepted as-is: the first download
    is precisely what establishes the hash. Once a hash is known, a mismatch is an
    error, not something to work around.

    Failing is the default rather than re-downloading: silently replacing a file that
    failed verification destroys the only evidence of what went wrong, and for a frozen
    research corpus that evidence is worth more than the convenience.
    """
    fetcher = fetcher or _http_fetcher

    if destination.exists():
        actual = sha256_of_file(destination)
        if expected_sha256 is None or actual == expected_sha256:
            return destination
        if not redownload_on_mismatch:
            raise CorpusIntegrityError(
                f"{destination} has hash {actual}, expected {expected_sha256}"
            )
        print(f"[WARN] {destination.name} failed hash check, downloading again")

    payload = fetcher(url)
    actual = sha256_of_bytes(payload)
    if expected_sha256 is not None and actual != expected_sha256:
        raise CorpusIntegrityError(f"downloaded file has hash {actual}, expected {expected_sha256}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination
