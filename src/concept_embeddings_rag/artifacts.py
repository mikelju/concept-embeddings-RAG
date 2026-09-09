"""Writing an artifact without ever leaving a half-written one on disk.

`Path.write_text` and `np.savez_compressed` both truncate the target before they
start writing it. An interrupted run - Ctrl-C, a full disk, a crash mid-serialize -
therefore leaves a zero-length or truncated file where a valid artifact used to be,
and the next run reads it back as corruption rather than as absence. Two concurrent
runs of the same stage can interleave into the same file for the same reason.

Writing to a unique temporary neighbour and renaming it into place closes both: the
rename is atomic on every filesystem this project runs on, so a reader sees either
the previous artifact or the new one, never a mixture. It costs one extra file that
lives for milliseconds.
"""

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def _temporary_neighbour(path: Path, suffix: str = ".tmp") -> Path:
    """A unique sibling of `path`, so two concurrent writers never share a temporary."""
    return path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}{suffix}")


def write_text_atomic(path: Path | str, text: str, encoding: str = "utf-8") -> Path:
    """Write `text` to `path` as a single visible step."""
    path = Path(path)
    temporary = _temporary_neighbour(path)
    try:
        temporary.write_text(text, encoding=encoding)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def savez_compressed_atomic(path: Path | str, **arrays: np.ndarray) -> Path:
    """`np.savez_compressed` into `path`, visible only once it is complete.

    The temporary already ends in `.npz` because numpy appends that extension to
    any filename lacking it, which would leave the real temporary unrenamed.
    """
    path = Path(path)
    temporary = _temporary_neighbour(path, suffix=".tmp.npz")
    try:
        np.savez_compressed(temporary, allow_pickle=False, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def digest_of(*parts: Any) -> str:
    """A sha256 over an ordered list of arrays, strings and numbers.

    The digest of an artifact has to be computed from the exact bytes that get
    written, not from the objects they came from: an array is hashed after the
    `astype` its writer applies, so the value recorded in the sidecar is the one a
    reader recomputes from the file it just read.
    """
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            hasher.update(np.ascontiguousarray(part).tobytes())
        elif isinstance(part, str):
            hasher.update(part.encode("utf-8"))
        else:
            hasher.update(repr(part).encode("utf-8"))
        # A separator, so two adjacent parts cannot be re-split differently.
        hasher.update(b"\x00")
    return hasher.hexdigest()
