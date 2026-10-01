"""Build the Phase 17 paper: regenerate the tables, run Tectonic, report the PDF.

    uv run python docs/paper/build.py

Steps, each named when it fails:
  1. `uv run cer p17-tables`  (tables, macros, figures, R5 guard);
  2. Tectonic on `docs/paper/main.tex` (path from the TECTONIC environment variable);
  3. the page count and the SHA-256 of `docs/paper/main.pdf`.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import zlib
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent
DEFAULT_TECTONIC = Path.home() / ".local" / "bin" / "tectonic-0.17.0" / "tectonic.exe"


class BuildError(RuntimeError):
    """A build step failed; the message names the step."""


def run(step: str, command: list[str], cwd: Path) -> None:
    print(f"[..] {step}: {' '.join(command)}")
    try:
        result = subprocess.run(command, cwd=cwd, check=False)  # noqa: S603
    except FileNotFoundError as error:
        raise BuildError(f"{step}: cannot start {command[0]!r} ({error})") from error
    if result.returncode != 0:
        raise BuildError(f"{step} failed with exit code {result.returncode}")
    print(f"[OK] {step}")


def tectonic_path() -> Path:
    return Path(os.environ.get("TECTONIC", str(DEFAULT_TECTONIC)))


def page_count(pdf: Path) -> int:
    """Pages of a PDF: pypdf when installed, else a count of `/Type /Page` objects."""
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found,unused-ignore]

        return len(PdfReader(str(pdf)).pages)
    except ImportError:
        pass
    data = pdf.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page(?![A-Za-z])", data))
    if pages:
        return pages
    # PDF 1.5 writers keep page objects inside compressed object streams: inflate every stream.
    found = 0
    for match in re.finditer(rb"stream\r?\n", data):
        end = data.find(b"endstream", match.end())
        try:
            inflated = zlib.decompress(data[match.end() : end])
        except zlib.error:
            continue
        found += len(re.findall(rb"/Type\s*/Page(?![A-Za-z])", inflated))
    return found


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> Path:
    run("tables", ["uv", "run", "cer", "p17-tables"], REPO_ROOT)
    engine = tectonic_path()
    if not engine.exists():
        raise BuildError(f"tectonic: {engine} does not exist; set the TECTONIC variable")
    run("tectonic", [str(engine), "main.tex"], PAPER_DIR)
    pdf = PAPER_DIR / "main.pdf"
    if not pdf.exists():
        raise BuildError("tectonic: main.pdf was not written")
    return pdf


def main() -> int:
    try:
        pdf = build()
    except BuildError as error:
        print(f"[FAIL] {error}")
        return 1
    print(f"[OK] pages: {page_count(pdf)}")
    print(f"[OK] sha256: {sha256_of(pdf)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
