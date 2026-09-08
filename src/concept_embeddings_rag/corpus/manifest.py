"""The corpus manifest: the only door to the frozen data.

Nothing downstream reads the raw benchmark file directly. Everything goes through
a manifest that records where the data came from, its hash, and the seed that
selected the subset, so that any number can be traced back to its inputs.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from concept_embeddings_rag import __version__


class ManifestError(Exception):
    """The manifest is missing fields or is internally inconsistent."""


@dataclass(frozen=True)
class CorpusManifest:
    dataset: str
    source_url: str
    sha256: str
    downloaded_at: str
    seed: int
    n_questions: int
    split_sizes: dict[str, int]
    n_units: int | None = None
    unit_set_hash: str | None = None
    tool_version: str = field(default=__version__)

    def __post_init__(self) -> None:
        total = sum(self.split_sizes.values())
        if total != self.n_questions:
            raise ManifestError(
                f"split sizes add up to {total} but n_questions is {self.n_questions}"
            )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: dict) -> "CorpusManifest":
        # Optional fields carry an explicit default; required ones do not.
        required = {
            f.name
            for f in fields(cls)
            if f.name not in {"n_units", "unit_set_hash", "tool_version"}
        }
        missing = required - payload.keys()
        if missing:
            raise ManifestError(f"manifest is missing required fields: {sorted(missing)}")
        known = {f.name for f in fields(cls)}
        unknown = payload.keys() - known
        if unknown:
            raise ManifestError(f"manifest has unknown fields: {sorted(unknown)}")
        return cls(**payload)

    @classmethod
    def load(cls, path: Path) -> "CorpusManifest":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest at {path} is not valid JSON") from exc
        return cls.from_dict(payload)
