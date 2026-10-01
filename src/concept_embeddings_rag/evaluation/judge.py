"""Phase 17 (D2): a pinned zero-shot cross-encoder, its identity, its truncations and its
scores.

The judge is `sentence_transformers.CrossEncoder` at the pinned revision, its raw logit as
the score (`activation_fn = Identity`: the sigmoid is monotone but saturates in float32 and
would manufacture ties), float32 weights from `model.safetensors` only, matmuls at
`highest` precision, no `trust_remote_code`, the maximum length left to the model. Before
any pair is scored, the snapshot the Hub served must carry the pinned revision as its
directory name and the pinned SHA-256 as its weights' digest.

`torch` and `sentence-transformers` are imported inside `load` only, so the scoring and
truncation code runs against a stub in tests without the real model.
"""

import importlib.metadata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.embeddings.backend import snapshot_directory, weights_sha256

# What the snapshot check fetches: configuration, tokenizer files and the safetensors
# weights; never the ONNX, OpenVINO, Flax or PyTorch-pickle copies a repository also holds.
SNAPSHOT_PATTERNS: tuple[str, ...] = ("*.json", config.PHASE_17_WEIGHTS_FILE, "vocab.txt")
LIBRARIES: tuple[str, ...] = (
    "torch",
    "transformers",
    "sentence-transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
)


class JudgeError(Exception):
    """The judge is not the pinned one, or it returned a score that cannot be used."""


@dataclass
class LoadedJudge:
    key: str
    pin: Mapping[str, Any]
    model: Any
    snapshot: Path
    served_revision: str
    weights_sha256: str


def check_snapshot(pin: Mapping[str, Any], snapshot: Path) -> dict[str, str]:
    """The served snapshot's revision (its directory name) and weights SHA-256, refused
    unless both are the pin."""
    served = Path(snapshot).name
    if served != pin["revision"]:
        raise JudgeError(
            f"{pin['name']}: the Hub served revision {served}, not the pinned {pin['revision']}"
        )
    digest = weights_sha256(snapshot, config.PHASE_17_WEIGHTS_FILE)
    if digest != pin["weights_sha256"]:
        raise JudgeError(
            f"{pin['name']}: {config.PHASE_17_WEIGHTS_FILE} has SHA-256 {digest}, not the "
            f"pinned {pin['weights_sha256']}"
        )
    return {"served_revision": served, "weights_sha256": digest}


def load(
    key: str,
    *,
    device: str | None = None,
    resolve_snapshot: Callable[..., Path] = snapshot_directory,
) -> LoadedJudge:
    """The pinned judge `key` (`light` or `strong`), checked before any pair is scored."""
    pin = config.PHASE_17_JUDGES[key]
    snapshot = resolve_snapshot(pin["name"], pin["revision"], allow_patterns=SNAPSHOT_PATTERNS)
    served = check_snapshot(pin, Path(snapshot))

    import torch
    from sentence_transformers import CrossEncoder

    torch.set_float32_matmul_precision(config.PHASE_17_MATMUL_PRECISION)
    print(f"[INFO] loading judge {pin['name']} (revision {pin['revision']})", flush=True)
    # No `trust_remote_code`: both checkpoints are stock `*ForSequenceClassification`.
    model = CrossEncoder(
        pin["name"],
        revision=pin["revision"],
        device=device,
        activation_fn=torch.nn.Identity(),
        model_kwargs={"dtype": torch.float32, "use_safetensors": True},
    )
    max_length = int(model.max_length)
    if max_length != int(pin["max_length"]):
        raise JudgeError(
            f"{pin['name']} resolved a maximum length of {max_length}, not {pin['max_length']}"
        )
    return LoadedJudge(
        key=key,
        pin=pin,
        model=model,
        snapshot=Path(snapshot),
        served_revision=served["served_revision"],
        weights_sha256=served["weights_sha256"],
    )


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def identity(loaded: LoadedJudge) -> dict[str, Any]:
    """What the scoring manifest records of the judge (D2)."""
    import torch

    model = loaded.model
    parameters = sum(int(p.numel()) for p in model.model.parameters())
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in model.model.parameters()})
    return {
        "key": loaded.key,
        "label": loaded.pin["label"],
        "name": loaded.pin["name"],
        "revision_requested": loaded.pin["revision"],
        "revision_served": loaded.served_revision,
        "weights_file": config.PHASE_17_WEIGHTS_FILE,
        "weights_sha256": loaded.weights_sha256,
        "architecture": type(model.model).__name__,
        "parameters": parameters,
        "max_length": int(model.max_length),
        "dtype": dtypes,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "activation": type(model.activation_fn).__name__,
        "batch_size": int(loaded.pin["batch_size"]),
        "trust_remote_code": False,
        "library_versions": {name: _version(name) for name in LIBRARIES},
    }


def truncations(
    tokenizer: Any, question: str, texts: Sequence[str], max_length: int
) -> dict[str, int]:
    """Each (question, text) pair tokenized untruncated by the judge's own tokenizer, special
    tokens included; a pair longer than `max_length` counts as truncated."""
    if not texts:
        return {"pairs": 0, "truncated": 0, "longest": 0}
    encoded = tokenizer(
        [question] * len(texts), list(texts), truncation=False, add_special_tokens=True
    )
    lengths = [len(ids) for ids in encoded["input_ids"]]
    return {
        "pairs": len(lengths),
        "truncated": sum(1 for n in lengths if n > max_length),
        "longest": max(lengths),
    }


def score(model: Any, pairs: Sequence[tuple[str, str]], batch_size: int) -> np.ndarray:
    """The raw logits of `pairs`, float32, aligned with `pairs`; a non-finite one refuses."""
    if not pairs:
        return np.zeros(0, dtype=np.float32)
    raw = model.predict(
        list(pairs), batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    )
    scores = np.asarray(raw, dtype=np.float32).reshape(-1)
    if scores.shape[0] != len(pairs):
        raise JudgeError(f"{scores.shape[0]} scores for {len(pairs)} pairs")
    bad = np.flatnonzero(~np.isfinite(scores))
    if bad.size:
        raise JudgeError(f"{bad.size} non-finite scores, the first at pair {int(bad[0])}")
    return scores


def score_rows(model: Any, rows: Sequence[Mapping[str, Any]], batch_size: int) -> list[np.ndarray]:
    """One `predict` call over every pair of `rows` (a shard), split back per question in
    the rows' order and, inside each, in the order of its texts."""
    pairs: list[tuple[str, str]] = []
    sizes: list[int] = []
    for row in rows:
        texts = list(row["texts"])
        if len(texts) != len(row["unit_ids"]):
            raise JudgeError(f"{row['qid']}: {len(texts)} texts for {len(row['unit_ids'])} units")
        pairs.extend((str(row["question"]), str(text)) for text in texts)
        sizes.append(len(texts))
    scores = score(model, pairs, batch_size)
    bounds = np.cumsum([0, *sizes])
    return [scores[int(a) : int(b)] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
