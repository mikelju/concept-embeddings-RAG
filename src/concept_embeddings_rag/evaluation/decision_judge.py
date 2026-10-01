"""Phase 17 (deviation 17.1): J-decision, CLM-8B's two projection heads over a frozen Qwen3-8B
that vLLM serves, embed-once per set, the pre-softmax logit as the score.

A bi-encoder: the question (the "state") and each unit's `indexable_text` are embedded apart
by the encoder, projected by the state head and the action head to 512-d L2-normalised
vectors, and the score is `scale * dot(z_unit, z_state)`, the quantity `Engine.answer` forms
before its softmax. This module uses neither `Engine`, `Engine.rank` nor `clm-serve`; it calls
the library's own parts, `clm.heads.make_head` and `clm.embedder.Embedder`, imported inside
`load` only (the package needs vLLM and does not install on the ARM64 laptop), so the scoring,
alignment and truncation code runs against stubs in tests.

Before the first request: the heads file's SHA-256 is checked against the pin, and only then
`torch.load(..., weights_only=True)` reads it; every encoder shard's SHA-256 is checked
against the pin in the local Hub snapshot vLLM served from. Truncation is counted with
Qwen3-8B's own tokenizer at the pinned revision (tokenizer files only; no model).
"""

import contextlib
import gzip
import importlib.metadata
import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.embeddings.backend import snapshot_directory, weights_sha256
from concept_embeddings_rag.evaluation.judge import JudgeError
from concept_embeddings_rag.evaluation.phase17 import sha256_file

DEFAULT_EMB_URL = "http://127.0.0.1:8090/v1/embeddings"
DEFAULT_EMB_MODEL = "qwen3-8b"
LIBRARIES: tuple[str, ...] = (
    "vllm",
    "contrastive-lm",
    "torch",
    "transformers",
    "huggingface-hub",
    "safetensors",
)
# What the encoder check fetches from the Hub when the snapshot is not already local: the
# shards and the small files, never the pickle or other copies a repository might hold.
ENCODER_PATTERNS: tuple[str, ...] = ("*.json", "model-*.safetensors")
SCALE_CEILING = 100.0
# Texts embedded, projected and discarded per call: bounds the 4,096-d buffer (the heads' input).
EMBED_CHUNK = 512
TOKENIZE_BATCH = 2000


def truncation_name(set_name: str) -> str:
    return f"truncation-decision-{set_name}.json"


def pin() -> Mapping[str, Any]:
    return config.PHASE_17_JUDGES[config.PHASE_17_DECISION]


def library_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


# --- Heads -----------------------------------------------------------------------------------


@dataclass
class LoadedHeads:
    state_head: Any
    action_head: Any
    scale: float
    cfg: dict[str, Any]
    parameters: int
    path: Path
    sha256: str
    bytes: int
    device: str


def fetch_heads(judge_pin: Mapping[str, Any]) -> Path:
    """The heads file at the pinned revision (not `clm-download`, which fetches `main`)."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=judge_pin["name"],
            filename=judge_pin["weights_file"],
            revision=judge_pin["revision"],
        )
    )


def check_heads_file(judge_pin: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """The file's size and SHA-256, refused unless both are the pin. Nothing is unpickled."""
    path = Path(path)
    size = path.stat().st_size
    if size != int(judge_pin["weights_bytes"]):
        raise JudgeError(
            f"{path.name} is {size} bytes, not the pinned {judge_pin['weights_bytes']}"
        )
    digest = sha256_file(path)
    if digest != judge_pin["weights_sha256"]:
        raise JudgeError(
            f"{path.name} has SHA-256 {digest}, not the pinned {judge_pin['weights_sha256']}"
        )
    return {"file": path.name, "sha256": digest, "bytes": size}


def _make_head_function() -> Callable[..., Any]:
    from clm.heads import make_head

    return make_head  # type: ignore[no-any-return]


def load_heads(
    judge_pin: Mapping[str, Any],
    *,
    device: str = "cpu",
    path: Path | None = None,
    download: Callable[[Mapping[str, Any]], Path] = fetch_heads,
    make_head: Callable[..., Any] | None = None,
) -> LoadedHeads:
    """The two heads of the pinned checkpoint: float32, `eval()`, built from the checkpoint's
    `cfg` as `HeadPair._load` does; scale `exp(logit_scale)` clamped at 100."""
    fetched = path is None
    if path is None:
        path = download(judge_pin)
    path = Path(path)
    if fetched and path.parent.name != judge_pin["revision"]:
        raise JudgeError(
            f"{judge_pin['name']}: the Hub served revision {path.parent.name}, not the pinned "
            f"{judge_pin['revision']}"
        )
    checked = check_heads_file(judge_pin, path)  # before torch.load reads a byte
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = dict(checkpoint["cfg"])
    build = make_head if make_head is not None else _make_head_function()
    kwargs: dict[str, Any] = {
        "width": cfg["width"],
        "depth": cfg["depth"],
        "proj": checkpoint.get("projection_dim", cfg.get("projection_dim", 512)),
        "activation": cfg.get("activation", "gelu"),
        "layernorm": cfg.get("layernorm", False),
        "residual": cfg.get("residual", False),
        "hidden": cfg.get("hidden_size", 4096),
    }
    state_head, action_head = build(**kwargs), build(**kwargs)
    state_head.load_state_dict(checkpoint["state_head"])
    action_head.load_state_dict(checkpoint["action_head"])
    for head in (state_head, action_head):
        head.float().eval().to(device)
    scale = float(torch.as_tensor(checkpoint["logit_scale"]).float().exp().clamp(max=SCALE_CEILING))
    parameters = sum(int(p.numel()) for h in (state_head, action_head) for p in h.parameters())
    return LoadedHeads(
        state_head=state_head,
        action_head=action_head,
        scale=scale,
        cfg=cfg,
        parameters=parameters,
        path=path,
        sha256=checked["sha256"],
        bytes=checked["bytes"],
        device=device,
    )


# --- Encoder ---------------------------------------------------------------------------------


def encoder_snapshot(
    judge_pin: Mapping[str, Any],
    encoder_dir: Path | None = None,
    *,
    resolve_snapshot: Callable[..., Path] = snapshot_directory,
) -> Path:
    """The local snapshot directory of the pinned encoder: `encoder_dir` as given, else the
    Hub cache's snapshot at the pinned revision (a cache hit when vLLM served from it)."""
    encoder = judge_pin["encoder"]
    if encoder_dir is not None:
        return Path(encoder_dir)
    return Path(
        resolve_snapshot(encoder["name"], encoder["revision"], allow_patterns=ENCODER_PATTERNS)
    )


def check_encoder(judge_pin: Mapping[str, Any], snapshot: Path) -> dict[str, Any]:
    """Every pinned shard present with its pinned size and SHA-256 (and, when the directory is
    a Hub snapshot, its name the pinned revision), before the first request."""
    encoder = judge_pin["encoder"]
    snapshot = Path(snapshot)
    if snapshot.parent.name == "snapshots" and snapshot.name != encoder["revision"]:
        raise JudgeError(
            f"{encoder['name']}: the snapshot is revision {snapshot.name}, not the pinned "
            f"{encoder['revision']}"
        )
    verified: dict[str, dict[str, Any]] = {}
    for filename, shard in encoder["shards"].items():
        file = snapshot / filename
        if not file.is_file():
            raise JudgeError(f"{encoder['name']}: {file} does not exist")
        if file.stat().st_size != int(shard["bytes"]):
            raise JudgeError(f"{filename} is {file.stat().st_size} bytes, not {shard['bytes']}")
        digest = weights_sha256(snapshot, filename)
        if digest != shard["sha256"]:
            raise JudgeError(
                f"{encoder['name']}: {filename} has SHA-256 {digest}, not the pinned "
                f"{shard['sha256']}"
            )
        verified[filename] = {"sha256": digest, "bytes": int(shard["bytes"])}
    return {
        "name": encoder["name"],
        "revision": encoder["revision"],
        "snapshot_directory": snapshot.name,
        "shards_verified": verified,
        "total_bytes": sum(v["bytes"] for v in verified.values()),
    }


# --- Truncation ------------------------------------------------------------------------------


def load_tokenizer(judge_pin: Mapping[str, Any]) -> Any:
    """Qwen3-8B's tokenizer at the pinned revision: tokenizer files only, no model."""
    from transformers import AutoTokenizer

    encoder = judge_pin["encoder"]
    return AutoTokenizer.from_pretrained(encoder["name"], revision=encoder["revision"])


def token_lengths(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    """Each text tokenized untruncated, special tokens included."""
    if not texts:
        return []
    encoded = tokenizer(list(texts), add_special_tokens=True, truncation=False)
    return [len(ids) for ids in encoded["input_ids"]]


@dataclass
class TokenScan:
    units: dict[str, int]  # unit id -> untruncated tokens
    states: list[int]  # per question, in the rows' order
    seconds: float


def scan_lengths(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    batch: int = TOKENIZE_BATCH,
    progress: Callable[[int, int], None] | None = None,
) -> TokenScan:
    """The untruncated token count of every distinct unit text and every state (`question`
    stripped) of `rows`, tokenized in batches as the rows stream past; a unit id met with
    two different texts refuses."""
    started = time.perf_counter()
    units: dict[str, int] = {}
    states: list[int] = []
    seen: dict[str, int] = {}  # unit id -> hash of its text, to catch an id with two texts
    pending_ids: list[str] = []
    pending_texts: list[str] = []
    pending_states: list[str] = []
    done = 0

    def flush() -> None:
        nonlocal done
        if pending_texts:
            for unit_id, n in zip(
                pending_ids, token_lengths(tokenizer, pending_texts), strict=True
            ):
                units[unit_id] = n
            done += len(pending_texts)
            if progress is not None:
                progress(done, len(states))
            pending_ids.clear()
            pending_texts.clear()
        if pending_states:
            states.extend(token_lengths(tokenizer, pending_states))
            pending_states.clear()

    for row in rows:
        pending_states.append(str(row["question"]).strip())
        for unit_id, text in zip(row["unit_ids"], row["texts"], strict=True):
            marker = hash(text)
            if unit_id in seen:
                if seen[unit_id] != marker:
                    raise JudgeError(f"unit {unit_id} appears with two different texts")
                continue
            seen[unit_id] = marker
            pending_ids.append(unit_id)
            pending_texts.append(text)
        if len(pending_texts) >= batch or len(pending_states) >= batch:
            flush()
    flush()
    return TokenScan(units=units, states=states, seconds=time.perf_counter() - started)


def _count(lengths: Iterable[int], max_length: int) -> dict[str, Any]:
    values = list(lengths)
    over = sum(1 for n in values if n > max_length)
    return {
        "texts": len(values),
        "truncated": over,
        "share": over / len(values) if values else 0.0,
        "longest": max(values, default=0),
    }


def truncation_summary(scan: TokenScan, max_length: int) -> dict[str, Any]:
    """Per set: how many texts the 2,048-token limit cuts, among the distinct units and the
    states, and together (count, share, longest)."""
    units = _count(scan.units.values(), max_length)
    states = _count(scan.states, max_length)
    texts = units["texts"] + states["texts"]
    count = units["truncated"] + states["truncated"]
    return {
        "max_length": max_length,
        "count": count,
        "share": count / texts if texts else 0.0,
        "longest": max(units["longest"], states["longest"]),
        "units": units,
        "states": states,
    }


def stream_pairs(path: Path) -> Iterator[dict[str, Any]]:
    """A pairs file's rows one at a time (1.3 M texts for HotpotQA never sit whole here)."""
    with gzip.open(path, "rb") as packed:
        for line in packed:
            yield json.loads(line)


# --- Embedding and scoring -------------------------------------------------------------------


def project(head: Any, vectors: np.ndarray, device: str) -> np.ndarray:
    """L2-normalised projections of `[n, hidden]` encoder embeddings, float32 on the host;
    a non-finite value refuses."""
    import torch

    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32)).to(device)
        out = torch.nn.functional.normalize(head(tensor), dim=-1)
    projected = out.cpu().numpy().astype(np.float32, copy=False)
    bad = np.flatnonzero(~np.isfinite(projected).all(axis=1))
    if bad.size:
        raise JudgeError(f"{bad.size} non-finite projections, the first at text {int(bad[0])}")
    return projected


class DecisionJudge:
    """The heads over an embedder, with the pin's identity. `embedder.embed(texts)` returns
    `([n, hidden] L2-normalised, tokens vLLM reported)`, as `clm.embedder.Embedder` does."""

    def __init__(
        self,
        *,
        judge_pin: Mapping[str, Any],
        heads: LoadedHeads,
        embedder: Any,
        encoder: Mapping[str, Any],
        served_revision: str,
        serving: Mapping[str, Any],
    ) -> None:
        self.key = config.PHASE_17_DECISION
        self.pin = judge_pin
        self.heads = heads
        self.embedder = embedder
        self.encoder = dict(encoder)
        self.served_revision = served_revision
        self.serving = dict(serving)

    @property
    def weights_sha256(self) -> str:
        return self.heads.sha256

    @property
    def scale(self) -> float:
        return self.heads.scale

    @contextlib.contextmanager
    def request_size(self, size: int) -> Iterator[None]:
        """Texts per embedding request, for the duration (the determinism re-score sends one)."""
        before = self.embedder.batch
        self.embedder.batch = size
        try:
            yield
        finally:
            self.embedder.batch = before

    def _embed(
        self, texts: Sequence[str], head: Any, progress: Callable[[int], None] | None = None
    ) -> tuple[np.ndarray, int]:
        out: list[np.ndarray] = []
        tokens = 0
        for start in range(0, len(texts), EMBED_CHUNK):
            chunk = list(texts[start : start + EMBED_CHUNK])
            try:
                vectors, reported = self.embedder.embed(chunk)
            except RuntimeError as error:  # clm.embedder.EmbedderError
                raise JudgeError(f"the embedding server failed: {error}") from error
            if vectors.shape[0] != len(chunk):
                raise JudgeError(f"{vectors.shape[0]} embeddings for {len(chunk)} texts")
            out.append(project(head, vectors, self.heads.device))
            tokens += int(reported)
            if progress is not None:
                progress(start + len(chunk))
        width = int(self.heads.cfg.get("projection_dim", 512))
        return (np.concatenate(out) if out else np.zeros((0, width), np.float32)), tokens

    def embed_units(
        self, texts: Sequence[str], progress: Callable[[int], None] | None = None
    ) -> tuple[np.ndarray, int]:
        """Unit texts verbatim -> action-head vectors, aligned with `texts`."""
        return self._embed(texts, self.heads.action_head, progress)

    def embed_states(self, questions: Sequence[str]) -> tuple[np.ndarray, int]:
        """`question.strip()` (no instruction) -> state-head vectors, aligned with `questions`."""
        return self._embed([q.strip() for q in questions], self.heads.state_head)

    def logits(self, state: np.ndarray, units: np.ndarray) -> np.ndarray:
        """`scale * dot(z_unit, z_state)`, float32, no softmax; a non-finite one refuses."""
        scores = (np.float32(self.scale) * (units @ state)).astype(np.float32).reshape(-1)
        bad = np.flatnonzero(~np.isfinite(scores))
        if bad.size:
            raise JudgeError(f"{bad.size} non-finite logits, the first at unit {int(bad[0])}")
        return scores

    def score_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        unit_vectors: np.ndarray,
        position: Mapping[str, int],
    ) -> tuple[list[np.ndarray], int]:
        """Per question of `rows`, the logits of its units in its own order, against the held
        `unit_vectors` (indexed through `position`, unit id -> row); also the state tokens
        vLLM reported."""
        states, tokens = self.embed_states([str(r["question"]) for r in rows])
        out: list[np.ndarray] = []
        for row, state in zip(rows, states, strict=True):
            ids = list(row["unit_ids"])
            if len(row["texts"]) != len(ids):
                raise JudgeError(f"{row['qid']}: {len(row['texts'])} texts for {len(ids)} units")
            index = [position[u] for u in ids]
            out.append(self.logits(state, unit_vectors[index]))
        return out, tokens

    def identity(self) -> dict[str, Any]:
        heads = self.heads
        return {
            "key": self.key,
            "label": self.pin["label"],
            "kind": self.pin["kind"],
            "name": self.pin["name"],
            "revision_requested": self.pin["revision"],
            "revision_served": self.served_revision,
            "weights_file": self.pin["weights_file"],
            "weights_sha256": heads.sha256,
            "weights_bytes": heads.bytes,
            "architecture": "state head and action head over a frozen Qwen3-8B (bi-encoder)",
            "parameters": heads.parameters,
            "heads": {
                "cfg": heads.cfg,
                "dtype": "float32",
                "parameters": heads.parameters,
                "scale": heads.scale,
                "scale_ceiling": SCALE_CEILING,
                "device": heads.device,
            },
            "encoder": {**self.encoder, "dtype": self.pin["dtype"]},
            "max_length": int(self.pin["max_length"]),
            "dtype": {"encoder": self.pin["dtype"], "heads": "float32"},
            "batch_size": int(self.pin["batch_size"]),
            "batch_unit": "texts per embedding request",
            "score": "logit",
            "trust_remote_code": False,
            "library_versions": {name: library_version(name) for name in LIBRARIES},
        }


def _embedder_class() -> Any:
    from clm.embedder import Embedder

    return Embedder


def load(
    *,
    emb_url: str = DEFAULT_EMB_URL,
    emb_model: str = DEFAULT_EMB_MODEL,
    encoder_dir: Path | None = None,
    device: str | None = None,
    serving: Mapping[str, Any] | None = None,
    resolve_snapshot: Callable[..., Path] = snapshot_directory,
    download: Callable[[Mapping[str, Any]], Path] = fetch_heads,
    make_head: Callable[..., Any] | None = None,
    embedder_class: Any = None,
) -> DecisionJudge:
    """The pinned J-decision, checked before any text is embedded: the heads file's digest,
    then the heads; the encoder's shards in the local snapshot."""
    judge_pin = pin()
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch

    torch.set_float32_matmul_precision(config.PHASE_17_MATMUL_PRECISION)
    print(f"[INFO] loading heads {judge_pin['weights_file']} (revision {judge_pin['revision']})")
    heads = load_heads(judge_pin, device=device, download=download, make_head=make_head)
    snapshot = encoder_snapshot(judge_pin, encoder_dir, resolve_snapshot=resolve_snapshot)
    print(f"[INFO] checking the encoder's shards in {snapshot}", flush=True)
    encoder = check_encoder(judge_pin, snapshot)
    klass = embedder_class if embedder_class is not None else _embedder_class()
    embedder = klass(
        emb_url,
        model=emb_model,
        max_tokens=int(judge_pin["max_length"]),
        batch=int(judge_pin["batch_size"]),
        cache_size=0,
    )
    return DecisionJudge(
        judge_pin=judge_pin,
        heads=heads,
        embedder=embedder,
        encoder=encoder,
        served_revision=heads.path.parent.name,
        serving={"url": emb_url, "model": emb_model, **(serving or {})},
    )
