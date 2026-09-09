"""Naming the concepts, for readers only.

An atom is a direction in embedding space. It can be read through the units that
activate it most strongly, and this module asks a model to compress that reading
into a name and one sentence. That is the whole purpose: **nothing in retrieval,
scoring or expansion ever reads a label**. A concept's embedding is its dictionary
atom, so a question maps to concepts by dot product and never through a name.

Three consequences follow, and each is a rule here rather than a preference:

- **This stage is optional.** Induction, coding, deduplication and diagnostics run
  fully offline with no key present. `anthropic` and `python-dotenv` live in an
  optional dependency group and are imported lazily, so reproducing the numbers of
  this experiment never requires an account.
- **This stage is not reproducible, and the artifact says so.** Current models
  reject `temperature`, so an identical prompt may return a different name on a
  different day. The on-disk cache is therefore not an optimization: it is the
  thing that fixes the result. A dictionary is labelled once and that artifact is
  what the report cites. This is admissible precisely because no metric depends on
  it.
- **The key is a real secret.** It is read from the environment, loaded from
  `.env`, and never printed, logged, written into an artifact, or included in a
  cache key. The tests assert the last two rather than trusting the reading.
"""

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from concept_embeddings_rag import config
from concept_embeddings_rag.concepts.dictionary import ConceptArtifactError

API_KEY_VARIABLE = "ANTHROPIC_API_KEY"

PROMPT_TEMPLATE = """You are reading paragraphs that all activate the same latent \
dimension of a concept space induced from a document corpus. The dimension has no \
name yet; these paragraphs are the only evidence of what it means.

Paragraphs:
{evidence}

Name what these paragraphs have in common as a concept. Base the name only on the \
evidence above. If they have little in common, say so plainly in the gloss rather \
than inventing a connection.

Answer with a short noun-phrase name and a one-sentence gloss."""

PROMPT_TEMPLATE_HASH = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:16]

LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "A short noun-phrase name for the concept"},
        "gloss": {"type": "string", "description": "One sentence on what the concept covers"},
    },
    "required": ["name", "gloss"],
    "additionalProperties": False,
}

# The request, minus the model and the prompt. `temperature` is deliberately absent:
# Sonnet 5 rejects it, which is the reason HU-6 declares this stage non-reproducible
# and treats the cache as what fixes the result. Thinking is disabled because naming
# a concept from ten paragraphs needs no reasoning budget, and it keeps the output
# token count - and therefore the spend - predictable.
REQUEST_SHAPE: dict[str, Any] = {
    "thinking": {"type": "disabled"},
    "output_config": {"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
    "max_tokens": 300,
}

# How many times the SDK retries a 429 or a 5xx before giving up, with its own
# exponential backoff. Raised from the SDK default of 2 by the D9 amendment: two
# retries assume failures arrive isolated, and a sustained 529 overload window
# breaks that assumption - the run then dies on its first uncached call and
# re-running resumes only to die again. Retrying does not re-spend: a call that
# never returned was never billed, and an answered one is on disk.
MAX_RETRIES = 8

# Most specific first, per decision D9. A bad key must abort rather than retry; a
# rate limit and a 5xx are retried with backoff by the SDK up to `MAX_RETRIES`, and
# the disk cache means an aborted run resumes where it stopped instead of re-spending.
EXCEPTION_ORDER: tuple[tuple[str, str], ...] = (
    ("AuthenticationError", "the API key was rejected; fix the key rather than retrying"),
    ("RateLimitError", "rate limited after the SDK's own retries; re-run to resume from the cache"),
    ("APIStatusError", "the API returned an error status"),
    ("APIConnectionError", "could not reach the API; re-run to resume from the cache"),
)

REPRODUCIBILITY_NOTE = (
    "Labelling is not reproducible. Current models reject a temperature setting, so an "
    "identical prompt may return a different name on a different day. This artifact, "
    "together with the on-disk prompt cache that produced it, is therefore the record: "
    "re-running does not regenerate these labels, it reads them back. This is admissible "
    "because no metric in this project depends on a label - a concept's embedding is its "
    "dictionary atom, and nothing in retrieval, scoring or expansion reads a name."
)

# Only for the estimate printed and recorded beside the actual spend. Claude's
# tokenizer is not this project's, so this is an approximation and is named as one.
_CHARS_PER_TOKEN = 4


class LabelingError(Exception):
    """The labelling stage could not complete."""


class MissingAPIKey(LabelingError):
    """No API key is available, and this stage will not proceed with a placeholder."""


@dataclass(frozen=True)
class LabelResponse:
    """One answer, with the tokens it actually cost."""

    name: str
    gloss: str
    input_tokens: int
    output_tokens: int


class LabelingClient(Protocol):
    """What `label_concepts` needs. A fake satisfies it, which is how the tests run."""

    model: str

    def label(self, prompt: str) -> LabelResponse: ...


@dataclass(frozen=True)
class ConceptLabels:
    """The labelled concept map of one dictionary, as an artifact."""

    dictionary_key: str
    labels: dict[str, dict[str, Any]]
    model: str
    prompt_hash: str
    n_units_shown: int
    created_at: str
    cost: dict[str, float]
    reproducibility_note: str = REPRODUCIBILITY_NOTE


def build_prompt(concept_index: int, evidence: Sequence[tuple[str, str]]) -> str:
    """The prompt for one concept, built from the units that activate it most.

    The concept index is deliberately absent from the text: it would be a label
    the model could anchor on, and the evidence is supposed to be the only thing
    it reads. It still separates two concepts in the cache key, because the units
    shown differ.
    """
    del concept_index
    shown = "\n\n".join(f"[{position + 1}] {text}" for position, (_, text) in enumerate(evidence))
    return PROMPT_TEMPLATE.format(evidence=shown)


def prompt_cache_key(model: str, prompt: str) -> str:
    """Address one answer by the exact question that produced it.

    Model and prompt only. No key material and no environment enter here: a cache
    key travels into filenames and logs, and a secret must not.
    """
    payload = f"{model}\n{PROMPT_TEMPLATE_HASH}\n{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _cached_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / "label-cache" / f"{key}.json"


def _read_cached(cache_dir: Path, key: str) -> LabelResponse | None:
    path = _cached_path(cache_dir, key)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LabelResponse(
        name=str(payload["name"]),
        gloss=str(payload["gloss"]),
        input_tokens=int(payload["input_tokens"]),
        output_tokens=int(payload["output_tokens"]),
    )


def _write_cached(cache_dir: Path, key: str, response: LabelResponse) -> None:
    path = _cached_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": response.name,
                "gloss": response.gloss,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def estimate_cost_usd(prompts: Sequence[str], max_output_tokens: int) -> float:
    """A rough forecast of what a run will spend, before it spends it.

    Approximate by construction: Claude's tokenizer is not the one this project
    embeds with, and the output allowance is a ceiling rather than a measurement.
    It is recorded as `estimated_usd` beside the measured `actual_usd`.
    """
    input_tokens = sum(len(prompt) for prompt in prompts) / _CHARS_PER_TOKEN
    output_tokens = len(prompts) * max_output_tokens
    return (
        input_tokens / 1e6 * config.LABELING_INPUT_USD_PER_MTOK
        + output_tokens / 1e6 * config.LABELING_OUTPUT_USD_PER_MTOK
    )


def label_concepts(
    *,
    dictionary_key: str,
    evidence: Mapping[int, Sequence[tuple[str, str]]],
    client: LabelingClient,
    cache_dir: Path | str,
    n_units_shown: int = config.LABEL_EVIDENCE_UNITS,
) -> ConceptLabels:
    """Label every concept from its own most strongly activating units.

    `evidence` maps a concept index to `(unit_id, text)` pairs, best first - the
    `top_units_per_concept` of the diagnostics, resolved against the pool. A
    concept with no evidence is not labelled: there is nothing to read.

    Every answer is cached on disk by the exact prompt that produced it, so a
    second run over an unchanged dictionary makes no call at all.
    """
    cache_dir = Path(cache_dir)
    labels: dict[str, dict[str, Any]] = {}
    prompts: list[str] = []
    calls = cached = input_tokens = output_tokens = 0

    for concept in sorted(evidence):
        shown = list(evidence[concept])[:n_units_shown]
        if not shown:
            continue
        prompt = build_prompt(concept, shown)
        prompts.append(prompt)
        key = prompt_cache_key(client.model, prompt)

        response = _read_cached(cache_dir, key)
        if response is None:
            response = client.label(prompt)
            _write_cached(cache_dir, key, response)
            calls += 1
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
        else:
            cached += 1

        labels[str(concept)] = {
            "name": response.name,
            "gloss": response.gloss,
            "evidence_unit_ids": [unit_id for unit_id, _ in shown],
        }

    return ConceptLabels(
        dictionary_key=dictionary_key,
        labels=labels,
        model=client.model,
        prompt_hash=PROMPT_TEMPLATE_HASH,
        n_units_shown=n_units_shown,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        cost={
            "calls": calls,
            "cached": cached,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_usd": estimate_cost_usd(prompts, int(REQUEST_SHAPE["max_tokens"])),
            "actual_usd": (
                input_tokens / 1e6 * config.LABELING_INPUT_USD_PER_MTOK
                + output_tokens / 1e6 * config.LABELING_OUTPUT_USD_PER_MTOK
            ),
        },
    )


def _path_for(directory: Path, dictionary_key: str) -> Path:
    return Path(directory) / f"labels-{dictionary_key}.json"


def save_labels(labels: ConceptLabels, directory: Path | str) -> Path:
    """Write the labelled concept map. It carries no secret, and a test asserts it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = _path_for(directory, labels.dictionary_key)
    path.write_text(
        json.dumps(
            {
                "dictionary_key": labels.dictionary_key,
                "labels": labels.labels,
                "model": labels.model,
                "prompt_hash": labels.prompt_hash,
                "n_units_shown": labels.n_units_shown,
                "created_at": labels.created_at,
                "cost": labels.cost,
                "reproducibility_note": labels.reproducibility_note,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def load_labels(dictionary_key: str, directory: Path | str) -> ConceptLabels:
    """Read back the labelled concept map of one dictionary."""
    path = _path_for(Path(directory), dictionary_key)
    if not path.exists():
        raise ConceptArtifactError(f"no labels for dictionary {dictionary_key} in {directory}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    return ConceptLabels(
        dictionary_key=str(payload["dictionary_key"]),
        labels=dict(payload["labels"]),
        model=str(payload["model"]),
        prompt_hash=str(payload["prompt_hash"]),
        n_units_shown=int(payload["n_units_shown"]),
        created_at=str(payload["created_at"]),
        cost=dict(payload["cost"]),
        reproducibility_note=str(payload["reproducibility_note"]),
    )


def _sdk() -> Any:
    """Import the optional SDK, saying what to install rather than failing raw."""
    try:
        import anthropic
    except ImportError as error:
        raise LabelingError(
            "the labelling stage needs the optional dependency group: "
            "run 'uv sync --group labeling'"
        ) from error
    return anthropic


def read_api_key(env: Mapping[str, str] | None = None) -> str:
    """The key, from the environment or from `.env`, never from a default.

    `python-dotenv` is optional like the SDK: absent it, the environment alone is
    consulted. The value is returned and never printed, logged or stored.
    """
    if env is None:
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:
            pass
        env = os.environ

    key = env.get(API_KEY_VARIABLE, "").strip()
    if not key:
        raise MissingAPIKey(
            f"{API_KEY_VARIABLE} is not set. Copy .env.example to .env and put the key "
            "there, or export it. Labelling is the only stage that needs it: induction, "
            "coding, deduplication and diagnostics all run without it"
        )
    return key


class AnthropicLabelingClient:
    """The real client. Constructed only once a key is known to exist."""

    def __init__(self, *, model: str = config.LABELING_MODEL, api_key: str | None = None) -> None:
        anthropic = _sdk()
        self.model = model
        # `read_api_key` raises before any HTTP client is built, so an absent key
        # never fails deep inside a connection attempt.
        self._client = anthropic.Anthropic(
            api_key=api_key or read_api_key(), max_retries=MAX_RETRIES
        )

    def label(self, prompt: str) -> LabelResponse:
        anthropic = _sdk()
        try:
            message = self._client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **REQUEST_SHAPE,
            )
        except Exception as error:
            for name, explanation in EXCEPTION_ORDER:
                if isinstance(error, getattr(anthropic, name)):
                    raise LabelingError(f"{name}: {explanation}") from error
            raise

        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        try:
            answer = json.loads(text)
        except json.JSONDecodeError as error:
            raise LabelingError(
                "the model did not return the structured answer the schema asked for"
            ) from error

        return LabelResponse(
            name=str(answer["name"]),
            gloss=str(answer["gloss"]),
            input_tokens=int(message.usage.input_tokens),
            output_tokens=int(message.usage.output_tokens),
        )
