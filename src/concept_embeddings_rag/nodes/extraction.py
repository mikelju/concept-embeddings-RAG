"""Reading entities and concepts out of each paragraph, offline (Phase 5, HU-2).

The representation Phase 5 tests is the one the proposal describes: what a paragraph is
about, read from its text. A model reads every paragraph of the pool once, before any hop
runs, and returns two lists - entities and concepts - in a fixed structure. Nothing here
is ever called from a retrieval or an evaluation.

Three rules, each a decision of the phase plan:

- **A request is a pure function of one paragraph** (D3). `build_request` takes an
  `IndexingUnit` and the model name and nothing else, so a question, an answer, a gold
  annotation or a split cannot reach the extractor by construction.
- **An answer outside its bounds is a failure, never a repair** (D3). Structured outputs
  cannot express length limits, so they are checked here; unlike `label`, which truncates,
  nothing is cut to fit, because the spec forbids silent repair.
- **The cache is the record** (D5, spec restrictions). Current models reject `temperature`,
  so the same paragraph can come back with different nodes on another day. Every answer is
  cached per paragraph, bound to the model and the prompt digest it came from, and an entry
  that does not say so is refused rather than trusted.

The key is never touched here: this module builds requests and reads answers. The clients
that send them live in T6 and read the key through `labeling.read_api_key`.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import IndexingUnit

OK = "ok"
FAILED = "failed"

# Why a record failed. `refusal` and `max_tokens` come from the stop reason; `not_json`,
# `schema` and `bounds` from reading the answer. All five are permanent for a given
# prompt, so they are cached like a success and never paid for twice.
FAILURE_REASONS: tuple[str, ...] = ("refusal", "max_tokens", "not_json", "schema", "bounds")

PROMPT_TEMPLATE = """Read the paragraph below and list what it is about, in two lists.

"entities": the specific named things the paragraph mentions or is about - people, \
organizations, places, works such as books, films, songs, albums or television series, \
events, and anything else with a proper name. Write each as the name itself, in its most \
complete common form, never a pronoun or a description.

"concepts": the general topics and notions the paragraph is about - kinds of things, \
fields, activities, roles and properties - that paragraphs about other subjects could \
share. A concept is never a proper name. Write each as a short noun phrase of one to three \
words, in its most common general wording and in the singular.

Use only what the paragraph itself says, without outside knowledge. Do not repeat an item \
within a list. List at most 50 items in each list, and return a list empty when nothing \
fits it.

Paragraph:
{paragraph}"""

PROMPT_DIGEST = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:16]

NODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific named things: people, organizations, places, works, events",
        },
        "concepts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "General topics and notions, never proper names",
        },
    },
    "required": ["entities", "concepts"],
    "additionalProperties": False,
}

# Everything a request carries besides the model and the paragraph. Thinking is disabled
# as in `label`: extraction needs no reasoning budget, and thinking is billed as output.
REQUEST_SHAPE: dict[str, Any] = {
    "thinking": {"type": "disabled"},
    "output_config": {"format": {"type": "json_schema", "schema": NODE_SCHEMA}},
    "max_tokens": config.EXTRACTION_MAX_TOKENS,
}

_CACHE_FIELDS = (
    "unit_id",
    "model",
    "prompt_digest",
    "status",
    "entities",
    "concepts",
    "failure",
    "input_tokens",
    "output_tokens",
)


class ExtractionError(Exception):
    """An extraction artifact or cache entry is not what it claims to be."""


@dataclass(frozen=True)
class RawResponse:
    """What a client hands back, stripped of the SDK: enough to judge the answer and its cost."""

    stop_reason: str
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ExtractionRecord:
    unit_id: str
    model: str
    prompt_digest: str
    status: str
    entities: tuple[str, ...]
    concepts: tuple[str, ...]
    failure: str | None
    input_tokens: int
    output_tokens: int


def render_prompt(unit: IndexingUnit) -> str:
    """The frozen template, filled with what both retrievers index: title and paragraph."""
    return PROMPT_TEMPLATE.format(paragraph=unit.indexable_text)


def build_request(unit: IndexingUnit, *, model: str = config.EXTRACTION_MODEL) -> dict[str, Any]:
    """One paragraph, one request. The signature is the guarantee HU-8 relies on."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": render_prompt(unit)}],
        **REQUEST_SHAPE,
    }


class _Rejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _validated_lists(answer: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The two lists, exactly as returned, or the reason they cannot be accepted."""
    if not isinstance(answer, dict) or set(answer) != {"entities", "concepts"}:
        raise _Rejected("schema")
    lists = []
    for field in ("entities", "concepts"):
        items = answer[field]
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise _Rejected("schema")
        if len(items) > config.MAX_NODES_PER_TYPE:
            raise _Rejected("bounds")
        if any(len(item) > config.MAX_NODE_CHARS for item in items):
            raise _Rejected("bounds")
        lists.append(tuple(items))
    return lists[0], lists[1]


def record_from_response(unit_id: str, response: RawResponse, *, model: str) -> ExtractionRecord:
    """Judge one answer. It is external input: nothing reaches a record unchecked."""

    def failed(reason: str) -> ExtractionRecord:
        return ExtractionRecord(
            unit_id=unit_id,
            model=model,
            prompt_digest=PROMPT_DIGEST,
            status=FAILED,
            entities=(),
            concepts=(),
            failure=reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    if response.stop_reason == "refusal":
        return failed("refusal")
    if response.stop_reason == "max_tokens":
        return failed("max_tokens")
    if response.stop_reason != "end_turn":
        # A request with no tools and no stop sequences has no other legitimate way to end;
        # whatever it returned is not the structured answer the schema asked for.
        return failed("schema")
    try:
        decoded = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return failed("not_json")
    try:
        entities, concepts = _validated_lists(decoded)
    except _Rejected as rejection:
        return failed(rejection.reason)

    return ExtractionRecord(
        unit_id=unit_id,
        model=model,
        prompt_digest=PROMPT_DIGEST,
        status=OK,
        entities=entities,
        concepts=concepts,
        failure=None,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def _cache_path(cache_dir: Path | str, unit_id: str, model: str, prompt_digest: str) -> Path:
    """One file per (model, prompt, paragraph). No key material ever enters the name."""
    key = hashlib.sha256(f"{model}\n{prompt_digest}\n{unit_id}".encode()).hexdigest()[:32]
    return Path(cache_dir) / "extraction-cache" / f"{key}.json"


def write_cached(cache_dir: Path | str, record: ExtractionRecord) -> Path:
    path = _cache_path(cache_dir, record.unit_id, record.model, record.prompt_digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit_id": record.unit_id,
        "model": record.model,
        "prompt_digest": record.prompt_digest,
        "status": record.status,
        "entities": list(record.entities),
        "concepts": list(record.concepts),
        "failure": record.failure,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
    }
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))
    return path


def read_cached(
    cache_dir: Path | str,
    unit_id: str,
    *,
    model: str,
    prompt_digest: str = PROMPT_DIGEST,
) -> ExtractionRecord | None:
    """The cached record for this paragraph under this model and prompt, or nothing.

    A lookup under another model or prompt finds another file, so an answer to a different
    question is never reused. An entry that exists but does not bind itself to the key it is
    filed under is refused: this cache fixes a non-reproducible result, and an edited entry
    would silently rewrite what the pilot measured.
    """
    path = _cache_path(cache_dir, unit_id, model, prompt_digest)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ExtractionError(
            f"the cached extraction in {path} is not readable JSON; delete that file to "
            "re-request it"
        ) from error
    if not isinstance(payload, dict) or set(payload) != set(_CACHE_FIELDS):
        raise ExtractionError(
            f"the cached extraction in {path} is not an extraction entry; delete that file to "
            "re-request it"
        )
    if (payload["unit_id"], payload["model"], payload["prompt_digest"]) != (
        unit_id,
        model,
        prompt_digest,
    ):
        raise ExtractionError(
            f"the cached extraction in {path} claims another paragraph, model or prompt than "
            "the one it is filed under; delete that file to re-request it"
        )
    status = payload["status"]
    failure = payload["failure"]
    if status not in (OK, FAILED) or (status == FAILED) != (failure in FAILURE_REASONS):
        raise ExtractionError(
            f"the cached extraction in {path} has an inconsistent status; delete that file to "
            "re-request it"
        )
    try:
        entities, concepts = _validated_lists(
            {"entities": payload["entities"], "concepts": payload["concepts"]}
        )
    except _Rejected as rejection:
        raise ExtractionError(
            f"the cached extraction in {path} fails its own bounds ({rejection.reason}); delete "
            "that file to re-request it"
        ) from rejection
    return ExtractionRecord(
        unit_id=unit_id,
        model=model,
        prompt_digest=prompt_digest,
        status=status,
        entities=entities,
        concepts=concepts,
        failure=failure,
        input_tokens=int(payload["input_tokens"]),
        output_tokens=int(payload["output_tokens"]),
    )
