"""The gate of Phase 5: GO, NAMES ONLY or STOP, computed from the run and nothing else.

HU-7 of the Phase 5 spec, decision D11 of its plan, exactly as the spec's table writes it:

| Outcome    | Condition                                                     |
|------------|---------------------------------------------------------------|
| GO         | EC beats the comparator and EC beats E                        |
| NAMES ONLY | E beats the comparator and not GO                             |
| STOP       | every other pattern, ambiguous ones included, as anomalies    |

"X beats Y" is a paired sign test over the pilot questions on hit@10: questions where the two
hops score the same are ties and are dropped, and the one-sided exact binomial p-value of X
winning must be below the declared alpha. The comparator is BM25 on the question, frozen in
`config` before any number existed.

NAMES ONLY is never assigned without positive evidence that entities alone navigate. Whether
concept-only beats the comparator is computed and reported, because one anomaly needs it, and
never changes the outcome. Nothing else the run holds - hit@100, new retrievals, concentration,
the subgroup - is read here.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.stats import binomtest

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.evaluation.navigation import ARM_HOPS
from concept_embeddings_rag.evaluation.selection import digest_of_payload, serialize_payload

GO = "GO"
NAMES_ONLY = "NAMES_ONLY"
STOP = "STOP"

ENTITY_HOP = ARM_HOPS["entity"]
CONCEPT_HOP = ARM_HOPS["concept"]
BOTH_HOP = ARM_HOPS["entity+concept"]

GATE_FILENAME = "gate-decision.json"
TEST_DESCRIPTION = (
    "paired sign test over pilot questions on hit@10; ties dropped; one-sided exact binomial "
    "(scipy.stats.binomtest, p = 0.5, alternative='greater'); no discordant question gives p = 1"
)


class GateError(Exception):
    """The gate decision is missing, modified, or bound to another run."""


@dataclass(frozen=True)
class SignTest:
    hop: str
    against: str
    wins: int
    losses: int
    ties: int
    p_value: float
    passed: bool

    def as_payload(self) -> dict[str, Any]:
        return {
            "hop": self.hop,
            "against": self.against,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "p_value": self.p_value,
            "passed": self.passed,
        }


def sign_test(
    per_question: Sequence[Mapping[str, Any]],
    hop: str,
    against: str,
    *,
    depth: int = config.GATE_METRIC_DEPTH,
    alpha: float = config.GATE_ALPHA,
) -> SignTest:
    """Does `hop` score higher than `against` on significantly more questions?"""
    key = str(depth)
    wins = losses = ties = 0
    for entry in per_question:
        mine = entry["scores"][hop][key]
        theirs = entry["scores"][against][key]
        if mine > theirs:
            wins += 1
        elif mine < theirs:
            losses += 1
        else:
            ties += 1
    discordant = wins + losses
    p_value = (
        float(binomtest(wins, discordant, 0.5, alternative="greater").pvalue) if discordant else 1.0
    )
    return SignTest(hop, against, wins, losses, ties, p_value, p_value < alpha)


def decide(
    *,
    ec_beats_comparator: bool,
    ec_beats_entity: bool,
    entity_beats_comparator: bool,
    concept_beats_comparator: bool,
) -> tuple[str, list[str]]:
    """The spec's table, and the anomalies that resolve to STOP."""
    if ec_beats_comparator and ec_beats_entity:
        return GO, []
    if entity_beats_comparator:
        return NAMES_ONLY, []
    anomalies = []
    if ec_beats_comparator:
        anomalies.append(
            "entity+concept beats the comparator but beats neither entity-only nor lets "
            "entity-only beat the comparator on its own: neither concepts adding something nor "
            "names sufficing is shown, so the pattern resolves to STOP"
        )
    if concept_beats_comparator and not ec_beats_comparator:
        anomalies.append(
            "concept-only beats the comparator while neither entity+concept nor entity-only "
            "does: reported, and the outcome stays STOP as the table computes it"
        )
    return STOP, anomalies


def compute_gate(
    hop_run: Mapping[str, Any], *, comparator: str = config.GATE_COMPARATOR
) -> dict[str, Any]:
    """The decision, from the verified hop run's per-question scores alone."""
    per_question = hop_run["per_question"]
    tests = {
        "EC beats BM25": sign_test(per_question, BOTH_HOP, comparator),
        "EC beats E": sign_test(per_question, BOTH_HOP, ENTITY_HOP),
        "E beats BM25": sign_test(per_question, ENTITY_HOP, comparator),
    }
    reported = {"C beats BM25": sign_test(per_question, CONCEPT_HOP, comparator)}
    outcome, anomalies = decide(
        ec_beats_comparator=tests["EC beats BM25"].passed,
        ec_beats_entity=tests["EC beats E"].passed,
        entity_beats_comparator=tests["E beats BM25"].passed,
        concept_beats_comparator=reported["C beats BM25"].passed,
    )
    return {
        "outcome": outcome,
        "tests": {name: test.as_payload() for name, test in tests.items()},
        "reported": {name: test.as_payload() for name, test in reported.items()},
        "anomalies": anomalies,
        "metric": {
            "depth": config.GATE_METRIC_DEPTH,
            "alpha": config.GATE_ALPHA,
            "comparator": comparator,
            "unit": "question",
            "test": TEST_DESCRIPTION,
        },
        "n_questions": len(per_question),
        "hop_run_digest": hop_run["digest"],
    }


def save_gate(decision: Mapping[str, Any], directory: Path | str) -> Path:
    """Write the decision once: the gate is never re-run on the same phase."""
    path = Path(directory) / GATE_FILENAME
    if path.exists():
        raise GateError(
            f"a gate decision is already written at {path}; the gate is applied once and never "
            "re-run with a different metric, test, threshold, comparator or set"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(decision)
    payload["digest"] = digest_of_payload(payload)
    write_text_atomic(path, serialize_payload(payload))
    return path


def load_gate(directory: Path | str, *, hop_run_digest: str) -> dict[str, Any]:
    path = Path(directory) / GATE_FILENAME
    if not path.exists():
        raise GateError(f"no gate decision in {directory}: run `cer navigate`")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("digest") != digest_of_payload(payload):
        raise GateError(f"{path.name} does not match its digest; it has been modified")
    if payload.get("hop_run_digest") != hop_run_digest:
        raise GateError(f"{path.name} was computed from another run than the one asked for")
    return payload
