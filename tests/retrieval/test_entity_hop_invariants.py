"""Phase 6, T5: what the entity stage cannot do, asserted rather than remembered (HU-3, HU-9).

- **Nothing tunable.** Its constructor takes the node index and the weights and nothing
  else: no types, no depth, no threshold. The types are `config.ENTITY_HOP_TYPES`, the read
  depth is `config.PILOT_READ_DEPTH`, and the maximum depth is whatever the hybrid asks for.
- **No answer key.** The Phase 3 scoring-path predicate - the very function that guards
  `retrieval/` - passes on `retrieval/entity_hop.py` and on `evaluation/second_hop.py`, which
  sits outside the globbed directory and is applied here explicitly. A mutated copy that
  reads `gold_unit_ids` fails it, so the guard is seen to work.
"""

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.retrieval import entity_hop
from concept_embeddings_rag.retrieval.entity_hop import EntityHopStage
from concept_embeddings_rag.retrieval.fusion import SECOND_STAGES

TESTS = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parents[2] / "src" / "concept_embeddings_rag"


def phase_3_guard() -> ModuleType:
    """The Phase 3 guard module itself, loaded by path, so its predicate is not a copy."""
    path = TESTS / "concepts" / "test_labels_not_in_retrieval_path.py"
    spec = importlib.util.spec_from_file_location("phase_3_scoring_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_constructor_takes_the_index_and_the_weights_and_nothing_else():
    parameters = list(inspect.signature(EntityHopStage.__init__).parameters)
    assert parameters == ["self", "index", "weights"]


def test_the_stage_reads_the_declared_types_and_depth_from_config():
    assert entity_hop.TYPES == config.ENTITY_HOP_TYPES == ("entity",)
    assert entity_hop.READ_DEPTH == config.PILOT_READ_DEPTH == 10


def test_the_stage_is_named_as_the_declared_second_stage():
    assert EntityHopStage.name == config.ENTITY_HOP_NAME
    assert EntityHopStage.name in SECOND_STAGES


def test_the_stage_module_names_no_threshold_expansion_or_canonicalization():
    source = (SOURCE / "retrieval" / "entity_hop.py").read_text(encoding="utf-8")
    used = phase_3_guard().identifiers_used(source)
    for forbidden in ("threshold", "min_score", "alias", "canonical", "cluster", "relation"):
        assert not any(forbidden in name.lower() for name in used), forbidden


@pytest.mark.parametrize("module", ["retrieval/entity_hop.py", "evaluation/second_hop.py"])
def test_the_phase_3_scoring_path_predicate_passes_on_the_entity_path(module: str):
    guard = phase_3_guard()
    used = guard.identifiers_used((SOURCE / module).read_text(encoding="utf-8"))
    assert not used & set(guard.FORBIDDEN_ANNOTATIONS)


def test_the_predicate_fails_on_a_mutated_copy_of_the_stage():
    guard = phase_3_guard()
    clean = (SOURCE / "retrieval" / "entity_hop.py").read_text(encoding="utf-8")
    anchor = "    def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:"
    offending = clean.replace(
        anchor,
        "    def propose(\n        self, first: Sequence[Hit], top_k: int, gold_unit_ids=()\n"
        "    ) -> list[Hit]:",
    )

    assert offending != clean, "the mutation did not land; the guard was not exercised"
    assert guard.identifiers_used(offending) & set(guard.FORBIDDEN_ANNOTATIONS) == {"gold_unit_ids"}


def test_the_predicate_fails_on_a_mutated_copy_of_the_second_hop_module():
    guard = phase_3_guard()
    clean = (SOURCE / "evaluation" / "second_hop.py").read_text(encoding="utf-8")
    offending = clean.replace("    incidence = index.incidence\n", "    split = index.split\n", 1)

    assert offending != clean
    assert "split" in guard.identifiers_used(offending) & set(guard.FORBIDDEN_ANNOTATIONS)


def test_the_retrieval_directory_guard_already_globs_the_stage():
    guard = phase_3_guard()
    paths = {path.name for path in guard.modules_on_the_scoring_path()}
    assert "entity_hop.py" in paths
