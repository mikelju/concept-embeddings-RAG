"""Scaffold verification: the package imports and the numerical stack is operational.

The import test is not decorative: this machine is Windows on ARM (win_arm64) and much of
the ecosystem publishes no wheels for that platform. If any of these imports fails, the
environment is broken before we measure anything at all.
"""

import importlib


def test_package_imports():
    module = importlib.import_module("concept_embeddings_rag")
    assert module is not None


def test_numerical_stack_available():
    for name in ("numpy", "scipy", "sklearn"):
        assert importlib.import_module(name) is not None


def test_torch_available_on_arm64():
    torch = importlib.import_module("torch")
    tensor = torch.ones(3)
    assert tensor.sum().item() == 3.0


def test_sentence_transformers_imports():
    assert importlib.import_module("sentence_transformers") is not None
