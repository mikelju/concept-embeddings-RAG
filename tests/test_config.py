"""T1: configuration constants and project paths.

Paths must resolve relative to the project root, never to the working directory:
the CLI has to behave the same whether it is invoked from the repo root or from
anywhere else.
"""

from pathlib import Path

from concept_embeddings_rag import config


def test_project_root_contains_pyproject():
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()


def test_paths_are_absolute_and_under_project_root():
    for path in (config.DATA_DIR, config.CACHE_DIR, config.RESULTS_DIR):
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert config.PROJECT_ROOT in path.parents or path == config.PROJECT_ROOT


def test_context_budgets_are_the_four_declared_in_the_spec():
    assert config.CONTEXT_BUDGETS == (512, 1024, 2048, 4096)


def test_recall_at_k_values():
    assert config.RECALL_AT_K == (2, 5, 10)


def test_split_sizes_match_the_spec():
    assert config.N_QUESTIONS == 2000
    assert config.N_DEV == 600
    assert config.N_TEST == 1400
    assert config.N_DEV + config.N_TEST == config.N_QUESTIONS


def test_seed_is_an_explicit_constant():
    assert isinstance(config.DEFAULT_SEED, int)


def test_embedding_model_is_pinned_by_name_and_revision():
    assert config.EMBEDDING_MODEL
    assert config.EMBEDDING_REVISION
    assert config.EMBEDDING_DIM == 384


def test_corpus_source_url_is_declared():
    assert config.HOTPOTQA_DEV_DISTRACTOR_URL.startswith("http")
