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


# --- Phase 2 (T1): the constants that decide what the concept space is -------


def test_concept_k_sweep_is_the_four_declared_sizes():
    assert config.CONCEPT_K_SWEEP == (512, 1024, 2048, 4096)


def test_concept_seed_is_an_explicit_constant():
    assert isinstance(config.CONCEPT_SEED, int)


def test_sparsity_target_band_sits_above_the_rejection_floor():
    low, high = config.SPARSITY_TARGET_BAND
    assert config.SPARSITY_FLOOR == 3
    assert low > config.SPARSITY_FLOOR
    assert low < high


def test_merge_threshold_and_merge_budget_are_declared_fractions():
    assert 0.0 < config.MERGE_COSINE_THRESHOLD < 1.0
    assert 0.0 < config.MAX_MERGED_FRACTION < 1.0


def test_concepts_dir_is_under_data_and_absolute():
    assert config.CONCEPTS_DIR.is_absolute()
    assert config.DATA_DIR in config.CONCEPTS_DIR.parents


def test_labelling_targets_exactly_one_dictionary_of_the_sweep():
    assert config.LABELED_K == 2048
    assert config.LABELED_K in config.CONCEPT_K_SWEEP
    assert config.LABELING_MODEL == "claude-sonnet-5"
    assert config.LABEL_EVIDENCE_UNITS >= 1


def test_coherence_is_measured_over_more_units_than_are_shown_as_evidence():
    """Ten units are what a human reads; a mean needs a sample, not an anecdote."""
    assert config.COHERENCE_TOP_UNITS > config.TOP_UNITS_PER_CONCEPT
    assert config.COHERENCE_MIN_UNITS >= 3
    assert config.COHERENCE_MIN_UNITS <= config.COHERENCE_TOP_UNITS


def test_the_null_baseline_is_drawn_from_enough_samples_to_have_a_spread():
    assert config.COHERENCE_NULL_SAMPLES >= 30


def test_induction_batching_constants_are_positive():
    assert config.CONCEPT_BATCH_SIZE > 0
    assert config.CONCEPT_MAX_ITER > 0
    assert config.CODING_BATCH_SIZE > 0
