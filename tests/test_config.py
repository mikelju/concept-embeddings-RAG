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


# --- Phase 3 (T1): the constants that decide what Phase 3 measures -----------


def test_query_operator_arms_are_the_three_declared_in_decision_d4():
    """Truncation is part of the operator, not a knob swept independently."""
    assert config.QUERY_OPERATORS == ("projection_top16", "projection_full", "sparse_coding")


def test_query_truncation_matches_the_dense_end_of_the_corpus_sparsity_band():
    """A top-m query describes the question at the resolution the units are described at."""
    _, band_high = config.SPARSITY_TARGET_BAND
    assert config.QUERY_TOP_M == band_high == 16


def test_both_views_of_x_are_declared_so_neither_is_assumed():
    assert config.CONCEPT_VIEWS == ("raw", "row_normalized")


def test_damping_modes_are_two_and_include_the_identity():
    assert config.DAMPING_MODES == ("none", "idf")


def test_fusion_schemes_are_the_weighted_one_and_the_parameter_free_reference():
    assert config.FUSION_SCHEMES == ("weighted", "rrf")


def test_fusion_weight_grid_has_eleven_points_and_includes_both_endpoints():
    """w = 1.0 must be reachable: it is how the curve says the second signal adds nothing."""
    grid = config.FUSION_WEIGHT_GRID
    assert len(grid) == 11
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert list(grid) == sorted(grid)
    assert all(0.0 <= w <= 1.0 for w in grid)


def test_rrf_constant_is_the_standard_sixty():
    assert config.RRF_K == 60


def test_selection_budget_is_one_of_the_declared_context_budgets():
    """Asserted rather than assumed: a selection budget outside the table is unreadable."""
    assert config.SELECTION_BUDGET == 2048
    assert config.SELECTION_BUDGET in config.CONTEXT_BUDGETS


def test_selection_metric_is_declared_as_a_constant():
    assert config.SELECTION_METRIC == "gold_recall"


def test_thin_concept_threshold_is_declared():
    assert config.THIN_CONCEPT_MAX_UNITS == 30
    assert config.THIN_CONCEPT_MAX_UNITS > config.DEAD_ATOM_MIN_UNITS


def test_phase_3_directories_are_absolute_and_under_data():
    for path in (config.SELECTION_DIR, config.QUESTION_CACHE_DIR):
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert config.DATA_DIR in path.parents


def test_ensure_directories_creates_the_phase_3_directories(tmp_path, monkeypatch):
    """A stage must not have to remember to create its own output directory."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "data" / "cache")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "data" / "results")
    monkeypatch.setattr(config, "CONCEPTS_DIR", tmp_path / "data" / "concepts")
    monkeypatch.setattr(config, "SELECTION_DIR", tmp_path / "data" / "selection")
    monkeypatch.setattr(config, "QUESTION_CACHE_DIR", tmp_path / "data" / "cache" / "questions")

    config.ensure_directories()

    assert config.SELECTION_DIR.is_dir()
    assert config.QUESTION_CACHE_DIR.is_dir()


# --- Phase 4 (T1): the constants that decide what Phase 4 measures -----------


def test_restart_grid_is_the_four_declared_points_and_excludes_the_endpoints():
    """restart = 1.0 is the seed and is asserted by a test, not paid for with a dev evaluation."""
    grid = config.RESTART_GRID
    assert grid == (0.2, 0.4, 0.6, 0.8)
    assert 1.0 not in grid
    assert 0.0 not in grid
    assert list(grid) == sorted(grid)


def test_normalization_arms_are_the_three_declared_in_decision_d3():
    assert config.NORMALIZATION_ARMS == ("none", "symmetric", "stochastic")


def test_stop_threshold_and_cap_are_declared_constants_not_grids():
    """HU-4 forbids a third degree of freedom: these are values, not alternatives."""
    assert isinstance(config.STOP_THRESHOLD, float)
    assert 0.0 < config.STOP_THRESHOLD < 1.0
    assert isinstance(config.MAX_ITERATIONS, int)
    assert config.MAX_ITERATIONS >= 2


def test_the_two_seed_arms_are_declared_before_measurement():
    assert config.SEED_ARMS == ("dense", "conceptual")


def test_seed_depth_equals_the_depth_the_harness_reads():
    """Decision D5: asking the seed for more would make restart = 1.0 hold at one depth only."""
    assert config.SEED_TOP_K == config.EVALUATION_TOP_K == 100


def test_the_dev_evaluation_cap_is_declared_and_larger_than_the_grid():
    """24 cells of the grid against a cap of 40, so the slack is visible rather than assumed."""
    grid_cells = len(config.RESTART_GRID) * len(config.NORMALIZATION_ARMS) * len(config.SEED_ARMS)
    assert config.DEV_EVALUATION_CAP == 40
    assert grid_cells == 24
    assert grid_cells < config.DEV_EVALUATION_CAP


def test_the_trace_widths_are_declared_and_decide_only_what_a_human_reads():
    """T8's two constants: how wide a round of the trace is, never what the walk computed."""
    assert isinstance(config.TRACE_TOP_CONCEPTS, int)
    assert isinstance(config.TRACE_TOP_UNITS, int)
    assert config.TRACE_TOP_CONCEPTS > 0
    assert config.TRACE_TOP_UNITS > 0
    # Narrower than the depth the harness reads, or the trace would be the ranking.
    assert config.TRACE_TOP_UNITS < config.SEED_TOP_K


def test_phase_4_directories_are_absolute_and_under_data():
    for path in (config.EXPANSION_DIR, config.TRACES_DIR):
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert config.DATA_DIR in path.parents


def test_ensure_directories_creates_the_phase_4_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "data" / "cache")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "data" / "results")
    monkeypatch.setattr(config, "CONCEPTS_DIR", tmp_path / "data" / "concepts")
    monkeypatch.setattr(config, "SELECTION_DIR", tmp_path / "data" / "selection")
    monkeypatch.setattr(config, "QUESTION_CACHE_DIR", tmp_path / "data" / "cache" / "questions")
    monkeypatch.setattr(config, "EXPANSION_DIR", tmp_path / "data" / "expansion")
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path / "data" / "traces")

    config.ensure_directories()

    assert config.EXPANSION_DIR.is_dir()
    assert config.TRACES_DIR.is_dir()


# --- Phase 5: text-derived concepts, navigation pilot ------------------------


def test_phase_5_directories_are_absolute_and_under_data():
    for path in (
        config.PILOT_DIR,
        config.EXTRACTION_DIR,
        config.NODES_DIR,
        config.NAVIGATION_DIR,
    ):
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert config.DATA_DIR in path.parents


def test_the_second_hop_protocol_is_the_diagnostics():
    """HU-5: the pilot reuses the diagnostic's read depth and its two depths, unchanged."""
    assert config.PILOT_READ_DEPTH == 10
    assert config.SECOND_HOP_DEPTHS == (10, 100)


def test_the_extractor_and_its_rates_are_declared_before_any_call():
    """Decided in the spec conversation: Sonnet 5, at half price through the Batches API."""
    assert config.EXTRACTION_MODEL == "claude-sonnet-5"
    assert config.EXTRACTION_INPUT_USD_PER_MTOK == 2.00
    assert config.EXTRACTION_OUTPUT_USD_PER_MTOK == 10.00
    assert config.BATCH_PRICE_FACTOR == 0.5
    assert config.EXTRACTION_MAX_TOKENS == 1024


def test_the_extraction_bounds_are_declared_and_exceeding_them_is_a_failure():
    """Decision D3: bounds are enforced client-side, and an answer outside them is not truncated."""
    assert config.MAX_NODES_PER_TYPE == 50
    assert config.MAX_NODE_CHARS == 120


def test_the_cost_guards_are_declared_before_money_is_spent():
    """HU-2 and D4: a sample first, a margin on its estimate, and a ceiling above which it stops."""
    assert config.EXTRACTION_SAMPLE_SIZE == 20
    assert config.EXTRACTION_ESTIMATE_MARGIN == 1.5
    assert config.EXTRACTION_COST_CEILING_USD == 25.0
    assert config.EXTRACTION_BATCH_SIZE == 5000
    assert config.EXTRACTION_FAILURE_FINDING == 0.01


def test_node_types_and_arms_separate_names_from_concepts():
    """HU-4: two node types, three arms, and nothing else."""
    assert config.NODE_TYPES == ("entity", "concept")
    assert config.NAVIGATION_ARMS == ("entity", "concept", "entity+concept")
    assert config.NORMALIZATION_VERSION == "normalization-v1"


def test_the_gate_is_declared_in_config_exactly_as_the_spec_writes_it():
    """HU-7: hit@10 per question, one-sided p < 0.05, BM25 on the question as the comparator."""
    assert config.GATE_METRIC_DEPTH == 10
    assert config.GATE_METRIC_DEPTH in config.SECOND_HOP_DEPTHS
    assert config.GATE_ALPHA == 0.05
    assert config.GATE_COMPARATOR == "bm25-question"


def test_the_reporting_constants_decide_only_what_a_human_reads():
    assert config.TRACES_READ_PER_ARM == 5
    assert config.REFERENCE_CONCEPT_K == 512
    assert config.REFERENCE_CONCEPT_K in config.CONCEPT_K_SWEEP


def test_ensure_directories_creates_the_phase_5_directories(tmp_path, monkeypatch):
    for name in ("PILOT_DIR", "EXTRACTION_DIR", "NODES_DIR", "NAVIGATION_DIR"):
        monkeypatch.setattr(config, name, tmp_path / "data" / name.lower())
    for name in (
        "DATA_DIR",
        "CACHE_DIR",
        "RESULTS_DIR",
        "CONCEPTS_DIR",
        "SELECTION_DIR",
        "QUESTION_CACHE_DIR",
        "EXPANSION_DIR",
        "TRACES_DIR",
    ):
        monkeypatch.setattr(config, name, tmp_path / "other" / name.lower())

    config.ensure_directories()

    for name in ("PILOT_DIR", "EXTRACTION_DIR", "NODES_DIR", "NAVIGATION_DIR"):
        assert getattr(config, name).is_dir()
