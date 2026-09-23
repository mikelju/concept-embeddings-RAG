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
    # Phase 6 added a directory; patched so this test writes nothing under the real data/.
    monkeypatch.setattr(config, "REPLACEMENT_DIR", tmp_path / "data" / "replacement")
    # Phase 8 added another; same reason.
    monkeypatch.setattr(config, "PHASE_8_DIR", tmp_path / "data" / "phase8")

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
    monkeypatch.setattr(config, "REPLACEMENT_DIR", tmp_path / "data" / "replacement")
    monkeypatch.setattr(config, "PHASE_8_DIR", tmp_path / "data" / "phase8")

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
    # Raised from 25.0 by deviation 5.1, after the sample measured 33.17 USD with its margin.
    assert config.EXTRACTION_COST_CEILING_USD == 35.0
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
    monkeypatch.setattr(config, "REPLACEMENT_DIR", tmp_path / "other" / "replacement")
    for name in (
        "DATA_DIR",
        "CACHE_DIR",
        "RESULTS_DIR",
        "CONCEPTS_DIR",
        "SELECTION_DIR",
        "QUESTION_CACHE_DIR",
        "EXPANSION_DIR",
        "TRACES_DIR",
        "PHASE_8_DIR",
    ):
        monkeypatch.setattr(config, name, tmp_path / "other" / name.lower())

    config.ensure_directories()

    for name in ("PILOT_DIR", "EXTRACTION_DIR", "NODES_DIR", "NAVIGATION_DIR"):
        assert getattr(config, name).is_dir()


# --- Phase 6: dense + entity navigation, end to end ----------------------------


def test_phase_6_writes_to_its_own_directory_under_data_and_not_to_results():
    """D1: a Phase 6 result under data/results/ would become the Phase 3 control for a reader."""
    assert config.REPLACEMENT_DIR == config.DATA_DIR / "replacement"
    assert config.REPLACEMENT_DIR.is_absolute()
    assert config.REPLACEMENT_DIR != config.RESULTS_DIR


def test_the_phase_6_system_and_component_names_follow_the_hybrid_naming_rule():
    assert config.ENTITY_HOP_NAME == "entity-hop"
    assert config.ENTITY_HOP_TYPES == ("entity",)
    assert config.PHASE_6_SYSTEMS == ("dense", "hybrid-bm25", "hybrid-entity-hop")
    assert config.PHASE_6_SYSTEMS[2] == f"hybrid-{config.ENTITY_HOP_NAME}"


def test_the_entity_hop_depth_is_the_depth_every_component_is_asked_for():
    assert config.ENTITY_HOP_MAX_DEPTH == config.EVALUATION_TOP_K == 100
    assert config.PILOT_READ_DEPTH == 10


# D2's full digests, as literals. Each is the sha256 of a public, pipeline-written artifact; the
# inline pragma records that the secret scanner's finding on it was audited (T20).
PINNED_DIGEST_LITERALS = (
    "91daa10ef0a75b6eba377d18a55ac868467b01b09fb7284c7835a84d4e4e602d",  # pragma: allowlist secret
    "e0f0af8f468dbf3332d9311948f416b75e4e5bf0d9971a0bd332b04acff2c364",  # pragma: allowlist secret
    "49847f3cedb635406416bbb15dfa1913f4fa10782ed07a6461ccd56d5ded246a",  # pragma: allowlist secret
    "163777a044e1d6fcded400495a6e8f9917411e614978aa1aa43427a6cc8d5925",  # pragma: allowlist secret
    "8e59854118ae5f4d83d88ce967098801f9f922309d7beb6b3e3deaf06fb1f12f",  # pragma: allowlist secret
    "b498f99418389b2f7849c680cda1749ea58700a9997fbfed7ce474d3b0a0559a",  # pragma: allowlist secret
)


def test_the_inherited_artifacts_are_pinned_by_their_full_digests():
    """D2: the spec quotes each by prefix; config holds the whole value, copied once."""
    pinned = (
        config.PINNED_SELECTION_DIGEST,
        config.PINNED_PILOT_DIGEST,
        config.PINNED_HOP_RUN_DIGEST,
        config.PINNED_NAVIGATION_TRACES_DIGEST,
        config.PINNED_EXTRACTION_DIGEST,
        config.PINNED_NODE_INDEX_DIGEST,
    )
    assert pinned == PINNED_DIGEST_LITERALS
    assert config.PINNED_EXTRACTION_PROMPT_DIGEST == "0107de3ae9b4e4a3"  # pragma: allowlist secret
    assert config.PINNED_NORMALIZATION_VERSION == "normalization-v1"
    assert config.PINNED_NORMALIZATION_VERSION == config.NORMALIZATION_VERSION
    assert config.PINNED_SELECTION_FROZEN_AT == "2026-09-11T13:30:36+00:00"
    assert config.PHASE_1_UNIT_SET_HASH == "101f564fdcca620c"  # pragma: allowlist secret


def test_the_pinned_digests_are_the_prefixes_the_spec_quotes():
    assert config.PINNED_SELECTION_DIGEST.startswith("91daa10e")
    assert config.PINNED_PILOT_DIGEST.startswith("e0f0af8f")
    assert config.PINNED_HOP_RUN_DIGEST.startswith("49847f3c")
    assert config.PINNED_NAVIGATION_TRACES_DIGEST.startswith("163777a0")
    assert config.PINNED_EXTRACTION_DIGEST.startswith("8e598541")
    assert config.PINNED_NODE_INDEX_DIGEST.startswith("b498f994")


def test_the_four_historical_result_files_are_pinned_by_name():
    assert config.HISTORICAL_DEV_RESULT_FILES == {
        "dense": "run-dense-dev-20260911T133154+0000.json",
        "hybrid-bm25": "run-hybrid-bm25-dev-20260911T133156+0000.json",
    }
    assert config.HISTORICAL_TEST_RESULT_FILES == {
        "dense": "run-dense-test-20260911T133159+0000.json",
        "hybrid-bm25": "run-hybrid-bm25-test-20260911T133204+0000.json",
    }
    assert config.HISTORICAL_CONFIG_KEYS == (
        "unit_set_hash",
        "tokenizer",
        "seed",
        "top_k",
        "model",
        "revision",
    )
    assert config.HISTORICAL_HYBRID_CONFIG_KEYS == ("fusion_scheme", "fusion_weights")


def test_the_expected_counts_are_the_ones_the_spec_quotes():
    assert config.EXPECTED_N_UNITS == 19366
    assert config.EXPECTED_ENTITY_NODES == 96416
    assert config.EXPECTED_FAILED_EXTRACTIONS == 8
    assert config.EXPECTED_UNITS_WITHOUT_ENTITY_NODE == 163
    assert config.EXPECTED_PILOT_QUESTIONS == 152
    assert config.EXPECTED_ENTITY_TRACES == 116


def test_ensure_directories_creates_the_phase_6_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPLACEMENT_DIR", tmp_path / "data" / "replacement")
    for name in (
        "DATA_DIR",
        "CACHE_DIR",
        "RESULTS_DIR",
        "CONCEPTS_DIR",
        "SELECTION_DIR",
        "QUESTION_CACHE_DIR",
        "EXPANSION_DIR",
        "TRACES_DIR",
        "PILOT_DIR",
        "EXTRACTION_DIR",
        "NODES_DIR",
        "NAVIGATION_DIR",
        "PHASE_7_DIR",
        "PHASE_8_DIR",
    ):
        monkeypatch.setattr(config, name, tmp_path / "other" / name.lower())

    config.ensure_directories()

    assert config.REPLACEMENT_DIR.is_dir()


# --- Phase 7 (S1): the constants that decide what a cheap extractor is -------
#
# The spec settles every value before a dev figure exists (decision 10), so these
# assertions are not a restatement of the module: they are the only thing that makes
# "fixed before measuring" checkable afterwards. A changed pin, label set or bar has
# to break a test rather than quietly redefine what the phase measured.


def test_the_phase_7_directory_is_its_own_and_under_the_data_directory():
    assert config.PHASE_7_DIR == config.DATA_DIR / "phase7"
    assert config.PHASE_7_DIR not in (
        config.EXTRACTION_DIR,
        config.NODES_DIR,
        config.REPLACEMENT_DIR,
        config.RESULTS_DIR,
    )


def test_ensure_directories_creates_the_phase_7_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PHASE_7_DIR", tmp_path / "data" / "phase7")
    for name in (
        "DATA_DIR",
        "CACHE_DIR",
        "RESULTS_DIR",
        "CONCEPTS_DIR",
        "SELECTION_DIR",
        "QUESTION_CACHE_DIR",
        "EXPANSION_DIR",
        "TRACES_DIR",
        "PILOT_DIR",
        "EXTRACTION_DIR",
        "NODES_DIR",
        "NAVIGATION_DIR",
        "REPLACEMENT_DIR",
        # Added with Phase 8: without it, this test creates the real `data/phase8/`.
        "PHASE_8_DIR",
    ):
        monkeypatch.setattr(config, name, tmp_path / "other" / name.lower())

    config.ensure_directories()

    assert config.PHASE_7_DIR.is_dir()


def test_the_phase_has_exactly_two_local_candidates():
    """Three arms in all: the Claude reference, quoted rather than re-run, plus these."""
    assert config.PHASE_7_EXTRACTORS == ("gliner", "spacy")
    assert config.GLINER_EXTRACTOR == "gliner"
    assert config.SPACY_EXTRACTOR == "spacy"


def test_gliner_is_pinned_to_a_commit_and_to_the_five_declared_labels():
    assert config.GLINER_MODEL == "urchade/gliner_medium-v2.1"
    assert config.GLINER_REVISION == (
        "40ec419335d09393f298636f471328b722c6da9e"  # pragma: allowlist secret
    )
    assert config.GLINER_TOKENIZER_MODEL == "microsoft/deberta-v3-base"
    assert config.GLINER_TOKENIZER_REVISION == (
        "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"  # pragma: allowlist secret
    )
    for revision in (config.GLINER_REVISION, config.GLINER_TOKENIZER_REVISION):
        assert len(revision) == 40
        assert all(char in "0123456789abcdef" for char in revision)
    assert config.GLINER_LABELS == (
        "person",
        "organization",
        "location",
        "work of art",
        "event",
    )
    assert config.GLINER_THRESHOLD == 0.5
    assert config.GLINER_FLAT_NER is True
    assert config.GLINER_MULTI_LABEL is False
    assert config.GLINER_BATCH_SIZE == 8
    assert config.GLINER_MAX_LEN == 384
    assert config.GLINER_MAX_WIDTH == 12


def test_no_pickle_archive_is_in_either_gliner_allow_list():
    """D14: what is never downloaded cannot be deserialized."""
    allowed = set(config.GLINER_ALLOW_PATTERNS) | set(config.GLINER_TOKENIZER_ALLOW_PATTERNS)
    assert "model.safetensors" in config.GLINER_ALLOW_PATTERNS
    assert not any(
        name in allowed for name in ("pytorch_model.bin", "tf_model.h5", "rust_model.ot")
    )
    assert not any(name.endswith((".bin", ".h5", ".ot", ".pt", ".pth", ".pkl")) for name in allowed)


def test_spacy_is_pinned_to_one_pipeline_version_from_its_official_wheel():
    assert config.SPACY_MODEL == "en_core_web_sm"
    assert config.SPACY_MODEL_VERSION == "3.8.0"
    assert config.SPACY_MODEL_WHEEL_URL.startswith("https://github.com/explosion/spacy-models/")
    assert config.SPACY_MODEL_WHEEL_URL.endswith("en_core_web_sm-3.8.0-py3-none-any.whl")
    assert config.SPACY_EXCLUDE == (
        "tagger",
        "parser",
        "attribute_ruler",
        "lemmatizer",
        "senter",
    )
    assert "ner" not in config.SPACY_EXCLUDE
    assert "tok2vec" not in config.SPACY_EXCLUDE
    assert config.SPACY_BATCH_SIZE == 64
    assert config.SPACY_PROCESSES == 1


def test_the_spacy_label_rule_keeps_things_and_drops_quantities_and_times():
    """D5: one line, declared in advance, over the whole OntoNotes ontology."""
    assert set(config.SPACY_LABELS) & set(config.SPACY_DROPPED_LABELS) == set()
    assert len(config.SPACY_LABELS) == 11
    assert config.SPACY_DROPPED_LABELS == (
        "CARDINAL",
        "DATE",
        "MONEY",
        "ORDINAL",
        "PERCENT",
        "QUANTITY",
        "TIME",
    )
    assert len(set(config.SPACY_LABELS) | set(config.SPACY_DROPPED_LABELS)) == 18


def test_the_retention_bar_is_the_declared_share_of_the_claude_dev_gain():
    """The bar is the arithmetic, not a number typed beside it."""
    dense = config.PHASE_7_REFERENCE_DEV_FULL_SUPPORT["dense"]
    claude = config.PHASE_7_REFERENCE_DEV_FULL_SUPPORT["hybrid-entity-hop"]
    bar = dense + config.PHASE_7_RETENTION_SHARE * (claude - dense)

    assert config.PHASE_7_RETENTION_SHARE == 0.75
    assert round(bar, 4) == config.PHASE_7_RETENTION_BAR
    # The bar lands exactly on a whole number of dev questions, which is what the rule
    # is applied on; the float in the spec is that count rounded for reading.
    assert bar * config.N_DEV == config.PHASE_7_RETENTION_BAR_QUESTIONS
    assert config.PHASE_7_RETENTION_BAR_QUESTIONS == 517


def test_the_economic_bar_is_the_one_the_spec_settles():
    assert config.PHASE_7_SOFTWARE_COST_CEILING_USD == 0.0
    assert config.PHASE_7_FULLWIKI_HOURS_CEILING == 48.0
    assert config.PHASE_7_INFRASTRUCTURE_CEILING_USD == 30.0
    assert config.PROJECTION_PARAGRAPHS == 5_000_000


def test_the_inherited_figures_are_the_ones_the_phase_6_artifacts_record():
    """Data-dependent: the reference a candidate is read against is never typed twice.

    Skipped rather than failed when the Phase 6 runs are not on disk, exactly as the
    other artifact-reading tests of this project are.
    """
    import json

    directory = config.REPLACEMENT_DIR
    if not directory.is_dir():
        import pytest

        pytest.skip("the Phase 6 replacement artifacts are not on this checkout")

    for figures, files in (
        (config.PHASE_7_REFERENCE_DEV_FULL_SUPPORT, config.PHASE_7_REFERENCE_DEV_FILES),
        (config.PHASE_7_REFERENCE_HELD_OUT_FULL_SUPPORT, config.PHASE_7_REFERENCE_HELD_OUT_FILES),
    ):
        assert sorted(figures) == sorted(files) == ["dense", "hybrid-bm25", "hybrid-entity-hop"]
        for system, filename in files.items():
            path = directory / filename
            if not path.exists():
                import pytest

                pytest.skip(f"{filename} is not on this checkout")
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["system"] == system
            measured = payload["metrics"][f"budget_{config.SELECTION_BUDGET}"]["full_support"]
            assert measured == figures[system]


# --- Phase 8 (S1): the constants that decide what "Strong Dense" is ----------
#
# The Phase 8 literals, at the indentation the line length allows. Each is a public
# commit sha or the sha256 of a pipeline-written artifact; the inline pragma records
# that the secret scanner's finding on it was audited, exactly as D2's are above.
PHASE_8_LITERALS = (
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",  # pragma: allowlist secret
    "840b4f78325d5c3393588e82312b29e168bc265e42127896c588e75a422a0aba",  # pragma: allowlist secret
    "0aec5c32440ff600c1abbc7d866ad143ddf5c5263bf8252f1dbacdffc1496be5",  # pragma: allowlist secret
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",  # pragma: allowlist secret
)
PHASE_8_DENSE_REVISION_LITERAL = PHASE_8_LITERALS[0]
PHASE_8_EXTRACTION_DIGEST_LITERAL = PHASE_8_LITERALS[1]
PHASE_8_INDEX_DIGEST_LITERAL = PHASE_8_LITERALS[2]
BUDGET_TOKENIZER_REVISION_LITERAL = PHASE_8_LITERALS[3]
#
# Every value is fixed by the approved spec before the first Phase 8 number exists,
# exactly as Phase 7's were (decision 10 there, decisions 1-3 and 6 here). These
# assertions are what makes "fixed before measuring" checkable afterwards: a changed
# model, revision, dimension, prompt or bar has to break a test rather than quietly
# redefine what the phase measured.


def test_the_phase_8_directory_is_its_own_and_under_the_data_directory():
    assert config.PHASE_8_DIR == config.DATA_DIR / "phase8"
    assert config.PHASE_8_DIR not in (
        config.PHASE_7_DIR,
        config.EXTRACTION_DIR,
        config.NODES_DIR,
        config.REPLACEMENT_DIR,
        config.RESULTS_DIR,
    )


def test_ensure_directories_creates_the_phase_8_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PHASE_8_DIR", tmp_path / "data" / "phase8")
    for name in (
        "DATA_DIR",
        "CACHE_DIR",
        "RESULTS_DIR",
        "CONCEPTS_DIR",
        "SELECTION_DIR",
        "QUESTION_CACHE_DIR",
        "EXPANSION_DIR",
        "TRACES_DIR",
        "PILOT_DIR",
        "EXTRACTION_DIR",
        "NODES_DIR",
        "NAVIGATION_DIR",
        "REPLACEMENT_DIR",
        "PHASE_7_DIR",
    ):
        monkeypatch.setattr(config, name, tmp_path / "other" / name.lower())

    config.ensure_directories()

    assert config.PHASE_8_DIR.is_dir()


def test_the_dense_model_is_pinned_to_a_commit_and_to_its_full_width():
    """Decision 1: one model, one commit, 1024 dimensions and no MRL truncation."""
    assert config.PHASE_8_DENSE_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert config.PHASE_8_DENSE_REVISION == PHASE_8_DENSE_REVISION_LITERAL
    assert config.PHASE_8_DENSE_REVISION.startswith("97b0c614")
    assert len(config.PHASE_8_DENSE_REVISION) == 40
    assert all(char in "0123456789abcdef" for char in config.PHASE_8_DENSE_REVISION)
    assert config.PHASE_8_DENSE_DIM == 1024
    assert config.PHASE_8_DENSE_MODEL != config.EMBEDDING_MODEL


def test_the_query_prompt_is_the_exact_string_the_spec_settles():
    """Restriction R2: one newline, no trailing space, never tuned for this benchmark."""
    prompt = config.PHASE_8_QUERY_PROMPT

    assert prompt == (
        "Instruct: Given a web search query, retrieve relevant passages that answer "
        "the query\nQuery:"
    )
    assert prompt.count("\n") == 1
    assert prompt.endswith("Query:")
    assert prompt == prompt.rstrip()
    assert prompt.isascii()


def test_the_entity_index_is_inherited_from_phase_7_by_its_digests():
    """The phase changes the Dense retriever and nothing else: GLiNER is pinned, not rebuilt."""
    assert config.PHASE_8_GLINER_DIR == config.PHASE_7_DIR / "gliner"
    assert config.PHASE_8_GLINER_EXTRACTION_DIGEST == PHASE_8_EXTRACTION_DIGEST_LITERAL
    assert config.PHASE_8_GLINER_INDEX_DIGEST == PHASE_8_INDEX_DIGEST_LITERAL
    # The prefixes the spec quotes, so a swapped pair fails rather than passes quietly.
    assert config.PHASE_8_GLINER_EXTRACTION_DIGEST.startswith("840b4f78")
    assert config.PHASE_8_GLINER_INDEX_DIGEST.startswith("0aec5c32")
    for digest in (
        config.PHASE_8_GLINER_EXTRACTION_DIGEST,
        config.PHASE_8_GLINER_INDEX_DIGEST,
    ):
        assert len(digest) == 64
        assert all(char in "0123456789abcdef" for char in digest)


def test_the_three_systems_are_the_ones_the_spec_measures():
    assert config.PHASE_8_SYSTEMS == ("dense", "hybrid-bm25", "hybrid-entity-hop")


def test_the_inherited_figures_carry_the_gliner_row_the_phase_is_read_against():
    """Phase 8 inherits the GLiNER index, so the GLiNER row is its comparison baseline."""
    for figures in (
        config.PHASE_8_INHERITED_DEV_FULL_SUPPORT,
        config.PHASE_8_INHERITED_HELD_OUT_FULL_SUPPORT,
    ):
        assert sorted(figures) == [
            "dense",
            "hybrid-bm25",
            "hybrid-entity-hop-claude",
            "hybrid-entity-hop-gliner",
        ]
    assert sorted(config.PHASE_8_INHERITED_DEV_FILES) == sorted(
        config.PHASE_8_INHERITED_DEV_FULL_SUPPORT
    )
    assert sorted(config.PHASE_8_INHERITED_HELD_OUT_FILES) == sorted(
        config.PHASE_8_INHERITED_HELD_OUT_FULL_SUPPORT
    )
    # The Phase 6 rows are the ones Phase 7 already pinned, never typed a second time.
    dev, test = (
        config.PHASE_8_INHERITED_DEV_FULL_SUPPORT,
        config.PHASE_8_INHERITED_HELD_OUT_FULL_SUPPORT,
    )
    assert dev["dense"] == config.PHASE_7_REFERENCE_DEV_FULL_SUPPORT["dense"]
    assert dev["hybrid-bm25"] == config.PHASE_7_REFERENCE_DEV_FULL_SUPPORT["hybrid-bm25"]
    assert (
        dev["hybrid-entity-hop-claude"]
        == config.PHASE_7_REFERENCE_DEV_FULL_SUPPORT["hybrid-entity-hop"]
    )
    assert test["dense"] == config.PHASE_7_REFERENCE_HELD_OUT_FULL_SUPPORT["dense"]


def test_the_stop_rule_bar_is_the_inherited_dense_dev_reading_in_whole_questions():
    """Decision 4 and D10: 487/600 is a tie, and a tie is not 'substantially stronger'."""
    dense = config.PHASE_8_INHERITED_DEV_FULL_SUPPORT["dense"]

    assert round(dense * config.N_DEV) == config.PHASE_8_STOP_RULE_BASELINE_QUESTIONS
    assert config.PHASE_8_STOP_RULE_BASELINE_QUESTIONS == 487


def test_the_budget_tokenizer_is_declared_independently_of_the_embedding_model():
    """Restriction R1: the ruler does not move when the Dense model does.

    The two constants hold exactly the values `TOKENIZER_ID = EMBEDDING_MODEL` resolved
    to in Phases 1-7, so this is a change of meaning and not of number; what it buys is
    that repointing the Dense model can no longer move the context budget.
    """
    assert config.BUDGET_TOKENIZER_ID == "BAAI/bge-small-en-v1.5"
    assert config.BUDGET_TOKENIZER_REVISION == BUDGET_TOKENIZER_REVISION_LITERAL
    assert config.BUDGET_TOKENIZER_REVISION == config.EMBEDDING_REVISION
    assert config.TOKENIZER_ID == config.BUDGET_TOKENIZER_ID
    assert config.BUDGET_TOKENIZER_ID != config.PHASE_8_DENSE_MODEL
    assert config.BUDGET_TOKENIZER_REVISION != config.PHASE_8_DENSE_REVISION


def test_the_inherited_gliner_figures_are_the_ones_its_artifacts_record():
    """Data-dependent: the baseline this phase is read against is never typed twice."""
    import json

    import pytest

    dev_path = config.PHASE_8_GLINER_DIR / "dev.json"
    test_path = config.PHASE_7_DIR / "test.json"
    if not dev_path.exists() or not test_path.exists():
        pytest.skip("the Phase 7 GLiNER artifacts are not on this checkout")

    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    budget = f"budget_{config.SELECTION_BUDGET}"

    assert dev["index"]["digest"] == config.PHASE_8_GLINER_INDEX_DIGEST
    assert dev["extraction"]["digest"] == config.PHASE_8_GLINER_EXTRACTION_DIGEST
    assert (
        dev["metrics"][budget]["full_support"]
        == config.PHASE_8_INHERITED_DEV_FULL_SUPPORT["hybrid-entity-hop-gliner"]
    )
    assert (
        test["metrics"][budget]["full_support"]
        == config.PHASE_8_INHERITED_HELD_OUT_FULL_SUPPORT["hybrid-entity-hop-gliner"]
    )
    for figures, files in (
        (config.PHASE_8_INHERITED_DEV_FULL_SUPPORT, config.PHASE_8_INHERITED_DEV_FILES),
        (config.PHASE_8_INHERITED_HELD_OUT_FULL_SUPPORT, config.PHASE_8_INHERITED_HELD_OUT_FILES),
    ):
        for system, relative in files.items():
            path = config.DATA_DIR / relative
            if not path.exists():
                pytest.skip(f"{relative} is not on this checkout")
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["metrics"][budget]["full_support"] == figures[system]


# --- Deviation 8.1: the constants that decide what the scale diagnostic is ----
#
# Every threshold, ceiling, size and required count below is fixed before the first
# 8.1 number exists, for the reason the Phase 7 and Phase 8 blocks above are: a rule
# adjusted after seeing a scale result is not a rule. These assertions are what make
# "fixed before measuring" checkable afterwards.
#
# The published MD5 is a **claim** copied from the deviation's recollection of the
# release page, not a measurement. S1 reads the live page and the downloaded bytes and
# corrects it in place if they differ; until then the artifact records both.
PHASE_8_1_DECLARED_MD5_LITERAL = "01edf64cd120ecc03a2745352779514c"  # pragma: allowlist secret


def test_the_deviation_8_1_directory_is_its_own_and_under_the_data_directory():
    assert config.PHASE_8_1_DIR == config.DATA_DIR / "phase8_1"
    assert config.PHASE_8_1_DIR != config.PHASE_8_DIR
    assert config.PHASE_8_1_CACHE_DIR == config.PHASE_8_1_DIR / "cache"
    assert config.PHASE_8_1_QUESTION_CACHE_DIR == config.PHASE_8_1_CACHE_DIR / "questions"
    # The whole point of the separate directories: `question_cache_key` does not depend
    # on the corpus, so the 600 dev questions would hash onto the historical file.
    assert config.PHASE_8_1_CACHE_DIR != config.CACHE_DIR
    assert config.PHASE_8_1_QUESTION_CACHE_DIR != config.QUESTION_CACHE_DIR


def test_the_official_source_is_declared_with_its_published_provenance():
    assert config.PHASE_8_1_SOURCE_URL.startswith("https://")
    assert config.PHASE_8_1_SOURCE_ARCHIVE.endswith(".tar.bz2")
    assert config.PHASE_8_1_SOURCE_LICENSE
    assert config.PHASE_8_1_DECLARED_BYTES == 1_553_565_403
    assert config.PHASE_8_1_DECLARED_MD5 == PHASE_8_1_DECLARED_MD5_LITERAL
    assert len(config.PHASE_8_1_DECLARED_MD5) == 32
    assert all(character in "0123456789abcdef" for character in config.PHASE_8_1_DECLARED_MD5)
    # S1 verified the declared identity against staged bytes, not the live source page.
    basis = config.PHASE_8_1_DECLARED_BASIS.lower()
    assert "verified against the staged archive bytes" in basis
    assert "live hotpotqa page was not checked" in basis


def test_the_four_corpus_sizes_and_their_prefixes_are_exactly_the_declared_arithmetic():
    assert config.PHASE_8_1_CORPUS_SIZES == (19366, 100000, 250000, 500000)
    assert config.PHASE_8_1_CORPUS_LABELS == ("c19", "c100", "c250", "c500")
    assert config.PHASE_8_1_DISTRACTOR_PREFIXES == (80634, 230634, 480634)
    assert config.PHASE_8_1_CORPUS_SIZES[0] == config.EXPECTED_N_UNITS
    # C100 = C19 + 80,634, and so on: the prefixes are the sizes minus the C19 block.
    for size, prefix in zip(
        config.PHASE_8_1_CORPUS_SIZES[1:], config.PHASE_8_1_DISTRACTOR_PREFIXES, strict=True
    ):
        assert config.PHASE_8_1_CORPUS_SIZES[0] + prefix == size
    assert len(config.PHASE_8_1_CORPUS_SIZES) == len(config.PHASE_8_1_CORPUS_LABELS)
    prefixes = config.PHASE_8_1_DISTRACTOR_PREFIXES
    assert tuple(sorted(prefixes)) == prefixes


def test_the_ordering_rule_declares_42_as_a_hash_salt_and_not_as_rng_state():
    assert config.PHASE_8_1_ORDER_SALT == "42"
    assert config.PHASE_8_1_ORDER_RULE == "sha256-salted-title-plaintext-v1"
    assert "salt" in config.PHASE_8_1_ORDER_SALT_BASIS.lower()
    assert "rng" in config.PHASE_8_1_ORDER_SALT_BASIS.lower()
    assert str(config.DEFAULT_SEED) == config.PHASE_8_1_ORDER_SALT


def test_the_reconciliation_ceiling_is_fifty_units():
    assert config.PHASE_8_1_UNMATCHED_CEILING == 50
    # 0.26% of the frozen pool: large enough for source drift, small enough to be safe.
    assert config.PHASE_8_1_UNMATCHED_CEILING / config.EXPECTED_N_UNITS < 0.003


def test_the_two_reproduction_counts_are_the_historical_readings_keyed_by_model():
    """Keyed by model, never by split: `config` is on the Phase 3 selection's closure."""
    assert config.PHASE_8_1_C19_DEV_SUPPORTED == {"bge": 487, "qwen": 446}
    assert sorted(config.PHASE_8_1_C19_DEV_SUPPORTED) == sorted(config.PHASE_8_1_MODELS)
    assert config.PHASE_8_1_MODELS == ("bge", "qwen")
    assert config.PHASE_8_1_REENCODE_TOLERANCE == 1
    # The BGE reading is the same 487 Phase 8's stop rule was read against.
    assert config.PHASE_8_1_C19_DEV_SUPPORTED["bge"] == config.PHASE_8_STOP_RULE_BASELINE_QUESTIONS
    assert round(config.PHASE_8_1_C19_DEV_SUPPORTED["bge"] / config.N_DEV, 4) == 0.8117


def test_the_outcome_thresholds_are_the_declared_arithmetic_on_the_c19_deficit():
    assert config.PHASE_8_1_CONVERGENCE_QUESTIONS == 20
    assert config.PHASE_8_1_BGE_RETENTION_FLOOR == 244
    deficit = config.PHASE_8_1_C19_DEV_SUPPORTED["qwen"] - config.PHASE_8_1_C19_DEV_SUPPORTED["bge"]
    assert deficit == -41
    # "at least half the initial deficit removed", on a 41-question deficit.
    assert abs(deficit) // 2 >= config.PHASE_8_1_CONVERGENCE_QUESTIONS
    # The floor is half BGE's own C19 reading, rounded up.
    half_of_bge = -(-config.PHASE_8_1_C19_DEV_SUPPORTED["bge"] // 2)
    assert half_of_bge == config.PHASE_8_1_BGE_RETENTION_FLOOR


def test_the_four_outcomes_and_two_stops_are_the_only_terminal_labels():
    assert config.PHASE_8_1_OUTCOMES == (
        "crossover",
        "convergence",
        "both_degrade",
        "stable_ranking",
    )
    assert config.PHASE_8_1_STOPS == ("data_stop", "reproduction_stop")
    assert not set(config.PHASE_8_1_OUTCOMES) & set(config.PHASE_8_1_STOPS)


def test_the_historical_cache_keys_are_the_ones_the_recorded_counts_were_read_from():
    assert config.PHASE_8_1_HISTORICAL_CORPUS_KEYS == {
        "bge": "bacd74e7f468f0d4",
        "qwen": "66ef5c19a95f6cbe",
    }
    assert config.PHASE_8_1_HISTORICAL_DEV_QUERY_KEYS == {
        "bge": "35329d9e75da1d4f",
        "qwen": "c9c589c29c226fb3",
    }
    for keys in (
        config.PHASE_8_1_HISTORICAL_CORPUS_KEYS,
        config.PHASE_8_1_HISTORICAL_DEV_QUERY_KEYS,
    ):
        assert sorted(keys) == sorted(config.PHASE_8_1_MODELS)
        for key in keys.values():
            assert len(key) == 16
            assert all(character in "0123456789abcdef" for character in key)
    # `a28365b7ed90ed10` holds the same vectors under revision "main" and is a
    # migration artifact; the revision-pinned key is the one 8.1 reads.
    assert config.PHASE_8_1_HISTORICAL_CORPUS_KEYS["bge"] != "a28365b7ed90ed10"


def test_the_archive_ceilings_are_finite_and_ordered():
    assert config.PHASE_8_1_MAX_MEMBER_BYTES > 0
    assert config.PHASE_8_1_MAX_LINE_BYTES > 0
    assert config.PHASE_8_1_MAX_TOTAL_RECORDS > 0
    # A line cannot be larger than the member that holds it.
    assert config.PHASE_8_1_MAX_LINE_BYTES < config.PHASE_8_1_MAX_MEMBER_BYTES
    # Room above the 5M paragraphs the official release holds.
    assert config.PHASE_8_1_MAX_TOTAL_RECORDS > config.PROJECTION_PARAGRAPHS
    assert config.PHASE_8_1_PROBE_MEMBERS > 0
    assert config.PHASE_8_1_TOKENIZE_BATCH > 0
    # Batching is the point: a batch that held every distractor would not be one.
    assert config.PHASE_8_1_DISTRACTOR_PREFIXES[-1] > config.PHASE_8_1_TOKENIZE_BATCH
    assert config.PHASE_8_1_QUERY_SECONDS_CEILING > 0.0


def test_the_schema_candidates_are_ordered_preferences_and_not_a_single_assumption():
    """S1 records the schema it observed; the parser dispatches on the record.

    The candidate lists exist so a probe can *resolve* a field name against what the
    archive actually holds and refuse when none of them is there - not so the code can
    assume one.
    """
    for candidates in (
        config.PHASE_8_1_TITLE_FIELDS,
        config.PHASE_8_1_SENTENCES_FIELDS,
        config.PHASE_8_1_PAGE_ID_FIELDS,
    ):
        assert len(candidates) >= 1
        assert len(set(candidates)) == len(candidates)
    assert config.PHASE_8_1_TITLE_FIELDS[0] == "title"
    assert config.PHASE_8_1_SENTENCES_FIELDS[0] == "text"


def test_the_deviation_8_1_directory_is_created_by_ensure_directories():
    config.ensure_directories()
    assert config.PHASE_8_1_DIR.is_dir()
    assert config.PHASE_8_1_CACHE_DIR.is_dir()
    assert config.PHASE_8_1_QUESTION_CACHE_DIR.is_dir()


def test_the_historical_bge_reproduction_count_is_the_one_its_run_file_records():
    """Data-dependent: the count the gate requires is never a number typed from memory."""
    import json

    import pytest

    path = config.DATA_DIR / config.PHASE_8_INHERITED_DEV_FILES["dense"]
    if not path.exists():
        pytest.skip(f"{path.name} is not on this checkout")
    payload = json.loads(path.read_text(encoding="utf-8"))
    share = payload["metrics"][f"budget_{config.SELECTION_BUDGET}"]["full_support"]

    assert round(share * config.N_DEV) == config.PHASE_8_1_C19_DEV_SUPPORTED["bge"]


def test_the_historical_qwen_reproduction_count_is_the_one_phase_8_recorded():
    """Same rule for the Qwen side; `data/phase8/dev.json` is untracked, hence the skip."""
    import json

    import pytest

    path = config.PHASE_8_DIR / "dev.json"
    if not path.exists():
        pytest.skip("data/phase8/dev.json is not on this checkout")
    payload = json.loads(path.read_text(encoding="utf-8"))
    dense = payload["systems"]["dense"]

    assert dense["supported_questions"] == config.PHASE_8_1_C19_DEV_SUPPORTED["qwen"]
    assert dense["config"]["corpus_cache_key"] == config.PHASE_8_1_HISTORICAL_CORPUS_KEYS["qwen"]
    assert (
        dense["config"]["question_cache_key"] == config.PHASE_8_1_HISTORICAL_DEV_QUERY_KEYS["qwen"]
    )
