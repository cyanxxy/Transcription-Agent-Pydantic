import pytest
from types import SimpleNamespace

from dependencies import (
    TranscriptionDeps,
    EditingDeps,
    QualityDeps,
    ExportDeps,
    AppDeps,
    SUPPORTED_GEMINI_MODELS,
    resolve_dual_gemini_secondary_model,
)
from streamlit_deps import build_app_deps_from_streamlit


# --- TranscriptionDeps validation ---


def test_chunk_overlap_validation() -> None:
    with pytest.raises(ValueError):
        TranscriptionDeps(
            api_key="test-key",
            chunk_duration_ms=1000,
            chunk_overlap_ms=1000,
        )


def test_pro_rejects_unknown_thinking_level() -> None:
    with pytest.raises(ValueError):
        TranscriptionDeps(
            api_key="test-key",
            model_name="gemini-3.1-pro-preview",
            transcription_thinking_level="ultra",
            judge_thinking_level="low",
        )


def test_valid_flash_defaults() -> None:
    deps = TranscriptionDeps(api_key="test-key")
    assert deps.model_name == "gemini-3-flash-preview"
    assert deps.judge_model_name == "gemini-3.1-pro-preview"
    assert deps.candidate_strategy == "dual_gemini"
    assert deps.transcription_thinking_level == "high"
    assert deps.judge_thinking_level == "high"
    assert deps.thinking_level == "high"
    assert deps.chunk_duration_ms == 120000
    assert deps.chunk_overlap_ms == 5000
    assert deps.max_file_size_mb == 200
    assert deps.use_judge_pipeline is True
    assert "gemini-3.1-flash-lite-preview" in SUPPORTED_GEMINI_MODELS


def test_valid_pro_creation() -> None:
    deps = TranscriptionDeps(
        api_key="test-key",
        model_name="gemini-3.1-pro-preview",
        transcription_thinking_level="low",
        judge_thinking_level="high",
    )
    assert deps.model_name == "gemini-3.1-pro-preview"
    assert deps.transcription_thinking_level == "low"
    assert deps.judge_thinking_level == "high"


def test_valid_flash_lite_creation() -> None:
    deps = TranscriptionDeps(
        api_key="test-key",
        model_name="gemini-3.1-flash-lite-preview",
        judge_model_name="gemini-3.1-flash-lite-preview",
        transcription_thinking_level="minimal",
        judge_thinking_level="medium",
    )
    assert deps.model_name == "gemini-3.1-flash-lite-preview"
    assert deps.transcription_thinking_level == "minimal"
    assert deps.judge_thinking_level == "medium"


def test_pro_accepts_low_and_high_thinking_levels() -> None:
    deps = TranscriptionDeps(
        api_key="test-key",
        model_name="gemini-3.1-pro-preview",
        transcription_thinking_level="high",
        judge_thinking_level="low",
    )
    assert deps.transcription_thinking_level == "high"
    assert deps.judge_thinking_level == "low"


def test_pro_normalizes_legacy_thinking_levels_to_high() -> None:
    deps = TranscriptionDeps(
        api_key="test-key",
        model_name="gemini-3.1-pro-preview",
        judge_model_name="gemini-3.1-pro-preview",
        transcription_thinking_level="medium",
        judge_thinking_level="minimal",
    )
    assert deps.transcription_thinking_level == "high"
    assert deps.judge_thinking_level == "high"


def test_normalizes_deprecated_gemini_3_pro_name() -> None:
    deps = TranscriptionDeps(
        api_key="test-key",
        model_name="google-gla:gemini-3-pro-preview",
        judge_model_name="gemini-3-pro-preview",
    )
    assert deps.model_name == "gemini-3.1-pro-preview"
    assert deps.judge_model_name == "gemini-3.1-pro-preview"
    deps.cleanup()


def test_strips_google_gla_prefix() -> None:
    deps = TranscriptionDeps(
        api_key="test-key",
        model_name="google-gla:gemini-3-flash-preview",
    )
    assert deps.model_name == "gemini-3-flash-preview"


def test_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        TranscriptionDeps(api_key="test-key", model_name="gpt-4")


def test_invalid_thinking_level() -> None:
    with pytest.raises(ValueError, match="Invalid transcription_thinking_level"):
        TranscriptionDeps(
            api_key="test-key",
            transcription_thinking_level="ultra",
            judge_thinking_level="medium",
        )


def test_invalid_candidate_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported candidate_strategy"):
        TranscriptionDeps(api_key="test-key", candidate_strategy="triple_gemini")


def test_negative_chunk_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap_ms must be >= 0"):
        TranscriptionDeps(api_key="test-key", chunk_overlap_ms=-1)


def test_zero_chunk_duration() -> None:
    with pytest.raises(ValueError, match="chunk_duration_ms must be > 0"):
        TranscriptionDeps(api_key="test-key", chunk_duration_ms=0)


def test_chunk_size_bytes() -> None:
    deps = TranscriptionDeps(api_key="test-key", chunk_duration_ms=60000)
    # 1 minute = 1 MB
    assert deps.chunk_size_bytes == 1024 * 1024


def test_temp_dir_created() -> None:
    import os

    deps = TranscriptionDeps(api_key="test-key")
    assert os.path.isdir(deps.temp_dir)
    deps.cleanup()


def test_resolve_candidate_specs_for_dual_gemini() -> None:
    deps = TranscriptionDeps(api_key="test-key", model_name="gemini-3-flash-preview")
    specs = deps.resolve_candidate_specs()
    assert len(specs) == 2
    assert specs[0]["kind"] == "gemini"
    assert specs[1]["model_name"] == "gemini-3.1-flash-lite-preview"
    assert (
        resolve_dual_gemini_secondary_model("gemini-3-flash-preview")
        == "gemini-3.1-flash-lite-preview"
    )
    deps.cleanup()


def test_resolve_candidate_specs_for_flash_lite_dual_gemini() -> None:
    deps = TranscriptionDeps(
        api_key="test-key",
        model_name="gemini-3.1-flash-lite-preview",
    )
    specs = deps.resolve_candidate_specs()
    assert len(specs) == 2
    assert specs[1]["model_name"] == "gemini-3-flash-preview"
    deps.cleanup()


def test_resolve_candidate_specs_for_parakeet_strategy() -> None:
    deps = TranscriptionDeps(
        api_key="test-key",
        candidate_strategy="gemini_plus_parakeet",
    )
    specs = deps.resolve_candidate_specs()
    assert len(specs) == 2
    assert specs[1]["kind"] == "parakeet"
    assert specs[1]["model_name"] == deps.parakeet_model
    deps.cleanup()


# --- EditingDeps ---


def test_editing_deps_defaults() -> None:
    deps = EditingDeps()
    assert deps.enable_auto_correct is True
    assert deps.remove_fillers is False
    assert isinstance(deps.filler_words, list)
    assert "um" in deps.filler_words
    assert isinstance(deps.replacements, dict)
    assert deps.replacements["gonna"] == "going to"


# --- QualityDeps ---


def test_quality_deps_defaults() -> None:
    deps = QualityDeps()
    assert deps.target_readability_score == 70.0
    assert deps.min_vocabulary_richness == 30.0
    assert "readability" in deps.weights
    assert abs(sum(deps.weights.values()) - 1.0) < 0.01


# --- ExportDeps ---


def test_export_deps_defaults() -> None:
    deps = ExportDeps()
    assert deps.srt_max_line_length == 42
    assert deps.json_pretty_print is True


# --- AppDeps ---


def test_app_deps_from_config() -> None:
    deps = AppDeps.from_config(
        api_key="test-key",
        model_name="gemini-3.1-flash-lite-preview",
        judge_model_name="gemini-3.1-pro-preview",
        candidate_strategy="single_gemini",
        auto_format=True,
        remove_fillers=True,
        transcription_thinking_level="low",
        judge_thinking_level="medium",
    )
    assert deps.transcription.api_key == "test-key"
    assert deps.transcription.candidate_strategy == "single_gemini"
    assert deps.transcription.auto_format is True
    assert deps.editing.remove_fillers is True
    assert deps.transcription.model_name == "gemini-3.1-flash-lite-preview"
    assert deps.transcription.transcription_thinking_level == "low"
    assert deps.transcription.judge_thinking_level == "high"
    deps.cleanup()


def test_app_deps_from_config_accepts_legacy_thinking_level_alias() -> None:
    deps = AppDeps.from_config(
        api_key="test-key",
        thinking_level="low",
    )
    assert deps.transcription.transcription_thinking_level == "low"
    assert deps.transcription.judge_thinking_level == "high"
    deps.cleanup()


def test_app_deps_from_config_with_judge_pipeline() -> None:
    deps = AppDeps.from_config(
        api_key="test-key",
        use_judge_pipeline=False,
    )
    assert deps.transcription.use_judge_pipeline is False
    deps.cleanup()


def test_app_deps_from_config_accepts_legacy_orchestrator_flag() -> None:
    deps = AppDeps.from_config(
        api_key="test-key",
        use_orchestrator=False,
    )
    assert deps.transcription.use_judge_pipeline is False
    deps.cleanup()


def test_build_app_deps_from_streamlit_reads_typed_app_state() -> None:
    session_state = SimpleNamespace(
        api_key="test-key",
        model_name="gemini-3.1-pro-preview",
        judge_model_name="gemini-3-flash-preview",
        candidate_strategy="single_gemini",
        use_judge_pipeline=False,
        auto_format=False,
        remove_fillers=True,
        transcription_thinking_level="medium",
        judge_thinking_level="low",
    )
    st_module = SimpleNamespace(
        session_state=session_state,
        secrets={},
    )

    deps = build_app_deps_from_streamlit(st_module)

    assert deps is not None
    assert deps.transcription.model_name == "gemini-3.1-pro-preview"
    assert deps.transcription.judge_model_name == "gemini-3-flash-preview"
    assert deps.transcription.candidate_strategy == "single_gemini"
    assert deps.transcription.use_judge_pipeline is False
    assert deps.transcription.auto_format is False
    assert deps.transcription.remove_fillers is True
    assert deps.transcription.transcription_thinking_level == "high"
    assert deps.transcription.judge_thinking_level == "low"
    deps.cleanup()


def test_build_app_deps_from_streamlit_returns_none_without_streamlit(
    monkeypatch,
) -> None:
    def raise_import_error(name: str):
        if name == "streamlit":
            raise ImportError("streamlit is not installed")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("streamlit_deps.importlib.import_module", raise_import_error)

    assert build_app_deps_from_streamlit() is None
