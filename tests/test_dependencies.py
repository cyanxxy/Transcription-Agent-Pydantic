import pytest

from dependencies import (
    TranscriptionDeps,
    EditingDeps,
    QualityDeps,
    ExportDeps,
    AppDeps,
)


# --- TranscriptionDeps validation ---


def test_chunk_overlap_validation() -> None:
    with pytest.raises(ValueError):
        TranscriptionDeps(
            api_key="test-key",
            chunk_duration_ms=1000,
            chunk_overlap_ms=1000,
        )


def test_thinking_level_validation_for_pro() -> None:
    with pytest.raises(ValueError):
        TranscriptionDeps(
            api_key="test-key",
            model_name="gemini-3-pro-preview",
            thinking_level="medium",
        )


def test_valid_flash_defaults() -> None:
    deps = TranscriptionDeps(api_key="test-key")
    assert deps.model_name == "gemini-3-flash-preview"
    assert deps.thinking_level == "high"
    assert deps.chunk_duration_ms == 120000
    assert deps.chunk_overlap_ms == 5000
    assert deps.max_file_size_mb == 200
    assert deps.use_orchestrator is True


def test_valid_pro_creation() -> None:
    deps = TranscriptionDeps(
        api_key="test-key",
        model_name="gemini-3-pro-preview",
        thinking_level="high",
    )
    assert deps.model_name == "gemini-3-pro-preview"
    assert deps.thinking_level == "high"


def test_pro_low_thinking_level() -> None:
    deps = TranscriptionDeps(
        api_key="test-key",
        model_name="gemini-3-pro-preview",
        thinking_level="low",
    )
    assert deps.thinking_level == "low"


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
    with pytest.raises(ValueError, match="Invalid thinking_level"):
        TranscriptionDeps(api_key="test-key", thinking_level="ultra")


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
        model_name="gemini-3-flash-preview",
        auto_format=True,
        remove_fillers=True,
    )
    assert deps.transcription.api_key == "test-key"
    assert deps.transcription.auto_format is True
    assert deps.editing.remove_fillers is True
    deps.cleanup()


def test_app_deps_from_config_with_orchestrator() -> None:
    deps = AppDeps.from_config(
        api_key="test-key",
        use_orchestrator=False,
    )
    assert deps.transcription.use_orchestrator is False
    deps.cleanup()
