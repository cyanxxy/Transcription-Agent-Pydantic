import pytest

from dependencies import TranscriptionDeps


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
