from types import SimpleNamespace
from typing import cast

from models import TranscriptSegment
from dependencies import QualityDeps
from agents.quality_validator import (
    calculate_quality_metrics,
    calculate_overall_score,
    calculate_timestamp_coverage,
    calculate_speaker_consistency,
    parse_timestamp_to_seconds,
)


def _seg(ts: str, speaker: str, text: str) -> TranscriptSegment:
    return TranscriptSegment(timestamp=ts, speaker=speaker, text=text)


# --- calculate_quality_metrics ---


def test_metrics_empty_segments() -> None:
    deps = QualityDeps()
    metrics = calculate_quality_metrics(deps, [])
    assert metrics["readability"] == 0
    assert metrics["warnings"] == ["No segments to analyze"]


def test_metrics_single_segment() -> None:
    deps = QualityDeps()
    segments = [_seg("[00:00:00]", "Alice", "Hello, this is a test sentence.")]
    metrics = calculate_quality_metrics(deps, segments)
    assert 0 <= metrics["readability"] <= 100
    assert 0 <= metrics["punctuation_density"] <= 1
    assert metrics["timestamp_coverage"] == 100.0
    assert isinstance(metrics["warnings"], list)


def test_metrics_multiple_segments() -> None:
    deps = QualityDeps()
    segments = [
        _seg("[00:00:00]", "Alice", "Hello, how are you doing today?"),
        _seg("[00:00:05]", "Bob", "I am doing well, thank you for asking."),
        _seg("[00:00:10]", "Alice", "That is great to hear, let us continue."),
        _seg("[00:00:15]", "Bob", "Sure, what would you like to discuss?"),
    ]
    metrics = calculate_quality_metrics(deps, segments)
    assert metrics["readability"] > 0
    assert metrics["vocabulary_richness"] > 0
    assert metrics["speaker_consistency"] > 0


def test_metrics_warnings_for_low_readability() -> None:
    deps = QualityDeps(target_readability_score=99.0)
    segments = [
        _seg("[00:00:00]", "Alice", "Hello, this is a moderately long test sentence that contains several words."),
    ]
    metrics = calculate_quality_metrics(deps, segments)
    # Should warn about low readability since threshold is very high
    assert any("readability" in w.lower() for w in metrics["warnings"])


def test_metrics_return_type_includes_warnings_list() -> None:
    """Verify the return type fix: warnings is List[str], not float"""
    deps = QualityDeps()
    segments = [_seg("[00:00:00]", "Alice", "Hello world.")]
    metrics = calculate_quality_metrics(deps, segments)
    assert isinstance(metrics["warnings"], list)
    for w in metrics["warnings"]:
        assert isinstance(w, str)


# --- calculate_overall_score ---


def test_overall_score_basic() -> None:
    deps = QualityDeps()
    metrics = {
        "readability": 80,
        "vocabulary_richness": 70,
        "sentence_variety": 60,
        "punctuation_density": 0.05,
        "speaker_consistency": 90,
    }
    score = calculate_overall_score(deps, metrics)
    assert 0 <= score <= 100


def test_overall_score_perfect() -> None:
    deps = QualityDeps()
    metrics = {
        "readability": 100,
        "vocabulary_richness": 100,
        "sentence_variety": 100,
        "punctuation_density": 0.0,
        "speaker_consistency": 100,
    }
    score = calculate_overall_score(deps, metrics)
    assert score == 100.0


def test_overall_score_empty_metrics() -> None:
    deps = QualityDeps()
    score = calculate_overall_score(deps, {})
    assert score == 50.0  # default middle score


def test_overall_score_high_punctuation_penalty() -> None:
    deps = QualityDeps()
    low_punct = {
        "readability": 80,
        "vocabulary_richness": 70,
        "sentence_variety": 60,
        "punctuation_density": 0.01,
        "speaker_consistency": 90,
    }
    high_punct = {
        "readability": 80,
        "vocabulary_richness": 70,
        "sentence_variety": 60,
        "punctuation_density": 0.5,
        "speaker_consistency": 90,
    }
    score_low = calculate_overall_score(deps, low_punct)
    score_high = calculate_overall_score(deps, high_punct)
    assert score_low > score_high


# --- calculate_timestamp_coverage ---


def test_timestamp_coverage_all_valid() -> None:
    segments = [
        _seg("[00:00:00]", "A", "Hello"),
        _seg("[00:00:05]", "A", "World"),
    ]
    assert calculate_timestamp_coverage(segments) == 100.0


def test_timestamp_coverage_without_duration_preserves_validity_ratio() -> None:
    segments = [
        _seg("[00:00:00]", "A", "Hello"),
        cast(
            TranscriptSegment,
            SimpleNamespace(timestamp="invalid", speaker="A", text="World"),
        ),
    ]
    assert calculate_timestamp_coverage(segments) == 50.0


def test_timestamp_coverage_empty() -> None:
    assert calculate_timestamp_coverage([]) == 0.0


def test_timestamp_coverage_with_duration_repeated_zero_not_full() -> None:
    segments = [
        _seg("[00:00:00]", "A", "Hello"),
        _seg("[00:00:00]", "A", "World"),
    ]
    assert calculate_timestamp_coverage(segments, audio_duration=60.0) == 0.0


def test_timestamp_coverage_single_segment_at_zero_is_full_when_duration_known() -> None:
    segments = [_seg("[00:00:00]", "A", "Hello world")]
    assert calculate_timestamp_coverage(segments, audio_duration=60.0) == 100.0


# --- calculate_speaker_consistency ---


def test_speaker_consistency_single_speaker() -> None:
    segments = [
        _seg("[00:00:00]", "Alice", "Hello"),
        _seg("[00:00:05]", "Alice", "World"),
    ]
    score = calculate_speaker_consistency(segments)
    assert score == 100.0  # Single speaker is perfectly consistent


def test_speaker_consistency_balanced() -> None:
    segments = [
        _seg("[00:00:00]", "Alice", "Hello"),
        _seg("[00:00:05]", "Bob", "Hi"),
        _seg("[00:00:10]", "Alice", "Bye"),
        _seg("[00:00:15]", "Bob", "See ya"),
    ]
    score = calculate_speaker_consistency(segments)
    assert score > 80  # Balanced 2 speakers should score well


def test_speaker_consistency_many_speakers() -> None:
    segments = [
        _seg("[00:00:00]", f"Speaker {i}", "Text")
        for i in range(15)
    ]
    score = calculate_speaker_consistency(segments)
    assert score < 80  # Too many speakers should lower consistency


def test_speaker_consistency_empty() -> None:
    assert calculate_speaker_consistency([]) == 0.0


# --- parse_timestamp_to_seconds ---


def test_parse_timestamp_valid() -> None:
    assert parse_timestamp_to_seconds("[00:00:00]") == 0.0
    assert parse_timestamp_to_seconds("[00:01:30]") == 90.0
    assert parse_timestamp_to_seconds("[01:00:00]") == 3600.0
    assert parse_timestamp_to_seconds("[01:30:45]") == 5445.0


def test_parse_timestamp_invalid() -> None:
    assert parse_timestamp_to_seconds("invalid") == 0.0
    assert parse_timestamp_to_seconds("00:00:00") == 0.0  # missing brackets
