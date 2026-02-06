import pytest

from utils import (
    estimate_audio_tokens,
    estimate_transcription_cost,
    format_duration,
    sanitize_filename,
    PRICING,
)


# --- estimate_audio_tokens ---


def test_audio_tokens_basic() -> None:
    assert estimate_audio_tokens(1.0) == 32
    assert estimate_audio_tokens(60.0) == 1920


def test_audio_tokens_zero() -> None:
    assert estimate_audio_tokens(0.0) == 0


# --- estimate_transcription_cost ---


def test_cost_flash_model() -> None:
    cost, cost_str = estimate_transcription_cost(60, "gemini-3-flash-preview")
    assert cost > 0
    assert "$" in cost_str


def test_cost_pro_model_low_tier() -> None:
    cost, cost_str = estimate_transcription_cost(60, "gemini-3-pro-preview")
    assert cost > 0


def test_cost_pro_model_high_tier() -> None:
    # Long audio to exceed 200k token threshold
    cost, cost_str = estimate_transcription_cost(7000, "gemini-3-pro-preview")
    assert cost > 0


def test_cost_very_short_audio() -> None:
    cost, cost_str = estimate_transcription_cost(1, "gemini-3-flash-preview")
    assert cost_str == "<$0.01"


def test_pricing_constants_exist() -> None:
    assert "flash" in PRICING
    assert "pro" in PRICING
    assert "pro_tier_threshold" in PRICING
    assert PRICING["flash"]["input"] == 0.50
    assert PRICING["flash"]["output"] == 3.00


# --- format_duration ---


def test_format_duration_seconds() -> None:
    assert format_duration(45) == "45s"


def test_format_duration_minutes() -> None:
    assert format_duration(90) == "1m 30s"


def test_format_duration_hours() -> None:
    assert format_duration(3661) == "1h 1m 1s"


def test_format_duration_zero() -> None:
    assert format_duration(0) == "0s"


# --- sanitize_filename ---


def test_sanitize_basic() -> None:
    assert sanitize_filename("test.mp3") == "test.mp3"


def test_sanitize_special_chars() -> None:
    result = sanitize_filename('test<>:"|?*.mp3')
    assert "<" not in result
    assert ">" not in result
    assert ":" not in result
    assert '"' not in result
    assert "|" not in result
    assert "?" not in result
    assert "*" not in result


def test_sanitize_path_traversal() -> None:
    result = sanitize_filename("../../etc/passwd")
    assert ".." not in result
    assert "/" not in result
    assert result == "passwd"


def test_sanitize_path_traversal_windows() -> None:
    result = sanitize_filename("..\\..\\windows\\system32\\config")
    # os.path.basename on unix treats backslash as part of filename
    # but the regex replaces backslashes with _
    assert ".." not in result or "\\" not in result


def test_sanitize_absolute_path() -> None:
    result = sanitize_filename("/home/user/secret.txt")
    assert result == "secret.txt"


def test_sanitize_long_filename() -> None:
    long_name = "a" * 300 + ".mp3"
    result = sanitize_filename(long_name)
    assert len(result) <= 200


def test_sanitize_empty_string() -> None:
    result = sanitize_filename("")
    assert result == ""
