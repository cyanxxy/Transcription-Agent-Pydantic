from utils import (
    estimate_audio_tokens,
    estimate_judge_pipeline_cost,
    estimate_transcription_cost,
    format_duration,
    sanitize_filename,
    PRICING,
)
from dependencies import resolve_dual_gemini_secondary_model


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


def test_cost_flash_lite_model() -> None:
    cost, cost_str = estimate_transcription_cost(60, "gemini-3.1-flash-lite-preview")
    assert cost > 0
    assert "$" in cost_str


def test_cost_pro_model_low_tier() -> None:
    cost, cost_str = estimate_transcription_cost(60, "gemini-3.1-pro-preview")
    assert cost > 0


def test_cost_pro_model_high_tier() -> None:
    # Long audio to exceed 200k token threshold
    cost, cost_str = estimate_transcription_cost(7000, "gemini-3.1-pro-preview")
    assert cost > 0


def test_cost_very_short_audio() -> None:
    cost, cost_str = estimate_transcription_cost(1, "gemini-3-flash-preview")
    assert cost_str == "<$0.01"


def test_judge_pipeline_cost_matches_direct_mode_when_disabled() -> None:
    direct_cost, _ = estimate_transcription_cost(180, "gemini-3-flash-preview")
    pipeline_cost, _ = estimate_judge_pipeline_cost(
        180,
        "gemini-3-flash-preview",
        "dual_gemini",
        False,
    )
    assert pipeline_cost == direct_cost


def test_judge_pipeline_cost_dual_gemini_exceeds_single_gemini() -> None:
    single_cost, _ = estimate_judge_pipeline_cost(
        180,
        "gemini-3-flash-preview",
        "single_gemini",
        True,
    )
    dual_cost, dual_cost_str = estimate_judge_pipeline_cost(
        180,
        "gemini-3-flash-preview",
        "dual_gemini",
        True,
    )
    assert dual_cost > single_cost
    assert "$" in dual_cost_str


def test_dual_gemini_secondary_mapping_prefers_flash_lite_for_flash() -> None:
    assert resolve_dual_gemini_secondary_model("gemini-3-flash-preview") == "gemini-3.1-flash-lite-preview"
    assert resolve_dual_gemini_secondary_model("gemini-3.1-flash-lite-preview") == "gemini-3-flash-preview"
    assert resolve_dual_gemini_secondary_model("gemini-3.1-pro-preview") == "gemini-3-flash-preview"


def test_judge_pipeline_cost_parakeet_strategy_is_cheaper_than_dual_gemini() -> None:
    parakeet_cost, _ = estimate_judge_pipeline_cost(
        180,
        "gemini-3-flash-preview",
        "gemini_plus_parakeet",
        True,
    )
    dual_cost, _ = estimate_judge_pipeline_cost(
        180,
        "gemini-3-flash-preview",
        "dual_gemini",
        True,
    )
    assert parakeet_cost < dual_cost


def test_pricing_constants_exist() -> None:
    assert "flash" in PRICING
    assert "flash_lite" in PRICING
    assert "pro" in PRICING
    assert "pro_tier_threshold" in PRICING
    assert PRICING["flash"]["input_text"] == 0.50
    assert PRICING["flash"]["input_audio"] == 1.00
    assert PRICING["flash"]["output"] == 3.00
    assert PRICING["flash_lite"]["input_text"] == 0.25
    assert PRICING["flash_lite"]["input_audio"] == 0.50
    assert PRICING["flash_lite"]["output"] == 1.50


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
