from models import TranscriptSegment
from agents.transcription_agent import (
    adjust_timestamp,
    map_speakers_to_context,
    ensure_speaker_consistency,
    build_transcription_prompt,
)


# --- adjust_timestamp ---


def test_adjust_timestamp_no_offset() -> None:
    assert adjust_timestamp("[00:00:00]", 0) == "[00:00:00]"


def test_adjust_timestamp_seconds() -> None:
    assert adjust_timestamp("[00:00:30]", 15) == "[00:00:45]"


def test_adjust_timestamp_minute_rollover() -> None:
    assert adjust_timestamp("[00:00:50]", 15) == "[00:01:05]"


def test_adjust_timestamp_hour_rollover() -> None:
    assert adjust_timestamp("[00:59:50]", 15) == "[01:00:05]"


def test_adjust_timestamp_large_offset() -> None:
    result = adjust_timestamp("[00:00:00]", 3661)
    assert result == "[01:01:01]"


def test_adjust_timestamp_invalid_format() -> None:
    assert adjust_timestamp("invalid", 10) == "invalid"


# --- map_speakers_to_context ---


def _seg(ts: str, speaker: str, text: str) -> TranscriptSegment:
    return TranscriptSegment(timestamp=ts, speaker=speaker, text=text)


def test_map_speakers_no_names() -> None:
    segments = [_seg("[00:00:00]", "Speaker 1", "Hello")]
    result = map_speakers_to_context(segments, None)
    assert result[0].speaker == "Speaker 1"


def test_map_speakers_empty_segments() -> None:
    result = map_speakers_to_context([], ["Alice"])
    assert result == []


def test_map_speakers_generic_to_names() -> None:
    segments = [
        _seg("[00:00:00]", "Speaker 1", "Hello"),
        _seg("[00:00:05]", "Speaker 2", "Hi"),
        _seg("[00:00:10]", "Speaker 1", "Bye"),
    ]
    result = map_speakers_to_context(segments, ["Alice", "Bob"])
    assert result[0].speaker == "Alice"
    assert result[1].speaker == "Bob"
    assert result[2].speaker == "Alice"


def test_map_speakers_preserves_real_names() -> None:
    segments = [
        _seg("[00:00:00]", "Dr. Smith", "Hello"),
        _seg("[00:00:05]", "Speaker 1", "Hi"),
    ]
    result = map_speakers_to_context(segments, ["Alice"])
    assert result[0].speaker == "Dr. Smith"  # Not a generic label
    assert result[1].speaker == "Alice"  # Speaker 1 -> Alice


def test_map_speakers_more_speakers_than_names() -> None:
    segments = [
        _seg("[00:00:00]", "Speaker 1", "Hello"),
        _seg("[00:00:05]", "Speaker 2", "Hi"),
        _seg("[00:00:10]", "Speaker 3", "Hey"),
    ]
    result = map_speakers_to_context(segments, ["Alice"])
    assert result[0].speaker == "Alice"
    assert result[1].speaker == "Speaker 2"  # No name for Speaker 2
    assert result[2].speaker == "Speaker 3"  # No name for Speaker 3


# --- ensure_speaker_consistency ---


def test_consistency_empty() -> None:
    assert ensure_speaker_consistency([]) == []


def test_consistency_preserves_names() -> None:
    segments = [
        _seg("[00:00:00]", "Alice", "Hello"),
        _seg("[00:00:05]", "Bob", "Hi"),
        _seg("[00:00:10]", "Alice", "Bye"),
    ]
    result = ensure_speaker_consistency(segments, preserve_names=True)
    assert result[0].speaker == "Alice"
    assert result[1].speaker == "Bob"
    assert result[2].speaker == "Alice"


def test_consistency_keeps_generic_labels() -> None:
    segments = [
        _seg("[00:00:00]", "Speaker 1", "Hello"),
        _seg("[00:00:05]", "Speaker 2", "Hi"),
    ]
    result = ensure_speaker_consistency(segments, preserve_names=True)
    assert result[0].speaker == "Speaker 1"
    assert result[1].speaker == "Speaker 2"


# --- build_transcription_prompt ---


def test_prompt_basic() -> None:
    prompt = build_transcription_prompt(None, None, None, None)
    assert "Transcribe" in prompt
    assert "TranscriptSegment" in prompt


def test_prompt_with_speakers() -> None:
    prompt = build_transcription_prompt(None, None, None, ["Alice", "Bob"])
    assert "Alice" in prompt
    assert "Bob" in prompt
    assert "KNOWN SPEAKERS" in prompt


def test_prompt_with_context() -> None:
    prompt = build_transcription_prompt(
        "Medical discussion", None, None, None
    )
    assert "Medical discussion" in prompt


def test_prompt_with_chunk_info() -> None:
    chunk = {"index": 2, "start_ms": 120000}
    prompt = build_transcription_prompt(None, None, chunk, None)
    assert "chunk 3" in prompt  # index + 1
    assert "120.0" in prompt


def test_prompt_with_previous_context() -> None:
    prompt = build_transcription_prompt(None, "Previous speaker said hello", None, None)
    assert "PREVIOUS CONTEXT" in prompt
    assert "Previous speaker said hello" in prompt
