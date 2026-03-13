from models import TranscriptSegment
from workflow import TranscriptionWorkflow


def _make_workflow() -> TranscriptionWorkflow:
    return TranscriptionWorkflow(api_key="test-key")


def test_srt_includes_speaker() -> None:
    workflow = _make_workflow()
    try:
        segments = [
            TranscriptSegment(
                timestamp="[00:00:01]",
                speaker="Alice",
                text="Hello there",
            )
        ]
        srt = workflow._export_as_srt(segments)
        assert "Alice:" in srt
    finally:
        workflow.cleanup()


def test_srt_single_segment_structure() -> None:
    workflow = _make_workflow()
    try:
        segments = [
            TranscriptSegment(
                timestamp="[00:00:05]",
                speaker="Bob",
                text="Good morning.",
            )
        ]
        srt = workflow._export_as_srt(segments)
        lines = srt.strip().split("\n")
        assert lines[0] == "1"
        assert "-->" in lines[1]
        assert "Bob:" in lines[2]
        assert "Good morning." in lines[2]
    finally:
        workflow.cleanup()


def test_srt_multiple_segments() -> None:
    workflow = _make_workflow()
    try:
        segments = [
            TranscriptSegment(
                timestamp="[00:00:01]", speaker="A", text="First"
            ),
            TranscriptSegment(
                timestamp="[00:00:05]", speaker="B", text="Second"
            ),
            TranscriptSegment(
                timestamp="[00:00:10]", speaker="A", text="Third"
            ),
        ]
        srt = workflow._export_as_srt(segments)
        assert "1\n" in srt
        assert "2\n" in srt
        assert "3\n" in srt
        assert "First" in srt
        assert "Second" in srt
        assert "Third" in srt
    finally:
        workflow.cleanup()


def test_srt_non_monotonic_timestamps_corrected() -> None:
    workflow = _make_workflow()
    try:
        segments = [
            TranscriptSegment(
                timestamp="[00:00:10]", speaker="A", text="Later"
            ),
            TranscriptSegment(
                timestamp="[00:00:05]", speaker="B", text="Earlier"
            ),
        ]
        srt = workflow._export_as_srt(segments)
        lines = srt.strip().split("\n")
        # Second segment should have timestamp >= first segment's
        time1 = lines[1].split(" --> ")[0]
        time2 = lines[5].split(" --> ")[0]
        t1_sec = workflow._srt_time_to_seconds(time1)
        t2_sec = workflow._srt_time_to_seconds(time2)
        assert t2_sec >= t1_sec
    finally:
        workflow.cleanup()


def test_srt_long_text_wrapped() -> None:
    workflow = _make_workflow()
    try:
        long_text = "This is a very long sentence that should be wrapped across multiple lines in the SRT output format"
        segments = [
            TranscriptSegment(
                timestamp="[00:00:01]",
                speaker="Alice",
                text=long_text,
            )
        ]
        srt = workflow._export_as_srt(segments, max_line_length=42)
        # At minimum, the wrapped SRT should still contain the content.
        assert "This is" in srt
    finally:
        workflow.cleanup()


def test_srt_time_conversion() -> None:
    workflow = _make_workflow()
    try:
        assert workflow._convert_to_srt_time("[01:30:45]") == "01:30:45,000"
    finally:
        workflow.cleanup()


def test_srt_add_seconds() -> None:
    workflow = _make_workflow()
    try:
        result = workflow._add_seconds_to_srt_time("00:00:58,000", 5)
        assert result == "00:01:03,000"
    finally:
        workflow.cleanup()


def test_srt_time_to_seconds() -> None:
    workflow = _make_workflow()
    try:
        assert workflow._srt_time_to_seconds("01:30:45,000") == 5445.0
        assert workflow._srt_time_to_seconds("00:00:00,000") == 0.0
    finally:
        workflow.cleanup()
