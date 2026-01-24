from models import TranscriptSegment
from workflow import TranscriptionWorkflow


def test_srt_includes_speaker() -> None:
    workflow = TranscriptionWorkflow(api_key="test-key")
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
