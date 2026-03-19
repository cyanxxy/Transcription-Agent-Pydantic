import pytest

from models import (
    TranscriptSegment,
    AudioMetadata,
    AudioFormat,
    TranscriptCandidate,
    TranscriptQuality,
    JudgeDecision,
    TranscriptResult,
    EditOperation,
    TranscriptContext,
    ErrorDetail,
    AppState,
    ProcessingStatus,
)


# --- TranscriptSegment ---


def test_valid_segment() -> None:
    seg = TranscriptSegment(
        timestamp="[00:01:30]",
        speaker="Alice",
        text="Hello world",
        confidence=0.95,
    )
    assert seg.timestamp == "[00:01:30]"
    assert seg.speaker == "Alice"
    assert seg.text == "Hello world"
    assert seg.confidence == 0.95


def test_segment_strips_speaker_whitespace() -> None:
    seg = TranscriptSegment(
        timestamp="[00:00:00]",
        speaker="  Bob  ",
        text="Hi",
    )
    assert seg.speaker == "Bob"


def test_segment_strips_text_whitespace() -> None:
    seg = TranscriptSegment(
        timestamp="[00:00:00]",
        speaker="Alice",
        text="  Hello  ",
    )
    assert seg.text == "Hello"


def test_invalid_timestamp_format() -> None:
    with pytest.raises(Exception):
        TranscriptSegment(
            timestamp="00:01:30",
            speaker="Alice",
            text="Hello",
        )


def test_invalid_timestamp_minutes() -> None:
    with pytest.raises(Exception):
        TranscriptSegment(
            timestamp="[99:99:99]",
            speaker="Alice",
            text="Hello",
        )


def test_empty_speaker() -> None:
    with pytest.raises(Exception):
        TranscriptSegment(
            timestamp="[00:00:00]",
            speaker="   ",
            text="Hello",
        )


def test_empty_text() -> None:
    with pytest.raises(Exception):
        TranscriptSegment(
            timestamp="[00:00:00]",
            speaker="Alice",
            text="",
        )


def test_confidence_out_of_range() -> None:
    with pytest.raises(Exception):
        TranscriptSegment(
            timestamp="[00:00:00]",
            speaker="Alice",
            text="Hello",
            confidence=1.5,
        )


def test_confidence_negative() -> None:
    with pytest.raises(Exception):
        TranscriptSegment(
            timestamp="[00:00:00]",
            speaker="Alice",
            text="Hello",
            confidence=-0.1,
        )


def test_confidence_optional() -> None:
    seg = TranscriptSegment(
        timestamp="[00:00:00]",
        speaker="Alice",
        text="Hello",
    )
    assert seg.confidence is None


# --- AudioMetadata ---


def test_audio_metadata_valid() -> None:
    meta = AudioMetadata(
        filename="test.mp3",
        duration=120.5,
        size_mb=5.0,
        format=AudioFormat.MP3,
    )
    assert meta.filename == "test.mp3"
    assert meta.needs_chunking is False


def test_audio_metadata_allows_large_size_values() -> None:
    meta = AudioMetadata(
        filename="big.wav",
        duration=3600,
        size_mb=250,
        format=AudioFormat.WAV,
    )
    assert meta.size_mb == 250


def test_audio_metadata_with_chunking() -> None:
    meta = AudioMetadata(
        filename="long.mp3",
        duration=600,
        size_mb=50,
        format=AudioFormat.MP3,
        needs_chunking=True,
        chunk_count=5,
    )
    assert meta.needs_chunking is True
    assert meta.chunk_count == 5


# --- TranscriptQuality ---


def test_quality_excellent() -> None:
    q = TranscriptQuality(
        overall_score=85,
        readability=80,
        punctuation_density=0.05,
        sentence_variety=70,
        vocabulary_richness=75,
    )
    assert q.quality_assessment == "Excellent"


def test_quality_good() -> None:
    q = TranscriptQuality(
        overall_score=65,
        readability=60,
        punctuation_density=0.05,
        sentence_variety=60,
        vocabulary_richness=60,
    )
    assert q.quality_assessment == "Good"


def test_quality_fair() -> None:
    q = TranscriptQuality(
        overall_score=45,
        readability=40,
        punctuation_density=0.05,
        sentence_variety=40,
        vocabulary_richness=40,
    )
    assert q.quality_assessment == "Fair"


def test_quality_poor() -> None:
    q = TranscriptQuality(
        overall_score=20,
        readability=20,
        punctuation_density=0.05,
        sentence_variety=20,
        vocabulary_richness=20,
    )
    assert q.quality_assessment == "Poor"


# --- TranscriptResult ---


def test_transcript_candidate_defaults() -> None:
    seg = TranscriptSegment(timestamp="[00:00:00]", speaker="Alice", text="Hello")
    candidate = TranscriptCandidate(
        candidate_id="gemini_flash",
        label="Gemini Flash",
        kind="gemini",
        model_name="gemini-3-flash-preview",
        segments=[seg],
    )
    assert candidate.quality_score is None
    assert candidate.notes == []


def test_judge_decision_defaults() -> None:
    decision = JudgeDecision()
    assert decision.segments == []
    assert decision.selected_candidate_ids == []
    assert decision.processing_notes == []


def test_result_created_at_has_timezone() -> None:
    seg = TranscriptSegment(timestamp="[00:00:00]", speaker="Alice", text="Hello")
    meta = AudioMetadata(
        filename="test.mp3", duration=10, size_mb=1, format=AudioFormat.MP3
    )
    quality = TranscriptQuality(
        overall_score=80,
        readability=80,
        punctuation_density=0.05,
        sentence_variety=70,
        vocabulary_richness=75,
    )
    result = TranscriptResult(
        segments=[seg],
        metadata=meta,
        quality=quality,
        processing_time=5.0,
        model_used="gemini-3-flash-preview",
    )
    assert result.created_at.tzinfo is not None


def test_result_serializes_datetime() -> None:
    seg = TranscriptSegment(timestamp="[00:00:00]", speaker="Alice", text="Hello")
    meta = AudioMetadata(
        filename="test.mp3", duration=10, size_mb=1, format=AudioFormat.MP3
    )
    quality = TranscriptQuality(
        overall_score=80,
        readability=80,
        punctuation_density=0.05,
        sentence_variety=70,
        vocabulary_richness=75,
    )
    result = TranscriptResult(
        segments=[seg],
        metadata=meta,
        quality=quality,
        processing_time=5.0,
        model_used="gemini-3-flash-preview",
    )
    data = result.model_dump()
    # field_serializer should produce an ISO string
    assert isinstance(data["created_at"], str)
    assert "T" in data["created_at"]


def test_result_full_text() -> None:
    segs = [
        TranscriptSegment(timestamp="[00:00:00]", speaker="A", text="Hello"),
        TranscriptSegment(timestamp="[00:00:05]", speaker="B", text="World"),
    ]
    meta = AudioMetadata(
        filename="test.mp3", duration=10, size_mb=1, format=AudioFormat.MP3
    )
    quality = TranscriptQuality(
        overall_score=80,
        readability=80,
        punctuation_density=0.05,
        sentence_variety=70,
        vocabulary_richness=75,
    )
    result = TranscriptResult(
        segments=segs,
        metadata=meta,
        quality=quality,
        processing_time=1.0,
        model_used="test",
    )
    assert result.full_text == "Hello World"


def test_result_formatted_text() -> None:
    segs = [
        TranscriptSegment(timestamp="[00:00:00]", speaker="Alice", text="Hi"),
    ]
    meta = AudioMetadata(
        filename="test.mp3", duration=10, size_mb=1, format=AudioFormat.MP3
    )
    quality = TranscriptQuality(
        overall_score=80,
        readability=80,
        punctuation_density=0.05,
        sentence_variety=70,
        vocabulary_richness=75,
    )
    result = TranscriptResult(
        segments=segs,
        metadata=meta,
        quality=quality,
        processing_time=1.0,
        model_used="test",
    )
    assert "[00:00:00] Alice: Hi" in result.formatted_text


def test_result_unique_speakers() -> None:
    segs = [
        TranscriptSegment(timestamp="[00:00:00]", speaker="Alice", text="Hi"),
        TranscriptSegment(timestamp="[00:00:05]", speaker="Bob", text="Hey"),
        TranscriptSegment(timestamp="[00:00:10]", speaker="Alice", text="Bye"),
    ]
    meta = AudioMetadata(
        filename="test.mp3", duration=15, size_mb=1, format=AudioFormat.MP3
    )
    quality = TranscriptQuality(
        overall_score=80,
        readability=80,
        punctuation_density=0.05,
        sentence_variety=70,
        vocabulary_richness=75,
    )
    result = TranscriptResult(
        segments=segs,
        metadata=meta,
        quality=quality,
        processing_time=1.0,
        model_used="test",
    )
    assert set(result.unique_speakers) == {"Alice", "Bob"}


def test_result_get_speaker_segments() -> None:
    segs = [
        TranscriptSegment(timestamp="[00:00:00]", speaker="Alice", text="Hi"),
        TranscriptSegment(timestamp="[00:00:05]", speaker="Bob", text="Hey"),
        TranscriptSegment(timestamp="[00:00:10]", speaker="Alice", text="Bye"),
    ]
    meta = AudioMetadata(
        filename="test.mp3", duration=15, size_mb=1, format=AudioFormat.MP3
    )
    quality = TranscriptQuality(
        overall_score=80,
        readability=80,
        punctuation_density=0.05,
        sentence_variety=70,
        vocabulary_richness=75,
    )
    result = TranscriptResult(
        segments=segs,
        metadata=meta,
        quality=quality,
        processing_time=1.0,
        model_used="test",
    )
    alice_segs = result.get_speaker_segments("Alice")
    assert len(alice_segs) == 2


def test_result_stores_judge_metadata() -> None:
    seg = TranscriptSegment(timestamp="[00:00:00]", speaker="Alice", text="Hello")
    meta = AudioMetadata(
        filename="test.mp3", duration=10, size_mb=1, format=AudioFormat.MP3
    )
    quality = TranscriptQuality(
        overall_score=80,
        readability=80,
        punctuation_density=0.05,
        sentence_variety=70,
        vocabulary_richness=75,
    )
    candidate = TranscriptCandidate(
        candidate_id="gemini_flash",
        label="Gemini Flash",
        kind="gemini",
        model_name="gemini-3-flash-preview",
        segments=[seg],
        quality_score=75,
    )
    result = TranscriptResult(
        segments=[seg],
        metadata=meta,
        quality=quality,
        processing_time=1.0,
        model_used="gemini-3-flash-preview",
        candidate_strategy="dual_gemini",
        candidates=[candidate],
        judge_used=True,
        judge_model_used="gemini-3.1-pro-preview",
        judge_selected_candidate_ids=["gemini_flash"],
        judge_notes=["Judge selected gemini_flash"],
    )
    assert result.judge_used is True
    assert result.judge_model_used == "gemini-3.1-pro-preview"
    assert result.judge_selected_candidate_ids == ["gemini_flash"]
    assert result.candidates[0].candidate_id == "gemini_flash"
    assert result.judge_notes == ["Judge selected gemini_flash"]


# --- EditOperation ---


def test_edit_operation_has_utc_timestamp() -> None:
    op = EditOperation(operation="replace", target="old", replacement="new")
    assert op.timestamp.tzinfo is not None


# --- ErrorDetail ---


def test_error_detail_has_utc_timestamp() -> None:
    err = ErrorDetail(code="E001", message="test", category="test")
    assert err.timestamp.tzinfo is not None


# --- TranscriptContext ---


def test_context_defaults() -> None:
    ctx = TranscriptContext()
    assert ctx.speaker_names == []
    assert ctx.topic is None
    assert ctx.technical_terms == []


# --- AppState ---


def test_app_state_reset() -> None:
    state = AppState()
    state.status = ProcessingStatus.PROCESSING
    state.current_file = "test.mp3"
    state.processing_progress = 0.5
    state.reset()
    assert state.status == ProcessingStatus.IDLE
    assert state.current_file is None
    assert state.processing_progress == 0.0


def test_app_state_defaults_include_pipeline_config() -> None:
    state = AppState()
    assert state.model_name == "gemini-3-flash-preview"
    assert state.judge_model_name == "gemini-3.1-pro-preview"
    assert state.candidate_strategy == "dual_gemini"
    assert state.use_judge_pipeline is True
    assert state.auto_format is True
    assert state.remove_fillers is False


def test_app_state_reset_preserves_pipeline_config() -> None:
    state = AppState(
        model_name="gemini-3.1-pro-preview",
        judge_model_name="gemini-3-flash-preview",
        candidate_strategy="single_gemini",
        use_judge_pipeline=False,
        auto_format=False,
        remove_fillers=True,
    )
    state.status = ProcessingStatus.ERROR
    state.current_file = "test.mp3"
    state.processing_progress = 0.9

    state.reset()

    assert state.model_name == "gemini-3.1-pro-preview"
    assert state.judge_model_name == "gemini-3-flash-preview"
    assert state.candidate_strategy == "single_gemini"
    assert state.use_judge_pipeline is False
    assert state.auto_format is False
    assert state.remove_fillers is True
