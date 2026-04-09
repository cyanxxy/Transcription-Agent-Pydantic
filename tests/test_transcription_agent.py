from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from pydantic_ai import AudioUrl
from models import TranscriptSegment
from agents.transcription_agent import (
    adjust_timestamp,
    map_speakers_to_context,
    ensure_speaker_consistency,
    build_transcription_prompt,
    process_audio_file,
    chunk_audio,
    merge_chunks,
    run_transcription_agent,
    _detect_overlap_boundary,
)
from dependencies import TranscriptionDeps
import pytest


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
    prompt = build_transcription_prompt("Medical discussion", None, None, None)
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


class FakeFilesClient:
    def __init__(self) -> None:
        self.upload_calls: list[str] = []
        self.delete_calls: list[str] = []

    async def upload(self, *, file, config=None):
        del config
        self.upload_calls.append(str(file))
        return SimpleNamespace(
            uri="https://example.com/audio.wav",
            mime_type="audio/wav",
            name="files/123",
        )

    async def delete(self, *, name, config=None):
        del config
        self.delete_calls.append(name)
        return SimpleNamespace()


class FakeAgent:
    def __init__(
        self,
        files_client: FakeFilesClient,
        output: Optional[List[TranscriptSegment]] = None,
        exc: Optional[Exception] = None,
    ) -> None:
        self.model = SimpleNamespace(
            client=SimpleNamespace(aio=SimpleNamespace(files=files_client))
        )
        self.output = output
        self.exc = exc
        self.inputs: list[list[object]] = []

    async def run(self, inputs, deps, model_settings, usage_limits):
        del deps, model_settings, usage_limits
        self.inputs.append(inputs)
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(output=self.output)


@pytest.mark.asyncio
async def test_run_transcription_agent_uploads_audio_url_and_deletes_file(
    tmp_path,
) -> None:
    file_path = tmp_path / "audio.wav"
    file_path.write_bytes(b"fake")
    deps = TranscriptionDeps(api_key="test-key")
    files_client = FakeFilesClient()
    agent = FakeAgent(files_client, output=[_seg("[00:00:00]", "Speaker 1", "hello")])

    try:
        segments = await run_transcription_agent(agent, deps, str(file_path))
    finally:
        deps.cleanup()

    assert segments == [_seg("[00:00:00]", "Speaker 1", "hello")]
    assert files_client.upload_calls == [str(file_path)]
    assert files_client.delete_calls == ["files/123"]
    assert isinstance(agent.inputs[0][1], AudioUrl)


@pytest.mark.asyncio
async def test_run_transcription_agent_deletes_uploaded_file_on_failure(tmp_path) -> None:
    file_path = tmp_path / "audio.wav"
    file_path.write_bytes(b"fake")
    deps = TranscriptionDeps(api_key="test-key")
    files_client = FakeFilesClient()
    agent = FakeAgent(files_client, exc=RuntimeError("boom"))

    try:
        with pytest.raises(RuntimeError, match="boom"):
            await run_transcription_agent(agent, deps, str(file_path))
    finally:
        deps.cleanup()

    assert files_client.upload_calls == [str(file_path)]
    assert files_client.delete_calls == ["files/123"]


@pytest.mark.asyncio
async def test_process_audio_file_counts_overlapping_chunks(
    monkeypatch, tmp_path
) -> None:
    class FakeAudio:
        frame_rate = 16_000
        channels = 1

        def __len__(self):
            return 10_000

    file_path = tmp_path / "audio.wav"
    file_path.write_bytes(b"fake")

    monkeypatch.setattr(
        "agents.transcription_agent.AudioSegment.from_file",
        lambda path: FakeAudio(),
    )

    deps = TranscriptionDeps(
        api_key="test-key",
        chunk_duration_ms=4_000,
        chunk_overlap_ms=1_000,
    )

    try:
        metadata = await process_audio_file(deps, str(file_path))
    finally:
        deps.cleanup()

    assert metadata.needs_chunking is True
    assert metadata.chunk_count == 3


@pytest.mark.asyncio
async def test_chunk_audio_skips_fully_overlapped_tail_chunk(
    monkeypatch, tmp_path
) -> None:
    class FakeChunk:
        def __init__(self, start_ms: int, end_ms: int) -> None:
            self.start_ms = start_ms
            self.end_ms = end_ms

        def export(self, path, format="wav"):
            del format
            Path(path).write_bytes(f"{self.start_ms}-{self.end_ms}".encode())

    class FakeAudio:
        frame_rate = 16_000
        channels = 1

        def __len__(self):
            return 10_000

        def __getitem__(self, item):
            assert isinstance(item, slice)
            start_ms = 0 if item.start is None else item.start
            end_ms = len(self) if item.stop is None else item.stop
            return FakeChunk(start_ms, end_ms)

    file_path = tmp_path / "audio.wav"
    file_path.write_bytes(b"fake")

    monkeypatch.setattr(
        "agents.transcription_agent.AudioSegment.from_file",
        lambda path: FakeAudio(),
    )

    deps = TranscriptionDeps(
        api_key="test-key",
        chunk_duration_ms=4_000,
        chunk_overlap_ms=1_000,
    )

    try:
        chunks = await chunk_audio(deps, str(file_path))
    finally:
        deps.cleanup()

    assert len(chunks) == 3
    assert [chunk["start_ms"] for chunk in chunks] == [0, 3_000, 6_000]
    assert [chunk["end_ms"] for chunk in chunks] == [4_000, 7_000, 10_000]


def test_detect_overlap_boundary_for_duplicate_segments() -> None:
    previous_segments = [_seg("[00:00:05]", "Speaker 1", "hello there")]
    next_segments = [
        _seg("[00:00:05]", "Speaker 1", "hello there"),
        _seg("[00:00:07]", "Speaker 1", "new content"),
    ]

    assert _detect_overlap_boundary(previous_segments, next_segments) == 1


@pytest.mark.asyncio
async def test_merge_chunks_preserves_repeated_phrase_when_timestamps_differ() -> None:
    deps = TranscriptionDeps(api_key="test-key")
    chunk_results = [
        [_seg("[00:00:05]", "Speaker 1", "hello there")],
        [_seg("[00:00:15]", "Speaker 1", "hello there")],
    ]

    try:
        merged = await merge_chunks(deps, chunk_results)
    finally:
        deps.cleanup()

    assert len(merged) == 2
    assert merged[0].timestamp == "[00:00:05]"
    assert merged[1].timestamp == "[00:00:15]"
