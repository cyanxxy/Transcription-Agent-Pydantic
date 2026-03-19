from __future__ import annotations

import os
import pytest

import workflow as workflow_module
from dependencies import AppDeps
from models import (
    AudioFormat,
    AudioMetadata,
    JudgeDecision,
    TranscriptQuality,
    TranscriptCandidate,
    TranscriptSegment,
)
from workflow import TranscriptionWorkflow


def _segment(text: str, timestamp: str = "[00:00:00]") -> TranscriptSegment:
    return TranscriptSegment(timestamp=timestamp, speaker="Speaker 1", text=text)


def _candidate(
    candidate_id: str, text: str, notes: list[str] | None = None
) -> TranscriptCandidate:
    return TranscriptCandidate(
        candidate_id=candidate_id,
        label=candidate_id,
        kind="gemini",
        model_name="gemini-3-flash-preview",
        segments=[_segment(text)],
        notes=notes or [],
    )


@pytest.fixture
def workflow_fixture():
    workflow = TranscriptionWorkflow.__new__(TranscriptionWorkflow)
    workflow.deps = AppDeps.from_config(api_key="test-key")
    workflow._judge_agent = object()
    workflow._run_transcription_deps = None
    yield workflow
    workflow.deps.cleanup()


@pytest.mark.asyncio
async def test_run_unit_with_judge_returns_notes_when_all_candidates_are_empty(
    workflow_fixture, monkeypatch
) -> None:
    empty_candidate = TranscriptCandidate(
        candidate_id="flash",
        label="flash",
        kind="gemini",
        model_name="gemini-3-flash-preview",
        segments=[],
        notes=["flash produced no segments"],
    )

    async def fake_generate_candidates(self, *args, **kwargs):
        del self, args, kwargs
        return [empty_candidate]

    monkeypatch.setattr(
        TranscriptionWorkflow,
        "_generate_candidates",
        fake_generate_candidates,
    )

    result = await workflow_fixture._run_unit_with_judge(
        "audio.wav",
        None,
        None,
        None,
        None,
        "the full audio file",
    )

    assert result.final_segments == []
    assert result.selected_candidate_ids == []
    assert result.judge_notes == ["flash produced no segments"]


@pytest.mark.asyncio
async def test_run_unit_with_judge_uses_judge_decision(
    workflow_fixture, monkeypatch
) -> None:
    workflow_fixture.deps.cleanup()
    workflow_fixture.deps = AppDeps.from_config(
        api_key="test-key",
        transcription_thinking_level="minimal",
        judge_thinking_level="low",
    )
    candidates = [_candidate("flash", "hello"), _candidate("pro", "hello there")]
    judged_segments = [_segment("hello there", "[00:00:02]")]

    async def fake_generate_candidates(self, *args, **kwargs):
        del self, args, kwargs
        return candidates

    async def fake_run_judge_agent(*args, **kwargs):
        assert kwargs["deps"].transcription.judge_thinking_level == "low"
        return JudgeDecision(
            segments=judged_segments,
            selected_candidate_ids=["pro"],
            processing_notes=["Preferred pro wording."],
        )

    monkeypatch.setattr(
        TranscriptionWorkflow,
        "_generate_candidates",
        fake_generate_candidates,
    )
    monkeypatch.setattr(workflow_module, "run_judge_agent", fake_run_judge_agent)

    result = await workflow_fixture._run_unit_with_judge(
        "audio.wav",
        None,
        None,
        None,
        None,
        "the full audio file",
    )

    assert result.final_segments == judged_segments
    assert result.selected_candidate_ids == ["pro"]
    assert result.judge_notes == [
        "Preferred pro wording.",
        "Judge selected: pro",
    ]


@pytest.mark.asyncio
async def test_run_candidate_spec_uses_transcription_thinking_level(
    workflow_fixture, monkeypatch
) -> None:
    workflow_fixture.deps.cleanup()
    workflow_fixture.deps = AppDeps.from_config(
        api_key="test-key",
        transcription_thinking_level="minimal",
        judge_thinking_level="medium",
    )
    workflow_fixture._run_transcription_deps = workflow_fixture.deps.transcription
    captured: dict[str, str] = {}

    def fake_create_transcription_agent(deps):
        captured["model_name"] = deps.model_name
        captured["thinking_level"] = deps.transcription_thinking_level
        return object()

    async def fake_run_transcription_agent(
        agent,
        deps,
        audio_path,
        custom_prompt,
        chunk_info,
        previous_context,
        speaker_names,
    ):
        del (
            agent,
            audio_path,
            custom_prompt,
            chunk_info,
            previous_context,
            speaker_names,
        )
        captured["runtime_thinking_level"] = deps.transcription_thinking_level
        return [_segment("hello there")]

    monkeypatch.setattr(
        workflow_module, "create_transcription_agent", fake_create_transcription_agent
    )
    monkeypatch.setattr(
        workflow_module, "run_transcription_agent", fake_run_transcription_agent
    )

    candidate = await workflow_fixture._run_candidate_spec(
        {
            "candidate_id": "flash",
            "label": "flash",
            "kind": "gemini",
            "model_name": "gemini-3-flash-preview",
        },
        "audio.wav",
        None,
        None,
        None,
        None,
    )

    assert candidate.segments == [_segment("hello there")]
    assert captured["thinking_level"] == "minimal"
    assert captured["runtime_thinking_level"] == "minimal"


@pytest.mark.asyncio
async def test_review_timestamps_skips_when_analysis_says_skip(
    workflow_fixture, monkeypatch
) -> None:
    metadata = AudioMetadata(
        filename="test.wav",
        duration=120,
        size_mb=1,
        format=AudioFormat.WAV,
    )
    segments = [_segment("hello")]

    monkeypatch.setattr(
        workflow_module,
        "analyze_timestamp_quality",
        lambda segments, audio_duration: {
            "reason": "Timestamps look good (score: 90)",
            "recommendation": "skip",
        },
    )

    corrected, timestamps_corrected, notes = await workflow_fixture._review_timestamps(
        "audio.wav",
        metadata,
        segments,
    )

    assert corrected == segments
    assert timestamps_corrected is False
    assert notes == ["Timestamp review: Timestamps look good (score: 90)"]


@pytest.mark.asyncio
async def test_review_timestamps_applies_alignment_when_recommended(
    workflow_fixture, monkeypatch
) -> None:
    metadata = AudioMetadata(
        filename="test.wav",
        duration=120,
        size_mb=1,
        format=AudioFormat.WAV,
    )
    segments = [_segment("hello")]
    corrected_segments = [_segment("hello", "[00:00:03]")]

    monkeypatch.setattr(
        workflow_module,
        "analyze_timestamp_quality",
        lambda segments, audio_duration: {
            "reason": "Timestamps need correction (score: 40)",
            "recommendation": "fix",
        },
    )

    async def fake_fix_timestamps(deps, audio_path, segments):
        del deps, audio_path, segments
        return corrected_segments

    monkeypatch.setattr(
        workflow_module,
        "fix_timestamps_with_parakeet",
        fake_fix_timestamps,
    )

    corrected, timestamps_corrected, notes = await workflow_fixture._review_timestamps(
        "audio.wav",
        metadata,
        segments,
    )

    assert corrected == corrected_segments
    assert timestamps_corrected is True
    assert notes == [
        "Timestamp review: Timestamps need correction (score: 40)",
        "Applied Parakeet alignment after judging.",
    ]


@pytest.mark.asyncio
async def test_transcribe_audio_reuses_workflow_with_run_scoped_temp_dirs(
    monkeypatch,
) -> None:
    workflow = TranscriptionWorkflow(api_key="test-key", use_judge_pipeline=False)
    run_temp_dirs: list[str] = []

    async def fake_validate_audio_file(deps, file_data, filename):
        del file_data, filename
        run_temp_dirs.append(deps.temp_dir)
        assert os.path.isdir(deps.temp_dir)
        temp_path = os.path.join(deps.temp_dir, "upload.wav")
        with open(temp_path, "wb") as handle:
            handle.write(b"fake")
        return {"valid": True, "temp_path": temp_path}

    async def fake_process_audio_file(deps, file_path):
        del deps, file_path
        return AudioMetadata(
            filename="upload.wav",
            duration=30,
            size_mb=1,
            format=AudioFormat.WAV,
        )

    async def fake_transcribe_direct(
        self,
        audio_path,
        metadata,
        progress_callback,
        custom_prompt,
        speaker_names,
    ):
        del self, audio_path, metadata, progress_callback, custom_prompt, speaker_names
        return [_segment("hello")]

    async def fake_calculate_quality(self, segments):
        del self, segments
        return TranscriptQuality(
            overall_score=80,
            readability=80,
            punctuation_density=0.05,
            sentence_variety=70,
            vocabulary_richness=75,
        )

    monkeypatch.setattr(
        workflow_module, "validate_audio_file", fake_validate_audio_file
    )
    monkeypatch.setattr(workflow_module, "process_audio_file", fake_process_audio_file)
    monkeypatch.setattr(
        TranscriptionWorkflow, "_transcribe_direct", fake_transcribe_direct
    )
    monkeypatch.setattr(
        TranscriptionWorkflow, "_calculate_quality", fake_calculate_quality
    )

    try:
        await workflow.transcribe_audio(b"first", "first.wav")
        first_temp_dir = run_temp_dirs[-1]
        assert not os.path.exists(first_temp_dir)

        await workflow.transcribe_audio(b"second", "second.wav")
        second_temp_dir = run_temp_dirs[-1]
        assert not os.path.exists(second_temp_dir)

        assert len(run_temp_dirs) == 2
        assert first_temp_dir != second_temp_dir
    finally:
        workflow.cleanup()
