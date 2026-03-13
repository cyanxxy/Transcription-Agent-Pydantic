from __future__ import annotations

import pytest

import workflow as workflow_module
from dependencies import AppDeps
from models import (
    AudioFormat,
    AudioMetadata,
    JudgeDecision,
    TranscriptCandidate,
    TranscriptSegment,
)
from workflow import TranscriptionWorkflow


def _segment(text: str, timestamp: str = "[00:00:00]") -> TranscriptSegment:
    return TranscriptSegment(timestamp=timestamp, speaker="Speaker 1", text=text)


def _candidate(candidate_id: str, text: str, notes: list[str] | None = None) -> TranscriptCandidate:
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
    candidates = [_candidate("flash", "hello"), _candidate("pro", "hello there")]
    judged_segments = [_segment("hello there", "[00:00:02]")]

    async def fake_generate_candidates(self, *args, **kwargs):
        del self, args, kwargs
        return candidates

    async def fake_run_judge_agent(*args, **kwargs):
        del args, kwargs
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
