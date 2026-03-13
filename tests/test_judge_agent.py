from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.judge_agent import run_judge_agent
from dependencies import AppDeps
from models import JudgeDecision, TranscriptCandidate, TranscriptSegment


def _segment(text: str, timestamp: str = "[00:00:00]") -> TranscriptSegment:
    return TranscriptSegment(timestamp=timestamp, speaker="Speaker 1", text=text)


def _candidate(candidate_id: str, text: str) -> TranscriptCandidate:
    return TranscriptCandidate(
        candidate_id=candidate_id,
        label=candidate_id,
        kind="gemini",
        model_name="gemini-3-flash-preview",
        segments=[_segment(text)],
    )


class FakeAgent:
    def __init__(self, output: JudgeDecision | None = None, exc: Exception | None = None):
        self.output = output
        self.exc = exc
        self.prompts: list[str] = []

    async def run(self, prompt, deps, model_settings, usage_limits):
        del deps, model_settings, usage_limits
        self.prompts.append(prompt)
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(output=self.output)


@pytest.mark.asyncio
async def test_run_judge_agent_without_candidates() -> None:
    deps = AppDeps.from_config(api_key="test-key")
    try:
        result = await run_judge_agent(deps, [])
    finally:
        deps.cleanup()

    assert result.segments == []
    assert result.selected_candidate_ids == []
    assert result.processing_notes == ["Judge skipped because no candidates were available."]


@pytest.mark.asyncio
async def test_run_judge_agent_filters_unknown_selected_candidates() -> None:
    deps = AppDeps.from_config(api_key="test-key")
    candidates = [_candidate("flash", "hello"), _candidate("pro", "hello there")]
    fake_agent = FakeAgent(
        JudgeDecision(
            segments=candidates[1].segments,
            selected_candidate_ids=["missing"],
            processing_notes=["Merged candidates."],
        )
    )

    try:
        result = await run_judge_agent(
            deps,
            candidates,
            context_prompt="Medical interview",
            speaker_names=["Dr. A"],
            agent=fake_agent,
        )
    finally:
        deps.cleanup()

    assert result.segments == candidates[1].segments
    assert result.selected_candidate_ids == ["flash"]
    assert "Candidate ID: flash" in fake_agent.prompts[0]
    assert "Candidate ID: pro" in fake_agent.prompts[0]
    assert "Medical interview" in fake_agent.prompts[0]


@pytest.mark.asyncio
async def test_run_judge_agent_falls_back_to_first_candidate_on_failure() -> None:
    deps = AppDeps.from_config(api_key="test-key")
    candidates = [_candidate("flash", "hello"), _candidate("pro", "hello there")]
    fake_agent = FakeAgent(exc=RuntimeError("judge exploded"))

    try:
        result = await run_judge_agent(deps, candidates, agent=fake_agent)
    finally:
        deps.cleanup()

    assert result.segments == candidates[0].segments
    assert result.selected_candidate_ids == ["flash"]
    assert "Judge failed, falling back to flash" in result.processing_notes[0]
