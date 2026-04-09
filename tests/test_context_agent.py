from __future__ import annotations

import pytest

from agents.context_agent import create_context_prompt, process_context
from models import TranscriptContext


@pytest.mark.parametrize(
    "expected_format",
    ["meeting", "interview", "lecture", "podcast", "legal", "medical", "technical"],
)
def test_expected_format_is_transcript_safe(expected_format: str) -> None:
    prompt = create_context_prompt(expected_format=expected_format)

    lowered = prompt.lower()
    assert prompt
    assert "format as" not in lowered
    assert "meeting minutes" not in lowered
    assert "main points and examples clearly marked" not in lowered


def test_process_context_keeps_format_context_but_not_rewrite_instructions() -> None:
    context = TranscriptContext(
        expected_format="meeting",
        topic="Budget review",
    )

    prompt = process_context(context)
    lowered = prompt.lower()

    assert "budget review" in lowered
    assert "format as" not in lowered
    assert "meeting minutes" not in lowered
    assert "rewrite" not in lowered


def test_interview_format_does_not_relabel_or_force_qa() -> None:
    prompt = create_context_prompt(expected_format="interview")
    lowered = prompt.lower()

    assert "q&a" not in lowered
    assert "interviewer" not in lowered
    assert "interviewee" not in lowered
    assert "do not rewrite" in lowered
