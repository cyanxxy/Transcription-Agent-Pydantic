"""
Judge agent for candidate transcript selection and correction.

This agent compares one or more transcript candidates for the same audio span and
returns the best final transcript, optionally merging candidates when one has
better wording and another has better timestamps or speaker labels.
"""

from typing import List, Optional
import logging

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.usage import UsageLimits

from dependencies import AppDeps
from models import JudgeDecision, TranscriptCandidate

logger = logging.getLogger(__name__)


def create_judge_agent(deps: AppDeps) -> Agent:
    """Create the transcript judge agent."""
    model = GoogleModel(
        deps.transcription.judge_model_name,
        provider=GoogleProvider(api_key=deps.transcription.api_key),
    )

    agent: Agent[AppDeps, JudgeDecision] = Agent(
        model,
        deps_type=AppDeps,
        output_type=JudgeDecision,
        instructions="""You are an expert transcript judge and editor.

OBJECTIVE:
- Compare transcript candidates for the same audio span.
- Select the strongest candidate or merge candidates when it clearly improves accuracy.
- Return only structured data that validates against JudgeDecision.

RULES:
- Do not invent content that is not supported by at least one candidate.
- Prefer the more conservative wording when candidates disagree.
- Preserve the exact timestamp format [HH:MM:SS].
- Preserve speaker labels and provided speaker names when a candidate already does so well.
- If one candidate has better wording and another has better timestamps, combine them carefully.
- If there is only one candidate, lightly correct obvious transcript issues but stay faithful.
- Use [inaudible] instead of guessing.

OUTPUT:
- segments: final transcript segments
- selected_candidate_ids: the candidate ids you relied on most
- processing_notes: short notes explaining your decision and corrections
""",
    )

    @agent.output_validator
    async def validate_output(
        ctx: RunContext[AppDeps], output: JudgeDecision
    ) -> JudgeDecision:
        del ctx
        if not output.segments:
            raise ValueError("Judge returned no segments despite having candidates.")

        return output

    return agent


def _format_segments(candidate: TranscriptCandidate) -> str:
    lines = [
        f"{segment.timestamp} {segment.speaker}: {segment.text}"
        for segment in candidate.segments
    ]
    return "\n".join(lines) if lines else "[empty candidate]"


def _build_judge_prompt(
    candidates: List[TranscriptCandidate],
    context_prompt: Optional[str],
    speaker_names: Optional[List[str]],
    chunk_label: str,
) -> str:
    candidate_blocks = []
    for candidate in candidates:
        quality = (
            f"{candidate.quality_score:.1f}"
            if candidate.quality_score is not None
            else "unknown"
        )
        notes = "; ".join(candidate.notes) if candidate.notes else "none"
        candidate_blocks.append(
            "\n".join(
                [
                    f"Candidate ID: {candidate.candidate_id}",
                    f"Label: {candidate.label}",
                    f"Kind: {candidate.kind}",
                    f"Model: {candidate.model_name}",
                    f"Quality score: {quality}",
                    f"Notes: {notes}",
                    "Transcript:",
                    _format_segments(candidate),
                ]
            )
        )

    speakers = ", ".join(speaker_names) if speaker_names else "Auto-detect speakers"
    context = context_prompt or "None provided"

    return "\n\n".join(
        [
            f"Judge these transcript candidates for {chunk_label}.",
            f"Context: {context}",
            f"Known speakers: {speakers}",
            "Candidates:",
            "\n\n---\n\n".join(candidate_blocks),
            "Return the final corrected transcript for this audio span.",
        ]
    )


async def run_judge_agent(
    deps: AppDeps,
    candidates: List[TranscriptCandidate],
    context_prompt: Optional[str] = None,
    speaker_names: Optional[List[str]] = None,
    chunk_label: str = "the current audio span",
    agent: Optional[Agent] = None,
) -> JudgeDecision:
    """Run the judge agent over one or more transcript candidates."""
    if not candidates:
        return JudgeDecision(
            segments=[],
            selected_candidate_ids=[],
            processing_notes=["Judge skipped because no candidates were available."],
        )

    if agent is None:
        agent = create_judge_agent(deps)

    prompt = _build_judge_prompt(candidates, context_prompt, speaker_names, chunk_label)
    model_settings = GoogleModelSettings(
        temperature=1.0,
        max_tokens=deps.transcription.max_output_tokens,
        google_thinking_config={
            "thinking_level": deps.transcription.thinking_level,
        },
    )

    try:
        result = await agent.run(
            prompt,
            deps=deps,
            model_settings=model_settings,
            usage_limits=UsageLimits(request_limit=5),
        )
        valid_candidate_ids = {candidate.candidate_id for candidate in candidates}
        result.output.selected_candidate_ids = [
            candidate_id
            for candidate_id in result.output.selected_candidate_ids
            if candidate_id in valid_candidate_ids
        ]
        if not result.output.selected_candidate_ids:
            result.output.selected_candidate_ids = [candidates[0].candidate_id]
        logger.info(
            "Judge selected %s candidates and returned %s segments",
            len(result.output.selected_candidate_ids),
            len(result.output.segments),
        )
        return result.output
    except Exception as exc:
        logger.error("Judge agent failed: %s", exc)
        return JudgeDecision(
            segments=candidates[0].segments,
            selected_candidate_ids=[candidates[0].candidate_id],
            processing_notes=[
                f"Judge failed, falling back to {candidates[0].label}: {exc}"
            ],
        )
