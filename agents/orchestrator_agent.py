"""
Agentic Transcription Orchestrator using Pydantic AI

This module provides an intelligent orchestrator agent that coordinates:
1. Gemini transcription for high-quality text
2. Parakeet timestamp correction for accurate timing
3. Quality validation and assessment

The orchestrator uses Pydantic AI's tool pattern to make decisions about
when to call each component based on the transcription needs.
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModelSettings
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import os

from models import TranscriptSegment, TranscriptQuality
from dependencies import TranscriptionDeps, AppDeps

logger = logging.getLogger(__name__)


class OrchestratorOutput(BaseModel):
    """Output from the orchestrator agent"""

    segments: List[TranscriptSegment] = Field(
        default_factory=list,
        description="Final transcript segments with corrected timestamps",
    )
    quality_score: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Overall quality score (0-100)",
    )
    timestamp_corrected: bool = Field(
        default=False,
        description="Whether timestamps were corrected using Parakeet",
    )
    processing_notes: List[str] = Field(
        default_factory=list,
        description="Notes about processing decisions and any issues encountered",
    )


def create_orchestrator_agent(deps: AppDeps) -> Agent:
    """
    Create the agentic orchestrator with tools for transcription coordination.

    The orchestrator is a Pydantic AI Agent that decides when to call:
    - transcribe_audio: Get quality transcription from Gemini
    - fix_timestamps: Correct timestamps using Parakeet NFA
    - check_quality: Validate transcript quality metrics

    Args:
        deps: Application dependencies containing configuration

    Returns:
        Configured Pydantic AI Agent with registered tools
    """
    # Ensure API key is set
    os.environ["GOOGLE_API_KEY"] = deps.transcription.api_key

    model_name = f"google-gla:{deps.transcription.model_name}"

    agent: Agent[AppDeps, OrchestratorOutput] = Agent(
        model_name,
        deps_type=AppDeps,
        output_type=OrchestratorOutput,
        instructions="""You are an intelligent transcription orchestrator agent that autonomously produces high-quality transcripts.

OBJECTIVE:
Analyze transcription results and make smart decisions about what tools to use.

WORKFLOW:
1. ALWAYS call transcribe_audio() first to get transcription from Gemini
2. Call analyze_timestamps() to check if timestamps need correction
3. Based on the analysis, DECIDE whether to call fix_timestamps()
4. Call check_quality() to validate the final result
5. Return complete OrchestratorOutput with your reasoning

AUTONOMOUS DECISION RULES:
You must DECIDE whether to fix timestamps based on analysis. Consider:
- If analyze_timestamps reports >20% irregular gaps → FIX timestamps
- If analyze_timestamps reports timestamp drift detected → FIX timestamps
- If analyze_timestamps reports poor alignment score (<70) → FIX timestamps
- If analyze_timestamps reports timestamps look good (score >85) → SKIP fixing
- If audio is very short (<30 seconds) → SKIP fixing (not worth the overhead)

ALWAYS explain your decision in processing_notes:
- "Decided to fix timestamps because: [reason]"
- "Skipped timestamp correction because: [reason]"

ERROR HANDLING:
- If any tool fails, capture the error in processing_notes
- If timestamp correction fails, use Gemini's timestamps and note why
- Provide partial results when possible rather than failing completely

Return the final OrchestratorOutput with:
- segments: The final transcript segments
- quality_score: From check_quality results
- timestamp_corrected: True ONLY if fix_timestamps was called AND succeeded
- processing_notes: Your decisions and reasoning
""",
    )

    # Register the orchestrator tools
    _register_tools(agent)

    return agent


def _register_tools(agent: Agent[AppDeps, OrchestratorOutput]) -> None:
    """Register all orchestrator tools with the agent"""

    @agent.tool
    async def transcribe_audio(
        ctx: RunContext[AppDeps],
        audio_path: str,
        context_prompt: Optional[str] = None,
        speaker_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Gemini for quality text and context understanding.

        This tool uses the existing Gemini transcription agent to produce
        high-quality transcription with speaker identification and context awareness.

        Args:
            ctx: Run context with dependencies
            audio_path: Path to audio file
            context_prompt: Optional context (topic, technical terms)
            speaker_names: Optional list of speaker names

        Returns:
            Dict with segments, count, duration, and chunking info
        """
        from agents.transcription_agent import (
            create_transcription_agent,
            run_transcription_agent,
            process_audio_file,
            chunk_audio,
            merge_chunks,
        )

        logger.info(f"Tool: transcribe_audio called for {audio_path}")

        try:
            # Get audio metadata
            metadata = await process_audio_file(ctx.deps.transcription, audio_path)

            # Create Gemini transcription agent
            gemini_agent = create_transcription_agent(ctx.deps.transcription)

            if metadata.needs_chunking:
                # Handle large files with chunking
                chunks = await chunk_audio(ctx.deps.transcription, audio_path)
                all_segments: List[List[TranscriptSegment]] = []
                previous_context: Optional[str] = None

                for chunk_info in chunks:
                    chunk_segments = await run_transcription_agent(
                        gemini_agent,
                        ctx.deps.transcription,
                        chunk_info["path"],
                        context_prompt,
                        chunk_info,
                        previous_context,
                        speaker_names,
                    )
                    all_segments.append(chunk_segments)

                    # Build context for next chunk (only if preserve_context is enabled)
                    if chunk_segments and ctx.deps.transcription.preserve_context:
                        context_segments = chunk_segments[-5:]
                        previous_context = "\n".join(
                            [f"{seg.speaker}: {seg.text}" for seg in context_segments]
                        )

                segments = await merge_chunks(ctx.deps.transcription, all_segments)
            else:
                # Direct transcription for smaller files
                segments = await run_transcription_agent(
                    gemini_agent,
                    ctx.deps.transcription,
                    audio_path,
                    context_prompt,
                    None,
                    None,
                    speaker_names,
                )

            return {
                "segments": [seg.model_dump() for seg in segments],
                "count": len(segments),
                "duration": metadata.duration,
                "chunked": metadata.needs_chunking,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "segments": [],
                "count": 0,
                "duration": 0,
                "chunked": False,
                "error": str(e),
            }

    @agent.tool
    async def fix_timestamps(
        ctx: RunContext[AppDeps],
        audio_path: str,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Fix timestamps using NVIDIA Parakeet + NeMo Forced Aligner.

        YOU decide when to call this based on analyze_timestamps() results.
        This uses Parakeet CTC model for precise word-level timing alignment.

        Args:
            ctx: Run context with dependencies
            audio_path: Path to audio file
            segments: Segments from transcribe_audio tool (as dicts)

        Returns:
            Dict with corrected segments, success flag, and any errors
        """
        from agents.timestamp_tool import fix_timestamps_with_parakeet

        logger.info(f"Tool: fix_timestamps called with {len(segments)} segments")

        if not segments:
            return {
                "segments": [],
                "corrected": False,
                "error": "No segments provided",
            }

        try:
            # Convert dicts back to TranscriptSegment objects
            transcript_segments = [
                TranscriptSegment(**seg) for seg in segments
            ]

            corrected = await fix_timestamps_with_parakeet(
                ctx.deps.transcription,
                audio_path,
                transcript_segments,
            )

            return {
                "segments": [seg.model_dump() for seg in corrected],
                "corrected": True,
                "error": None,
            }

        except ImportError as e:
            logger.warning(f"NeMo not installed, skipping timestamp correction: {e}")
            return {
                "segments": segments,
                "corrected": False,
                "error": f"NeMo not available: {e}",
            }
        except Exception as e:
            logger.warning(f"Timestamp correction failed: {e}")
            return {
                "segments": segments,  # Return original on failure
                "corrected": False,
                "error": str(e),
            }

    @agent.tool
    async def analyze_timestamps(
        ctx: RunContext[AppDeps],
        segments: List[Dict[str, Any]],
        audio_duration: float,
    ) -> Dict[str, Any]:
        """
        Analyze timestamp quality to help decide if correction is needed.

        This tool examines the timestamps from Gemini transcription and provides
        metrics to help you decide whether to call fix_timestamps().

        Args:
            ctx: Run context with dependencies
            segments: Segments from transcribe_audio (as dicts)
            audio_duration: Total audio duration in seconds

        Returns:
            Dict with alignment_score (0-100), issues found, and recommendation
        """
        logger.info(f"Tool: analyze_timestamps called with {len(segments)} segments")

        if not segments:
            return {
                "alignment_score": 0,
                "issues": ["No segments to analyze"],
                "recommendation": "skip",
                "reason": "No segments available",
            }

        if audio_duration < 30:
            return {
                "alignment_score": 80,
                "issues": [],
                "recommendation": "skip",
                "reason": "Audio too short (<30s), correction overhead not worth it",
            }

        try:
            import re

            issues = []
            timestamps_seconds = []

            # Parse all timestamps
            for seg in segments:
                ts = seg.get("timestamp", "[00:00:00]")
                match = re.match(r"^\[(\d{2}):(\d{2}):(\d{2})\]$", ts)
                if match:
                    h, m, s = map(int, match.groups())
                    timestamps_seconds.append(h * 3600 + m * 60 + s)
                else:
                    issues.append(f"Invalid timestamp format: {ts}")

            if len(timestamps_seconds) < 2:
                return {
                    "alignment_score": 50,
                    "issues": ["Too few segments for analysis"],
                    "recommendation": "skip",
                    "reason": "Not enough segments to analyze patterns",
                }

            # Check for monotonicity (timestamps should increase)
            non_monotonic = 0
            for i in range(1, len(timestamps_seconds)):
                if timestamps_seconds[i] < timestamps_seconds[i - 1]:
                    non_monotonic += 1
                    issues.append(f"Non-monotonic at segment {i}")

            # Check for irregular gaps
            gaps = [timestamps_seconds[i] - timestamps_seconds[i - 1]
                    for i in range(1, len(timestamps_seconds))]

            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                irregular_gaps = sum(1 for g in gaps if g < 0 or g > avg_gap * 3)
                irregular_pct = (irregular_gaps / len(gaps)) * 100

                if irregular_pct > 20:
                    issues.append(f"Irregular gaps: {irregular_pct:.0f}% of segments")

            # Check timestamp coverage (last timestamp vs audio duration)
            if timestamps_seconds:
                last_ts = max(timestamps_seconds)
                coverage = (last_ts / audio_duration) * 100 if audio_duration > 0 else 0

                if coverage < 70:
                    issues.append(f"Poor coverage: timestamps only reach {coverage:.0f}% of audio")
                elif coverage > 110:
                    issues.append(f"Timestamp drift: timestamps exceed audio duration")

            # Calculate alignment score
            score = 100
            score -= non_monotonic * 15  # Heavy penalty for non-monotonic
            score -= len([i for i in issues if "Irregular" in i]) * 10
            score -= len([i for i in issues if "coverage" in i.lower()]) * 15
            score -= len([i for i in issues if "drift" in i.lower()]) * 20
            score = max(0, min(100, score))

            # Make recommendation
            if score >= 85:
                recommendation = "skip"
                reason = f"Timestamps look good (score: {score})"
            elif score >= 70:
                recommendation = "optional"
                reason = f"Timestamps acceptable but could improve (score: {score})"
            else:
                recommendation = "fix"
                reason = f"Timestamps need correction (score: {score})"

            return {
                "alignment_score": score,
                "issues": issues,
                "recommendation": recommendation,
                "reason": reason,
            }

        except Exception as e:
            logger.error(f"Timestamp analysis failed: {e}")
            return {
                "alignment_score": 50,
                "issues": [f"Analysis error: {e}"],
                "recommendation": "fix",
                "reason": "Analysis failed, recommend fixing to be safe",
            }

    @agent.tool
    async def check_quality(
        ctx: RunContext[AppDeps],
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check transcript quality and return comprehensive metrics.

        Analyzes the transcript for readability, vocabulary richness,
        timestamp coverage, speaker consistency, and other quality indicators.

        Args:
            ctx: Run context with dependencies
            segments: Transcript segments to evaluate (as dicts)

        Returns:
            Dict with overall_score, assessment level, detailed metrics, and issues
        """
        from agents.quality_validator import (
            calculate_quality_metrics,
            calculate_overall_score,
        )

        logger.info(f"Tool: check_quality called with {len(segments)} segments")

        if not segments:
            return {
                "overall_score": 0.0,
                "assessment": "poor",
                "metrics": {},
                "issues": ["No segments to analyze"],
            }

        try:
            # Convert dicts to TranscriptSegment objects
            transcript_segments = [
                TranscriptSegment(**seg) for seg in segments
            ]

            # Calculate detailed metrics
            metrics = calculate_quality_metrics(ctx.deps.quality, transcript_segments)

            # Calculate overall weighted score
            overall_score = calculate_overall_score(ctx.deps.quality, metrics)

            # Determine quality assessment level
            if overall_score >= 80:
                assessment = "excellent"
            elif overall_score >= 60:
                assessment = "good"
            elif overall_score >= 40:
                assessment = "fair"
            else:
                assessment = "poor"

            # Extract warnings/issues from metrics
            issues = metrics.get("warnings", [])

            return {
                "overall_score": overall_score,
                "assessment": assessment,
                "metrics": {
                    "readability": metrics.get("readability", 0),
                    "vocabulary_richness": metrics.get("vocabulary_richness", 0),
                    "sentence_variety": metrics.get("sentence_variety", 0),
                    "punctuation_density": metrics.get("punctuation_density", 0),
                    "timestamp_coverage": metrics.get("timestamp_coverage", 0),
                    "speaker_consistency": metrics.get("speaker_consistency", 0),
                },
                "issues": issues,
            }

        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return {
                "overall_score": 0.0,
                "assessment": "unknown",
                "metrics": {},
                "issues": [f"Quality check error: {e}"],
            }


async def run_orchestrator(
    deps: AppDeps,
    audio_path: str,
    context_prompt: Optional[str] = None,
    speaker_names: Optional[List[str]] = None,
) -> OrchestratorOutput:
    """
    Run the orchestrator agent to transcribe audio.

    The agent will autonomously:
    1. Transcribe with Gemini for quality text
    2. Fix timestamps with Parakeet for accurate timing
    3. Check quality metrics
    4. Return complete results with processing notes

    Args:
        deps: Application dependencies
        audio_path: Path to the audio file to transcribe
        context_prompt: Optional context information for better accuracy
        speaker_names: Optional list of known speaker names

    Returns:
        OrchestratorOutput with segments, quality score, and processing notes
    """
    logger.info(f"Starting orchestrator for {audio_path}")

    agent = create_orchestrator_agent(deps)

    # Build user message for orchestrator
    # Let the agent make autonomous decisions
    speaker_info = ", ".join(speaker_names) if speaker_names else "Auto-detect speakers"

    user_message = f"""Transcribe the audio file at: {audio_path}

Context information: {context_prompt or 'None provided'}
Speaker names: {speaker_info}

Execute your workflow and make autonomous decisions:
1. Call transcribe_audio() with the audio path and context
2. Call analyze_timestamps() to assess timestamp quality
3. Based on the analysis, DECIDE whether to call fix_timestamps()
   - If recommendation is "fix" or alignment_score < 70 → call fix_timestamps
   - If recommendation is "skip" or alignment_score > 85 → skip it
   - Explain your decision in processing_notes
4. Call check_quality() on the final segments
5. Return OrchestratorOutput with your reasoning in processing_notes
"""

    # Configure model settings for Gemini 3
    # Note: google_thinking_config only supports thinking_level (not include_thoughts)
    model_settings = GoogleModelSettings(
        temperature=1.0,  # Gemini 3 default
        max_tokens=deps.transcription.max_output_tokens,
        google_thinking_config={
            "thinking_level": deps.transcription.thinking_level,
        },
    )

    try:
        result = await agent.run(
            user_message,
            deps=deps,
            model_settings=model_settings,
        )

        logger.info(
            f"Orchestrator complete: {len(result.output.segments)} segments, "
            f"quality={result.output.quality_score:.1f}"
        )

        return result.output

    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        # Return a valid but empty output on failure
        return OrchestratorOutput(
            segments=[],
            quality_score=0.0,
            timestamp_corrected=False,
            processing_notes=[f"Orchestrator error: {e}"],
        )
