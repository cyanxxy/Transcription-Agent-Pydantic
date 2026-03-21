"""
Main Workflow Coordinator using Pydantic AI
Orchestrates the entire transcription pipeline

Supports two modes:
1. Judge Pipeline (default): Generate candidate transcripts, then judge and align
2. Direct Pipeline: Manual single-model transcription without judging
"""

from typing import Optional, Callable, List, Dict, Any, Literal, cast
from dataclasses import dataclass, replace
import asyncio
import logging
from datetime import datetime
import os
import shutil
import tempfile

from pydantic_ai import Agent
from models import (
    JudgeDecision,
    TranscriptResult,
    TranscriptSegment,
    TranscriptCandidate,
    AudioMetadata,
    TranscriptQuality,
    ProcessingStatus,
)
from dependencies import AppDeps, TranscriptionDeps

# Import agent tools
from agents.transcription_agent import (
    create_transcription_agent,
    validate_audio_file,
    process_audio_file,
    chunk_audio,
    merge_chunks,
    map_speakers_to_context,
    run_transcription_agent,
    adjust_timestamp,
)

from agents.judge_agent import (
    create_judge_agent,
    run_judge_agent,
)
from agents.timestamp_tool import (
    analyze_timestamp_quality,
    fix_timestamps_with_parakeet,
    transcribe_with_parakeet,
)

logger = logging.getLogger(__name__)


@dataclass
class JudgedUnitResult:
    """Intermediate result for a chunk or full audio span."""

    final_segments: List[TranscriptSegment]
    candidates: List[TranscriptCandidate]
    selected_candidate_ids: List[str]
    judge_notes: List[str]


class TranscriptionWorkflow:
    """Main workflow coordinator for transcription pipeline"""

    def __init__(self, api_key: str, **kwargs):
        """Initialize workflow with dependencies"""
        self.deps = AppDeps.from_config(api_key=api_key, **kwargs)

        # Create agents once and reuse (avoids repeated GoogleProvider creation)
        self.transcription_agent = create_transcription_agent(self.deps.transcription)
        self._judge_agent: Optional[Agent[AppDeps, JudgeDecision]] = None

        # Track processing state
        self.current_status = ProcessingStatus.IDLE
        self.current_file: Optional[str] = None
        self.processing_start: Optional[datetime] = None
        self._run_transcription_deps: Optional[TranscriptionDeps] = None

    @property
    def judge_agent(self) -> Agent[AppDeps, JudgeDecision]:
        """Lazily create and cache the judge agent."""
        if self._judge_agent is None:
            self._judge_agent = create_judge_agent(self.deps)
        return self._judge_agent

    @property
    def active_transcription_deps(self) -> TranscriptionDeps:
        """Return run-scoped transcription deps when a run is active."""
        return self._run_transcription_deps or self.deps.transcription

    async def transcribe_audio(
        self,
        file_data: bytes,
        filename: str,
        progress_callback: Optional[Callable] = None,
        custom_prompt: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> TranscriptResult:
        """Main transcription workflow with streaming progress

        Supports two modes:
        1. Judge Pipeline (default): candidate generation plus judge selection
        2. Direct Pipeline: Manual step-by-step processing
        """

        self.current_status = ProcessingStatus.PROCESSING
        self.current_file = filename
        self.processing_start = datetime.now()
        self._run_transcription_deps = self._create_run_transcription_deps()

        try:
            # Step 1: Validate file
            if progress_callback:
                progress_callback("Validating audio file...", 0.1)

            validation = await validate_audio_file(
                self.active_transcription_deps, file_data, filename
            )

            if not validation.get("valid"):
                raise ValueError(validation.get("error", "Invalid audio file"))

            # Step 2: Process audio metadata
            if progress_callback:
                progress_callback("Processing audio...", 0.2)

            temp_path: str = validation["temp_path"]
            metadata = await process_audio_file(
                self.active_transcription_deps, temp_path
            )

            # Step 2.5: Process user context if provided
            context_prompt = ""
            speaker_names = None
            if user_context:
                from agents.context_agent import create_context_prompt

                # Extract speaker names for later mapping
                speaker_names = user_context.get("speakers")
                context_prompt = create_context_prompt(
                    speaker_names=speaker_names,
                    topic=user_context.get("topic"),
                    technical_terms=user_context.get("terms"),
                    custom_instructions=user_context.get("instructions"),
                    language_hints=user_context.get("language"),
                    expected_format=user_context.get("format"),
                )

                # Combine with custom prompt if provided
                if custom_prompt:
                    custom_prompt = f"{context_prompt}\n\n{custom_prompt}"
                else:
                    custom_prompt = context_prompt

            # Check if the judge pipeline is enabled
            use_judge_pipeline = getattr(
                self.deps.transcription, "use_judge_pipeline", True
            )
            candidates: List[TranscriptCandidate] = []
            judge_notes: List[str] = []
            judge_used = False
            judge_model_used: Optional[str] = None
            judge_selected_candidate_ids: List[str] = []

            if use_judge_pipeline:
                # Use the multi-agent judge pipeline
                (
                    segments,
                    quality,
                    timestamps_corrected,
                    candidates,
                    judge_selected_candidate_ids,
                    judge_notes,
                ) = await self._transcribe_with_judge_pipeline(
                    temp_path,
                    metadata,
                    progress_callback,
                    custom_prompt,
                    speaker_names,
                )
                judge_used = True
                judge_model_used = self.deps.transcription.judge_model_name
            else:
                # Use direct pipeline (legacy mode)
                segments = await self._transcribe_direct(
                    temp_path,
                    metadata,
                    progress_callback,
                    custom_prompt,
                    speaker_names,
                )

                # Calculate quality metrics
                if progress_callback:
                    progress_callback("Analyzing quality...", 0.8)
                quality = await self._calculate_quality(segments, metadata.duration)
                timestamps_corrected = False  # Direct mode doesn't use Parakeet

            # Step 6: Create final result
            assert self.processing_start is not None
            processing_time = (datetime.now() - self.processing_start).total_seconds()

            result = TranscriptResult(
                segments=segments,
                metadata=metadata,
                quality=quality,
                processing_time=processing_time,
                model_used=self.deps.transcription.model_name,
                edited=False,
                timestamps_corrected=timestamps_corrected,
                candidate_strategy=(
                    self.deps.transcription.candidate_strategy
                    if judge_used
                    else "single_gemini"
                ),
                candidates=candidates,
                judge_used=judge_used,
                judge_model_used=judge_model_used,
                judge_selected_candidate_ids=judge_selected_candidate_ids,
                judge_notes=judge_notes,
            )

            # Step 7: Finalize
            if progress_callback:
                progress_callback("Finalizing...", 0.9)

            self.current_status = ProcessingStatus.COMPLETE

            if progress_callback:
                progress_callback("Complete!", 1.0)

            return result

        except Exception as e:
            self.current_status = ProcessingStatus.ERROR
            logger.error(f"Transcription failed: {e}")
            raise

        finally:
            # Cleanup temp files
            self._cleanup_temp_files()

    async def _transcribe_with_judge_pipeline(
        self,
        audio_path: str,
        metadata: AudioMetadata,
        progress_callback: Optional[Callable],
        custom_prompt: Optional[str],
        speaker_names: Optional[List[str]] = None,
    ) -> tuple[
        List[TranscriptSegment],
        TranscriptQuality,
        bool,
        List[TranscriptCandidate],
        List[str],
        List[str],
    ]:
        """Run the candidate fan-out and judge fan-in pipeline."""
        if progress_callback:
            progress_callback("Generating transcript candidates...", 0.3)

        if metadata.needs_chunking:
            audio_units = await chunk_audio(self.active_transcription_deps, audio_path)
        else:
            audio_units = [
                {
                    "path": audio_path,
                    "index": 0,
                    "start_ms": 0,
                    "end_ms": int(metadata.duration * 1000),
                    "duration_ms": int(metadata.duration * 1000),
                }
            ]

        judged_chunks: List[List[TranscriptSegment]] = []
        candidate_chunks: Dict[str, Dict[str, Any]] = {}
        selected_candidate_ids: List[str] = []
        judge_notes: List[str] = []
        previous_context: Optional[str] = None

        for index, unit_info in enumerate(audio_units):
            if progress_callback:
                base_progress = 0.3
                step = 0.4 / max(len(audio_units), 1)
                progress_callback(
                    f"Judging transcript candidates {index + 1}/{len(audio_units)}...",
                    base_progress + (step * index),
                )

            chunk_label = (
                f"chunk {index + 1} of {len(audio_units)}"
                if metadata.needs_chunking
                else "the full audio file"
            )
            unit_chunk_info = unit_info if metadata.needs_chunking else None
            unit_result = await self._run_unit_with_judge(
                unit_info["path"],
                unit_chunk_info,
                custom_prompt,
                previous_context,
                speaker_names,
                chunk_label,
                (
                    (unit_info["duration_ms"] / 1000.0)
                    if metadata.needs_chunking
                    else metadata.duration
                ),
            )
            judged_chunks.append(unit_result.final_segments)
            selected_candidate_ids.extend(unit_result.selected_candidate_ids)
            if metadata.needs_chunking:
                judge_notes.extend(
                    [f"{chunk_label}: {note}" for note in unit_result.judge_notes]
                )
            else:
                judge_notes.extend(unit_result.judge_notes)

            for candidate in unit_result.candidates:
                bucket = candidate_chunks.setdefault(
                    candidate.candidate_id,
                    {
                        "candidate": candidate.model_copy(
                            update={"segments": [], "quality_score": None, "notes": []}
                        ),
                        "chunks": [],
                        "notes": [],
                    },
                )
                bucket["chunks"].append(candidate.segments)
                bucket["notes"].extend(candidate.notes)

            if (
                unit_result.final_segments
                and self.active_transcription_deps.preserve_context
            ):
                previous_context = self._build_followup_context(
                    unit_result.final_segments
                )

        if metadata.needs_chunking:
            final_segments = await merge_chunks(
                self.active_transcription_deps, judged_chunks
            )
        else:
            final_segments = judged_chunks[0] if judged_chunks else []

        candidates = await self._merge_candidate_chunks(
            candidate_chunks,
            speaker_names,
            metadata.duration,
        )

        if speaker_names and final_segments:
            final_segments = map_speakers_to_context(final_segments, speaker_names)

        final_segments, timestamps_corrected, timestamp_notes = (
            await self._review_timestamps(audio_path, metadata, final_segments)
        )
        judge_notes.extend(timestamp_notes)

        if final_segments:
            quality = await self._calculate_quality(final_segments, metadata.duration)
            if judge_notes:
                quality = quality.model_copy(
                    update={"warnings": quality.warnings + judge_notes}
                )
        else:
            quality = TranscriptQuality(
                overall_score=0.0,
                readability=0.0,
                punctuation_density=0.0,
                sentence_variety=0.0,
                vocabulary_richness=0.0,
                timestamp_coverage=0.0,
                speaker_consistency=0.0,
                issues=[{"type": "error", "message": "No segments transcribed"}],
                warnings=judge_notes,
            )

        logger.info(
            f"Judge pipeline complete: {len(final_segments)} segments, "
            f"quality={quality.overall_score:.1f}, "
            f"timestamps_corrected={timestamps_corrected}"
        )

        return (
            final_segments,
            quality,
            timestamps_corrected,
            candidates,
            list(dict.fromkeys(selected_candidate_ids)),
            judge_notes,
        )

    async def _run_unit_with_judge(
        self,
        audio_path: str,
        chunk_info: Optional[Dict[str, Any]],
        custom_prompt: Optional[str],
        previous_context: Optional[str],
        speaker_names: Optional[List[str]],
        chunk_label: str,
        audio_duration: Optional[float] = None,
    ) -> JudgedUnitResult:
        """Generate candidates for one audio span and judge them."""
        candidates = await self._generate_candidates(
            audio_path,
            chunk_info,
            custom_prompt,
            previous_context,
            speaker_names,
            audio_duration,
        )
        valid_candidates = [candidate for candidate in candidates if candidate.segments]

        if not valid_candidates:
            notes = []
            for candidate in candidates:
                notes.extend(candidate.notes)
            if not notes:
                notes = ["No candidate transcription produced segments."]
            return JudgedUnitResult(
                final_segments=[],
                candidates=candidates,
                selected_candidate_ids=[],
                judge_notes=notes,
            )

        judge_decision = await run_judge_agent(
            deps=self.deps,
            candidates=valid_candidates,
            context_prompt=custom_prompt,
            speaker_names=speaker_names,
            chunk_label=chunk_label,
            agent=self.judge_agent,
        )
        judge_notes = list(judge_decision.processing_notes)
        if judge_decision.selected_candidate_ids:
            judge_notes.append(
                "Judge selected: " + ", ".join(judge_decision.selected_candidate_ids)
            )

        final_segments = judge_decision.segments or valid_candidates[0].segments
        return JudgedUnitResult(
            final_segments=final_segments,
            candidates=candidates,
            selected_candidate_ids=list(judge_decision.selected_candidate_ids),
            judge_notes=judge_notes,
        )

    async def _generate_candidates(
        self,
        audio_path: str,
        chunk_info: Optional[Dict[str, Any]],
        custom_prompt: Optional[str],
        previous_context: Optional[str],
        speaker_names: Optional[List[str]],
        audio_duration: Optional[float] = None,
    ) -> List[TranscriptCandidate]:
        """Run the configured candidate transcription plan for one audio span."""
        specs = self.active_transcription_deps.resolve_candidate_specs()
        tasks = [
            self._run_candidate_spec(
                spec,
                audio_path,
                chunk_info,
                custom_prompt,
                previous_context,
                speaker_names,
                audio_duration,
            )
            for spec in specs
        ]
        return list(await asyncio.gather(*tasks))

    async def _run_candidate_spec(
        self,
        spec: Dict[str, str],
        audio_path: str,
        chunk_info: Optional[Dict[str, Any]],
        custom_prompt: Optional[str],
        previous_context: Optional[str],
        speaker_names: Optional[List[str]],
        audio_duration: Optional[float] = None,
    ) -> TranscriptCandidate:
        """Execute a single candidate transcription run."""
        notes: List[str] = []
        segments: List[TranscriptSegment] = []

        try:
            if spec["kind"] == "gemini":
                candidate_deps = replace(
                    self.active_transcription_deps, model_name=spec["model_name"]
                )
                candidate_agent = create_transcription_agent(candidate_deps)
                segments = await run_transcription_agent(
                    candidate_agent,
                    candidate_deps,
                    audio_path,
                    custom_prompt,
                    chunk_info,
                    previous_context,
                    speaker_names,
                )
            else:
                segments = await transcribe_with_parakeet(
                    self.active_transcription_deps,
                    audio_path,
                    speaker_names,
                )
                if chunk_info and chunk_info.get("start_ms"):
                    offset_seconds = chunk_info["start_ms"] / 1000.0
                    if offset_seconds > 0:
                        segments = [
                            segment.model_copy(
                                update={
                                    "timestamp": adjust_timestamp(
                                        segment.timestamp, offset_seconds
                                    )
                                }
                            )
                            for segment in segments
                        ]
            notes.append(f"Generated by {spec['model_name']}")
        except ImportError as exc:
            notes.append(f"{spec['label']} unavailable: {exc}")
        except Exception as exc:
            notes.append(f"{spec['label']} failed: {exc}")

        quality_score = None
        if segments:
            candidate_quality = await self._calculate_quality(segments, audio_duration)
            quality_score = candidate_quality.overall_score

        return TranscriptCandidate(
            candidate_id=spec["candidate_id"],
            label=spec["label"],
            kind=cast(Literal["gemini", "parakeet"], spec["kind"]),
            model_name=spec["model_name"],
            segments=segments,
            quality_score=quality_score,
            notes=notes,
        )

    async def _merge_candidate_chunks(
        self,
        candidate_chunks: Dict[str, Dict[str, Any]],
        speaker_names: Optional[List[str]],
        audio_duration: Optional[float] = None,
    ) -> List[TranscriptCandidate]:
        """Merge per-chunk candidates back into full-audio candidates."""
        merged_candidates = []

        for bucket in candidate_chunks.values():
            candidate = bucket["candidate"]
            chunks = bucket["chunks"]
            if not chunks:
                merged_segments: List[TranscriptSegment] = []
            elif len(chunks) == 1:
                merged_segments = chunks[0]
            else:
                merged_segments = await merge_chunks(
                    self.active_transcription_deps, chunks
                )

            if speaker_names and merged_segments:
                merged_segments = map_speakers_to_context(
                    merged_segments, speaker_names
                )

            quality_score = None
            if merged_segments:
                candidate_quality = await self._calculate_quality(
                    merged_segments,
                    audio_duration,
                )
                quality_score = candidate_quality.overall_score

            merged_candidates.append(
                candidate.model_copy(
                    update={
                        "segments": merged_segments,
                        "quality_score": quality_score,
                        "notes": list(dict.fromkeys(bucket["notes"])),
                    }
                )
            )

        return merged_candidates

    async def _review_timestamps(
        self,
        audio_path: str,
        metadata: AudioMetadata,
        segments: List[TranscriptSegment],
    ) -> tuple[List[TranscriptSegment], bool, List[str]]:
        """Optionally align timestamps after judging."""
        if not segments:
            return segments, False, []

        analysis = analyze_timestamp_quality(segments, metadata.duration)
        notes = [f"Timestamp review: {analysis['reason']}"]
        if analysis["recommendation"] != "fix":
            return segments, False, notes

        try:
            corrected = await fix_timestamps_with_parakeet(
                self.active_transcription_deps,
                audio_path,
                segments,
            )
            notes.append("Applied Parakeet alignment after judging.")
            return corrected, True, notes
        except ImportError as exc:
            notes.append(
                f"Skipped Parakeet alignment because NeMo is unavailable: {exc}"
            )
        except Exception as exc:
            notes.append(f"Parakeet alignment failed: {exc}")

        return segments, False, notes

    def _build_followup_context(self, segments: List[TranscriptSegment]) -> str:
        """Build short context from the tail of the judged transcript."""
        context_segments = segments[-5:] if len(segments) > 5 else segments
        return "\n".join(
            f"{segment.speaker}: {segment.text}" for segment in context_segments
        )

    async def _transcribe_direct(
        self,
        audio_path: str,
        metadata: AudioMetadata,
        progress_callback: Optional[Callable],
        custom_prompt: Optional[str],
        speaker_names: Optional[List[str]] = None,
    ) -> List[TranscriptSegment]:
        """Direct transcription pipeline (legacy mode, no judge)"""

        # Step 3: Transcribe (with chunking if needed)
        if metadata.needs_chunking:
            segments = await self._transcribe_with_chunks(
                audio_path,
                metadata,
                progress_callback,
                custom_prompt,
                speaker_names,
            )
        else:
            if progress_callback:
                progress_callback("Transcribing audio...", 0.5)

            segments = await run_transcription_agent(
                self.transcription_agent,
                self.active_transcription_deps,
                audio_path,
                custom_prompt,
                None,
                None,
                speaker_names,
            )

        # Step 3.5: Map speakers to context names if provided
        if speaker_names:
            segments = map_speakers_to_context(segments, speaker_names)

        # Step 4: Auto-format if enabled
        if self.active_transcription_deps.auto_format:
            if progress_callback:
                progress_callback("Formatting transcript...", 0.7)
            # Auto-formatting handled by editing tools if needed

        return segments

    async def _transcribe_with_chunks(
        self,
        audio_path: str,
        metadata: AudioMetadata,
        progress_callback: Optional[Callable],
        custom_prompt: Optional[str],
        speaker_names: Optional[List[str]] = None,
    ) -> List[TranscriptSegment]:
        """Handle chunked transcription for large files"""

        if progress_callback:
            progress_callback("Splitting audio into chunks...", 0.3)

        # Create chunks directly
        chunks = await chunk_audio(self.active_transcription_deps, audio_path)

        # Transcribe each chunk with context from previous chunks
        all_segments = []
        previous_context = None

        for i, chunk_info in enumerate(chunks):
            if progress_callback:
                progress = 0.3 + (0.4 * (i / len(chunks)))
                progress_callback(
                    f"Transcribing chunk {i+1}/{len(chunks)}...", progress
                )

            # Direct transcription call through the agent
            chunk_segments = await run_transcription_agent(
                self.transcription_agent,
                self.active_transcription_deps,
                chunk_info["path"],
                custom_prompt,
                chunk_info,
                previous_context,
                speaker_names,
            )
            all_segments.append(chunk_segments)

            # Build context for next chunk (last 30 seconds of text)
            if chunk_segments and self.active_transcription_deps.preserve_context:
                # Get last few segments as context
                context_segments = (
                    chunk_segments[-5:] if len(chunk_segments) > 5 else chunk_segments
                )
                previous_context = "\n".join(
                    [f"{seg.speaker}: {seg.text}" for seg in context_segments]
                )

        # Merge chunks directly
        if progress_callback:
            progress_callback("Merging transcription chunks...", 0.7)

        merged_segments = await merge_chunks(
            self.active_transcription_deps, all_segments
        )

        return merged_segments

    async def _calculate_quality(
        self,
        segments: List[TranscriptSegment],
        audio_duration: Optional[float] = None,
    ) -> TranscriptQuality:
        """Calculate quality metrics for transcript"""

        from agents.quality_validator import (
            calculate_quality_metrics,
            calculate_overall_score,
        )

        # Calculate metrics
        metrics = calculate_quality_metrics(
            self.deps.quality,
            segments,
            audio_duration,
        )

        # Calculate overall score
        overall_score = calculate_overall_score(self.deps.quality, metrics)

        # Detect issues - simplified for now
        issues: List[Dict[str, Any]] = []

        return TranscriptQuality(
            overall_score=overall_score,
            readability=metrics["readability"],
            punctuation_density=metrics["punctuation_density"],
            sentence_variety=metrics["sentence_variety"],
            vocabulary_richness=metrics["vocabulary_richness"],
            timestamp_coverage=metrics["timestamp_coverage"],
            speaker_consistency=metrics["speaker_consistency"],
            issues=issues,
            warnings=metrics.get("warnings", []),
        )

    def _cleanup_temp_files(self):
        """Clean up run-scoped temporary files."""
        try:
            if self._run_transcription_deps is not None:
                shutil.rmtree(self._run_transcription_deps.temp_dir, ignore_errors=True)
                self._run_transcription_deps = None
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")

    def _create_run_transcription_deps(self) -> TranscriptionDeps:
        """Create per-run transcription deps with an isolated temp workspace."""
        os.makedirs(self.deps.transcription.temp_dir, exist_ok=True)
        run_temp_dir = tempfile.mkdtemp(
            prefix="run_",
            dir=self.deps.transcription.temp_dir,
        )
        return replace(self.deps.transcription, temp_dir=run_temp_dir)

    async def edit_transcript(
        self, result: TranscriptResult, operation: str, **kwargs
    ) -> TranscriptResult:
        """Apply editing operation to transcript"""

        from agents.editing_tools import (
            auto_format_transcript,
            find_and_replace,
            fix_capitalization,
        )

        if operation == "auto_format":
            # Apply auto-formatting using editing utilities
            formatted_segments, changes = auto_format_transcript(
                self.deps.editing, result.segments
            )
            result.segments = formatted_segments
            result.edited = True

        elif operation == "find_replace":
            # Apply find and replace
            find_text = kwargs.get("find", "")
            replace_text = kwargs.get("replace", "")
            case_sensitive = kwargs.get("case_sensitive", False)
            whole_word = kwargs.get("whole_word", False)

            if find_text:
                replace_result = find_and_replace(
                    result.segments, find_text, replace_text, case_sensitive, whole_word
                )
                result.segments = replace_result["segments"]
                result.edited = True

        elif operation == "fix_capitalization":
            # Apply capitalization fixes
            result.segments = fix_capitalization(result.segments)
            result.edited = True

        # Recalculate quality after editing
        if result.edited:
            result.quality = await self._calculate_quality(
                result.segments,
                result.metadata.duration,
            )

        return result

    async def export_transcript(
        self, result: TranscriptResult, format: str, **options
    ) -> str:
        """Export transcript in specified format"""

        if format == "txt":
            return result.formatted_text

        elif format == "srt":
            return self._export_as_srt(result.segments, **options)

        elif format == "json":
            import json

            return json.dumps(result.model_dump(), indent=2, default=str)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_as_srt(
        self, segments: List[TranscriptSegment], max_line_length: int = 42, **options
    ) -> str:
        """Export segments as SRT subtitle format with safety checks"""

        srt_lines = []
        previous_start = None

        for i, segment in enumerate(segments, 1):
            # Convert timestamp format
            start_time = self._convert_to_srt_time(segment.timestamp)

            # Safety check: ensure monotonic timestamps
            if previous_start:
                start_seconds = self._srt_time_to_seconds(start_time)
                prev_seconds = self._srt_time_to_seconds(previous_start)
                if start_seconds < prev_seconds:
                    # Use previous time + 0.1 seconds for non-monotonic timestamps
                    start_time = self._add_seconds_to_srt_time(previous_start, 0.1)

            previous_start = start_time

            # Calculate end time (approximation)
            if i < len(segments):
                next_start = self._convert_to_srt_time(segments[i].timestamp)
                # Ensure end time >= start time
                if self._srt_time_to_seconds(next_start) <= self._srt_time_to_seconds(
                    start_time
                ):
                    # Use start + 2 seconds if next timestamp is invalid
                    end_time = self._add_seconds_to_srt_time(start_time, 2)
                else:
                    end_time = next_start
            else:
                # Add 3 seconds for last segment
                end_time = self._add_seconds_to_srt_time(start_time, 3)

            # Format text with line breaks
            text = segment.text
            if len(text) > max_line_length:
                # Split into lines
                words = text.split()
                lines: List[str] = []
                current_line: List[str] = []

                for word in words:
                    if len(" ".join(current_line + [word])) <= max_line_length:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(" ".join(current_line))
                        current_line = [word]

                if current_line:
                    lines.append(" ".join(current_line))

                text = "\n".join(lines[:2])  # Max 2 lines

            # Add SRT entry with speaker label
            srt_lines.append(str(i))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(f"{segment.speaker}: {text}")
            srt_lines.append("")  # Empty line between entries

        return "\n".join(srt_lines)

    def _convert_to_srt_time(self, timestamp: str) -> str:
        """Convert [HH:MM:SS] to SRT time format HH:MM:SS,mmm"""
        # Remove brackets and keep colons, add milliseconds
        time_str = timestamp.strip("[]")
        return time_str + ",000"

    def _add_seconds_to_srt_time(self, srt_time: str, seconds: float) -> str:
        """Add seconds (supports fractional) to SRT timestamp"""
        # Parse SRT time (HH:MM:SS,mmm format)
        time_part, _, millis_part = srt_time.partition(",")
        hours, minutes, secs = map(int, time_part.split(":"))
        existing_ms = int(millis_part or "000")

        total_ms = ((hours * 3600) + (minutes * 60) + secs) * 1000 + existing_ms
        delta_ms = int(round(seconds * 1000))
        total_ms = max(0, total_ms + delta_ms)

        new_hours, remainder = divmod(total_ms, 3600000)
        new_minutes, remainder = divmod(remainder, 60000)
        new_seconds, new_milliseconds = divmod(remainder, 1000)

        return f"{new_hours:02d}:{new_minutes:02d}:{new_seconds:02d},{new_milliseconds:03d}"

    def _srt_time_to_seconds(self, srt_time: str) -> float:
        """Convert SRT timestamp to seconds for comparison"""
        # Parse HH:MM:SS,mmm format
        time_part = srt_time.split(",")[0]
        parts = time_part.split(":")
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return 0.0

    def cleanup(self):
        """Clean up all resources"""
        self.deps.cleanup()
