"""
Transcription Agent using Pydantic AI
Handles audio processing and transcription with Google Gemini
"""

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.google import GoogleModelSettings
from typing import List, Optional, Dict, Any
import asyncio
import logging
from pathlib import Path
import os
import re
import uuid
import mimetypes

from pydub import AudioSegment
import aiofiles

from models import TranscriptSegment, AudioMetadata, AudioFormat
from dependencies import TranscriptionDeps

logger = logging.getLogger(__name__)


# Create main transcription agent with proper Pydantic AI setup
def create_transcription_agent(deps: TranscriptionDeps) -> Agent:
    """Create a properly configured transcription agent"""
    # Note: API key is already set in TranscriptionDeps.__post_init__

    # Use model name string - Pydantic AI handles the rest
    # Format: 'google-gla:model-name' for Google models
    model_name = (
        f"google-gla:{deps.model_name}"
        if not deps.model_name.startswith("google-gla:")
        else deps.model_name
    )

    agent = Agent(
        model_name,
        deps_type=TranscriptionDeps,
        output_type=List[TranscriptSegment],
        system_prompt="""You are an expert audio transcription specialist using Gemini's advanced capabilities.

    OBJECTIVE:
    - Produce highly accurate transcripts for any supplied audio.
    - Return only data that validates against the TranscriptSegment schema:
      * timestamp: string in the form [HH:MM:SS]
      * speaker: consistent label or provided speaker name
      * text: cleaned utterance with natural punctuation
      * confidence: optional float between 0 and 1 when you can estimate certainty

    THINKING APPROACH:
    - Analyze audio quality and speaker patterns before transcribing
    - Use context clues to disambiguate unclear speech
    - Consider domain-specific terminology and proper nouns

    DELIVERY RULES:
    - Maintain consistent speaker labels throughout
    - Insert non-speech events as [MUSIC], [SILENCE], [NOISE], [APPLAUSE], etc.
    - Preserve readability with sentence-level punctuation
    - Prefer accuracy over speed; use [inaudible] rather than guessing
    - Do not include commentary outside of the structured transcript""",
    )

    return agent


# Initialize default agent for backward compatibility
transcription_agent: Optional[Agent[TranscriptionDeps, List[TranscriptSegment]]] = None


async def validate_audio_file(
    deps: TranscriptionDeps, file_data: bytes, filename: str
) -> Dict[str, Any]:
    """Validate audio file before processing"""
    try:
        # Check file size
        size_mb = len(file_data) / (1024 * 1024)
        if size_mb > deps.max_file_size_mb:
            return {
                "valid": False,
                "error": f"File size ({size_mb:.1f}MB) exceeds limit ({deps.max_file_size_mb}MB)",
            }

        # Get file extension
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in ["mp3", "wav", "m4a", "ogg", "flac"]:
            return {"valid": False, "error": f"Unsupported file format: {ext}"}

        # Create safe temp filename with UUID to avoid collisions
        # Sanitize original filename and add UUID
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(filename).stem)
        unique_id = str(uuid.uuid4())[:8]
        temp_filename = f"{safe_name}_{unique_id}.{ext}"
        temp_path = os.path.join(deps.temp_dir, temp_filename)
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(file_data)

        # Load with pydub to validate (run in thread to avoid blocking)
        try:
            audio = await asyncio.to_thread(AudioSegment.from_file, temp_path)
            duration = len(audio) / 1000.0  # Convert to seconds

            return {
                "valid": True,
                "temp_path": temp_path,
                "duration": duration,
                "size_mb": size_mb,
                "format": ext,
                "channels": audio.channels,
                "frame_rate": audio.frame_rate,
            }
        except Exception as e:
            return {"valid": False, "error": f"Failed to load audio file: {str(e)}"}

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {"valid": False, "error": str(e)}


async def process_audio_file(
    deps: TranscriptionDeps, file_path: str
) -> AudioMetadata:
    """Process and analyze audio file"""
    try:
        # Load audio in thread to avoid blocking
        audio = await asyncio.to_thread(AudioSegment.from_file, file_path)

        # Get file info
        file_stat = os.stat(file_path)
        filename = Path(file_path).name
        ext = Path(file_path).suffix.lower().lstrip(".")

        # Determine if chunking is needed
        duration_ms = len(audio)
        needs_chunking = duration_ms > deps.chunk_duration_ms

        chunk_count = None
        if needs_chunking:
            chunk_count = (
                duration_ms + deps.chunk_duration_ms - 1
            ) // deps.chunk_duration_ms

        return AudioMetadata(
            filename=filename,
            duration=duration_ms / 1000.0,
            size_mb=file_stat.st_size / (1024 * 1024),
            format=AudioFormat(ext),
            sample_rate=audio.frame_rate,
            channels=audio.channels,
            needs_chunking=needs_chunking,
            chunk_count=chunk_count,
        )
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise


async def chunk_audio(
    deps: TranscriptionDeps, audio_path: str
) -> List[Dict[str, Any]]:
    """Split audio into chunks for processing"""
    try:
        # Load audio in thread to avoid blocking
        audio = await asyncio.to_thread(AudioSegment.from_file, audio_path)
        chunks = []

        chunk_duration = deps.chunk_duration_ms
        overlap = deps.chunk_overlap_ms

        # Fix: Use proper step size without double-applying overlap
        step_size = chunk_duration - overlap

        for i in range(0, len(audio), step_size):
            # Calculate chunk boundaries - overlap is already handled by step_size
            start_ms = i
            end_ms = min(len(audio), i + chunk_duration)

            chunk = audio[start_ms:end_ms]

            # Save chunk (in thread to avoid blocking)
            chunk_filename = f"chunk_{len(chunks):03d}.wav"
            chunk_path = os.path.join(deps.temp_dir, chunk_filename)
            await asyncio.to_thread(chunk.export, chunk_path, format="wav")

            chunks.append(
                {
                    "path": chunk_path,
                    "index": len(chunks),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
                }
            )

        logger.info(f"Created {len(chunks)} chunks from audio file")
        return chunks

    except Exception as e:
        logger.error(f"Error chunking audio: {e}")
        raise


async def run_transcription_agent(
    agent: Agent,
    deps: TranscriptionDeps,
    audio_path: str,
    custom_prompt: Optional[str] = None,
    chunk_info: Optional[Dict[str, Any]] = None,
    previous_context: Optional[str] = None,
    speaker_names: Optional[List[str]] = None,
) -> List[TranscriptSegment]:
    """Transcribe audio file or chunk using the configured Pydantic AI agent"""

    async with aiofiles.open(audio_path, "rb") as audio_file:
        audio_bytes = await audio_file.read()

    media_type = _guess_media_type(audio_path)

    # Validate content type
    content = BinaryContent(data=audio_bytes, media_type=media_type)
    if not content.is_audio:
        logger.warning(f"Content may not be audio: {media_type}")

    prompt = build_transcription_prompt(
        custom_prompt, previous_context, chunk_info, speaker_names
    )

    model_settings = _build_google_settings(deps)

    logger.info(f"Running transcription agent for {audio_path}")
    result = await agent.run(
        [prompt, content],
        deps=deps,
        model_settings=model_settings,
    )

    segments: List[TranscriptSegment] = result.output or []

    # Warn if transcription returned empty
    if not segments:
        logger.warning(f"Transcription returned empty result for {audio_path}")

    # Adjust timestamps if chunked
    if chunk_info and chunk_info.get("start_ms"):
        offset_seconds = chunk_info["start_ms"] / 1000.0
        if offset_seconds > 0:
            segments = [
                segment.model_copy(
                    update={
                        "timestamp": adjust_timestamp(segment.timestamp, offset_seconds)
                    }
                )
                for segment in segments
            ]

    # Note: Speaker mapping is handled by workflow.py after all chunks are merged

    logger.info(f"Transcribed {len(segments)} segments via agent")
    return segments


def _guess_media_type(path: str) -> str:
    """Infer media type from file path for Gemini uploads"""
    media_type, _ = mimetypes.guess_type(path)
    return media_type or "audio/wav"


def _build_google_settings(deps: TranscriptionDeps) -> GoogleModelSettings:
    """Build Google model settings from dependencies"""
    settings_kwargs: Dict[str, Any] = {
        "temperature": deps.temperature,
        "max_tokens": deps.max_output_tokens,
    }

    thinking_config: Dict[str, Any] = {}
    if deps.thinking_budget >= 0:
        thinking_config["thinking_budget"] = deps.thinking_budget
    if deps.enable_thought_summaries:
        thinking_config["include_thoughts"] = True
    if thinking_config:
        settings_kwargs["google_thinking_config"] = thinking_config

    return GoogleModelSettings(**settings_kwargs)


def build_transcription_prompt(
    custom_prompt: Optional[str],
    previous_context: Optional[str],
    chunk_info: Optional[Dict[str, Any]],
    speaker_names: Optional[List[str]],
) -> str:
    """Build a comprehensive transcription prompt"""

    base_prompt = """Transcribe this audio with maximum accuracy.

Return structured transcript data that conforms to the TranscriptSegment schema.
For each segment include:
- timestamp: string formatted as [HH:MM:SS]
- speaker: consistent speaker label or provided name
- text: cleaned spoken content with natural punctuation
- confidence: optional float between 0 and 1 when you can estimate certainty

Focus on accuracy, preserve technical terms, and avoid speculative guesses (use [inaudible] when unsure)."""

    parts = [base_prompt]

    if speaker_names:
        speakers_str = ", ".join(speaker_names)
        parts.append(f"\nKNOWN SPEAKERS: {speakers_str}")
        parts.append("Use these exact speaker names in your transcription.")

    if previous_context:
        parts.append(f"\nPREVIOUS CONTEXT:\n{previous_context}")

    if chunk_info:
        chunk_number = chunk_info["index"] + 1
        chunk_offset = chunk_info["start_ms"] / 1000.0
        parts.append(
            f"\nCHUNK INFO: This is chunk {chunk_number} starting at {chunk_offset:.1f} seconds."
        )

    if custom_prompt:
        parts.append(f"\nADDITIONAL INSTRUCTIONS:\n{custom_prompt}")

    return "\n".join(parts)


async def merge_chunks(
    deps: TranscriptionDeps, chunk_results: List[List[TranscriptSegment]]
) -> List[TranscriptSegment]:
    """Merge transcription chunks into a single transcript"""
    merged = []

    for chunk_segments in chunk_results:
        # Remove duplicates from overlap regions
        if merged and chunk_segments:
            # Check for overlap with last segment of previous chunk
            last_text = merged[-1].text.lower().strip()
            first_text = chunk_segments[0].text.lower().strip()

            # Simple duplicate detection
            if last_text == first_text or last_text.endswith(first_text[:20]):
                chunk_segments = chunk_segments[1:]  # Skip first segment

        merged.extend(chunk_segments)

    # Ensure speaker consistency across chunks (preserve real names)
    merged = ensure_speaker_consistency(merged, preserve_names=True)

    return merged


def adjust_timestamp(timestamp: str, offset_seconds: float) -> str:
    """Adjust timestamp by adding offset"""
    # Parse [HH:MM:SS] format
    time_str = timestamp.strip("[]")
    parts = time_str.split(":")

    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])

        # Convert to total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds + offset_seconds

        # Convert back to HH:MM:SS
        new_hours = int(total_seconds // 3600)
        new_minutes = int((total_seconds % 3600) // 60)
        new_seconds = int(total_seconds % 60)

        return f"[{new_hours:02d}:{new_minutes:02d}:{new_seconds:02d}]"

    return timestamp


def map_speakers_to_context(
    segments: List[TranscriptSegment], speaker_names: Optional[List[str]] = None
) -> List[TranscriptSegment]:
    """Map generic speaker labels to actual names from context

    Args:
        segments: List of transcript segments
        speaker_names: List of actual speaker names from user context
    """
    if not segments or not speaker_names:
        return segments

    # Build a mapping from generic labels to provided names
    generic_pattern = re.compile(r"^Speaker\s*(\d+)$", re.IGNORECASE)

    mapped_segments = []
    for segment in segments:
        speaker = segment.speaker.strip()

        # Check if this is a generic label
        match = generic_pattern.match(speaker)
        if match:
            speaker_num = int(match.group(1))
            # Map to provided name if available (1-indexed)
            if speaker_num <= len(speaker_names):
                mapped_speaker = speaker_names[speaker_num - 1]
            else:
                mapped_speaker = speaker  # Keep generic if no name provided
        else:
            # Not a generic label, keep as is (might already be a real name)
            mapped_speaker = speaker

        mapped_segments.append(
            TranscriptSegment(
                timestamp=segment.timestamp,
                speaker=mapped_speaker,
                text=segment.text,
                confidence=segment.confidence,
            )
        )

    return mapped_segments


def ensure_speaker_consistency(
    segments: List[TranscriptSegment], preserve_names: bool = True
) -> List[TranscriptSegment]:
    """Ensure speaker labels are consistent throughout transcript

    Args:
        segments: List of transcript segments
        preserve_names: If True, keep actual names; if False, normalize to Speaker X
    """
    if not segments:
        return segments

    # Map inconsistent speaker labels
    speaker_map = {}
    normalized_segments = []
    speaker_counter = 1

    for segment in segments:
        speaker = segment.speaker.strip()

        # Check if this looks like a real name (not "Speaker X" format)
        is_generic_label = speaker.lower().startswith("speaker") and any(
            c.isdigit() for c in speaker
        )

        if preserve_names and not is_generic_label:
            # Keep the actual name, just ensure consistency
            if speaker not in speaker_map:
                speaker_map[speaker] = speaker
            normalized_speaker = speaker_map[speaker]
        elif is_generic_label:
            # Already in Speaker X format, keep it
            normalized_speaker = speaker
        else:
            # Map unknown/inconsistent labels to Speaker X
            if speaker not in speaker_map:
                speaker_map[speaker] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            normalized_speaker = speaker_map[speaker]

        # Create new segment with normalized speaker
        normalized_segments.append(
            TranscriptSegment(
                timestamp=segment.timestamp,
                speaker=normalized_speaker,
                text=segment.text,
                confidence=segment.confidence,
            )
        )

    return normalized_segments
