"""
Transcription Agent using Pydantic AI
Handles audio processing and transcription with Google Gemini
"""

from pydantic_ai import Agent, RunContext
from typing import List, Optional, Dict, Any
import tempfile
import asyncio
import logging
from pathlib import Path
import hashlib
import json
import os

from pydub import AudioSegment
import aiofiles
import google.generativeai as genai

from models import (
    TranscriptResult, TranscriptSegment, AudioMetadata, 
    TranscriptQuality, AudioFormat
)
from dependencies import TranscriptionDeps

logger = logging.getLogger(__name__)


# Create main transcription agent
# Use the model name directly without prefix as per Pydantic AI docs
transcription_agent = Agent(
    'gemini-2.5-flash',
    deps_type=TranscriptionDeps,
    output_type=TranscriptResult,
    system_prompt="""You are an expert audio transcription specialist using Gemini 2.5's advanced capabilities.
    
    THINKING APPROACH:
    - Analyze audio quality and speaker patterns before transcribing
    - Use context clues to disambiguate unclear speech
    - Consider domain-specific terminology and proper nouns
    
    FORMAT REQUIREMENTS:
    - Use format: [HH:MM:SS] Speaker X: Text...
    - Maintain consistent speaker labels throughout
    - Include natural paragraph breaks for topic changes
    - Mark non-speech audio as [MUSIC], [SILENCE], [NOISE], [APPLAUSE], etc.
    - End transcription with [END]
    
    QUALITY STANDARDS:
    - Prioritize accuracy over speed
    - Use proper punctuation including commas, periods, question marks
    - Maintain sentence flow and readability
    - Preserve technical terms, acronyms, and proper nouns accurately
    - Add [inaudible] for unclear sections rather than guessing"""
)


@transcription_agent.tool
async def validate_audio_file(
    ctx: RunContext[TranscriptionDeps],
    file_data: bytes,
    filename: str
) -> Dict[str, Any]:
    """Validate audio file before processing"""
    try:
        # Check file size
        size_mb = len(file_data) / (1024 * 1024)
        if size_mb > ctx.deps.max_file_size_mb:
            return {
                "valid": False,
                "error": f"File size ({size_mb:.1f}MB) exceeds limit ({ctx.deps.max_file_size_mb}MB)"
            }
        
        # Get file extension
        ext = Path(filename).suffix.lower().lstrip('.')
        if ext not in ['mp3', 'wav', 'm4a', 'ogg', 'flac']:
            return {
                "valid": False,
                "error": f"Unsupported file format: {ext}"
            }
        
        # Save to temp file for processing
        temp_path = os.path.join(ctx.deps.temp_dir, filename)
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(file_data)
        
        # Load with pydub to validate
        try:
            audio = AudioSegment.from_file(temp_path)
            duration = len(audio) / 1000.0  # Convert to seconds
            
            return {
                "valid": True,
                "temp_path": temp_path,
                "duration": duration,
                "size_mb": size_mb,
                "format": ext,
                "channels": audio.channels,
                "frame_rate": audio.frame_rate
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Failed to load audio file: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "valid": False,
            "error": str(e)
        }


@transcription_agent.tool
async def process_audio_file(
    ctx: RunContext[TranscriptionDeps],
    file_path: str
) -> AudioMetadata:
    """Process and analyze audio file"""
    try:
        audio = AudioSegment.from_file(file_path)
        
        # Get file info
        file_stat = os.stat(file_path)
        filename = Path(file_path).name
        ext = Path(file_path).suffix.lower().lstrip('.')
        
        # Determine if chunking is needed
        duration_ms = len(audio)
        needs_chunking = duration_ms > ctx.deps.chunk_duration_ms
        
        chunk_count = None
        if needs_chunking:
            chunk_count = (duration_ms + ctx.deps.chunk_duration_ms - 1) // ctx.deps.chunk_duration_ms
        
        return AudioMetadata(
            filename=filename,
            duration=duration_ms / 1000.0,
            size_mb=file_stat.st_size / (1024 * 1024),
            format=AudioFormat(ext),
            sample_rate=audio.frame_rate,
            channels=audio.channels,
            needs_chunking=needs_chunking,
            chunk_count=chunk_count
        )
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise


@transcription_agent.tool
async def chunk_audio(
    ctx: RunContext[TranscriptionDeps],
    audio_path: str
) -> List[Dict[str, Any]]:
    """Split audio into chunks for processing"""
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = []
        
        chunk_duration = ctx.deps.chunk_duration_ms
        overlap = ctx.deps.chunk_overlap_ms
        
        for i in range(0, len(audio), chunk_duration - overlap):
            # Calculate chunk boundaries with overlap
            start_ms = max(0, i - (overlap if i > 0 else 0))
            end_ms = min(len(audio), i + chunk_duration)
            
            chunk = audio[start_ms:end_ms]
            
            # Save chunk
            chunk_filename = f"chunk_{len(chunks):03d}.wav"
            chunk_path = os.path.join(ctx.deps.temp_dir, chunk_filename)
            chunk.export(chunk_path, format="wav")
            
            chunks.append({
                "path": chunk_path,
                "index": len(chunks),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms
            })
        
        logger.info(f"Created {len(chunks)} chunks from audio file")
        return chunks
        
    except Exception as e:
        logger.error(f"Error chunking audio: {e}")
        raise


@transcription_agent.tool
async def transcribe_audio(
    ctx: RunContext[TranscriptionDeps],
    audio_path: str,
    custom_prompt: Optional[str] = None,
    chunk_info: Optional[Dict[str, Any]] = None,
    previous_context: Optional[str] = None
) -> List[TranscriptSegment]:
    """Transcribe audio file or chunk using Gemini"""
    try:
        # Check cache first if enabled
        cache_key = None
        if ctx.deps.use_cache:
            # Generate cache key from file content
            with open(audio_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            cache_key = f"{file_hash}_{ctx.deps.model_name}"
            cache_path = os.path.join(ctx.deps.cache_dir, f"{cache_key}.json")
            
            if os.path.exists(cache_path):
                logger.info(f"Using cached transcription for {audio_path}")
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                return [TranscriptSegment(**seg) for seg in cached_data]
        
        # Upload audio to Gemini
        with open(audio_path, 'rb') as f:
            uploaded_file = genai.upload_file(f, mime_type="audio/wav")
        
        # Prepare prompt with Gemini 2.5 optimizations
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = """Using your advanced thinking capabilities, transcribe this audio with maximum accuracy.
            
            REQUIREMENTS:
            1. Include precise timestamps in [HH:MM:SS] format
            2. Identify and label speakers consistently (Speaker 1, Speaker 2, etc.)
            3. Include all spoken content with proper punctuation
            4. Format each line as: [HH:MM:SS] Speaker X: Dialogue text
            5. Mark non-speech audio appropriately ([MUSIC], [SILENCE], [APPLAUSE], etc.)
            6. Add paragraph breaks for topic changes
            7. End with [END]
            
            THINK ABOUT:
            - Audio quality and background noise
            - Speaker characteristics and patterns
            - Context and domain-specific terminology
            - Proper nouns and technical terms"""
        
        # Add previous context for better continuity
        if ctx.deps.preserve_context and previous_context:
            prompt += f"\n\nPREVIOUS CONTEXT (for continuity):\n{previous_context}"
        
        # Add chunk context if provided
        if chunk_info:
            chunk_offset = chunk_info['start_ms'] / 1000.0
            prompt += f"\n\nCHUNK INFO: This is chunk {chunk_info['index']} starting at {chunk_offset:.1f} seconds. Adjust timestamps accordingly."
        
        # Generate transcription with Gemini 2.5 structured output
        generation_config = {
            "temperature": ctx.deps.temperature,
            "max_output_tokens": ctx.deps.max_output_tokens,
            "response_mime_type": "text/plain"  # Ensure text output for transcription
        }
        
        model = genai.GenerativeModel(
            model_name=ctx.deps.model_name,
            generation_config=generation_config
        )
        
        # Add thinking budget for complex audio
        response = await model.generate_content_async([uploaded_file, prompt])
        
        # Parse response into segments
        segments = parse_transcript_response(response.text, chunk_info)
        
        # Cache result if enabled
        if ctx.deps.use_cache and cache_key:
            cache_data = [seg.model_dump() for seg in segments]
            os.makedirs(ctx.deps.cache_dir, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        
        # Clean up uploaded file
        uploaded_file.delete()
        
        return segments
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise


@transcription_agent.tool
async def merge_chunks(
    ctx: RunContext[TranscriptionDeps],
    chunk_results: List[List[TranscriptSegment]]
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
    
    # Ensure speaker consistency across chunks
    merged = ensure_speaker_consistency(merged)
    
    return merged


def parse_transcript_response(
    text: str, 
    chunk_info: Optional[Dict[str, Any]] = None
) -> List[TranscriptSegment]:
    """Parse Gemini response into transcript segments"""
    segments = []
    
    # Calculate time offset for chunks
    time_offset_seconds = 0
    if chunk_info:
        time_offset_seconds = chunk_info['start_ms'] / 1000.0
    
    for line in text.split('\n'):
        line = line.strip()
        if not line or line == '[END]':
            continue
        
        # Parse timestamp pattern [HH:MM:SS]
        if line.startswith('[') and '] ' in line:
            parts = line.split('] ', 1)
            timestamp_str = parts[0] + ']'
            
            # Parse speaker and text
            if len(parts) > 1:
                content = parts[1]
                if ': ' in content:
                    speaker, text = content.split(': ', 1)
                    
                    # Adjust timestamp for chunk offset
                    if chunk_info and time_offset_seconds > 0:
                        timestamp_str = adjust_timestamp(timestamp_str, time_offset_seconds)
                    
                    segments.append(TranscriptSegment(
                        timestamp=timestamp_str,
                        speaker=speaker.strip(),
                        text=text.strip()
                    ))
    
    return segments


def adjust_timestamp(timestamp: str, offset_seconds: float) -> str:
    """Adjust timestamp by adding offset"""
    # Parse [HH:MM:SS] format
    time_str = timestamp.strip('[]')
    parts = time_str.split(':')
    
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


def ensure_speaker_consistency(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """Ensure speaker labels are consistent throughout transcript"""
    if not segments:
        return segments
    
    # Map inconsistent speaker labels
    speaker_map = {}
    normalized_segments = []
    
    for segment in segments:
        speaker = segment.speaker.strip()
        
        # Normalize speaker label
        if speaker.lower().startswith('speaker'):
            # Already in correct format
            normalized_speaker = speaker
        else:
            # Map to Speaker X format
            if speaker not in speaker_map:
                speaker_map[speaker] = f"Speaker {len(speaker_map) + 1}"
            normalized_speaker = speaker_map[speaker]
        
        # Create new segment with normalized speaker
        normalized_segments.append(TranscriptSegment(
            timestamp=segment.timestamp,
            speaker=normalized_speaker,
            text=segment.text,
            confidence=segment.confidence
        ))
    
    return normalized_segments