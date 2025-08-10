"""
Main Workflow Coordinator using Pydantic AI
Orchestrates the entire transcription pipeline
"""

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from typing import Optional, Callable, List, Dict, Any
import asyncio
import logging
from datetime import datetime
import os

from models import (
    TranscriptResult, TranscriptSegment, AudioMetadata,
    TranscriptQuality, ProcessingStatus, ErrorDetail
)
from dependencies import AppDeps, TranscriptionDeps

# Import agent tools
from agents.transcription_agent import (
    transcription_agent, 
    validate_audio_file,
    process_audio_file,
    chunk_audio,
    transcribe_audio,
    merge_chunks
)
from agents.quality_validator import (
    quality_agent,
    validate_transcript_quality,
    analyze_readability,
    detect_errors
)
from agents.editing_tools import (
    editing_agent,
    auto_format_transcript,
    find_and_replace,
    fix_capitalization
)

logger = logging.getLogger(__name__)


class TranscriptionWorkflow:
    """Main workflow coordinator for transcription pipeline"""
    
    def __init__(self, api_key: str, **kwargs):
        """Initialize workflow with dependencies"""
        self.deps = AppDeps.from_config(api_key=api_key, **kwargs)
        self.model = self.deps.transcription.get_google_model()
        
        # Initialize orchestrator agent
        self.orchestrator = Agent(
            self.model,
            deps_type=AppDeps,
            result_type=TranscriptResult,
            system_prompt="""You are the transcription workflow orchestrator.
            Coordinate the transcription pipeline:
            1. Validate and process audio files
            2. Chunk if necessary for large files
            3. Transcribe using Gemini
            4. Validate quality
            5. Apply auto-formatting if enabled
            6. Return final transcript with quality metrics"""
        )
        
        # Register all tools from sub-agents
        self._register_tools()
        
        # Track processing state
        self.current_status = ProcessingStatus.IDLE
        self.current_file = None
        self.processing_start = None
    
    def _register_tools(self):
        """Register all workflow tools"""
        # Transcription tools
        self.orchestrator.tools.extend([
            validate_audio_file,
            process_audio_file,
            chunk_audio,
            transcribe_audio,
            merge_chunks
        ])
        
        # Quality tools
        self.orchestrator.tools.extend([
            analyze_readability,
            detect_errors
        ])
        
        # Editing tools
        self.orchestrator.tools.extend([
            auto_format_transcript,
            fix_capitalization
        ])
    
    async def transcribe_audio(
        self,
        file_data: bytes,
        filename: str,
        progress_callback: Optional[Callable] = None,
        custom_prompt: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> TranscriptResult:
        """Main transcription workflow with streaming progress"""
        
        self.current_status = ProcessingStatus.PROCESSING
        self.current_file = filename
        self.processing_start = datetime.now()
        
        try:
            # Step 1: Validate file
            if progress_callback:
                progress_callback("Validating audio file...", 0.1)
            
            validation = await validate_audio_file.fn(
                self.deps.transcription,
                file_data,
                filename
            )
            
            if not validation["valid"]:
                raise ValueError(validation.get("error", "Invalid audio file"))
            
            # Step 2: Process audio metadata
            if progress_callback:
                progress_callback("Processing audio...", 0.2)
            
            metadata = await process_audio_file.fn(
                self.deps.transcription,
                validation["temp_path"]
            )
            
            # Step 2.5: Process user context if provided
            context_prompt = ""
            if user_context:
                from agents.context_agent import create_context_prompt
                context_prompt = create_context_prompt(
                    speaker_names=user_context.get('speakers'),
                    topic=user_context.get('topic'),
                    technical_terms=user_context.get('terms'),
                    custom_instructions=user_context.get('instructions'),
                    language_hints=user_context.get('language'),
                    expected_format=user_context.get('format')
                )
                
                # Combine with custom prompt if provided
                if custom_prompt:
                    custom_prompt = f"{context_prompt}\n\n{custom_prompt}"
                else:
                    custom_prompt = context_prompt
            
            # Step 3: Transcribe (with chunking if needed)
            if metadata.needs_chunking:
                segments = await self._transcribe_with_chunks(
                    validation["temp_path"],
                    metadata,
                    progress_callback,
                    custom_prompt
                )
            else:
                if progress_callback:
                    progress_callback("Transcribing audio...", 0.5)
                
                segments = await transcribe_audio.fn(
                    self.deps.transcription,
                    validation["temp_path"],
                    custom_prompt
                )
            
            # Step 4: Auto-format if enabled
            if self.deps.transcription.auto_format:
                if progress_callback:
                    progress_callback("Formatting transcript...", 0.7)
                
                segments, _ = await auto_format_transcript.fn(
                    self.deps.editing,
                    segments
                )
            
            # Step 5: Calculate quality metrics
            if progress_callback:
                progress_callback("Analyzing quality...", 0.8)
            
            quality = await self._calculate_quality(segments)
            
            # Step 6: Create final result
            processing_time = (datetime.now() - self.processing_start).total_seconds()
            
            result = TranscriptResult(
                segments=segments,
                metadata=metadata,
                quality=quality,
                processing_time=processing_time,
                model_used=self.deps.transcription.model_name,
                edited=False
            )
            
            # Step 7: Validate with quality agent
            if progress_callback:
                progress_callback("Finalizing...", 0.9)
            
            result = await validate_transcript_quality.fn(
                self.deps.quality,
                result
            )
            
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
    
    async def _transcribe_with_chunks(
        self,
        audio_path: str,
        metadata: AudioMetadata,
        progress_callback: Optional[Callable],
        custom_prompt: Optional[str]
    ) -> List[TranscriptSegment]:
        """Handle chunked transcription for large files"""
        
        if progress_callback:
            progress_callback("Splitting audio into chunks...", 0.3)
        
        # Create chunks
        chunks = await chunk_audio.fn(
            self.deps.transcription,
            audio_path
        )
        
        # Transcribe each chunk
        all_segments = []
        for i, chunk_info in enumerate(chunks):
            if progress_callback:
                progress = 0.3 + (0.4 * (i / len(chunks)))
                progress_callback(
                    f"Transcribing chunk {i+1}/{len(chunks)}...",
                    progress
                )
            
            chunk_segments = await transcribe_audio.fn(
                self.deps.transcription,
                chunk_info["path"],
                custom_prompt,
                chunk_info
            )
            all_segments.append(chunk_segments)
        
        # Merge chunks
        if progress_callback:
            progress_callback("Merging transcription chunks...", 0.7)
        
        merged_segments = await merge_chunks.fn(
            self.deps.transcription,
            all_segments
        )
        
        return merged_segments
    
    async def _calculate_quality(self, segments: List[TranscriptSegment]) -> TranscriptQuality:
        """Calculate quality metrics for transcript"""
        
        from agents.quality_validator import calculate_quality_metrics, calculate_overall_score
        
        # Calculate metrics
        metrics = calculate_quality_metrics(self.deps.quality, segments)
        
        # Calculate overall score
        overall_score = calculate_overall_score(self.deps.quality, metrics)
        
        # Detect issues
        issues = await detect_errors.fn(self.deps.quality, segments)
        
        return TranscriptQuality(
            overall_score=overall_score,
            readability=metrics['readability'],
            punctuation_density=metrics['punctuation_density'],
            sentence_variety=metrics['sentence_variety'],
            vocabulary_richness=metrics['vocabulary_richness'],
            timestamp_coverage=metrics['timestamp_coverage'],
            speaker_consistency=metrics['speaker_consistency'],
            issues=issues,
            warnings=metrics.get('warnings', [])
        )
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            temp_dir = self.deps.transcription.temp_dir
            
            if os.path.exists(temp_dir):
                # Only clean up audio files, keep cache
                for file in os.listdir(temp_dir):
                    if file.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
                        file_path = os.path.join(temp_dir, file)
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
                
                logger.info(f"Cleaned up temp files in {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")
    
    async def edit_transcript(
        self,
        result: TranscriptResult,
        operation: str,
        **kwargs
    ) -> TranscriptResult:
        """Apply editing operation to transcript"""
        
        if operation == "auto_format":
            segments, changes = await auto_format_transcript.fn(
                self.deps.editing,
                result.segments
            )
            result.segments = segments
            result.edited = True
            
        elif operation == "find_replace":
            response = await find_and_replace.fn(
                self.deps.editing,
                result.segments,
                kwargs.get('find', ''),
                kwargs.get('replace', ''),
                kwargs.get('case_sensitive', False),
                kwargs.get('whole_word', False)
            )
            if response['success']:
                result.segments = response['segments']
                result.edited = True
        
        elif operation == "fix_capitalization":
            segments = await fix_capitalization.fn(
                self.deps.editing,
                result.segments
            )
            result.segments = segments
            result.edited = True
        
        # Recalculate quality after editing
        if result.edited:
            result.quality = await self._calculate_quality(result.segments)
        
        return result
    
    async def export_transcript(
        self,
        result: TranscriptResult,
        format: str,
        **options
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
        self,
        segments: List[TranscriptSegment],
        max_line_length: int = 42,
        **options
    ) -> str:
        """Export segments as SRT subtitle format"""
        
        srt_lines = []
        
        for i, segment in enumerate(segments, 1):
            # Convert timestamp format
            start_time = self._convert_to_srt_time(segment.timestamp)
            
            # Calculate end time (approximation)
            if i < len(segments):
                end_time = self._convert_to_srt_time(segments[i].timestamp)
            else:
                # Add 3 seconds for last segment
                end_time = self._add_seconds_to_srt_time(start_time, 3)
            
            # Format text with line breaks
            text = segment.text
            if len(text) > max_line_length:
                # Split into lines
                words = text.split()
                lines = []
                current_line = []
                
                for word in words:
                    if len(' '.join(current_line + [word])) <= max_line_length:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                text = '\n'.join(lines[:2])  # Max 2 lines
            
            # Add SRT entry
            srt_lines.append(str(i))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between entries
        
        return '\n'.join(srt_lines)
    
    def _convert_to_srt_time(self, timestamp: str) -> str:
        """Convert [HH:MM:SS] to SRT time format"""
        # Remove brackets and convert to SRT format
        time_str = timestamp.strip('[]')
        return time_str.replace(':', ',', 2) + ',000'
    
    def _add_seconds_to_srt_time(self, srt_time: str, seconds: int) -> str:
        """Add seconds to SRT timestamp"""
        # Parse SRT time
        parts = srt_time.replace(',', ':').split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        secs = int(parts[2])
        
        # Add seconds
        secs += seconds
        if secs >= 60:
            minutes += secs // 60
            secs = secs % 60
        if minutes >= 60:
            hours += minutes // 60
            minutes = minutes % 60
        
        return f"{hours:02d},{minutes:02d},{secs:02d},000"
    
    def cleanup(self):
        """Clean up all resources"""
        self.deps.cleanup()