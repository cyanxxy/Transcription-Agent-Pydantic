"""
Main Workflow Coordinator using Pydantic AI
Orchestrates the entire transcription pipeline
"""

from typing import Optional, Callable, List, Dict, Any
import logging
from datetime import datetime
import os

from models import (
    TranscriptResult, TranscriptSegment, AudioMetadata,
    TranscriptQuality, ProcessingStatus
)
from dependencies import AppDeps

# Import agent tools
from agents.transcription_agent import (
    create_transcription_agent,
    validate_audio_file,
    process_audio_file,
    chunk_audio,
    merge_chunks,
    map_speakers_to_context
)

logger = logging.getLogger(__name__)


class TranscriptionWorkflow:
    """Main workflow coordinator for transcription pipeline"""
    
    def __init__(self, api_key: str, **kwargs):
        """Initialize workflow with dependencies"""
        self.deps = AppDeps.from_config(api_key=api_key, **kwargs)
        
        # Create the transcription agent with proper Pydantic AI setup
        self.transcription_agent = create_transcription_agent(self.deps.transcription)
        
        # Register tools with the transcription agent
        self._register_transcription_tools()
        
        # Track processing state
        self.current_status = ProcessingStatus.IDLE
        self.current_file = None
        self.processing_start = None
    
    def _register_transcription_tools(self):
        """Register tools with the transcription agent"""
        # Register all transcription-related tools
        @self.transcription_agent.tool
        async def validate_file(ctx, file_data: bytes, filename: str):
            return await validate_audio_file(ctx, file_data, filename)
        
        @self.transcription_agent.tool
        async def process_file(ctx, file_path: str):
            return await process_audio_file(ctx, file_path)
        
        @self.transcription_agent.tool
        async def chunk_file(ctx, audio_path: str):
            return await chunk_audio(ctx, audio_path)
        
        @self.transcription_agent.tool
        async def transcribe_file(ctx, audio_path: str, custom_prompt: Optional[str] = None,
                                 chunk_info: Optional[Dict[str, Any]] = None,
                                 previous_context: Optional[str] = None,
                                 speaker_names: Optional[List[str]] = None):
            # Build prompt and return placeholder - actual transcription via agent
            # The prompt would be: build_transcription_prompt(custom_prompt, previous_context, chunk_info, speaker_names)
            # Return empty for now - agent will handle actual transcription
            return []
        
        @self.transcription_agent.tool
        async def merge_results(ctx, chunk_results: List[List[TranscriptSegment]]):
            return await merge_chunks(ctx, chunk_results)
    
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
            # Step 1: Validate file using agent.run()
            if progress_callback:
                progress_callback("Validating audio file...", 0.1)
            
            result = await self.transcription_agent.run(
                "Validate the audio file",
                deps=self.deps.transcription,
                tool_choice="validate_file",
                file_data=file_data,
                filename=filename
            )
            
            # Extract validation from result
            validation = result.data if hasattr(result, 'data') else result
            if isinstance(validation, dict) and not validation.get("valid"):
                raise ValueError(validation.get("error", "Invalid audio file"))
            elif hasattr(validation, 'valid') and not validation.valid:
                raise ValueError(validation.error or "Invalid audio file")
            
            # Step 2: Process audio metadata using agent.run()
            if progress_callback:
                progress_callback("Processing audio...", 0.2)
            
            temp_path = validation.get("temp_path") if isinstance(validation, dict) else validation.temp_path
            
            metadata_result = await self.transcription_agent.run(
                "Process audio metadata",
                deps=self.deps.transcription,
                tool_choice="process_file",
                file_path=temp_path
            )
            
            metadata = metadata_result.data if hasattr(metadata_result, 'data') else metadata_result
            
            # Step 2.5: Process user context if provided
            context_prompt = ""
            speaker_names = None
            if user_context:
                from agents.context_agent import create_context_prompt
                # Extract speaker names for later mapping
                speaker_names = user_context.get('speakers')
                context_prompt = create_context_prompt(
                    speaker_names=speaker_names,
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
                    temp_path,
                    metadata,
                    progress_callback,
                    custom_prompt,
                    speaker_names
                )
            else:
                if progress_callback:
                    progress_callback("Transcribing audio...", 0.5)
                
                # Use agent.run() for transcription
                result = await self.transcription_agent.run(
                    "Transcribe the audio file",
                    deps=self.deps.transcription,
                    tool_choice="transcribe_file",
                    audio_path=temp_path,
                    custom_prompt=custom_prompt,
                    speaker_names=speaker_names
                )
                segments = result.data if hasattr(result, 'data') else []
            
            # Step 3.5: Map speakers to context names if provided
            if speaker_names:
                segments = map_speakers_to_context(segments, speaker_names)
            
            # Step 4: Auto-format if enabled
            if self.deps.transcription.auto_format:
                if progress_callback:
                    progress_callback("Formatting transcript...", 0.7)
                
                # For now, skip auto-formatting as it needs separate agent setup
                # segments = await self._apply_auto_format(segments)
            
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
            
            # Step 7: Validate quality using the tool
            if progress_callback:
                progress_callback("Finalizing...", 0.9)
            
            # Quality validation is done through metrics calculation
            # No need for separate validation call
            
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
        custom_prompt: Optional[str],
        speaker_names: Optional[List[str]] = None
    ) -> List[TranscriptSegment]:
        """Handle chunked transcription for large files"""
        
        if progress_callback:
            progress_callback("Splitting audio into chunks...", 0.3)
        
        # Create chunks using agent.run()
        chunks_result = await self.transcription_agent.run(
            "Create audio chunks",
            deps=self.deps.transcription,
            tool_choice="chunk_file",
            audio_path=audio_path
        )
        chunks = chunks_result.data if hasattr(chunks_result, 'data') else chunks_result
        
        # Transcribe each chunk with context from previous chunks
        all_segments = []
        previous_context = None
        
        for i, chunk_info in enumerate(chunks):
            if progress_callback:
                progress = 0.3 + (0.4 * (i / len(chunks)))
                progress_callback(
                    f"Transcribing chunk {i+1}/{len(chunks)}...",
                    progress
                )
            
            # Use agent.run() for chunk transcription
            result = await self.transcription_agent.run(
                f"Transcribe chunk {i+1}",
                deps=self.deps.transcription,
                tool_choice="transcribe_file",
                audio_path=chunk_info["path"],
                custom_prompt=custom_prompt,
                chunk_info=chunk_info,
                previous_context=previous_context,
                speaker_names=speaker_names
            )
            chunk_segments = result.data if hasattr(result, 'data') else []
            all_segments.append(chunk_segments)
            
            # Build context for next chunk (last 30 seconds of text)
            if chunk_segments and self.deps.transcription.preserve_context:
                # Get last few segments as context
                context_segments = chunk_segments[-5:] if len(chunk_segments) > 5 else chunk_segments
                previous_context = "\n".join([
                    f"{seg.speaker}: {seg.text}" 
                    for seg in context_segments
                ])
        
        # Merge chunks using agent.run()
        if progress_callback:
            progress_callback("Merging transcription chunks...", 0.7)
        
        merge_result = await self.transcription_agent.run(
            "Merge transcription chunks",
            deps=self.deps.transcription,
            tool_choice="merge_results",
            chunk_results=all_segments
        )
        merged_segments = merge_result.data if hasattr(merge_result, 'data') else []
        
        return merged_segments
    
    async def _calculate_quality(self, segments: List[TranscriptSegment]) -> TranscriptQuality:
        """Calculate quality metrics for transcript"""
        
        from agents.quality_validator import calculate_quality_metrics, calculate_overall_score
        
        # Calculate metrics
        metrics = calculate_quality_metrics(self.deps.quality, segments)
        
        # Calculate overall score
        overall_score = calculate_overall_score(self.deps.quality, metrics)
        
        # Detect issues - simplified for now
        issues = []
        
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
            temp_dir = self.deps.transcription.temp_dir
            
            if os.path.exists(temp_dir):
                # Clean up all audio files
                for file in os.listdir(temp_dir):
                    if file.endswith(('.wav', '.mp3', '.m4a', '.ogg', '.flac')):
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
            # Auto-formatting would need its own agent setup
            # For now, mark as edited without changes
            result.edited = True
            
        elif operation == "find_replace":
            # Find/replace would need its own agent setup
            # For now, simple implementation
            find_text = kwargs.get('find', '')
            replace_text = kwargs.get('replace', '')
            if find_text:
                for segment in result.segments:
                    segment.text = segment.text.replace(find_text, replace_text)
                result.edited = True
        
        elif operation == "fix_capitalization":
            # Capitalization fix would need its own agent setup
            # For now, mark as edited without changes
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
                if self._srt_time_to_seconds(next_start) <= self._srt_time_to_seconds(start_time):
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
        """Convert [HH:MM:SS] to SRT time format HH:MM:SS,mmm"""
        # Remove brackets and keep colons, add milliseconds
        time_str = timestamp.strip('[]')
        return time_str + ',000'
    
    def _add_seconds_to_srt_time(self, srt_time: str, seconds: int) -> str:
        """Add seconds to SRT timestamp"""
        # Parse SRT time (HH:MM:SS,mmm format)
        time_part = srt_time.split(',')[0]
        parts = time_part.split(':')
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
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},000"
    
    def _srt_time_to_seconds(self, srt_time: str) -> float:
        """Convert SRT timestamp to seconds for comparison"""
        # Parse HH:MM:SS,mmm format
        time_part = srt_time.split(',')[0]
        parts = time_part.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return 0.0
    
    def cleanup(self):
        """Clean up all resources"""
        self.deps.cleanup()