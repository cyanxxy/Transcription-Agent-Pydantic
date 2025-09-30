"""
Pydantic models for ExactTranscriber v2.0
Type-safe data models for the transcription workflow
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """Workflow processing status"""

    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class AudioFormat(str, Enum):
    """Supported audio formats"""

    MP3 = "mp3"
    WAV = "wav"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"


class AudioMetadata(BaseModel):
    """Audio file metadata"""

    filename: str
    duration: float = Field(..., description="Duration in seconds")
    size_mb: float = Field(..., description="File size in megabytes")
    format: AudioFormat
    sample_rate: Optional[int] = Field(None, description="Sample rate in Hz")
    channels: Optional[int] = Field(None, description="Number of audio channels")
    needs_chunking: bool = Field(False, description="Whether file needs to be chunked")
    chunk_count: Optional[int] = Field(None, description="Number of chunks if chunked")

    @field_validator("size_mb")
    @classmethod
    def validate_size(cls, v):
        if v > 200:
            raise ValueError("File size exceeds 200MB limit")
        return v


class TranscriptSegment(BaseModel):
    """Individual transcript segment with speaker and timestamp"""

    timestamp: str = Field(..., pattern=r"^\[\d{2}:\d{2}:\d{2}\]$")
    speaker: str = Field(..., description="Speaker identifier (e.g., 'Speaker 1')")
    text: str = Field(..., min_length=1)
    confidence: Optional[float] = Field(None, ge=0, le=1)

    @field_validator("speaker")
    @classmethod
    def validate_speaker(cls, v):
        if not v.strip():
            raise ValueError("Speaker cannot be empty")
        return v.strip()

    @field_validator("text")
    @classmethod
    def clean_text(cls, v):
        return v.strip()


class TranscriptQuality(BaseModel):
    """Quality metrics for transcript"""

    overall_score: float = Field(..., ge=0, le=100)
    readability: float = Field(..., ge=0, le=100)
    punctuation_density: float = Field(..., ge=0, le=1)
    sentence_variety: float = Field(..., ge=0, le=100)
    vocabulary_richness: float = Field(..., ge=0, le=100)
    timestamp_coverage: float = Field(100.0, ge=0, le=100)
    speaker_consistency: float = Field(100.0, ge=0, le=100)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @property
    def quality_assessment(self) -> str:
        """Get human-readable quality assessment"""
        if self.overall_score >= 80:
            return "Excellent"
        elif self.overall_score >= 60:
            return "Good"
        elif self.overall_score >= 40:
            return "Fair"
        else:
            return "Poor"


class TranscriptResult(BaseModel):
    """Complete transcription result"""

    segments: List[TranscriptSegment]
    metadata: AudioMetadata
    quality: TranscriptQuality
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str
    created_at: datetime = Field(default_factory=datetime.now)
    edited: bool = Field(False)
    export_formats_available: List[str] = Field(
        default_factory=lambda: ["txt", "srt", "json"]
    )

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    @property
    def full_text(self) -> str:
        """Get full transcript text without timestamps"""
        return " ".join(seg.text for seg in self.segments)

    @property
    def formatted_text(self) -> str:
        """Get formatted transcript with timestamps and speakers"""
        return "\n".join(
            f"{seg.timestamp} {seg.speaker}: {seg.text}" for seg in self.segments
        )

    def get_speaker_segments(self, speaker: str) -> List[TranscriptSegment]:
        """Get all segments for a specific speaker"""
        return [seg for seg in self.segments if seg.speaker == speaker]

    @property
    def unique_speakers(self) -> List[str]:
        """Get list of unique speakers"""
        return list(set(seg.speaker for seg in self.segments))


class EditOperation(BaseModel):
    """Edit operation for transcript modification"""

    operation: Literal["replace", "format", "remove", "insert"]
    target: Optional[str] = None
    replacement: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class ExportConfig(BaseModel):
    """Configuration for export operations"""

    format: Literal["txt", "srt", "json", "vtt", "xml"]
    include_timestamps: bool = True
    include_speakers: bool = True
    include_confidence: bool = False
    max_line_length: Optional[int] = None
    custom_template: Optional[str] = None


class TranscriptContext(BaseModel):
    """User-provided context for improving transcription accuracy"""

    speaker_names: List[str] = Field(
        default_factory=list, description="Names of speakers in the audio"
    )
    topic: Optional[str] = Field(
        None, description="Main topic or domain of the conversation"
    )
    technical_terms: List[str] = Field(
        default_factory=list, description="Technical terms or jargon to watch for"
    )
    custom_instructions: Optional[str] = Field(
        None, description="Custom instructions for transcription"
    )
    language_hints: Optional[str] = Field(
        None, description="Language or accent information"
    )
    expected_format: Optional[str] = Field(
        None, description="Expected format (meeting, interview, lecture, etc.)"
    )
    keywords: List[str] = Field(
        default_factory=list, description="Important keywords to watch for"
    )


class ErrorDetail(BaseModel):
    """Structured error information"""

    code: str
    message: str
    category: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    recoverable: bool = True
    suggested_action: Optional[str] = None


class AppState(BaseModel):
    """Application state model"""

    status: ProcessingStatus = ProcessingStatus.IDLE
    current_file: Optional[str] = None
    transcript_result: Optional[TranscriptResult] = None
    edit_history: List[EditOperation] = Field(default_factory=list)
    error: Optional[ErrorDetail] = None
    api_key_configured: bool = False
    model_name: str = "gemini-2.5-flash"
    processing_progress: float = Field(0.0, ge=0, le=1)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def reset(self):
        """Reset state to initial values"""
        self.status = ProcessingStatus.IDLE
        self.current_file = None
        self.transcript_result = None
        self.edit_history = []
        self.error = None
        self.processing_progress = 0.0

    def add_edit(self, operation: EditOperation):
        """Add edit operation to history"""
        self.edit_history.append(operation)
        if self.transcript_result:
            self.transcript_result.edited = True
