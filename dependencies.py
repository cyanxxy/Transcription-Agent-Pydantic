"""
Dependency injection for Pydantic AI agents
Centralized configuration and dependencies
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import os
import tempfile
from pathlib import Path
import logging
import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionDeps:
    """Dependencies for transcription agent"""

    api_key: str
    model_name: str = "gemini-2.5-flash"
    max_file_size_mb: int = 200
    chunk_duration_ms: int = 120000  # 2 minutes
    chunk_overlap_ms: int = 5000  # 5 seconds overlap
    temp_dir: str = field(
        default_factory=lambda: tempfile.mkdtemp(prefix="transcriber_")
    )
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: int = 300

    # Gemini 2.5 specific settings
    thinking_budget: int = -1  # -1 for dynamic, 0 to disable, or specific token count
    enable_thought_summaries: bool = True
    max_input_tokens: int = 1048576  # 1M tokens for Gemini 2.5
    max_output_tokens: int = 65536  # 65K tokens output limit
    enable_structured_output: bool = True
    temperature: float = 0.2  # Lower for more consistent transcription

    # Smart chunking settings
    adaptive_chunk_size: bool = True  # Adjust chunk size based on audio complexity
    preserve_context: bool = True  # Pass previous chunk summary to next chunk
    smart_overlap: bool = True  # Detect sentence boundaries for overlap

    # Quality thresholds
    min_quality_score: float = 40.0
    target_quality_score: float = 70.0

    # Processing options
    auto_format: bool = True
    remove_fillers: bool = False
    fix_capitalization: bool = True

    # Cache settings (removed - no caching)
    use_cache: bool = False
    cache_dir: Optional[str] = None

    def __post_init__(self):
        """Initialize dependencies"""
        # Create temp directory if it doesn't exist
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        # Set API key in environment for Pydantic AI and Google GenAI SDK
        os.environ["GOOGLE_API_KEY"] = self.api_key
        os.environ["GEMINI_API_KEY"] = self.api_key  # For backward compatibility

    @property
    def chunk_size_bytes(self) -> int:
        """Calculate chunk size in bytes based on duration"""
        # Approximate: 1 minute of audio ≈ 1 MB for MP3
        return int((self.chunk_duration_ms / 60000) * 1024 * 1024)

    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil

            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")


@dataclass
class EditingDeps:
    """Dependencies for editing agent"""

    enable_auto_correct: bool = True
    preserve_timestamps: bool = True
    max_undo_history: int = 50
    remove_fillers: bool = False

    # Formatting rules
    sentence_case: bool = True
    remove_extra_spaces: bool = True
    fix_punctuation_spacing: bool = True

    # Filler words to remove
    filler_words: list = field(
        default_factory=lambda: [
            "um",
            "uh",
            "like",
            "you know",
            "I mean",
            "sort of",
            "kind of",
            "basically",
            "actually",
        ]
    )

    # Common replacements
    replacements: Dict[str, str] = field(
        default_factory=lambda: {
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "kinda": "kind of",
            "sorta": "sort of",
        }
    )


@dataclass
class QualityDeps:
    """Dependencies for quality assurance"""

    min_sentence_length: int = 3
    max_sentence_length: int = 50
    target_readability_score: float = 70.0

    # Quality check thresholds
    min_vocabulary_richness: float = 30.0
    max_punctuation_density: float = 0.15
    min_timestamp_coverage: float = 80.0

    # Issue detection
    detect_grammar_issues: bool = True
    detect_consistency_issues: bool = True
    detect_formatting_issues: bool = True

    # Scoring weights
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "readability": 0.3,
            "vocabulary": 0.2,
            "sentence_variety": 0.2,
            "punctuation": 0.15,
            "consistency": 0.15,
        }
    )


@dataclass
class ExportDeps:
    """Dependencies for export operations"""

    output_dir: str = field(default_factory=lambda: os.path.expanduser("~/Downloads"))

    # Format-specific settings
    srt_max_line_length: int = 42
    srt_max_duration: float = 3.0  # seconds

    json_pretty_print: bool = True
    json_include_metadata: bool = True

    txt_include_headers: bool = True
    txt_separator: str = "\n" + "=" * 50 + "\n"

    # Templates
    templates_dir: Optional[str] = None
    use_custom_templates: bool = False


@dataclass
class ContextDeps:
    """Dependencies for context agent"""

    max_speakers: int = 10
    max_terms: int = 50
    max_keywords: int = 30

    # Domain detection
    auto_detect_domain: bool = True
    suggest_terms: bool = True

    # Context enhancement
    enhance_with_common_terms: bool = True
    validate_speaker_names: bool = True


@dataclass
class AppDeps:
    """Main application dependencies container"""

    transcription: TranscriptionDeps
    editing: EditingDeps = field(default_factory=EditingDeps)
    quality: QualityDeps = field(default_factory=QualityDeps)
    export: ExportDeps = field(default_factory=ExportDeps)

    # Application settings
    debug_mode: bool = False
    log_level: str = "INFO"
    enable_metrics: bool = True

    @classmethod
    def from_config(cls, api_key: str, **kwargs) -> "AppDeps":
        """Create dependencies from configuration"""
        transcription = TranscriptionDeps(api_key=api_key, **kwargs)
        editing = EditingDeps(remove_fillers=transcription.remove_fillers)
        return cls(transcription=transcription, editing=editing)

    @classmethod
    def from_streamlit(cls) -> Optional["AppDeps"]:
        """Create dependencies from Streamlit session state"""
        # Try to get API key from session state or secrets
        api_key = None

        # Check session state first
        if hasattr(st.session_state, "api_key"):
            api_key = st.session_state.api_key

        # Then check secrets
        if not api_key and hasattr(st, "secrets"):
            api_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get(
                "GEMINI_API_KEY"
            )

        # Finally check environment
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
                "GEMINI_API_KEY"
            )

        if not api_key:
            return None

        # Get other settings from session state
        model_name = getattr(st.session_state, "model_name", "gemini-2.5-flash")
        auto_format = getattr(st.session_state, "auto_format", True)
        remove_fillers = getattr(st.session_state, "remove_fillers", False)

        return cls.from_config(
            api_key=api_key,
            model_name=model_name,
            auto_format=auto_format,
            remove_fillers=remove_fillers,
        )

    def cleanup(self):
        """Clean up all resources"""
        self.transcription.cleanup()
