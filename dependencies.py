"""
Dependency injection for Pydantic AI agents
Centralized configuration and dependencies
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import os
import tempfile
from pathlib import Path
import logging
import streamlit as st

logger = logging.getLogger(__name__)

SUPPORTED_GEMINI_MODELS = {
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
}
GEMINI_MODEL_ALIASES = {
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
}
SUPPORTED_CANDIDATE_STRATEGIES = {
    "single_gemini",
    "dual_gemini",
    "gemini_plus_parakeet",
}

FLASH_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
PRO_THINKING_LEVELS = {"low", "high"}


def normalize_gemini_model_name(model_name: str) -> str:
    """Accept optional provider prefixes and return the bare Gemini model name."""
    if model_name.startswith("google-gla:"):
        model_name = model_name.split(":", 1)[1]
    return GEMINI_MODEL_ALIASES.get(model_name, model_name)


@dataclass
class TranscriptionDeps:
    """Dependencies for transcription agent"""

    api_key: str
    model_name: str = "gemini-3-flash-preview"  # Gemini 3 Flash (default)
    judge_model_name: str = "gemini-3.1-pro-preview"
    candidate_strategy: str = "dual_gemini"
    max_file_size_mb: int = 200
    chunk_duration_ms: int = 120000  # 2 minutes
    chunk_overlap_ms: int = 5000  # 5 seconds overlap
    temp_dir: str = field(
        default_factory=lambda: tempfile.mkdtemp(prefix="transcriber_")
    )

    # Gemini 3 specific settings
    thinking_level: str = (
        "high"  # Options: "minimal", "low", "medium" (Flash only), "high"
    )
    max_output_tokens: int = 65536  # 65K tokens output limit

    # Chunking settings
    preserve_context: bool = True  # Pass previous chunk summary to next chunk

    # Processing options
    auto_format: bool = True
    remove_fillers: bool = False
    fix_capitalization: bool = True

    # Judge pipeline settings
    parakeet_model: str = "nvidia/parakeet-ctc-0.6b"
    use_judge_pipeline: bool = True

    def __post_init__(self):
        """Initialize dependencies"""
        # Normalize model name (accept optional provider prefix)
        self.model_name = normalize_gemini_model_name(self.model_name)
        self.judge_model_name = normalize_gemini_model_name(self.judge_model_name)

        if self.model_name not in SUPPORTED_GEMINI_MODELS:
            raise ValueError(
                f"Unsupported model: {self.model_name}. "
                f"Supported models: {sorted(SUPPORTED_GEMINI_MODELS)}"
            )
        if self.judge_model_name not in SUPPORTED_GEMINI_MODELS:
            raise ValueError(
                f"Unsupported judge_model_name: {self.judge_model_name}. "
                f"Supported models: {sorted(SUPPORTED_GEMINI_MODELS)}"
            )
        if self.candidate_strategy not in SUPPORTED_CANDIDATE_STRATEGIES:
            raise ValueError(
                f"Unsupported candidate_strategy: {self.candidate_strategy}. "
                f"Supported strategies: {sorted(SUPPORTED_CANDIDATE_STRATEGIES)}"
            )

        if self.thinking_level not in FLASH_THINKING_LEVELS:
            raise ValueError(
                f"Invalid thinking_level: {self.thinking_level}. "
                f"Allowed: {sorted(FLASH_THINKING_LEVELS)}"
            )

        if (
            "-pro" in self.model_name
            and self.thinking_level not in PRO_THINKING_LEVELS
        ):
            raise ValueError(
                f"thinking_level '{self.thinking_level}' is not supported by {self.model_name}. "
                f"Allowed: {sorted(PRO_THINKING_LEVELS)}"
            )

        if self.chunk_duration_ms <= 0:
            raise ValueError("chunk_duration_ms must be > 0")
        if self.chunk_overlap_ms < 0:
            raise ValueError("chunk_overlap_ms must be >= 0")
        if self.chunk_overlap_ms >= self.chunk_duration_ms:
            raise ValueError("chunk_overlap_ms must be less than chunk_duration_ms")

        # Create temp directory if it doesn't exist
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

    def resolve_candidate_specs(self) -> List[Dict[str, str]]:
        """Return the configured transcript candidate plan for the judge pipeline."""
        specs = [
            {
                "candidate_id": self.model_name.replace("-", "_"),
                "label": f"Gemini {self.model_name}",
                "kind": "gemini",
                "model_name": self.model_name,
            }
        ]

        if self.candidate_strategy == "dual_gemini":
            secondary_model = (
                "gemini-3.1-pro-preview"
                if self.model_name != "gemini-3.1-pro-preview"
                else "gemini-3-flash-preview"
            )
            specs.append(
                {
                    "candidate_id": secondary_model.replace("-", "_"),
                    "label": f"Gemini {secondary_model}",
                    "kind": "gemini",
                    "model_name": secondary_model,
                }
            )
        elif self.candidate_strategy == "gemini_plus_parakeet":
            specs.append(
                {
                    "candidate_id": "parakeet_audio",
                    "label": "Parakeet Audio",
                    "kind": "parakeet",
                    "model_name": self.parakeet_model,
                }
            )

        return specs

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
    filler_words: List[str] = field(
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
        """Create dependencies from configuration

        Supported kwargs:
        - model_name: Gemini model to use
        - judge_model_name: Gemini model used by the judge agent
        - candidate_strategy: single_gemini, dual_gemini, gemini_plus_parakeet
        - auto_format: Enable auto-formatting
        - remove_fillers: Remove filler words
        - use_judge_pipeline: Enable multi-agent judge pipeline
        - parakeet_model: NeMo model used for Parakeet candidates/alignment

        Note: Post-judge timestamp alignment is only applied when the pipeline
        detects that the judged transcript needs it.
        """
        # Extract transcription-specific kwargs
        transcription_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "model_name",
                "judge_model_name",
                "candidate_strategy",
                "auto_format",
                "remove_fillers",
                "use_judge_pipeline",
                "parakeet_model",
                "thinking_level",
                "max_file_size_mb",
                "chunk_duration_ms",
                "chunk_overlap_ms",
                "preserve_context",
            ]
        }
        if "use_orchestrator" in kwargs and "use_judge_pipeline" not in kwargs:
            transcription_kwargs["use_judge_pipeline"] = kwargs["use_orchestrator"]
        transcription = TranscriptionDeps(api_key=api_key, **transcription_kwargs)
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

        app_state = getattr(st.session_state, "app_state", None)

        def _session_value(name: str, default):
            if hasattr(st.session_state, name):
                return getattr(st.session_state, name)
            if app_state is not None and hasattr(app_state, name):
                return getattr(app_state, name)
            return default

        # Get other settings from session state
        model_name = _session_value("model_name", "gemini-3-flash-preview")
        judge_model_name = _session_value(
            "judge_model_name", "gemini-3.1-pro-preview"
        )
        candidate_strategy = _session_value("candidate_strategy", "dual_gemini")
        auto_format = _session_value("auto_format", True)
        remove_fillers = _session_value("remove_fillers", False)
        use_judge_pipeline = _session_value("use_judge_pipeline", True)

        return cls.from_config(
            api_key=api_key,
            model_name=model_name,
            judge_model_name=judge_model_name,
            candidate_strategy=candidate_strategy,
            auto_format=auto_format,
            remove_fillers=remove_fillers,
            use_judge_pipeline=use_judge_pipeline,
        )

    def cleanup(self):
        """Clean up all resources"""
        self.transcription.cleanup()
