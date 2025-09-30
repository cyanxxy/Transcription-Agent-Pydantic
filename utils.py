"""
Utility functions for ExactTranscriber v2.0
"""

from typing import Tuple


def estimate_audio_tokens(duration_seconds: float) -> int:
    """
    Estimate token count for audio input to Gemini.
    Based on Gemini's audio processing: ~32 tokens per second
    """
    return int(duration_seconds * 32)


def estimate_transcription_cost(
    duration_seconds: float, model_name: str = "gemini-2.5-flash"
) -> Tuple[float, str]:
    """
    Estimate cost for transcribing audio.

    Gemini 2.5 Flash pricing (as of 2024):
    - Input: $0.075 per 1M tokens (first 1B tokens/month)
    - Output: $0.30 per 1M tokens

    Gemini 2.5 Pro pricing:
    - Input: $1.25 per 1M tokens
    - Output: $5.00 per 1M tokens
    """

    # Estimate tokens
    input_tokens = estimate_audio_tokens(duration_seconds)
    # Estimate output tokens (roughly 150 words per minute, ~200 tokens)
    output_tokens = int((duration_seconds / 60) * 200)

    # Calculate costs based on model
    if "flash" in model_name.lower():
        input_cost = (input_tokens / 1_000_000) * 0.075
        output_cost = (output_tokens / 1_000_000) * 0.30
    else:  # Pro model
        input_cost = (input_tokens / 1_000_000) * 1.25
        output_cost = (output_tokens / 1_000_000) * 5.00

    total_cost = input_cost + output_cost

    # Format cost string
    if total_cost < 0.01:
        cost_str = "<$0.01"
    else:
        cost_str = f"${total_cost:.2f}"

    return total_cost, cost_str


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def calculate_optimal_chunk_size(
    file_size_mb: float, duration_seconds: float, max_chunk_mb: float = 20
) -> int:
    """
    Calculate optimal chunk size based on file characteristics.
    Returns chunk duration in milliseconds.
    """
    # Base chunk size: 2 minutes
    base_chunk_ms = 120000

    # Adjust based on file size
    if file_size_mb > 100:
        # Very large file, use smaller chunks
        return 60000  # 1 minute
    elif file_size_mb > 50:
        return 90000  # 1.5 minutes
    else:
        return base_chunk_ms


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage"""
    import re

    # Remove invalid characters
    clean = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Limit length
    if len(clean) > 200:
        clean = clean[:200]
    return clean
