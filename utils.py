"""
Utility functions for ExactTranscriber v2.0
"""

import asyncio
from typing import Any, Coroutine, Tuple, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run async coroutine in Streamlit context.

    Creates a new event loop, runs the coroutine, and properly cleans up.
    Use this wrapper for all async operations in the Streamlit UI.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def estimate_audio_tokens(duration_seconds: float) -> int:
    """
    Estimate token count for audio input to Gemini.
    Based on Gemini's audio processing: ~32 tokens per second
    """
    return int(duration_seconds * 32)


def estimate_transcription_cost(
    duration_seconds: float, model_name: str = "gemini-3-flash-preview"
) -> Tuple[float, str]:
    """
    Estimate cost for transcribing audio.

    Gemini 3 Flash Preview pricing (per 1M tokens):
    - Input: $0.50
    - Output: $3.00

    Gemini 3 Pro Preview pricing (per 1M tokens):
    - <200k tokens: Input $2.00, Output $12.00
    - >200k tokens: Input $4.00, Output $18.00
    """

    # Estimate tokens
    input_tokens = estimate_audio_tokens(duration_seconds)
    # Estimate output tokens (roughly 150 words per minute, ~200 tokens)
    output_tokens = int((duration_seconds / 60) * 200)

    # Calculate costs based on model
    if "flash" in model_name.lower():
        input_cost = (input_tokens / 1_000_000) * 0.50
        output_cost = (output_tokens / 1_000_000) * 3.00
    else:  # Pro model
        # Apply tiered pricing for Gemini 3 Pro
        if max(input_tokens, output_tokens) > 200_000:
            input_rate = 4.00
            output_rate = 18.00
        else:
            input_rate = 2.00
            output_rate = 12.00
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate

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


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage"""
    import re

    # Remove invalid characters
    clean = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Limit length
    if len(clean) > 200:
        clean = clean[:200]
    return clean
