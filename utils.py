"""
Utility functions for ExactTranscriber v2.0
"""

import asyncio
from typing import Any, Coroutine, Tuple, TypeVar

T = TypeVar("T")

# Gemini 3 pricing per 1M tokens (update when Google changes pricing)
PRICING = {
    "flash": {"input": 0.50, "output": 3.00},
    "pro": {
        "input_low": 2.00,
        "output_low": 12.00,
        "input_high": 4.00,
        "output_high": 18.00,
    },
    "pro_tier_threshold": 200_000,  # token threshold for higher pricing tier
}


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run async coroutine in Streamlit context.

    Uses the existing event loop (patched by nest_asyncio) when available,
    otherwise creates a new one. This avoids conflicts with nest_asyncio.apply().
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Loop is closed")
        return loop.run_until_complete(coro)
    except RuntimeError:
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
        input_cost = (input_tokens / 1_000_000) * PRICING["flash"]["input"]
        output_cost = (output_tokens / 1_000_000) * PRICING["flash"]["output"]
    else:  # Pro model
        # Apply tiered pricing for Gemini 3 Pro
        threshold = PRICING["pro_tier_threshold"]
        if max(input_tokens, output_tokens) > threshold:
            input_rate = PRICING["pro"]["input_high"]
            output_rate = PRICING["pro"]["output_high"]
        else:
            input_rate = PRICING["pro"]["input_low"]
            output_rate = PRICING["pro"]["output_low"]
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
    import os

    # Strip directory components to prevent path traversal
    filename = os.path.basename(filename)
    # Remove invalid characters
    clean = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Limit length
    if len(clean) > 200:
        clean = clean[:200]
    return clean
