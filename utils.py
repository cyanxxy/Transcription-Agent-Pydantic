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


def estimate_transcript_tokens(duration_seconds: float) -> int:
    """
    Estimate transcript text tokens from audio duration.

    Roughly 150 words per minute, about 200 tokens per minute.
    """
    return int((duration_seconds / 60) * 200)


def _estimate_model_token_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: str = "gemini-3-flash-preview",
) -> float:
    """Estimate model cost from explicit token counts."""
    if "flash" in model_name.lower():
        input_cost = (input_tokens / 1_000_000) * PRICING["flash"]["input"]
        output_cost = (output_tokens / 1_000_000) * PRICING["flash"]["output"]
    else:
        threshold = PRICING["pro_tier_threshold"]
        if max(input_tokens, output_tokens) > threshold:
            input_rate = PRICING["pro"]["input_high"]
            output_rate = PRICING["pro"]["output_high"]
        else:
            input_rate = PRICING["pro"]["input_low"]
            output_rate = PRICING["pro"]["output_low"]
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate

    return input_cost + output_cost


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
    output_tokens = estimate_transcript_tokens(duration_seconds)
    total_cost = _estimate_model_token_cost(input_tokens, output_tokens, model_name)

    # Format cost string
    if total_cost < 0.01:
        cost_str = "<$0.01"
    else:
        cost_str = f"${total_cost:.2f}"

    return total_cost, cost_str


def estimate_judge_pipeline_cost(
    duration_seconds: float,
    model_name: str = "gemini-3-flash-preview",
    candidate_strategy: str = "dual_gemini",
    use_judge_pipeline: bool = True,
    judge_model_name: str = "gemini-3.1-pro-preview",
) -> Tuple[float, str]:
    """
    Estimate total Gemini API cost for the visible pipeline choice.

    Notes:
    - Parakeet is treated as a local model with no Gemini API cost.
    - Judge cost is approximated from transcript text tokens, not audio tokens.
    """
    if not use_judge_pipeline:
        return estimate_transcription_cost(duration_seconds, model_name)

    total_cost, _ = estimate_transcription_cost(duration_seconds, model_name)
    transcript_tokens = estimate_transcript_tokens(duration_seconds)

    if candidate_strategy == "dual_gemini":
        secondary_model = (
            "gemini-3.1-pro-preview"
            if model_name != "gemini-3.1-pro-preview"
            else "gemini-3-flash-preview"
        )
        secondary_cost, _ = estimate_transcription_cost(duration_seconds, secondary_model)
        total_cost += secondary_cost
        judge_candidate_count = 2
    elif candidate_strategy == "gemini_plus_parakeet":
        judge_candidate_count = 2
    else:
        judge_candidate_count = 1

    judge_input_tokens = (transcript_tokens * judge_candidate_count) + 800
    judge_output_tokens = transcript_tokens
    total_cost += _estimate_model_token_cost(
        judge_input_tokens,
        judge_output_tokens,
        judge_model_name,
    )

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
