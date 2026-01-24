"""
Timestamp correction using NVIDIA Parakeet + NeMo Forced Aligner

This module provides accurate timestamp alignment by using Parakeet CTC model
with NeMo's Forced Aligner to align Gemini's quality transcription text
to precise audio timestamps.
"""

from typing import List, Dict, Any, Optional
import asyncio
import logging
import re

from models import TranscriptSegment
from dependencies import TranscriptionDeps

logger = logging.getLogger(__name__)

# Lazy-loaded Parakeet model (cached at module level)
_PARAKEET_MODEL = None
_MODEL_LOCK = asyncio.Lock()


async def _get_parakeet_model(model_name: str = "nvidia/parakeet-ctc-0.6b"):
    """Load Parakeet CTC model (lazy, cached, thread-safe)"""
    global _PARAKEET_MODEL

    async with _MODEL_LOCK:
        if _PARAKEET_MODEL is None:
            logger.info(f"Loading Parakeet model: {model_name}")

            def _load_model():
                import nemo.collections.asr as nemo_asr

                return nemo_asr.models.ASRModel.from_pretrained(model_name)

            _PARAKEET_MODEL = await asyncio.to_thread(_load_model)
            logger.info("Parakeet model loaded successfully")

    return _PARAKEET_MODEL


async def fix_timestamps_with_parakeet(
    deps: TranscriptionDeps,
    audio_path: str,
    segments: List[TranscriptSegment],
) -> List[TranscriptSegment]:
    """
    Align Gemini's text to accurate timestamps using NeMo Forced Aligner.

    Process:
    1. Extract text from Gemini segments
    2. Run NFA with Parakeet CTC to get word-level timestamps
    3. Map timestamps back to segment boundaries

    Args:
        deps: Transcription dependencies (includes parakeet_model setting)
        audio_path: Path to the audio file
        segments: List of TranscriptSegment from Gemini transcription

    Returns:
        List of TranscriptSegment with corrected timestamps
    """
    if not segments:
        logger.warning("No segments provided for timestamp correction")
        return segments

    logger.info(f"Fixing timestamps for {len(segments)} segments using Parakeet")

    try:
        # Get model name from deps or use default
        model_name = getattr(deps, "parakeet_model", "nvidia/parakeet-ctc-0.6b")

        # Load model (cached)
        model = await _get_parakeet_model(model_name)

        # Run alignment in thread pool (NeMo is synchronous)
        corrected = await asyncio.to_thread(
            _run_alignment,
            model,
            audio_path,
            segments,
        )

        logger.info(f"Timestamp correction complete for {len(corrected)} segments")
        return corrected

    except ImportError as e:
        logger.warning(f"NeMo not available, skipping timestamp correction: {e}")
        return segments
    except Exception as e:
        logger.error(f"Timestamp correction failed: {e}")
        raise


def _run_alignment(
    model,
    audio_path: str,
    segments: List[TranscriptSegment],
) -> List[TranscriptSegment]:
    """Synchronous NFA alignment using Parakeet model"""
    try:
        from nemo.collections.asr.parts.utils.nfa import NFA

        # Combine all segment text for alignment
        full_text = " ".join([_clean_text_for_alignment(seg.text) for seg in segments])

        # Create NFA instance
        nfa = NFA(model=model)

        # Run forced alignment
        # NFA aligns reference text to audio frames and returns word-level timestamps
        alignment_result = nfa.align(
            audio_file=audio_path,
            text=full_text,
        )

        # Extract word timestamps from alignment result
        word_timestamps = _extract_word_timestamps(alignment_result)

        # Map word timestamps back to segments
        return _map_to_segments(segments, word_timestamps)

    except ImportError:
        # Fallback: try alternative alignment approach
        return _run_alignment_fallback(model, audio_path, segments)


def _run_alignment_fallback(
    model,
    audio_path: str,
    segments: List[TranscriptSegment],
) -> List[TranscriptSegment]:
    """Fallback alignment using direct model transcription for timing reference"""
    try:
        # Use Parakeet to transcribe and get word-level timing
        # This gives us accurate timestamps we can cross-reference
        transcription_result = model.transcribe(
            [audio_path],
            return_hypotheses=True,
            timestamps=True,
        )

        if not transcription_result or not transcription_result[0]:
            logger.warning("Fallback alignment returned no results")
            return segments

        # Extract word timestamps from Parakeet transcription
        word_timestamps = _extract_timestamps_from_transcription(transcription_result[0])

        # Map to original segments using text matching
        return _map_to_segments_with_matching(segments, word_timestamps)

    except Exception as e:
        logger.warning(f"Fallback alignment failed: {e}")
        return segments


def _clean_text_for_alignment(text: str) -> str:
    """Clean text for forced alignment (remove special markers)"""
    # Remove non-speech markers like [MUSIC], [SILENCE], etc.
    cleaned = re.sub(r"\[[A-Z]+\]", "", text)
    # Remove multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_word_timestamps(alignment_result) -> List[Dict[str, Any]]:
    """Extract word-level timestamps from NFA alignment result"""
    word_timestamps = []

    if hasattr(alignment_result, "word_timestamps"):
        for word_info in alignment_result.word_timestamps:
            word_timestamps.append({
                "word": word_info.get("word", ""),
                "start": word_info.get("start", 0.0),
                "end": word_info.get("end", 0.0),
            })
    elif isinstance(alignment_result, list):
        for item in alignment_result:
            if isinstance(item, dict):
                word_timestamps.append({
                    "word": item.get("word", item.get("text", "")),
                    "start": item.get("start", item.get("start_time", 0.0)),
                    "end": item.get("end", item.get("end_time", 0.0)),
                })

    return word_timestamps


def _extract_timestamps_from_transcription(transcription) -> List[Dict[str, Any]]:
    """Extract word timestamps from Parakeet transcription result"""
    word_timestamps = []

    # Handle different transcription result formats
    if hasattr(transcription, "timestep") and transcription.timestep:
        # NeMo timestep format
        for ts in transcription.timestep:
            word_timestamps.append({
                "word": ts.get("word", ""),
                "start": ts.get("start", 0.0),
                "end": ts.get("end", 0.0),
            })
    elif hasattr(transcription, "words"):
        # Alternative format with words attribute
        for word_info in transcription.words:
            word_timestamps.append({
                "word": getattr(word_info, "word", ""),
                "start": getattr(word_info, "start", 0.0),
                "end": getattr(word_info, "end", 0.0),
            })

    return word_timestamps


def _map_to_segments(
    segments: List[TranscriptSegment],
    word_timestamps: List[Dict[str, Any]],
) -> List[TranscriptSegment]:
    """Map word-level timestamps to segment boundaries"""
    if not word_timestamps:
        logger.warning("No word timestamps available, returning original segments")
        return segments

    corrected = []
    word_idx = 0

    for segment in segments:
        # Count words in this segment (excluding special markers)
        segment_text = _clean_text_for_alignment(segment.text)
        words_in_segment = len(segment_text.split()) if segment_text else 0

        if word_idx < len(word_timestamps) and words_in_segment > 0:
            # Get start time from first word of this segment
            start_time = word_timestamps[word_idx].get("start", 0.0)

            # Format as [HH:MM:SS]
            accurate_ts = _format_timestamp(start_time)
        else:
            # Fallback to original timestamp
            accurate_ts = segment.timestamp

        corrected.append(TranscriptSegment(
            timestamp=accurate_ts,
            speaker=segment.speaker,
            text=segment.text,
            confidence=segment.confidence,
        ))

        # Advance word index
        word_idx += words_in_segment

    return corrected


def _map_to_segments_with_matching(
    segments: List[TranscriptSegment],
    word_timestamps: List[Dict[str, Any]],
) -> List[TranscriptSegment]:
    """Map segments using fuzzy text matching against Parakeet timestamps"""
    if not word_timestamps:
        return segments

    corrected = []
    parakeet_idx = 0

    for segment in segments:
        # Get first few words of segment for matching
        segment_words = _clean_text_for_alignment(segment.text).lower().split()[:3]

        if not segment_words:
            corrected.append(segment)
            continue

        # Find matching position in Parakeet timestamps
        best_match_idx = _find_best_match(
            segment_words,
            word_timestamps,
            parakeet_idx,
        )

        if best_match_idx is not None and best_match_idx < len(word_timestamps):
            start_time = word_timestamps[best_match_idx].get("start", 0.0)
            accurate_ts = _format_timestamp(start_time)
            parakeet_idx = best_match_idx + len(segment_words)
        else:
            accurate_ts = segment.timestamp

        corrected.append(TranscriptSegment(
            timestamp=accurate_ts,
            speaker=segment.speaker,
            text=segment.text,
            confidence=segment.confidence,
        ))

    return corrected


def _find_best_match(
    target_words: List[str],
    word_timestamps: List[Dict[str, Any]],
    start_idx: int,
    search_window: int = 50,
) -> Optional[int]:
    """Find best matching position for target words in timestamp list"""
    if not target_words or not word_timestamps:
        return None

    first_word = target_words[0].lower()
    end_idx = min(start_idx + search_window, len(word_timestamps))

    for i in range(start_idx, end_idx):
        parakeet_word = word_timestamps[i].get("word", "").lower()

        # Check for match (allowing for slight variations)
        if parakeet_word == first_word or _is_similar(parakeet_word, first_word):
            return i

    return None


def _is_similar(word1: str, word2: str, threshold: float = 0.8) -> bool:
    """Check if two words are similar (simple Levenshtein-like check)"""
    if not word1 or not word2:
        return False

    # Exact match
    if word1 == word2:
        return True

    # One is prefix of other
    if word1.startswith(word2) or word2.startswith(word1):
        return True

    # Length difference too large
    if abs(len(word1) - len(word2)) > 2:
        return False

    # Simple character overlap check
    common = sum(1 for c in word1 if c in word2)
    similarity = (2 * common) / (len(word1) + len(word2))

    return similarity >= threshold


def _format_timestamp(seconds: float) -> str:
    """Format seconds as [HH:MM:SS] timestamp"""
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def _parse_timestamp(timestamp: str) -> float:
    """Parse [HH:MM:SS] timestamp to seconds"""
    match = re.match(r"^\[(\d{2}):(\d{2}):(\d{2})\]$", timestamp)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return 0.0
