"""
Quality Validator using utility functions
Analyzes and validates transcript quality
"""

from typing import List, Dict, Any
import re
import statistics
import logging

from models import TranscriptSegment
from dependencies import QualityDeps

logger = logging.getLogger(__name__)


def calculate_quality_metrics(
    deps: QualityDeps, segments: List[TranscriptSegment]
) -> Dict[str, float]:
    """Calculate comprehensive quality metrics"""

    if not segments:
        return {
            "readability": 0,
            "punctuation_density": 0,
            "sentence_variety": 0,
            "vocabulary_richness": 0,
            "timestamp_coverage": 0,
            "speaker_consistency": 0,
            "warnings": ["No segments to analyze"],
        }

    # Combine all text
    full_text = " ".join(seg.text for seg in segments)
    words = full_text.lower().split()

    # Readability
    sentences = [s.strip() for s in re.split(r"[.!?]+", full_text) if s.strip()]
    avg_sentence_length = len(words) / max(len(sentences), 1)

    # Normalize readability (optimal is around 15 words per sentence)
    readability = min(100, max(0, 100 - abs(avg_sentence_length - 15) * 3))

    # Punctuation density
    punct_count = sum(1 for c in full_text if c in ".,;:!?")
    punctuation_density = punct_count / max(len(full_text), 1)

    # Sentence variety (standard deviation of sentence lengths)
    if sentences:
        sentence_lengths = [len(s.split()) for s in sentences]
        sentence_variety = (
            min(100, statistics.stdev(sentence_lengths) * 10)
            if len(sentences) > 1
            else 50
        )
    else:
        sentence_variety = 0

    # Vocabulary richness (unique words / total words)
    unique_words = len(set(words))
    vocabulary_richness = min(100, (unique_words / max(len(words), 1)) * 200)

    # Timestamp coverage (check if timestamps are sequential)
    timestamp_coverage = calculate_timestamp_coverage(segments)

    # Speaker consistency
    speaker_consistency = calculate_speaker_consistency(segments)

    # Compile warnings
    warnings = []
    if readability < deps.target_readability_score:
        warnings.append(f"Low readability score: {readability:.1f}")
    if vocabulary_richness < deps.min_vocabulary_richness:
        warnings.append(f"Low vocabulary richness: {vocabulary_richness:.1f}")
    if punctuation_density > deps.max_punctuation_density:
        warnings.append(f"High punctuation density: {punctuation_density:.3f}")
    if timestamp_coverage < deps.min_timestamp_coverage:
        warnings.append(f"Low timestamp coverage: {timestamp_coverage:.1f}%")

    return {
        "readability": readability,
        "punctuation_density": punctuation_density,
        "sentence_variety": sentence_variety,
        "vocabulary_richness": vocabulary_richness,
        "timestamp_coverage": timestamp_coverage,
        "speaker_consistency": speaker_consistency,
        "warnings": warnings,
    }


def detect_quality_issues(
    deps: QualityDeps, segments: List[TranscriptSegment]
) -> List[Dict[str, Any]]:
    """Detect quality issues in transcript"""

    issues = []

    # Check for very short or long sentences
    for i, segment in enumerate(segments):
        words = segment.text.split()
        word_count = len(words)

        if word_count < deps.min_sentence_length:
            issues.append(
                {
                    "type": "sentence_length",
                    "segment": i,
                    "message": f"Very short segment ({word_count} words)",
                    "severity": "low",
                }
            )
        elif word_count > deps.max_sentence_length:
            issues.append(
                {
                    "type": "sentence_length",
                    "segment": i,
                    "message": f"Very long segment ({word_count} words)",
                    "severity": "medium",
                }
            )

    # Check for speaker changes
    speakers = set(seg.speaker for seg in segments)
    if len(speakers) > 10:
        issues.append(
            {
                "type": "speaker_count",
                "message": f"Unusually high number of speakers: {len(speakers)}",
                "severity": "medium",
            }
        )

    # Check for timestamp gaps
    prev_timestamp = None
    for i, segment in enumerate(segments):
        if prev_timestamp:
            gap = parse_timestamp_to_seconds(
                segment.timestamp
            ) - parse_timestamp_to_seconds(prev_timestamp)
            if gap > 30:  # More than 30 seconds gap
                issues.append(
                    {
                        "type": "timestamp_gap",
                        "segment": i,
                        "message": f"Large gap in timestamps ({gap:.1f} seconds)",
                        "severity": "medium",
                    }
                )
        prev_timestamp = segment.timestamp

    return issues


def calculate_overall_score(deps: QualityDeps, metrics: Dict[str, float]) -> float:
    """Calculate weighted overall quality score"""

    weights = deps.weights

    score = 0.0
    total_weight = 0.0

    # Map metrics to weight keys
    metric_mapping = {
        "readability": "readability",
        "vocabulary_richness": "vocabulary",
        "sentence_variety": "sentence_variety",
        "punctuation_density": "punctuation",
        "speaker_consistency": "consistency",
    }

    for metric_key, weight_key in metric_mapping.items():
        if metric_key in metrics and weight_key in weights:
            metric_value = metrics[metric_key]

            # Special handling for punctuation density (lower is better)
            if metric_key == "punctuation_density":
                metric_value = max(0, 100 - metric_value * 100)

            score += metric_value * weights[weight_key]
            total_weight += weights[weight_key]

    # Normalize score
    if total_weight > 0:
        overall_score = score / total_weight
    else:
        overall_score = 50.0  # Default middle score

    return min(100, max(0, overall_score))


def count_syllables(word: str) -> int:
    """Simple syllable counter"""
    word = word.lower()
    vowels = "aeiou"
    syllables = 0
    previous_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllables += 1
        previous_was_vowel = is_vowel

    # Ensure at least one syllable
    return max(1, syllables)


def calculate_timestamp_coverage(segments: List[TranscriptSegment]) -> float:
    """Calculate how well timestamps cover the transcript"""

    if not segments:
        return 0.0

    valid_timestamps = sum(
        1 for seg in segments if re.match(r"^\[\d{2}:\d{2}:\d{2}\]$", seg.timestamp)
    )

    coverage = (valid_timestamps / len(segments)) * 100
    return coverage


def calculate_speaker_consistency(segments: List[TranscriptSegment]) -> float:
    """Calculate speaker labeling consistency

    Rewards stable speaker identification, whether using real names or generic labels.
    The key is consistency - same speaker should have same label throughout.
    """

    if not segments:
        return 0.0

    # Count unique speakers and their frequency
    speaker_counts = {}
    for seg in segments:
        speaker = seg.speaker.strip()
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

    # Calculate consistency based on speaker stability
    # Good: Few unique speakers with many segments each (stable identification)
    # Bad: Many unique speakers with few segments each (inconsistent labeling)

    unique_speakers = len(speaker_counts)

    if unique_speakers == 0:
        return 0.0

    # Consistency score based on how well distributed the segments are
    # Perfect score if 2-4 speakers with balanced distribution
    # Lower score for too many speakers or very unbalanced distribution

    if unique_speakers <= 10:  # Reasonable number of speakers
        # Check distribution balance using standard deviation
        import statistics

        segment_counts = list(speaker_counts.values())

        if len(segment_counts) > 1:
            mean_count = statistics.mean(segment_counts)
            stdev = statistics.stdev(segment_counts)
            # Lower stdev relative to mean = more balanced = better consistency
            balance_score = max(0, 100 - (stdev / mean_count) * 50)
        else:
            balance_score = 100  # Single speaker is perfectly consistent

        # Penalize too many speakers slightly
        speaker_penalty = max(0, (10 - unique_speakers) * 5)

        consistency = min(100, balance_score + speaker_penalty)
    else:
        # Too many unique speakers indicates poor consistency
        consistency = max(0, 100 - (unique_speakers - 10) * 10)

    return consistency


def parse_timestamp_to_seconds(timestamp: str) -> float:
    """Parse [HH:MM:SS] to seconds"""

    match = re.match(r"^\[(\d{2}):(\d{2}):(\d{2})\]$", timestamp)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return 0.0
