"""
Quality Validator Agent using Pydantic AI
Analyzes and validates transcript quality
"""

from pydantic_ai import Agent, RunContext
from typing import List, Dict, Any, Tuple
import re
import statistics
import logging

from models import (
    TranscriptResult, TranscriptSegment, TranscriptQuality,
    ErrorDetail
)
from dependencies import QualityDeps

logger = logging.getLogger(__name__)


# Create quality assessment agent
# Use the model name directly without prefix as per Pydantic AI docs
quality_agent = Agent(
    'gemini-2.5-flash',
    deps_type=QualityDeps,
    output_type=TranscriptQuality,
    system_prompt="""You are a transcript quality analyst.
    Analyze transcripts for quality issues and provide detailed assessments.
    
    Focus on:
    - Readability and clarity
    - Grammar and punctuation accuracy
    - Speaker consistency
    - Timestamp completeness
    - Overall coherence"""
)


@quality_agent.output_validator
async def validate_transcript_quality(
    ctx: RunContext[QualityDeps],
    result: TranscriptResult
) -> TranscriptResult:
    """Validate and score transcript quality"""
    
    if not result.segments:
        raise ValueError("Transcript has no segments")
    
    # Calculate quality metrics
    quality_metrics = calculate_quality_metrics(ctx.deps, result.segments)
    
    # Check for issues
    issues = detect_quality_issues(ctx.deps, result.segments)
    
    # Calculate overall score
    overall_score = calculate_overall_score(ctx.deps, quality_metrics)
    
    # Create quality assessment
    result.quality = TranscriptQuality(
        overall_score=overall_score,
        readability=quality_metrics['readability'],
        punctuation_density=quality_metrics['punctuation_density'],
        sentence_variety=quality_metrics['sentence_variety'],
        vocabulary_richness=quality_metrics['vocabulary_richness'],
        timestamp_coverage=quality_metrics['timestamp_coverage'],
        speaker_consistency=quality_metrics['speaker_consistency'],
        issues=issues,
        warnings=quality_metrics.get('warnings', [])
    )
    
    # Validate minimum quality threshold
    if overall_score < ctx.deps.min_quality_score:
        logger.warning(f"Transcript quality below threshold: {overall_score:.1f}")
        result.quality.warnings.append(
            f"Quality score ({overall_score:.1f}) below minimum threshold ({ctx.deps.min_quality_score})"
        )
    
    return result


@quality_agent.tool
async def analyze_readability(
    ctx: RunContext[QualityDeps],
    segments: List[TranscriptSegment]
) -> float:
    """Analyze transcript readability"""
    
    if not segments:
        return 0.0
    
    # Combine all text
    full_text = ' '.join(seg.text for seg in segments)
    
    # Calculate readability metrics
    sentences = re.split(r'[.!?]+', full_text)
    words = full_text.split()
    
    if not sentences or not words:
        return 0.0
    
    # Average sentence length
    avg_sentence_length = len(words) / len(sentences)
    
    # Calculate syllables (simplified)
    total_syllables = sum(count_syllables(word) for word in words)
    avg_syllables_per_word = total_syllables / len(words) if words else 0
    
    # Flesch Reading Ease formula (simplified)
    reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
    
    # Normalize to 0-100 scale
    readability = max(0, min(100, reading_ease))
    
    return readability


@quality_agent.tool
async def detect_errors(
    ctx: RunContext[QualityDeps],
    segments: List[TranscriptSegment]
) -> List[Dict[str, Any]]:
    """Detect errors and issues in transcript"""
    
    errors = []
    
    for i, segment in enumerate(segments):
        # Check timestamp format
        if not re.match(r'^\[\d{2}:\d{2}:\d{2}\]$', segment.timestamp):
            errors.append({
                "type": "timestamp_format",
                "segment": i,
                "message": f"Invalid timestamp format: {segment.timestamp}",
                "severity": "medium"
            })
        
        # Check speaker format
        if not segment.speaker or not segment.speaker.strip():
            errors.append({
                "type": "missing_speaker",
                "segment": i,
                "message": "Missing speaker identification",
                "severity": "high"
            })
        
        # Check for empty text
        if not segment.text or not segment.text.strip():
            errors.append({
                "type": "empty_text",
                "segment": i,
                "message": "Empty transcript text",
                "severity": "high"
            })
            continue
        
        # Grammar and punctuation checks
        text = segment.text
        
        # Check for missing punctuation at end
        if text and text[-1] not in '.!?,;:':
            errors.append({
                "type": "missing_punctuation",
                "segment": i,
                "message": "Missing punctuation at end of segment",
                "severity": "low",
                "suggestion": text + "."
            })
        
        # Check for double spaces
        if '  ' in text:
            errors.append({
                "type": "formatting",
                "segment": i,
                "message": "Multiple consecutive spaces detected",
                "severity": "low",
                "suggestion": re.sub(r'\s+', ' ', text)
            })
        
        # Check for uncapitalized sentences
        if text and text[0].islower():
            errors.append({
                "type": "capitalization",
                "segment": i,
                "message": "Sentence doesn't start with capital letter",
                "severity": "low",
                "suggestion": text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            })
    
    return errors


def calculate_quality_metrics(
    deps: QualityDeps,
    segments: List[TranscriptSegment]
) -> Dict[str, float]:
    """Calculate comprehensive quality metrics"""
    
    if not segments:
        return {
            'readability': 0,
            'punctuation_density': 0,
            'sentence_variety': 0,
            'vocabulary_richness': 0,
            'timestamp_coverage': 0,
            'speaker_consistency': 0,
            'warnings': ["No segments to analyze"]
        }
    
    # Combine all text
    full_text = ' '.join(seg.text for seg in segments)
    words = full_text.lower().split()
    
    # Readability
    sentences = [s.strip() for s in re.split(r'[.!?]+', full_text) if s.strip()]
    avg_sentence_length = len(words) / max(len(sentences), 1)
    
    # Normalize readability (optimal is around 15 words per sentence)
    readability = min(100, max(0, 100 - abs(avg_sentence_length - 15) * 3))
    
    # Punctuation density
    punct_count = sum(1 for c in full_text if c in '.,;:!?')
    punctuation_density = punct_count / max(len(full_text), 1)
    
    # Sentence variety (standard deviation of sentence lengths)
    if sentences:
        sentence_lengths = [len(s.split()) for s in sentences]
        sentence_variety = min(100, statistics.stdev(sentence_lengths) * 10) if len(sentences) > 1 else 50
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
        'readability': readability,
        'punctuation_density': punctuation_density,
        'sentence_variety': sentence_variety,
        'vocabulary_richness': vocabulary_richness,
        'timestamp_coverage': timestamp_coverage,
        'speaker_consistency': speaker_consistency,
        'warnings': warnings
    }


def detect_quality_issues(
    deps: QualityDeps,
    segments: List[TranscriptSegment]
) -> List[Dict[str, Any]]:
    """Detect quality issues in transcript"""
    
    issues = []
    
    # Check for very short or long sentences
    for i, segment in enumerate(segments):
        words = segment.text.split()
        word_count = len(words)
        
        if word_count < deps.min_sentence_length:
            issues.append({
                "type": "sentence_length",
                "segment": i,
                "message": f"Very short segment ({word_count} words)",
                "severity": "low"
            })
        elif word_count > deps.max_sentence_length:
            issues.append({
                "type": "sentence_length",
                "segment": i,
                "message": f"Very long segment ({word_count} words)",
                "severity": "medium"
            })
    
    # Check for speaker changes
    speakers = set(seg.speaker for seg in segments)
    if len(speakers) > 10:
        issues.append({
            "type": "speaker_count",
            "message": f"Unusually high number of speakers: {len(speakers)}",
            "severity": "medium"
        })
    
    # Check for timestamp gaps
    prev_timestamp = None
    for i, segment in enumerate(segments):
        if prev_timestamp:
            gap = parse_timestamp_to_seconds(segment.timestamp) - parse_timestamp_to_seconds(prev_timestamp)
            if gap > 30:  # More than 30 seconds gap
                issues.append({
                    "type": "timestamp_gap",
                    "segment": i,
                    "message": f"Large gap in timestamps ({gap:.1f} seconds)",
                    "severity": "medium"
                })
        prev_timestamp = segment.timestamp
    
    return issues


def calculate_overall_score(
    deps: QualityDeps,
    metrics: Dict[str, float]
) -> float:
    """Calculate weighted overall quality score"""
    
    weights = deps.weights
    
    score = 0.0
    total_weight = 0.0
    
    # Map metrics to weight keys
    metric_mapping = {
        'readability': 'readability',
        'vocabulary_richness': 'vocabulary',
        'sentence_variety': 'sentence_variety',
        'punctuation_density': 'punctuation',
        'speaker_consistency': 'consistency'
    }
    
    for metric_key, weight_key in metric_mapping.items():
        if metric_key in metrics and weight_key in weights:
            metric_value = metrics[metric_key]
            
            # Special handling for punctuation density (lower is better)
            if metric_key == 'punctuation_density':
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
        1 for seg in segments 
        if re.match(r'^\[\d{2}:\d{2}:\d{2}\]$', seg.timestamp)
    )
    
    coverage = (valid_timestamps / len(segments)) * 100
    return coverage


def calculate_speaker_consistency(segments: List[TranscriptSegment]) -> float:
    """Calculate speaker labeling consistency"""
    
    if not segments:
        return 0.0
    
    # Check for consistent speaker format
    speaker_pattern = r'^Speaker \d+$'
    consistent_speakers = sum(
        1 for seg in segments
        if re.match(speaker_pattern, seg.speaker.strip())
    )
    
    consistency = (consistent_speakers / len(segments)) * 100
    return consistency


def parse_timestamp_to_seconds(timestamp: str) -> float:
    """Parse [HH:MM:SS] to seconds"""
    
    match = re.match(r'^\[(\d{2}):(\d{2}):(\d{2})\]$', timestamp)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return 0.0