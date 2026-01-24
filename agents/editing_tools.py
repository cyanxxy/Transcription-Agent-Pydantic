"""
Editing Tools utilities
Provides smart editing and formatting capabilities
"""

from typing import Dict, Any, List, Tuple
import re
import logging

from models import TranscriptSegment
from dependencies import EditingDeps

logger = logging.getLogger(__name__)


def auto_format_transcript(
    deps: EditingDeps, segments: List[TranscriptSegment]
) -> Tuple[List[TranscriptSegment], List[str]]:
    """Auto-format transcript segments"""

    formatted_segments = []
    changes_applied = []

    for segment in segments:
        text = segment.text
        original_text = text

        # Apply formatting based on settings
        if deps.remove_extra_spaces:
            text = re.sub(r"\s+", " ", text).strip()
            if text != original_text:
                changes_applied.append("Removed extra spaces")

        if deps.fix_punctuation_spacing:
            text = fix_punctuation_spacing(text)
            if text != original_text:
                changes_applied.append("Fixed punctuation spacing")

        if deps.remove_fillers and deps.filler_words:
            text = remove_filler_words_helper(text, deps.filler_words)
            if text != original_text:
                changes_applied.append("Removed filler words")

        if deps.sentence_case:
            text = apply_sentence_case(text)
            if text != original_text:
                changes_applied.append("Applied sentence case")

        # Apply common replacements
        if deps.replacements:
            for old, new in deps.replacements.items():
                pattern = r"\b" + re.escape(old) + r"\b"
                text = re.sub(pattern, new, text, flags=re.IGNORECASE)

        # Create formatted segment
        formatted_segment = TranscriptSegment(
            timestamp=segment.timestamp,
            speaker=segment.speaker,
            text=text,
            confidence=segment.confidence,
        )
        formatted_segments.append(formatted_segment)

    # Remove duplicate change descriptions
    changes_applied = list(set(changes_applied))

    return formatted_segments, changes_applied


def find_and_replace(
    segments: List[TranscriptSegment],
    find: str,
    replace: str,
    case_sensitive: bool = False,
    whole_word: bool = False,
) -> Dict[str, Any]:
    """Find and replace text in transcript segments"""

    modified_segments = []
    total_replacements = 0
    affected_segments = []

    for i, segment in enumerate(segments):
        text = segment.text

        # Build regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE

        if whole_word:
            pattern = r"\b" + re.escape(find) + r"\b"
        else:
            pattern = re.escape(find)

        # Count replacements
        matches = len(re.findall(pattern, text, flags=flags))

        if matches > 0:
            # Perform replacement
            new_text = re.sub(pattern, replace, text, flags=flags)
            total_replacements += matches
            affected_segments.append(i)

            # Create modified segment
            modified_segment = TranscriptSegment(
                timestamp=segment.timestamp,
                speaker=segment.speaker,
                text=new_text,
                confidence=segment.confidence,
            )
        else:
            modified_segment = segment

        modified_segments.append(modified_segment)

    return {
        "segments": modified_segments,
        "replacement_count": total_replacements,
        "affected_segments": affected_segments,
        "success": total_replacements > 0,
    }


def remove_filler_words_helper(text: str, filler_words: List[str]) -> str:
    """Remove filler words from text"""

    if not filler_words:
        return text

    # Create pattern for filler words
    filler_pattern = (
        r"\b(" + "|".join(re.escape(word) for word in filler_words) + r")\b"
    )

    # Remove fillers
    cleaned = re.sub(filler_pattern, "", text, flags=re.IGNORECASE)

    # Clean up extra spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Fix punctuation after removal
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)

    return cleaned


def fix_capitalization(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """Fix capitalization in transcript segments"""

    formatted_segments = []

    for segment in segments:
        text = segment.text

        # Split into sentences
        sentences = re.split(r"([.!?]+)", text)
        formatted_text = []

        for i, part in enumerate(sentences):
            if i % 2 == 0 and part:  # Text parts (not punctuation)
                part = part.strip()
                if part:
                    # Capitalize first letter
                    part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()

                    # Capitalize 'I'
                    part = re.sub(r"\bi\b", "I", part)

                    # Capitalize proper nouns (simplified)
                    part = capitalize_proper_nouns(part)

                formatted_text.append(part)
            else:
                formatted_text.append(part)

        # Join back together
        final_text = "".join(formatted_text)

        # Create formatted segment
        formatted_segment = TranscriptSegment(
            timestamp=segment.timestamp,
            speaker=segment.speaker,
            text=final_text,
            confidence=segment.confidence,
        )
        formatted_segments.append(formatted_segment)

    return formatted_segments


# Helper functions


def fix_punctuation_spacing(text: str) -> str:
    """Fix spacing around punctuation marks"""

    # Remove space before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Add space after punctuation if missing
    text = re.sub(r"([,.!?;:])([A-Za-z])", r"\1 \2", text)

    # Fix multiple punctuation marks
    text = re.sub(r"([.!?])\1+", r"\1", text)

    # Fix ellipsis
    text = re.sub(r"\.{2,}", "...", text)

    return text


def apply_sentence_case(text: str) -> str:
    """Apply sentence case to text"""

    if not text:
        return text

    # Split by sentence-ending punctuation
    sentences = re.split(r"([.!?]+)", text)
    result = []

    for i, part in enumerate(sentences):
        if i % 2 == 0 and part:  # Text parts
            part = part.strip()
            if part:
                # Capitalize first letter
                part = (
                    part[0].upper() + part[1:].lower()
                    if len(part) > 1
                    else part.upper()
                )

                # Keep 'I' capitalized
                part = re.sub(r"\bi\b", "I", part)

            result.append(part)
        else:
            result.append(part)

    return "".join(result)


def capitalize_proper_nouns(text: str) -> str:
    """Capitalize common proper nouns"""

    # Common proper nouns to capitalize
    proper_nouns = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "English",
        "Spanish",
        "French",
        "German",
        "Chinese",
        "Japanese",
        "America",
        "American",
        "Europe",
        "European",
        "Asia",
        "Asian",
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Facebook",
        "Twitter",
    ]

    for noun in proper_nouns:
        pattern = r"\b" + noun.lower() + r"\b"
        text = re.sub(pattern, noun, text, flags=re.IGNORECASE)

    return text
