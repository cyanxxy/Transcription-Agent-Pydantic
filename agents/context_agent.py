"""
Context Agent using Pydantic AI
Manages user-provided context to improve transcription accuracy
"""

from typing import List, Optional
import logging

from models import TranscriptContext

logger = logging.getLogger(__name__)


def process_context(user_context: TranscriptContext) -> str:
    """Process user context and generate enhanced prompt instructions"""

    prompt_parts = []

    # Add speaker information
    if user_context.speaker_names:
        speakers_str = ", ".join(user_context.speaker_names)
        prompt_parts.append(f"KNOWN SPEAKERS: {speakers_str}")
        prompt_parts.append(
            "IMPORTANT: You MUST use these exact speaker names in your transcription."
        )
        prompt_parts.append(
            "Replace any 'Speaker 1', 'Speaker 2' etc. with the actual names provided above."
        )
        prompt_parts.append(
            "If you can distinguish between voices, map them to these names based on context and voice characteristics."
        )

    # Add topic context
    if user_context.topic:
        prompt_parts.append(f"TOPIC/DOMAIN: {user_context.topic}")
        prompt_parts.append(
            f"Pay special attention to technical terms and jargon related to: {user_context.topic}"
        )

    # Add technical terms
    if user_context.technical_terms:
        terms_str = ", ".join(user_context.technical_terms)
        prompt_parts.append(f"TECHNICAL VOCABULARY: {terms_str}")
        prompt_parts.append(
            "Ensure these terms are spelled correctly and used appropriately."
        )

    # Add custom instructions
    if user_context.custom_instructions:
        prompt_parts.append(f"SPECIAL INSTRUCTIONS: {user_context.custom_instructions}")

    # Add language/accent info
    if user_context.language_hints:
        prompt_parts.append(f"LANGUAGE/ACCENT INFO: {user_context.language_hints}")

    # Format expectations based on context
    if user_context.expected_format:
        format_instructions = {
            "meeting": "Format as meeting minutes with clear speaker turns and action items.",
            "interview": "Format as Q&A with clear distinction between interviewer and interviewee.",
            "lecture": "Format as educational content with main points and examples clearly marked.",
            "podcast": "Format as conversational dialogue with natural flow and speaker personalities.",
            "legal": "Format with precise language, proper legal terminology, and clear attribution.",
            "medical": "Use proper medical terminology and maintain HIPAA-appropriate formatting.",
            "technical": "Include technical details, code snippets, and specifications accurately.",
        }
        if user_context.expected_format in format_instructions:
            prompt_parts.append(
                f"FORMAT STYLE: {format_instructions[user_context.expected_format]}"
            )

    # Combine all parts
    if prompt_parts:
        context_prompt = (
            "\n\n=== USER-PROVIDED CONTEXT ===\n"
            + "\n".join(prompt_parts)
            + "\n=== END CONTEXT ===\n"
        )
        return context_prompt

    return ""


def create_context_prompt(
    speaker_names: Optional[List[str]] = None,
    topic: Optional[str] = None,
    technical_terms: Optional[List[str]] = None,
    custom_instructions: Optional[str] = None,
    language_hints: Optional[str] = None,
    expected_format: Optional[str] = None,
) -> str:
    """Simple function to create context prompt"""

    context = TranscriptContext(
        speaker_names=speaker_names or [],
        topic=topic,
        technical_terms=technical_terms or [],
        custom_instructions=custom_instructions,
        language_hints=language_hints,
        expected_format=expected_format,
    )

    # Build prompt directly
    prompt = process_context(context)

    return prompt
