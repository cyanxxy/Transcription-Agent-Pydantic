"""
Context Agent using Pydantic AI
Manages user-provided context to improve transcription accuracy
"""

from pydantic_ai import Agent, RunContext
from typing import List, Optional
import logging

from models import TranscriptContext
from dependencies import ContextDeps

logger = logging.getLogger(__name__)


# Create context processing agent
context_agent = Agent(
    'gemini-2.5-flash',
    deps_type=ContextDeps,
    output_type=str,  # Returns formatted context prompt
    system_prompt="""You are a context specialist that helps improve transcription accuracy.
    
    Your role is to:
    - Analyze user-provided context (speaker names, topics, technical terms)
    - Generate helpful instructions for the transcription agent
    - Identify domain-specific vocabulary
    - Suggest formatting preferences based on context
    """
)


@context_agent.tool
def process_context(
    ctx: RunContext[ContextDeps],
    user_context: TranscriptContext
) -> str:
    """Process user context and generate enhanced prompt instructions"""
    
    prompt_parts = []
    
    # Add speaker information
    if user_context.speaker_names:
        speakers_str = ", ".join(user_context.speaker_names)
        prompt_parts.append(f"KNOWN SPEAKERS: {speakers_str}")
        prompt_parts.append("IMPORTANT: You MUST use these exact speaker names in your transcription.")
        prompt_parts.append("Replace any 'Speaker 1', 'Speaker 2' etc. with the actual names provided above.")
        prompt_parts.append("If you can distinguish between voices, map them to these names based on context and voice characteristics.")
    
    # Add topic context
    if user_context.topic:
        prompt_parts.append(f"TOPIC/DOMAIN: {user_context.topic}")
        prompt_parts.append(f"Pay special attention to technical terms and jargon related to: {user_context.topic}")
    
    # Add technical terms
    if user_context.technical_terms:
        terms_str = ", ".join(user_context.technical_terms)
        prompt_parts.append(f"TECHNICAL VOCABULARY: {terms_str}")
        prompt_parts.append("Ensure these terms are spelled correctly and used appropriately.")
    
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
            "technical": "Include technical details, code snippets, and specifications accurately."
        }
        if user_context.expected_format in format_instructions:
            prompt_parts.append(f"FORMAT STYLE: {format_instructions[user_context.expected_format]}")
    
    # Combine all parts
    if prompt_parts:
        context_prompt = "\n\n=== USER-PROVIDED CONTEXT ===\n" + "\n".join(prompt_parts) + "\n=== END CONTEXT ===\n"
        return context_prompt
    
    return ""


@context_agent.tool  
def extract_keywords(
    ctx: RunContext[ContextDeps],
    text: str
) -> List[str]:
    """Extract important keywords from user-provided context"""
    
    import re
    
    # Basic keyword extraction
    # Remove common words
    common_words = {
        "the", "is", "at", "which", "on", "and", "a", "an", "as", "are", 
        "was", "were", "to", "in", "for", "of", "with", "from", "by"
    }
    
    # Split and clean text
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter out common words and short words
    keywords = [
        word for word in words 
        if word not in common_words and len(word) > 3
    ]
    
    # Get unique keywords
    unique_keywords = list(dict.fromkeys(keywords))
    
    return unique_keywords[:20]  # Return top 20 keywords


@context_agent.tool
def suggest_terminology(
    ctx: RunContext[ContextDeps],
    domain: str
) -> List[str]:
    """Suggest common terminology for a given domain"""
    
    domain_terms = {
        "medical": ["patient", "diagnosis", "treatment", "symptoms", "medication", "procedure", "vital signs", "history"],
        "legal": ["plaintiff", "defendant", "counsel", "objection", "evidence", "testimony", "jurisdiction", "precedent"],
        "technical": ["API", "database", "function", "variable", "deployment", "architecture", "algorithm", "framework"],
        "business": ["revenue", "ROI", "KPI", "stakeholder", "strategy", "quarterly", "forecast", "margin"],
        "academic": ["hypothesis", "methodology", "analysis", "conclusion", "literature", "research", "study", "findings"],
        "finance": ["portfolio", "investment", "dividend", "equity", "asset", "liability", "liquidity", "valuation"]
    }
    
    domain_lower = domain.lower()
    for key, terms in domain_terms.items():
        if key in domain_lower or domain_lower in key:
            return terms
    
    return []


def create_context_prompt(
    speaker_names: Optional[List[str]] = None,
    topic: Optional[str] = None,
    technical_terms: Optional[List[str]] = None,
    custom_instructions: Optional[str] = None,
    language_hints: Optional[str] = None,
    expected_format: Optional[str] = None
) -> str:
    """Simple function to create context prompt without agent"""
    
    context = TranscriptContext(
        speaker_names=speaker_names or [],
        topic=topic,
        technical_terms=technical_terms or [],
        custom_instructions=custom_instructions,
        language_hints=language_hints,
        expected_format=expected_format
    )
    
    # Create context dependencies
    deps = ContextDeps()
    
    # Process context
    ctx = RunContext(deps=deps, agent=context_agent)
    # Build prompt directly without agent call
    prompt = process_context(ctx, context)
    
    return prompt