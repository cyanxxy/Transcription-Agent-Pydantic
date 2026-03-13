"""
ExactTranscriber v2.0 - Main Application
Powered by Pydantic AI for robust transcription workflows
"""

import streamlit as st
import logging
from pathlib import Path
import nest_asyncio

# Allow nested event loops for Streamlit
nest_asyncio.apply()

# Must import after nest_asyncio.apply()
from workflow import TranscriptionWorkflow  # noqa: E402
from state_manager import StateManager  # noqa: E402
from models import ProcessingStatus  # noqa: E402
from styles import apply_custom_styles  # noqa: E402
from utils import run_async  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _format_strategy_label(strategy: str) -> str:
    """Human-readable strategy label."""
    labels = {
        "single_gemini": "Single Gemini + Judge",
        "dual_gemini": "Dual Gemini + Judge",
        "gemini_plus_parakeet": "Gemini + Parakeet + Judge",
    }
    return labels.get(strategy, strategy.replace("_", " ").title())


def _format_timing_status(result) -> str:
    """Summarize timing status for the result header."""
    if result.timestamps_corrected:
        return "Aligned"
    if result.judge_used:
        return "As Judged"
    return "Direct"


def _get_workflow(**kwargs) -> TranscriptionWorkflow:
    """Get or create a cached TranscriptionWorkflow from session state"""
    api_key = StateManager.get_api_key()
    model_name = StateManager.get_model_name()
    judge_model_name = StateManager.get_judge_model_name()
    candidate_strategy = StateManager.get_candidate_strategy()
    auto_format = StateManager.get_auto_format()
    remove_fillers = StateManager.get_remove_fillers()
    use_judge_pipeline = StateManager.get_use_judge_pipeline()

    # Build a cache key from current settings
    cache_key = (
        api_key,
        model_name,
        judge_model_name,
        candidate_strategy,
        auto_format,
        remove_fillers,
        use_judge_pipeline,
    )

    # Reuse cached workflow if settings haven't changed
    if (
        "workflow" in st.session_state
        and st.session_state.get("workflow_cache_key") == cache_key
    ):
        return st.session_state.workflow

    workflow = TranscriptionWorkflow(
        api_key=api_key,
        model_name=model_name,
        judge_model_name=judge_model_name,
        candidate_strategy=candidate_strategy,
        auto_format=auto_format,
        remove_fillers=remove_fillers,
        use_judge_pipeline=use_judge_pipeline,
        **kwargs,
    )
    st.session_state.workflow = workflow
    st.session_state.workflow_cache_key = cache_key
    return workflow


def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="ExactTranscriber v2.0",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()


async def handle_transcription(workflow: TranscriptionWorkflow, file):
    """Handle file transcription with new workflow"""

    StateManager.set_processing(file.name)

    try:
        # Read file data
        file_data = file.read()

        # Create progress container
        progress_container = st.container()

        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(msg, pct):
                progress_bar.progress(pct)
                status_text.text(msg)
                StateManager.update_progress(pct)

            # Get user context if provided
            user_context = st.session_state.get("user_context", None)

            # Run transcription with context
            result = await workflow.transcribe_audio(
                file_data,
                file.name,
                progress_callback=update_progress,
                user_context=user_context,
            )

            # Update state with result
            StateManager.set_complete(result)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Show success metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Quality Score", f"{result.quality.overall_score:.1f}/100")
            with col2:
                st.metric("Processing Time", f"{result.processing_time:.1f}s")
            with col3:
                st.metric("Segments", len(result.segments))
            with col4:
                st.metric("Candidates", len(result.candidates) or 1)
            with col5:
                st.metric("Timing", _format_timing_status(result))

            st.success("✅ Transcription complete!")

            # Force a rerun to show the transcript tabs
            st.rerun()

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        StateManager.set_error(str(e))
        st.error(f"❌ Transcription failed: {e}")


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.markdown("### Settings")

        # Model Selection - Gemini 3 only
        model_options = {
            "Flash (Fast)": "gemini-3-flash-preview",
            "3.1 Pro (Quality)": "gemini-3.1-pro-preview",
        }

        current_model = StateManager.get_model_name()
        model_values = list(model_options.values())
        if current_model not in model_values:
            current_model = model_values[0]
            StateManager.set_model_name(current_model)
        selected_model = st.radio(
            "Model",
            options=list(model_options.keys()),
            index=model_values.index(current_model),
            horizontal=True,
        )

        if model_options[selected_model] != current_model:
            StateManager.set_model_name(model_options[selected_model])

        # Processing Options - Compact
        st.markdown("**Options**")
        col1, col2 = st.columns(2)
        with col1:
            auto_format = st.checkbox(
                "Auto-format",
                value=StateManager.get_auto_format(),
                key="auto_fmt",
            )
        with col2:
            remove_fillers = st.checkbox(
                "No fillers",
                value=StateManager.get_remove_fillers(),
                key="no_fill",
            )

        st.session_state.auto_format = auto_format
        st.session_state.remove_fillers = remove_fillers
        StateManager.set_auto_format(auto_format)
        StateManager.set_remove_fillers(remove_fillers)

        # Judge Pipeline Options
        st.markdown("**Advanced**")
        use_judge_pipeline = st.checkbox(
            "Use Judge Pipeline",
            value=StateManager.get_use_judge_pipeline(),
            key="use_judge_pipeline_toggle",
            help="Generate transcript candidates first, then use a judge agent to pick or merge the final transcript.",
        )

        st.session_state.use_judge_pipeline = use_judge_pipeline
        StateManager.set_use_judge_pipeline(use_judge_pipeline)

        if use_judge_pipeline:
            strategy_options = {
                "Single Gemini + Judge": "single_gemini",
                "Dual Gemini + Judge": "dual_gemini",
                "Gemini + Parakeet + Judge": "gemini_plus_parakeet",
            }
            current_strategy = StateManager.get_candidate_strategy()
            strategy_values = list(strategy_options.values())
            if current_strategy not in strategy_values:
                current_strategy = "dual_gemini"
                StateManager.set_candidate_strategy(current_strategy)
                st.session_state.candidate_strategy = current_strategy

            selected_strategy = st.selectbox(
                "Candidate Strategy",
                options=list(strategy_options.keys()),
                index=strategy_values.index(current_strategy),
                help="Primary model comes from the Model selector above. Dual Gemini adds the other Gemini model. Gemini + Parakeet uses Parakeet as the second transcript candidate.",
            )
            selected_strategy_value = strategy_options[selected_strategy]
            st.session_state.candidate_strategy = selected_strategy_value
            StateManager.set_candidate_strategy(selected_strategy_value)
            st.caption(
                "Judge agent runs on Gemini 3.1 Pro by default and can still align timestamps after judging."
            )
        else:
            st.session_state.candidate_strategy = "single_gemini"
            StateManager.set_candidate_strategy("single_gemini")
            st.caption("Legacy direct single-model transcription")

        # Context Section
        st.markdown("---")
        st.markdown("**Context (Optional)**")

        with st.expander("Add context for better accuracy"):
            # Topic/Domain
            topic = st.text_input(
                "Topic/Domain",
                placeholder="e.g., Medical, Legal, Tech, Business",
                key="context_topic",
                help="Main subject or field of discussion",
            )

            # Speaker names
            speakers = st.text_input(
                "Speaker Names",
                placeholder="e.g., John, Sarah, Dr. Smith",
                key="context_speakers",
                help="Comma-separated names of speakers",
            )

            # Technical terms
            terms = st.text_area(
                "Technical Terms",
                placeholder="e.g., API, microservices, Kubernetes",
                key="context_terms",
                height=60,
                help="Important terms or jargon (one per line)",
            )

            # Format type
            format_type = st.selectbox(
                "Format Type",
                options=[
                    "Auto",
                    "Meeting",
                    "Interview",
                    "Lecture",
                    "Podcast",
                    "Legal",
                    "Medical",
                ],
                key="context_format",
            )

            # Save context to session state
            st.session_state.user_context = {
                "topic": topic if topic else None,
                "speakers": (
                    [s.strip() for s in speakers.split(",")] if speakers else None
                ),
                "terms": (
                    [t.strip() for t in terms.split("\n") if t.strip()]
                    if terms
                    else None
                ),
                "format": format_type.lower() if format_type != "Auto" else None,
            }

        # Footer - Minimal
        st.markdown("---")
        st.caption(
            "v2.1 (Judge Pipeline) | [Docs](https://github.com) | [API Key](https://makersuite.google.com)"
        )


def render_transcript_display():
    """Render transcript display and editing interface"""
    state = StateManager.get_state()

    if not state.transcript_result:
        return

    result = state.transcript_result

    # Display tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Transcript", "✏️ Edit", "📊 Quality", "💾 Export"]
    )

    with tab1:
        # Display transcript
        st.subheader("Transcript")

        if result.judge_used:
            st.info(
                "Judge pipeline: "
                f"{_format_strategy_label(result.candidate_strategy)} | "
                f"{len(result.candidates)} candidates | "
                f"judge model: {result.judge_model_used}"
            )

            if result.judge_selected_candidate_ids:
                st.caption(
                    "Selected candidates: "
                    + ", ".join(result.judge_selected_candidate_ids)
                )

            if result.judge_notes:
                with st.expander("Judge Notes"):
                    for note in result.judge_notes:
                        st.markdown(f"- {note}")

            if result.candidates:
                with st.expander("Candidate Summaries"):
                    for candidate in result.candidates:
                        quality_text = (
                            f"{candidate.quality_score:.1f}/100"
                            if candidate.quality_score is not None
                            else "n/a"
                        )
                        selected_suffix = (
                            " selected"
                            if candidate.candidate_id in result.judge_selected_candidate_ids
                            else ""
                        )
                        st.markdown(
                            f"**{candidate.label}** "
                            f"({candidate.kind}, quality {quality_text}{selected_suffix})"
                        )
                        for note in candidate.notes:
                            st.caption(note)
        else:
            st.info("Direct mode: single Gemini transcription without judge arbitration")

        # Show speakers
        speakers = result.unique_speakers
        if speakers:
            st.info(f"Speakers detected: {', '.join(speakers)}")

        # Display segments
        for segment in result.segments:
            with st.container():
                col1, col2 = st.columns([1, 10])
                with col1:
                    st.text(segment.timestamp)
                with col2:
                    st.markdown(f"**{segment.speaker}:** {segment.text}")

    with tab2:
        # Editing tools
        st.subheader("Editing Tools")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔧 Auto-Format", use_container_width=True):
                with st.spinner("Formatting..."):
                    workflow = _get_workflow()
                    edited_result = run_async(
                        workflow.edit_transcript(result, "auto_format")
                    )
                    StateManager.set_complete(edited_result)
                    st.success("Formatting applied!")
                    st.rerun()

            if st.button("🔤 Fix Capitalization", use_container_width=True):
                with st.spinner("Fixing capitalization..."):
                    workflow = _get_workflow()
                    edited_result = run_async(
                        workflow.edit_transcript(result, "fix_capitalization")
                    )
                    StateManager.set_complete(edited_result)
                    st.success("Capitalization fixed!")
                    st.rerun()

        with col2:
            st.markdown("**Find & Replace**")
            find_text = st.text_input("Find:")
            replace_text = st.text_input("Replace with:")

            col3, col4 = st.columns(2)
            with col3:
                case_sensitive = st.checkbox("Case sensitive")
            with col4:
                whole_word = st.checkbox("Whole words only")

            if st.button("Replace All", type="primary", use_container_width=True):
                if find_text:
                    with st.spinner("Replacing..."):
                        workflow = _get_workflow()
                        edited_result = run_async(
                            workflow.edit_transcript(
                                result,
                                "find_replace",
                                find=find_text,
                                replace=replace_text,
                                case_sensitive=case_sensitive,
                                whole_word=whole_word,
                            )
                        )
                        StateManager.set_complete(edited_result)
                        st.success("Replacements complete!")
                        st.rerun()

    with tab3:
        # Quality metrics
        st.subheader("Quality Analysis")

        quality = result.quality

        st.markdown(
            f"""
            <div style='text-align: center; padding: 20px; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 10px; color: white;'>
                <h1 style='margin: 0;'>{quality.overall_score:.0f}/100</h1>
                <p style='margin: 5px 0;'>{quality.quality_assessment}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Detailed metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Readability", f"{quality.readability:.0f}/100")
            st.metric("Timestamp Coverage", f"{quality.timestamp_coverage:.0f}%")

        with col2:
            st.metric("Vocabulary Richness", f"{quality.vocabulary_richness:.0f}/100")
            st.metric("Speaker Consistency", f"{quality.speaker_consistency:.0f}%")

        with col3:
            st.metric("Sentence Variety", f"{quality.sentence_variety:.0f}/100")
            st.metric("Punctuation Density", f"{quality.punctuation_density:.2%}")

        # Issues and warnings
        if quality.issues:
            st.warning(f"Found {len(quality.issues)} issues")
            with st.expander("View Issues"):
                for issue in quality.issues[:10]:
                    st.markdown(
                        f"- **{issue.get('type', 'Unknown')}**: {issue.get('message', '')}"
                    )

        if quality.warnings:
            with st.expander("Warnings"):
                for warning in quality.warnings:
                    st.warning(warning)

    with tab4:
        # Export options
        st.subheader("Export Transcript")

        export_format = st.selectbox(
            "Select Format",
            options=["Text (.txt)", "Subtitles (.srt)", "JSON (.json)"],
            index=0,
        )

        format_map = {
            "Text (.txt)": "txt",
            "Subtitles (.srt)": "srt",
            "JSON (.json)": "json",
        }

        if st.button("📥 Download", type="primary", use_container_width=True):
            workflow = _get_workflow()
            export_content = run_async(
                workflow.export_transcript(result, format_map[export_format])
            )

            # Create download button
            st.download_button(
                label=f"Download {export_format}",
                data=export_content,
                file_name=f"transcript.{format_map[export_format]}",
                mime="text/plain",
            )


def main():
    """Main application entry point"""

    # Setup page
    setup_page()

    # Initialize state
    state = StateManager.get_state()

    # Header - Compact
    col1, col2 = st.columns([10, 1])
    with col1:
        st.title("ExactTranscriber")
        st.caption("Multi-agent transcription with candidate models, a judge agent, and Parakeet alignment")
    with col2:
        if state.transcript_result and st.button("New"):
            StateManager.reset_state()
            st.rerun()

    # Render sidebar
    render_sidebar()

    # Check for API key silently - don't show anything in UI
    if not StateManager.get_api_key():
        st.error("Configuration error. Please check setup.")
        st.stop()

    # Main content area
    if state.status == ProcessingStatus.IDLE:
        # File upload interface
        st.markdown("### Upload Audio File")

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac"],
            help="Maximum file size: 200MB",
        )

        if uploaded_file:
            # Display file info
            from utils import estimate_judge_pipeline_cost

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("File", uploaded_file.name)
            with col2:
                size_mb = uploaded_file.size / (1024 * 1024)
                st.metric("Size", f"{size_mb:.1f} MB")
            with col3:
                file_type = Path(uploaded_file.name).suffix.lstrip(".")
                st.metric("Format", file_type.upper())
            with col4:
                use_judge_pipeline = StateManager.get_use_judge_pipeline()
                candidate_strategy = StateManager.get_candidate_strategy()
                pipeline_label = (
                    _format_strategy_label(candidate_strategy)
                    if use_judge_pipeline
                    else "Direct Gemini"
                )
                st.metric("Pipeline", pipeline_label)
            with col5:
                # Estimate cost (rough estimate based on file size)
                estimated_duration = size_mb * 60  # Rough: 1MB ≈ 1 minute for MP3
                _, cost_str = estimate_judge_pipeline_cost(
                    estimated_duration,
                    StateManager.get_model_name(),
                    StateManager.get_candidate_strategy(),
                    StateManager.get_use_judge_pipeline(),
                    StateManager.get_judge_model_name(),
                )
                st.metric("Est. API Cost", cost_str)

            # Show active context if any
            if hasattr(st.session_state, "user_context"):
                ctx = st.session_state.user_context
                active_context = []
                if ctx.get("topic"):
                    active_context.append(f"Topic: {ctx['topic']}")
                if ctx.get("speakers"):
                    active_context.append(f"Speakers: {', '.join(ctx['speakers'])}")
                if ctx.get("format"):
                    active_context.append(f"Format: {ctx['format']}")

                if active_context:
                    st.info("**Using context:** " + " | ".join(active_context))

            # Transcribe button
            if st.button(
                "🚀 Start Transcription", type="primary", use_container_width=True
            ):
                # Initialize workflow with the current pipeline configuration.
                workflow = _get_workflow()

                # Run transcription with proper event loop handling
                run_async(handle_transcription(workflow, uploaded_file))

    elif state.status == ProcessingStatus.PROCESSING:
        # Show processing status
        st.info(f"Processing: {state.current_file}")
        st.progress(state.processing_progress)

    elif state.status == ProcessingStatus.COMPLETE:
        # Show transcript and tools
        render_transcript_display()

    elif state.status == ProcessingStatus.ERROR:
        # Show error
        st.error(f"Error: {state.error.message if state.error else 'Unknown error'}")

        if state.error and state.error.recoverable:
            if st.button("🔄 Try Again"):
                StateManager.reset_state()
                st.rerun()


if __name__ == "__main__":
    main()
