"""
ExactTranscriber v2.0 - Main Application
Powered by Pydantic AI for robust transcription workflows
"""

import streamlit as st
import asyncio
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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Quality Score", f"{result.quality.overall_score:.1f}/100")
            with col2:
                st.metric("Processing Time", f"{result.processing_time:.1f}s")
            with col3:
                st.metric("Segments", len(result.segments))

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

        # Model Selection - Compact
        model_options = {
            "Flash (Fast)": "gemini-2.5-flash",
            "Pro (Quality)": "gemini-2.5-pro",
        }

        current_model = StateManager.get_model_name()
        selected_model = st.radio(
            "Model",
            options=list(model_options.keys()),
            index=list(model_options.values()).index(current_model),
            horizontal=True,
        )

        if model_options[selected_model] != current_model:
            StateManager.set_model_name(model_options[selected_model])

        # Processing Options - Compact
        st.markdown("**Options**")
        col1, col2 = st.columns(2)
        with col1:
            auto_format = st.checkbox("Auto-format", value=True, key="auto_fmt")
        with col2:
            remove_fillers = st.checkbox("No fillers", value=False, key="no_fill")

        st.session_state.auto_format = auto_format
        st.session_state.remove_fillers = remove_fillers

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
            "v2.0 | [Docs](https://github.com) | [API Key](https://makersuite.google.com)"
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
                    # Apply auto-formatting
                    workflow = TranscriptionWorkflow(StateManager.get_api_key())
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    edited_result = loop.run_until_complete(
                        workflow.edit_transcript(result, "auto_format")
                    )
                    loop.close()
                    StateManager.set_complete(edited_result)
                    st.success("Formatting applied!")
                    st.rerun()

            if st.button("🔤 Fix Capitalization", use_container_width=True):
                with st.spinner("Fixing capitalization..."):
                    workflow = TranscriptionWorkflow(StateManager.get_api_key())
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    edited_result = loop.run_until_complete(
                        workflow.edit_transcript(result, "fix_capitalization")
                    )
                    loop.close()
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
                        workflow = TranscriptionWorkflow(StateManager.get_api_key())
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        edited_result = loop.run_until_complete(
                            workflow.edit_transcript(
                                result,
                                "find_replace",
                                find=find_text,
                                replace=replace_text,
                                case_sensitive=case_sensitive,
                                whole_word=whole_word,
                            )
                        )
                        loop.close()
                        StateManager.set_complete(edited_result)
                        st.success("Replacements complete!")
                        st.rerun()

    with tab3:
        # Quality metrics
        st.subheader("Quality Analysis")

        quality = result.quality

        # Overall score with color
        # Unused: score_color = "green" if quality.overall_score >= 70 else "orange" if quality.overall_score >= 40 else "red"
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
            workflow = TranscriptionWorkflow(StateManager.get_api_key())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            export_content = loop.run_until_complete(
                workflow.export_transcript(result, format_map[export_format])
            )
            loop.close()

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
        st.caption("Audio transcription with Google Gemini 2.5")
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
            from utils import estimate_transcription_cost

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("File", uploaded_file.name)
            with col2:
                size_mb = uploaded_file.size / (1024 * 1024)
                st.metric("Size", f"{size_mb:.1f} MB")
            with col3:
                file_type = Path(uploaded_file.name).suffix.lstrip(".")
                st.metric("Format", file_type.upper())
            with col4:
                # Estimate cost (rough estimate based on file size)
                estimated_duration = size_mb * 60  # Rough: 1MB ≈ 1 minute for MP3
                _, cost_str = estimate_transcription_cost(
                    estimated_duration, StateManager.get_model_name()
                )
                st.metric("Est. Cost", cost_str)

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
                # Initialize workflow
                workflow = TranscriptionWorkflow(
                    api_key=StateManager.get_api_key(),
                    model_name=StateManager.get_model_name(),
                    auto_format=st.session_state.get("auto_format", True),
                    remove_fillers=st.session_state.get("remove_fillers", False),
                )

                # Run transcription with proper event loop handling
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(handle_transcription(workflow, uploaded_file))
                loop.close()

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
