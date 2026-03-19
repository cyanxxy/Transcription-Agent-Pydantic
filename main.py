"""
ExactTranscriber v2.0 - Main Application
Powered by Pydantic AI for robust transcription workflows
"""

import html as html_mod
import logging
from pathlib import Path

import nest_asyncio
import streamlit as st

# Allow nested event loops for Streamlit
nest_asyncio.apply()

# Must import after nest_asyncio.apply()
from workflow import TranscriptionWorkflow  # noqa: E402
from state_manager import StateManager  # noqa: E402
from models import ProcessingStatus  # noqa: E402
from styles import apply_custom_styles  # noqa: E402
from utils import run_async  # noqa: E402
from streamlit_deps import build_app_deps_from_streamlit  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────
#  Helpers
# ──────────────────────────────────────


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


def _esc(text: str) -> str:
    """HTML-escape user content."""
    return html_mod.escape(str(text))


def _get_workflow(**kwargs) -> TranscriptionWorkflow:
    """Get or create a cached TranscriptionWorkflow from session state."""
    app_deps = build_app_deps_from_streamlit(st)
    if app_deps is not None:
        transcription = app_deps.transcription
        api_key = transcription.api_key
        model_name = transcription.model_name
        judge_model_name = transcription.judge_model_name
        candidate_strategy = transcription.candidate_strategy
        auto_format = transcription.auto_format
        remove_fillers = transcription.remove_fillers
        use_judge_pipeline = transcription.use_judge_pipeline
        transcription_thinking_level = transcription.transcription_thinking_level
        judge_thinking_level = transcription.judge_thinking_level
    else:
        api_key = StateManager.get_api_key() or ""
        model_name = StateManager.get_model_name()
        judge_model_name = StateManager.get_judge_model_name()
        candidate_strategy = StateManager.get_candidate_strategy()
        auto_format = StateManager.get_auto_format()
        remove_fillers = StateManager.get_remove_fillers()
        use_judge_pipeline = StateManager.get_use_judge_pipeline()
        transcription_thinking_level = "high"
        judge_thinking_level = "medium"

    cache_key = (
        api_key,
        model_name,
        judge_model_name,
        candidate_strategy,
        auto_format,
        remove_fillers,
        use_judge_pipeline,
        transcription_thinking_level,
        judge_thinking_level,
    )

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
        transcription_thinking_level=transcription_thinking_level,
        judge_thinking_level=judge_thinking_level,
        **kwargs,
    )
    st.session_state.workflow = workflow
    st.session_state.workflow_cache_key = cache_key
    return workflow


# ──────────────────────────────────────
#  Custom HTML Renderers
# ──────────────────────────────────────


def _render_header():
    """Render the branded app header."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-badge">v2.1 &middot; Judge Pipeline</div>
            <div class="app-title">Exact<span>Transcriber</span></div>
            <div class="app-subtitle">
                Multi-agent transcription with candidate models, judge arbitration, and Parakeet alignment
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_info_grid(items):
    """Render a grid of info cards. items: list of (label, value, is_accent)."""
    cards = []
    for label, value, accent in items:
        cls = " accent" if accent else ""
        cards.append(
            f'<div class="info-card">'
            f'<div class="info-label">{_esc(label)}</div>'
            f'<div class="info-value{cls}">{_esc(str(value))}</div>'
            f"</div>"
        )
    st.markdown(
        f'<div class="info-grid">{"".join(cards)}</div>', unsafe_allow_html=True
    )


def _render_context_badges(ctx):
    """Render context as inline badges."""
    parts = []
    if ctx.get("topic"):
        parts.append(f'<span class="ctx-badge">Topic: {_esc(ctx["topic"])}</span>')
    if ctx.get("speakers"):
        names = ", ".join(_esc(s) for s in ctx["speakers"])
        parts.append(f'<span class="ctx-badge">Speakers: {names}</span>')
    if ctx.get("format"):
        parts.append(f'<span class="ctx-badge">Format: {_esc(ctx["format"])}</span>')
    if ctx.get("terms"):
        count = len(ctx["terms"])
        parts.append(f'<span class="ctx-badge">{count} term(s)</span>')
    if parts:
        st.markdown(
            f'<div class="ctx-badges">{"".join(parts)}</div>',
            unsafe_allow_html=True,
        )


def _render_result_strip(result):
    """Render a compact summary strip for the completed result."""

    def _stat(value, label, accent=False):
        cls = " accent" if accent else ""
        return (
            f'<div class="result-stat">'
            f'<div class="stat-value{cls}">{_esc(str(value))}</div>'
            f'<div class="stat-label">{_esc(label)}</div>'
            f"</div>"
        )

    sep = '<div class="result-sep"></div>'
    parts = [
        _stat(f"{result.quality.overall_score:.0f}/100", "Quality", accent=True),
        sep,
        _stat(f"{result.processing_time:.1f}s", "Time"),
        sep,
        _stat(str(len(result.segments)), "Segments"),
        sep,
        _stat(str(len(result.candidates) or 1), "Candidates"),
        sep,
        _stat(_format_timing_status(result), "Timing"),
    ]
    st.markdown(
        f'<div class="result-strip animate-in">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def _render_segments_html(segments):
    """Render transcript segments as styled HTML."""
    parts = ['<div class="segment-container">']
    for seg in segments:
        parts.append(
            f'<div class="segment">'
            f'<div class="segment-time">{_esc(seg.timestamp)}</div>'
            f'<div class="segment-body">'
            f'<div class="segment-speaker">{_esc(seg.speaker)}</div>'
            f'<div class="segment-text">{_esc(seg.text)}</div>'
            f"</div></div>"
        )
    parts.append("</div>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def _render_quality_ring(score, assessment):
    """Render an SVG quality score ring."""
    import math

    radius = 54
    circumference = 2 * math.pi * radius
    progress = max(0, min(score, 100)) / 100
    offset = circumference * (1 - progress)

    if score >= 80:
        color, css_cls = "#34d399", "excellent"
    elif score >= 60:
        color, css_cls = "#f0a500", "good"
    elif score >= 40:
        color, css_cls = "#fbbf24", "fair"
    else:
        color, css_cls = "#f87171", "poor"

    st.markdown(
        f"""
        <div class="quality-hero">
            <div class="quality-ring">
                <svg viewBox="0 0 120 120" width="140" height="140">
                    <circle cx="60" cy="60" r="{radius}" fill="none"
                            stroke="rgba(255,255,255,0.06)" stroke-width="8"/>
                    <circle cx="60" cy="60" r="{radius}" fill="none"
                            stroke="{color}" stroke-width="8"
                            stroke-dasharray="{circumference:.1f}"
                            stroke-dashoffset="{offset:.1f}"
                            stroke-linecap="round"
                            transform="rotate(-90 60 60)"
                            style="transition: stroke-dashoffset 0.8s ease-out;"/>
                </svg>
                <div class="inner">
                    <div class="quality-val">{score:.0f}</div>
                    <div class="quality-max">/100</div>
                </div>
            </div>
            <div class="quality-assess {css_cls}">{_esc(assessment)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pipeline_banner(result):
    """Render judge pipeline info banner."""
    strategy = _format_strategy_label(result.candidate_strategy)
    model = _esc(str(result.judge_model_used or "n/a"))
    n_candidates = len(result.candidates) or 1
    st.markdown(
        f'<div class="pipeline-banner">'
        f"<strong>Judge Pipeline</strong> &mdash; "
        f"{_esc(strategy)} &middot; "
        f"{n_candidates} candidate(s) &middot; "
        f"judge model: {model}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_candidates_html(candidates, selected_ids):
    """Render candidate summaries as styled HTML."""
    parts = []
    for c in candidates:
        is_sel = c.candidate_id in selected_ids
        cls = "cand-item selected" if is_sel else "cand-item"
        quality = f"{c.quality_score:.1f}/100" if c.quality_score is not None else "n/a"
        sel_tag = " &bull; selected" if is_sel else ""
        parts.append(
            f'<div class="{cls}">'
            f'<div class="cand-label">{_esc(c.label)}</div>'
            f'<div class="cand-meta">{_esc(c.kind)} &middot; quality {_esc(quality)}{sel_tag}</div>'
        )
        for note in c.notes:
            parts.append(f'<div class="cand-note">{_esc(note)}</div>')
        parts.append("</div>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


# ──────────────────────────────────────
#  Page Setup
# ──────────────────────────────────────


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="ExactTranscriber v2.0",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()


# ──────────────────────────────────────
#  Transcription Handler
# ──────────────────────────────────────


async def handle_transcription(workflow: TranscriptionWorkflow, file):
    """Handle file transcription with new workflow."""

    StateManager.set_processing(file.name)

    try:
        file_data = file.read()

        progress_container = st.container()

        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(msg, pct):
                progress_bar.progress(pct)
                status_text.text(msg)
                StateManager.update_progress(pct)

            user_context = st.session_state.get("user_context", None)

            result = await workflow.transcribe_audio(
                file_data,
                file.name,
                progress_callback=update_progress,
                user_context=user_context,
            )

            StateManager.set_complete(result)

            progress_bar.empty()
            status_text.empty()

            st.success("Transcription complete!")
            st.rerun()

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        StateManager.set_error(str(e))
        st.error(f"Transcription failed: {e}")


# ──────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────


def render_sidebar():
    """Render sidebar with settings."""
    with st.sidebar:
        # Brand
        st.markdown(
            '<div class="sidebar-brand">'
            '<span class="brand-text">Exact<span class="brand-accent">T</span></span>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Model")

        model_options = {
            "Flash (Balanced)": "gemini-3-flash-preview",
            "3.1 Flash-Lite (Fastest/Cheapest)": "gemini-3.1-flash-lite-preview",
            "3.1 Pro (Judge/Quality)": "gemini-3.1-pro-preview",
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
            label_visibility="collapsed",
        )

        if model_options[selected_model] != current_model:
            StateManager.set_model_name(model_options[selected_model])

        st.markdown("### Options")
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

        st.markdown("### Pipeline")
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
                help="Primary model comes from the Model selector above. "
                "Dual Gemini pairs Flash with Flash-Lite, and pairs Pro with Flash. "
                "Gemini + Parakeet uses Parakeet as the second candidate.",
            )
            selected_strategy_value = strategy_options[selected_strategy]
            st.session_state.candidate_strategy = selected_strategy_value
            StateManager.set_candidate_strategy(selected_strategy_value)
            st.caption(
                "Judge agent runs on Gemini 3.1 Pro by default; Flash-Lite is the low-cost secondary candidate."
            )
        else:
            st.session_state.candidate_strategy = "single_gemini"
            StateManager.set_candidate_strategy("single_gemini")
            st.caption("Direct single-model transcription")

        # Context Section
        st.markdown("---")
        st.markdown("### Context")

        with st.expander("Add context for better accuracy"):
            topic = st.text_input(
                "Topic / Domain",
                placeholder="e.g., Medical, Legal, Tech",
                key="context_topic",
            )

            speakers = st.text_input(
                "Speaker Names",
                placeholder="e.g., John, Sarah, Dr. Smith",
                key="context_speakers",
            )

            terms = st.text_area(
                "Technical Terms",
                placeholder="One per line",
                key="context_terms",
                height=60,
            )

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

        # Footer
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;">'
            '<span style="font-size:0.68rem;color:var(--text-muted);">'
            "v2.1 &middot; Judge Pipeline &middot; "
            '<a href="https://makersuite.google.com" '
            'style="color:var(--accent);text-decoration:none;">'
            "API Key</a></span></div>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────
#  Transcript Display
# ──────────────────────────────────────


def render_transcript_display():
    """Render transcript display and editing interface."""
    state = StateManager.get_state()

    if not state.transcript_result:
        return

    result = state.transcript_result

    tab1, tab2, tab3, tab4 = st.tabs(["Transcript", "Edit", "Quality", "Export"])

    with tab1:
        # Pipeline info
        if result.judge_used:
            _render_pipeline_banner(result)

            if result.judge_selected_candidate_ids:
                st.caption(
                    "Selected: " + ", ".join(result.judge_selected_candidate_ids)
                )

            if result.judge_notes:
                with st.expander("Judge Notes"):
                    for note in result.judge_notes:
                        st.markdown(f"- {note}")

            if result.candidates:
                with st.expander("Candidate Summaries"):
                    _render_candidates_html(
                        result.candidates, result.judge_selected_candidate_ids
                    )
        else:
            st.markdown(
                '<div class="pipeline-banner">'
                "<strong>Direct Mode</strong> &mdash; "
                "single Gemini transcription without judge arbitration"
                "</div>",
                unsafe_allow_html=True,
            )

        # Speaker info
        speakers = result.unique_speakers
        if speakers:
            st.caption(f"Speakers: {', '.join(speakers)}")

        # Segments
        _render_segments_html(result.segments)

    with tab2:
        st.markdown(
            '<div class="section-hdr"><span class="dot"></span>Editing Tools</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Auto-Format", use_container_width=True):
                with st.spinner("Formatting..."):
                    workflow = _get_workflow()
                    edited_result = run_async(
                        workflow.edit_transcript(result, "auto_format")
                    )
                    StateManager.set_complete(edited_result)
                    st.success("Formatting applied!")
                    st.rerun()

            if st.button("Fix Capitalization", use_container_width=True):
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
            find_text = st.text_input("Find:", key="edit_find")
            replace_text = st.text_input("Replace with:", key="edit_replace")

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
        quality = result.quality
        _render_quality_ring(quality.overall_score, quality.quality_assessment)

        st.divider()

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

        if quality.issues:
            st.warning(f"Found {len(quality.issues)} issues")
            with st.expander("View Issues"):
                for issue in quality.issues[:10]:
                    st.markdown(
                        f"- **{issue.get('type', 'Unknown')}**: "
                        f"{issue.get('message', '')}"
                    )

        if quality.warnings:
            with st.expander("Warnings"):
                for warning in quality.warnings:
                    st.warning(warning)

    with tab4:
        st.markdown(
            '<div class="section-hdr"><span class="dot"></span>Export Transcript</div>',
            unsafe_allow_html=True,
        )

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

        if st.button("Download", type="primary", use_container_width=True):
            workflow = _get_workflow()
            export_content = run_async(
                workflow.export_transcript(result, format_map[export_format])
            )

            st.download_button(
                label=f"Save {export_format}",
                data=export_content,
                file_name=f"transcript.{format_map[export_format]}",
                mime="text/plain",
            )


# ──────────────────────────────────────
#  Main
# ──────────────────────────────────────


def main():
    """Main application entry point."""

    setup_page()
    state = StateManager.get_state()

    # Custom branded header
    _render_header()

    # Sidebar
    render_sidebar()

    # API key check
    if not StateManager.get_api_key():
        st.error("Configuration error. Please check setup.")
        st.stop()

    # ── IDLE: Upload ─────────────────────
    if state.status == ProcessingStatus.IDLE:
        st.markdown(
            '<div class="section-hdr"><span class="dot"></span>Upload Audio</div>'
            '<div class="upload-hint">'
            "Drag and drop or browse &mdash; supports MP3, WAV, M4A, OGG, FLAC up to 200 MB"
            "</div>",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            from utils import estimate_judge_pipeline_cost

            size_mb = uploaded_file.size / (1024 * 1024)
            file_type = Path(uploaded_file.name).suffix.lstrip(".")
            use_judge = StateManager.get_use_judge_pipeline()
            strategy = StateManager.get_candidate_strategy()
            pipeline_label = (
                _format_strategy_label(strategy) if use_judge else "Direct Gemini"
            )
            estimated_duration = size_mb * 60
            _, cost_str = estimate_judge_pipeline_cost(
                estimated_duration,
                StateManager.get_model_name(),
                strategy,
                use_judge,
                StateManager.get_judge_model_name(),
            )

            _render_info_grid(
                [
                    ("File", uploaded_file.name, False),
                    ("Size", f"{size_mb:.1f} MB", False),
                    ("Format", file_type.upper(), False),
                    ("Pipeline", pipeline_label, True),
                    ("Est. Cost", cost_str, True),
                ]
            )

            # Context badges
            if hasattr(st.session_state, "user_context"):
                _render_context_badges(st.session_state.user_context)

            # Transcribe button
            if st.button(
                "Start Transcription", type="primary", use_container_width=True
            ):
                workflow = _get_workflow()
                run_async(handle_transcription(workflow, uploaded_file))

    # ── PROCESSING ───────────────────────
    elif state.status == ProcessingStatus.PROCESSING:
        st.info(f"Processing: {state.current_file}")
        st.progress(state.processing_progress)

    # ── COMPLETE ─────────────────────────
    elif state.status == ProcessingStatus.COMPLETE:
        # New session button
        cols = st.columns([9, 1])
        with cols[1]:
            if st.button("New", use_container_width=True):
                StateManager.reset_state()
                st.rerun()

        # Result summary strip
        if state.transcript_result:
            _render_result_strip(state.transcript_result)

        render_transcript_display()

    # ── ERROR ────────────────────────────
    elif state.status == ProcessingStatus.ERROR:
        st.error(f"Error: {state.error.message if state.error else 'Unknown error'}")

        if state.error and state.error.recoverable:
            if st.button("Try Again"):
                StateManager.reset_state()
                st.rerun()


if __name__ == "__main__":
    main()
