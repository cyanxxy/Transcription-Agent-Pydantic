import streamlit as st


@st.cache_resource
def _get_custom_css() -> str:
    """Returns cached custom CSS for the studio dark theme."""
    return """<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ═══════════════════════════════════════
       CSS VARIABLES
       ═══════════════════════════════════════ */
    :root {
        --accent: #f0a500;
        --accent-hover: #ffb627;
        --accent-dim: rgba(240, 165, 0, 0.10);
        --accent-glow: rgba(240, 165, 0, 0.22);
        --bg-deep: #0c0e12;
        --bg-card: #14171d;
        --bg-elevated: #1b1f27;
        --bg-hover: #242830;
        --text-primary: #e8ebf0;
        --text-secondary: #8891a0;
        --text-muted: #555d6e;
        --border: rgba(255, 255, 255, 0.07);
        --border-strong: rgba(255, 255, 255, 0.12);
        --success: #34d399;
        --error: #f87171;
        --warning: #fbbf24;
        --radius: 10px;
        --radius-sm: 6px;
        --radius-lg: 14px;
    }

    /* ═══════════════════════════════════════
       GLOBAL TYPOGRAPHY
       ═══════════════════════════════════════ */
    body, .stApp,
    div.stButton > button,
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox, .stRadio, .stCheckbox,
    .stFileUploader,
    .stMarkdown, .stCaption,
    p, span, label {
        font-family: 'Outfit', -apple-system, sans-serif !important;
    }

    h1, h2, h3 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
    }

    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ═══════════════════════════════════════
       MAIN LAYOUT
       ═══════════════════════════════════════ */
    .stApp {
        background-color: var(--bg-deep) !important;
    }

    .main .block-container {
        padding: 2rem 2.5rem !important;
        max-width: 1100px !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    footer { display: none !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 99px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    /* ═══════════════════════════════════════
       SIDEBAR
       ═══════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 1.5rem 1.25rem !important;
    }

    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-bottom: 0.75rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] strong {
        color: var(--text-secondary) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: var(--text-muted) !important;
        font-size: 0.72rem !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
        margin: 1rem 0 !important;
    }

    /* ═══════════════════════════════════════
       BUTTONS
       ═══════════════════════════════════════ */
    div.stButton > button {
        border-radius: var(--radius) !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.6rem 1.25rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid var(--border-strong) !important;
        letter-spacing: 0.01em !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent) 0%, #d49200 100%) !important;
        color: #0c0e12 !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(240, 165, 0, 0.25) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--accent-hover) 0%, var(--accent) 100%) !important;
        box-shadow: 0 4px 16px rgba(240, 165, 0, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    div.stButton > button[kind="primary"]:active {
        transform: translateY(0) !important;
        box-shadow: 0 1px 4px rgba(240, 165, 0, 0.2) !important;
    }

    div.stButton > button:not([kind="primary"]) {
        background-color: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-strong) !important;
    }

    div.stButton > button:not([kind="primary"]):hover {
        background-color: var(--bg-hover) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* Download button */
    [data-testid="stDownloadButton"] > button {
        border-radius: var(--radius) !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    /* ═══════════════════════════════════════
       TABS
       ═══════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--bg-card) !important;
        border-radius: var(--radius-lg) !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid var(--border) !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius) !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.84rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background-color: var(--bg-elevated) !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: #0c0e12 !important;
        font-weight: 600 !important;
        border-radius: var(--radius) !important;
    }

    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ═══════════════════════════════════════
       METRICS
       ═══════════════════════════════════════ */
    [data-testid="stMetric"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.875rem 1rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
    }

    /* ═══════════════════════════════════════
       FILE UPLOADER
       ═══════════════════════════════════════ */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border-strong) !important;
        border-radius: var(--radius-lg) !important;
        padding: 2rem !important;
        transition: border-color 0.3s ease, background-color 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        background-color: var(--accent-dim) !important;
    }

    [data-testid="stFileUploader"] label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* ═══════════════════════════════════════
       PROGRESS BAR
       ═══════════════════════════════════════ */
    .stProgress > div > div {
        background-color: var(--bg-elevated) !important;
        border-radius: 99px !important;
        height: 6px !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-hover)) !important;
        border-radius: 99px !important;
    }

    /* ═══════════════════════════════════════
       ALERTS
       ═══════════════════════════════════════ */
    [data-testid="stAlert"] {
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        font-size: 0.88rem !important;
    }

    /* ═══════════════════════════════════════
       TEXT INPUTS
       ═══════════════════════════════════════ */
    .stTextInput input,
    .stTextArea textarea {
        background-color: var(--bg-elevated) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        transition: border-color 0.2s ease !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-dim) !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* ═══════════════════════════════════════
       SELECTBOX
       ═══════════════════════════════════════ */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: var(--bg-elevated) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ═══════════════════════════════════════
       RADIO BUTTONS
       ═══════════════════════════════════════ */
    .stRadio > div[role="radiogroup"] > label {
        background-color: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        padding: 8px 14px !important;
        border-radius: var(--radius-sm) !important;
        transition: all 0.2s ease !important;
    }

    .stRadio > div[role="radiogroup"] > label:hover {
        border-color: var(--accent) !important;
        background-color: var(--accent-dim) !important;
    }

    .stRadio > div[role="radiogroup"] > label[data-is-checked="true"] {
        background-color: var(--accent-dim) !important;
        border-color: var(--accent) !important;
    }

    /* ═══════════════════════════════════════
       EXPANDERS
       ═══════════════════════════════════════ */
    [data-testid="stExpander"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }

    .streamlit-expanderHeader {
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        font-size: 0.88rem !important;
    }

    .streamlit-expanderHeader:hover {
        color: var(--accent) !important;
    }

    /* ═══════════════════════════════════════
       DIVIDERS
       ═══════════════════════════════════════ */
    hr, [data-testid="stDivider"] {
        border-color: var(--border) !important;
    }

    /* ═══════════════════════════════════════
       CUSTOM: APP HEADER
       ═══════════════════════════════════════ */
    .app-header {
        text-align: center;
        padding: 1rem 0 1.75rem;
        position: relative;
    }

    .app-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 48px;
        height: 3px;
        background: var(--accent);
        border-radius: 99px;
    }

    .app-badge {
        display: inline-block;
        background: var(--accent-dim);
        color: var(--accent);
        font-size: 0.65rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 99px;
        border: 1px solid rgba(240, 165, 0, 0.18);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        font-family: 'JetBrains Mono', monospace !important;
    }

    .app-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        letter-spacing: -0.03em !important;
        text-align: center !important;
        line-height: 1.15 !important;
    }

    .app-title span {
        background: linear-gradient(135deg, var(--accent), var(--accent-hover));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .app-subtitle {
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
        margin-top: 0.4rem !important;
        font-weight: 400 !important;
        text-align: center !important;
    }

    /* ═══════════════════════════════════════
       CUSTOM: INFO GRID
       ═══════════════════════════════════════ */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 10px;
        margin: 1rem 0;
    }

    .info-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.875rem 1rem;
        text-align: center;
        transition: border-color 0.2s ease;
    }

    .info-card:hover {
        border-color: var(--border-strong);
    }

    .info-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
        margin-bottom: 4px;
    }

    .info-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--text-primary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .info-value.accent {
        color: var(--accent);
    }

    /* ═══════════════════════════════════════
       CUSTOM: SECTION HEADERS
       ═══════════════════════════════════════ */
    .section-hdr {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .section-hdr .dot {
        width: 8px;
        height: 8px;
        background: var(--accent);
        border-radius: 50%;
        flex-shrink: 0;
    }

    .upload-hint {
        color: var(--text-muted);
        font-size: 0.82rem;
        margin-bottom: 1.25rem;
    }

    /* ═══════════════════════════════════════
       CUSTOM: CONTEXT BADGES
       ═══════════════════════════════════════ */
    .ctx-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 0.75rem 0 1rem;
    }

    .ctx-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: var(--accent-dim);
        color: var(--accent);
        font-size: 0.78rem;
        padding: 4px 12px;
        border-radius: 99px;
        border: 1px solid rgba(240, 165, 0, 0.15);
        font-weight: 500;
    }

    /* ═══════════════════════════════════════
       CUSTOM: PIPELINE BANNER
       ═══════════════════════════════════════ */
    .pipeline-banner {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: var(--radius);
        padding: 0.75rem 1rem;
        font-size: 0.84rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
        line-height: 1.5;
    }

    .pipeline-banner strong {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* ═══════════════════════════════════════
       CUSTOM: TRANSCRIPT SEGMENTS
       ═══════════════════════════════════════ */
    .segment-container {
        margin: 1rem 0;
    }

    .segment {
        display: flex;
        gap: 1rem;
        padding: 0.75rem 0.875rem;
        border-radius: var(--radius-sm);
        margin-bottom: 1px;
        transition: background-color 0.15s ease;
        border-left: 3px solid transparent;
    }

    .segment:nth-child(even) {
        background-color: rgba(255, 255, 255, 0.012);
    }

    .segment:hover {
        background-color: var(--bg-card);
        border-left-color: var(--accent);
    }

    .segment-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--accent);
        font-weight: 500;
        flex-shrink: 0;
        padding-top: 2px;
        opacity: 0.75;
    }

    .segment:hover .segment-time {
        opacity: 1;
    }

    .segment-body {
        flex: 1;
        min-width: 0;
    }

    .segment-speaker {
        font-weight: 600;
        color: var(--text-secondary);
        font-size: 0.76rem;
        margin-bottom: 2px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .segment-text {
        color: var(--text-primary);
        line-height: 1.65;
        font-size: 0.92rem;
    }

    /* ═══════════════════════════════════════
       CUSTOM: QUALITY SCORE RING
       ═══════════════════════════════════════ */
    .quality-hero {
        text-align: center;
        padding: 2rem;
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }

    .quality-ring {
        position: relative;
        width: 140px;
        height: 140px;
        margin: 0 auto 1rem;
    }

    .quality-ring .inner {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .quality-val {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1;
    }

    .quality-max {
        font-size: 0.9rem;
        color: var(--text-muted);
        font-weight: 400;
    }

    .quality-assess {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .quality-assess.excellent { color: var(--success); }
    .quality-assess.good { color: var(--accent); }
    .quality-assess.fair { color: var(--warning); }
    .quality-assess.poor { color: var(--error); }

    /* ═══════════════════════════════════════
       CUSTOM: CANDIDATE ITEMS
       ═══════════════════════════════════════ */
    .cand-item {
        background: var(--bg-elevated);
        border-radius: var(--radius-sm);
        padding: 0.75rem 1rem;
        margin-bottom: 8px;
        border: 1px solid var(--border);
    }

    .cand-item.selected {
        border-color: var(--accent);
        background: var(--accent-dim);
    }

    .cand-item .cand-label {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.88rem;
    }

    .cand-item .cand-meta {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 2px;
    }

    .cand-item .cand-note {
        color: var(--text-secondary);
        font-size: 0.76rem;
        margin-top: 4px;
        font-style: italic;
    }

    /* ═══════════════════════════════════════
       CUSTOM: RESULT SUMMARY STRIP
       ═══════════════════════════════════════ */
    .result-strip {
        display: flex;
        gap: 1.5rem;
        align-items: center;
        justify-content: center;
        padding: 0.875rem 1.25rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        margin-bottom: 1.25rem;
        flex-wrap: wrap;
    }

    .result-stat {
        text-align: center;
    }

    .result-stat .stat-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--text-primary);
    }

    .result-stat .stat-value.accent {
        color: var(--accent);
    }

    .result-stat .stat-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 1px;
    }

    .result-sep {
        width: 1px;
        height: 28px;
        background: var(--border);
    }

    /* ═══════════════════════════════════════
       SIDEBAR BRAND
       ═══════════════════════════════════════ */
    .sidebar-brand {
        text-align: center;
        padding: 0.25rem 0 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border);
    }

    .sidebar-brand .brand-text {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 1rem;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    .sidebar-brand .brand-accent {
        color: var(--accent);
    }

    /* ═══════════════════════════════════════
       ANIMATION
       ═══════════════════════════════════════ */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: fadeIn 0.35s ease-out forwards;
    }
</style>"""


def apply_custom_styles():
    """Applies custom CSS styles to the Streamlit app."""
    st.markdown(_get_custom_css(), unsafe_allow_html=True)
