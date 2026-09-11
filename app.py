"""
Competitive Intelligence Briefing Crew – Streamlit Application.

Professional dashboard providing:
- Sidebar controls (topic, sources, steps, RAG upload)
- Live workflow progress and agent status
- Execution trace viewer
- Report viewer with section tabs
- Citation viewer
- Human approval gate
- Markdown and PDF download
- Audit log viewer
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="Competitive Intelligence Crew",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Lazy imports (defer heavy imports until after page config) ────────────────
@st.cache_resource
def _import_workflow():
    from graph.workflow import run_workflow
    return run_workflow


@st.cache_resource
def _import_exporters():
    from tools.markdown_export import export_markdown
    from tools.pdf_export import export_pdf
    return export_markdown, export_pdf


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #080b11;
    --bg-elevated: #10141c;
    --bg-elevated-2: #141926;
    --border-subtle: #1d2330;
    --border-strong: #262e40;
    --teal: #2dd4bf;
    --teal-soft: rgba(45, 212, 191, 0.14);
    --teal-border: rgba(45, 212, 191, 0.38);
    --teal-glow: rgba(45, 212, 191, 0.16);
    --blue-link: #7dd3fc;
    --text-primary: #eef1f8;
    --text-secondary: #8b93ab;
    --text-muted: #545c72;
    --warn: #fbbf24;
    --danger: #f87171;
    --success: #34d399;
}

html, body, [data-testid="stAppViewContainer"] * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Global background & base text ── */
[data-testid="stAppViewContainer"] {
    background: var(--bg-primary);
    color: var(--text-primary);
}
[data-testid="stHeader"] {
    background: var(--bg-primary);
}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div,
[data-testid="stAppViewContainer"] label {
    color: var(--text-primary);
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {
    color: #ffffff;
    font-weight: 700;
    letter-spacing: -0.01em;
}

/* ── Main content block ── */
[data-testid="block-container"] {
    background: var(--bg-primary);
    padding-top: 1.5rem;
    max-width: 1280px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #060811;
    border-right: 1px solid var(--border-subtle);
}
[data-testid="stSidebar"] > div {
    padding-top: 1.25rem;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stTextArea textarea {
    background: var(--bg-elevated) !important;
    color: #ffffff !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stTextArea textarea:focus {
    border: 1px solid var(--teal) !important;
    box-shadow: 0 0 0 1px var(--teal-border) !important;
}
[data-testid="stSidebar"] .stButton button {
    background: var(--bg-elevated) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: var(--bg-elevated-2) !important;
    color: #ffffff !important;
    border: 1px solid var(--teal-border) !important;
}

/* ── Primary run button ── */
[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: var(--teal) !important;
    color: #052e2b !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 0 20px var(--teal-glow);
}
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
    background: #5eead4 !important;
    color: #052e2b !important;
}

/* ── Sidebar section labels (uppercase eyebrow headers) ── */
.sb-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    margin: 4px 0 10px 0;
}

/* ── Slider ── */
[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--teal) !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--text-muted) !important;
}

/* ── Toggle ── */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-elevated);
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--teal);
    padding: 12px 16px;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--text-primary) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-weight: 700 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 22px;
    border-bottom: 1px solid var(--border-subtle);
}
[data-testid="stTabs"] [role="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 600;
    font-size: 14px;
    padding: 6px 2px 12px 2px;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--text-primary) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid var(--teal);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: var(--teal) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: var(--text-primary) !important;
    font-weight: 600;
}

/* ── Info / warning / error / success boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
}
[data-testid="stAlert"] p {
    color: #ffffff !important;
}

/* ── Code blocks ── */
[data-testid="stCode"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
}
[data-testid="stCode"] code {
    color: var(--success) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border-subtle);
}

/* ═══════════════════════════════════════════════════
   Agent pipeline — stage cards (matches reference UI)
   ═══════════════════════════════════════════════════ */
.stage-row {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 12px;
    margin: 4px 0 6px 0;
}
.stage-card {
    min-width: 0;
    border-radius: 16px;
    padding: 18px 8px 14px 8px;
    text-align: center;
    position: relative;
    transition: all .2s ease;
}
.stage-icon {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px auto;
}
.stage-num {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    display: block;
    margin-bottom: 2px;
}
.stage-label {
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    display: block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.stage-completed {
    background: linear-gradient(180deg, rgba(45,212,191,0.14), rgba(15,23,32,0.5));
    border: 1px solid var(--teal-border);
    box-shadow: 0 0 26px rgba(45,212,191,0.10);
}
.stage-completed .stage-icon { background: rgba(45,212,191,0.16); border: 1px solid var(--teal-border); }
.stage-completed .stage-icon svg { stroke: var(--teal); }
.stage-completed .stage-num { color: var(--teal); }
.stage-completed .stage-label { color: #ffffff; }

.stage-running {
    background: linear-gradient(180deg, rgba(251,191,36,0.14), rgba(15,23,32,0.5));
    border: 1px solid rgba(251,191,36,0.45);
    box-shadow: 0 0 26px rgba(251,191,36,0.14);
}
.stage-running .stage-icon { background: rgba(251,191,36,0.18); border: 1px solid rgba(251,191,36,0.5); }
.stage-running .stage-icon svg { stroke: var(--warn); }
.stage-running .stage-num { color: var(--warn); }
.stage-running .stage-label { color: #ffffff; }

.stage-pending {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    opacity: 0.55;
}
.stage-pending .stage-icon { background: rgba(255,255,255,0.04); border: 1px solid var(--border-subtle); }
.stage-pending .stage-icon svg { stroke: var(--text-muted); }
.stage-pending .stage-num { color: var(--text-muted); }
.stage-pending .stage-label { color: var(--text-secondary); }

.stage-failed {
    background: linear-gradient(180deg, rgba(248,113,113,0.14), rgba(15,23,32,0.5));
    border: 1px solid rgba(248,113,113,0.45);
    box-shadow: 0 0 26px rgba(248,113,113,0.12);
}
.stage-failed .stage-icon { background: rgba(248,113,113,0.18); border: 1px solid rgba(248,113,113,0.5); }
.stage-failed .stage-icon svg { stroke: var(--danger); }
.stage-failed .stage-num { color: var(--danger); }
.stage-failed .stage-label { color: #ffffff; }

.stage-skipped {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
}
.stage-skipped .stage-icon { background: rgba(255,255,255,0.04); border: 1px solid var(--border-strong); }
.stage-skipped .stage-icon svg { stroke: var(--text-secondary); }
.stage-skipped .stage-num { color: var(--text-secondary); }
.stage-skipped .stage-label { color: var(--text-secondary); }

/* ── Pills (status chips, e.g. sidebar / audit) ── */
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}
.pill-teal   { background: rgba(45,212,191,0.12); color: var(--teal); border: 1px solid var(--teal-border); }
.pill-muted  { background: rgba(148,163,184,0.10); color: var(--text-secondary); border: 1px solid var(--border-strong); }
.pill-danger { background: rgba(248,113,113,0.12); color: var(--danger); border: 1px solid rgba(248,113,113,0.4); }

/* ── Report container (unchanged layout, refreshed skin) ── */
.report-container {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border-radius: 12px;
    padding: 24px;
    line-height: 1.8;
    border: 1px solid var(--border-subtle);
    /* Prevent long tokens (prices, URLs, formulas) from blowing out the layout */
    word-wrap: break-word;
    overflow-wrap: break-word;
    overflow-x: hidden;
    white-space: normal;
}
.report-container h1,
.report-container h2,
.report-container h3 {
    color: var(--blue-link);
}
.report-container a {
    color: var(--blue-link);
}

/* ── Source cards ── */
.source-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px;
    margin: 6px 0;
    transition: border-color .15s ease;
}
.source-card:hover {
    border-color: var(--teal-border);
}
.source-card .src-title {
    font-weight: 700;
    color: var(--blue-link);
    font-size: 14.5px;
    line-height: 1.4;
}
.source-card .src-meta {
    font-size: 11.5px;
    color: var(--text-secondary);
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}
.source-card .src-snippet {
    font-size: 12.5px;
    color: #c3c9de;
    margin-top: 8px;
    line-height: 1.6;
}
.source-card .src-link {
    margin-top: 10px;
    display: block;
    font-size: 11.5px;
}
.source-card .src-link a {
    color: var(--blue-link);
    text-decoration: none;
}
.source-card .src-link a:hover {
    text-decoration: underline;
}
.badge-general {
    color: var(--text-secondary);
}
.badge-trusted {
    color: var(--teal);
}

/* ── Trace rows ── */
.trace-row {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 5px 0;
    border-bottom: 1px solid var(--border-subtle);
    color: #c3c9de;
}

/* ── Caption / small text ── */
[data-testid="stCaptionContainer"] p {
    color: var(--text-secondary) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: var(--bg-elevated-2) !important;
    color: #ffffff !important;
    border: 1px solid var(--teal-border) !important;
}

/* ── App header ── */
.app-header-title {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.2;
}
.app-header-sub {
    color: var(--text-secondary);
    font-size: 13.5px;
    margin-top: 4px;
}
.app-logo-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
}
.app-logo-word {
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.02em;
    color: var(--text-secondary);
}

/* ── Progress bar (teal, everywhere) ── */
[data-testid="stProgress"] [role="progressbar"] {
    background: var(--bg-elevated-2) !important;
}
[data-testid="stProgress"] [role="progressbar"] > div,
[data-testid="stProgress"] div[style*="background-color"],
[data-testid="stProgress"] div[style*="background"] {
    background-color: #2dd4bf !important;
    background: linear-gradient(90deg, #14b8a6, #2dd4bf) !important;
    box-shadow: 0 0 12px rgba(45,212,191,0.35);
}
[data-testid="stProgress"] > div > div {
    background: var(--bg-elevated-2) !important;
}
[data-testid="stProgress"] p {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

/* ── Buttons everywhere (main content, not just sidebar) ── */
.stButton button {
    background: var(--bg-elevated) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all .15s ease;
}
.stButton button:hover {
    background: var(--bg-elevated-2) !important;
    color: #ffffff !important;
    border: 1px solid var(--teal-border) !important;
}
.stButton button[kind="primary"] {
    background: var(--teal) !important;
    color: #052e2b !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 0 20px var(--teal-glow);
}
.stButton button[kind="primary"]:hover {
    background: #5eead4 !important;
    color: #052e2b !important;
}
.stButton button[kind="secondary"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-strong) !important;
}
.stButton button[kind="secondary"]:hover {
    color: #ffffff !important;
    border: 1px solid var(--border-strong) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] div {
    border-top-color: var(--teal) !important;
}

/* ── Toggle switch ── */
[data-testid="stSidebar"] [data-baseweb="toggle"] {
    background: var(--border-strong) !important;
}
[data-testid="stSidebar"] [aria-checked="true"][data-baseweb="toggle"] {
    background: var(--teal) !important;
}

/* ═══════════════════════════════════════════════════
   Evaluation dashboard
   ═══════════════════════════════════════════════════ */
.eval-section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 22px 0 10px 0;
}
.eval-section-label:first-child {
    margin-top: 4px;
}
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
}
.stat-tile {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.stat-tile .stat-icon {
    width: 34px;
    height: 34px;
    min-width: 34px;
    border-radius: 9px;
    background: var(--teal-soft);
    border: 1px solid var(--teal-border);
    display: flex;
    align-items: center;
    justify-content: center;
}
.stat-tile .stat-icon svg { stroke: var(--teal); }
.stat-tile.tile-warn .stat-icon { background: rgba(251,191,36,0.14); border: 1px solid rgba(251,191,36,0.4); }
.stat-tile.tile-warn .stat-icon svg { stroke: var(--warn); }
.stat-tile.tile-danger .stat-icon { background: rgba(248,113,113,0.14); border: 1px solid rgba(248,113,113,0.4); }
.stat-tile.tile-danger .stat-icon svg { stroke: var(--danger); }
.stat-tile .stat-value {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
}
.stat-tile .stat-label {
    font-size: 11.5px;
    color: var(--text-secondary);
    margin-top: 2px;
}

.meter-block { margin-bottom: 16px; }
.meter-head {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    color: var(--text-secondary);
    margin-bottom: 6px;
}
.meter-head b { color: #ffffff; font-size: 13px; }
.meter-track {
    width: 100%;
    height: 8px;
    border-radius: 999px;
    background: var(--bg-elevated-2);
    border: 1px solid var(--border-subtle);
    overflow: hidden;
}
.meter-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #14b8a6, #2dd4bf);
    box-shadow: 0 0 10px rgba(45,212,191,0.4);
}

.claims-bar {
    display: flex;
    width: 100%;
    height: 14px;
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
    margin: 6px 0 12px 0;
}
.claims-seg { height: 100%; }
.claims-seg.verified   { background: var(--teal); }
.claims-seg.unverified { background: var(--warn); }
.claims-seg.rejected   { background: var(--danger); }
.claims-legend {
    display: flex;
    gap: 18px;
    font-size: 12px;
    color: var(--text-secondary);
    flex-wrap: wrap;
}
.claims-legend .dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-verified   { background: var(--teal); }
.dot-unverified { background: var(--warn); }
.dot-rejected   { background: var(--danger); }

.gov-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def init_session_state() -> None:
    """Initialise all Streamlit session state keys."""
    defaults = {
        "workflow_state": None,
        "running": False,
        "run_complete": False,
        "approval_pending": False,
        "approved": None,
        "error": None,
        "progress": 0.0,
        "status_text": "Ready",
        "log_queue": queue.Queue(),
        "uploaded_kb": False,
        "last_run_id": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> Dict[str, Any]:
    """Render sidebar controls and return configuration dict."""
    with st.sidebar:
        st.markdown(
            '''<div class="app-logo-wrap">
                <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="14" cy="14" r="13" fill="rgba(45,212,191,0.16)" stroke="rgba(45,212,191,0.4)" stroke-width="1"/>
                    <path d="M4 15 L8 15 L11 7 L15 21 L18 11 L20 15 L24 15" stroke="#2dd4bf" stroke-width="2"
                    stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span style="font-size:17px;font-weight:800;color:#ffffff;">Intel Crew</span>
            </div>''',
            unsafe_allow_html=True,
        )
        st.caption("Competitive Intelligence")
        st.divider()

        # Topic input
        st.markdown('<div class="sb-eyebrow">Research Topic</div>', unsafe_allow_html=True)
        topic = st.text_area(
            "Enter topic",
            placeholder="e.g. Artificial Intelligence CRM market 2025",
            height=100,
            label_visibility="collapsed",
        )

        # Quick topic buttons
        st.markdown('<div class="sb-eyebrow">Sample Topics</div>', unsafe_allow_html=True)
        sample_topics = [
            "AI CRM software market 2025",
            "Electric Vehicles competitors",
            "Cloud Computing AWS Azure GCP",
            "Cybersecurity vendors 2025",
            "Healthcare AI diagnostics",
        ]
        for sample in sample_topics:
            if st.button(sample, key=f"sample_{sample[:15]}", use_container_width=True):
                st.session_state["_sample_topic"] = sample
                st.rerun()

        # Apply sample topic if set
        if "_sample_topic" in st.session_state:
            topic = st.session_state.pop("_sample_topic")

        st.divider()
        st.markdown('<div class="sb-eyebrow">Configuration</div>', unsafe_allow_html=True)

        max_sources = st.slider("Max Sources", min_value=3, max_value=20, value=8, step=1)
        max_steps = st.slider("Max Workflow Steps", min_value=5, max_value=30, value=20, step=5)

        rag_enabled = st.toggle("Enable RAG Knowledge Base", value=False)

        if rag_enabled:
            uploaded_files = st.file_uploader(
                "Upload documents (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt", "csv"],
                accept_multiple_files=True,
            )
            if uploaded_files and st.button("🗄️ Index Documents", use_container_width=True):
                _index_uploaded_documents(uploaded_files)
        else:
            uploaded_files = []

        st.divider()
        st.markdown('<div class="sb-eyebrow">API Status</div>', unsafe_allow_html=True)
        import os
        has_key = bool(
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GROQ_API_KEY")
        )
        if has_key:
            st.markdown('<span class="pill pill-teal">● LLM key configured</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill-danger">● No LLM key found</span>', unsafe_allow_html=True)
            st.caption("Set OPENROUTER_API_KEY in .env")

        # Tavily search status
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        tavily_active = bool(tavily_key) and len(tavily_key) >= 20 and not tavily_key.startswith("your_")
        if tavily_active:
            st.markdown('<span class="pill pill-teal">🔍 Tavily search active</span>', unsafe_allow_html=True)
            st.caption("Real-time web search enabled")
        else:
            st.markdown('<span class="pill pill-muted">🔍 Tavily not configured</span>', unsafe_allow_html=True)
            st.caption("Set TAVILY_API_KEY in .env for real-time data")

        st.divider()
        # Run button
        run_clicked = st.button(
            "🚀 Run Intelligence Briefing",
            type="primary",
            use_container_width=True,
            disabled=st.session_state["running"],
        )

    return {
        "topic": topic,
        "max_sources": max_sources,
        "max_steps": max_steps,
        "rag_enabled": rag_enabled,
        "run_clicked": run_clicked,
    }


def _index_uploaded_documents(files: list) -> None:
    """Index uploaded documents into the FAISS knowledge base."""
    from pathlib import Path
    import tempfile
    from rag.knowledge_base import build_knowledge_base

    with st.spinner("Indexing documents..."):
        tmp_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in files:
                dest = Path(tmpdir) / f.name
                dest.write_bytes(f.read())
                tmp_paths.append(dest)
            try:
                vs = build_knowledge_base(file_paths=tmp_paths, persist=True)
                if vs:
                    st.session_state["uploaded_kb"] = True
                    st.success(f"✅ Indexed {len(files)} document(s) into knowledge base")
                else:
                    st.warning("No content extracted from documents.")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")


# ── Agent status display ───────────────────────────────────────────────────────
# Minimal line-style SVG icons (stroke color is set via CSS per stage state)
_STAGE_ICONS = {
    "supervisor": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M12 1v4M12 19v4M4.2 4.2l2.9 2.9M16.9 16.9l2.9 2.9M1 12h4M19 12h4M4.2 19.8l2.9-2.9M16.9 7.1l2.9-2.9"></path></svg>',
    "research": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"></circle><path d="M21 21l-4.3-4.3"></path></svg>',
    "analyst": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"></path><rect x="7" y="12" width="3" height="6"></rect><rect x="12" y="8" width="3" height="10"></rect><rect x="17" y="5" width="3" height="13"></rect></svg>',
    "fact_verification": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="M9 12l2 2 4-4"></path></svg>',
    "writer": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"></path><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z"></path></svg>',
    "governance": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12c0-5 3-7 8-9 5 2 8 4 8 9 0 5-4 8-8 9-4-1-8-4-8-9z"></path><circle cx="12" cy="11" r="2.2"></circle></svg>',
}


def render_agent_status(workflow_state: Optional[Dict]) -> None:
    """Render agent pipeline as numbered, glowing stage cards."""
    agents = ["supervisor", "research", "analyst", "fact_verification", "writer", "governance"]
    agent_labels = {
        "supervisor": "Coordinate",
        "research": "Research",
        "analyst": "Analyze",
        "fact_verification": "Verify",
        "writer": "Write",
        "governance": "Govern",
    }

    tracker = None
    if workflow_state and "agent_status" in workflow_state:
        tracker = workflow_state["agent_status"]

    cards_html = ['<div class="stage-row">']
    for i, agent_key in enumerate(agents, start=1):
        status = "pending"
        if tracker and hasattr(tracker, agent_key):
            status = getattr(tracker, agent_key).value
        icon = _STAGE_ICONS.get(agent_key, "")
        label = agent_labels[agent_key]
        cards_html.append(
            f'<div class="stage-card stage-{status}">'
            f'<div class="stage-icon">{icon}</div>'
            f'<span class="stage-num">{i:02d}</span>'
            f'<span class="stage-label">{label}</span>'
            f'</div>'
        )
    cards_html.append("</div>")
    st.markdown("".join(cards_html), unsafe_allow_html=True)


# ── Progress bar & status ──────────────────────────────────────────────────────
def render_progress(state: Optional[Dict], running: bool) -> None:
    """Render the workflow progress bar."""
    if running:
        progress = st.session_state.get("progress", 0.0)
        st.progress(progress, text=st.session_state.get("status_text", "Running..."))
    elif state:
        phase = state.get("workflow_phase", "")
        if "completed" in phase:
            st.progress(1.0, text="✅ Workflow completed")
        else:
            st.progress(0.5, text=f"Phase: {phase}")


# ── Execution trace ────────────────────────────────────────────────────────────
def render_execution_trace(state: Dict) -> None:
    """Render the live execution trace."""
    trace = state.get("execution_trace", [])
    if not trace:
        st.info("No trace entries yet.")
        return

    st.markdown(f"**{len(trace)} trace entries**")
    rows_html = []
    for entry in reversed(trace[-50:]):  # Show latest 50
        ts = entry.timestamp[:19] if hasattr(entry, "timestamp") else ""
        agent = entry.agent if hasattr(entry, "agent") else ""
        phase = entry.phase if hasattr(entry, "phase") else ""
        msg = entry.message if hasattr(entry, "message") else ""
        dur = f"{entry.duration_ms:.0f}ms" if (hasattr(entry, "duration_ms") and entry.duration_ms) else ""
        rows_html.append(
            f'<div class="trace-row">'
            f'<span style="color:#545c72">{ts}</span> '
            f'<span style="color:#2dd4bf;font-weight:600">[{agent}]</span> '
            f'<span style="color:#7dd3fc">{phase}</span> '
            f'<span>{msg[:100]}</span> '
            f'<span style="color:#545c72">{dur}</span>'
            f'</div>'
        )
    st.markdown("\n".join(rows_html), unsafe_allow_html=True)


# ── Report viewer ──────────────────────────────────────────────────────────────
def _section_to_html(text: str) -> str:
    """
    Convert a report section (plain markdown prose) to HTML for safe rendering
    inside a report-container div.

    Converts via the `markdown` library when available; falls back to a minimal
    regex path. Either way, dollar signs are escaped BEFORE conversion so
    Streamlit never interprets $...$ sequences as LaTeX math mode (which strips
    spaces and produces the horizontal-overflow italic runs seen in the UI).
    """
    # Escape $ so Streamlit/MathJax never treats it as LaTeX delimiter.
    # Use the HTML entity &#36; — invisible to the reader, safe in HTML.
    safe_text = text.replace("$", "&#36;")

    try:
        import markdown as _md
        return _md.markdown(
            safe_text,
            extensions=["tables", "fenced_code", "nl2br"],
        )
    except ImportError:
        pass

    # Minimal regex fallback
    import re as _re
    t = safe_text
    t = _re.sub(r"^## (.+)$",  r"<h2>\1</h2>",  t, flags=_re.M)
    t = _re.sub(r"^### (.+)$", r"<h3>\1</h3>",  t, flags=_re.M)
    t = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = _re.sub(r"_(.+?)_",    r"<em>\1</em>",   t)
    t = _re.sub(r"^[-*] (.+)$", r"<li>\1</li>",  t, flags=_re.M)
    t = _re.sub(r"(<li>.*?</li>\n?)+", r"<ul>\g<0></ul>", t, flags=_re.S)
    t = _re.sub(r"^(?!<)(.+)$", r"<p>\1</p>",    t, flags=_re.M)
    return t


def render_report(state: Dict) -> None:
    """Render the final report in tabbed sections."""
    report = state.get("final_report")
    if not report:
        st.warning("No report generated yet.")
        return

    # Metrics header
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Word Count", f"{report.word_count:,}")
    c2.metric("Citations", str(len(report.references)))
    c3.metric("Confidence", f"{report.overall_confidence:.0%}")
    c4.metric("Coverage", f"{report.citation_coverage:.0%}")

    # Section tabs
    tabs = st.tabs([
        "📋 Executive Summary",
        "💰 Pricing",
        "🚀 Products",
        "📈 Market Signals",
        "⚠️ Risks",
        "💡 Recommendations",
        "🌟 Opportunities",
        "📄 Full Report",
    ])

    with tabs[0]:
        st.markdown(f'<div class="report-container">{_section_to_html(report.executive_summary or "")}</div>', unsafe_allow_html=True)
    with tabs[1]:
        st.markdown(f'<div class="report-container">{_section_to_html(report.competitor_pricing or "")}</div>', unsafe_allow_html=True)
    with tabs[2]:
        st.markdown(f'<div class="report-container">{_section_to_html(report.product_updates or "")}</div>', unsafe_allow_html=True)
    with tabs[3]:
        st.markdown(f'<div class="report-container">{_section_to_html(report.market_signals or "")}</div>', unsafe_allow_html=True)
    with tabs[4]:
        st.markdown(f'<div class="report-container">{_section_to_html(report.business_risks or "")}</div>', unsafe_allow_html=True)
    with tabs[5]:
        st.markdown(f'<div class="report-container">{_section_to_html(report.strategic_recommendations or "")}</div>', unsafe_allow_html=True)
    with tabs[6]:
        opportunities = getattr(report, "opportunities", "") or "_No opportunities section generated._"
        st.markdown(f'<div class="report-container">{_section_to_html(opportunities)}</div>', unsafe_allow_html=True)
    with tabs[7]:
        # Full report tab: escape $ for display, use st.markdown with raw=False
        st.markdown(report.markdown_content.replace("$", r"\$"), unsafe_allow_html=False)


# ── Source cards ───────────────────────────────────────────────────────────────
def render_source_cards(state: Dict) -> None:
    """Render source cards showing all collected research sources."""
    research = state.get("research_result")
    if not research or not research.sources:
        st.info("No sources collected.")
        return

    sources = research.sources
    st.markdown(f"**{len(sources)} source(s) collected**")

    _globe_icon = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-1px"><circle cx="12" cy="12" r="10"></circle><path d="M2 12h20M12 2a15 15 0 0 1 0 20 15 15 0 0 1 0-20z"></path></svg>'
    _link_icon = '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-1px"><path d="M10 13a5 5 0 0 0 7.5.5l2-2a5 5 0 0 0-7-7l-1.5 1.5"></path><path d="M14 11a5 5 0 0 0-7.5-.5l-2 2a5 5 0 0 0 7 7l1.5-1.5"></path></svg>'

    cols = st.columns(2)
    for i, src in enumerate(sources):
        with cols[i % 2]:
            is_trusted = getattr(src, "is_trusted_domain", False)
            badge_class = "badge-trusted" if is_trusted else "badge-general"
            badge_text = "Trusted" if is_trusted else "General"
            relevance_pct = f"{src.relevance_score:.0%}" if src.relevance_score else "N/A"
            date_str = f" · {src.publication_date}" if src.publication_date else ""
            st.markdown(
                f"""<div class="source-card">
                <div class="src-title">{src.title[:70]}</div>
                <div class="src-meta">
                    <span>{src.domain or 'unknown'}{date_str}</span>
                    <span>·</span>
                    <span class="{badge_class}">{_globe_icon} {badge_text}</span>
                    <span>·</span>
                    <span>Relevance: {relevance_pct}</span>
                </div>
                <div class="src-snippet">{(src.summary or '')[:160]}</div>
                <div class="src-link"><a href="{src.url}" target="_blank">{_link_icon} {src.url[:60]}...</a></div>
                </div>""",
                unsafe_allow_html=True,
            )


# ── Evaluation dashboard ───────────────────────────────────────────────────────
_EVAL_ICONS = {
    "steps":    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"></path></svg>',
    "sources":  '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M3 12h18M12 3a13 13 0 0 1 0 18 13 13 0 0 1 0-18z"></path></svg>',
    "queries":  '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"></circle><path d="M21 21l-4.3-4.3"></path></svg>',
    "tools":    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a4 4 0 0 1-5.4 5.4L4 17l3 3 5.3-5.3a4 4 0 0 1 5.4-5.4L21 6l-3-3-3.3 3.3z"></path></svg>',
    "clock":    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 7v5l3 3"></path></svg>',
    "total":    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 12h6M9 16h6M9 8h6"></path><rect x="4" y="3" width="16" height="18" rx="2"></rect></svg>',
    "check":    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"></path></svg>',
    "question": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.1 9a3 3 0 1 1 5.2 2c-.6.6-1.3 1-1.3 2"></path><circle cx="12" cy="17" r="0.4" fill="currentColor"></circle><circle cx="12" cy="12" r="9"></circle></svg>',
    "x":        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"></path></svg>',
    "articles": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h12v16l-6-3-6 3z"></path><path d="M18 8h2v14l-2-1"></path></svg>',
}


def _stat_tile(icon_key: str, value: Any, label: str, variant: str = "") -> str:
    cls = f"stat-tile {variant}".strip()
    icon = _EVAL_ICONS.get(icon_key, "")
    return (
        f'<div class="{cls}"><div class="stat-icon">{icon}</div>'
        f'<div><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div></div>'
    )


def render_evaluation_dashboard(state: Dict) -> None:
    """Render evaluation metrics dashboard."""
    report = state.get("final_report")
    verification = state.get("verification_result")
    research = state.get("research_result")
    meta = state.get("run_metadata")
    gov = state.get("governance_result")

    # ── Run overview ─────────────────────────────────────────────────────────
    st.markdown('<div class="eval-section-label">Run Overview</div>', unsafe_allow_html=True)
    duration = f"{meta.duration_seconds:.1f}s" if meta and meta.duration_seconds else "–"
    tiles = [
        _stat_tile("steps", meta.total_steps if meta else "–", "Workflow Steps"),
        _stat_tile("sources", meta.total_sources if meta else "–", "Sources Collected"),
        _stat_tile("queries", meta.search_queries if meta else "–", "Search Queries"),
        _stat_tile("tools", meta.tool_calls if meta else "–", "Tool Calls"),
        _stat_tile("clock", duration, "Execution Time"),
    ]
    st.markdown(f'<div class="stat-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)

    # ── Claims & verification ───────────────────────────────────────────────
    st.markdown('<div class="eval-section-label">Claims &amp; Verification</div>', unsafe_allow_html=True)
    total_claims = meta.total_claims if meta else 0
    verified = meta.verified_claims if meta else 0
    rejected = meta.rejected_claims if meta else 0
    unverified = total_claims - verified - rejected if total_claims else 0

    claim_tiles = [
        _stat_tile("total", total_claims, "Total Claims"),
        _stat_tile("check", verified, "Verified"),
        _stat_tile("question", unverified, "Unverified", variant="tile-warn"),
        _stat_tile("x", rejected, "Rejected", variant="tile-danger"),
    ]
    st.markdown(f'<div class="stat-grid">{"".join(claim_tiles)}</div>', unsafe_allow_html=True)

    if total_claims:
        v_pct = verified / total_claims * 100
        u_pct = unverified / total_claims * 100
        r_pct = rejected / total_claims * 100
        st.markdown(
            f"""<div class="claims-bar">
                <div class="claims-seg verified" style="width:{v_pct}%"></div>
                <div class="claims-seg unverified" style="width:{u_pct}%"></div>
                <div class="claims-seg rejected" style="width:{r_pct}%"></div>
            </div>
            <div class="claims-legend">
                <span><span class="dot dot-verified"></span>Verified ({verified})</span>
                <span><span class="dot dot-unverified"></span>Unverified ({unverified})</span>
                <span><span class="dot dot-rejected"></span>Rejected ({rejected})</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Quality meters ───────────────────────────────────────────────────────
    st.markdown('<div class="eval-section-label">Quality Meters</div>', unsafe_allow_html=True)
    coverage_pct = verification.citation_coverage * 100 if verification else 0
    confidence_pct = report.overall_confidence * 100 if report else 0
    st.markdown(
        f"""<div class="meter-block">
            <div class="meter-head"><span>Citation Coverage</span><b>{coverage_pct:.0f}%</b></div>
            <div class="meter-track"><div class="meter-fill" style="width:{coverage_pct}%"></div></div>
        </div>
        <div class="meter-block">
            <div class="meter-head"><span>Confidence Score</span><b>{confidence_pct:.0f}%</b></div>
            <div class="meter-track"><div class="meter-fill" style="width:{confidence_pct}%"></div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Governance ────────────────────────────────────────────────────────────
    if gov:
        st.markdown('<div class="eval-section-label">Governance</div>', unsafe_allow_html=True)
        passed = not gov.refusal_triggered
        pill_cls = "pill-teal" if passed else "pill-danger"
        pill_text = "✓ Passed" if passed else "✕ Refused"
        articles_parsed = len([s for s in research.sources if s.raw_content]) if research else "–"
        st.markdown(
            f"""<div class="gov-card">
                <div><div class="stat-label">Governance Status</div>
                    <span class="pill {pill_cls}" style="margin-top:4px;display:inline-block">{pill_text}</span></div>
                <div><div class="stat-label">Governance Confidence</div>
                    <div class="stat-value" style="font-size:17px">{gov.confidence_score:.0%}</div></div>
                <div><div class="stat-label">Articles Parsed</div>
                    <div class="stat-value" style="font-size:17px">{articles_parsed}</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Errors count
    errors = state.get("errors", [])
    if errors:
        st.warning(f"⚠️ {len(errors)} error(s) recorded during execution.")


# ── Citation viewer ────────────────────────────────────────────────────────────
def render_citations(state: Dict) -> None:
    """Render the citations table."""
    report = state.get("final_report")
    if not report or not report.references:
        st.info("No citations available.")
        return

    st.markdown(f"**{len(report.references)} source(s) cited**")
    for cit in report.references:
        with st.expander(f"[{cit.number}] {cit.title[:70]}", expanded=False):
            st.markdown(f"**URL:** [{cit.url}]({cit.url})")
            st.markdown(f"**Domain:** `{cit.domain or 'unknown'}`")
            st.markdown(f"**Accessed:** {cit.accessed_date}")


# ── Download buttons ───────────────────────────────────────────────────────────
def render_downloads(state: Dict) -> None:
    """Render Markdown and PDF download buttons."""
    report = state.get("final_report")
    if not report:
        return

    topic = report.topic
    run_id = state.get("run_id", "")
    md_content = report.markdown_content
    # Use the lean pdf_markdown (no Run Metadata / Audit pages) for PDF export.
    # Fall back to full markdown_content if pdf_markdown wasn't populated (old reports).
    pdf_content = getattr(report, "pdf_markdown", None) or md_content

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="⬇️ Download Markdown",
            data=md_content.encode("utf-8"),
            file_name=f"intel_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col2:
        if st.button("⬇️ Download PDF", use_container_width=True):
            with st.spinner("Generating PDF..."):
                try:
                    export_md, export_pdf = _import_exporters()
                    pdf_path = export_pdf(pdf_content, topic, run_id=run_id)
                    if pdf_path and Path(pdf_path).exists():
                        pdf_bytes = Path(pdf_path).read_bytes()
                        st.download_button(
                            label="📥 Click to save PDF",
                            data=pdf_bytes,
                            file_name=Path(pdf_path).name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    else:
                        st.warning("PDF generation failed – download Markdown instead.")
                except Exception as exc:
                    st.error(f"PDF export error: {exc}")


# ── Audit log viewer ───────────────────────────────────────────────────────────
def render_audit_log(state: Dict) -> None:
    """Render the audit log table."""
    audit_log = state.get("audit_log", [])
    if not audit_log:
        st.info("No audit entries.")
        return

    st.markdown(f"**{len(audit_log)} audit entries**")

    sev_styles = {
        "info":    ("rgba(45,212,191,0.08)",  "#1d2330", "#8b93ab"),
        "warning": ("rgba(251,191,36,0.08)",  "rgba(251,191,36,0.3)", "#fbbf24"),
        "error":   ("rgba(248,113,113,0.08)", "rgba(248,113,113,0.3)", "#f87171"),
    }
    for entry in audit_log:
        sev = entry.severity if hasattr(entry, "severity") else "info"
        bg, border, accent = sev_styles.get(sev, sev_styles["info"])
        ts = entry.timestamp[:19] if hasattr(entry, "timestamp") else ""
        agent = entry.agent if hasattr(entry, "agent") else ""
        ev = entry.event_type if hasattr(entry, "event_type") else ""
        desc = entry.description if hasattr(entry, "description") else ""
        st.markdown(
            f'<div style="background:{bg};border:1px solid {border};padding:8px 12px;'
            f'border-radius:8px;margin:4px 0;font-size:13px;color:#c3c9de;font-family:\'JetBrains Mono\',monospace;">'
            f'<span style="color:#545c72">{ts}</span> &nbsp;'
            f'<span style="color:{accent};font-weight:600">{agent}</span> &nbsp;'
            f'<span style="color:#8b93ab">{ev}</span> &nbsp; {desc}</div>',
            unsafe_allow_html=True,
        )


# ── Human approval gate ────────────────────────────────────────────────────────
def render_approval_gate(state: Dict) -> Optional[bool]:
    """Render the human approval UI. Returns True/False/None."""
    report = state.get("final_report")
    if not report:
        return None

    st.markdown("---")
    st.markdown("## 👤 Human Review & Approval")
    st.info(
        f"The report is ready for review. "
        f"Confidence: **{report.overall_confidence:.0%}** | "
        f"Citations: **{len(report.references)}** | "
        f"Words: **{report.word_count:,}**"
    )

    # Show governance issues if any
    gov = state.get("governance_result")
    if gov and gov.issues:
        with st.expander("⚠️ Governance Issues", expanded=True):
            for issue in gov.issues:
                st.warning(issue)
    if gov and gov.warnings:
        with st.expander("ℹ️ Governance Warnings"):
            for w in gov.warnings:
                st.info(w)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Report", type="primary", use_container_width=True):
            return True
    with col2:
        if st.button("🔄 Request Revision", type="secondary", use_container_width=True):
            return False
    return None


# ── Workflow runner (background thread) ───────────────────────────────────────
def _run_workflow_thread(
    topic: str,
    max_sources: int,
    max_steps: int,
    rag_enabled: bool,
    result_holder: dict,
    progress_queue: queue.Queue,
) -> None:
    """Execute the workflow in a background thread."""
    try:
        run_workflow = _import_workflow()
        final_state = run_workflow(
            topic=topic,
            max_steps=max_steps,
            max_sources=max_sources,
            rag_enabled=rag_enabled,
            human_approved=None,  # UI handles approval
        )
        result_holder["state"] = final_state
        result_holder["error"] = None
    except Exception as exc:
        result_holder["state"] = None
        result_holder["error"] = str(exc)
    finally:
        progress_queue.put("done")


# ── Main application ───────────────────────────────────────────────────────────
def main() -> None:
    """Main Streamlit application entry point."""

    # ── Header ────────────────────────────────────────────────────────────────
    state_for_header = st.session_state.get("workflow_state")
    meta_for_header = state_for_header.get("run_metadata") if state_for_header else None
    if meta_for_header:
        header_meta = f"{meta_for_header.total_sources} sources · multi-agent crew"
    else:
        header_meta = "Powered by LangGraph · Multi-Agent AI · Automated Competitive Research"

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:2px;">
            <svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="19" cy="19" r="18" fill="rgba(45,212,191,0.16)" stroke="rgba(45,212,191,0.4)" stroke-width="1.5"/>
                <path d="M6 20 L11 20 L14 10 L18 29 L22 14 L25 20 L32 20" stroke="#2dd4bf" stroke-width="2.2"
                stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div>
                <div class="app-header-title">Competitive Intelligence Briefing Crew</div>
                <div class="app-header-sub">{header_meta}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    config = render_sidebar()

    # ── Trigger workflow run ──────────────────────────────────────────────────
    if config["run_clicked"] and config["topic"].strip():
        topic = config["topic"].strip()
        st.session_state["running"] = True
        st.session_state["run_complete"] = False
        st.session_state["approval_pending"] = False
        st.session_state["approved"] = None
        st.session_state["workflow_state"] = None
        st.session_state["error"] = None
        st.session_state["progress"] = 0.05
        st.session_state["status_text"] = "Starting workflow..."

        # Run synchronously (Streamlit doesn't support true async here)
        with st.spinner(f"🔄 Running intelligence briefing for: **{topic}**"):
            progress_bar = st.progress(0.05, text="Initialising agents...")
            status_placeholder = st.empty()

            phase_progress = {
                "initializing": 0.05,
                "researching": 0.20,
                "analyzing": 0.45,
                "verifying": 0.65,
                "writing": 0.80,
                "governance": 0.92,
                "awaiting_approval": 0.97,
                "completed": 1.0,
            }

            try:
                run_workflow = _import_workflow()
                from agents.state import WorkflowPhase

                # Use streaming to show progress
                from graph.workflow import stream_workflow

                final_state = None
                for snapshot in stream_workflow(
                    topic=topic,
                    max_steps=config["max_steps"],
                    max_sources=config["max_sources"],
                    rag_enabled=config["rag_enabled"],
                    human_approved=None,
                ):
                    # snapshot is a dict of {node_name: state}
                    for node_name, node_state in snapshot.items():
                        phase = node_state.get("workflow_phase", "initializing")
                        prog = phase_progress.get(phase, 0.5)
                        progress_bar.progress(prog, text=f"⚙️ Agent: {node_name} | Phase: {phase}")
                        final_state = node_state

                if final_state is None:
                    final_state = run_workflow(
                        topic=topic,
                        max_steps=config["max_steps"],
                        max_sources=config["max_sources"],
                        rag_enabled=config["rag_enabled"],
                        human_approved=None,
                    )

                progress_bar.progress(0.97, text="🛡️ Governance check complete – awaiting approval")
                st.session_state["workflow_state"] = final_state
                st.session_state["running"] = False
                st.session_state["approval_pending"] = True
                st.session_state["last_run_id"] = final_state.get("run_id", "")

            except Exception as exc:
                st.session_state["error"] = str(exc)
                st.session_state["running"] = False
                progress_bar.progress(1.0, text="❌ Error")

    elif config["run_clicked"] and not config["topic"].strip():
        st.error("Please enter a research topic before running.")

    # ── Display error if any ──────────────────────────────────────────────────
    if st.session_state.get("error"):
        st.error(f"❌ Workflow Error: {st.session_state['error']}")
        with st.expander("Error Details"):
            st.code(st.session_state["error"])

    # ── Main content area ──────────────────────────────────────────────────────
    state = st.session_state.get("workflow_state")

    if state is None and not st.session_state.get("running"):
        # Welcome / instructions
        st.markdown("### 👋 Welcome")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**What this does:**
- 🔎 Researches your topic across the web
- 📊 Extracts competitor intelligence
- ✅ Verifies all claims with sources
- ✍️ Writes a professional report
""")
        with col2:
            st.markdown("""
**How to use:**
1. Enter a topic in the sidebar
2. Adjust max sources / steps
3. Click **Run Intelligence Briefing**
4. Review and approve the report
5. Download as Markdown or PDF
""")
        with col3:
            st.markdown("""
**Sample topics:**
- AI CRM market 2025
- Electric vehicle competitors
- Cloud computing AWS/Azure/GCP
- Cybersecurity vendors
- Healthcare AI diagnostics
""")
        return

    if state:
        # ── Progress ──────────────────────────────────────────────────────────
        render_progress(state, st.session_state.get("running", False))

        # ── Agent status ──────────────────────────────────────────────────────
        st.markdown("### 🤖 Agent Pipeline")
        render_agent_status(state)

        # ── Run metrics ───────────────────────────────────────────────────────
        meta = state.get("run_metadata")
        if meta:
            st.markdown("### 📊 Run Statistics")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Steps", meta.total_steps)
            c2.metric("Sources", meta.total_sources)
            c3.metric("Claims", meta.total_claims)
            c4.metric("Verified", meta.verified_claims)
            c5.metric("Errors", meta.errors)

        # ── Human approval gate ───────────────────────────────────────────────
        if st.session_state.get("approval_pending") and not st.session_state.get("approved"):
            decision = render_approval_gate(state)
            if decision is True:
                st.session_state["approved"] = True
                st.session_state["approval_pending"] = False
                state["human_approved"] = True
                # Mark report as published
                report = state.get("final_report")
                if report:
                    report.run_metadata["published"] = True
                    report.run_metadata["approved_at"] = datetime.utcnow().isoformat()
                st.session_state["workflow_state"] = state
                st.session_state["run_complete"] = True
                st.rerun()
            elif decision is False:
                st.session_state["approval_pending"] = False
                st.warning("🔄 Revision requested. Please re-run with a refined topic or settings.")
                st.session_state["workflow_state"] = None

        # ── Tabs for report, trace, citations, audit ──────────────────────────
        if st.session_state.get("approved") or st.session_state.get("run_complete"):
            report = state.get("final_report")
            if report:
                st.success("✅ Report approved and ready for download")

            main_tabs = st.tabs([
                "📄 Report",
                "🔗 Citations",
                "🗂️ Sources",
                "⬇️ Download",
                "📊 Evaluation",
                "🔍 Execution Trace",
                "📋 Audit Log",
                "❌ Errors",
            ])

            with main_tabs[0]:
                render_report(state)

            with main_tabs[1]:
                render_citations(state)

            with main_tabs[2]:
                render_source_cards(state)

            with main_tabs[3]:
                render_downloads(state)

            with main_tabs[4]:
                render_evaluation_dashboard(state)

            with main_tabs[5]:
                render_execution_trace(state)

            with main_tabs[6]:
                render_audit_log(state)

            with main_tabs[7]:
                errors = state.get("errors", [])
                if not errors:
                    st.success("No errors recorded.")
                else:
                    st.markdown(f"**{len(errors)} error(s)**")
                    for e in errors:
                        st.error(e)

        elif st.session_state.get("approval_pending"):
            # Show report preview while awaiting approval
            with st.expander("👁️ Report Preview", expanded=True):
                render_report(state)

            main_tabs = st.tabs(["🔍 Execution Trace", "📋 Audit Log"])
            with main_tabs[0]:
                render_execution_trace(state)
            with main_tabs[1]:
                render_audit_log(state)

        else:
            # In-progress or failed state — show what went wrong
            errors = state.get("errors", [])
            report = state.get("final_report")
            research = state.get("research_result")

            if not report:
                st.error("❌ Workflow completed but no report was generated.")

                if not research or not getattr(research, "sources", None):
                    st.warning(
                        "🔎 Research returned 0 sources. Possible causes:\n"
                        "- DuckDuckGo rate-limited your IP (wait 1–2 minutes and retry)\n"
                        "- The LLM API key is invalid or over quota\n"
                        "- Network issue reaching search providers\n\n"
                        "**Tip:** Add a free Tavily API key at https://tavily.com and set "
                        "`TAVILY_API_KEY` in your `.env` for more reliable search results."
                    )
                elif errors:
                    st.warning(f"Research found {len(research.sources)} sources but the workflow failed during processing.")

            if errors:
                with st.expander("❌ Errors", expanded=True):
                    for e in errors:
                        st.error(e)

            with st.expander("🔍 Execution Trace", expanded=True):
                render_execution_trace(state)


if __name__ == "__main__":
    main()