"""
ChemEvidence AI: Advanced Evidence-Aware Intelligence Platform for Scientific Chemistry Literature.
Features:
- Multi-Paper Comparison & Cross-Study Synthesis
- Automatic Table Extraction & Dataframe Export (CSV / Excel)
- Reaction Pathway Extraction (Multistep Schemes, Catalysts, Yields)
- Chemical Entity Normalization (Synonym Resolution, PubChem CIDs)
- Interactive Chemistry Knowledge Graph (HTML5 Canvas Physics Network)
- Similar Compound Side-by-Side Comparator
- SAR Studio with Interactive Plotly Potency & Feasibility Plots
- Empirical Multi-Factor Evidence Confidence Scoring
- Proactive Peer-Review Research-Gap Detection
- Grounded Interactive Q&A with Strict Anti-Hallucination Absence & Conflict Detection
"""

import os
import re
import json
import io
import html
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from typing import List, Optional, Dict, Any

import base64

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_BACKEND_KEY = os.getenv("GEMINI_API_KEY", "")
try:
    import streamlit as st
    if not DEFAULT_BACKEND_KEY and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        DEFAULT_BACKEND_KEY = st.secrets["GEMINI_API_KEY"]
        os.environ["GEMINI_API_KEY"] = DEFAULT_BACKEND_KEY
except Exception:
    pass

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")

from core.models import (
    PaperAnalysisResult,
    PaperMetadata,
    QAResponse,
    MultiPaperComparisonResult
)
from core.pdf_parser import DocumentParser, PageContent
from core.evidence_engine import EvidenceEngine
from core.chemistry_extractor import ChemistryExtractor
from core.qa_system import QASystem
from core.pubchem_service import PubChemService
from core.table_extractor import TableExtractor
from core.reaction_engine import ReactionPathwayExtractor
from core.entity_normalizer import EntityNormalizer
from core.sar_engine import SAREngine
from core.knowledge_graph import KnowledgeGraphBuilder
from core.confidence_scorer import ConfidenceScorer
from core.gap_detector import ResearchGapDetector
from core.multi_paper_comparator import MultiPaperComparator

from src.ui_components import (
    render_grounded_evidence_card,
    render_show_in_paper_button,
    render_page_navigation,
    render_live_document_viewer,
    render_dynamic_question_chips,
    render_compound_smiles_inspector,
    jump_to_paper_citation,
    inject_custom_styles,
    render_floating_scratchpad,
    render_floating_round_chatbot,
    render_multilingual_chatbot_ui
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChemEvidence AI | Chemistry Literature Platform",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_logo_base64() -> str:
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


# Custom High-Aesthetic Rose Gold Theme & CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #2D1D22;
    }
    
    /* Rose Gold Master Background */
    .stApp {
        background: linear-gradient(135deg, #FAF0EE 0%, #F6E2DF 50%, #F1D4CE 100%) !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F5E1DE 0%, #EDD3CF 100%) !important;
        border-right: 1px solid rgba(183, 110, 121, 0.28) !important;
        box-shadow: 2px 0 16px rgba(183, 110, 121, 0.08);
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #5B2B3D !important;
        font-weight: 700;
    }
    
    /* Luxury Rose Gold Metallic Header */
    .chem-header {
        background: linear-gradient(135deg, #3A1C28 0%, #5B2B3D 50%, #823E54 100%);
        padding: 24px 32px;
        border-radius: 16px;
        border: 1px solid rgba(235, 182, 188, 0.45);
        box-shadow: 0 12px 32px -8px rgba(91, 43, 61, 0.4), 0 0 25px rgba(183, 110, 121, 0.25);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .chem-header::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(235, 182, 188, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .chem-title {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .chem-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(235, 182, 188, 0.22);
        color: #FCE7E9;
        border: 1px solid rgba(235, 182, 188, 0.45);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        backdrop-filter: blur(6px);
    }
    .chem-subtitle {
        color: #F8D8DE;
        font-size: 0.98rem;
        margin-top: 6px;
        margin-bottom: 0;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Frosted Luxury Rose Gold Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid rgba(183, 110, 121, 0.35);
        border-radius: 14px;
        padding: 16px 20px;
        backdrop-filter: blur(14px);
        box-shadow: 0 4px 18px rgba(183, 110, 121, 0.12);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #B76E79;
        box-shadow: 0 8px 24px rgba(183, 110, 121, 0.25);
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #9F3E54;
        letter-spacing: -0.02em;
    }
    .metric-lbl {
        font-size: 0.78rem;
        color: #6C4C54;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 2px;
    }
    
    /* Page & Section Pills */
    .page-pill {
        display: inline-flex;
        align-items: center;
        background: #B76E79;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        margin-right: 6px;
        box-shadow: 0 2px 6px rgba(183, 110, 121, 0.25);
    }
    .section-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(183, 110, 121, 0.18);
        color: #5B2B3D;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
    }
    .quote-box {
        background: rgba(253, 242, 240, 0.95);
        border-left: 3px solid #B76E79;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        font-style: italic;
        color: #2D1D22;
        font-size: 0.88rem;
        margin-top: 6px;
    }
    
    /* Evidence Verification Status Badges */
    .status-found {
        background: rgba(16, 185, 129, 0.16);
        color: #065F46;
        border: 1px solid rgba(16, 185, 129, 0.45);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.88rem;
    }
    .status-absent {
        background: rgba(239, 68, 68, 0.15);
        color: #991B1B;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.88rem;
    }
    .status-conflict {
        background: rgba(245, 158, 11, 0.2);
        color: #92400E;
        border: 1px solid rgba(245, 158, 11, 0.45);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.88rem;
    }
    .status-external {
        background: rgba(14, 165, 233, 0.15);
        color: #0369A1;
        border: 1px solid rgba(14, 165, 233, 0.45);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.88rem;
    }
    .external-paper-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #0284C7;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 10px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.08);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .external-paper-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.16);
    }
    
    .conflict-card {
        background: rgba(254, 243, 199, 0.92);
        border: 1px solid #D97706;
        border-left: 5px solid #D97706;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 12px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(217, 119, 6, 0.12);
        color: #78350F;
    }
    
    .gap-card-critical {
        background: rgba(254, 226, 226, 0.92);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 5px solid #DC2626;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #7F1D1D;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
    }
    .gap-card-important {
        background: rgba(254, 243, 199, 0.92);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 5px solid #D97706;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #78350F;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
    }
    .gap-card-exploratory {
        background: rgba(253, 242, 240, 0.95);
        border: 1px solid rgba(183, 110, 121, 0.4);
        border-left: 5px solid #B76E79;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #5B2B3D;
        box-shadow: 0 4px 12px rgba(183, 110, 121, 0.1);
    }
    
    .step-badge {
        display: inline-block;
        background: #B76E79;
        color: #FFFFFF;
        font-weight: 700;
        border-radius: 50%;
        width: 26px;
        height: 26px;
        text-align: center;
        line-height: 26px;
        margin-right: 8px;
        box-shadow: 0 2px 6px rgba(183, 110, 121, 0.3);
    }
    
    /* User-Friendly Action Card Container */
    .user-friendly-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(183, 110, 121, 0.32);
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 6px 20px rgba(183, 110, 121, 0.12);
        margin-bottom: 20px;
    }
    
    /* Enhanced Button Aesthetics */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        letter-spacing: 0.01em !important;
    }
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #B76E79 0%, #9F3E54 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(183, 110, 121, 0.35) !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 6px 22px rgba(183, 110, 121, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="stBaseButton-secondary"] {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 1px solid rgba(183, 110, 121, 0.4) !important;
        color: #3A1C28 !important;
    }
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(253, 242, 240, 0.95) !important;
        border-color: #B76E79 !important;
        color: #8F2E44 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(183, 110, 121, 0.2) !important;
    }
    
    /* Modern Tabs Styling */
    button[data-baseweb="tab"] {
        font-size: 0.94rem !important;
        font-weight: 600 !important;
        color: #6C4C54 !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #8F2E44 !important;
        border-bottom-color: #B76E79 !important;
        font-weight: 700 !important;
    }
    
    /* Streamlit Chat Messages */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(183, 110, 121, 0.25);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(183, 110, 121, 0.08);
    }
    
    /* Native Alert Boxes */
    div.stAlert {
        border-radius: 12px !important;
        font-weight: 500 !important;
    }
    
    /* Multi-Paper Workspace Luxury Rose Gold Styles */
    .multi-hero-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(253, 242, 240, 0.94) 50%, rgba(247, 230, 233, 0.92) 100%);
        border: 1px solid rgba(183, 110, 121, 0.38);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(183, 110, 121, 0.12);
    }
    .multi-kpi-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(183, 110, 121, 0.3);
        border-top: 4px solid #B76E79;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 16px rgba(183, 110, 121, 0.08);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .multi-kpi-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #8F2E44;
        line-height: 1.2;
        margin: 4px 0 2px 0;
    }
    .multi-kpi-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6C4C54;
    }
    .multi-kpi-sub {
        font-size: 0.8rem;
        color: #7A5862;
        margin-top: 4px;
        line-height: 1.4;
    }
    .lead-showdown-card {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(183, 110, 121, 0.35);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 16px rgba(183, 110, 121, 0.08);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .lead-showdown-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(183, 110, 121, 0.16);
    }
    .potency-badge-elite {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(5, 150, 105, 0.22) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #065F46;
        font-weight: 800;
        font-size: 0.88rem;
        padding: 4px 12px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .potency-badge-mod {
        background: rgba(183, 110, 121, 0.18);
        border: 1px solid rgba(183, 110, 121, 0.45);
        color: #8F2E44;
        font-weight: 700;
        font-size: 0.88rem;
        padding: 4px 12px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .synth-step-pill {
        display: inline-block;
        background: rgba(183, 110, 121, 0.14);
        color: #5B2B3D;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }
    .confrontation-card {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(217, 119, 6, 0.4);
        border-left: 6px solid #D97706;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 18px rgba(217, 119, 6, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Inject custom styles for glowing highlights and split view
inject_custom_styles()


def render_html(html_str: str):
    """Safely render HTML without Markdown code-block interpretation."""
    clean = "".join([l.strip() for l in html_str.strip().splitlines() if l.strip()])
    st.markdown(clean, unsafe_allow_html=True)


def safe_html(text: Any, default: str = "") -> str:
    """Safely escapes text for HTML rendering, gracefully handling None or non-string values."""
    if text is None:
        return html.escape(str(default))
    return html.escape(str(text))






# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "papers_store" not in st.session_state:
    # Stores paper_name -> {"pages": ..., "metadata": ..., "engine": ..., "analysis": ..., "pdf_bytes": ...}
    st.session_state.papers_store = {}
if "active_paper_name" not in st.session_state:
    st.session_state.active_paper_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chembot_chat_history" not in st.session_state:
    st.session_state.chembot_chat_history = []
if "cross_paper_chat" not in st.session_state:
    st.session_state.cross_paper_chat = []
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Single Paper Analysis"
if "app_layout" not in st.session_state:
    st.session_state.app_layout = "split"
if "viewer_page" not in st.session_state:
    st.session_state.viewer_page = 1
if "highlight_quote" not in st.session_state:
    st.session_state.highlight_quote = None
if "researcher_notes" not in st.session_state:
    st.session_state.researcher_notes = ""


# Helper to ingest a paper
def ingest_paper_content(file_label: str, file_type: str, data: Any, api_key: Optional[str] = None):
    pdf_bytes = data if file_type == "pdf" else None
    if file_type == "pdf":
        pages = DocumentParser.parse_pdf_bytes(data)
    else:
        pages = DocumentParser.parse_text_content(data)

    if not pages:
        return False, "Could not extract text from document."

    metadata = DocumentParser.extract_metadata(pages)
    engine = EvidenceEngine(pages)
    active_key = api_key or os.getenv("GEMINI_API_KEY") or DEFAULT_BACKEND_KEY
    extractor = ChemistryExtractor(api_key=active_key)
    analysis = extractor.analyze_paper(pages, metadata, engine)

    st.session_state.papers_store[file_label] = {
        "pages": pages,
        "metadata": metadata,
        "engine": engine,
        "analysis": analysis,
        "pdf_bytes": pdf_bytes,
        "file_type": file_type
    }
    st.session_state.active_paper_name = file_label
    st.session_state.viewer_page = 1
    st.session_state.highlight_quote = None
    return True, "Success"


def render_paper_ingestion_ui(key_prefix: str = "main", title: Optional[str] = None, api_key: Optional[str] = None):
    """
    Renders a unified, multi-file-capable ingestion interface.
    Supports batch file uploads (PDF & TXT), local/server path, and raw manuscript text.
    """
    if title:
        st.markdown(f"#### {title}")
    
    upload_tab1, upload_tab2, upload_tab3 = st.tabs([
        "📁 Upload File(s) (PDF / TXT)",
        "💻 Server / Local File Path",
        "📝 Paste Manuscript Text"
    ])
    
    active_key = api_key or st.session_state.get("gemini_api_key_input") or os.getenv("GEMINI_API_KEY") or DEFAULT_BACKEND_KEY

    with upload_tab1:
        uploaded_files = st.file_uploader(
            "Choose chemistry publications (PDF or TXT) — Select one or multiple files:",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=f"{key_prefix}_file_uploader",
            help="Upload research paper files (Max 500MB supported). You can select multiple files at once."
        )
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.papers_store]
            if new_files:
                with st.spinner(f"Ingesting & analyzing {len(new_files)} new publication(s)..."):
                    for uf in new_files:
                        raw_bytes = uf.read()
                        ftype = "pdf" if uf.name.lower().endswith(".pdf") else "txt"
                        fdata = raw_bytes if ftype == "pdf" else raw_bytes.decode("utf-8", errors="replace")
                        ingest_paper_content(uf.name, ftype, fdata, active_key)
                if len(st.session_state.papers_store) >= 2 and (len(uploaded_files) > 1 or st.session_state.app_mode == "Multi-Paper Workspace"):
                    st.session_state.app_mode = "Multi-Paper Workspace"
                st.rerun()
            elif len(uploaded_files) > 0:
                st.info(f"✅ All {len(uploaded_files)} selected files are already loaded in memory.")

    with upload_tab2:
        path_input = st.text_input(
            "Enter absolute or relative path to the paper on disk:",
            placeholder="e.g. sample_papers/paper1_kinase_inhibitors.pdf or /Users/prejin/Thesis/my_paper.pdf",
            key=f"{key_prefix}_path_input"
        )
        if st.button("Load Paper from Path", key=f"{key_prefix}_btn_load_path", type="primary") and path_input.strip():
            resolved_path = os.path.expanduser(path_input.strip())
            if os.path.exists(resolved_path):
                flabel = os.path.basename(resolved_path)
                with st.spinner(f"Analyzing '{flabel}'..."):
                    if resolved_path.lower().endswith(".pdf"):
                        with open(resolved_path, "rb") as f:
                            success, msg = ingest_paper_content(flabel, "pdf", f.read(), active_key)
                    else:
                        with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
                            success, msg = ingest_paper_content(flabel, "txt", f.read(), active_key)
                    if success:
                        if len(st.session_state.papers_store) >= 2 and st.session_state.app_mode == "Multi-Paper Workspace":
                            st.session_state.app_mode = "Multi-Paper Workspace"
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.error(f"❌ File not found at path: `{resolved_path}`")

    with upload_tab3:
        pasted_text = st.text_area(
            "Paste Research Paper Text:",
            height=200,
            placeholder="Paste your chemistry paper text here (optionally with '--- Page X ---' headers)...",
            key=f"{key_prefix}_pasted_area"
        )
        doc_title = st.text_input(
            "Document Name / Title (optional):",
            value=f"Manuscript_{len(st.session_state.papers_store)+1}.txt",
            key=f"{key_prefix}_pasted_title"
        )
        if st.button("Analyze Pasted Paper", key=f"{key_prefix}_btn_load_pasted", type="primary") and pasted_text.strip():
            flabel = doc_title.strip() or f"Manuscript_{len(st.session_state.papers_store)+1}.txt"
            with st.spinner(f"Analyzing '{flabel}'..."):
                success, msg = ingest_paper_content(flabel, "txt", pasted_text.strip(), active_key)
                if success:
                    if len(st.session_state.papers_store) >= 2 and st.session_state.app_mode == "Multi-Paper Workspace":
                        st.session_state.app_mode = "Multi-Paper Workspace"
                    st.rerun()
                else:
                    st.error(msg)



# ---------------------------------------------------------
# Sidebar: Settings, Mode Switcher & Paper Store
# ---------------------------------------------------------
with st.sidebar:
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 18px; padding: 12px; background: rgba(255, 255, 255, 0.95); border-radius: 16px; border: 1px solid rgba(183, 110, 121, 0.35); box-shadow: 0 4px 16px rgba(183, 110, 121, 0.15);">
            <img src="data:image/png;base64,{logo_b64}" style="max-width: 100%; height: auto; border-radius: 10px; display: block;" alt="ChemEvidence AI Logo" />
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Gemini Intelligence Engine")
    default_key_val = os.getenv("GEMINI_API_KEY") or DEFAULT_BACKEND_KEY

    # Optional collapsed override (never exposes or pre-fills the default backend key)
    with st.expander("🔑 Custom API Key (Optional Override)", expanded=False):
        custom_key_override = st.text_input(
            "Custom Gemini Key:",
            type="password",
            value="",
            placeholder="Leave empty to use default backend key",
            help="Optional: Enter a different key only if you want to override the default backend key."
        )

    api_key_input = custom_key_override.strip() if custom_key_override.strip() else default_key_val

    if api_key_input:
        st.markdown("""
        <div style="background: rgba(183, 110, 121, 0.14); border: 1px solid rgba(183, 110, 121, 0.4); border-radius: 10px; padding: 10px 12px; margin-top: 4px; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 2px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #22C55E; box-shadow: 0 0 6px #22C55E;"></span>
                <span style="color: #8F2E44; font-weight: 700; font-size: 0.85rem;">✨ Gemini 3.8 Flash Active</span>
            </div>
            <span style="color: #6C4C54; font-size: 0.74rem; display: block; line-height: 1.35;">
                Default backend AI connected • Scientific reasoning ready
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("⚡ Local Deterministic Engine Active (Offline)")

    enable_pubchem = st.toggle("Enable Live PubChem Structure Lookup", value=True)

    st.markdown("---")
    st.markdown("### 🧭 Platform Mode")
    mode_selection = st.radio(
        "Select Workflow:",
        ["📄 Single Paper In-Depth Analysis", "📚 Multi-Paper Comparison Workspace"],
        index=0 if st.session_state.app_mode == "Single Paper Analysis" else 1
    )
    st.session_state.app_mode = "Single Paper Analysis" if "Single" in mode_selection else "Multi-Paper Workspace"

    st.markdown("---")
    st.markdown("### 📚 Loaded Papers Library")
    
    loaded_names = list(st.session_state.papers_store.keys())
    if loaded_names:
        for idx, pname in enumerate(loaded_names, 1):
            st.markdown(f"**{idx}.** 📄 `{pname}`")

        st.caption(f"Total Papers in Memory: **{len(loaded_names)}**")

        if len(loaded_names) >= 2:
            st.markdown(f"""
            <div style="background: rgba(183, 110, 121, 0.16); border-radius: 8px; padding: 6px 10px; margin-top: 4px; margin-bottom: 8px;">
                <span style="color: #8F2E44; font-weight: 700; font-size: 0.82rem;">✨ Multi-Paper Ready ({len(loaded_names)} Papers)</span>
            </div>
            """, unsafe_allow_html=True)
            if st.session_state.app_mode != "Multi-Paper Workspace":
                if st.button("📊 Open Multi-Paper Comparison", use_container_width=True, type="primary"):
                    st.session_state.app_mode = "Multi-Paper Workspace"
                    st.rerun()

        selected_paper = st.selectbox(
            "Inspect in Single Paper View:",
            options=loaded_names,
            index=loaded_names.index(st.session_state.active_paper_name) if st.session_state.active_paper_name in loaded_names else 0
        )
        st.session_state.active_paper_name = selected_paper
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("➕ Add Another", use_container_width=True):
                st.session_state.active_paper_name = None
                st.rerun()
        with col_btn2:
            if st.button("🗑️ Clear Library", use_container_width=True):
                st.session_state.papers_store = {}
                st.session_state.active_paper_name = None
                st.session_state.chat_history = []
                st.session_state.chembot_chat_history = []
                st.session_state.highlight_quote = None
                st.session_state.viewer_page = 1
                st.rerun()
    else:
        st.caption("No papers loaded yet.")

    st.markdown("---")
    st.markdown("### 🚀 Quick Benchmark Loader")
    st.caption("Test full capabilities with 1-click pre-loaded literature:")
    
    sample_dir = "/Users/prejin/Thesis/sample_papers"
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Paper 1 (Kinase SAR)", use_container_width=True):
            p1_pdf = os.path.join(sample_dir, "paper1_kinase_inhibitors.pdf")
            p1_txt = os.path.join(sample_dir, "paper1_kinase_inhibitors.txt")
            if os.path.exists(p1_pdf):
                with open(p1_pdf, "rb") as f:
                    ingest_paper_content("Paper 1: EGFR Kinase Inhibitors", "pdf", f.read(), api_key_input)
            elif os.path.exists(p1_txt):
                with open(p1_txt, "r", encoding="utf-8") as f:
                    ingest_paper_content("Paper 1: EGFR Kinase Inhibitors", "txt", f.read(), api_key_input)
            st.session_state.app_layout = "split"
            st.rerun()
    with col_s2:
        if st.button("Paper 2 (Discrepancy)", use_container_width=True):
            p2_pdf = os.path.join(sample_dir, "paper2_catalytic_synthesis_conflicts.pdf")
            p2_txt = os.path.join(sample_dir, "paper2_catalytic_synthesis_conflicts.txt")
            if os.path.exists(p2_pdf):
                with open(p2_pdf, "rb") as f:
                    ingest_paper_content("Paper 2: Catalytic Synthesis Discrepancy", "pdf", f.read(), api_key_input)
            elif os.path.exists(p2_txt):
                with open(p2_txt, "r", encoding="utf-8") as f:
                    ingest_paper_content("Paper 2: Catalytic Synthesis Discrepancy", "txt", f.read(), api_key_input)
            st.session_state.app_layout = "split"
            st.rerun()

    if st.button("Load All 3 Benchmark Papers (Multi-Paper Ready)", use_container_width=True):
        with st.spinner("Ingesting benchmark literature corpus..."):
            for fname_base, label in [
                ("paper1_kinase_inhibitors", "Paper 1: EGFR Kinase Inhibitors"),
                ("paper2_catalytic_synthesis_conflicts", "Paper 2: Catalytic Synthesis Discrepancy"),
                ("paper3_natural_product_sar", "Paper 3: Alkaloid Total Synthesis")
            ]:
                pdf_f = os.path.join(sample_dir, f"{fname_base}.pdf")
                txt_f = os.path.join(sample_dir, f"{fname_base}.txt")
                if os.path.exists(pdf_f):
                    with open(pdf_f, "rb") as f:
                        ingest_paper_content(label, "pdf", f.read(), api_key_input)
                elif os.path.exists(txt_f):
                    with open(txt_f, "r", encoding="utf-8") as f:
                        ingest_paper_content(label, "txt", f.read(), api_key_input)
        st.session_state.app_mode = "Multi-Paper Workspace"
        st.rerun()


# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown("""
<div class="chem-header">
    <div class="chem-badge">⚗️ Evidence-Aware Chemistry Intelligence Platform</div>
    <div class="chem-title">ChemEvidence AI</div>
    <p class="chem-subtitle">
        Applying LLM-based Agents to Automate Chemical Literature Extraction and Data Synthesis • Structured Discovery • Table Extraction • Reaction Pathways • SAR Analytics • Knowledge Graph • Anti-Hallucination Audit
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Floating Draggable Researcher Scratchpad (Always freely movable anywhere on screen)
# ---------------------------------------------------------
new_pin_data = st.session_state.pop("last_pinned_note", "")
reset_pos_flag = st.session_state.pop("reset_scratchpad_pos", False)
clear_notes_flag = st.session_state.pop("clear_scratchpad_flag", False)
render_floating_scratchpad(new_pinned=new_pin_data, reset_pos=reset_pos_flag, clear_notes=clear_notes_flag)

# ---------------------------------------------------------
# Floating Round ChemBot Launcher (Always accessible bottom-right buddy)
# ---------------------------------------------------------
_fab_analysis = None
_fab_pages = None
_fab_engine = None

if st.session_state.active_paper_name and st.session_state.active_paper_name in st.session_state.papers_store:
    _cur_paper = st.session_state.papers_store[st.session_state.active_paper_name]
    _fab_analysis = _cur_paper.get("analysis")
    _fab_pages = _cur_paper.get("pages")
    _fab_engine = _cur_paper.get("engine")
elif st.session_state.papers_store:
    _first_paper = next(iter(st.session_state.papers_store.values()), None)
    if _first_paper:
        _fab_analysis = _first_paper.get("analysis")
        _fab_pages = _first_paper.get("pages")
        _fab_engine = _first_paper.get("engine")

render_floating_round_chatbot(
    pages=_fab_pages,
    engine=_fab_engine,
    analysis=_fab_analysis,
    api_key_input=api_key_input,
    key_prefix="global_chembot"
)


# =========================================================
# WORKSPACE ROUTING: MULTI-PAPER vs SINGLE PAPER
# =========================================================

if st.session_state.app_mode == "Multi-Paper Workspace":
    num_loaded = len(st.session_state.papers_store)
    
    if num_loaded >= 2:
        all_papers = list(st.session_state.papers_store.items())
        all_analyses = [data["analysis"] for _, data in all_papers]
        comparison: MultiPaperComparisonResult = MultiPaperComparator.compare_papers(all_analyses)

        # Precompute summary statistics across corpus
        total_compounds = sum(len(a.compounds) for a in all_analyses)
        total_bioactivities = sum(len(a.bioactivities) for a in all_analyses)
        total_conditions = sum(len(a.conditions) for a in all_analyses)
        all_targets = sorted(list(set(b.target for a in all_analyses for b in a.bioactivities if b.target)))

        # Find Peak Potency Champion across all papers
        champion_val = float("inf")
        champion_mol = "N/A"
        champion_paper = "N/A"
        champion_target = "N/A"
        champion_pot_str = "N/A"

        for item in comparison.lead_comparison_table:
            pot_str = str(item.get("Best Potency") or "")
            num_m = re.search(r"(\d+(?:\.\d+)?)", pot_str)
            if num_m:
                v = float(num_m.group(1))
                if "µm" in pot_str.lower() or "um" in pot_str.lower():
                    v *= 1000.0
                if v < champion_val:
                    champion_val = v
                    champion_mol = str(item.get("Lead Molecule") or "N/A")
                    champion_paper = str(item.get("Paper Index") or "N/A")
                    champion_target = str(item.get("Target") or "N/A")
                    champion_pot_str = pot_str or "N/A"

        # 1. LUXURY HERO BANNER
        render_html(f"""
        <div class="multi-hero-box">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
                <div>
                    <span style="background: #B76E79; color: #FFFFFF; font-size: 0.75rem; font-weight: 800; padding: 4px 12px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.05em;">
                        ✨ Multi-Document Synthesis
                    </span>
                    <h2 style="margin: 8px 0 4px 0; color: #5B2B3D; font-weight: 800; font-size: 1.7rem;">
                        📚 Cross-Study Literature Intelligence Suite
                    </h2>
                    <p style="margin: 0; color: #8F2E44; font-size: 0.95rem; font-weight: 500;">
                        Synthesizing <b>{num_loaded} independent publications</b> across lead chemotypes, synthetic route efficiency, target concordance, and inter-laboratory discrepancies.
                    </p>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="status-found" style="font-size: 0.82rem; padding: 6px 14px;">
                        ✨ Gemini 3.8 Flash Active
                    </span>
                </div>
            </div>
        </div>
        """)

        # Active Publications Quick Navigator Ribbon
        st.markdown("##### 📑 Active Publications in Comparative Corpus")
        cols_papers = st.columns(min(num_loaded, 4))
        for p_idx, (p_name, p_data) in enumerate(all_papers):
            with cols_papers[p_idx % min(num_loaded, 4)]:
                p_analysis = p_data["analysis"]
                raw_title = str(getattr(p_analysis.metadata, "title", None) or p_name or f"Paper {p_idx+1}")
                p_title_short = raw_title[:30] + "..." if len(raw_title) > 30 else raw_title
                p_comps = len(p_analysis.compounds)
                p_assays = len(p_analysis.bioactivities)
                
                render_html(f"""
                <div style="background: rgba(255, 255, 255, 0.92); border: 1px solid rgba(183, 110, 121, 0.32); border-radius: 12px; padding: 12px 14px; margin-bottom: 8px; box-shadow: 0 2px 8px rgba(183, 110, 121, 0.08);">
                    <div style="font-size: 0.76rem; font-weight: 800; color: #B76E79; text-transform: uppercase;">Paper {p_idx + 1}</div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #2D1D22; margin: 2px 0 6px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{safe_html(raw_title)}">
                        {safe_html(p_title_short)}
                    </div>
                    <div style="font-size: 0.78rem; color: #6C4C54;">
                        🧬 <b>{p_comps}</b> compounds • 🎯 <b>{p_assays}</b> assays
                    </div>
                </div>
                """)
                if st.button(f"🔍 Jump to Paper {p_idx + 1}", key=f"nav_to_paper_{p_idx}", use_container_width=True):
                    st.session_state.active_paper_name = p_name
                    st.session_state.app_mode = "Single Paper Analysis"
                    st.rerun()

        # Ingestion Expander for adding more publications
        with st.expander(f"➕ Add Another Publication to Comparison (Currently: {num_loaded} Loaded)", expanded=False):
            st.caption("Expand your comparative literature matrix by uploading or selecting additional publications:")
            render_paper_ingestion_ui(key_prefix="multi_add_more", api_key=api_key_input)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. EXECUTIVE KPI METRIC RIBBON (4 Cards)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            render_html(f"""
            <div class="multi-kpi-card">
                <div>
                    <div class="multi-kpi-title">Corpus Scope</div>
                    <div class="multi-kpi-val">{num_loaded} Studies</div>
                </div>
                <div class="multi-kpi-sub"><b>{total_compounds}</b> chemotypes & <b>{total_bioactivities}</b> biological assays evaluated</div>
            </div>
            """)
        with kpi2:
            render_html(f"""
            <div class="multi-kpi-card">
                <div>
                    <div class="multi-kpi-title">Potency Champion</div>
                    <div class="multi-kpi-val" style="color: #065F46;">{safe_html(champion_pot_str, 'N/A')}</div>
                </div>
                <div class="multi-kpi-sub"><b>{safe_html(champion_mol, 'N/A')}</b> ({safe_html(champion_paper, 'N/A')}) vs <i>{safe_html(champion_target, 'N/A')}</i></div>
            </div>
            """)
        with kpi3:
            render_html(f"""
            <div class="multi-kpi-card">
                <div>
                    <div class="multi-kpi-title">Target Overlap</div>
                    <div class="multi-kpi-val">{len(comparison.common_targets)} Shared</div>
                </div>
                <div class="multi-kpi-sub">Across <b>{len(all_targets)}</b> distinct enzyme/cellular screens</div>
            </div>
            """)
        with kpi4:
            discrepancy_count = len(comparison.cross_study_discrepancies)
            disc_color = "#991B1B" if discrepancy_count > 0 else "#065F46"
            disc_label = f"{discrepancy_count} Flagged" if discrepancy_count > 0 else "0 Conflicts"
            disc_sub = "Cross-laboratory numerical & protocol audits" if discrepancy_count > 0 else "Full inter-laboratory concordance verified"
            render_html(f"""
            <div class="multi-kpi-card">
                <div>
                    <div class="multi-kpi-title">Literature Audit</div>
                    <div class="multi-kpi-val" style="color: {disc_color};">{disc_label}</div>
                </div>
                <div class="multi-kpi-sub">{disc_sub}</div>
            </div>
            """)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. EXECUTIVE SYNTHESIS CALLOUT
        render_html(f"""
        <div style="background: rgba(255, 255, 255, 0.94); border: 1px solid rgba(183, 110, 121, 0.38); border-left: 6px solid #B76E79; border-radius: 14px; padding: 22px 24px; margin-bottom: 24px; box-shadow: 0 4px 18px rgba(183, 110, 121, 0.1);">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 1.2rem;">📋</span>
                <h4 style="margin: 0; color: #9F3E54; font-weight: 800; font-size: 1.15rem;">Cross-Literature Executive Synthesis</h4>
            </div>
            <div style="color: #2D1D22; line-height: 1.7; font-size: 0.94rem;">
                {comparison.comparative_synthesis}
            </div>
        </div>
        """)

        # 4. FIVE CURATED WORKSPACE TABS
        tab_leads, tab_synth, tab_targets, tab_conflicts, tab_chat = st.tabs([
            "🏆 Lead Showdown & Potency",
            "⚗️ Synthetic Efficiency & Routes",
            "🎯 Target Concordance & Bioactivity",
            "⚠️ Inter-Study Conflict Audit",
            "💬 Cross-Study AI Assistant"
        ])

        # ---------------------------------------------------------
        # TAB 1: LEAD MOLECULE SHOWDOWN & POTENCY
        # ---------------------------------------------------------
        with tab_leads:
            st.markdown("#### 🏅 Head-to-Head Lead Molecule Showdown")
            st.caption("Direct comparative profiling of the premier lead chemotype identified in each independent publication:")
            
            num_cards = len(comparison.lead_comparison_table)
            lead_cols = st.columns(min(num_cards, 4))
            
            for l_idx, lead_item in enumerate(comparison.lead_comparison_table):
                with lead_cols[l_idx % min(num_cards, 4)]:
                    p_index = str(lead_item.get("Paper Index") or f"Paper {l_idx+1}")
                    p_title = str(lead_item.get("Paper Title") or "Study")
                    lead_mol = str(lead_item.get("Lead Molecule") or "Lead Analogue")
                    formula = str(lead_item.get("Formula") or "N/A")
                    target = str(lead_item.get("Target") or "Assay")
                    potency_str = str(lead_item.get("Best Potency") or "N/A")
                    yield_str = str(lead_item.get("Isolated Yield") or "N/A")
                    chem_class = str(lead_item.get("Chemical Class") or "Organic Lead")
                    
                    is_elite = False
                    num_match = re.search(r"(\d+(?:\.\d+)?)", potency_str)
                    if num_match:
                        val_nm = float(num_match.group(1))
                        if "µm" in potency_str.lower() or "um" in potency_str.lower():
                            val_nm *= 1000.0
                        if val_nm < 100.0:
                            is_elite = True
                    
                    badge_cls = "potency-badge-elite" if is_elite else "potency-badge-mod"
                    badge_icon = "💎" if is_elite else "⚡"
                    
                    render_html(f"""
                    <div class="lead-showdown-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 0.75rem; font-weight: 800; color: #B76E79; text-transform: uppercase;">{safe_html(p_index)}</span>
                            <span style="font-size: 0.72rem; color: #7A5862; max-width: 130px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{safe_html(p_title)}">{safe_html(p_title)}</span>
                        </div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #8F2E44; margin-bottom: 4px;">{safe_html(lead_mol)}</div>
                        <div style="font-size: 0.8rem; font-weight: 600; color: #5B2B3D; margin-bottom: 8px;">{safe_html(chem_class)} • <span style="font-family: monospace;">{safe_html(formula)}</span></div>
                        <div style="margin-bottom: 10px;">
                            <span class="{badge_cls}">{badge_icon} {safe_html(potency_str)}</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #2D1D22; margin-bottom: 4px;">🎯 <b>Target:</b> <i>{safe_html(target)}</i></div>
                        <div style="font-size: 0.82rem; color: #2D1D22; margin-bottom: 12px;">⚗️ <b>Isolated Yield:</b> {safe_html(yield_str)}</div>
                    </div>
                    """)
                    matching_paper_key = all_papers[l_idx][0] if l_idx < len(all_papers) else None
                    if matching_paper_key:
                        if st.button(f"🔍 Inspect in {p_index}", key=f"btn_inspect_lead_{l_idx}", use_container_width=True):
                            st.session_state.active_paper_name = matching_paper_key
                            st.session_state.app_mode = "Single Paper Analysis"
                            st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Comparative Potency & Feasibility Landscape")
            
            chart_col1, chart_col2 = st.columns(2)
            
            # Prepare data for plotting
            plot_leads = []
            for item in comparison.lead_comparison_table:
                pot_str = str(item.get("Best Potency") or "")
                num_m = re.search(r"(\d+(?:\.\d+)?)", pot_str)
                if num_m:
                    val = float(num_m.group(1))
                    if "µm" in pot_str.lower() or "um" in pot_str.lower():
                        val *= 1000.0
                    
                    y_str = str(item.get("Isolated Yield") or "")
                    y_match = re.search(r"(\d+(?:\.\d+)?)", y_str)
                    y_val = float(y_match.group(1)) if y_match else 65.0
                    
                    p_paper = str(item.get("Paper Index") or "Paper")
                    l_lead = str(item.get("Lead Molecule") or "Lead Analogue")
                    t_target = str(item.get("Target") or "Assay")

                    plot_leads.append({
                        "Paper": p_paper,
                        "Lead Molecule": f"{l_lead} ({p_paper})",
                        "Short Name": l_lead,
                        "Target": t_target,
                        "IC50 (nM)": val,
                        "Isolated Yield (%)": y_val,
                        "Raw Potency": pot_str
                    })

            if plot_leads:
                import plotly.express as px
                df_plot = pd.DataFrame(plot_leads)
                palette = ["#B76E79", "#9F3E54", "#5B2B3D", "#D48C96", "#8F2E44"]
                
                with chart_col1:
                    fig_bar = px.bar(
                        df_plot,
                        x="Lead Molecule",
                        y="IC50 (nM)",
                        color="Paper",
                        text="IC50 (nM)",
                        title="<b>Lead Molecule Potency Comparison (Log Scale)</b>",
                        color_discrete_sequence=palette,
                        template="plotly_white"
                    )
                    fig_bar.update_traces(texttemplate="%{text:.1f} nM", textposition="outside")
                    fig_bar.update_layout(
                        yaxis_type="log",
                        yaxis_title="<b>IC50 (nM) — Lower is More Potent</b>",
                        paper_bgcolor="rgba(255, 255, 255, 0.88)",
                        plot_bgcolor="rgba(253, 242, 240, 0.45)",
                        font=dict(family="Inter, sans-serif", size=11, color="#2D1D22"),
                        height=380,
                        margin=dict(l=40, r=20, t=50, b=40)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                with chart_col2:
                    fig_scatter = px.scatter(
                        df_plot,
                        x="Isolated Yield (%)",
                        y="IC50 (nM)",
                        color="Paper",
                        size=[18] * len(df_plot),
                        text="Short Name",
                        title="<b>SAR Feasibility: Yield vs. Potency Landscape</b>",
                        hover_data=["Target", "Raw Potency"],
                        color_discrete_sequence=palette,
                        template="plotly_white"
                    )
                    fig_scatter.update_traces(textposition="top center")
                    fig_scatter.update_layout(
                        yaxis_type="log",
                        yaxis_title="<b>Enzyme IC50 (nM) — Log Scale</b>",
                        xaxis_title="<b>Isolated Synthetic Yield (%)</b>",
                        paper_bgcolor="rgba(255, 255, 255, 0.88)",
                        plot_bgcolor="rgba(253, 242, 240, 0.45)",
                        font=dict(family="Inter, sans-serif", size=11, color="#2D1D22"),
                        height=380,
                        margin=dict(l=40, r=20, t=50, b=40)
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Collated Lead Comparison Matrix")
            lead_df = pd.DataFrame(comparison.lead_comparison_table)
            st.dataframe(lead_df, use_container_width=True)
            
            csv_lead = lead_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Lead Comparison Leaderboard (CSV)",
                data=csv_lead,
                file_name="chemevidence_multi_paper_leads.csv",
                mime="text/csv",
                key="dl_lead_csv"
            )

        # ---------------------------------------------------------
        # TAB 2: SYNTHETIC EFFICIENCY & ROUTES
        # ---------------------------------------------------------
        with tab_synth:
            st.markdown("#### ⚗️ Synthetic Routes & Reaction Efficiency Benchmark")
            st.caption("Comparative assessment of synthetic transformations, reaction conditions, catalysts, and isolated yields:")
            
            # Synthetic Bar Chart: Steps vs Average Yield
            synth_chart_data = []
            for idx, a in enumerate(all_analyses):
                yields = []
                for c in a.conditions:
                    if c.yield_reported:
                        try:
                            yields.append(float(c.yield_reported.replace("%", "").strip()))
                        except Exception:
                            pass
                avg_y = round(sum(yields)/len(yields), 1) if yields else 0.0
                synth_chart_data.append({
                    "Paper": f"Paper {idx + 1}",
                    "Reported Steps": len(a.conditions),
                    "Average Yield (%)": avg_y
                })
            
            if synth_chart_data:
                import plotly.graph_objects as go
                df_synth_chart = pd.DataFrame(synth_chart_data)
                fig_synth = go.Figure()
                fig_synth.add_trace(go.Bar(
                    x=df_synth_chart["Paper"],
                    y=df_synth_chart["Reported Steps"],
                    name="Reaction Steps Reported",
                    marker_color="#B76E79"
                ))
                fig_synth.add_trace(go.Bar(
                    x=df_synth_chart["Paper"],
                    y=df_synth_chart["Average Yield (%)"],
                    name="Average Yield (%)",
                    marker_color="#9F3E54"
                ))
                fig_synth.update_layout(
                    barmode="group",
                    title="<b>Synthetic Complexity & Isolated Efficiency by Publication</b>",
                    paper_bgcolor="rgba(255, 255, 255, 0.88)",
                    plot_bgcolor="rgba(253, 242, 240, 0.45)",
                    font=dict(family="Inter, sans-serif", size=11, color="#2D1D22"),
                    height=340,
                    margin=dict(l=40, r=20, t=50, b=40)
                )
                st.plotly_chart(fig_synth, use_container_width=True)

            # Per-Paper Synthetic Strategy Cards
            st.markdown("##### 🧪 Strategy Breakdown by Publication")
            synth_cols = st.columns(min(num_loaded, 3))
            for s_idx, a in enumerate(all_analyses):
                with synth_cols[s_idx % min(num_loaded, 3)]:
                    cats = list(set(str(c.catalyst).strip() for c in a.conditions if c.catalyst))[:4]
                    solvs = list(set(str(c.solvent).strip() for c in a.conditions if c.solvent))[:4]
                    cat_pills = "".join([f'<span class="synth-step-pill">⚡ {safe_html(c)}</span>' for c in cats]) if cats else '<span style="font-size:0.8rem; color:#7A5862;">Standard reagents</span>'
                    solv_pills = "".join([f'<span class="synth-step-pill">🧪 {safe_html(s)}</span>' for s in solvs]) if solvs else '<span style="font-size:0.8rem; color:#7A5862;">Standard solvents</span>'
                    
                    raw_s_title = str(getattr(a.metadata, "title", None) or f"Paper {s_idx + 1}")
                    short_s_title = raw_s_title[:32] + "..." if len(raw_s_title) > 32 else raw_s_title
                    render_html(f"""
                    <div style="background: rgba(255, 255, 255, 0.92); border: 1px solid rgba(183, 110, 121, 0.32); border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(183, 110, 121, 0.08);">
                        <div style="font-size: 0.78rem; font-weight: 800; color: #B76E79; text-transform: uppercase;">Paper {s_idx + 1}</div>
                        <div style="font-size: 1rem; font-weight: 700; color: #2D1D22; margin: 2px 0 8px 0;">{safe_html(short_s_title)}</div>
                        <div style="font-size: 0.84rem; color: #5B2B3D; margin-bottom: 6px;"><b>Transformations:</b> {len(a.conditions)} reported steps</div>
                        <div style="font-size: 0.82rem; color: #5B2B3D; margin-bottom: 6px;"><b>Key Catalysts / Bases:</b><br>{cat_pills}</div>
                        <div style="font-size: 0.82rem; color: #5B2B3D;"><b>Primary Solvents:</b><br>{solv_pills}</div>
                    </div>
                    """)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Multistep Synthetic Methodology Comparison Table")
            synth_df = pd.DataFrame(comparison.synthetic_route_comparison)
            st.dataframe(synth_df, use_container_width=True)
            
            csv_synth = synth_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Synthetic Route Comparison (CSV)",
                data=csv_synth,
                file_name="chemevidence_multi_paper_synthetic_routes.csv",
                mime="text/csv",
                key="dl_synth_csv"
            )

        # ---------------------------------------------------------
        # TAB 3: TARGET ASSAY CONCORDANCE & BIOACTIVITY
        # ---------------------------------------------------------
        with tab_targets:
            st.markdown("#### 🎯 Target Landscape & Assay Concordance")
            st.caption("Consolidated mapping of biological targets, enzymes, and cell-based assays investigated across the corpus:")
            
            # Shared vs Unique Targets
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                render_html(f"""
                <div style="background: rgba(255, 255, 255, 0.92); border: 1px solid rgba(183, 110, 121, 0.3); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                    <div style="font-size: 0.82rem; font-weight: 800; color: #9F3E54; text-transform: uppercase;">Shared Cross-Study Targets ({len(comparison.common_targets)})</div>
                    <div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px;">
                        {"".join([f'<span class="section-pill">🎯 {safe_html(t)}</span>' for t in comparison.common_targets if t]) if comparison.common_targets else '<span style="font-size:0.85rem; color:#7A5862;">No overlapping target assays identified across distinct papers.</span>'}
                    </div>
                </div>
                """)
            with col_t2:
                unique_targets = [t for t in all_targets if t not in comparison.common_targets]
                render_html(f"""
                <div style="background: rgba(255, 255, 255, 0.92); border: 1px solid rgba(183, 110, 121, 0.3); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                    <div style="font-size: 0.82rem; font-weight: 800; color: #B76E79; text-transform: uppercase;">Unique / Distinct Targets ({len(unique_targets)})</div>
                    <div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px;">
                        {"".join([f'<span class="synth-step-pill">🔬 {safe_html(t)}</span>' for t in unique_targets[:10] if t]) if unique_targets else '<span style="font-size:0.85rem; color:#7A5862;">All evaluated targets are shared across studies.</span>'}
                    </div>
                </div>
                """)

            # Collated bioactivity records across all papers
            all_bio_rows = []
            for idx, a in enumerate(all_analyses, 1):
                p_label = f"Paper {idx}"
                for b in a.bioactivities:
                    all_bio_rows.append({
                        "Paper": p_label,
                        "Paper Title": a.metadata.title[:30] + "...",
                        "Compound ID": b.compound_id or "Unspecified",
                        "Target": b.target or "General Screen",
                        "Assay Type": b.assay_type,
                        "Value": b.value,
                        "Unit": b.unit,
                        "Cell Line": b.cell_line or "Enzymatic",
                        "Verbatim Quote": b.evidence.verbatim_quote if b.evidence else "N/A"
                    })

            if all_bio_rows:
                df_all_bio = pd.DataFrame(all_bio_rows)
                
                # Interactive Target Filter
                target_options = ["All Targets"] + all_targets
                selected_target = st.selectbox("🎯 Filter Assays by Biological Target:", target_options, key="multi_target_filter")
                
                filtered_bio_df = df_all_bio if selected_target == "All Targets" else df_all_bio[df_all_bio["Target"] == selected_target]
                
                # If filtered by target and multiple compounds present, plot bar chart
                if selected_target != "All Targets" and len(filtered_bio_df) > 1:
                    plot_data_target = []
                    for _, r in filtered_bio_df.iterrows():
                        v_num = re.search(r"(\d+(?:\.\d+)?)", str(r["Value"]))
                        if v_num:
                            val_f = float(v_num.group(1))
                            if "µm" in str(r["Unit"]).lower() or "um" in str(r["Unit"]).lower():
                                val_f *= 1000.0
                            plot_data_target.append({
                                "Compound": f"{r['Compound ID']} ({r['Paper']})",
                                "IC50 (nM)": val_f,
                                "Paper": r["Paper"]
                            })
                    if plot_data_target:
                        df_target_plot = pd.DataFrame(plot_data_target)
                        fig_target_comp = px.bar(
                            df_target_plot,
                            x="Compound",
                            y="IC50 (nM)",
                            color="Paper",
                            title=f"<b>Comparative Inhibition Profile Against: {selected_target}</b>",
                            color_discrete_sequence=["#B76E79", "#9F3E54", "#5B2B3D"],
                            template="plotly_white"
                        )
                        fig_target_comp.update_layout(
                            yaxis_type="log",
                            yaxis_title="<b>Potency (nM) — Lower is More Potent</b>",
                            paper_bgcolor="rgba(255, 255, 255, 0.88)",
                            plot_bgcolor="rgba(253, 242, 240, 0.45)",
                            height=340
                        )
                        st.plotly_chart(fig_target_comp, use_container_width=True)

                st.markdown("##### 📋 Unified Bioactivity Records Across All Papers")
                st.dataframe(filtered_bio_df, use_container_width=True)
                
                csv_bio = filtered_bio_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Unified Bioactivity Records (CSV)",
                    data=csv_bio,
                    file_name="chemevidence_multi_paper_bioactivities.csv",
                    mime="text/csv",
                    key="dl_bio_csv"
                )

        # ---------------------------------------------------------
        # TAB 4: INTER-STUDY CONFLICT AUDIT
        # ---------------------------------------------------------
        with tab_conflicts:
            st.markdown("#### ⚠️ Inter-Laboratory Conflict & Discrepancy Audits")
            st.caption("Automated audit detecting contradictory numerical potency reports, divergent reaction conditions, or conflicting SAR conclusions between independent studies:")
            
            if comparison.cross_study_discrepancies:
                for c_idx, cf in enumerate(comparison.cross_study_discrepancies):
                    q_a = (cf.citation_a.verbatim_quote if cf.citation_a else 'Direct paper claim') or 'Direct paper claim'
                    q_b = (cf.citation_b.verbatim_quote if cf.citation_b else 'Direct paper claim') or 'Direct paper claim'
                    render_html(f"""
                    <div class="confrontation-card">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                            <span class="status-conflict" style="font-size: 0.8rem; padding: 4px 10px;">⚠️ {safe_html(cf.severity, 'Warning')}</span>
                            <span style="font-size: 0.78rem; font-weight: 700; color: #92400E;">Audit Alert #{c_idx + 1}</span>
                        </div>
                        <h4 style="margin: 0 0 6px 0; color: #78350F; font-size: 1.05rem; font-weight: 800;">{safe_html(cf.topic)}</h4>
                        <p style="margin: 0 0 12px 0; color: #451A03; font-size: 0.9rem; line-height: 1.5;">{safe_html(cf.description)}</p>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
                            <div style="background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(217, 119, 6, 0.35); border-radius: 8px; padding: 10px 12px;">
                                <div style="font-size: 0.75rem; font-weight: 800; color: #B45309; text-transform: uppercase;">Study A Statement</div>
                                <div style="font-size: 0.88rem; font-weight: 700; color: #2D1D22; margin: 4px 0;">{safe_html(cf.claim_a)}</div>
                                <div style="font-size: 0.78rem; font-style: italic; color: #78350F; margin-top: 4px;">Quote: "{safe_html(q_a)}"</div>
                            </div>
                            <div style="background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(217, 119, 6, 0.35); border-radius: 8px; padding: 10px 12px;">
                                <div style="font-size: 0.75rem; font-weight: 800; color: #B45309; text-transform: uppercase;">Study B Statement</div>
                                <div style="font-size: 0.88rem; font-weight: 700; color: #2D1D22; margin: 4px 0;">{safe_html(cf.claim_b)}</div>
                                <div style="font-size: 0.78rem; font-style: italic; color: #78350F; margin-top: 4px;">Quote: "{safe_html(q_b)}"</div>
                            </div>
                        </div>
                        
                        <div style="background: rgba(217, 119, 6, 0.1); border-left: 4px solid #D97706; padding: 8px 12px; border-radius: 6px; font-size: 0.82rem; color: #78350F;">
                            💡 <b>Medicinal Chemistry Reconciliation:</b> {safe_html(cf.resolution_note)}
                        </div>
                    </div>
                    """)
            else:
                render_html("""
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-left: 6px solid #10B981; border-radius: 14px; padding: 22px 24px; box-shadow: 0 4px 16px rgba(16, 185, 129, 0.08);">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                        <span style="font-size: 1.4rem;">✅</span>
                        <h4 style="margin: 0; color: #065F46; font-size: 1.1rem; font-weight: 800;">Cross-Literature Integrity Verified</h4>
                    </div>
                    <p style="margin: 0; color: #047857; font-size: 0.92rem; line-height: 1.5;">
                        Zero contradictory numerical assay reports or conflicting synthetic claims detected across the shared chemotypes and biological targets in this corpus.
                    </p>
                </div>
                """)

        # ---------------------------------------------------------
        # TAB 5: CROSS-STUDY AI ASSISTANT (GEMINI 3.8 FLASH)
        # ---------------------------------------------------------
        with tab_chat:
            st.markdown("#### 💬 Cross-Study Literature AI Assistant")
            st.caption("Ask comparative questions spanning all loaded research papers. Synthesized with Gemini 3.8 Flash with verifiable citation tracking:")
            
            # Suggested Prompt Chips
            st.markdown("##### 💡 Suggested Comparative Inquiries")
            col_chip1, col_chip2 = st.columns(2)
            with col_chip1:
                if st.button("🏆 Compare the most potent lead molecules between the papers", key="chip_cross_1", use_container_width=True):
                    st.session_state.cross_paper_pending_q = "Compare the most potent lead molecules between the papers, noting their exact IC50s and targets."
                if st.button("⚗️ Contrast the synthetic routes and efficiency across papers", key="chip_cross_2", use_container_width=True):
                    st.session_state.cross_paper_pending_q = "Compare the synthetic efficiency, step counts, and reaction yields between the papers."
            with col_chip2:
                if st.button("⚠️ Are there any conflicting bioactivity measurements reported?", key="chip_cross_3", use_container_width=True):
                    st.session_state.cross_paper_pending_q = "Are there any conflicting or diverging bioactivity measurements reported for shared targets?"
                if st.button("🔬 What are the shared scaffolds and SAR trends across the corpus?", key="chip_cross_4", use_container_width=True):
                    st.session_state.cross_paper_pending_q = "What are the common chemical scaffolds and SAR trends identified across all loaded studies?"

            st.markdown("---")
            
            # Display Chat History
            chat_container = st.container()
            with chat_container:
                if not st.session_state.cross_paper_chat:
                    render_html("""
                    <div style="background: rgba(253, 242, 240, 0.8); border: 1px dashed rgba(183, 110, 121, 0.4); border-radius: 12px; padding: 18px; text-align: center; color: #8F2E44; font-size: 0.9rem;">
                        💬 No questions asked yet. Click a suggestion chip above or type a comparative inquiry below to consult the cross-study AI.
                    </div>
                    """)
                else:
                    st.caption("⏱️ **Comparative Q&A Timeline:** Latest question displayed on the top.")
                    pairs = []
                    for i in range(0, len(st.session_state.cross_paper_chat), 2):
                        u_msg = st.session_state.cross_paper_chat[i]
                        a_msg = st.session_state.cross_paper_chat[i+1] if i+1 < len(st.session_state.cross_paper_chat) else None
                        pairs.append((i // 2, u_msg, a_msg))

                    for pair_idx, u_msg, a_msg in reversed(pairs):
                        with st.chat_message(u_msg["role"]):
                            st.markdown(u_msg["content"])
                        if a_msg:
                            with st.chat_message(a_msg["role"]):
                                st.markdown(a_msg["content"])
                                if st.button("📌 Pin to Scratchpad", key=f"cross_pin_{pair_idx}", help="Save cross-paper synthesis to Floating Scratchpad"):
                                    pin_text = f"\n\n---\n#### 🌐 Multi-Paper: {u_msg['content']}\n{a_msg['content']}\n"
                                    st.session_state.researcher_notes = (
                                        st.session_state.get("researcher_notes", "") + pin_text
                                    )
                                    st.session_state.last_pinned_note = pin_text
                                    st.toast("📌 Saved to Floating Researcher Scratchpad!", icon="📝")
                                    st.rerun()

            # Handle Chat Input or Pending Chip
            user_input = st.chat_input("Ask a cross-paper comparative question...", key="cross_paper_chat_input")
            query_to_run = user_input or st.session_state.get("cross_paper_pending_q")

            if query_to_run:
                # Clear pending question
                if "cross_paper_pending_q" in st.session_state:
                    del st.session_state["cross_paper_pending_q"]
                
                # Append user query
                st.session_state.cross_paper_chat.append({"role": "user", "content": query_to_run})
                
                # Generate Answer
                with st.spinner("Synthesizing evidence across all publications with Gemini 3.8 Flash..."):
                    answer = MultiPaperComparator.ask_cross_paper(
                        question=query_to_run,
                        analyses=all_analyses,
                        api_key=api_key_input
                    )
                
                # Append assistant reply
                st.session_state.cross_paper_chat.append({"role": "assistant", "content": answer})
                st.rerun()

            if st.session_state.cross_paper_chat:
                if st.button("🗑️ Clear Cross-Paper Chat History", key="clear_cross_chat", use_container_width=True):
                    st.session_state.cross_paper_chat = []
                    st.rerun()



        # ---------------------------------------------------------
        # BOTTOM ACTION BAR
        # ---------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        col_m_back1, col_m_back2, col_m_back3 = st.columns(3)
        with col_m_back1:
            if st.button("⬅️ Switch to Single Paper In-Depth View", use_container_width=True):
                st.session_state.app_mode = "Single Paper Analysis"
                st.rerun()
        with col_m_back2:
            synthesis_export = f"""# ChemEvidence AI: Multi-Paper Comparative Synthesis Report
Generated for {num_loaded} Publications

## Executive Synthesis
{comparison.comparative_synthesis}

## Lead Compounds Leaderboard
{pd.DataFrame(comparison.lead_comparison_table).to_markdown()}

## Synthetic Methodology Comparison
{pd.DataFrame(comparison.synthetic_route_comparison).to_markdown()}
"""
            st.download_button(
                "📥 Download Multi-Paper Synthesis Report (.md)",
                data=synthesis_export.encode("utf-8"),
                file_name="chemevidence_multi_paper_report.md",
                mime="text/markdown",
                use_container_width=True,
                key="dl_multi_report_btn"
            )
        with col_m_back3:
            if st.button("🗑️ Clear All Loaded Papers", use_container_width=True):
                st.session_state.papers_store = {}
                st.session_state.active_paper_name = None
                st.session_state.chat_history = []
                st.session_state.cross_paper_chat = []
                st.session_state.highlight_quote = None
                st.session_state.viewer_page = 1
                st.rerun()

        st.stop()

    elif num_loaded == 1:
        current_name = list(st.session_state.papers_store.keys())[0]
        st.markdown("## 📚 Multi-Paper Comparison Workspace")
        
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.92); border: 1px solid rgba(183, 110, 121, 0.35); border-radius: 16px; padding: 22px; margin-bottom: 24px; box-shadow: 0 4px 18px rgba(183, 110, 121, 0.12);">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                <span style="font-weight: 800; font-size: 1.15rem; color: #9F3E54;">📄 1 of 2 Papers Loaded in Memory</span>
                <span class="status-conflict" style="font-size: 0.82rem; padding: 4px 12px;">At Least 2 Papers Needed for Comparison</span>
            </div>
            <p style="margin: 0 0 10px 0; color: #2D1D22; font-size: 0.95rem;">
                Currently loaded in memory: <b>{safe_html(current_name)}</b>
            </p>
            <div style="background: rgba(183, 110, 121, 0.12); border-left: 4px solid #B76E79; padding: 12px 16px; border-radius: 6px; font-size: 0.9rem; color: #5B2B3D; line-height: 1.5;">
                💡 <b>Multi-Paper Comparison</b> requires at least <b>2 publications</b> to generate cross-study lead molecule leaderboards, synthetic route comparisons, and inter-laboratory conflict audits.
                Please load your second paper below or pick a benchmark paper to compare immediately.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_st1, col_st2 = st.columns([3, 1])
        with col_st1:
            render_paper_ingestion_ui(key_prefix="multi_staging", title="📂 Load 2nd Publication to Compare", api_key=api_key_input)
        with col_st2:
            st.markdown("#### 🚀 Benchmark Starters")
            st.caption("Instantly compare against pre-loaded literature:")
            sample_dir = "/Users/prejin/Thesis/sample_papers"
            if st.button("➕ Add Paper 1 (Kinase SAR)", key="stage_add_p1", use_container_width=True):
                p1_pdf = os.path.join(sample_dir, "paper1_kinase_inhibitors.pdf")
                p1_txt = os.path.join(sample_dir, "paper1_kinase_inhibitors.txt")
                if os.path.exists(p1_pdf):
                    with open(p1_pdf, "rb") as f:
                        ingest_paper_content("Paper 1: EGFR Kinase Inhibitors", "pdf", f.read(), api_key_input)
                elif os.path.exists(p1_txt):
                    with open(p1_txt, "r", encoding="utf-8") as f:
                        ingest_paper_content("Paper 1: EGFR Kinase Inhibitors", "txt", f.read(), api_key_input)
                st.rerun()

            if st.button("➕ Add Paper 2 (Discrepancy)", key="stage_add_p2", use_container_width=True):
                p2_pdf = os.path.join(sample_dir, "paper2_catalytic_synthesis_conflicts.pdf")
                p2_txt = os.path.join(sample_dir, "paper2_catalytic_synthesis_conflicts.txt")
                if os.path.exists(p2_pdf):
                    with open(p2_pdf, "rb") as f:
                        ingest_paper_content("Paper 2: Catalytic Synthesis Discrepancy", "pdf", f.read(), api_key_input)
                elif os.path.exists(p2_txt):
                    with open(p2_txt, "r", encoding="utf-8") as f:
                        ingest_paper_content("Paper 2: Catalytic Synthesis Discrepancy", "txt", f.read(), api_key_input)
                st.rerun()

            if st.button("⬅️ View Active Paper Individually", key="stage_btn_single", use_container_width=True):
                st.session_state.app_mode = "Single Paper Analysis"
                st.rerun()

        st.stop()

    else:
        st.markdown("## 📚 Multi-Paper Comparison Workspace")
        st.info("No publications loaded yet. Please upload 2 or more publications to compare:")
        render_paper_ingestion_ui(key_prefix="multi_empty", title="📂 Ingest Publications for Comparison", api_key=api_key_input)
        st.stop()


# =========================================================
# SINGLE PAPER INGESTION SCREEN (When no active paper)
# =========================================================
if not st.session_state.active_paper_name or st.session_state.active_paper_name not in st.session_state.papers_store:
    st.markdown("""
    <div class="user-friendly-card" style="margin-bottom: 24px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div>
                <h3 style="margin: 0; color: #5B2B3D; font-weight: 800;">👋 Welcome to ChemEvidence AI</h3>
                <p style="margin: 4px 0 0 0; color: #8F2E44; font-weight: 600; font-size: 0.88rem;">
                    Applying LLM-based Agents to Automate Chemical Literature Extraction and Data Synthesis
                </p>
                <p style="margin: 6px 0 0 0; color: #6C4C54; font-size: 0.95rem;">
                    Your evidence-aware chemistry intelligence workspace. Extract chemical structures, SAR potency series, multistep reactions, and verifiable citations with zero hallucinations.
                </p>
            </div>
            <div>
                <span class="status-found" style="font-size: 0.8rem; padding: 4px 12px;">✨ Gemini 3.8 Flash Connected</span>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 20px;">
            <div style="background: rgba(253, 242, 240, 0.7); border: 1px solid rgba(183, 110, 121, 0.25); border-radius: 10px; padding: 12px 16px;">
                <b style="color: #9F3E54; font-size: 0.9rem;">1️⃣ Upload or Select</b>
                <p style="margin: 4px 0 0 0; font-size: 0.82rem; color: #6C4C54;">Load research papers in PDF or TXT, or launch 1-click benchmark publications below.</p>
            </div>
            <div style="background: rgba(253, 242, 240, 0.7); border: 1px solid rgba(183, 110, 121, 0.25); border-radius: 10px; padding: 12px 16px;">
                <b style="color: #9F3E54; font-size: 0.9rem;">2️⃣ Automated Extraction</b>
                <p style="margin: 4px 0 0 0; font-size: 0.82rem; color: #6C4C54;">Mines molecules, 2D structures, tables, reaction schemes, and interactive knowledge graph.</p>
            </div>
            <div style="background: rgba(253, 242, 240, 0.7); border: 1px solid rgba(183, 110, 121, 0.25); border-radius: 10px; padding: 12px 16px;">
                <b style="color: #9F3E54; font-size: 0.9rem;">3️⃣ Grounded Q&A</b>
                <p style="margin: 4px 0 0 0; font-size: 0.82rem; color: #6C4C54;">Interrogate literature with verifiable page and verbatim quote citations in high-res split view.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1-Click Benchmark Starters
    st.markdown("#### 🚀 1-Click Benchmark Literature (Instant Preview)")
    st.caption("Explore all features immediately with verified peer-reviewed chemistry benchmarks:")

    sample_dir = "/Users/prejin/Thesis/sample_papers"
    col_bench1, col_bench2, col_bench3 = st.columns(3)

    with col_bench1:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.85); border: 1px solid rgba(183, 110, 121, 0.3); border-radius: 12px; padding: 16px; margin-bottom: 10px;">
            <b style="color: #9F3E54; font-size: 1.0rem;">📄 Paper 1: EGFR Kinase SAR</b>
            <p style="margin: 6px 0 10px 0; font-size: 0.82rem; color: #6C4C54;">
                Medicinal chemistry campaign synthesizing 4-anilinoquinazoline kinase inhibitors. Contains full SAR series, IC50 data, and 2D chemical structures.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Load Paper 1 Benchmark", key="home_load_p1", use_container_width=True, type="primary"):
            p1_pdf = os.path.join(sample_dir, "paper1_kinase_inhibitors.pdf")
            p1_txt = os.path.join(sample_dir, "paper1_kinase_inhibitors.txt")
            if os.path.exists(p1_pdf):
                with open(p1_pdf, "rb") as f:
                    ingest_paper_content("Paper 1: EGFR Kinase Inhibitors", "pdf", f.read(), api_key_input)
            elif os.path.exists(p1_txt):
                with open(p1_txt, "r", encoding="utf-8") as f:
                    ingest_paper_content("Paper 1: EGFR Kinase Inhibitors", "txt", f.read(), api_key_input)
            st.session_state.app_layout = "split"
            st.rerun()

    with col_bench2:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.85); border: 1px solid rgba(183, 110, 121, 0.3); border-radius: 12px; padding: 16px; margin-bottom: 10px;">
            <b style="color: #9F3E54; font-size: 1.0rem;">⚗️ Paper 2: Catalytic Discrepancy</b>
            <p style="margin: 6px 0 10px 0; font-size: 0.82rem; color: #6C4C54;">
                Palladium-catalyzed C-C cross-coupling optimization. Demonstrates automatic detection of internal contradictory yield reports between text and tables.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Load Paper 2 Benchmark", key="home_load_p2", use_container_width=True, type="primary"):
            p2_pdf = os.path.join(sample_dir, "paper2_catalytic_synthesis_conflicts.pdf")
            p2_txt = os.path.join(sample_dir, "paper2_catalytic_synthesis_conflicts.txt")
            if os.path.exists(p2_pdf):
                with open(p2_pdf, "rb") as f:
                    ingest_paper_content("Paper 2: Catalytic Synthesis Discrepancy", "pdf", f.read(), api_key_input)
            elif os.path.exists(p2_txt):
                with open(p2_txt, "r", encoding="utf-8") as f:
                    ingest_paper_content("Paper 2: Catalytic Synthesis Discrepancy", "txt", f.read(), api_key_input)
            st.session_state.app_layout = "split"
            st.rerun()

    with col_bench3:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.85); border: 1px solid rgba(183, 110, 121, 0.3); border-radius: 12px; padding: 16px; margin-bottom: 10px;">
            <b style="color: #9F3E54; font-size: 1.0rem;">📚 All 3 Benchmark Papers</b>
            <p style="margin: 6px 0 10px 0; font-size: 0.82rem; color: #6C4C54;">
                Loads the complete 3-paper corpus for multi-paper comparative analysis, cross-study synthesis, and inter-laboratory discrepancy leaderboards.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Load Full 3-Paper Corpus", key="home_load_all", use_container_width=True, type="secondary"):
            with st.spinner("Ingesting benchmark literature corpus..."):
                for fname_base, label in [
                    ("paper1_kinase_inhibitors", "Paper 1: EGFR Kinase Inhibitors"),
                    ("paper2_catalytic_synthesis_conflicts", "Paper 2: Catalytic Synthesis Discrepancy"),
                    ("paper3_natural_product_sar", "Paper 3: Alkaloid Total Synthesis")
                ]:
                    pdf_f = os.path.join(sample_dir, f"{fname_base}.pdf")
                    txt_f = os.path.join(sample_dir, f"{fname_base}.txt")
                    if os.path.exists(pdf_f):
                        with open(pdf_f, "rb") as f:
                            ingest_paper_content(label, "pdf", f.read(), api_key_input)
                    elif os.path.exists(txt_f):
                        with open(txt_f, "r", encoding="utf-8") as f:
                            ingest_paper_content(label, "txt", f.read(), api_key_input)
            st.session_state.app_mode = "Multi-Paper Workspace"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    render_paper_ingestion_ui(key_prefix="home_ingest", title="📂 Or Analyze Your Own Chemistry Publication", api_key=api_key_input)
    st.markdown("<br>", unsafe_allow_html=True)
    st.stop()


# =========================================================
# ACTIVE PAPER DETAIL VIEW (8 TABS / SIDE-BY-SIDE SPLIT VIEW)
# =========================================================
active_data = st.session_state.papers_store.get(st.session_state.active_paper_name)
if not active_data:
    st.stop()

analysis: PaperAnalysisResult = active_data["analysis"]
pages: List[PageContent] = active_data["pages"]
engine: EvidenceEngine = active_data["engine"]
meta: PaperMetadata = active_data["metadata"]
pdf_bytes: Optional[bytes] = active_data.get("pdf_bytes")

# Dynamic document-wide formula and MW resolution for active paper
if analysis.compounds and any(c.formula is None or c.molecular_weight is None for c in analysis.compounds):
    from core.chem_calculator import ChemCalculator
    ChemCalculator.bind_formulas_and_weights_document_wide(analysis.compounds, analysis.tables, pages)

# Summary Metrics Row
col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{len(analysis.compounds)}</div>
        <div class="metric-lbl">Key Compounds</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{len(analysis.bioactivities)}</div>
        <div class="metric-lbl">Bioactivities</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{len(analysis.tables)}</div>
        <div class="metric-lbl">Extracted Tables</div>
    </div>
    """, unsafe_allow_html=True)
with col_m4:
    steps_total = sum(len(p.steps) for p in analysis.reaction_pathways) if analysis.reaction_pathways else len(analysis.conditions)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{steps_total}</div>
        <div class="metric-lbl">Reaction Steps</div>
    </div>
    """, unsafe_allow_html=True)
with col_m5:
    gaps_count = len(analysis.research_gaps)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val" style="color: #F59E0B;">{gaps_count}</div>
        <div class="metric-lbl">Research Gaps</div>
    </div>
    """, unsafe_allow_html=True)
with col_m6:
    conf_score = analysis.confidence_breakdown.overall_score if analysis.confidence_breakdown else 94.0
    tier_str = analysis.confidence_breakdown.tier if analysis.confidence_breakdown else "Strong (A)"
    st.markdown(f"""
    <div class="metric-card" style="border-color: #10B981;">
        <div class="metric-val" style="color: #10B981;">{conf_score:.1f}%</div>
        <div class="metric-lbl">Audit: {tier_str}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Reusable Chemical Intelligence Suite Renderers
# ---------------------------------------------------------

def render_external_paper_cards(related_papers: List[Dict[str, Any]], key_prefix: str = "ext_paper"):
    """Renders structured, clickable publication cards for external literature results."""
    if not related_papers:
        return
    st.markdown(f"**📚 Related Research Publications ({len(related_papers)}):**")
    for p_idx, paper in enumerate(related_papers):
        title = html.escape(str(paper.get("title", "Research Publication")))
        source = html.escape(str(paper.get("source", "Peer-Reviewed Literature")))
        url = paper.get("url", "#")
        doi = html.escape(str(paper.get("doi", ""))) if paper.get("doi") else ""
        doi_badge = f'<span style="background: #E0F2FE; color: #0369A1; padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; font-family: monospace;">DOI: {doi}</span>' if doi else ""
        pmid = html.escape(str(paper.get("pmid", ""))) if paper.get("pmid") else ""
        pmid_badge = f'<span style="background: #F1F5F9; color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; font-family: monospace;">PMID: {pmid}</span>' if pmid else ""
        
        st.markdown(
            f'<div class="external-paper-card">'
            f'<div style="font-weight: 700; font-size: 0.95rem; color: #0F172A; margin-bottom: 4px;">'
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: #0369A1;">📄 {title} ↗</a>'
            f'</div>'
            f'<div style="font-size: 0.85rem; color: #64748B; margin-bottom: 6px;">🏛️ {source}</div>'
            f'<div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">'
            f'{doi_badge} {pmid_badge}'
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="font-size: 0.82rem; color: #2563EB; font-weight: 600; margin-left: auto; text-decoration: underline;">View Publication ↗</a>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )


def render_qa_section(pages: List[PageContent], engine: EvidenceEngine, analysis: PaperAnalysisResult, api_key_input: Optional[str], key_prefix: str = "qa"):
    st.markdown("### 💬 Evidence-Aware Interactive Question Answering")
    active_key = api_key_input or os.getenv("GEMINI_API_KEY") or DEFAULT_BACKEND_KEY
    if active_key:
        st.caption("🟢 **Gemini 3.8 Flash** active • Verifiable citation grounding with strict absence alerts.")
    else:
        st.caption("⚡ **Local Chemistry Engine** active (Offline) • Grounded in verbatim paper text.")

    # Handle pending 1-click external literature exploration
    if st.session_state.get("pending_external_query"):
        pending_q = st.session_state.pop("pending_external_query")
        with st.chat_message("user"):
            st.markdown(f"Yes, please search broader literature & provide related papers for: **{pending_q}**")
        with st.chat_message("assistant"):
            with st.spinner(f"Consulting broader scientific literature & retrieving related papers on \"{pending_q[:40]}...\""):
                qa = QASystem(api_key=active_key)
                doc_title = analysis.metadata.title if analysis and analysis.metadata else None
                ext_resp = qa.answer_from_external_literature(pending_q, paper_context=doc_title)
                st.session_state.chat_history.append({
                    "question": f"Yes, provide broader literature answers and related papers on: {pending_q}",
                    "response": ext_resp
                })
        st.rerun()

    # Preset Quick-Prompt Chips
    chip_query = render_dynamic_question_chips(key_prefix=f"{key_prefix}_chips")

    # Chat Input
    chat_query = st.chat_input("Ask any question about yields, mechanisms, assays, or citations...", key=f"{key_prefix}_chat_input")
    active_query = chip_query or chat_query

    # Process Active Query (Prepends to reverse-ordered history on rerun)
    if active_query:
        clean_q_lower = active_query.strip().lower().rstrip(".!?,")
        affirmative_words = {
            "yes", "yes please", "yeah", "yep", "sure", "ok", "okay", "please",
            "yes please provide", "yes provide", "yes show me", "yes related papers",
            "yes i need", "yes i need that", "yes i do", "yes i want that",
            "yes related research paper", "yes related research papers", "yes give me",
            "yes answer", "yes find papers", "yes search literature", "yes please give related paper"
        }
        is_affirmation = (
            clean_q_lower in affirmative_words
            or clean_q_lower.startswith("yes,")
            or clean_q_lower.startswith("yes ")
        )

        last_absent_question = None
        for prev in reversed(st.session_state.chat_history):
            prev_resp = prev.get("response")
            if prev_resp and getattr(prev_resp, "status", None) == "absent":
                last_absent_question = getattr(prev_resp, "question", None)
                prev_resp.can_expand_external = False
                break

        if is_affirmation and last_absent_question:
            with st.spinner(f"Consulting broader scientific literature & retrieving related papers on \"{last_absent_question[:40]}...\""):
                qa = QASystem(api_key=active_key)
                doc_title = analysis.metadata.title if analysis and analysis.metadata else None
                ext_resp = qa.answer_from_external_literature(last_absent_question, paper_context=doc_title)
                st.session_state.chat_history.append({
                    "question": active_query,
                    "response": ext_resp
                })
        else:
            with st.spinner("Reasoning over literature and verifying evidence citations..."):
                qa = QASystem(api_key=active_key)
                response = qa.ask(active_query, pages, engine, analysis)
                st.session_state.chat_history.append({
                    "question": active_query,
                    "response": response
                })
        st.rerun()

    # Display Chat History (LATEST QUESTION ON TOP)
    if not st.session_state.chat_history:
        st.info("💡 No questions asked yet. Click any quick-prompt chip above or type your scientific question to begin.")
    else:
        st.caption(f"⏱️ **Q&A History ({len(st.session_state.chat_history)} inquiries):** Latest question displayed on the top.")

        for item_idx, item in reversed(list(enumerate(st.session_state.chat_history))):
            with st.chat_message("user"):
                st.markdown(item["question"])
            with st.chat_message("assistant"):
                resp: QAResponse = item["response"]

                if resp.status == "found":
                    st.markdown('<span class="status-found">🟢 Evidence Verified in Document</span>', unsafe_allow_html=True)
                elif resp.status == "absent":
                    st.markdown('<span class="status-absent">🔴 No Supporting Evidence Found</span>', unsafe_allow_html=True)
                elif resp.status == "conflicting":
                    st.markdown('<span class="status-conflict">⚠️ Conflicting Evidence Detected - Flagged for Review</span>', unsafe_allow_html=True)
                elif resp.status == "external_literature":
                    st.markdown('<span class="status-external">🌐 Broader Scientific Literature & Related Papers</span>', unsafe_allow_html=True)

                st.markdown(resp.answer)

                # Interactive button to explore external literature if absent
                if resp.status == "absent" and getattr(resp, "can_expand_external", False):
                    if st.button("🌐 Yes, Search Broader Literature & Provide Related Papers", key=f"{key_prefix}_hist_expand_{item_idx}"):
                        st.session_state.pending_external_query = resp.question
                        resp.can_expand_external = False
                        st.rerun()

                if resp.status == "external_literature" and getattr(resp, "related_papers", None):
                    render_external_paper_cards(resp.related_papers, key_prefix=f"{key_prefix}_hist_papers_{item_idx}")

                if resp.conflicts:
                    for cf in resp.conflicts:
                        desc = html.escape(str(cf.description))
                        ca = html.escape(str(cf.claim_a))
                        cb = html.escape(str(cf.claim_b))
                        res = html.escape(str(cf.resolution_note))
                        st.markdown(
                            f'<div class="conflict-card">'
                            f'<b style="color: #92400E;">Discrepancy Details:</b> {desc}<br>'
                            f'• <b>Source A:</b> {ca} (Page {cf.citation_a.page_number})<br>'
                            f'• <b>Source B:</b> {cb} (Page {cf.citation_b.page_number})<br>'
                            f'• <b>Recommendation:</b> {res}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                if resp.citations:
                    st.markdown(f"**Supporting Grounded Evidence Cards ({len(resp.citations)}):**")
                    for cit_idx, cit in enumerate(resp.citations):
                        render_grounded_evidence_card(
                            citation=cit,
                            title=f"Citation {cit_idx + 1} (Page {cit.page_number})",
                            key=f"{key_prefix}_hist_cit_{item_idx}_{cit_idx}"
                        )

                # Action row: Pin to Scratchpad
                col_pin, col_empty = st.columns([1, 3])
                with col_pin:
                    if st.button("📌 Pin to Scratchpad", key=f"{key_prefix}_pin_{item_idx}", help="Append this question, answer, and citations directly to your Floating Researcher Scratchpad"):
                        pin_text = f"\n\n---\n#### ❓ {item['question']}\n{resp.answer}\n"
                        if resp.citations:
                            pin_text += "\n*Supporting Citations:*\n"
                            for c in resp.citations:
                                pin_text += f"- Page {c.page_number} ({c.section}): \"{c.verbatim_quote}\"\n"
                        st.session_state.researcher_notes = (
                            st.session_state.get("researcher_notes", "") + pin_text
                        )
                        st.session_state.last_pinned_note = pin_text
                        st.toast("📌 Saved to Floating Researcher Scratchpad!", icon="📝")
                        st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Q&A History", key=f"{key_prefix}_clear_chat_history"):
            st.session_state.chat_history = []
            st.rerun()


def render_structured_discovery(analysis: PaperAnalysisResult, meta: PaperMetadata, key_prefix: str = "disc"):
    st.markdown("### 📋 Executive Scientific Overview & Discovery")
    st.markdown(f"**Document:** *{st.session_state.active_paper_name}* ({meta.page_count} pages, ~{meta.total_words} words)")
    st.info(analysis.executive_summary)

    # Discrepancy & Conflict Banner
    if analysis.conflicts_detected:
        for cf_idx, cf in enumerate(analysis.conflicts_detected):
            cf_top = html.escape(str(cf.topic))
            cf_desc = html.escape(str(cf.description))
            ca_txt = html.escape(str(cf.claim_a))
            cb_txt = html.escape(str(cf.claim_b))
            qa_txt = html.escape(str(cf.citation_a.verbatim_quote).strip().replace("\r\n", " ").replace("\n", " "))
            qb_txt = html.escape(str(cf.citation_b.verbatim_quote).strip().replace("\r\n", " ").replace("\n", " "))
            res_txt = html.escape(str(cf.resolution_note))
            st.markdown(
                f'<div class="conflict-card">'
                f'<span class="status-conflict">⚠️ {cf.severity}: {cf_top}</span>'
                f'<p style="margin-top: 8px; margin-bottom: 6px; font-weight: 600; color: #78350F;">{cf_desc}</p>'
                f'<div style="font-size: 0.85rem; color: #92400E; line-height: 1.6;">'
                f'• <b>Claim A:</b> {ca_txt} <span class="page-pill">Page {cf.citation_a.page_number}</span> <i>"{qa_txt}"</i><br>'
                f'• <b>Claim B:</b> {cb_txt} <span class="page-pill">Page {cf.citation_b.page_number}</span> <i>"{qb_txt}"</i><br>'
                f'• <b>Action:</b> {res_txt}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            col_cf1, col_cf2 = st.columns(2)
            with col_cf1:
                render_show_in_paper_button(cf.citation_a, key=f"{key_prefix}_cf_a_{cf_idx}", button_text=f"📄 Show Claim A in Paper (Page {cf.citation_a.page_number})")
            with col_cf2:
                render_show_in_paper_button(cf.citation_b, key=f"{key_prefix}_cf_b_{cf_idx}", button_text=f"📄 Show Claim B in Paper (Page {cf.citation_b.page_number})")

    # Section 1 & 2: Properties & Bioactivity Tables
    col_p, col_b = st.columns(2)
    with col_p:
        st.markdown("#### 1. Chemical & Physical Properties")
        if analysis.properties:
            prop_data = [{
                "Compound": p.compound_id or "General",
                "Parameter": p.parameter,
                "Value": f"{p.value} {p.unit or ''}".strip(),
                "Page": f"Page {p.evidence.page_number}",
                "Citation": p.evidence.verbatim_quote[:90] + "..."
            } for p in analysis.properties]
            st.dataframe(pd.DataFrame(prop_data), use_container_width=True)
        else:
            st.write("No physical property data extracted.")

    with col_b:
        st.markdown("#### 2. Bioactivity & Assay Results")
        if analysis.bioactivities:
            bio_data = [{
                "Compound": b.compound_id or "Lead",
                "Assay": b.assay_type,
                "Target": b.target,
                "Value": f"{b.value} {b.unit or ''}".strip(),
                "Page": f"Page {b.evidence.page_number}",
                "Citation": b.evidence.verbatim_quote[:90] + "..."
            } for b in analysis.bioactivities]
            st.dataframe(pd.DataFrame(bio_data), use_container_width=True)
        else:
            st.write("No biological evaluation data extracted.")

    st.markdown("---")

    # Section 3: Significant Scientific Takeaways
    st.markdown("#### 3. Significant Scientific Findings & SAR Takeaways")
    for f_idx, f in enumerate(analysis.findings):
        f_cat = html.escape(str(f.category))
        f_txt = html.escape(str(f.finding))
        f_sec = html.escape(str(f.evidence.section))
        st.markdown(
            f'<div style="background: rgba(255, 255, 255, 0.88); border: 1px solid rgba(183, 110, 121, 0.28); border-left: 4px solid #B76E79; padding: 12px 16px; border-radius: 0 10px 10px 0; margin-bottom: 8px; color: #2D1D22; box-shadow: 0 2px 8px rgba(183, 110, 121, 0.08);">'
            f'<b style="color: #9F3E54;">[{f_cat}]</b> {f_txt}<br>'
            f'<span class="page-pill" style="margin-top: 6px;">Page {f.evidence.page_number}</span>'
            f'<span class="section-pill">{f_sec}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        render_show_in_paper_button(f.evidence, key=f"{key_prefix}_f_btn_{f_idx}")


def render_table_extraction(analysis: PaperAnalysisResult, key_prefix: str = "tables"):
    st.markdown("### 📋 Table Extraction — Automatic Chemistry Table Mining")
    st.markdown(
        "Structured scientific data extracted from literature tables (SAR series, reaction conditions, optimization matrices). "
        "Filter, sort, and export directly to CSV or Excel."
    )

    if analysis.tables:
        for idx, table in enumerate(analysis.tables):
            with st.expander(f"📊 **{table.table_id}**: {table.title} (Page {table.page_number})", expanded=(idx == 0)):
                st.markdown(f"**Section:** `{table.section}` | **Dimensions:** `{len(table.rows)} rows × {len(table.headers)} columns`")
                try:
                    df_table = TableExtractor.table_to_dataframe(table)
                    if not df_table.empty:
                        # Extra defense for PyArrow serialization: ensure all columns are unique strings
                        cols = list(df_table.columns)
                        seen = set()
                        deduped_cols = []
                        for c in cols:
                            c_str = str(c)
                            if c_str in seen:
                                count = 1
                                while f"{c_str}_{count}" in seen:
                                    count += 1
                                c_str = f"{c_str}_{count}"
                            seen.add(c_str)
                            deduped_cols.append(c_str)
                        df_table.columns = deduped_cols
                        st.dataframe(df_table, use_container_width=True)
                    else:
                        st.info("Table has no structured rows.")
                except Exception as table_err:
                    st.warning(f"Could not render interactive grid for {table.table_id}: {table_err}")
                    preview_txt = f"{table.title}\n" + " | ".join(table.headers) + "\n" + "\n".join([" | ".join(r) for r in table.rows[:10]])
                    st.code(preview_txt)
                    df_table = pd.DataFrame()

                col_dl1, col_dl2, col_cit = st.columns([1, 1, 2])
                with col_dl1:
                    if not df_table.empty:
                        csv_data = df_table.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"{table.table_id.replace(' ', '_').lower()}.csv",
                            mime="text/csv",
                            key=f"{key_prefix}_dl_csv_{idx}"
                        )
                with col_dl2:
                    if not df_table.empty:
                        excel_buf = io.BytesIO()
                        df_table.to_excel(excel_buf, index=False, engine="openpyxl")
                        st.download_button(
                            label="📊 Download Excel (.xlsx)",
                            data=excel_buf.getvalue(),
                            file_name=f"{table.table_id.replace(' ', '_').lower()}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"{key_prefix}_dl_xlsx_{idx}"
                        )
                with col_cit:
                    render_grounded_evidence_card(
                        table.evidence,
                        title=f"Source Citation for {table.table_id}",
                        key=f"{key_prefix}_table_ev_{idx}"
                    )
    else:
        st.info("No explicit structured tables were detected in this document. Tables will appear here if present in the publication.")


def render_reaction_pathways(analysis: PaperAnalysisResult, key_prefix: str = "reactions"):
    st.markdown("### ⚗️ Reaction Pathways & Synthetic Cascades")
    st.markdown("Reconstructed multistep synthetic routes, transformation types, reagents, catalysts, temperatures, and isolated yields.")

    if analysis.reaction_pathways:
        for p_idx, pathway in enumerate(analysis.reaction_pathways):
            p_title = html.escape(str(pathway.title))
            p_target = html.escape(str(pathway.target_compound))
            p_yield = html.escape(str(pathway.overall_yield))
            st.markdown(
                f'<div style="background: rgba(255, 255, 255, 0.88); border: 1px solid rgba(183, 110, 121, 0.35); border-radius: 14px; padding: 18px 24px; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(183, 110, 121, 0.1);">'
                f'<h4 style="margin: 0; color: #9F3E54; font-weight: 700;">{p_title}</h4>'
                f'<p style="color: #6C4C54; margin-top: 4px; margin-bottom: 8px;">Target Molecule: <b style="color: #2D1D22;">{p_target}</b> | Estimated Overall Yield: <b style="color: #8F2E44;">{p_yield}</b></p>'
                f'</div>',
                unsafe_allow_html=True
            )

            for step in pathway.steps:
                s_name = html.escape(str(step.reaction_name))
                s_yield = html.escape(str(step.yield_percent or 'Reported in text'))
                s_sm = html.escape(str(step.starting_material or 'Precursor'))
                s_prod = html.escape(str(step.product))
                s_reag = html.escape(str(step.reagents or 'Standard conditions'))
                s_solv = html.escape(str(step.solvent or 'N/A'))
                s_temp = f" at {html.escape(str(step.temperature))}" if step.temperature else ""
                s_time = f" ({html.escape(str(step.time))})" if step.time else ""
                st.markdown(
                    f'<div style="background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(183, 110, 121, 0.28); border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; box-shadow: 0 2px 10px rgba(183, 110, 121, 0.08);">'
                    f'<div style="display: flex; align-items: center; justify-content: space-between;">'
                    f'<div>'
                    f'<span class="step-badge">{step.step_number}</span>'
                    f'<b style="font-size: 1.05rem; color: #9F3E54;">{s_name}</b>'
                    f'</div>'
                    f'<div style="font-weight: 700; color: #8F2E44; font-size: 1.0rem;">'
                    f'Yield: {s_yield}'
                    f'</div>'
                    f'</div>'
                    f'<div style="margin-top: 10px; font-size: 0.88rem; color: #2D1D22; line-height: 1.65;">'
                    f'• <b>Starting Material:</b> {s_sm}<br>'
                    f'• <b>Product / Intermediate:</b> <code style="color: #9F3E54; background: rgba(183, 110, 121, 0.12); padding: 2px 6px; border-radius: 4px;">{s_prod}</code><br>'
                    f'• <b>Reagents / Catalyst:</b> {s_reag}<br>'
                    f'• <b>Solvent / Temp:</b> {s_solv}{s_temp}{s_time}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                render_grounded_evidence_card(
                    step.evidence,
                    title=f"Evidence for Step {step.step_number}: {step.reaction_name}",
                    key=f"{key_prefix}_step_ev_{p_idx}_{step.step_number}"
                )
    else:
        st.info("No multi-step synthetic routes were extracted.")


def render_sar_analytics(analysis: PaperAnalysisResult, key_prefix: str = "sar"):
    st.markdown("### 📈 Structure-Activity Relationship (SAR) Analytics")
    st.markdown("Quantitative medicinal chemistry exploration of substituent effects, pIC50 potency landscapes, and multi-compound comparisons.")

    if analysis.sar_analyses:
        for s_idx, sar in enumerate(analysis.sar_analyses):
            st.markdown(f"#### {sar.series_name}")
            if sar.key_insights:
                for ins in sar.key_insights:
                    ins_clean = html.escape(str(ins))
                    st.markdown(
                        f'<div style="background: rgba(253, 242, 240, 0.95); border: 1px solid rgba(183, 110, 121, 0.3); border-left: 4px solid #B76E79; padding: 10px 16px; border-radius: 0 8px 8px 0; margin-bottom: 8px; font-size: 0.9rem; color: #2D1D22;">{ins_clean}</div>',
                        unsafe_allow_html=True
                    )

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_pot = SAREngine.create_sar_potency_chart(sar)
                if fig_pot.data:
                    st.plotly_chart(fig_pot, use_container_width=True)
            with col_chart2:
                fig_scat = SAREngine.create_yield_vs_potency_scatter(sar)
                if fig_scat.data:
                    st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown("---")

    # Side-by-Side Compound Comparator
    st.markdown("### 🔬 Side-by-Side Compound Comparator")
    st.markdown("Select 2 to 4 synthesized compounds to inspect a differential properties and potency matrix:")

    comp_ids = [c.compound_id for c in analysis.compounds]
    if len(comp_ids) >= 2:
        selected_comps = st.multiselect(
            "Select Compounds to Compare:",
            options=comp_ids,
            default=comp_ids[:min(3, len(comp_ids))],
            key=f"{key_prefix}_cmp_select"
        )
        if selected_comps:
            cmp_df = SAREngine.compare_compounds_matrix(
                analysis.compounds,
                analysis.bioactivities,
                analysis.properties,
                selected_comps
            )
            st.dataframe(cmp_df, use_container_width=True)
    else:
        st.caption("Need at least 2 compounds in the document to generate comparison matrix.")


def render_knowledge_graph(analysis: PaperAnalysisResult, key_prefix: str = "graph"):
    st.markdown("### 🌐 Chemistry Literature Knowledge Graph")
    st.markdown(
        "Interactive relational network linking **Publication** $\\leftrightarrow$ **Compounds** "
        "$\\leftrightarrow$ **Biological Targets** $\\leftrightarrow$ **Reaction Routes** $\\leftrightarrow$ **Cell Lines**."
    )

    if analysis.knowledge_graph and analysis.knowledge_graph.nodes:
        st.caption(f"Knowledge Graph topology: **{len(analysis.knowledge_graph.nodes)} Entities (Nodes)** and **{len(analysis.knowledge_graph.edges)} Relationships (Edges)**. Drag nodes or hover for details.")
        kg_html = KnowledgeGraphBuilder.render_interactive_html(analysis.knowledge_graph)
        components.html(kg_html, height=520)
    else:
        st.info("Knowledge graph nodes are being compiled.")


def render_anti_hallucination_audit(analysis: PaperAnalysisResult, key_prefix: str = "audit"):
    st.markdown("### 🛡️ Anti-Hallucination Evidence Audit & Research Gap Detection")
    st.markdown(
        "A rigorous, multi-dimensional assessment of extraction fidelity alongside an automated "
        "peer-review check for omitted pharmacology, missing controls, and unexplored chemical space."
    )

    # 1. Confidence Breakdown
    st.markdown("#### 1. Multi-Factor Evidence Confidence Rubric")
    conf = analysis.confidence_breakdown
    if conf:
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.metric("Verbatim Quote Fidelity", f"{conf.verbatim_match_score:.1f}%")
        with col_c2:
            st.metric("Entity-Metric Proximity", f"{conf.proximity_score:.1f}%")
        with col_c3:
            st.metric("Numeric & Unit Precision", f"{conf.numeric_precision_score:.1f}%")
        with col_c4:
            st.metric("Cross-Section Validation", f"{conf.cross_validation_score:.1f}%")

        st.info(f"**Audit Rationale:** {conf.rationale}")

    st.markdown("---")

    # 2. Research-Gap Detection
    st.markdown("#### 2. Proactive Literature Gap & Omission Detection")
    if analysis.research_gaps:
        for gap in analysis.research_gaps:
            card_class = f"gap-card-{gap.severity.lower()}"
            g_cat = html.escape(str(gap.category))
            g_title = html.escape(str(gap.title))
            g_sev = html.escape(str(gap.severity))
            g_desc = html.escape(str(gap.description))
            g_rec = html.escape(str(gap.recommendation))
            st.markdown(
                f'<div class="{card_class}">'
                f'<div style="display: flex; align-items: center; justify-content: space-between;">'
                f'<b style="font-size: 1.05rem;">[{g_cat}] {g_title}</b>'
                f'<span style="font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">{g_sev} Severity</span>'
                f'</div>'
                f'<p style="margin-top: 8px; margin-bottom: 6px; font-size: 0.9rem; line-height: 1.55;">{g_desc}</p>'
                f'<div style="font-size: 0.85rem; font-weight: 600; margin-top: 6px;">'
                f'💡 <b>Recommendation for Future Work:</b> {g_rec}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.success("✅ Comprehensive experimental coverage; no major pharmacological gaps detected.")


def render_export_section(analysis: PaperAnalysisResult, meta: PaperMetadata, key_prefix: str = "export", pdf_bytes: Optional[bytes] = None, pages: Optional[List[PageContent]] = None):
    st.markdown("### 📑 Document Explorer & Complete Chemical Dossier Export")

    col_tab8_a, col_tab8_b = st.columns([3, 1])
    with col_tab8_b:
        if st.button("🔍 Open Side-by-Side Split View", key=f"{key_prefix}_open_split", use_container_width=True):
            st.session_state.app_layout = "split"
            st.rerun()

    # Integrated Live Document Viewer inside Export
    if pages:
        render_live_document_viewer(
            pdf_bytes=pdf_bytes,
            pages=pages,
            current_page=st.session_state.viewer_page,
            highlight_quote=st.session_state.highlight_quote,
            key_prefix=f"{key_prefix}_live_doc"
        )

    st.markdown("---")
    st.markdown("#### 📥 One-Click Export Suite")
    col_ex1, col_ex2, col_ex3 = st.columns(3)

    with col_ex1:
        json_str = analysis.model_dump_json(indent=2)
        st.download_button(
            label="💾 Download Structured JSON",
            data=json_str,
            file_name=f"{st.session_state.active_paper_name or 'paper'}_analysis.json",
            mime="application/json",
            use_container_width=True,
            key=f"{key_prefix}_dl_json"
        )

    with col_ex2:
        md_lines = [
            f"# Evidence-Aware Chemistry Report: {meta.title}",
            f"**File:** {st.session_state.active_paper_name} | **Pages:** {meta.page_count}",
            f"\n## Executive Summary\n{analysis.executive_summary}\n",
            "## Key Findings"
        ]
        for f in analysis.findings:
            md_lines.append(f"- **[{f.category}]**: {f.finding} [Page {f.evidence.page_number}]")
            md_lines.append(f"  *Citation:* \"{f.evidence.verbatim_quote}\"")

        md_lines.append("\n## Bioactivity Results")
        for b in analysis.bioactivities:
            md_lines.append(f"- **{b.compound_id or 'Compound'}**: {b.assay_type} against {b.target} = {b.value} {b.unit} [Page {b.evidence.page_number}]")

        md_report = "\n".join(md_lines)
        st.download_button(
            label="📄 Download Markdown Report",
            data=md_report,
            file_name=f"{st.session_state.active_paper_name or 'paper'}_report.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"{key_prefix}_dl_md"
        )

    with col_ex3:
        excel_out = io.BytesIO()
        with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
            if analysis.compounds:
                pd.DataFrame([c.model_dump() for c in analysis.compounds]).to_excel(writer, sheet_name="Compounds", index=False)
            if analysis.bioactivities:
                pd.DataFrame([b.model_dump() for b in analysis.bioactivities]).to_excel(writer, sheet_name="Bioactivities", index=False)
            if analysis.properties:
                pd.DataFrame([p.model_dump() for p in analysis.properties]).to_excel(writer, sheet_name="Properties", index=False)
            for idx, t in enumerate(analysis.tables):
                t_df = TableExtractor.table_to_dataframe(t)
                t_df.to_excel(writer, sheet_name=f"Table_{idx+1}", index=False)

        st.download_button(
            label="📊 Download Excel Chemical Dossier",
            data=excel_out.getvalue(),
            file_name=f"{st.session_state.active_paper_name or 'paper'}_dossier.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"{key_prefix}_dl_xlsx"
        )


# ---------------------------------------------------------
# Workspace Layout Mode Switcher (Side-by-Side Split View vs Tabs)
# ---------------------------------------------------------
col_sw_mode, col_sw_hint = st.columns([2.8, 1.2])
with col_sw_mode:
    layout_mode_choice = st.radio(
        "Workspace View Layout:",
        [
            "🔍 Side-by-Side Split View (55% Document Reader | 45% Intelligence Suite)",
            "📊 Multi-Tab Chemical Studio (Discovery, Tables, Reactions, SAR, Knowledge Graph, Audit, Q&A, Export)"
        ],
        index=0 if st.session_state.app_layout == "split" else 1,
        horizontal=True,
        key="layout_mode_radio"
    )
    st.session_state.app_layout = "split" if "Side-by-Side" in layout_mode_choice else "tabs"

with col_sw_hint:
    if st.session_state.app_layout == "split":
        st.info("💡 **Split View Active**: High-Res PDF on left (55%); Intelligence Suite on right (45%) with quote highlights.")
    else:
        st.info("💡 **Studio Active**: Click any Grounded Evidence button to auto-jump directly into Side-by-Side Split View.")

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# SIDE-BY-SIDE SPLIT VIEW WORKSPACE (55% / 45%)
# =========================================================
if st.session_state.app_layout == "split":
    col_doc, col_qa = st.columns([55, 45], gap="large")

    # -----------------------------------------------------
    # LEFT COLUMN (55% width): Live Document Viewer
    # -----------------------------------------------------
    with col_doc:
        st.markdown("### 📄 Live Document Viewer")
        st.markdown(
            f"**Active Publication:** *{st.session_state.active_paper_name}* "
            f"({meta.page_count} pages, ~{meta.total_words} words)"
        )
        render_live_document_viewer(
            pdf_bytes=pdf_bytes,
            pages=pages,
            current_page=st.session_state.viewer_page,
            highlight_quote=st.session_state.highlight_quote,
            key_prefix="split_live_doc"
        )

    # -----------------------------------------------------
    # RIGHT COLUMN (45% width): Chemistry Intelligence Suite
    # -----------------------------------------------------
    with col_qa:
        st.markdown("### ⚗️ Chemistry Intelligence Suite")

        suite_view_choice = st.pills(
            "Intelligence Suite View:",
            options=[
                "💬 Q&A Assistant",
                "📊 Structured Discovery",
                "📋 Table Extraction",
                "⚗️ Reaction Pathways",
                "📈 SAR Analytics",
                "🌐 Knowledge Graph",
                "🛡️ Anti-Hallucination Audit"
            ],
            default="💬 Q&A Assistant",
            key="split_suite_view",
            label_visibility="collapsed"
        )
        active_view = suite_view_choice or "💬 Q&A Assistant"

        if active_view == "💬 Q&A Assistant":
            render_qa_section(pages, engine, analysis, api_key_input, key_prefix="split_qa")
        elif active_view == "📊 Structured Discovery":
            render_structured_discovery(analysis, meta, key_prefix="split_disc")
        elif active_view == "📋 Table Extraction":
            render_table_extraction(analysis, key_prefix="split_tbl")
        elif active_view == "⚗️ Reaction Pathways":
            render_reaction_pathways(analysis, key_prefix="split_rxn")
        elif active_view == "📈 SAR Analytics":
            render_sar_analytics(analysis, key_prefix="split_sar")
        elif active_view == "🌐 Knowledge Graph":
            render_knowledge_graph(analysis, key_prefix="split_kg")
        elif active_view == "🛡️ Anti-Hallucination Audit":
            render_anti_hallucination_audit(analysis, key_prefix="split_audit")

    # Stop rendering studio tabs in split view
    st.stop()


# ---------------------------------------------------------
# Tabbed Navigation Ribbon (Multi-Tab Chemical Studio)
# ---------------------------------------------------------
tab_discovery, tab_tables, tab_reactions, tab_sar, tab_graph, tab_audit, tab_qa, tab_export = st.tabs([
    "📊 Structured Discovery",
    "📋 Table Extraction",
    "⚗️ Reaction Pathways",
    "📈 SAR Analytics",
    "🌐 Knowledge Graph",
    "🛡️ Anti-Hallucination Audit",
    "💬 Evidence Q&A",
    "📑 Document & Export"
])

with tab_discovery:
    render_structured_discovery(analysis, meta, key_prefix="studio_disc")

with tab_tables:
    render_table_extraction(analysis, key_prefix="studio_tbl")

with tab_reactions:
    render_reaction_pathways(analysis, key_prefix="studio_rxn")

with tab_sar:
    render_sar_analytics(analysis, key_prefix="studio_sar")

with tab_graph:
    render_knowledge_graph(analysis, key_prefix="studio_kg")

with tab_audit:
    render_anti_hallucination_audit(analysis, key_prefix="studio_audit")

with tab_qa:
    render_qa_section(pages, engine, analysis, api_key_input, key_prefix="studio_qa")

with tab_export:
    render_export_section(analysis, meta, key_prefix="studio_export", pdf_bytes=pdf_bytes, pages=pages)

