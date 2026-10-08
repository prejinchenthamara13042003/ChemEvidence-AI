"""
Reusable UI Components for ChemEvidence AI:
- High-Resolution PDF Page Viewer & Navigation Controls
- Grounded Evidence Card with "📄 Show in Paper & Highlight Quote (Page X)" Action
- Compound 2D SMILES Inspector
- Dynamic Question Chips
- Glowing Amber/Yellow Citation Highlighting Styles
"""

import html
import json
import urllib.parse
from typing import List, Optional, Dict, Any
import streamlit as st
import streamlit.components.v1 as components

from core.models import EvidenceCitation, CompoundInfo, PaperAnalysisResult, MultilingualBotResponse
from core.multilingual_bot import MultilingualChemBot
from src.pdf_parser import DocumentParser, PageContent


# ---------------------------------------------------------
# CSS Injection for Glowing Highlights & Split View Aesthetics
# ---------------------------------------------------------
SPLIT_VIEW_CSS = """
<style>
    /* Glowing Citation Mark for Text Reader */
    .glowing-citation-mark {
        background: rgba(224, 133, 147, 0.38) !important;
        color: #7A2437 !important;
        border-bottom: 2px solid #B76E79 !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-weight: 700 !important;
        box-shadow: 0 0 14px rgba(183, 110, 121, 0.55) !important;
        display: inline !important;
        animation: pulse-glow-rose 2.2s infinite ease-in-out !important;
    }

    @keyframes pulse-glow-rose {
        0% {
            box-shadow: 0 0 6px rgba(183, 110, 121, 0.3);
            background: rgba(224, 133, 147, 0.3);
        }
        50% {
            box-shadow: 0 0 16px rgba(183, 110, 121, 0.85);
            background: rgba(224, 133, 147, 0.5);
        }
        100% {
            box-shadow: 0 0 6px rgba(183, 110, 121, 0.3);
            background: rgba(224, 133, 147, 0.3);
        }
    }

    /* Grounded Evidence Card Styling */
    .evidence-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(183, 110, 121, 0.32);
        border-left: 4px solid #B76E79;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(183, 110, 121, 0.12);
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .evidence-card:hover {
        border-color: #B76E79;
        box-shadow: 0 6px 22px rgba(183, 110, 121, 0.25);
        transform: translateY(-1px);
    }
    .evidence-quote {
        font-family: 'Inter', -apple-system, sans-serif;
        font-style: italic;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #2D1D22;
        background: rgba(253, 242, 240, 0.95);
        padding: 10px 14px;
        border-radius: 8px;
        border-left: 3px solid #B76E79;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    /* Jump Banner */
    .jump-alert-banner {
        background: linear-gradient(90deg, rgba(235, 182, 188, 0.35) 0%, rgba(255, 255, 255, 0.92) 100%);
        border: 1px solid #B76E79;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 10px rgba(183, 110, 121, 0.12);
    }

    /* Viewer Container */
    .doc-viewer-container {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(183, 110, 121, 0.3);
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 8px 24px -4px rgba(183, 110, 121, 0.15);
    }

    .doc-text-body {
        font-family: 'Fira Code', 'SFMono-Regular', monospace;
        font-size: 0.88rem;
        line-height: 1.7;
        color: #2D1D22;
        white-space: pre-wrap;
        background: rgba(255, 255, 255, 0.96);
        border-radius: 10px;
        padding: 18px;
        max-height: 680px;
        overflow-y: auto;
        border: 1px solid rgba(183, 110, 121, 0.25);
    }

    /* SMILES Inspector Card */
    .smiles-inspector-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(183, 110, 121, 0.32);
        border-radius: 14px;
        padding: 18px;
        margin-top: 14px;
        box-shadow: 0 6px 20px -4px rgba(183, 110, 121, 0.12);
    }
</style>
"""


def inject_custom_styles():
    """Injects custom CSS styles into Streamlit app."""
    st.markdown(SPLIT_VIEW_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Action: Jump to Paper and Highlight Quote
# ---------------------------------------------------------
def jump_to_paper_citation(citation: EvidenceCitation):
    """
    Executes the jump action:
    1. Sets viewer page to the citation page number.
    2. Sets the verbatim quote to highlight.
    3. Automatically switches the layout to the Side-by-Side Split View.
    4. Records jump metadata for banner display.
    """
    st.session_state.viewer_page = citation.page_number
    st.session_state.highlight_quote = citation.verbatim_quote
    st.session_state.app_layout = "split"
    st.session_state.citation_jump_alert = {
        "page": citation.page_number,
        "quote": citation.verbatim_quote,
        "section": citation.section
    }
    st.rerun()


def render_show_in_paper_button(
    citation: EvidenceCitation,
    key: str,
    button_text: Optional[str] = None,
    use_container_width: bool = False
) -> bool:
    """
    Renders the prominent '📄 Show in Paper & Highlight Quote (Page X)' button.
    """
    label = button_text or f"📄 Show in Paper & Highlight Quote (Page {citation.page_number})"
    if st.button(label, key=key, use_container_width=use_container_width, type="secondary"):
        jump_to_paper_citation(citation)
        return True
    return False


# ---------------------------------------------------------
# Grounded Evidence Card
# ---------------------------------------------------------
def render_grounded_evidence_card(
    citation: EvidenceCitation,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    badge_label: Optional[str] = None,
    key: Optional[str] = None,
    button_label: Optional[str] = None
):
    """
    Renders an interactive Grounded Evidence Card with page badge, section badge,
    verbatim quote box, and the prominent '📄 Show in Paper & Highlight Quote (Page X)' button.
    """
    card_key = key or f"ev_card_{citation.page_number}_{hash(citation.verbatim_quote) % 100000}"

    clean_title = html.escape(str(title)) if title else ""
    clean_subtitle = html.escape(str(subtitle)) if subtitle else ""
    clean_section = html.escape(str(citation.section)) if citation.section else "Document Section"
    clean_badge = html.escape(str(badge_label)) if badge_label else ""
    # Sanitize verbatim quote: escape HTML and replace newlines with space to prevent markdown indentation blocks
    safe_quote = html.escape(str(citation.verbatim_quote).strip().replace("\r\n", " ").replace("\n", " "))

    title_html = f'<div style="color: #9F3E54; font-weight: 700; font-size: 0.95rem; margin-bottom: 2px;">{clean_title}</div>' if clean_title else ""
    subtitle_html = f'<div style="color: #6C4C54; font-size: 0.82rem; margin-bottom: 4px;">{clean_subtitle}</div>' if clean_subtitle else ""
    badge_html = f'<span style="background: rgba(183, 110, 121, 0.2); color: #8F2E44; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; font-weight: 600; margin-left: 4px;">{clean_badge}</span>' if clean_badge else ""

    card_html = (
        f'<div class="evidence-card">'
        f'<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 6px;">'
        f'<div>{title_html}{subtitle_html}</div>'
        f'<div><span class="page-pill">Page {citation.page_number}</span><span class="section-pill">{clean_section}</span>{badge_html}</div>'
        f'</div>'
        f'<div class="evidence-quote">"{safe_quote}"</div>'
        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)

    render_show_in_paper_button(
        citation,
        key=f"btn_{card_key}",
        button_text=button_label,
        use_container_width=True
    )


# ---------------------------------------------------------
# Page Navigation Controls (◄ Prev Page | Page X of Y | Next Page ►)
# ---------------------------------------------------------
def render_page_navigation(total_pages: int, current_page: int, key_prefix: str = "doc_nav") -> int:
    """
    Renders clean page navigation controls:
    ◄ Prev Page | Page X of Y | Next Page ►
    Returns the updated current page number.
    """
    total_pages = max(1, total_pages)
    current_page = max(1, min(current_page, total_pages))

    col_prev, col_mid, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button("◄ Prev Page", key=f"{key_prefix}_prev", disabled=(current_page <= 1), use_container_width=True):
            new_page = max(1, current_page - 1)
            st.session_state.viewer_page = new_page
            st.rerun()

    with col_mid:
        selected_page = st.selectbox(
            label="Page Selector",
            options=list(range(1, total_pages + 1)),
            index=current_page - 1,
            format_func=lambda p: f"Page {p} of {total_pages}",
            key=f"{key_prefix}_select",
            label_visibility="collapsed"
        )
        if selected_page != current_page:
            st.session_state.viewer_page = selected_page
            st.rerun()

    with col_next:
        if st.button("Next Page ►", key=f"{key_prefix}_next", disabled=(current_page >= total_pages), use_container_width=True):
            new_page = min(total_pages, current_page + 1)
            st.session_state.viewer_page = new_page
            st.rerun()

    return st.session_state.get("viewer_page", current_page)


# ---------------------------------------------------------
# Live Document Viewer Component (Left Column)
# ---------------------------------------------------------
def render_live_document_viewer(
    pdf_bytes: Optional[bytes],
    pages: List[PageContent],
    current_page: int,
    highlight_quote: Optional[str] = None,
    key_prefix: str = "live_doc"
):
    """
    Renders the Live Document Viewer for the Left Column (55% width).
    Displays high-resolution PDF page previews with PyMuPDF amber highlight boxes,
    or formatted text sections with glowing yellow <mark> tags.
    """
    inject_custom_styles()
    total_pages = len(pages) if pages else 1
    current_page = max(1, min(current_page, total_pages))
    curr_page_obj = pages[current_page - 1] if pages and current_page - 1 < len(pages) else None

    # Jump / Highlight Banner
    if highlight_quote:
        col_banner1, col_banner2 = st.columns([4, 1])
        with col_banner1:
            clean_hl = html.escape(str(highlight_quote)[:90].replace("\r\n", " ").replace("\n", " "))
            banner_html = (
                f'<div class="jump-alert-banner">'
                f'<div>'
                f'<span style="color: #F59E0B; font-weight: 700;">🎯 Page {current_page} Highlight Active</span><br>'
                f'<span style="font-size: 0.84rem; color: #CBD5E1; font-style: italic;">"{clean_hl}..."</span>'
                f'</div>'
                f'</div>'
            )
            st.markdown(banner_html, unsafe_allow_html=True)
        with col_banner2:
            if st.button("✖ Clear Highlight", key=f"{key_prefix}_clear_hl", use_container_width=True):
                st.session_state.highlight_quote = None
                if "citation_jump_alert" in st.session_state:
                    del st.session_state.citation_jump_alert
                st.rerun()

    # Navigation Controls Bar
    render_page_navigation(total_pages, current_page, key_prefix=f"{key_prefix}_nav")

    # Viewer Sub-controls: Zoom and Display View Toggle
    col_v1, col_v2 = st.columns([1, 1])
    with col_v1:
        view_mode = st.radio(
            "Display Mode:",
            ["🖼️ High-Res PDF Preview", "📝 Text & Glowing Quote"],
            horizontal=True,
            key=f"{key_prefix}_mode"
        )
    with col_v2:
        zoom_choice = st.selectbox(
            "Render Zoom / Fidelity:",
            options=[1.5, 2.0, 2.5],
            index=1,
            format_func=lambda z: f"{int(z * 100)}% ({'Crisp High-DPI' if z >= 2.0 else 'Standard'})",
            key=f"{key_prefix}_zoom"
        )

    # 1. High-Resolution PDF Page Preview
    if "High-Res" in view_mode:
        img_bytes = None
        if pdf_bytes:
            img_bytes = DocumentParser.render_pdf_page_image(
                pdf_bytes=pdf_bytes,
                page_number=current_page,
                zoom=zoom_choice,
                highlight_quote=highlight_quote
            )
        elif pages:
            # Generate on-the-fly PDF from pages if pure text was uploaded
            synth_pdf = DocumentParser.synthesize_pdf_from_pages(pages)
            if synth_pdf:
                img_bytes = DocumentParser.render_pdf_page_image(
                    pdf_bytes=synth_pdf,
                    page_number=current_page,
                    zoom=zoom_choice,
                    highlight_quote=highlight_quote
                )

        if img_bytes:
            st.markdown('<div class="doc-viewer-container">', unsafe_allow_html=True)
            st.image(
                img_bytes,
                caption=f"High-Resolution PDF Page {current_page} of {total_pages} (PyMuPDF High-DPI Engine)",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ High-resolution preview not available for this page. Switching to Text Inspector below.")
            view_mode = "📝 Text & Glowing Quote"

    # 2. Text & Glowing Quote Inspector
    if "Text" in view_mode:
        if curr_page_obj:
            if curr_page_obj.sections:
                badges = " ".join([f"<span class='section-pill'>{s['name']}</span>" for s in curr_page_obj.sections])
                st.markdown(f"**Detected Page Sections:** {badges}", unsafe_allow_html=True)

            highlighted_html = DocumentParser.highlight_quote_in_text(curr_page_obj.text, highlight_quote)
            doc_html = f'<div class="doc-text-body">{highlighted_html}</div>'
            st.markdown(doc_html, unsafe_allow_html=True)
        else:
            st.info("No text content available on this page.")


# ---------------------------------------------------------
# Dynamic Question Chips Component
# ---------------------------------------------------------
def render_dynamic_question_chips(key_prefix: str = "chips") -> Optional[str]:
    """
    Renders responsive dynamic prompt chips for instant literature interrogation.
    Returns the question text if a chip was clicked, otherwise None.
    """
    st.markdown("**⚡ Dynamic Research Queries:**")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    selected = None
    with c1:
        if st.button("📈 What was the isolated yield?", key=f"{key_prefix}_1", use_container_width=True):
            selected = "What was the reported isolated yield and under what reaction conditions?"
    with c2:
        if st.button("🎯 What are the key bioactivities?", key=f"{key_prefix}_2", use_container_width=True):
            selected = "What are the key bioactivity or IC50 assay results for the synthesized series?"
    with c3:
        if st.button("🚫 What was the rat bioavailability?", key=f"{key_prefix}_3", use_container_width=True):
            selected = "What was the oral bioavailability in rats and pharmacokinetic clearance?"
    with c4:
        if st.button("⚗️ What was the synthetic bottleneck?", key=f"{key_prefix}_4", use_container_width=True):
            selected = "What reaction conditions or synthetic transformations were optimized?"
    with c5:
        if st.button("🔬 What is the lead scaffold?", key=f"{key_prefix}_5", use_container_width=True):
            selected = "What is the core chemical scaffold and optimal lead molecule identified in this study?"
    with c6:
        if st.button("⚠️ Any experimental conflicts?", key=f"{key_prefix}_6", use_container_width=True):
            selected = "Are there any internal conflicts, conflicting yield reports, or literature discrepancies?"

    return selected


# ---------------------------------------------------------
# Compound 2D SMILES Inspector Component
# ---------------------------------------------------------
def render_compound_smiles_inspector(
    compounds: List[CompoundInfo],
    key_prefix: str = "smiles_insp"
):
    """
    Renders an interactive Compound 2D SMILES Inspector in the Right Column:
    - Compound selector
    - Live 2D chemical structure image
    - SMILES string, Formula, Molecular Weight, and Grounded Citation with Show-in-Paper button.
    """
    if not compounds:
        st.caption("No chemical compounds extracted from document.")
        return

    comp_options = [f"{c.compound_id} - {c.name}" for c in compounds]
    sel_idx = st.selectbox(
        "Select Compound to Inspect:",
        options=range(len(compounds)),
        format_func=lambda i: comp_options[i],
        key=f"{key_prefix}_select"
    )

    comp = compounds[sel_idx]

    st.markdown('<div class="smiles-inspector-card">', unsafe_allow_html=True)
    col_str1, col_str2 = st.columns([1, 1.2])

    with col_str1:
        # 2D Structure Rendering: via PubChem REST API
        rendered = False
        if comp.pubchem_cid:
            img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{comp.pubchem_cid}/PNG?image_size=300x300"
            st.image(img_url, caption=f"{comp.compound_id} (PubChem CID: {comp.pubchem_cid})", use_container_width=True)
            rendered = True
        elif comp.smiles:
            encoded_smiles = urllib.parse.quote(comp.smiles)
            img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/PNG?image_size=300x300"
            st.image(img_url, caption=f"{comp.compound_id} (2D SMILES Depiction)", use_container_width=True)
            rendered = True

        if not rendered:
            st.info(f"⚗️ Structure for {comp.compound_id} (Scaffold derivative)")

    with col_str2:
        st.markdown(f"#### {comp.compound_id}")
        st.markdown(f"<span style='color: #6C4C54; font-size: 0.88rem;'>{comp.name}</span>", unsafe_allow_html=True)

        form_val = comp.formula or "Scaffold Derivative"
        mw_val = f"{comp.molecular_weight:.1f} g/mol" if comp.molecular_weight else "Not reported in text"

        st.markdown(f"""
        <div style="font-size: 0.85rem; line-height: 1.6; color: #2D1D22; margin-top: 6px;">
            • <b>Formula:</b> <code style="color: #9F3E54; background: rgba(183, 110, 121, 0.12); padding: 2px 6px; border-radius: 4px;">{form_val}</code><br>
            • <b>Molecular Weight:</b> <b style="color: #8F2E44;">{mw_val}</b><br>
            • <b>Chemical Class:</b> {comp.chemical_class or 'Small Molecule'}<br>
            • <b>PubChem CID:</b> {f'<a href="https://pubchem.ncbi.nlm.nih.gov/compound/{comp.pubchem_cid}" target="_blank" style="color: #9F3E54; font-weight: 600;">{comp.pubchem_cid}</a>' if comp.pubchem_cid else 'Unregistered'}
        </div>
        """, unsafe_allow_html=True)

        if comp.smiles:
            st.markdown(f"**Canonical SMILES:**")
            st.code(comp.smiles, language="text")

    st.markdown("</div>", unsafe_allow_html=True)

    # Citation Card for the selected compound with Jump-in-Paper button!
    if comp.evidence:
        render_grounded_evidence_card(
            citation=comp.evidence,
            title=f"Original Citation for {comp.compound_id}",
            key=f"{key_prefix}_ev_{comp.compound_id}"
        )


def render_floating_scratchpad(new_pinned: str = "", reset_pos: bool = False, clear_notes: bool = False):
    """
    Renders a floating, draggable, persistent researcher scratchpad that can be moved
    freely anywhere on the screen, minimized, edited, and downloaded.
    """
    pinned_json = json.dumps(new_pinned)
    reset_json = "true" if reset_pos else "false"
    clear_json = "true" if clear_notes else "false"

    html_code = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0; padding:0; background:transparent;">
<script>
(function() {{
    let doc;
    try {{
        doc = (window.parent && window.parent.document) ? window.parent.document : document;
    }} catch (e) {{
        doc = document;
    }}

    const PinnedFromStreamlit = {pinned_json};
    const ResetPos = {reset_json};
    const ClearNotes = {clear_json};

    // If container already exists on parent doc, handle updates
    let container = doc.getElementById('chem-floating-scratchpad-root');
    if (container) {{
        if (ResetPos) {{
            container.style.left = 'auto';
            container.style.top = 'auto';
            container.style.bottom = '24px';
            container.style.right = '28px';
            container.classList.remove('minimized');
            localStorage.removeItem('chem_scratchpad_pos');
            localStorage.setItem('chem_scratchpad_minimized', 'false');
            const minBtn = doc.getElementById('chem-pad-min-btn');
            if (minBtn) minBtn.innerText = '_';
        }}
        if (ClearNotes) {{
            const textarea = doc.getElementById('chem-scratchpad-textarea');
            if (textarea) {{
                textarea.value = '';
                localStorage.setItem('chem_floating_notes', '');
                const countEl = doc.getElementById('chem-scratchpad-counts');
                if (countEl) countEl.innerText = '0 words • 0 chars';
                const titleText = doc.getElementById('chem-title-text');
                if (titleText) titleText.innerText = 'Floating Scratchpad (0w)';
            }}
        }}
        const textarea = doc.getElementById('chem-scratchpad-textarea');
        if (textarea && PinnedFromStreamlit && PinnedFromStreamlit.trim()) {{
            const current = textarea.value || "";
            const sections = PinnedFromStreamlit.split(/\\n\\s*---\\s*\\n/);
            let addedAny = false;
            let updatedText = current;
            for (let sec of sections) {{
                sec = sec.trim();
                if (sec && !updatedText.includes(sec)) {{
                    updatedText = (updatedText.trim() ? updatedText.trim() + "\\n\\n---\\n" : "") + sec;
                    addedAny = true;
                }}
            }}
            if (addedAny) {{
                textarea.value = updatedText;
                localStorage.setItem('chem_floating_notes', textarea.value);
                const wc = textarea.value.trim() ? textarea.value.trim().split(/\\s+/).length : 0;
                const cc = textarea.value.length;
                const countEl = doc.getElementById('chem-scratchpad-counts');
                if (countEl) countEl.innerText = `${{wc}} words • ${{cc}} chars`;
                const titleText = doc.getElementById('chem-title-text');
                if (titleText) titleText.innerText = `Floating Scratchpad (${{wc}}w)`;
                const header = doc.getElementById('chem-scratchpad-header');
                if (header) {{
                    header.style.boxShadow = 'inset 0 0 16px rgba(16, 185, 129, 0.9)';
                    setTimeout(() => {{ header.style.boxShadow = ''; }}, 1500);
                }}
            }}
        }}
        return;
    }}

    // Inject styles
    if (!doc.getElementById('chem-floating-scratchpad-styles')) {{
        const styleEl = doc.createElement('style');
        styleEl.id = 'chem-floating-scratchpad-styles';
        styleEl.innerHTML = `
            #chem-floating-scratchpad-root {{
                position: fixed;
                bottom: 96px;
                right: 28px;
                width: 380px;
                height: 420px;
                min-width: 280px;
                min-height: 200px;
                max-width: 95vw;
                max-height: 90vh;
                background: #FFFFFF;
                border-radius: 14px;
                box-shadow: 0 16px 44px rgba(0, 0, 0, 0.28), 0 0 0 1.5px rgba(183, 110, 121, 0.45);
                z-index: 9999999 !important;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                resize: both;
            }}
            #chem-floating-scratchpad-root.minimized {{
                height: 44px !important;
                width: 250px !important;
                min-height: 44px !important;
                resize: none !important;
                box-shadow: 0 8px 24px rgba(91, 43, 61, 0.38);
            }}
            #chem-scratchpad-header {{
                background: linear-gradient(135deg, #3A1C28 0%, #5B2B3D 50%, #823E54 100%);
                color: #FFFFFF;
                padding: 10px 14px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                cursor: grab;
                user-select: none;
                border-top-left-radius: 14px;
                border-top-right-radius: 14px;
                transition: box-shadow 0.2s ease;
            }}
            #chem-scratchpad-header:active {{
                cursor: grabbing;
            }}
            .chem-scratchpad-title {{
                font-size: 0.88rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 7px;
                color: #FFFFFF;
                letter-spacing: 0.01em;
            }}
            .chem-drag-handle {{
                font-size: 1.05rem;
                opacity: 0.8;
                cursor: grab;
                transition: opacity 0.15s ease;
            }}
            .chem-drag-handle:hover {{
                opacity: 1;
            }}
            .chem-scratchpad-actions {{
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            .chem-pad-btn {{
                background: rgba(255, 255, 255, 0.18);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: #FFFFFF;
                border-radius: 6px;
                width: 26px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.82rem;
                cursor: pointer;
                transition: all 0.15s ease;
                line-height: 1;
            }}
            .chem-pad-btn:hover {{
                background: rgba(255, 255, 255, 0.35);
                transform: scale(1.05);
            }}
            #chem-scratchpad-body {{
                flex: 1;
                display: flex;
                flex-direction: column;
                padding: 10px 12px;
                background: #F8FAFC;
                overflow: hidden;
            }}
            #chem-floating-scratchpad-root.minimized #chem-scratchpad-body {{
                display: none !important;
            }}
            #chem-scratchpad-textarea {{
                flex: 1;
                width: 100%;
                border: 1px solid #CBD5E1;
                border-radius: 8px;
                padding: 10px 12px;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                font-size: 0.83rem;
                line-height: 1.55;
                color: #0F172A;
                background: #FFFFFF;
                outline: none;
                resize: none;
                box-sizing: border-box;
                transition: border-color 0.15s ease, box-shadow 0.15s ease;
            }}
            #chem-scratchpad-textarea:focus {{
                border-color: #B76E79;
                box-shadow: 0 0 0 2px rgba(183, 110, 121, 0.2);
            }}
            #chem-scratchpad-footer {{
                padding-top: 8px;
                display: flex;
                flex-direction: column;
                gap: 6px;
            }}
            .chem-pad-meta {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.74rem;
                color: #64748B;
            }}
            .chem-pad-toolbar {{
                display: flex;
                gap: 6px;
                flex-wrap: wrap;
            }}
            .chem-tool-btn {{
                background: #FFFFFF;
                border: 1px solid #CBD5E1;
                color: #334155;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 0.74rem;
                font-weight: 600;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 4px;
                transition: all 0.15s ease;
            }}
            .chem-tool-btn:hover {{
                background: #F1F5F9;
                border-color: #94A3B8;
                color: #0F172A;
            }}
            .chem-tool-btn.primary {{
                background: #0284C7;
                border-color: #0284C7;
                color: #FFFFFF;
            }}
            .chem-tool-btn.primary:hover {{
                background: #0369A1;
            }}
            .chem-tool-btn.danger:hover {{
                background: #FEE2E2;
                border-color: #EF4444;
                color: #B91C1C;
            }}
        `;
        doc.head.appendChild(styleEl);
    }}

    container = doc.createElement('div');
    container.id = 'chem-floating-scratchpad-root';

    // Position restoration
    const savedPos = localStorage.getItem('chem_scratchpad_pos');
    if (savedPos && !ResetPos) {{
        try {{
            const pos = JSON.parse(savedPos);
            const winW = (doc.defaultView || window).innerWidth || 1200;
            const winH = (doc.defaultView || window).innerHeight || 800;
            if (pos.left !== undefined && pos.top !== undefined) {{
                const left = Math.max(10, Math.min(winW - 300, pos.left));
                const top = Math.max(10, Math.min(winH - 80, pos.top));
                container.style.left = `${{left}}px`;
                container.style.top = `${{top}}px`;
                container.style.bottom = 'auto';
                container.style.right = 'auto';
            }}
        }} catch (e) {{}}
    }}

    // Dimension restoration
    const savedDim = localStorage.getItem('chem_scratchpad_dim');
    if (savedDim && !ResetPos) {{
        try {{
            const dim = JSON.parse(savedDim);
            if (dim.width && dim.height) {{
                container.style.width = `${{Math.min(window.innerWidth - 40, Math.max(280, dim.width))}}px`;
                container.style.height = `${{Math.min(window.innerHeight - 40, Math.max(200, dim.height))}}px`;
            }}
        }} catch(e) {{}}
    }}

    const wasMinimized = !ResetPos && localStorage.getItem('chem_scratchpad_minimized') === 'true';
    if (wasMinimized) {{
        container.classList.add('minimized');
    }}

    container.innerHTML = `
        <div id="chem-scratchpad-header" title="Drag to move anywhere on screen • Double-click to collapse/expand">
            <div class="chem-scratchpad-title">
                <span class="chem-drag-handle" title="Drag anywhere">⠿</span>
                <span>📝</span>
                <span id="chem-title-text">Floating Scratchpad</span>
            </div>
            <div class="chem-scratchpad-actions">
                <button class="chem-pad-btn" id="chem-pad-min-btn" title="Minimize / Restore">${{wasMinimized ? '▢' : '_'}}</button>
            </div>
        </div>
        <div id="chem-scratchpad-body">
            <textarea id="chem-scratchpad-textarea" placeholder="Paste literature excerpts, reaction yields, IC50 data, or type research notes here..."></textarea>
            <div id="chem-scratchpad-footer">
                <div class="chem-pad-meta">
                    <span id="chem-scratchpad-counts">0 words • 0 chars</span>
                    <span style="color: #10B981; font-weight: 600;">● Auto-saved</span>
                </div>
                <div class="chem-pad-toolbar">
                    <button class="chem-tool-btn primary" id="chem-pad-dl-md" title="Download formatted Markdown (.md)">💾 .md</button>
                    <button class="chem-tool-btn" id="chem-pad-dl-txt" title="Download plain text (.txt)">💾 .txt</button>
                    <button class="chem-tool-btn" id="chem-pad-time" title="Insert current timestamp">⏱️ Time</button>
                    <button class="chem-tool-btn" id="chem-pad-copy" title="Copy all notes to clipboard">📋 Copy</button>
                    <button class="chem-tool-btn" id="chem-pad-dock" title="Re-dock to bottom-right corner">📍 Dock</button>
                    <button class="chem-tool-btn danger" id="chem-pad-clear" title="Clear notes">🧹 Clear</button>
                </div>
            </div>
        </div>
    `;

    doc.body.appendChild(container);

    // Fallback if doc is iframe itself
    if (doc === document && window.frameElement) {{
        window.frameElement.style.position = 'fixed';
        window.frameElement.style.bottom = '24px';
        window.frameElement.style.right = '28px';
        window.frameElement.style.zIndex = '9999999';
        window.frameElement.style.width = '380px';
        window.frameElement.style.height = '420px';
        window.frameElement.style.border = 'none';
        window.frameElement.style.background = 'transparent';
    }}

    const textarea = doc.getElementById('chem-scratchpad-textarea');
    const counts = doc.getElementById('chem-scratchpad-counts');
    const minBtn = doc.getElementById('chem-pad-min-btn');
    const header = doc.getElementById('chem-scratchpad-header');

    // Restore text
    let savedNotes = localStorage.getItem('chem_floating_notes') || "";
    if (ClearNotes) {{
        savedNotes = "";
        localStorage.setItem('chem_floating_notes', "");
    }} else if (PinnedFromStreamlit && PinnedFromStreamlit.trim()) {{
        const sections = PinnedFromStreamlit.split(/\\n\\s*---\\s*\\n/);
        for (let sec of sections) {{
            sec = sec.trim();
            if (sec && !savedNotes.includes(sec)) {{
                savedNotes = (savedNotes.trim() ? savedNotes.trim() + "\\n\\n---\\n" : "") + sec;
            }}
        }}
        localStorage.setItem('chem_floating_notes', savedNotes);
    }}
    textarea.value = savedNotes;
    updateCounts();

    function updateCounts() {{
        const text = textarea.value;
        const wc = text.trim() ? text.trim().split(/\\s+/).length : 0;
        const cc = text.length;
        if (counts) counts.innerText = `${{wc}} words • ${{cc}} chars`;
        const titleText = doc.getElementById('chem-title-text');
        if (container.classList.contains('minimized') && titleText) {{
            titleText.innerText = `Notes (${{wc}}w)`;
        }} else if (titleText) {{
            titleText.innerText = `Floating Scratchpad (${{wc}}w)`;
        }}
    }}

    textarea.addEventListener('input', function() {{
        localStorage.setItem('chem_floating_notes', textarea.value);
        updateCounts();
    }});

    // Minimize toggle
    function toggleMin(e) {{
        if (e) e.stopPropagation();
        const isMin = container.classList.toggle('minimized');
        localStorage.setItem('chem_scratchpad_minimized', isMin ? 'true' : 'false');
        minBtn.innerText = isMin ? '▢' : '_';
        minBtn.title = isMin ? 'Expand Floating Scratchpad' : 'Minimize to Floating Pill';
        updateCounts();
    }}
    minBtn.addEventListener('click', toggleMin);
    header.addEventListener('dblclick', toggleMin);

    // Draggable logic (mouse + touch)
    let isDragging = false;
    let dragStartX = 0, dragStartY = 0, elemStartX = 0, elemStartY = 0;

    function startDrag(clientX, clientY) {{
        isDragging = true;
        const rect = container.getBoundingClientRect();
        elemStartX = rect.left;
        elemStartY = rect.top;
        dragStartX = clientX;
        dragStartY = clientY;
        container.style.transition = 'none';
        header.style.cursor = 'grabbing';
        doc.body.style.userSelect = 'none';
    }}

    function doDrag(clientX, clientY) {{
        if (!isDragging) return;
        const deltaX = clientX - dragStartX;
        const deltaY = clientY - dragStartY;
        let newX = elemStartX + deltaX;
        let newY = elemStartY + deltaY;

        const winW = (doc.defaultView || window).innerWidth || 1200;
        const winH = (doc.defaultView || window).innerHeight || 800;
        const maxX = Math.max(0, winW - container.offsetWidth - 10);
        const maxY = Math.max(0, winH - container.offsetHeight - 10);
        newX = Math.max(10, Math.min(maxX, newX));
        newY = Math.max(10, Math.min(maxY, newY));

        container.style.left = `${{newX}}px`;
        container.style.top = `${{newY}}px`;
        container.style.bottom = 'auto';
        container.style.right = 'auto';
    }}

    function endDrag() {{
        if (isDragging) {{
            isDragging = false;
            header.style.cursor = 'grab';
            doc.body.style.userSelect = '';
            container.style.transition = '';
            const rect = container.getBoundingClientRect();
            try {{
                localStorage.setItem('chem_scratchpad_pos', JSON.stringify({{left: rect.left, top: rect.top}}));
            }} catch(e) {{}}
            if (!container.classList.contains('minimized')) {{
                const currentW = container.offsetWidth;
                const currentH = container.offsetHeight;
                if (currentW > 0 && currentH > 0) {{
                    localStorage.setItem('chem_scratchpad_dim', JSON.stringify({{width: currentW, height: currentH}}));
                }}
            }}
        }}
    }}

    header.addEventListener('mousedown', function(e) {{
        if (e.target.closest('button')) return;
        startDrag(e.clientX, e.clientY);
        e.preventDefault();
    }});

    doc.addEventListener('mousemove', function(e) {{
        if (isDragging) {{
            doDrag(e.clientX, e.clientY);
        }}
    }});

    doc.addEventListener('mouseup', function() {{
        endDrag();
    }});

    header.addEventListener('touchstart', function(e) {{
        if (e.target.closest('button')) return;
        const touch = e.touches[0];
        startDrag(touch.clientX, touch.clientY);
    }}, {{passive: false}});

    doc.addEventListener('touchmove', function(e) {{
        if (isDragging) {{
            const touch = e.touches[0];
            doDrag(touch.clientX, touch.clientY);
            e.preventDefault();
        }}
    }}, {{passive: false}});

    doc.addEventListener('touchend', function() {{
        endDrag();
    }});

    // Downloads
    function downloadFile(content, filename, type) {{
        const blob = new Blob([content], {{type: `${{type}};charset=utf-8`}});
        const url = URL.createObjectURL(blob);
        const a = doc.createElement('a');
        a.href = url;
        a.download = filename;
        doc.body.appendChild(a);
        a.click();
        doc.body.removeChild(a);
        URL.revokeObjectURL(url);
    }}

    doc.getElementById('chem-pad-dl-md').addEventListener('click', function() {{
        const dateStr = new Date().toISOString().slice(0, 10);
        downloadFile(textarea.value, `chem_research_notes_${{dateStr}}.md`, 'text/markdown');
    }});

    doc.getElementById('chem-pad-dl-txt').addEventListener('click', function() {{
        const dateStr = new Date().toISOString().slice(0, 10);
        downloadFile(textarea.value, `chem_research_notes_${{dateStr}}.txt`, 'text/plain');
    }});

    // Timestamp
    doc.getElementById('chem-pad-time').addEventListener('click', function() {{
        const now = new Date();
        const dateStr = now.toISOString().replace('T', ' ').substring(0, 16);
        const stamp = `\\n\\n#### ⏱️ [${{dateStr}}] — Notes:\\n`;
        const start = textarea.selectionStart || textarea.value.length;
        textarea.value = textarea.value.substring(0, start) + stamp + textarea.value.substring(start);
        localStorage.setItem('chem_floating_notes', textarea.value);
        updateCounts();
    }});

    // Copy
    doc.getElementById('chem-pad-copy').addEventListener('click', function() {{
        navigator.clipboard.writeText(textarea.value).then(() => {{
            const btn = doc.getElementById('chem-pad-copy');
            const orig = btn.innerText;
            btn.innerText = '✅ Copied!';
            setTimeout(() => {{ btn.innerText = orig; }}, 1800);
        }});
    }});

    // Re-dock
    doc.getElementById('chem-pad-dock').addEventListener('click', function() {{
        container.style.left = 'auto';
        container.style.top = 'auto';
        container.style.bottom = '96px';
        container.style.right = '28px';
        localStorage.removeItem('chem_scratchpad_pos');
    }});

    // Window resize auto-clamping
    (doc.defaultView || window).addEventListener('resize', function() {{
        const winW = (doc.defaultView || window).innerWidth || 1200;
        const winH = (doc.defaultView || window).innerHeight || 800;
        const rect = container.getBoundingClientRect();
        if (rect.left > winW - 60 || rect.top > winH - 60) {{
            const newX = Math.max(10, Math.min(winW - container.offsetWidth - 10, rect.left));
            const newY = Math.max(10, Math.min(winH - container.offsetHeight - 10, rect.top));
            container.style.left = `${{newX}}px`;
            container.style.top = `${{newY}}px`;
        }}
    }});

    // Clear
    doc.getElementById('chem-pad-clear').addEventListener('click', function() {{
        if (confirm('Are you sure you want to clear your scratchpad notes?')) {{
            textarea.value = '';
            localStorage.setItem('chem_floating_notes', '');
            updateCounts();
        }}
    }});
}})();
</script>
</body>
</html>"""
    components.html(html_code, height=0, width=0)


# ---------------------------------------------------------
# Floating Round ChemBot Popup Component
# ---------------------------------------------------------
def render_floating_round_chatbot(
    pages: Optional[List[PageContent]] = None,
    engine: Optional[Any] = None,
    analysis: Optional[PaperAnalysisResult] = None,
    api_key_input: Optional[str] = None,
    key_prefix: str = "fab_chembot"
):
    """
    Renders the Floating Round Chatbot Icon in the bottom-right corner.
    When touched, opens a friendly popup greeting: "How can I help you buddy? 😊"
    Gives a proper, simple answer in the user's language, with an optional "Go In-Depth" action.
    """
    import os
    from core.multilingual_bot import MultilingualChemBot

    if "chembot_chat_history" not in st.session_state:
        st.session_state.chembot_chat_history = []
    if "chembot_expanded_in_depth" not in st.session_state:
        st.session_state.chembot_expanded_in_depth = set()

    # CSS for Floating Round Launcher Button & Popover Window
    st.markdown("""
    <style>
    /* Floating Round Launcher Button in Bottom-Right Corner */
    div.st-key-chembot_fab_container {
        position: fixed !important;
        bottom: 24px !important;
        right: 28px !important;
        z-index: 999998 !important;
        width: auto !important;
        height: auto !important;
    }
    div.st-key-chembot_fab_container div[data-testid="stPopover"] > button {
        width: 60px !important;
        height: 60px !important;
        min-width: 60px !important;
        min-height: 60px !important;
        max-width: 60px !important;
        max-height: 60px !important;
        border-radius: 50% !important;
        background: #18181B !important;
        color: #FFFFFF !important;
        border: 2px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45), 0 2px 8px rgba(0, 0, 0, 0.25) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        cursor: pointer !important;
        transition: transform 0.22s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.22s ease, background 0.2s ease !important;
    }
    div.st-key-chembot_fab_container div[data-testid="stPopover"] > button:hover {
        transform: scale(1.08) !important;
        background: #27272A !important;
        box-shadow: 0 14px 38px rgba(0, 0, 0, 0.6) !important;
    }
    div.st-key-chembot_fab_container div[data-testid="stPopover"] > button:active {
        transform: scale(0.95) !important;
    }
    div.st-key-chembot_fab_container div[data-testid="stPopover"] > button span[data-testid="stIconMaterial"] {
        font-size: 30px !important;
        color: #FFFFFF !important;
        line-height: 1 !important;
        margin: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div.st-key-chembot_fab_container div[data-testid="stPopover"] > button p {
        display: none !important;
    }
    div.st-key-chembot_fab_container div[data-testid="stPopover"] > button svg {
        display: none !important;
    }
    div.st-key-chembot_fab_container div[data-testid="stPopover"] > button span[data-testid="stIconMaterial"] svg {
        display: inline-block !important;
    }

    /* Popover Body Window */
    div[data-testid="stPopoverBody"]:has(.chembot-fab-marker) {
        width: 420px !important;
        max-width: calc(100vw - 36px) !important;
        max-height: 580px !important;
        border-radius: 20px !important;
        border: 1px solid rgba(183, 110, 121, 0.35) !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3) !important;
        background: #FFFFFF !important;
        padding: 16px 18px !important;
        overflow-y: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container(key="chembot_fab_container"):
        with st.popover("", icon=":material/forum:", help="How can I help you buddy? 😊"):
            st.markdown('<div class="chembot-fab-marker"></div>', unsafe_allow_html=True)

            # Header
            has_paper = bool(analysis and getattr(analysis, "metadata", None) and analysis.metadata.title)
            doc_label = (analysis.metadata.title[:18] + "...") if has_paper else "Open Chemistry"

            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(183, 110, 121, 0.2); padding-bottom: 10px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, #5B2B3D, #B76E79); display: flex; align-items: center; justify-content: center; font-size: 1.25rem; box-shadow: 0 2px 8px rgba(183,110,121,0.3);">
                        🤖
                    </div>
                    <div>
                        <div style="font-weight: 800; color: #5B2B3D; font-size: 1.05rem; line-height: 1.2;">ChemBot</div>
                        <div style="font-size: 0.76rem; color: #8F2E44; font-weight: 500;">Your Chemistry Buddy 😊</div>
                    </div>
                </div>
                <span style="background: rgba(183, 110, 121, 0.14); color: #8F2E44; font-size: 0.72rem; font-weight: 700; padding: 3px 8px; border-radius: 999px;">
                    {html.escape(doc_label)}
                </span>
            </div>
            """, unsafe_allow_html=True)

            clicked_starter = None

            # Empty State: Friendly Greeting
            if not st.session_state.chembot_chat_history:
                st.markdown("""
                <div style="background: #FDF2F0; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; border: 1px solid rgba(183, 110, 121, 0.25);">
                    <div style="font-weight: 800; color: #5B2B3D; font-size: 1.12rem; margin-bottom: 4px;">
                        How can I help you buddy? 😊
                    </div>
                    <div style="font-size: 0.85rem; color: #6C4C54; line-height: 1.45;">
                        Ask me any question in your language. I will give you a simple, clear answer! If you need in-depth scientific details, just ask or click <i>🔍 Explain in depth</i>.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #8F2E44; margin-bottom: 6px;'>💡 Quick Questions:</div>", unsafe_allow_html=True)
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    if st.button("📄 Simple Summary", key=f"{key_prefix}_start_sum", use_container_width=True):
                        clicked_starter = "Explain this paper in simple words"
                with col_s2:
                    if st.button("🏆 Lead Molecule", key=f"{key_prefix}_start_lead", use_container_width=True):
                        clicked_starter = "Which compound is the most active and why?"
                with col_s3:
                    if st.button("❓ What is IC50?", key=f"{key_prefix}_start_ic50", use_container_width=True):
                        clicked_starter = "Explain what IC50 means in simple words"

            # Chat Message Stream
            else:
                with st.container(height=310):
                    for idx, item in enumerate(st.session_state.chembot_chat_history):
                        # User message
                        with st.chat_message("user"):
                            st.markdown(f"**{item['question']}**")

                        # Assistant message
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(item["simple_answer"])

                            is_depth = idx in st.session_state.chembot_expanded_in_depth
                            if not is_depth:
                                col_b1, col_b2 = st.columns([1.5, 1])
                                with col_b1:
                                    if st.button("🔍 Explain in depth", key=f"{key_prefix}_btn_depth_{idx}", help="View detailed mechanism, exact numbers, and citations"):
                                        st.session_state.chembot_expanded_in_depth.add(idx)
                                        st.rerun()
                                parse_lang = item.get("language", "")
                                with col_b2:
                                    if st.button("📌 Pin", key=f"{key_prefix}_btn_pin_{idx}", help="Save to Floating Scratchpad"):
                                        pin_text = f"\n\n---\n#### 🤖 ChemBot: {item['question']}\n{item['simple_answer']}\n"
                                        st.session_state.researcher_notes = (
                                            st.session_state.get("researcher_notes", "") + pin_text
                                        )
                                        st.session_state.last_pinned_note = pin_text
                                        st.toast("📌 Saved to Notes!", icon="📝")
                                        st.rerun()
                            else:
                                in_depth_text = item.get("in_depth_markdown") or item.get("full_text", "")
                                st.markdown(f"""
                                <div style="background: rgba(253, 242, 240, 0.7); border: 1px solid rgba(183, 110, 121, 0.3); border-radius: 10px; padding: 12px 14px; margin-top: 8px; margin-bottom: 6px; font-size: 0.85rem; color: #2D1D22;">
                                    <b style="color: #8F2E44; font-size: 0.9rem;">🔬 In-Depth Scientific Breakdown:</b><br>
                                    {in_depth_text}
                                </div>
                                """, unsafe_allow_html=True)
                                col_c1, col_c2 = st.columns([1.5, 1])
                                with col_c1:
                                    if st.button("🔼 Show less", key=f"{key_prefix}_btn_less_{idx}"):
                                        st.session_state.chembot_expanded_in_depth.discard(idx)
                                        st.rerun()
                                with col_c2:
                                    if st.button("📌 Pin all", key=f"{key_prefix}_btn_pinall_{idx}", help="Save in-depth notes to Floating Scratchpad"):
                                        pin_text = f"\n\n---\n#### 🤖 ChemBot: {item['question']}\n{item['simple_answer']}\n\n*In-Depth Details:*\n{in_depth_text}\n"
                                        st.session_state.researcher_notes = (
                                            st.session_state.get("researcher_notes", "") + pin_text
                                        )
                                        st.session_state.last_pinned_note = pin_text
                                        st.toast("📌 Saved in-depth notes!", icon="📝")
                                        st.rerun()

            # Chat Input
            chat_input_val = st.chat_input("Ask a question buddy...", key=f"{key_prefix}_chat_input")
            active_q = clicked_starter or chat_input_val

            if active_q:
                clean_q = active_q.strip().lower()
                is_depth_keyword = any(w in clean_q for w in [
                    "in depth", "in-depth", "go deep", "more detail", "more details", "tell me more",
                    "explain deeper", "deep", "കൂടുതൽ", "വിശദീകരിക്കാമോ", "गहराई"
                ])
                words = clean_q.split()
                is_followup_depth = is_depth_keyword and len(words) <= 4 and bool(st.session_state.chembot_chat_history)

                if is_followup_depth:
                    last_idx = len(st.session_state.chembot_chat_history) - 1
                    st.session_state.chembot_expanded_in_depth.add(last_idx)
                    st.rerun()
                else:
                    active_key = api_key_input or os.getenv("GEMINI_API_KEY")
                    bot = MultilingualChemBot(api_key=active_key)
                    with st.spinner("Thinking simply in your language..."):
                        bot_resp: MultilingualBotResponse = bot.answer(
                            query=active_q,
                            analysis=analysis,
                            pages=pages,
                            language_choice="Auto-Detect",
                            simplicity_level="Simple & Everyday"
                        )
                        new_idx = len(st.session_state.chembot_chat_history)
                        st.session_state.chembot_chat_history.append({
                            "question": active_q,
                            "simple_answer": bot_resp.simple_answer,
                            "in_depth_markdown": bot_resp.in_depth_markdown,
                            "full_text": bot_resp.full_formatted_text,
                            "language": bot_resp.detected_language
                        })
                        if is_depth_keyword:
                            st.session_state.chembot_expanded_in_depth.add(new_idx)
                    st.rerun()

            # Footer: Clear chat
            if st.session_state.chembot_chat_history:
                st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Clear Chat", key=f"{key_prefix}_clear_btn", help="Reset conversation"):
                    st.session_state.chembot_chat_history = []
                    st.session_state.chembot_expanded_in_depth = set()
                    st.rerun()


# Maintain alias for compatibility
render_multilingual_chatbot_ui = render_floating_round_chatbot


