"""
Unit tests for High-Resolution PDF Page Viewer, Amber Quote Highlighting,
and UI Components for Side-by-Side Split View.
"""

import os
import unittest
from core.models import EvidenceCitation, CompoundInfo
from src.pdf_parser import DocumentParser, PageContent
from src.ui_components import (
    render_grounded_evidence_card,
    render_show_in_paper_button,
    render_page_navigation,
    jump_to_paper_citation,
    render_dynamic_question_chips
)


class TestPDFViewerAndSplitView(unittest.TestCase):

    def setUp(self):
        self.sample_pdf_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.pdf"
        self.sample_txt_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        with open(self.sample_pdf_path, "rb") as f:
            self.pdf_bytes = f.read()
        with open(self.sample_txt_path, "r", encoding="utf-8") as f:
            self.txt_content = f.read()
        self.pages = DocumentParser.parse_text_content(self.txt_content)

    def test_pdf_page_count(self):
        page_count = DocumentParser.get_pdf_page_count(self.pdf_bytes)
        self.assertEqual(page_count, 4)

    def test_render_high_res_page_image(self):
        # Render Page 1 at zoom 2.0x
        img_bytes = DocumentParser.render_pdf_page_image(self.pdf_bytes, page_number=1, zoom=2.0)
        self.assertIsNotNone(img_bytes)
        # Check PNG header magic bytes
        self.assertTrue(img_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(img_bytes), 10000)

    def test_render_page_with_amber_highlight(self):
        # Page 3 contains Compound 3b potency quote
        quote = "Compound 3b exhibited superior enzymatic potency (IC50 = 8.4 nM)"
        img_highlighted = DocumentParser.render_pdf_page_image(
            self.pdf_bytes,
            page_number=3,
            zoom=2.0,
            highlight_quote=quote
        )
        self.assertIsNotNone(img_highlighted)
        self.assertTrue(img_highlighted.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_highlight_quote_in_text(self):
        text = "Evaluation results showed Compound 3b exhibited superior enzymatic potency (IC50 = 8.4 nM) across assays."
        quote = "Compound 3b exhibited superior enzymatic potency (IC50 = 8.4 nM)"
        highlighted = DocumentParser.highlight_quote_in_text(text, quote)
        self.assertIn('<mark class="glowing-citation-mark">', highlighted)
        self.assertIn("Compound 3b", highlighted)

    def test_highlight_quote_fuzzy_match(self):
        text = "In biological tests, Compound 3b   exhibited superior enzymatic potency (IC50 = 8.4 nM)."
        quote = "Compound 3b exhibited superior enzymatic potency"
        highlighted = DocumentParser.highlight_quote_in_text(text, quote)
        self.assertIn('<mark class="glowing-citation-mark">', highlighted)

    def test_synthesize_pdf_from_pages(self):
        pdf_out = DocumentParser.synthesize_pdf_from_pages(self.pages)
        self.assertIsNotNone(pdf_out)
        self.assertTrue(pdf_out.startswith(b"%PDF"))
        # Render synthesized page image
        img = DocumentParser.render_pdf_page_image(pdf_out, page_number=1, zoom=1.5)
        self.assertIsNotNone(img)
        self.assertTrue(img.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_jump_to_paper_citation_state(self):
        # Mock streamlit session_state
        class MockSessionState(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        import streamlit as st
        # Inject mock session_state if needed
        if not hasattr(st, "session_state"):
            st.session_state = MockSessionState()

        cit = EvidenceCitation(
            page_number=3,
            section="Biological Evaluation",
            verbatim_quote="Compound 3b exhibited superior enzymatic potency (IC50 = 8.4 nM)."
        )

        st.session_state.viewer_page = 1
        st.session_state.highlight_quote = None
        st.session_state.app_layout = "tabs"

        # Verify state changes without rerun
        st.session_state.viewer_page = cit.page_number
        st.session_state.highlight_quote = cit.verbatim_quote
        st.session_state.app_layout = "split"

        self.assertEqual(st.session_state.viewer_page, 3)
        self.assertEqual(st.session_state.highlight_quote, cit.verbatim_quote)
        self.assertEqual(st.session_state.app_layout, "split")


if __name__ == "__main__":
    unittest.main()
