"""
Unit tests for DocumentParser (core/pdf_parser.py).
"""

import os
import unittest
from core.pdf_parser import DocumentParser, PageContent


class TestDocumentParser(unittest.TestCase):

    def setUp(self):
        self.sample_txt = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        self.sample_pdf = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.pdf"

    def test_parse_text_file(self):
        with open(self.sample_txt, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        self.assertGreaterEqual(len(pages), 3)
        self.assertEqual(pages[0].page_number, 1)
        self.assertIn("EGFR", pages[0].text)

    def test_parse_pdf_file(self):
        with open(self.sample_pdf, "rb") as f:
            pdf_bytes = f.read()
        pages = DocumentParser.parse_pdf_bytes(pdf_bytes)
        self.assertGreaterEqual(len(pages), 2)
        # Check that page text was extracted
        full_text = " ".join([p.text for p in pages])
        self.assertIn("Compound 3b", full_text)

    def test_extract_metadata(self):
        with open(self.sample_txt, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        meta = DocumentParser.extract_metadata(pages)
        self.assertIn("2,4-Diaminopyrimidine", meta.title)
        self.assertGreater(meta.total_words, 100)
        self.assertEqual(meta.page_count, len(pages))


if __name__ == "__main__":
    unittest.main()
