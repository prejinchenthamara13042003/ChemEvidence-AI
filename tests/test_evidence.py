"""
Unit tests for EvidenceEngine (core/evidence_engine.py).
"""

import unittest
from core.pdf_parser import DocumentParser
from core.evidence_engine import EvidenceEngine


class TestEvidenceEngine(unittest.TestCase):

    def setUp(self):
        txt_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.pages = DocumentParser.parse_text_content(content)
        self.engine = EvidenceEngine(self.pages)

    def test_indexing_chunks(self):
        self.assertGreater(len(self.engine.chunks), 2)
        # Verify chunks have valid page numbers
        for c in self.engine.chunks:
            self.assertGreaterEqual(c.page_number, 1)
            self.assertLessEqual(c.page_number, len(self.pages))

    def test_search_retrieval(self):
        results = self.engine.search("yield of compound 3b microwave", top_k=3)
        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertIn("3b", top_chunk.text)
        self.assertGreater(score, 0.5)

    def test_verify_quote_exact(self):
        quote = "off-white crystalline solid in 82% isolated yield"
        cit = self.engine.verify_quote(quote)
        self.assertIsNotNone(cit)
        self.assertEqual(cit.page_number, 2)
        self.assertEqual(cit.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
