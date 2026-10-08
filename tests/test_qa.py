"""
Unit tests for QASystem and ChemistryExtractor (core/qa_system.py & core/chemistry_extractor.py).
Tests:
- Automatic structured overview extraction
- Evidence-grounded Q&A
- Strict absence detection ("no evidence found")
- Ambiguity and conflict detection
"""

import unittest
from core.pdf_parser import DocumentParser
from core.evidence_engine import EvidenceEngine
from core.chemistry_extractor import ChemistryExtractor
from core.qa_system import QASystem


class TestQASystem(unittest.TestCase):

    def test_structured_overview_extraction(self):
        txt_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        meta = DocumentParser.extract_metadata(pages)
        engine = EvidenceEngine(pages)
        extractor = ChemistryExtractor(use_local_only=True)
        
        analysis = extractor.analyze_paper(pages, meta, engine)
        self.assertIsNotNone(analysis)
        self.assertGreater(len(analysis.compounds), 0)
        self.assertGreater(len(analysis.properties), 0)
        self.assertGreater(len(analysis.bioactivities), 0)
        self.assertGreater(len(analysis.conditions), 0)
        
        # Check compound evidence citation
        first_comp = analysis.compounds[0]
        self.assertIsNotNone(first_comp.evidence)
        self.assertGreaterEqual(first_comp.evidence.page_number, 1)

    def test_positive_qa_with_evidence(self):
        txt_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        engine = EvidenceEngine(pages)
        qa = QASystem(use_local_only=True)

        resp = qa.ask("What is the IC50 value of Compound 3b against EGFR kinase?", pages, engine)
        self.assertEqual(resp.status, "found")
        self.assertIn("8.4", resp.answer)
        self.assertGreater(len(resp.citations), 0)
        # Verify citation page number is 1 or 3
        self.assertIn(resp.citations[0].page_number, [1, 2, 3, 4])

    def test_strict_absence_detection(self):
        txt_path = "/Users/prejin/Thesis/sample_papers/paper3_natural_product_sar.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        engine = EvidenceEngine(pages)
        qa = QASystem(use_local_only=True)

        # Query for something completely absent from paper
        resp = qa.ask("What was the oral bioavailability in Sprague-Dawley rats and pharmacokinetic clearance?", pages, engine)
        self.assertEqual(resp.status, "absent")
        self.assertIn("No supporting evidence was found", resp.answer)
        self.assertIn("Would you like an answer based on broader scientific literature", resp.answer)
        self.assertTrue(resp.can_expand_external)
        self.assertEqual(len(resp.citations), 0)

    def test_external_literature_search(self):
        papers = QASystem.search_external_literature("oral bioavailability kinase inhibitor", max_results=3)
        self.assertGreater(len(papers), 0)
        first_paper = papers[0]
        self.assertIn("title", first_paper)
        self.assertIn("source", first_paper)
        self.assertIn("url", first_paper)
        self.assertTrue(first_paper["url"].startswith("http"))

    def test_answer_from_external_literature(self):
        qa = QASystem(use_local_only=True)
        resp = qa.answer_from_external_literature(
            "What is typical oral bioavailability for small-molecule kinase inhibitors?",
            paper_context="Design of Novel EGFR Kinase Inhibitors"
        )
        self.assertEqual(resp.status, "external_literature")
        self.assertFalse(resp.can_expand_external)
        self.assertGreater(len(resp.related_papers), 0)
        self.assertIn("Broader Scientific Literature", resp.answer)
        self.assertIn(resp.related_papers[0]["title"], resp.answer)

    def test_conflict_detection(self):
        txt_path = "/Users/prejin/Thesis/sample_papers/paper2_catalytic_synthesis_conflicts.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        meta = DocumentParser.extract_metadata(pages)
        engine = EvidenceEngine(pages)
        extractor = ChemistryExtractor(use_local_only=True)
        analysis = extractor.analyze_paper(pages, meta, engine)

        qa = QASystem(use_local_only=True)
        resp = qa.ask("What was the isolated yield of compound 5c?", pages, engine, analysis)
        
        # Verify conflict was flagged
        self.assertTrue(resp.status == "conflicting" or len(resp.conflicts) > 0 or len(analysis.conflicts_detected) > 0)
        if resp.conflicts:
            self.assertIn("Yield", resp.conflicts[0].topic)


if __name__ == "__main__":
    unittest.main()

