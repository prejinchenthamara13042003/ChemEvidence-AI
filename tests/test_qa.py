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

    def test_metric_type_and_cell_line_disambiguation(self):
        """Verifies that enzyme IC50 (EGFR) is never confused with cellular growth inhibition (A549)."""
        txt_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        meta = DocumentParser.extract_metadata(pages)
        engine = EvidenceEngine(pages)
        extractor = ChemistryExtractor(use_local_only=True)
        analysis = extractor.analyze_paper(pages, meta, engine)
        qa = QASystem(use_local_only=True)

        # 1. Query for enzymatic EGFR IC50
        resp_egfr = qa.ask("What is the IC50 value of Compound 3b against EGFR kinase?", pages, engine, analysis)
        self.assertEqual(resp_egfr.status, "found")
        self.assertIn("8.4", resp_egfr.answer)
        self.assertIn("nM", resp_egfr.answer)

        # 2. Query for cellular A549 activity
        resp_a549 = qa.ask("What is the antiproliferative IC50 of Compound 3b in A549 cells?", pages, engine, analysis)
        self.assertEqual(resp_a549.status, "found")
        self.assertIn("0.62", resp_a549.answer)
        self.assertIn("A549", resp_a549.answer)

    def test_target_absence_detection(self):
        """Verifies that querying for an unmeasured target (HER2) triggers strict absence detection."""
        txt_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        meta = DocumentParser.extract_metadata(pages)
        engine = EvidenceEngine(pages)
        extractor = ChemistryExtractor(use_local_only=True)
        analysis = extractor.analyze_paper(pages, meta, engine)
        qa = QASystem(use_local_only=True)

        resp = qa.ask("What is the IC50 value of Compound 3b against HER2 kinase?", pages, engine, analysis)
        self.assertEqual(resp.status, "absent")
        self.assertIn("No supporting evidence was found", resp.answer)

    def test_paper3_antifungal_mic_extraction_and_qa(self):
        """Verifies that antimicrobial MIC values and fungal targets are correctly extracted and answered."""
        txt_path = "/Users/prejin/Thesis/sample_papers/paper3_natural_product_sar.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        meta = DocumentParser.extract_metadata(pages)
        engine = EvidenceEngine(pages)
        extractor = ChemistryExtractor(use_local_only=True)
        analysis = extractor.analyze_paper(pages, meta, engine)
        qa = QASystem(use_local_only=True)

        resp = qa.ask("What is the MIC of Compound 6b against Candida albicans?", pages, engine, analysis)
        self.assertEqual(resp.status, "found")
        self.assertIn("1.8", resp.answer)
        self.assertIn("Candida albicans", resp.answer)
        self.assertIn("MIC", resp.answer)
        # Ensure it does not report kinase or EGFR
        self.assertNotIn("EGFR", resp.answer)

    def test_authors_query(self):
        """Verifies that author-related questions correctly return the authors with Page 1 citation."""
        txt_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        meta = DocumentParser.extract_metadata(pages)
        engine = EvidenceEngine(pages)
        extractor = ChemistryExtractor(use_local_only=True)
        analysis = extractor.analyze_paper(pages, meta, engine)
        qa = QASystem(use_local_only=True)

        # Test query 1
        resp = qa.ask("who are the authors of the paper?", pages, engine, analysis)
        self.assertEqual(resp.status, "found")
        self.assertIn("Elena R. Vance", resp.answer)
        self.assertIn("Marcus K. Chen", resp.answer)
        self.assertIn("Aris Thorne", resp.answer)
        self.assertGreater(len(resp.citations), 0)
        self.assertEqual(resp.citations[0].page_number, 1)

        # Test query 2
        resp2 = qa.ask("who wrote this paper?", pages, engine, analysis)
        self.assertEqual(resp2.status, "found")
        self.assertIn("Elena R. Vance", resp2.answer)

    def test_summary_query(self):
        """Verifies that summary questions return a structured executive summary from the paper."""
        txt_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        meta = DocumentParser.extract_metadata(pages)
        engine = EvidenceEngine(pages)
        extractor = ChemistryExtractor(use_local_only=True)
        analysis = extractor.analyze_paper(pages, meta, engine)
        qa = QASystem(use_local_only=True)

        resp = qa.ask("summary of the paper", pages, engine, analysis)
        self.assertEqual(resp.status, "found")
        self.assertGreater(len(resp.citations), 0)
        # Contains key concepts from the paper
        self.assertTrue("kinase" in resp.answer.lower() or "inhibitor" in resp.answer.lower() or "synthesis" in resp.answer.lower())

        resp2 = qa.ask("what is this paper about?", pages, engine, analysis)
        self.assertEqual(resp2.status, "found")

    def test_conclusion_query(self):
        """Verifies that questions about conclusions return paper conclusions."""
        txt_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        meta = DocumentParser.extract_metadata(pages)
        engine = EvidenceEngine(pages)
        extractor = ChemistryExtractor(use_local_only=True)
        analysis = extractor.analyze_paper(pages, meta, engine)
        qa = QASystem(use_local_only=True)

        resp = qa.ask("what is the conclusion?", pages, engine, analysis)
        self.assertEqual(resp.status, "found")
        self.assertGreater(len(resp.citations), 0)

    def test_chemical_synonym_expansion_retrieval(self):
        """Verifies that queries using synonyms like 'synthetic efficiency' or 'antiproliferative potency' successfully retrieve relevant chunks."""
        txt_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        engine = EvidenceEngine(pages)
        
        # Test 1: 'synthetic efficiency' expands to yield/conversion
        results_yield = engine.search("What was the synthetic efficiency of compound 3b?", top_k=3)
        self.assertGreater(len(results_yield), 0)
        yield_text = " ".join([c.text.lower() for c, _ in results_yield])
        self.assertTrue("yield" in yield_text or "%" in yield_text or "3b" in yield_text)

        # Test 2: 'antiproliferative potency' expands to cytotoxicity/GI50
        results_pot = engine.search("antiproliferative potency in cancer cells", top_k=3)
        self.assertGreater(len(results_pot), 0)

    def test_structured_data_context_injection(self):
        """Verifies that _get_matching_structured_context extracts exact matching table rows and bioactivities."""
        txt_path = "/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pages = DocumentParser.parse_text_content(content)
        meta = DocumentParser.extract_metadata(pages)
        engine = EvidenceEngine(pages)
        extractor = ChemistryExtractor(use_local_only=True)
        analysis = extractor.analyze_paper(pages, meta, engine)
        qa = QASystem(use_local_only=True)

        structured_ctx = qa._get_matching_structured_context("What is the IC50 of compound 3b?", analysis)
        self.assertIn("3b", structured_ctx.lower())
        self.assertTrue("8.4" in structured_ctx or "ic50" in structured_ctx.lower())


if __name__ == "__main__":
    unittest.main()

