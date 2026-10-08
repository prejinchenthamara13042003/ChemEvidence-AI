"""
Unit tests for Advanced ChemEvidence AI Features:
- Table extraction
- Reaction pathway extraction
- Chemical entity normalization
- SAR analysis
- Knowledge graph construction
- Confidence scoring
- Research gap detection
- Multi-paper comparison
"""

import unittest
from core.pdf_parser import DocumentParser
from core.evidence_engine import EvidenceEngine
from core.chemistry_extractor import ChemistryExtractor
from core.table_extractor import TableExtractor
from core.reaction_engine import ReactionPathwayExtractor
from core.entity_normalizer import EntityNormalizer
from core.sar_engine import SAREngine
from core.knowledge_graph import KnowledgeGraphBuilder
from core.confidence_scorer import ConfidenceScorer
from core.gap_detector import ResearchGapDetector
from core.multi_paper_comparator import MultiPaperComparator


class TestAdvancedFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ingest Paper 1
        with open("/Users/prejin/Thesis/sample_papers/paper1_kinase_inhibitors.txt", "r", encoding="utf-8") as f:
            p1_text = f.read()
        cls.pages_p1 = DocumentParser.parse_text_content(p1_text)
        cls.meta_p1 = DocumentParser.extract_metadata(cls.pages_p1)
        cls.engine_p1 = EvidenceEngine(cls.pages_p1)
        extractor = ChemistryExtractor(use_local_only=True)
        cls.analysis_p1 = extractor.analyze_paper(cls.pages_p1, cls.meta_p1, cls.engine_p1)

        # Ingest Paper 2
        with open("/Users/prejin/Thesis/sample_papers/paper2_catalytic_synthesis_conflicts.txt", "r", encoding="utf-8") as f:
            p2_text = f.read()
        cls.pages_p2 = DocumentParser.parse_text_content(p2_text)
        cls.meta_p2 = DocumentParser.extract_metadata(cls.pages_p2)
        cls.engine_p2 = EvidenceEngine(cls.pages_p2)
        cls.analysis_p2 = extractor.analyze_paper(cls.pages_p2, cls.meta_p2, cls.engine_p2)

    def test_table_extraction(self):
        tables = self.analysis_p1.tables
        self.assertGreaterEqual(len(tables), 1)
        t = tables[0]
        self.assertIn("Compound", " ".join(t.headers))
        self.assertGreater(len(t.rows), 0)
        df = TableExtractor.table_to_dataframe(t)
        self.assertFalse(df.empty)

    def test_reaction_pathways(self):
        pathways = self.analysis_p1.reaction_pathways
        self.assertGreaterEqual(len(pathways), 1)
        p = pathways[0]
        self.assertGreater(len(p.steps), 0)
        self.assertIsNotNone(p.steps[0].evidence)

    def test_entity_normalization(self):
        norm_entities = self.analysis_p1.normalized_entities
        self.assertGreater(len(norm_entities), 0)
        # Check standard drug normalization or compound normalization
        names = [e.canonical_name for e in norm_entities]
        self.assertTrue(any("Compound" in n or "Erlotinib" in n for n in names))

    def test_sar_analysis_and_plot(self):
        sars = self.analysis_p1.sar_analyses
        self.assertGreaterEqual(len(sars), 1)
        sar = sars[0]
        self.assertGreater(len(sar.datapoints), 0)
        self.assertIsNotNone(sar.optimal_lead)
        # Check Plotly figures generation
        fig1 = SAREngine.create_sar_potency_chart(sar)
        self.assertIsNotNone(fig1)
        fig2 = SAREngine.create_yield_vs_potency_scatter(sar)
        self.assertIsNotNone(fig2)

    def test_knowledge_graph(self):
        kg = self.analysis_p1.knowledge_graph
        self.assertIsNotNone(kg)
        self.assertGreater(len(kg.nodes), 0)
        self.assertGreater(len(kg.edges), 0)
        html = KnowledgeGraphBuilder.render_interactive_html(kg)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("kgCanvas", html)

    def test_confidence_scoring(self):
        score = self.analysis_p1.confidence_breakdown
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score.overall_score, 0.0)
        self.assertLessEqual(score.overall_score, 100.0)
        self.assertIn(score.tier, ["High (A+)", "Strong (A)", "Moderate (B)", "Tentative (C)"])

    def test_research_gap_detection(self):
        gaps = self.analysis_p1.research_gaps
        self.assertGreater(len(gaps), 0)
        gap_categories = [g.category for g in gaps]
        self.assertIn("ADME & Pharmacokinetics", gap_categories)

    def test_multi_paper_comparison(self):
        comparison = MultiPaperComparator.compare_papers([self.analysis_p1, self.analysis_p2])
        self.assertIsNotNone(comparison)
        self.assertEqual(len(comparison.papers), 2)
        self.assertGreater(len(comparison.lead_comparison_table), 0)
        self.assertGreater(len(comparison.synthetic_route_comparison), 0)
        self.assertIn("Cross-Literature Synthesis", comparison.comparative_synthesis)

    def test_multi_paper_cross_qa(self):
        answer = MultiPaperComparator.ask_cross_paper(
            question="Which paper discovered the most potent lead compound and what is its IC50?",
            analyses=[self.analysis_p1, self.analysis_p2]
        )
    def test_multi_paper_comparison_with_missing_and_none_fields(self):
        from core.models import PaperAnalysisResult, PaperMetadata, BioactivityResult, ExperimentalCondition, EvidenceCitation
        cit = EvidenceCitation(page_number=1, verbatim_quote="quote", confidence=0.9)
        # Create an analysis with None compound_id in bioactivity, None target, and empty lists
        a_none = PaperAnalysisResult(
            metadata=PaperMetadata(title="Sparse Paper"),
            executive_summary="",
            compounds=[],
            properties=[],
            bioactivities=[
                BioactivityResult(compound_id=None, assay_type="IC50", target="Enzyme", value="42.5", unit="nM", evidence=cit)
            ],
            conditions=[
                ExperimentalCondition(reaction_step="Coupling", solvent=None, catalyst=None, evidence=cit)
            ],
            methodologies=[],
            findings=[],
            conflicts=[],
            extracted_tables=[],
            reaction_pathways=[],
            sar_analyses=[],
            confidence_breakdown=self.analysis_p1.confidence_breakdown,
            research_gaps=[]
        )
        comparison = MultiPaperComparator.compare_papers([self.analysis_p1, a_none])
        self.assertIsNotNone(comparison)
        self.assertEqual(len(comparison.lead_comparison_table), 2)
        for row in comparison.lead_comparison_table:
            for k, v in row.items():
                self.assertIsNotNone(v, f"Key '{k}' in lead table row should not be None")
                self.assertIsInstance(v, str, f"Key '{k}' should be string, got {type(v)}")


if __name__ == "__main__":
    unittest.main()
