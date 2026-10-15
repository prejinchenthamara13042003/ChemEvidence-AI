"""
Unit tests for the ChemEvidence AI Evaluation Framework (evaluation/).
Tests:
- Precision@K, Recall@K, Hit Rate@K, MRR
- Exact match, Token F1, and scientific text normalization
- Chemistry-specific accuracy: entity, target, metric type, and numerical value with unit conversion
- Strict disambiguation of biochemical metrics (IC50 vs GI50, IC50 vs EC50)
- Compound matching (e.g. 5a vs 5d)
- Absence detection and conflict detection metrics
- Unsupported claim rate and hallucination-free rate calculation
- Required regression tests (Test 1, Test 2, Test 3)
"""

import unittest
from evaluation.benchmark_loader import BenchmarkCase
from evaluation.retrieval_metrics import evaluate_retrieval_for_case
from evaluation.answer_metrics import (
    exact_match_score,
    token_f1_score,
    normalize_scientific_answer,
    normalized_answer_accuracy
)
from evaluation.chemistry_metrics import (
    evaluate_chemistry_accuracy,
    are_numbers_equivalent,
    convert_to_canonical_concentration,
    normalize_metric_type
)
from evaluation.citation_metrics import evaluate_citations
from evaluation.reliability_metrics import (
    evaluate_unsupported_claims,
    evaluate_absence_detection,
    evaluate_conflict_detection,
    compute_confusion_matrix
)
from evaluation.evaluator import Evaluator
from core.models import QAResponse, EvidenceCitation
from core.pdf_parser import PageContent
from core.evidence_engine import EvidenceChunk


class TestEvaluationMetrics(unittest.TestCase):

    def test_normalization_subscripts_and_units(self):
        """Verifies normalization handles subscripts and micro symbols."""
        raw = "Compound 3b IC₅₀ = 8.4 ± 0.6 nM, with a GI₅₀ of 0.62 µM (or 0.62 μM)."
        norm = normalize_scientific_answer(raw)
        self.assertIn("ic50", norm)
        self.assertIn("gi50", norm)
        self.assertIn("8.4", norm)
        self.assertIn("µm", norm)
        self.assertNotIn("₅", norm)
        self.assertNotIn("₀", norm)

    def test_answer_metrics_exact_match_and_f1(self):
        """Verifies exact match and token F1."""
        exp = "8.4 ± 0.6 nM"
        pred = "Compound 3b exhibited an IC50 of 8.4 ± 0.6 nM against EGFR kinase."
        
        # Exact match should recognise the phrase inside the text
        self.assertEqual(exact_match_score(pred, exp), 1.0)

        # Token F1
        f1 = token_f1_score(pred, exp)
        self.assertGreater(f1, 0.4)

        # Mismatch
        self.assertEqual(exact_match_score("120.5 nM", "8.4 nM"), 0.0)

    # -------------------------------------------------------------
    # User Required Specific Regression Test 1
    # -------------------------------------------------------------
    def test_regression_test_1_compound_and_metric_mismatch(self):
        """
        Test 1:
        Evidence: "Compound 5d showed a GI50 of 50 nM."
        Prediction: "Compound 5a showed an EGFR IC50 of 50 nM."
        Expected:
        metric_type_correct = false
        entity_correct = false
        overall_chemistry_correct = false
        """
        # Benchmark ground truth expecting Compound 5d with GI50
        case = BenchmarkCase(
            id="TEST_REG_1",
            question="What is the potency of compound 5d?",
            expected_answer="Compound 5d showed a GI50 of 50 nM.",
            expected_status="found",
            compound="5d",
            target=None,
            metric="GI50",
            value=50.0,
            unit="nM"
        )

        prediction = "Compound 5a showed an EGFR IC50 of 50 nM."

        res = evaluate_chemistry_accuracy(prediction, case)

        self.assertFalse(res["metric_type_correct"], "Expected metric_type_correct to be False (IC50 != GI50)")
        self.assertFalse(res["entity_correct"], "Expected entity_correct to be False (5a != 5d)")
        self.assertFalse(res["overall_chemistry_correct"], "Expected overall_chemistry_correct to be False")

    # -------------------------------------------------------------
    # User Required Specific Regression Test 2
    # -------------------------------------------------------------
    def test_regression_test_2_unit_conversion_equivalence(self):
        """
        Test 2:
        Expected: 87 nM
        Prediction: 0.087 µM
        Expected:
        numeric equivalence should be recognized after unit conversion (numeric_correct = True).
        """
        case = BenchmarkCase(
            id="TEST_REG_2",
            question="What is the IC50 of compound 5a?",
            expected_answer="87 nM",
            expected_status="found",
            compound="5a",
            metric="IC50",
            value=87.0,
            unit="nM"
        )

        prediction = "The IC50 of Compound 5a was measured to be 0.087 µM."

        res = evaluate_chemistry_accuracy(prediction, case)

        self.assertTrue(res["numeric_correct"], "Expected 0.087 µM to be numerically equivalent to 87 nM")
        self.assertTrue(res["entity_correct"])
        self.assertTrue(res["metric_type_correct"])
        self.assertTrue(res["overall_chemistry_correct"])

    # -------------------------------------------------------------
    # User Required Specific Regression Test 3
    # -------------------------------------------------------------
    def test_regression_test_3_absence_detection(self):
        """
        Test 3:
        Expected status: absent
        Prediction: "No supporting evidence was found in the uploaded manuscript."
        Expected:
        correct absence detection.
        """
        case = BenchmarkCase(
            id="TEST_REG_3",
            question="What was the Phase III human clinical dose of compound 5a?",
            expected_answer="No supporting evidence was found in the uploaded manuscript.",
            expected_status="absent"
        )

        qa_resp = QAResponse(
            question=case.question,
            answer="No supporting evidence was found in the uploaded manuscript.",
            status="absent",
            citations=[],
            confidence_score=1.0
        )

        pages = [PageContent(page_number=1, text="Sample text without clinical dose.")]
        eval_res = Evaluator.evaluate_question(case, qa_resp, [], pages)

        self.assertTrue(eval_res.is_status_correct, "Expected status to be correctly evaluated as absent")
        self.assertEqual(eval_res.predicted_status, "absent")
        self.assertEqual(eval_res.exact_match, 1.0)
        self.assertEqual(eval_res.token_f1, 1.0)

    def test_retrieval_metrics(self):
        """Verifies Precision@K, Recall@K, Hit Rate@K, and MRR calculations."""
        chunk1 = EvidenceChunk("c1", 3, "SAR", "Compound 3b EGFR IC50 8.4 nM in Table 1.")
        chunk2 = EvidenceChunk("c2", 2, "Chemistry", "Synthetic procedures for pyrimidines.")
        chunk3 = EvidenceChunk("c3", 1, "Abstract", "Novel 2,4-diaminopyrimidines as EGFR inhibitors.")

        case = BenchmarkCase(
            id="Q001",
            question="What is the IC50 of compound 3b?",
            expected_answer="8.4 nM",
            expected_page=3,
            expected_section="SAR",
            evidence_keywords=["3b", "EGFR", "8.4", "nM"],
            relevant_chunk_ids=["c1"]
        )

        ret_metrics = evaluate_retrieval_for_case([chunk1, chunk2, chunk3], case, k_list=[1, 3, 5])
        self.assertEqual(ret_metrics["hit_rate@1"], 1.0)
        self.assertEqual(ret_metrics["hit_rate@3"], 1.0)
        self.assertEqual(ret_metrics["precision@1"], 1.0)
        self.assertEqual(round(ret_metrics["precision@3"], 2), 0.33)
        self.assertEqual(ret_metrics["recall@1"], 1.0)
        self.assertEqual(ret_metrics["mrr"], 1.0)

    def test_absence_and_conflict_metrics(self):
        """Verifies precision, recall, and F1 calculations for absence and conflict detection."""
        # 3 questions: absent, found, conflicting
        preds = ["absent", "found", "conflicting"]
        truths = ["absent", "found", "conflicting"]

        abs_res = evaluate_absence_detection(preds, truths)
        self.assertEqual(abs_res["absence_f1"], 1.0)
        self.assertEqual(abs_res["absence_precision"], 1.0)
        self.assertEqual(abs_res["absence_recall"], 1.0)

        conf_res = evaluate_conflict_detection(preds, truths)
        self.assertEqual(conf_res["conflict_f1"], 1.0)

        matrix = compute_confusion_matrix(preds, truths)
        self.assertEqual(matrix["absent"]["absent"], 1)
        self.assertEqual(matrix["found"]["found"], 1)
        self.assertEqual(matrix["conflicting"]["conflicting"], 1)

    def test_unsupported_claim_rate(self):
        """Verifies unsupported claims and hallucination-free rate calculation."""
        evidence = "Compound 3b showed an EGFR IC50 of 8.4 nM."
        citations = "Page 3: Compound 3b IC50 8.4 nM."
        
        # Answer containing a supported claim and an unsupported hallucinated number (999 nM for compound 9z)
        ans = "Compound 3b exhibited an IC50 of 8.4 nM. Furthermore, compound 9z had an IC50 of 999 nM."
        
        rel = evaluate_unsupported_claims(ans, evidence, citations)
        self.assertIn(999.0, rel["unsupported_numbers"])
        self.assertIn("9z", rel["unsupported_compounds"])
        self.assertGreater(rel["unsupported_claim_rate"], 0.0)
        self.assertLess(rel["hallucination_free_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
