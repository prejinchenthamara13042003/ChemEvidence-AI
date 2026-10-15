"""
Master Evaluator Engine.
Coordinates retrieval, answer quality, chemistry factual accuracy,
citation verification, reliability, absence/conflict detection, and latency aggregation.
"""

import time
import math
import statistics
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from core.models import QAResponse, EvidenceCitation
from core.pdf_parser import PageContent
from evaluation.benchmark_loader import BenchmarkCase
from evaluation.retrieval_metrics import evaluate_retrieval_for_case, aggregate_retrieval_metrics
from evaluation.answer_metrics import evaluate_answer_quality
from evaluation.chemistry_metrics import evaluate_chemistry_accuracy
from evaluation.citation_metrics import evaluate_citations
from evaluation.reliability_metrics import (
    evaluate_unsupported_claims,
    compute_confusion_matrix,
    evaluate_absence_detection,
    evaluate_conflict_detection
)


class QuestionEvaluationResult(BaseModel):
    """Detailed evaluation metrics for a single benchmark question."""
    question_id: str
    category: str
    question: str
    expected_answer: str
    predicted_answer: str
    expected_status: str
    predicted_status: str
    is_status_correct: bool

    # Answer quality
    exact_match: float
    token_f1: float
    normalized_accuracy: float

    # Chemistry metrics
    entity_correct: bool
    target_correct: bool
    metric_type_correct: bool
    numeric_correct: bool
    unit_correct: bool
    overall_chemistry_correct: bool

    # Citations & Faithfulness
    citation_validity: float
    citation_accuracy: float
    citation_completeness: float
    faithfulness: float
    unsupported_claim_rate: float
    hallucination_free_rate: float

    # Latency & System
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    engine_used: str = "local"
    notes: str = ""
    retrieval_metrics: Dict[str, float] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Aggregated evaluation results across an entire benchmark dataset."""
    total_questions: int
    retrieval_metrics: Dict[str, float]
    answer_metrics: Dict[str, float]
    chemistry_metrics: Dict[str, float]
    citation_metrics: Dict[str, float]
    reliability_metrics: Dict[str, float]
    latency_metrics: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]
    composite_benchmark_score: float
    composite_weighting_explanation: str
    per_question_results: List[QuestionEvaluationResult]
    overall_summary: Dict[str, Any]


class Evaluator:
    """Evaluates chatbot predictions against ground truth benchmark test cases."""

    COMPOSITE_WEIGHTS = {
        "chemistry_accuracy": 0.25,
        "answer_f1": 0.25,
        "retrieval_hit_at_5": 0.20,
        "citation_accuracy": 0.15,
        "hallucination_free_rate": 0.15
    }

    @classmethod
    def evaluate_question(
        cls,
        case: BenchmarkCase,
        qa_response: QAResponse,
        retrieved_chunks: List[Any],
        pages: List[PageContent],
        retrieval_latency_ms: float = 0.0,
        generation_latency_ms: float = 0.0,
        engine_used: str = "local"
    ) -> QuestionEvaluationResult:
        """Evaluates a single question against its benchmark ground truth."""
        pred_ans = qa_response.answer or ""
        pred_status = qa_response.status or "found"

        # Check status correctness
        is_status_correct = (pred_status.lower().strip() == case.expected_status.lower().strip())

        # If answer is an absence response, normalize answer text check
        if case.expected_status == "absent":
            if "no supporting evidence was found" in pred_ans.lower() or pred_status == "absent":
                is_status_correct = True
                pred_status = "absent"

        # 1. Retrieval Metrics
        retrieval_res = evaluate_retrieval_for_case(retrieved_chunks, case)

        # 2. Answer Quality Metrics
        if case.expected_status == "absent" and is_status_correct:
            ans_res = {"exact_match": 1.0, "token_f1": 1.0, "normalized_accuracy": 1.0}
        else:
            ans_res = evaluate_answer_quality(pred_ans, case.expected_answer)

        # 3. Chemistry Metrics
        if case.expected_status == "absent":
            chem_res = {
                "entity_correct": is_status_correct,
                "target_correct": is_status_correct,
                "metric_type_correct": is_status_correct,
                "numeric_correct": is_status_correct,
                "unit_correct": is_status_correct,
                "overall_chemistry_correct": is_status_correct
            }
        else:
            chem_res = evaluate_chemistry_accuracy(pred_ans, case)

        # 4. Citation Metrics
        citations = getattr(qa_response, "citations", []) or []
        cit_res = evaluate_citations(citations, pages, case, predicted_answer=pred_ans)

        # 5. Reliability & Unsupported Claims
        retrieved_text = " ".join([getattr(c, "text", "") for c in retrieved_chunks])
        citations_text = " ".join([getattr(c, "verbatim_quote", "") for c in citations])
        rel_res = evaluate_unsupported_claims(pred_ans, retrieved_text, citations_text)

        total_lat = retrieval_latency_ms + generation_latency_ms

        notes_list = []
        if not is_status_correct:
            notes_list.append(f"Status mismatch: expected {case.expected_status}, got {pred_status}")
        if case.expected_status != "absent" and not chem_res["overall_chemistry_correct"]:
            if not chem_res["metric_type_correct"]:
                notes_list.append(f"Metric type incorrect (expected {case.metric})")
            if not chem_res["entity_correct"]:
                notes_list.append(f"Compound mismatch (expected {case.compound})")
            if not chem_res["numeric_correct"]:
                notes_list.append(f"Value mismatch (expected {case.value} {case.unit})")

        return QuestionEvaluationResult(
            question_id=case.id,
            category=case.category,
            question=case.question,
            expected_answer=case.expected_answer,
            predicted_answer=pred_ans,
            expected_status=case.expected_status,
            predicted_status=pred_status,
            is_status_correct=is_status_correct,
            exact_match=ans_res["exact_match"],
            token_f1=ans_res["token_f1"],
            normalized_accuracy=ans_res["normalized_accuracy"],
            entity_correct=chem_res["entity_correct"],
            target_correct=chem_res["target_correct"],
            metric_type_correct=chem_res["metric_type_correct"],
            numeric_correct=chem_res["numeric_correct"],
            unit_correct=chem_res["unit_correct"],
            overall_chemistry_correct=chem_res["overall_chemistry_correct"],
            citation_validity=cit_res["citation_validity"],
            citation_accuracy=cit_res["citation_accuracy"],
            citation_completeness=cit_res["citation_completeness"],
            faithfulness=cit_res["faithfulness"],
            unsupported_claim_rate=rel_res["unsupported_claim_rate"],
            hallucination_free_rate=rel_res["hallucination_free_rate"],
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round(total_lat, 2),
            engine_used=engine_used,
            notes="; ".join(notes_list),
            retrieval_metrics=retrieval_res
        )

    @classmethod
    def aggregate_results(
        cls,
        q_results: List[QuestionEvaluationResult]
    ) -> EvaluationResult:
        """Aggregates all question-level evaluation records into a unified benchmark summary."""
        N = len(q_results)
        if N == 0:
            raise ValueError("Cannot aggregate empty question evaluation list.")

        # Aggregate retrieval over findable questions (excluding questions where ground-truth is absent)
        findable_ret_dicts = [q.retrieval_metrics for q in q_results if q.retrieval_metrics and q.expected_status != "absent"]
        agg_retrieval = aggregate_retrieval_metrics(findable_ret_dicts or [q.retrieval_metrics for q in q_results if q.retrieval_metrics])

        # Aggregate answer quality
        agg_answer = {
            "exact_match": round(sum(q.exact_match for q in q_results) / N, 4),
            "token_f1": round(sum(q.token_f1 for q in q_results) / N, 4),
            "normalized_accuracy": round(sum(q.normalized_accuracy for q in q_results) / N, 4)
        }

        # Aggregate chemistry
        agg_chemistry = {
            "entity_accuracy": round(sum(1.0 if q.entity_correct else 0.0 for q in q_results) / N, 4),
            "target_accuracy": round(sum(1.0 if q.target_correct else 0.0 for q in q_results) / N, 4),
            "metric_type_accuracy": round(sum(1.0 if q.metric_type_correct else 0.0 for q in q_results) / N, 4),
            "numeric_accuracy": round(sum(1.0 if q.numeric_correct else 0.0 for q in q_results) / N, 4),
            "unit_accuracy": round(sum(1.0 if q.unit_correct else 0.0 for q in q_results) / N, 4),
            "overall_chemistry_accuracy": round(sum(1.0 if q.overall_chemistry_correct else 0.0 for q in q_results) / N, 4)
        }

        # Aggregate citations & reliability
        agg_citations = {
            "citation_validity": round(sum(q.citation_validity for q in q_results) / N, 4),
            "citation_accuracy": round(sum(q.citation_accuracy for q in q_results) / N, 4),
            "citation_completeness": round(sum(q.citation_completeness for q in q_results) / N, 4),
            "faithfulness": round(sum(q.faithfulness for q in q_results) / N, 4)
        }

        # Reliability & absence/conflict metrics
        pred_statuses = [q.predicted_status for q in q_results]
        true_statuses = [q.expected_status for q in q_results]
        conf_matrix = compute_confusion_matrix(pred_statuses, true_statuses)
        absence_eval = evaluate_absence_detection(pred_statuses, true_statuses)
        conflict_eval = evaluate_conflict_detection(pred_statuses, true_statuses)

        agg_reliability = {
            "unsupported_claim_rate": round(sum(q.unsupported_claim_rate for q in q_results) / N, 4),
            "hallucination_free_rate": round(sum(q.hallucination_free_rate for q in q_results) / N, 4),
            **absence_eval,
            **conflict_eval
        }

        # Latency statistics
        tot_lats = [q.total_latency_ms for q in q_results]
        sorted_lats = sorted(tot_lats)
        p95_idx = int(math.ceil(0.95 * len(sorted_lats))) - 1
        agg_latency = {
            "mean_total_latency_ms": round(statistics.mean(tot_lats), 2),
            "median_total_latency_ms": round(statistics.median(tot_lats), 2),
            "min_total_latency_ms": round(min(tot_lats), 2),
            "max_total_latency_ms": round(max(tot_lats), 2),
            "p95_total_latency_ms": round(sorted_lats[p95_idx], 2)
        }

        # Composite score calculation (strictly documented weights)
        c_chem = agg_chemistry["overall_chemistry_accuracy"]
        c_ans = agg_answer["token_f1"]
        c_ret = agg_retrieval.get("hit_rate@5", 1.0)
        c_cit = agg_citations["citation_accuracy"]
        c_hall = agg_reliability["hallucination_free_rate"]

        composite = (
            cls.COMPOSITE_WEIGHTS["chemistry_accuracy"] * c_chem +
            cls.COMPOSITE_WEIGHTS["answer_f1"] * c_ans +
            cls.COMPOSITE_WEIGHTS["retrieval_hit_at_5"] * c_ret +
            cls.COMPOSITE_WEIGHTS["citation_accuracy"] * c_cit +
            cls.COMPOSITE_WEIGHTS["hallucination_free_rate"] * c_hall
        )
        composite = round(composite * 100.0, 2)

        weighting_exp = (
            "Composite Benchmark Score is calculated as: "
            "25% Overall Chemistry Accuracy + 25% Token F1 + "
            "20% Retrieval Hit@5 + 15% Citation Accuracy + 15% Hallucination-Free Rate."
        )

        overall_summary = {
            "total_benchmark_questions": N,
            "status_accuracy": round(sum(1.0 if q.is_status_correct else 0.0 for q in q_results) / N, 4),
            "composite_score_percent": composite,
            "mean_latency_ms": agg_latency["mean_total_latency_ms"]
        }

        return EvaluationResult(
            total_questions=N,
            retrieval_metrics=agg_retrieval,
            answer_metrics=agg_answer,
            chemistry_metrics=agg_chemistry,
            citation_metrics=agg_citations,
            reliability_metrics=agg_reliability,
            latency_metrics=agg_latency,
            confusion_matrix=conf_matrix,
            composite_benchmark_score=composite,
            composite_weighting_explanation=weighting_exp,
            per_question_results=q_results,
            overall_summary=overall_summary
        )
