"""
ChemEvidence AI RAG & Chemical Extraction Evaluation Framework.
Provides quantitative, research-grade evaluation of retrieval quality,
answer accuracy, chemistry factual fidelity, citations, reliability, and latency.
"""

from evaluation.benchmark_loader import BenchmarkCase, load_benchmark
from evaluation.retrieval_metrics import (
    evaluate_retrieval_for_case,
    aggregate_retrieval_metrics,
    is_chunk_relevant
)
from evaluation.answer_metrics import (
    exact_match_score,
    token_f1_score,
    normalize_scientific_answer,
    normalized_answer_accuracy,
    evaluate_answer_quality
)
from evaluation.chemistry_metrics import (
    evaluate_chemistry_accuracy,
    parse_chemistry_entities_from_text,
    are_numbers_equivalent,
    convert_to_canonical_concentration,
    normalize_metric_type
)
from evaluation.citation_metrics import (
    evaluate_citations,
    verify_citation_in_pages,
    extract_factual_claims_from_answer
)
from evaluation.reliability_metrics import (
    evaluate_unsupported_claims,
    compute_confusion_matrix,
    evaluate_absence_detection,
    evaluate_conflict_detection
)
from evaluation.evaluator import (
    Evaluator,
    EvaluationResult,
    QuestionEvaluationResult
)
from evaluation.benchmark_runner import BenchmarkRunner

__all__ = [
    "BenchmarkCase",
    "load_benchmark",
    "evaluate_retrieval_for_case",
    "aggregate_retrieval_metrics",
    "is_chunk_relevant",
    "exact_match_score",
    "token_f1_score",
    "normalize_scientific_answer",
    "normalized_answer_accuracy",
    "evaluate_answer_quality",
    "evaluate_chemistry_accuracy",
    "parse_chemistry_entities_from_text",
    "are_numbers_equivalent",
    "convert_to_canonical_concentration",
    "normalize_metric_type",
    "evaluate_citations",
    "verify_citation_in_pages",
    "extract_factual_claims_from_answer",
    "evaluate_unsupported_claims",
    "compute_confusion_matrix",
    "evaluate_absence_detection",
    "evaluate_conflict_detection",
    "Evaluator",
    "EvaluationResult",
    "QuestionEvaluationResult",
    "BenchmarkRunner"
]
