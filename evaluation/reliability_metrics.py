"""
Reliability, Hallucination, and Detection Metrics.
Evaluates Unsupported Claim Rate, Hallucination-Free Rate,
Absence Detection (Precision/Recall/F1), Conflict Detection (Precision/Recall/F1),
and produces a 3x3 Confusion Matrix across found, absent, and conflicting statuses.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from evaluation.benchmark_loader import BenchmarkCase
from evaluation.citation_metrics import extract_factual_claims_from_answer
from evaluation.chemistry_metrics import parse_chemistry_entities_from_text


def evaluate_unsupported_claims(
    answer: str,
    retrieved_text: str,
    citations_text: str
) -> Dict[str, Any]:
    """
    Identifies claims, numbers, compounds, metrics, and targets made in the answer
    that lack textual support in the retrieved or cited passages.
    """
    evidence_corpus = f"{retrieved_text} {citations_text}".lower()
    parsed_ans = parse_chemistry_entities_from_text(answer)

    unsupported_numbers = []
    for num in parsed_ans["values"]:
        # Expose numbers not found in evidence
        num_str = str(num).rstrip("0").rstrip(".") if "." in str(num) else str(num)
        if num_str not in evidence_corpus:
            unsupported_numbers.append(num)

    unsupported_compounds = []
    for comp in parsed_ans["compounds"]:
        if comp not in evidence_corpus:
            unsupported_compounds.append(comp)

    unsupported_metrics = []
    for met in parsed_ans["metrics"]:
        if met.lower() not in evidence_corpus:
            unsupported_metrics.append(met)

    unsupported_targets = []
    for tgt in parsed_ans["targets"]:
        if tgt.lower() not in evidence_corpus:
            unsupported_targets.append(tgt)

    claims = extract_factual_claims_from_answer(answer)
    total_claims = max(1, len(claims))
    unsupported_claims = 0

    for claim in claims:
        claim_nums = re.findall(r"\b\d+(?:\.\d+)?\b", claim)
        if claim_nums and not any(n in evidence_corpus for n in claim_nums):
            unsupported_claims += 1

    unsupported_claim_rate = round(unsupported_claims / float(total_claims), 4)
    hallucination_free_rate = round(max(0.0, 1.0 - unsupported_claim_rate), 4)

    return {
        "unsupported_claim_rate": unsupported_claim_rate,
        "hallucination_free_rate": hallucination_free_rate,
        "unsupported_numbers": unsupported_numbers,
        "unsupported_compounds": unsupported_compounds,
        "unsupported_metrics": unsupported_metrics,
        "unsupported_targets": unsupported_targets,
        "total_claims_count": total_claims,
        "unsupported_claims_count": unsupported_claims
    }


def compute_confusion_matrix(
    predictions: List[str],
    ground_truths: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Computes a 3x3 confusion matrix across ['found', 'absent', 'conflicting'].
    Format: matrix[actual_label][predicted_label] = count
    """
    classes = ["found", "absent", "conflicting"]
    matrix: Dict[str, Dict[str, int]] = {
        c: {p: 0 for p in classes} for c in classes
    }

    for pred, truth in zip(predictions, ground_truths):
        p = pred.lower().strip() if pred else "found"
        t = truth.lower().strip() if truth else "found"

        if p not in classes:
            p = "found"
        if t not in classes:
            t = "found"

        matrix[t][p] += 1

    return matrix


def evaluate_absence_detection(
    predictions: List[str],
    ground_truths: List[str]
) -> Dict[str, float]:
    """
    Computes Precision, Recall, F1, and Accuracy for detecting 'absent' evidence.
    """
    tp = 0  # truth=absent, pred=absent
    fp = 0  # truth!=absent, pred=absent
    fn = 0  # truth=absent, pred!=absent
    tn = 0  # truth!=absent, pred!=absent

    for pred, truth in zip(predictions, ground_truths):
        p = pred.lower().strip()
        t = truth.lower().strip()

        is_pred_absent = (p == "absent")
        is_truth_absent = (t == "absent")

        if is_truth_absent and is_pred_absent:
            tp += 1
        elif not is_truth_absent and is_pred_absent:
            fp += 1
        elif is_truth_absent and not is_pred_absent:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / float(max(1, total))
    precision = tp / float(max(1, tp + fp)) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
    recall = tp / float(max(1, tp + fn)) if (tp + fn) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "absence_precision": round(precision, 4),
        "absence_recall": round(recall, 4),
        "absence_f1": round(f1, 4),
        "absence_accuracy": round(accuracy, 4)
    }


def evaluate_conflict_detection(
    predictions: List[str],
    ground_truths: List[str]
) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1 for detecting 'conflicting' evidence.
    """
    tp = 0
    fp = 0
    fn = 0

    for pred, truth in zip(predictions, ground_truths):
        p = pred.lower().strip()
        t = truth.lower().strip()

        is_pred_conf = (p == "conflicting")
        is_truth_conf = (t == "conflicting")

        if is_truth_conf and is_pred_conf:
            tp += 1
        elif not is_truth_conf and is_pred_conf:
            fp += 1
        elif is_truth_conf and not is_pred_conf:
            fn += 1

    precision = tp / float(max(1, tp + fp)) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
    recall = tp / float(max(1, tp + fn)) if (tp + fn) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "conflict_precision": round(precision, 4),
        "conflict_recall": round(recall, 4),
        "conflict_f1": round(f1, 4)
    }
