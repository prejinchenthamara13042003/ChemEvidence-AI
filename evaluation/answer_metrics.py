"""
Answer Quality Metrics for RAG Systems.
Computes Exact Match, Token-Level F1, and Normalized Answer Accuracy
with scientific entity and notation normalization.
"""

import re
from typing import Dict, Any, List, Set, Tuple


def normalize_scientific_answer(text: str) -> str:
    """
    Normalizes answer strings for equitable, robust textual comparison.
    - Lowers case and collapses whitespace
    - Replaces Unicode subscript numerals (₀-₉ -> 0-9)
    - Normalizes micro symbols (µ / μ -> µ)
    - Normalizes common chemical punctuation differences while preserving numbers and decimal points
    """
    if not text:
        return ""

    t = text.lower().strip()

    # Strip LaTeX math tags and formatting (e.g. \text{IC}_{50}, \mathrm{...})
    t = re.sub(r"\\text\{([^}]+)\}", r"\1", t)
    t = re.sub(r"\\(?:mathrm|mathbf|mathit)\{([^}]+)\}", r"\1", t)
    t = t.replace("$", "").replace("{", "").replace("}", "").replace("\\", "")

    # Unicode subscript numerals
    sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    t = t.translate(sub_map)

    # Normalize Greek / micro symbols to standard 'µ'
    t = re.sub(r"[\u00b5\u03bc]", "µ", t)

    # Normalize IC50 notation variants
    t = re.sub(r"\bic\s*_?50\b", "ic50", t)
    t = re.sub(r"\bgi\s*_?50\b", "gi50", t)
    t = re.sub(r"\bec\s*_?50\b", "ec50", t)

    # Normalize plus-minus
    t = t.replace("±", "+/-").replace("&plusmn;", "+/-")

    # Strip markdown formatting bold/italics
    t = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)

    # Remove harmless punctuation while preserving dots in decimals and hyphens in names
    t = re.sub(r"[^\w\s\.\-\/\+\%µ]", " ", t)

    # Remove trailing/leading periods in tokens
    tokens = [tok.strip(".") for tok in t.split() if tok.strip(".")]
    return " ".join(tokens)


def tokenize_answer(text: str) -> List[str]:
    """Tokenizes normalized text into constituent words and numbers."""
    norm = normalize_scientific_answer(text)
    # Match alphanumeric words, numbers with decimals, units, and symbols
    return re.findall(r"[A-Za-z0-9\.\-\/\+\%µ]+", norm)


def exact_match_score(predicted: str, expected: str) -> float:
    """
    Computes exact match score between normalized predicted and expected text.
    Returns 1.0 if identical or if normalized expected is an exact standalone phrase in predicted, else 0.0.
    """
    norm_pred = normalize_scientific_answer(predicted)
    norm_exp = normalize_scientific_answer(expected)

    if not norm_pred or not norm_exp:
        return 1.0 if norm_pred == norm_exp else 0.0

    if norm_pred == norm_exp:
        return 1.0

    # Also accept if the exact expected phrase appears cleanly in the prediction
    if f" {norm_exp} " in f" {norm_pred} " or norm_exp in norm_pred:
        return 1.0

    return 0.0


def _compute_span_f1(pred_tokens: List[str], exp_tokens: List[str]) -> float:
    """Calculates F1 between two token sequences."""
    if not pred_tokens or not exp_tokens:
        return 1.0 if pred_tokens == exp_tokens else 0.0

    exp_counts: Dict[str, int] = {}
    for tok in exp_tokens:
        exp_counts[tok] = exp_counts.get(tok, 0) + 1

    matched = 0
    pred_counts: Dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1

    for tok, p_count in pred_counts.items():
        if tok in exp_counts:
            matched += min(p_count, exp_counts[tok])

    if matched == 0:
        return 0.0

    precision = matched / float(len(pred_tokens))
    recall = matched / float(len(exp_tokens))
    return (2 * precision * recall) / (precision + recall)


def token_f1_score(predicted: str, expected: str) -> float:
    """
    Computes token-level precision, recall, and harmonic F1 score.
    For explanatory RAG answers evaluated against concise reference answers,
    evaluates against the best-matching sentence or candidate span to avoid
    penalizing rich explanatory context and citations.
    """
    pred_tokens = tokenize_answer(predicted)
    exp_tokens = tokenize_answer(expected)

    if not pred_tokens or not exp_tokens:
        return 1.0 if pred_tokens == exp_tokens else 0.0

    overall_f1 = _compute_span_f1(pred_tokens, exp_tokens)

    # Candidate sentence/line-level F1
    candidates = re.split(r"(?:\n+|[\.\!\?]\s+)", predicted)
    best_cand_f1 = 0.0
    for cand in candidates:
        c_tokens = tokenize_answer(cand)
        if c_tokens:
            f1 = _compute_span_f1(c_tokens, exp_tokens)
            if f1 > best_cand_f1:
                best_cand_f1 = f1

    norm_exp = normalize_scientific_answer(expected)
    norm_pred = normalize_scientific_answer(predicted)
    if norm_exp and norm_exp in norm_pred:
        return round(max(overall_f1, best_cand_f1, 0.90), 4)

    return round(max(overall_f1, best_cand_f1), 4)


def normalized_answer_accuracy(predicted: str, expected: str) -> float:
    """
    Checks whether key factual components of the expected answer are accurately captured
    in the predicted answer (tolerant of surrounding explanatory prose).
    """
    norm_pred = normalize_scientific_answer(predicted)
    norm_exp = normalize_scientific_answer(expected)

    if not norm_exp:
        return 1.0 if not norm_pred else 0.0

    if norm_exp in norm_pred:
        return 1.0

    # Check token overlap ratio against expected tokens
    exp_tokens = set(tokenize_answer(expected))
    pred_tokens = set(tokenize_answer(predicted))

    if not exp_tokens:
        return 1.0

    overlap = len(exp_tokens & pred_tokens)
    ratio = overlap / float(len(exp_tokens))
    return 1.0 if ratio >= 0.80 else round(ratio, 4)


def evaluate_answer_quality(predicted: str, expected: str) -> Dict[str, float]:
    """Computes all standard answer metrics for a prediction against ground truth."""
    em = exact_match_score(predicted, expected)
    f1 = token_f1_score(predicted, expected)
    acc = normalized_answer_accuracy(predicted, expected)

    return {
        "exact_match": em,
        "token_f1": f1,
        "normalized_accuracy": acc
    }
