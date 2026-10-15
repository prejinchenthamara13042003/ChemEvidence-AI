"""
Retrieval Quality Metrics for Evidence-Grounded RAG.
Computes Precision@K, Recall@K, Hit Rate@K, and MRR (Mean Reciprocal Rank)
across configurable K cutoffs (1, 3, 5, 10).
Supports chunk ID matching and robust fallback matching via page, section, and evidence keywords.
"""

import re
from typing import List, Dict, Any, Optional, Set
from evaluation.benchmark_loader import BenchmarkCase


def normalize_text_for_retrieval(text: str) -> str:
    """Normalizes text for fuzzy keyword and quote matching."""
    t = text.lower()
    t = re.sub(r"[₀₁₂₃₄₅₆₇₈₉]", lambda m: str("₀₁₂₃₄₅₆₇₈₉".index(m.group(0))), t)
    t = t.replace("µ", "u").replace("μ", "u")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_chunk_relevant(chunk: Any, case: BenchmarkCase) -> bool:
    """
    Determines whether an EvidenceChunk is relevant to a BenchmarkCase.
    Supports primary chunk ID matching and fallback heuristic matching.
    """
    chunk_id = getattr(chunk, "chunk_id", "")
    chunk_page = getattr(chunk, "page_number", None)
    chunk_sec = getattr(chunk, "section", "")
    chunk_text = getattr(chunk, "text", "")

    # 1. Primary match: explicit chunk IDs in ground truth
    if case.relevant_chunk_ids:
        if chunk_id and chunk_id in case.relevant_chunk_ids:
            return True

    # For absence questions, no chunks should be deemed relevant
    if case.expected_status == "absent":
        return False

    norm_chunk_text = normalize_text_for_retrieval(chunk_text)

    # 2. Evidence Keywords matching
    if case.evidence_keywords:
        matched_kw_count = 0
        for kw in case.evidence_keywords:
            norm_kw = normalize_text_for_retrieval(kw)
            if norm_kw in norm_chunk_text:
                matched_kw_count += 1
        
        # If all or high proportion (>= 60%) of keywords are found
        kw_ratio = matched_kw_count / len(case.evidence_keywords)
        if len(case.evidence_keywords) <= 2 and matched_kw_count == len(case.evidence_keywords):
            return True
        elif len(case.evidence_keywords) > 2 and kw_ratio >= 0.60:
            return True

    # 3. Page and Section fallback match
    if case.expected_page is not None and chunk_page is not None:
        if chunk_page == case.expected_page:
            # If section specified, verify overlap
            if case.expected_section:
                sec_match = (
                    case.expected_section.lower() in chunk_sec.lower() or
                    chunk_sec.lower() in case.expected_section.lower()
                )
                if sec_match:
                    return True
            else:
                return True

    # 4. Compound + Value match
    if case.compound and case.value is not None:
        c_norm = case.compound.lower()
        val_str = str(case.value)
        if c_norm in norm_chunk_text and val_str in norm_chunk_text:
            return True

    return False


def evaluate_retrieval_for_case(
    retrieved_chunks: List[Any],
    case: BenchmarkCase,
    total_relevant_count: Optional[int] = None,
    k_list: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Computes Precision@K, Recall@K, Hit Rate@K, and MRR for a single benchmark query.

    Args:
        retrieved_chunks: Ranked list of retrieved EvidenceChunk objects.
        case: Benchmark ground-truth case.
        total_relevant_count: Total relevant chunks existing in the corpus (defaults to at least 1).
        k_list: List of K thresholds.

    Returns:
        Dictionary of retrieval metrics.
    """
    metrics: Dict[str, float] = {}

    # For absence questions, retrieval metrics are evaluated against 0 expected relevant chunks
    if case.expected_status == "absent":
        for k in k_list:
            metrics[f"precision@{k}"] = 1.0 if not retrieved_chunks else 0.0
            metrics[f"recall@{k}"] = 1.0
            metrics[f"hit_rate@{k}"] = 0.0
        metrics["mrr"] = 0.0
        return metrics

    # Identify relevance for each retrieved chunk at its rank (1-indexed)
    relevance_flags: List[bool] = [is_chunk_relevant(c, case) for c in retrieved_chunks]
    
    # Calculate MRR (Reciprocal rank of the first relevant result)
    first_rank: Optional[int] = None
    for idx, is_rel in enumerate(relevance_flags):
        if is_rel:
            first_rank = idx + 1
            break
    metrics["mrr"] = (1.0 / first_rank) if first_rank is not None else 0.0

    # Determine total ground truth relevant items (fallback to at least 1 if found in retrieved, or len(relevant_chunk_ids))
    num_ground_truth = (
        len(case.relevant_chunk_ids)
        if case.relevant_chunk_ids
        else (total_relevant_count or max(1, sum(relevance_flags)))
    )

    for k in k_list:
        top_k_flags = relevance_flags[:k]
        rel_in_top_k = sum(top_k_flags)

        precision = rel_in_top_k / float(k)
        recall = rel_in_top_k / float(max(1, num_ground_truth))
        hit_rate = 1.0 if rel_in_top_k > 0 else 0.0

        metrics[f"precision@{k}"] = round(precision, 4)
        metrics[f"recall@{k}"] = round(min(1.0, recall), 4)
        metrics[f"hit_rate@{k}"] = round(hit_rate, 4)

    return metrics


def aggregate_retrieval_metrics(case_metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Computes mean retrieval metrics across all benchmark cases."""
    if not case_metrics_list:
        return {}
    
    keys = case_metrics_list[0].keys()
    aggregated: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in case_metrics_list if k in m]
        aggregated[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return aggregated
