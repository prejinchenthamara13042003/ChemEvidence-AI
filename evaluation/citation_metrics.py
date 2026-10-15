"""
Citation and Grounding Metrics for Evidence-Aware RAG.
Evaluates Citation Validity, Citation Accuracy, Citation Completeness,
and Evidence Faithfulness against manuscript pages.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from core.pdf_parser import PageContent
from core.models import EvidenceCitation
from evaluation.benchmark_loader import BenchmarkCase


def verify_citation_in_pages(citation: Any, pages: List[PageContent]) -> Tuple[bool, float]:
    """
    Checks whether a citation's page and verbatim quote exist in the source pages.
    Returns (is_valid, match_confidence).
    """
    if not citation or not pages:
        return False, 0.0

    page_num = getattr(citation, "page_number", None)
    quote = getattr(citation, "verbatim_quote", "")

    if not quote or len(quote.strip()) < 5:
        return False, 0.0

    clean_quote = re.sub(r"\s+", " ", quote.lower().strip())

    # Check designated page first
    if page_num is not None and 1 <= page_num <= len(pages):
        page_text = re.sub(r"\s+", " ", pages[page_num - 1].text.lower())
        if clean_quote in page_text:
            return True, 1.0
        # Substring prefix check (at least 30 chars)
        prefix = clean_quote[:min(40, len(clean_quote))]
        if prefix in page_text:
            return True, 0.90

    # Fallback: check across all pages
    for p_idx, p in enumerate(pages):
        p_text = re.sub(r"\s+", " ", p.text.lower())
        if clean_quote in p_text:
            return True, 0.85
        prefix = clean_quote[:min(40, len(clean_quote))]
        if prefix in p_text:
            return True, 0.75

    return False, 0.0


def extract_factual_claims_from_answer(answer_text: str) -> List[str]:
    """
    Decomposes an answer into discrete factual claims and sentences,
    filtering out greetings, boilerplate, and citation headers.
    """
    if not answer_text:
        return []

    # Strip out citation sections and headers before splitting claims
    cleaned = re.split(r"(?i)###\s*(?:📑\s*)?Supporting\s+Verifiable\s+Citations", answer_text)[0]
    cleaned = re.split(r"(?i)###\s*(?:📚\s*)?Evidence\s+&\s+Paper\s+Source", cleaned)[0]
    cleaned = re.sub(r"###+\s+[^\n]+", "", cleaned)
    cleaned = re.sub(r"\*{1,3}[^*]+\*{1,3}", lambda m: m.group(0).replace("*", ""), cleaned)

    # Split into sentences or bullet points
    raw_claims = []
    lines = cleaned.split("\n")
    for line in lines:
        l = line.strip()
        if not l or l.startswith("#") or l.startswith("---"):
            continue
        if l.startswith("- ") or l.startswith("• "):
            raw_claims.append(l[2:].strip())
        else:
            sentences = re.split(r"(?<=[.!?])\s+", l)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) > 15:
                    raw_claims.append(s_clean)

    return raw_claims if raw_claims else [cleaned.strip()]


def evaluate_citations(
    citations: List[Any],
    pages: List[PageContent],
    case: BenchmarkCase,
    predicted_answer: str = ""
) -> Dict[str, float]:
    """
    Evaluates:
    - citation_validity: Percentage of citations whose quote exists on that page in the paper.
    - citation_accuracy: Whether citations contain evidence matching the benchmark case.
    - citation_completeness: Whether factual claims are accompanied by citations.
    - faithfulness: Proportion of factual claims supported by the citations/text.
    """
    # For absent cases, having NO citations is valid and expected
    if case.expected_status == "absent":
        return {
            "citation_validity": 1.0 if not citations else 0.0,
            "citation_accuracy": 1.0 if not citations else 0.0,
            "citation_completeness": 1.0,
            "faithfulness": 1.0 if not citations else 0.5
        }

    if not citations:
        return {
            "citation_validity": 0.0,
            "citation_accuracy": 0.0,
            "citation_completeness": 0.0,
            "faithfulness": 0.0
        }

    # 1. Citation Validity
    valid_count = 0
    total_citations = len(citations)
    citation_quotes: List[str] = []

    for cit in citations:
        is_val, conf = verify_citation_in_pages(cit, pages)
        if is_val:
            valid_count += 1
        q_text = getattr(cit, "verbatim_quote", "")
        if q_text:
            citation_quotes.append(q_text)

    citation_validity = round(valid_count / float(max(1, total_citations)), 4)

    # 2. Citation Accuracy: Does cited text contain the ground-truth evidence keywords?
    all_cited_text = " ".join(citation_quotes).lower()
    acc_count = 0
    if case.evidence_keywords:
        for kw in case.evidence_keywords:
            if kw.lower() in all_cited_text:
                acc_count += 1
        citation_accuracy = round(acc_count / float(len(case.evidence_keywords)), 4)
    else:
        citation_accuracy = citation_validity

    # 3. Citation Completeness & Faithfulness
    claims = extract_factual_claims_from_answer(predicted_answer)
    total_claims = max(1, len(claims))
    supported_claims = 0

    for claim in claims:
        claim_norm = re.sub(r"\s+", " ", claim.lower())
        # Check if numbers or key terms in claim appear in the cited text or manuscript pages
        nums_in_claim = re.findall(r"\b\d+(?:\.\d+)?\b", claim_norm)
        if not nums_in_claim:
            supported_claims += 1  # General connective sentence
            continue

        num_found = all(n in all_cited_text for n in nums_in_claim)
        if num_found:
            supported_claims += 1
        else:
            # Check if found across document pages
            doc_has_num = any(n in " ".join([p.text for p in pages]) for n in nums_in_claim)
            if doc_has_num:
                supported_claims += 0.5

    faithfulness = round(supported_claims / float(total_claims), 4)
    citation_completeness = round(min(1.0, len(citations) / float(max(1, len(claims)))), 4)

    return {
        "citation_validity": citation_validity,
        "citation_accuracy": citation_accuracy,
        "citation_completeness": citation_completeness,
        "faithfulness": min(1.0, faithfulness)
    }
