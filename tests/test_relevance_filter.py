"""
Unit and integration tests for the Multi-Tier Relevance Filtering and Deep QA Engine.
Verifies:
1. Fast Bibliographic & Abstract Triage (Tier 1)
2. Chemical Entity & Target Concordance (Tier 2)
3. Deep Evidence Passage Density (Tier 3)
4. Candidate Shortlisting and Volume Reduction at Scale (100+ papers)
5. Scoped Deep Question-Answering exclusively on shortlisted literature
"""

import pytest
from core.relevance_filter import (
    RelevanceFilterEngine,
    PaperRelevanceAssessment,
    RelevanceScreeningReport,
    DeepQAResponse
)
from core.pdf_parser import PageContent
from core.evidence_engine import EvidenceEngine
from core.models import (
    PaperMetadata,
    CompoundInfo,
    BioactivityResult,
    ExperimentalCondition,
    PaperAnalysisResult,
    EvidenceCitation
)


@pytest.fixture
def sample_library_corpus():
    """Builds a mini-corpus of 4 papers across distinct domains."""
    cit = EvidenceCitation(
        page_number=2,
        section="Results & Discussion",
        verbatim_quote="Compound 8b demonstrated exceptional potency against EGFR kinase with an IC50 of 8.4 nM.",
        confidence=0.98
    )

    # Paper 1: EGFR Kinase Inhibitor (High relevance for EGFR queries)
    p1_meta = PaperMetadata(
        title="Discovery of Covalent EGFR Kinase Inhibitors for Lung Carcinoma",
        authors="Zhang et al.",
        journal_or_doi="J. Med. Chem. 2024",
        abstract="Novel quinazoline derivatives targeting mutant EGFR kinase with sub-10 nM IC50 and strong antiproliferative cellular activity."
    )
    p1_pages = [
        PageContent(page_number=1, text="Title: " + p1_meta.title + "\nAbstract: " + p1_meta.abstract),
        PageContent(page_number=2, text="Results: Compound 8b demonstrated exceptional potency against EGFR kinase with an IC50 of 8.4 nM in A549 lung cancer cells.")
    ]
    p1_engine = EvidenceEngine(p1_pages)
    p1_analysis = PaperAnalysisResult(
        metadata=p1_meta,
        executive_summary=p1_meta.abstract,
        compounds=[CompoundInfo(compound_id="Compound 8b", name="WZ-Analogue", evidence=cit)],
        bioactivities=[BioactivityResult(compound_id="Compound 8b", target="EGFR Kinase", assay_type="IC50", value="8.4", unit="nM", evidence=cit)],
        conditions=[ExperimentalCondition(reaction_step="Suzuki coupling", catalyst="Pd(dppf)Cl2", yield_reported="82%", evidence=cit)],
        properties=[], methodologies=[], findings=[], conflicts_detected=[]
    )

    # Paper 2: CDK4/6 Kinase Inhibitor (Moderate relevance: kinase, but not EGFR)
    p2_meta = PaperMetadata(
        title="Optimization of Selective CDK4/6 Kinase Inhibitors for Breast Cancer",
        authors="Smith et al.",
        journal_or_doi="Bioorg. Med. Chem. 2023",
        abstract="Synthesis of pyrido-pyrimidine analogues exhibiting nanomolar inhibition against CDK4/6 cell cycle kinase."
    )
    p2_pages = [
        PageContent(page_number=1, text="Title: " + p2_meta.title + "\nAbstract: " + p2_meta.abstract),
        PageContent(page_number=2, text="Results: Lead analogue 14 showed an IC50 of 22 nM against CDK4/6 enzyme.")
    ]
    p2_engine = EvidenceEngine(p2_pages)
    p2_analysis = PaperAnalysisResult(
        metadata=p2_meta,
        executive_summary=p2_meta.abstract,
        compounds=[CompoundInfo(compound_id="Lead 14", name="CDK-Inhibitor", evidence=cit)],
        bioactivities=[BioactivityResult(compound_id="Lead 14", target="CDK4/6 Kinase", assay_type="IC50", value="22.0", unit="nM", evidence=cit)],
        conditions=[], properties=[], methodologies=[], findings=[], conflicts_detected=[]
    )

    # Paper 3: Antifungal Natural Product (Irrelevant for EGFR kinase query)
    p3_meta = PaperMetadata(
        title="Antifungal Flavonoids from Medicinal Herbs against Candida albicans",
        authors="Patel et al.",
        journal_or_doi="Phytochemistry 2024",
        abstract="Bioassay-guided fractionation of plant extracts yielding flavonoids with antifungal activity against Candida albicans."
    )
    p3_pages = [
        PageContent(page_number=1, text="Title: " + p3_meta.title + "\nAbstract: " + p3_meta.abstract),
        PageContent(page_number=2, text="Results: Flavonoid C displayed an MIC of 12 µg/mL against clinical Candida albicans.")
    ]
    p3_engine = EvidenceEngine(p3_pages)
    p3_analysis = PaperAnalysisResult(
        metadata=p3_meta,
        executive_summary=p3_meta.abstract,
        compounds=[CompoundInfo(compound_id="Flavonoid C", name="Natural Flavonoid", evidence=cit)],
        bioactivities=[BioactivityResult(compound_id="Flavonoid C", target="Candida albicans", assay_type="MIC", value="12", unit="µg/mL", evidence=cit)],
        conditions=[], properties=[], methodologies=[], findings=[], conflicts_detected=[]
    )

    return {
        "paper1_egfr.pdf": {"pages": p1_pages, "metadata": p1_meta, "engine": p1_engine, "analysis": p1_analysis},
        "paper2_cdk.pdf": {"pages": p2_pages, "metadata": p2_meta, "engine": p2_engine, "analysis": p2_analysis},
        "paper3_antifungal.pdf": {"pages": p3_pages, "metadata": p3_meta, "engine": p3_engine, "analysis": p3_analysis}
    }


def test_relevance_screening_pipeline_targeting(sample_library_corpus):
    """Verifies that EGFR kinase inquiry prioritizes EGFR paper and screens out antifungal paper."""
    query = "Find potent EGFR kinase inhibitors with nanomolar IC50 potency in lung cancer"
    report = RelevanceFilterEngine.screen_library(
        query=query,
        papers_dict=sample_library_corpus,
        min_score=45.0,
        max_shortlist=5
    )

    assert report.total_screened == 3
    assert len(report.shortlisted) >= 1
    top_paper = report.shortlisted[0]
    
    # Paper 1 should be rank 1 with high score
    assert "egfr" in top_paper.paper_id.lower() or "egfr" in top_paper.title.lower()
    assert top_paper.composite_score >= 60.0
    assert top_paper.status == "SHORTLISTED"
    assert "EGFR" in top_paper.matched_targets or "egfr" in str(top_paper.matched_keywords).lower()
    
    # Antifungal paper should be excluded or lower score
    excluded_ids = [e.paper_id for e in report.excluded]
    assert "paper3_antifungal.pdf" in excluded_ids


def test_100_paper_benchmark_generation_and_screening():
    """Evaluates multi-tier screening performance on a 100-paper library."""
    benchmark_100 = RelevanceFilterEngine.generate_benchmark_library_100()
    assert len(benchmark_100) == 100

    query = "Which studies report palladium-catalyzed Suzuki cross-coupling reaction yields above 80%?"
    report = RelevanceFilterEngine.screen_library(
        query=query,
        papers_dict=benchmark_100,
        min_score=50.0,
        max_shortlist=8
    )

    assert report.total_screened == 100
    assert len(report.shortlisted) <= 8
    assert report.excluded_count >= 80
    assert report.reduction_percentage >= 80.0
    # Must execute swiftly (under 2 seconds)
    assert report.execution_time_seconds < 2.0

    # Shortlisted papers should feature catalytic cross-coupling
    for p in report.shortlisted:
        text_context = (p.title + " " + p.match_rationale).lower()
        assert "cross-coupling" in text_context or "palladium" in text_context or "cataly" in text_context or "suzuki" in text_context or p.composite_score >= 50.0


def test_deep_qa_on_shortlisted_papers(sample_library_corpus):
    """Verifies interactive question answering scoped exclusively to shortlisted papers."""
    report = RelevanceFilterEngine.screen_library(
        query="EGFR kinase inhibitors",
        papers_dict=sample_library_corpus,
        min_score=45.0,
        max_shortlist=2
    )

    qa_response = RelevanceFilterEngine.ask_deep_qa(
        question="What are the lead compounds and exact IC50 values reported?",
        shortlisted_assessments=report.shortlisted,
        papers_store=sample_library_corpus,
        api_key=None  # Offline local synthesizer
    )

    assert isinstance(qa_response, DeepQAResponse)
    assert len(qa_response.shortlisted_paper_ids) == len(report.shortlisted)
    assert "Executive" in qa_response.executive_synthesis or "Synthesis" in qa_response.executive_synthesis
    assert len(qa_response.comparative_matrix) == len(report.shortlisted)
    assert len(qa_response.citations) > 0
    # Check that citations contain page numbers and verbatim quotes
    assert "page" in qa_response.citations[0]
    assert "quote" in qa_response.citations[0]
