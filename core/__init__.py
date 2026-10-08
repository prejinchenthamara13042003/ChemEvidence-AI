"""
ChemEvidence AI Core Module.
Comprehensive evidence-grounded scientific literature analysis suite.
"""

from core.models import (
    PaperAnalysisResult,
    PaperMetadata,
    CompoundInfo,
    ChemicalProperty,
    BioactivityResult,
    ExperimentalCondition,
    Methodology,
    SignificantFinding,
    ConflictAlert,
    EvidenceCitation,
    QAResponse,
    ExtractedTable,
    ReactionStep,
    ReactionPathway,
    NormalizedEntity,
    KnowledgeGraphNode,
    KnowledgeGraphEdge,
    KnowledgeGraphData,
    SARDataPoint,
    SARAnalysis,
    EvidenceConfidenceBreakdown,
    ResearchGap,
    MultiPaperComparisonResult,
    MultilingualBotResponse
)
from core.pdf_parser import DocumentParser, PageContent
from core.evidence_engine import EvidenceEngine, EvidenceChunk
from core.pubchem_service import PubChemService
from core.chemistry_extractor import ChemistryExtractor
from core.qa_system import QASystem
from core.multilingual_bot import MultilingualChemBot
from core.table_extractor import TableExtractor
from core.reaction_engine import ReactionPathwayExtractor
from core.entity_normalizer import EntityNormalizer
from core.sar_engine import SAREngine
from core.knowledge_graph import KnowledgeGraphBuilder
from core.confidence_scorer import ConfidenceScorer
from core.gap_detector import ResearchGapDetector
from core.multi_paper_comparator import MultiPaperComparator

__all__ = [
    "PaperAnalysisResult",
    "PaperMetadata",
    "CompoundInfo",
    "ChemicalProperty",
    "BioactivityResult",
    "ExperimentalCondition",
    "Methodology",
    "SignificantFinding",
    "ConflictAlert",
    "EvidenceCitation",
    "QAResponse",
    "ExtractedTable",
    "ReactionStep",
    "ReactionPathway",
    "NormalizedEntity",
    "KnowledgeGraphNode",
    "KnowledgeGraphEdge",
    "KnowledgeGraphData",
    "SARDataPoint",
    "SARAnalysis",
    "EvidenceConfidenceBreakdown",
    "ResearchGap",
    "MultiPaperComparisonResult",
    "MultilingualBotResponse",
    "DocumentParser",
    "PageContent",
    "EvidenceEngine",
    "EvidenceChunk",
    "PubChemService",
    "ChemistryExtractor",
    "QASystem",
    "MultilingualChemBot",
    "TableExtractor",
    "ReactionPathwayExtractor",
    "EntityNormalizer",
    "SAREngine",
    "KnowledgeGraphBuilder",
    "ConfidenceScorer",
    "ResearchGapDetector",
    "MultiPaperComparator"
]
