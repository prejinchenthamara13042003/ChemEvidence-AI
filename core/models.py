"""
Pydantic data models for ChemEvidence AI.
Ensures rigorous schema compliance, typed evidence attribution,
and structured confidence/conflict tracking.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class EvidenceCitation(BaseModel):
    """Direct verifiable citation from the original document."""
    page_number: int = Field(description="Exact 1-based page number where the evidence appears")
    section: str = Field(default="Main Text", description="Section heading (e.g., Abstract, Experimental, Results, Table 1)")
    verbatim_quote: str = Field(description="Exact verbatim excerpt from the document text supporting the claim")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Confidence score of the evidence extraction")


class CompoundInfo(BaseModel):
    """Information regarding a chemical entity or compound identified in the paper."""
    compound_id: str = Field(description="Internal ID or label in the paper (e.g., Compound 3b, 4a, Inhibitor X)")
    name: str = Field(description="Chemical or common name, or IUPAC representation")
    formula: Optional[str] = Field(default=None, description="Chemical molecular formula (e.g., C18H21N3O2)")
    smiles: Optional[str] = Field(default=None, description="SMILES notation if reported or identified")
    chemical_class: Optional[str] = Field(default=None, description="Chemical class (e.g., Pyrimidine, Indole, Quinazoline)")
    pubchem_cid: Optional[int] = Field(default=None, description="PubChem Compound ID if mapped")
    molecular_weight: Optional[float] = Field(default=None, description="Molecular weight in g/mol")
    structure_svg: Optional[str] = Field(default=None, description="2D SVG string of molecular structure")
    evidence: EvidenceCitation = Field(description="Supporting evidence citation")


class ChemicalProperty(BaseModel):
    """Reported chemical, physical, or spectroscopic property."""
    compound_id: Optional[str] = Field(default=None, description="Associated compound identifier")
    parameter: str = Field(description="Property measured (e.g., Isolated Yield, Melting Point, LogP, Purity, Rf)")
    value: str = Field(description="Reported value or numerical range")
    unit: Optional[str] = Field(default=None, description="Measurement unit (e.g., %, °C, mg/mL, min)")
    evidence: EvidenceCitation = Field(description="Supporting evidence citation")


class BioactivityResult(BaseModel):
    """Biological assay outcome, enzyme inhibition, or cellular screening result."""
    compound_id: Optional[str] = Field(default=None, description="Associated compound identifier")
    assay_type: str = Field(description="Metric used (e.g., IC50, EC50, Ki, % Inhibition, GI50)")
    target: Optional[str] = Field(default="Enzyme Target", description="Biological target enzyme, protein, or receptor (e.g., EGFR Kinase, CDK4)")
    value: str = Field(description="Numerical activity metric (e.g., 14.2, 0.45)")
    unit: Optional[str] = Field(default="nM", description="Unit of measurement (e.g., nM, µM, mM, %)")
    cell_line: Optional[str] = Field(default=None, description="Cell line or organism if applicable (e.g., A549, HeLa)")
    evidence: EvidenceCitation = Field(description="Supporting evidence citation")


class ExperimentalCondition(BaseModel):
    """Reaction parameters, solvent, catalyst, temperature, and duration."""
    reaction_step: str = Field(description="Name or description of synthetic transformation (e.g., Cross-coupling, Alkylation)")
    solvent: Optional[str] = Field(default=None, description="Reaction solvent (e.g., DMF, THF, CH2Cl2, Toluene)")
    catalyst: Optional[str] = Field(default=None, description="Catalyst, ligand, or base (e.g., Pd(PPh3)4, K2CO3)")
    temperature: Optional[str] = Field(default=None, description="Reaction temperature (e.g., 80 °C, RT, reflux)")
    time: Optional[str] = Field(default=None, description="Reaction duration (e.g., 4 h, overnight, 30 min)")
    yield_reported: Optional[str] = Field(default=None, description="Reported conversion or yield")
    evidence: EvidenceCitation = Field(description="Supporting evidence citation")


class Methodology(BaseModel):
    """Analytical, synthetic, purification, or computational technique."""
    technique: str = Field(description="Technique name (e.g., 1H-NMR, Flash Chromatography, Molecular Docking, HPLC)")
    description: str = Field(description="Summary of methodology execution or instrument details")
    evidence: EvidenceCitation = Field(description="Supporting evidence citation")


class SignificantFinding(BaseModel):
    """Key scientific takeaway, conclusion, or SAR discovery."""
    finding: str = Field(description="Scientific finding or outcome statement")
    category: str = Field(default="General", description="Category: SAR Insight, Selectivity, Potency, Novel Mechanism, Efficiency")
    evidence: EvidenceCitation = Field(description="Supporting evidence citation")


class ConflictAlert(BaseModel):
    """Flagged discrepancy or conflicting data between sections of the paper."""
    topic: str = Field(description="Topic under conflict (e.g., Compound 3b Yield Discrepancy)")
    description: str = Field(description="Explanation of the discrepancy or conflict")
    claim_a: str = Field(description="First reported statement or value")
    citation_a: EvidenceCitation = Field(description="Citation for first statement")
    claim_b: str = Field(description="Contradicting statement or value found elsewhere in the paper")
    citation_b: EvidenceCitation = Field(description="Citation for conflicting statement")
    severity: Literal["Warning", "High Discrepancy", "Ambiguous"] = "Warning"
    resolution_note: str = Field(default="Flagged for human researcher review.", description="Recommendation or guidance")


class QAResponse(BaseModel):
    """Response structure for interactive evidence-aware Q&A."""
    question: str = Field(description="The user's query")
    answer: str = Field(description="Synthesized answer grounded in document evidence, or explicit absence notice")
    status: Literal["found", "absent", "conflicting", "external_literature"] = Field(
        default="found",
        description="Whether evidence was successfully found, absent from the paper, conflicting/ambiguous, or answered via external literature"
    )
    citations: List[EvidenceCitation] = Field(default_factory=list, description="Supporting verifiable citations")
    conflicts: List[ConflictAlert] = Field(default_factory=list, description="Any detected conflicts or discrepancies")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall confidence level in the answer")
    related_papers: List[Dict[str, Any]] = Field(default_factory=list, description="Related external research publications discovered")
    can_expand_external: bool = Field(default=False, description="True if query is absent from document and eligible for broader literature exploration")


class PaperMetadata(BaseModel):
    """Extracted or inferred publication metadata."""
    title: str = Field(default="Untitled Chemistry Document")
    authors: Optional[str] = Field(default=None)
    journal_or_doi: Optional[str] = Field(default=None)
    page_count: int = 1
    total_characters: int = 0
    total_words: int = 0


# =========================================================
# Advanced Extensions: Tables, Reactions, SAR, Graph, Gaps
# =========================================================

class ExtractedTable(BaseModel):
    """Structured representation of a table extracted from the publication."""
    table_id: str = Field(description="Table identifier (e.g., Table 1, Table 2)")
    title: str = Field(default="Chemistry Data Table", description="Table title or caption")
    headers: List[str] = Field(default_factory=list, description="Column header labels")
    rows: List[List[str]] = Field(default_factory=list, description="Row values matrix")
    page_number: int = Field(default=1, description="Page where table is located")
    section: str = Field(default="Results", description="Section containing the table")
    evidence: EvidenceCitation = Field(description="Verifiable source quote for the table")


class ReactionStep(BaseModel):
    """A discrete synthetic transformation step in a chemical reaction pathway."""
    step_number: int = Field(default=1, description="Sequential step index in synthesis")
    reaction_name: str = Field(default="Coupling / Functionalization", description="Type of reaction")
    starting_material: Optional[str] = Field(default=None, description="Starting reactant or intermediate")
    reagents: Optional[str] = Field(default=None, description="Reagents, catalysts, and additives")
    solvent: Optional[str] = Field(default=None, description="Solvent system")
    temperature: Optional[str] = Field(default=None, description="Reaction temperature")
    time: Optional[str] = Field(default=None, description="Reaction duration")
    product: str = Field(description="Product or intermediate formed")
    yield_percent: Optional[str] = Field(default=None, description="Isolated or analytical yield")
    evidence: EvidenceCitation = Field(description="Citation for synthetic step")


class ReactionPathway(BaseModel):
    """Multi-step synthetic route or Scheme leading to target molecule(s)."""
    scheme_id: str = Field(default="Scheme 1", description="Scheme or Pathway identifier")
    title: str = Field(default="Synthetic Route", description="Pathway description")
    target_compound: str = Field(description="Final lead compound or target molecule")
    steps: List[ReactionStep] = Field(default_factory=list, description="Sequential synthetic reaction steps")
    overall_yield: Optional[str] = Field(default=None, description="Estimated or reported overall yield")
    evidence: EvidenceCitation = Field(description="Supporting citation")


class NormalizedEntity(BaseModel):
    """Canonicalized chemical entity with synonym harmonization and database IDs."""
    original_text: str = Field(description="Raw string mentioned in the paper")
    canonical_name: str = Field(description="Standardized name")
    entity_type: Literal["Compound", "Target", "Solvent", "Catalyst", "Reagent"] = "Compound"
    formula: Optional[str] = Field(default=None)
    molecular_weight: Optional[float] = Field(default=None)
    smiles: Optional[str] = Field(default=None)
    inchi_key: Optional[str] = Field(default=None)
    pubchem_cid: Optional[int] = Field(default=None)
    chembl_id: Optional[str] = Field(default=None)
    synonyms: List[str] = Field(default_factory=list)


class KnowledgeGraphNode(BaseModel):
    """Node in the chemical literature knowledge graph."""
    id: str
    label: str
    category: Literal["Compound", "Target", "Reaction", "Property", "Paper", "CellLine", "Catalyst"]
    properties: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphEdge(BaseModel):
    """Relationship edge connecting entities in the knowledge graph."""
    source: str
    target: str
    relation: str  # e.g. "INHIBITS", "SYNTHESIZED_VIA", "TESTED_IN", "CATALYZED_BY", "EXHIBITS_PROPERTY"
    label: Optional[str] = None
    weight: float = 1.0


class KnowledgeGraphData(BaseModel):
    """Knowledge graph payload ready for interactive network rendering."""
    nodes: List[KnowledgeGraphNode] = Field(default_factory=list)
    edges: List[KnowledgeGraphEdge] = Field(default_factory=list)


class SARDataPoint(BaseModel):
    """Data point in a Structure-Activity Relationship (SAR) series."""
    compound_id: str
    substituent_position: Optional[str] = Field(default=None, description="e.g. C-5, 4-position, R1")
    substituent_group: Optional[str] = Field(default=None, description="e.g. -OEt, -OMe, -CF3, -Me, -H")
    ic50_nm: Optional[float] = Field(default=None, description="Numeric IC50 in nM")
    pic50: Optional[float] = Field(default=None, description="-log10(IC50 in M) potency score")
    cellular_ic50_um: Optional[float] = Field(default=None, description="Cellular IC50 in µM")
    yield_percent: Optional[float] = Field(default=None, description="Isolated yield %")
    fold_improvement: Optional[float] = Field(default=None, description="Fold improvement over baseline")
    evidence: EvidenceCitation


class SARAnalysis(BaseModel):
    """Complete Structure-Activity Relationship series analysis."""
    series_name: str = Field(default="Core SAR Series")
    core_scaffold: str = Field(default="Small Molecule Scaffold")
    target: str = Field(default="Biological Target")
    datapoints: List[SARDataPoint] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    optimal_lead: Optional[str] = None


class EvidenceConfidenceBreakdown(BaseModel):
    """Empirical multi-factor audit score of extraction confidence."""
    overall_score: float = Field(ge=0.0, le=100.0, description="Overall confidence 0-100%")
    tier: Literal["High (A+)", "Strong (A)", "Moderate (B)", "Tentative (C)"]
    verbatim_match_score: float = Field(description="Exact textual match fidelity")
    proximity_score: float = Field(description="Proximity between compound entity and assay metric")
    numeric_precision_score: float = Field(description="Proper unit and error margin verification")
    cross_validation_score: float = Field(description="Multi-section agreement verification")
    rationale: str = Field(description="Detailed explanation of the confidence score")


class ResearchGap(BaseModel):
    """Identified blind spot, missing assay, or unexplored avenue in the literature."""
    category: Literal[
        "ADME & Pharmacokinetics",
        "Target Selectivity & Mutants",
        "In Vivo Translation",
        "Safety & Toxicity",
        "Analytical Characterization",
        "Chemical Space & Analogs"
    ]
    title: str = Field(description="Headline of the identified gap")
    description: str = Field(description="Detailed scientific description of what was omitted")
    severity: Literal["Critical", "Important", "Exploratory"] = "Important"
    recommendation: str = Field(description="Suggested next experimental step or investigation")


class MultiPaperComparisonResult(BaseModel):
    """Cross-paper comparative synthesis across multiple uploaded documents."""
    papers: List[PaperMetadata] = Field(default_factory=list)
    lead_comparison_table: List[Dict[str, Any]] = Field(default_factory=list)
    common_targets: List[str] = Field(default_factory=list)
    synthetic_route_comparison: List[Dict[str, Any]] = Field(default_factory=list)
    cross_study_discrepancies: List[ConflictAlert] = Field(default_factory=list)
    comparative_synthesis: str = Field(description="Holistic comparison summary across literature")


class PaperAnalysisResult(BaseModel):
    """Complete structured output containing all extracted chemical information and audit evidence."""
    metadata: PaperMetadata
    executive_summary: str = Field(description="Executive scientific summary of the document")
    compounds: List[CompoundInfo] = Field(default_factory=list)
    properties: List[ChemicalProperty] = Field(default_factory=list)
    bioactivities: List[BioactivityResult] = Field(default_factory=list)
    conditions: List[ExperimentalCondition] = Field(default_factory=list)
    methodologies: List[Methodology] = Field(default_factory=list)
    findings: List[SignificantFinding] = Field(default_factory=list)
    conflicts_detected: List[ConflictAlert] = Field(default_factory=list)

    # Advanced Extensions
    tables: List[ExtractedTable] = Field(default_factory=list)
    reaction_pathways: List[ReactionPathway] = Field(default_factory=list)
    normalized_entities: List[NormalizedEntity] = Field(default_factory=list)
    sar_analyses: List[SARAnalysis] = Field(default_factory=list)
    research_gaps: List[ResearchGap] = Field(default_factory=list)
    confidence_breakdown: Optional[EvidenceConfidenceBreakdown] = None
    knowledge_graph: Optional[KnowledgeGraphData] = None


class MultilingualBotResponse(BaseModel):
    """Structured response from the Multilingual ChemBot assistant."""
    question: str = Field(description="The user's query")
    detected_language: str = Field(description="Name of detected or selected language (e.g. Malayalam, Hindi, English)")
    language_code: str = Field(default="en", description="ISO code for the language (e.g. ml, hi, ta, de, es, fr, ar, zh, en)")
    simple_answer: str = Field(description="Plain-language direct answer")
    key_facts: List[str] = Field(default_factory=list, description="Key chemical facts, numbers, and parameters")
    why_it_matters: str = Field(description="Everyday analogy or practical significance")
    evidence_source: Optional[str] = Field(default=None, description="Page reference, quote, or literature context")
    full_formatted_text: str = Field(description="Complete formatted markdown response in the user's language")
    model_used: str = Field(default="gemini-3.8-flash", description="Underlying model or engine used")

    @property
    def in_depth_markdown(self) -> str:
        """Formatted in-depth breakdown for users who request deep details."""
        parts = []
        if self.key_facts:
            facts_str = "\n".join([f"- {f}" for f in self.key_facts])
            parts.append(f"**🔬 Key Chemical Facts & Numbers:**\n{facts_str}")
        if self.why_it_matters:
            parts.append(f"**💡 Mechanism & Significance:**\n{self.why_it_matters}")
        if self.evidence_source:
            parts.append(f"**📚 Paper Evidence & Source:**\n{self.evidence_source}")
        return "\n\n".join(parts)

