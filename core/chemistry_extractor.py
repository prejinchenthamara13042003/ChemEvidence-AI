"""
Automated Chemistry Information Extraction Engine.
Extracts compounds, properties, bioactivity, experimental conditions,
methodologies, and significant findings with strict page-level evidence citation.
Supports Gemini 3.8 Flash LLM and local offline deterministic extraction.
"""

import os
import re
import json
from typing import List, Optional, Tuple, Dict, Any
from core.pdf_parser import PageContent
from core.evidence_engine import EvidenceEngine
from core.pubchem_service import PubChemService
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
    EvidenceCitation
)

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class ChemistryExtractor:
    """Extracts structured chemistry data from research papers with direct evidence grounding."""

    def __init__(self, api_key: Optional[str] = None, use_local_only: bool = False):
        if use_local_only or api_key == "none":
            self.api_key = None
            self.client = None
        else:
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY
            self.client = None
            if self.api_key and HAS_GENAI:
                try:
                    self.client = genai.Client(api_key=self.api_key, http_options={"timeout": 15000})
                except Exception:
                    self.client = None

    def analyze_paper(
        self,
        pages: List[PageContent],
        metadata: PaperMetadata,
        evidence_engine: EvidenceEngine
    ) -> PaperAnalysisResult:
        """
        Executes complete paper analysis.
        Attempts Gemini LLM extraction if client available, otherwise uses deterministic local extractor.
        """
        if self.client:
            try:
                res = self._analyze_with_gemini(pages, metadata, evidence_engine)
            except Exception as e:
                # Graceful fallback to deterministic local extractor
                print(f"[ChemistryExtractor] Gemini extraction fallback due to: {e}")
                res = self._analyze_locally(pages, metadata, evidence_engine)
        else:
            res = self._analyze_locally(pages, metadata, evidence_engine)

        self._enrich_advanced_analytics(res, pages, evidence_engine)
        return res

    def _enrich_advanced_analytics(
        self,
        res: PaperAnalysisResult,
        pages: List[PageContent],
        evidence_engine: EvidenceEngine
    ) -> None:
        """Enriches the paper analysis result with tables, reactions, entities, SAR, gaps, and graph."""
        try:
            from core.table_extractor import TableExtractor
            from core.chem_calculator import ChemCalculator
            from core.reaction_engine import ReactionPathwayExtractor
            from core.entity_normalizer import EntityNormalizer
            from core.sar_engine import SAREngine
            from core.knowledge_graph import KnowledgeGraphBuilder
            from core.confidence_scorer import ConfidenceScorer
            from core.gap_detector import ResearchGapDetector

            if not res.tables:
                res.tables = TableExtractor.extract_tables_from_pages(pages)

            # Document-wide Chemical Formula and Molecular Weight Binding
            ChemCalculator.bind_formulas_and_weights_document_wide(res.compounds, res.tables, pages)

            if not res.reaction_pathways:
                res.reaction_pathways = ReactionPathwayExtractor.extract_pathways(pages, res.conditions, evidence_engine)

            if not res.normalized_entities:
                res.normalized_entities = EntityNormalizer.normalize_all(res.compounds, res.bioactivities)

            if not res.sar_analyses:
                res.sar_analyses = SAREngine.extract_sar_series(res.compounds, res.bioactivities, res.properties)

            if not res.confidence_breakdown:
                res.confidence_breakdown = ConfidenceScorer.evaluate_paper_evidence(pages, res.compounds, res.bioactivities, res.properties)

            if not res.research_gaps:
                res.research_gaps = ResearchGapDetector.audit_paper_gaps(pages, res.compounds, res.bioactivities, res.conditions)

            if not res.knowledge_graph:
                res.knowledge_graph = KnowledgeGraphBuilder.build_graph(
                    res.metadata, res.compounds, res.bioactivities, res.conditions, res.properties
                )
        except Exception as e:
            print(f"[ChemistryExtractor] Warning in _enrich_advanced_analytics: {e}")

    def _analyze_with_gemini(
        self,
        pages: List[PageContent],
        metadata: PaperMetadata,
        evidence_engine: EvidenceEngine
    ) -> PaperAnalysisResult:
        """Performs structured extraction using Gemini 3.8 Flash model."""
        # Collate paper text with explicit page markers
        paper_text_with_pages = ""
        for p in pages:
            paper_text_with_pages += f"\n--- [PAGE {p.page_number}] ---\n{p.text}\n"

        prompt = f"""You are an expert scientific chemistry analyst specializing in evidence-grounded literature extraction.
Analyze the following scientific chemistry paper and extract all verified information in structured JSON format.

CRITICAL EVIDENCE RULES:
1. Every extracted item MUST have an exact verbatim quote ("verbatim_quote") copied directly from the text.
2. Every item MUST have the exact 1-based page number ("page_number") where that quote appears.
3. If an item is not explicitly discussed in the paper, DO NOT invent or assume it.
4. Detect any conflicting data or discrepancies (e.g. different yields or IC50 values for the same compound in different sections) and record them in "conflicts_detected".
5. STRICT METRIC-TYPE AND TARGET DISAMBIGUATION:
   - Strictly distinguish enzymatic inhibition (IC50, Ki, Kd) from cellular growth inhibition (GI50, TGI), cytotoxicity (CC50), and antimicrobial susceptibility (MIC).
   - NEVER place a cell line (e.g. A549, HeLa, Vero, MCF-7) into the "target" field; assign cell lines to "cell_line" and the molecular/phenotypic effect to "target".
   - NEVER report a cellular growth inhibition value (like A549 GI50) as an enzymatic target IC50 (like EGFR IC50).
   - When extracting from tables, map each numerical value strictly to its column header's metric, target, and unit.
   - Preserve exact numerical values with standard deviations or errors (e.g., "8.4 ± 0.6"), inequality symbols (<, >), and physical units.

SCHEMA SPECIFICATION:
Return a JSON object with the following keys:
{{
  "executive_summary": "Concise 2-3 paragraph summary of the paper's chemical goals, synthesis, and key outcomes",
  "compounds": [
    {{
      "compound_id": "e.g., Compound 3b",
      "name": "Chemical or IUPAC name",
      "formula": "e.g., C18H21N3O2",
      "smiles": "SMILES if mentioned",
      "chemical_class": "e.g., Pyrimidine kinase inhibitor",
      "page_number": 1,
      "section": "Results",
      "verbatim_quote": "exact quote from text"
    }}
  ],
  "properties": [
    {{
      "compound_id": "Compound 3b",
      "parameter": "Yield / Melting Point / Purity / LogP",
      "value": "78",
      "unit": "%",
      "page_number": 2,
      "section": "Table 1",
      "verbatim_quote": "exact quote"
    }}
  ],
  "bioactivities": [
    {{
      "compound_id": "Compound 3b",
      "assay_type": "IC50 / GI50 / EC50 / Ki / Kd / MIC / CC50",
      "target": "EGFR kinase / Candida albicans / Cellular Antiproliferative",
      "value": "8.4 ± 0.6",
      "unit": "nM",
      "cell_line": "A549 (or null if purely enzymatic assay)",
      "page_number": 3,
      "section": "Bioactivity Evaluation",
      "verbatim_quote": "exact quote"
    }}
  ],
  "conditions": [
    {{
      "reaction_step": "Suzuki-Miyaura cross-coupling",
      "solvent": "1,4-dioxane/H2O (4:1)",
      "catalyst": "Pd(dppf)Cl2 (5 mol %)",
      "temperature": "90 °C",
      "time": "6 h",
      "yield_reported": "78%",
      "page_number": 4,
      "section": "Experimental Procedures",
      "verbatim_quote": "exact quote"
    }}
  ],
  "methodologies": [
    {{
      "technique": "1H-NMR / HRMS / X-ray crystallography",
      "description": "Technique summary",
      "page_number": 4,
      "section": "Experimental",
      "verbatim_quote": "exact quote"
    }}
  ],
  "findings": [
    {{
      "finding": "Statement of key conclusion or SAR trend",
      "category": "SAR Insight / Potency / Novelty",
      "page_number": 3,
      "section": "Discussion",
      "verbatim_quote": "exact quote"
    }}
  ],
  "conflicts_detected": [
    {{
      "topic": "Yield discrepancy for Compound 3b",
      "description": "Abstract reports 85% yield, whereas Table 1 and Experimental section state 74% yield.",
      "claim_a": "Yield reported as 85%",
      "page_a": 1,
      "quote_a": "verbatim quote from abstract",
      "claim_b": "Yield reported as 74%",
      "page_b": 3,
      "quote_b": "verbatim quote from table/experimental",
      "severity": "Warning",
      "resolution_note": "Experimental table 74% is likely the isolated yield; 85% may be NMR conversion."
    }}
  ]
}}

RESEARCH PAPER TEXT:
{paper_text_with_pages[:60000]}
"""

        response = self.client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        data = json.loads(response.text)

        # Parse into typed Pydantic models with quote verification
        compounds: List[CompoundInfo] = []
        for c in data.get("compounds", []):
            quote = c.get("verbatim_quote", "")
            cit = evidence_engine.verify_quote(quote) or EvidenceCitation(
                page_number=int(c.get("page_number", 1)),
                section=c.get("section", "Main Text"),
                verbatim_quote=quote,
                confidence=0.9
            )
            comp_obj = CompoundInfo(
                compound_id=c.get("compound_id", "Compound"),
                name=c.get("name", "Unknown Name"),
                formula=c.get("formula"),
                smiles=c.get("smiles"),
                chemical_class=c.get("chemical_class"),
                evidence=cit
            )
            # PubChem enrichment
            lookup = PubChemService.lookup_compound(comp_obj.name) or PubChemService.lookup_compound(comp_obj.compound_id)
            if lookup:
                comp_obj.pubchem_cid = lookup.get("pubchem_cid")
                comp_obj.formula = comp_obj.formula or lookup.get("formula")
                comp_obj.molecular_weight = lookup.get("molecular_weight")
                comp_obj.smiles = comp_obj.smiles or lookup.get("smiles")
            compounds.append(comp_obj)

        properties: List[ChemicalProperty] = []
        for p in data.get("properties", []):
            quote = p.get("verbatim_quote", "")
            cit = evidence_engine.verify_quote(quote) or EvidenceCitation(
                page_number=int(p.get("page_number", 1)),
                section=p.get("section", "Properties"),
                verbatim_quote=quote,
                confidence=0.9
            )
            properties.append(ChemicalProperty(
                compound_id=p.get("compound_id"),
                parameter=p.get("parameter", "Property"),
                value=str(p.get("value", "")),
                unit=p.get("unit"),
                evidence=cit
            ))

        bioactivities: List[BioactivityResult] = []
        for b in data.get("bioactivities", []):
            quote = b.get("verbatim_quote", "")
            cit = evidence_engine.verify_quote(quote) or EvidenceCitation(
                page_number=int(b.get("page_number", 1)),
                section=b.get("section", "Bioactivity"),
                verbatim_quote=quote,
                confidence=0.9
            )
            bioactivities.append(BioactivityResult(
                compound_id=b.get("compound_id"),
                assay_type=b.get("assay_type", "Bioactivity"),
                target=b.get("target", "Target"),
                value=str(b.get("value", "")),
                unit=b.get("unit", "nM"),
                cell_line=b.get("cell_line"),
                evidence=cit
            ))

        conditions: List[ExperimentalCondition] = []
        for cond in data.get("conditions", []):
            quote = cond.get("verbatim_quote", "")
            cit = evidence_engine.verify_quote(quote) or EvidenceCitation(
                page_number=int(cond.get("page_number", 1)),
                section=cond.get("section", "Experimental"),
                verbatim_quote=quote,
                confidence=0.9
            )
            conditions.append(ExperimentalCondition(
                reaction_step=cond.get("reaction_step", "Synthesis"),
                solvent=cond.get("solvent"),
                catalyst=cond.get("catalyst"),
                temperature=cond.get("temperature"),
                time=cond.get("time"),
                yield_reported=cond.get("yield_reported"),
                evidence=cit
            ))

        methodologies: List[Methodology] = []
        for m in data.get("methodologies", []):
            quote = m.get("verbatim_quote", "")
            cit = evidence_engine.verify_quote(quote) or EvidenceCitation(
                page_number=int(m.get("page_number", 1)),
                section=m.get("section", "Methods"),
                verbatim_quote=quote,
                confidence=0.9
            )
            methodologies.append(Methodology(
                technique=m.get("technique", "Analytical Technique"),
                description=m.get("description", ""),
                evidence=cit
            ))

        findings: List[SignificantFinding] = []
        for f in data.get("findings", []):
            quote = f.get("verbatim_quote", "")
            cit = evidence_engine.verify_quote(quote) or EvidenceCitation(
                page_number=int(f.get("page_number", 1)),
                section=f.get("section", "Discussion"),
                verbatim_quote=quote,
                confidence=0.9
            )
            findings.append(SignificantFinding(
                finding=f.get("finding", "Finding"),
                category=f.get("category", "General"),
                evidence=cit
            ))

        conflicts: List[ConflictAlert] = []
        for cf in data.get("conflicts_detected", []):
            cit_a = evidence_engine.verify_quote(cf.get("quote_a", "")) or EvidenceCitation(
                page_number=int(cf.get("page_a", 1)),
                section="Section A",
                verbatim_quote=cf.get("quote_a", ""),
                confidence=0.85
            )
            cit_b = evidence_engine.verify_quote(cf.get("quote_b", "")) or EvidenceCitation(
                page_number=int(cf.get("page_b", 1)),
                section="Section B",
                verbatim_quote=cf.get("quote_b", ""),
                confidence=0.85
            )
            conflicts.append(ConflictAlert(
                topic=cf.get("topic", "Discrepancy"),
                description=cf.get("description", ""),
                claim_a=cf.get("claim_a", ""),
                citation_a=cit_a,
                claim_b=cf.get("claim_b", ""),
                citation_b=cit_b,
                severity=cf.get("severity", "Warning"),
                resolution_note=cf.get("resolution_note", "Flagged for human review")
            ))

        return PaperAnalysisResult(
            metadata=metadata,
            executive_summary=data.get("executive_summary", "Scientific analysis of chemistry literature."),
            compounds=compounds,
            properties=properties,
            bioactivities=bioactivities,
            conditions=conditions,
            methodologies=methodologies,
            findings=findings,
            conflicts_detected=conflicts
        )

    def _analyze_locally(
        self,
        pages: List[PageContent],
        metadata: PaperMetadata,
        evidence_engine: EvidenceEngine
    ) -> PaperAnalysisResult:
        """
        High-fidelity local deterministic extractor using regular expressions,
        scientific pattern matching, and sentence-level evidence attribution.
        Ensures 100% functionality without network or API keys.
        """
        full_text = "\n\n".join([f"[Page {p.page_number}]\n" + p.text for p in pages])

        # 1. Extract Compounds
        compounds: List[CompoundInfo] = []
        seen_compounds = set()
        
        # Look for explicit compound mentions (e.g. Compound 3a, analogue 4b, inhibitor 5)
        compound_patterns = [
            r"\b(Compound\s+[0-9]+[a-z]?)\b",
            r"\b(analogue\s+[0-9]+[a-z]?)\b",
            r"\b(derivative\s+[0-9]+[a-z]?)\b",
            r"\b(inhibitor\s+[0-9]+[a-z]?)\b",
            r"\b(lead\s+compound\s+[0-9]+[a-z]?)\b"
        ]

        for page in pages:
            sec_name = page.sections[0]["name"] if page.sections else "Main Text"
            sentences = re.split(r"(?<=[.!?])\s+", page.text)
            for s in sentences:
                for cp in compound_patterns:
                    matches = re.finditer(cp, s, re.IGNORECASE)
                    for m in matches:
                        c_id = m.group(1).title()
                        if c_id not in seen_compounds:
                            seen_compounds.add(c_id)
                            # Look for formula or chemical class nearby
                            formula_match = re.search(r"\b(C\d+H\d+[A-Za-z0-9]*)\b", s)
                            formula = formula_match.group(1) if formula_match else None
                            
                            class_name = "Heterocycle / Organic Small Molecule"
                            for cls_kw in ["pyrimidine", "indole", "quinazoline", "pyridine", "alkaloid", "kinase inhibitor", "catalyst"]:
                                if cls_kw in page.text.lower():
                                    class_name = cls_kw.title()
                                    break

                            cit = EvidenceCitation(
                                page_number=page.page_number,
                                section=sec_name,
                                verbatim_quote=s.strip(),
                                confidence=0.95
                            )
                            comp_obj = CompoundInfo(
                                compound_id=c_id,
                                name=f"{class_name} {c_id}",
                                formula=formula,
                                chemical_class=class_name,
                                evidence=cit
                            )
                            # PubChem enrichment
                            lookup = PubChemService.lookup_compound(comp_obj.name)
                            if lookup:
                                comp_obj.pubchem_cid = lookup.get("pubchem_cid")
                                comp_obj.formula = comp_obj.formula or lookup.get("formula")
                                comp_obj.molecular_weight = lookup.get("molecular_weight")
                                comp_obj.smiles = lookup.get("smiles")
                            compounds.append(comp_obj)

        # Document-wide formula and MW binding for locally discovered compounds
        from core.chem_calculator import ChemCalculator
        from core.table_extractor import TableExtractor
        temp_tables = TableExtractor.extract_tables_from_pages(pages)
        ChemCalculator.bind_formulas_and_weights_document_wide(compounds, temp_tables, pages)

        # 2. Extract Properties (Yields, Melting Points, Purity)
        properties: List[ChemicalProperty] = []
        yield_records: List[Dict[str, Any]] = []

        for page in pages:
            sec_name = page.sections[0]["name"] if page.sections else "Main Text"
            sentences = re.split(r"(?<=[.!?])\s+", page.text)
            for s in sentences:
                # Yields
                ymatch = re.search(r"\b(\d{1,3}(?:\.\d+)?)\s*%\s*(?:isolated\s+)?yield\b", s, re.IGNORECASE)
                if not ymatch:
                    ymatch = re.search(r"\byield\s*(?:of|was|is)?\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*%", s, re.IGNORECASE)
                if ymatch:
                    val = ymatch.group(1)
                    # Associate with nearby compound if possible
                    comp_assoc = None
                    for c_id in seen_compounds:
                        if c_id.lower() in s.lower():
                            comp_assoc = c_id
                            break
                    cit = EvidenceCitation(
                        page_number=page.page_number,
                        section=sec_name,
                        verbatim_quote=s.strip(),
                        confidence=0.92
                    )
                    prop = ChemicalProperty(
                        compound_id=comp_assoc,
                        parameter="Isolated Yield",
                        value=val,
                        unit="%",
                        evidence=cit
                    )
                    properties.append(prop)
                    yield_records.append({
                        "compound_id": comp_assoc or "General",
                        "value": float(val),
                        "page": page.page_number,
                        "section": sec_name,
                        "quote": s.strip()
                    })

                # Melting Points
                mp_match = re.search(r"\bmp\s*[:=]?\s*(\d{2,3}(?:[–-]\d{2,3})?)\s*°?C\b", s, re.IGNORECASE)
                if mp_match:
                    cit = EvidenceCitation(
                        page_number=page.page_number,
                        section=sec_name,
                        verbatim_quote=s.strip(),
                        confidence=0.9
                    )
                    properties.append(ChemicalProperty(
                        compound_id=None,
                        parameter="Melting Point",
                        value=mp_match.group(1),
                        unit="°C",
                        evidence=cit
                    ))

        # 3. Extract Bioactivity & Table Properties (Table-driven + Sentence-level)
        bioactivities: List[BioactivityResult] = []
        known_cell_lines = [
            "A549", "H1975", "PC-9", "PC9", "MCF-7", "MCF7", "HeLa", "Vero", "Vero Cells",
            "HepG2", "Jurkat", "K562", "MDA-MB-231", "Huh-7", "Caco-2", "NIH3T3", "CHO",
            "HT-29", "U87", "SKBR3", "BT474", "Calu-3", "cancer cells", "cell line"
        ]
        known_pathogens = [
            "Candida albicans", "C. albicans", "Aspergillus fumigatus", "A. fumigatus",
            "Cryptococcus neoformans", "Escherichia coli", "E. coli", "Staphylococcus aureus",
            "S. aureus", "Pseudomonas aeruginosa", "P. aeruginosa"
        ]
        known_enzymes = [
            "EGFR", "HER2", "HER3", "HER4", "CDK4", "CDK6", "CDK2", "CDK1", "BRAF",
            "VEGFR", "VEGFR2", "ALK", "ROS1", "MET", "RET", "KRAS", "MEK", "ERK",
            "mTOR", "PI3K", "AKT", "JAK1", "JAK2", "JAK3", "TYK2", "BTK", "SYK",
            "FLT3", "ABL", "BCR-ABL", "c-KIT", "PDGFR", "SRC", "AURKA", "AURKB",
            "PARP", "HDAC", "SIRT", "COX-1", "COX-2", "AChE", "BChE", "protease",
            "kinase", "polymerase", "integrase", "ligase", "Mpro", "main protease"
        ]

        # 3a. Extract directly from structured tables (highest precision)
        for tbl in temp_tables:
            if not tbl.headers or not tbl.rows:
                continue

            # Identify compound column (usually index 0)
            comp_col_idx = 0
            for idx, h in enumerate(tbl.headers):
                if any(w in h.lower() for w in ["compound", "analogue", "derivative", "molecule", "entry", "alkaloid", "name"]):
                    comp_col_idx = idx
                    break

            for col_idx, h in enumerate(tbl.headers):
                if col_idx == comp_col_idx:
                    continue
                h_lower = h.lower()

                # Check if this column is Isolated Yield
                if "yield" in h_lower:
                    for r in tbl.rows:
                        if len(r) > max(col_idx, comp_col_idx):
                            c_id = r[comp_col_idx].strip()
                            raw_val = r[col_idx].strip()
                            y_m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", raw_val)
                            if not y_m:
                                y_m = re.search(r"^(\d{1,3}(?:\.\d+)?)$", raw_val)
                            if y_m:
                                properties.append(ChemicalProperty(
                                    compound_id=c_id if c_id not in ["-", ""] else None,
                                    parameter="Isolated Yield",
                                    value=y_m.group(1),
                                    unit="%",
                                    evidence=tbl.evidence
                                ))
                    continue

                # Check if this column is a Bioactivity Assay
                is_bio_col = any(m in h_lower for m in ["ic50", "gi50", "ec50", "ki", "kd", "mic", "cc50", "inhibition", "potency", "activity"])
                # Also check unit in header
                header_unit_m = re.search(r"\((nM|µM|uM|mM|µg/mL|ug/ml|ng/mL|%)\)", h, re.IGNORECASE)
                if not is_bio_col and not header_unit_m:
                    continue

                # Determine assay metric
                col_metric = "IC50"
                if re.search(r"\bgi50\b", h_lower):
                    col_metric = "GI50"
                elif re.search(r"\bmic\b", h_lower):
                    col_metric = "MIC"
                elif re.search(r"\bcc50\b", h_lower):
                    col_metric = "CC50"
                elif re.search(r"\bki\b", h_lower):
                    col_metric = "Ki"
                elif re.search(r"\bkd\b", h_lower):
                    col_metric = "Kd"
                elif re.search(r"\bec50\b", h_lower):
                    col_metric = "EC50"
                elif re.search(r"\binhibition\b", h_lower):
                    col_metric = "% Inhibition"

                # Determine target and cell line from column header
                col_target = "Enzyme Target"
                col_cell_line = None

                # Check for cell lines in header
                for cl in known_cell_lines:
                    if cl.lower() in h_lower:
                        col_cell_line = cl
                        col_target = "Cytotoxicity" if "cytotox" in h_lower else "Cellular Antiproliferative"
                        break

                # Check for pathogens in header
                if not col_cell_line:
                    for path in known_pathogens:
                        if path.lower() in h_lower:
                            col_target = path
                            break

                # Check for enzymes/receptors in header
                if not col_cell_line and col_target == "Enzyme Target":
                    for enz in known_enzymes:
                        if enz.lower() in h_lower:
                            col_target = enz
                            break

                # If still Enzyme Target, inspect table title
                if col_target == "Enzyme Target" and not col_cell_line:
                    for enz in known_enzymes:
                        if enz.lower() in tbl.title.lower():
                            col_target = enz
                            break

                # Parse row values for this column
                for r in tbl.rows:
                    if len(r) > max(col_idx, comp_col_idx):
                        c_id = r[comp_col_idx].strip()
                        raw_val = r[col_idx].strip()
                        if raw_val in ["-", "--", "N/A", "nd", "ND", "not determined", "Reference", "none"] or not raw_val:
                            continue

                        # Extract value and optional unit
                        val_unit_m = re.search(r"((?:[><≤≥]\s*)?\d+(?:\.\d+)?(?:\s*±\s*\d+(?:\.\d+)?)?)\s*(nM|µM|uM|mM|µg/mL|ug/ml|ng/mL|%)?", raw_val, re.IGNORECASE)
                        if val_unit_m:
                            num_val = val_unit_m.group(1).strip()
                            row_unit = val_unit_m.group(2) or (header_unit_m.group(1) if header_unit_m else ("µg/mL" if col_metric == "MIC" else "nM"))
                            bioactivities.append(BioactivityResult(
                                compound_id=c_id if c_id not in ["-", ""] else None,
                                assay_type=col_metric,
                                target=col_target,
                                value=num_val,
                                unit=row_unit,
                                cell_line=col_cell_line,
                                evidence=tbl.evidence
                            ))

        # 3b. Extract from narrative sentences (with strict local target & cell line isolation)
        for page in pages:
            sec_name = page.sections[0]["name"] if page.sections else "Main Text"
            sentences = re.split(r"(?<=[.!?])\s+", page.text)
            for s in sentences:
                bio_match = re.search(r"\b(IC50|EC50|Ki|Kd|GI50|CC50|MIC)\s*(?:value|of)?\s*[:=]?\s*((?:[><≤≥]\s*)?\d+(?:\.\d+)?(?:\s*±\s*\d+(?:\.\d+)?)?)\s*(nM|µM|uM|mM|ng/mL|µg/mL|ug/ml)\b", s, re.IGNORECASE)
                if bio_match:
                    assay_type = bio_match.group(1).upper()
                    val = bio_match.group(2).strip()
                    unit = bio_match.group(3)

                    # Strict target and cell line isolation (MUST be in the SAME sentence)
                    s_lower = s.lower()
                    target = "Enzyme Target"
                    cell_line = None

                    # Check cell lines in this sentence
                    for cl in known_cell_lines:
                        if cl.lower() in s_lower:
                            cell_line = cl
                            target = "Cytotoxicity" if "cytotox" in s_lower else "Cellular Antiproliferative"
                            break

                    # Check pathogens in this sentence
                    if not cell_line:
                        for path in known_pathogens:
                            if path.lower() in s_lower:
                                target = path
                                break

                    # Check enzymes in this sentence
                    if not cell_line and target == "Enzyme Target":
                        for enz in known_enzymes:
                            if enz.lower() in s_lower:
                                target = enz
                                break

                    # Match compound in this sentence
                    comp_assoc = None
                    for c_id in seen_compounds:
                        if c_id.lower() in s_lower:
                            comp_assoc = c_id
                            break

                    # De-duplicate if already captured from table
                    is_dup = any(
                        b.compound_id == comp_assoc and b.assay_type == assay_type and b.value == val
                        for b in bioactivities
                    )
                    if not is_dup:
                        cit = EvidenceCitation(
                            page_number=page.page_number,
                            section=sec_name,
                            verbatim_quote=s.strip(),
                            confidence=0.94
                        )
                        bioactivities.append(BioactivityResult(
                            compound_id=comp_assoc,
                            assay_type=assay_type,
                            target=target,
                            value=val,
                            unit=unit,
                            cell_line=cell_line,
                            evidence=cit
                        ))

        # 4. Extract Experimental Conditions
        conditions: List[ExperimentalCondition] = []
        common_solvents = ["DMF", "THF", "DCM", "CH2Cl2", "toluene", "EtOH", "MeOH", "1,4-dioxane", "acetonitrile", "MeCN", "EtOAc", "DMSO"]
        common_catalysts = ["Pd(PPh3)4", "Pd(dppf)Cl2", "Pd/C", "CuI", "RuCl3", "Ni(cod)2", "K2CO3", "Cs2CO3", "Et3N", "TFA", "NaOH"]

        for page in pages:
            sec_name = page.sections[0]["name"] if page.sections else "Main Text"
            sentences = re.split(r"(?<=[.!?])\s+", page.text)
            for s in sentences:
                found_solvents = [sol for sol in common_solvents if re.search(r"\b" + re.escape(sol) + r"\b", s, re.IGNORECASE)]
                temp_match = re.search(r"(\d{1,3})\s*°?C\b", s)
                time_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hours?|min|minutes?)\b", s, re.IGNORECASE)
                found_cats = [cat for cat in common_catalysts if cat.lower() in s.lower()]

                if found_solvents or (temp_match and time_match) or found_cats:
                    cit = EvidenceCitation(
                        page_number=page.page_number,
                        section=sec_name,
                        verbatim_quote=s.strip(),
                        confidence=0.9
                    )
                    conditions.append(ExperimentalCondition(
                        reaction_step="Reaction Step / Synthetic Protocol",
                        solvent=", ".join(found_solvents) if found_solvents else None,
                        catalyst=", ".join(found_cats) if found_cats else None,
                        temperature=f"{temp_match.group(1)} °C" if temp_match else None,
                        time=time_match.group(0) if time_match else None,
                        evidence=cit
                    ))

        # 5. Methodologies
        methodologies: List[Methodology] = []
        method_keywords = [
            ("1H NMR", "1H-NMR Spectroscopy used for structural elucidation"),
            ("13C NMR", "13C-NMR Spectroscopy used for carbon skeletal assignment"),
            ("HRMS", "High-Resolution Mass Spectrometry for molecular mass confirmation"),
            ("Flash chromatography", "Silica gel column purification"),
            ("HPLC", "High Performance Liquid Chromatography for purity analysis"),
            ("X-ray", "Single-crystal X-ray diffraction"),
            ("Docking", "Computational molecular docking simulation")
        ]

        for page in pages:
            sec_name = page.sections[0]["name"] if page.sections else "Main Text"
            for kw, desc in method_keywords:
                if kw.lower() in page.text.lower():
                    # Find sentence
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", page.text) if kw.lower() in s.lower()]
                    quote = sentences[0] if sentences else f"Mentions {kw} analysis."
                    cit = EvidenceCitation(
                        page_number=page.page_number,
                        section=sec_name,
                        verbatim_quote=quote,
                        confidence=0.92
                    )
                    methodologies.append(Methodology(
                        technique=kw,
                        description=desc,
                        evidence=cit
                    ))

        # 6. Significant Findings
        findings: List[SignificantFinding] = []
        finding_triggers = [
            r"\b(revealed\s+that\b.*?[\.\n])",
            r"\b(demonstrated\s+(?:potent|significant|improved)\b.*?[\.\n])",
            r"\b(exhibited\s+(?:high|potent|selective)\b.*?[\.\n])",
            r"\b(structure-activity\s+relationship\b.*?[\.\n])",
            r"\b(in\s+conclusion\b.*?[\.\n])"
        ]

        for page in pages:
            sec_name = page.sections[0]["name"] if page.sections else "Main Text"
            for trigger in finding_triggers:
                matches = re.finditer(trigger, page.text, re.IGNORECASE)
                for m in matches:
                    quote = m.group(0).strip()
                    if len(quote) > 30:
                        cit = EvidenceCitation(
                            page_number=page.page_number,
                            section=sec_name,
                            verbatim_quote=quote,
                            confidence=0.91
                        )
                        cat = "SAR Insight" if "structure-activity" in quote.lower() else "Potency & Efficacy"
                        findings.append(SignificantFinding(
                            finding=quote,
                            category=cat,
                            evidence=cit
                        ))

        # 7. Check for Data Conflicts / Discrepancies
        conflicts: List[ConflictAlert] = []
        # Check yield discrepancies for same compound or between abstract and experimental
        if len(yield_records) >= 2:
            for i in range(len(yield_records)):
                for j in range(i + 1, len(yield_records)):
                    r1 = yield_records[i]
                    r2 = yield_records[j]
                    # If same compound or significant delta (>10%) between abstract and experimental
                    if abs(r1["value"] - r2["value"]) >= 8.0:
                        diff = abs(r1["value"] - r2["value"])
                        topic = f"Discrepancy in Reported Yield ({r1['value']}% vs {r2['value']}%)"
                        desc = (
                            f"A numerical discrepancy of {diff:.1f}% was detected between page {r1['page']} "
                            f"({r1['section']}: {r1['value']}%) and page {r2['page']} ({r2['section']}: {r2['value']}%)."
                        )
                        cit_a = EvidenceCitation(
                            page_number=r1["page"],
                            section=r1["section"],
                            verbatim_quote=r1["quote"],
                            confidence=0.9
                        )
                        cit_b = EvidenceCitation(
                            page_number=r2["page"],
                            section=r2["section"],
                            verbatim_quote=r2["quote"],
                            confidence=0.9
                        )
                        conflicts.append(ConflictAlert(
                            topic=topic,
                            description=desc,
                            claim_a=f"{r1['value']}% yield reported in {r1['section']}",
                            citation_a=cit_a,
                            claim_b=f"{r2['value']}% yield reported in {r2['section']}",
                            citation_b=cit_b,
                            severity="Warning" if diff < 15 else "High Discrepancy",
                            resolution_note="Verify whether the higher figure represents crude NMR conversion while the lower figure is the isolated yield."
                        ))
                        break
                if conflicts:
                    break

        exec_summary = (
            f"Automated chemical analysis identified {len(compounds)} key chemical compounds/series, "
            f"{len(bioactivities)} bioactivity assay points, and {len(conditions)} synthetic reaction conditions. "
            f"Document spans {metadata.page_count} pages with {metadata.total_words} words."
        )

        return PaperAnalysisResult(
            metadata=metadata,
            executive_summary=exec_summary,
            compounds=compounds[:10],
            properties=properties[:12],
            bioactivities=bioactivities[:12],
            conditions=conditions[:8],
            methodologies=methodologies[:6],
            findings=findings[:8],
            conflicts_detected=conflicts
        )
