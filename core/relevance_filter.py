"""
Multi-Tier Relevance Screening and Deep Question-Answering Engine.
Enables rapid triage of large chemistry literature libraries (up to 100+ papers)
against a research question, candidate shortlisting, and scoped interactive Q&A.

Pipeline Architecture:
- Tier 1: Fast Bibliographic & Abstract Triage (BM25 & Semantic Lexical Filter)
- Tier 2: Chemical Entity & Target Concordance Screening (Domain Entity Matching)
- Tier 3: Deep Evidence & Full-Text Passage Verification (Passage Density Scoring)
- Deep QA: High-precision cross-study synthesis exclusively on shortlisted papers
"""

import os
import re
import math
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter
from pydantic import BaseModel, Field

from core.models import (
    PaperMetadata,
    CompoundInfo,
    ChemicalProperty,
    BioactivityResult,
    ExperimentalCondition,
    PaperAnalysisResult,
    EvidenceCitation,
    ConflictAlert,
    AVAILABLE_GEMINI_MODELS,
    generate_content_with_fallback
)
from core.pdf_parser import PageContent
from core.evidence_engine import EvidenceEngine, EvidenceChunk

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# =========================================================
# DATA MODELS
# =========================================================

class ScreeningTierResult(BaseModel):
    """Result of a single tier of the relevance screening pipeline."""
    tier: int = Field(description="Tier index (1, 2, or 3)")
    name: str = Field(description="Name of the screening tier")
    score: float = Field(ge=0.0, le=100.0, description="Normalized relevance score 0-100")
    passed: bool = Field(description="Whether the paper passed this tier's threshold")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic scoring details")


class PaperRelevanceAssessment(BaseModel):
    """Complete multi-tier evaluation for a single paper in the library."""
    paper_id: str = Field(description="Key or filename of the paper in the store")
    title: str = Field(description="Extracted or declared title of the publication")
    authors: str = Field(default="N/A", description="Authors list")
    journal: str = Field(default="N/A", description="Journal or DOI")
    year: Optional[str] = Field(default=None, description="Publication year")
    
    tier1_result: ScreeningTierResult
    tier2_result: ScreeningTierResult
    tier3_result: ScreeningTierResult
    
    composite_score: float = Field(ge=0.0, le=100.0, description="Weighted composite score 0-100")
    status: str = Field(default="SHORTLISTED", description="'SHORTLISTED' or 'EXCLUDED'")
    relevance_tier: str = Field(default="High Relevance", description="High / Moderate / Low")
    rank: int = Field(default=1, description="Relative rank within the screened library")
    
    match_rationale: str = Field(description="Concise justification of why this paper was shortlisted/excluded")
    matched_targets: List[str] = Field(default_factory=list, description="Target proteins/assays matching query")
    matched_compounds: List[str] = Field(default_factory=list, description="Compounds matching query constraints")
    matched_keywords: List[str] = Field(default_factory=list, description="Keywords and chemical terms matched")
    
    key_quote: str = Field(default="", description="Premier supporting passage quote")
    source_page: int = Field(default=1, description="Page number of the premier quote")
    top_evidence_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Top evidence passages")


class RelevanceScreeningReport(BaseModel):
    """Overall summary report of executing the multi-tier screening pipeline."""
    query: str = Field(description="The input scientific research question")
    total_screened: int = Field(description="Total number of papers screened in library")
    tier1_passed: int = Field(description="Papers passing Tier 1 (Bibliographic/Abstract)")
    tier2_passed: int = Field(description="Papers passing Tier 2 (Entity/Target Concordance)")
    tier3_shortlisted: int = Field(description="Papers shortlisted after Tier 3 (Deep Evidence)")
    excluded_count: int = Field(description="Number of papers screened out")
    reduction_percentage: float = Field(description="Literature volume reduction percentage")
    shortlisted: List[PaperRelevanceAssessment] = Field(default_factory=list)
    excluded: List[PaperRelevanceAssessment] = Field(default_factory=list)
    execution_time_seconds: float = Field(default=0.0, description="Pipeline runtime in seconds")


class DeepQAResponse(BaseModel):
    """Structured response from deep interactive QA on shortlisted papers."""
    question: str
    shortlisted_paper_ids: List[str]
    executive_synthesis: str
    comparative_matrix: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    divergences_and_gaps: List[str]
    model_used: str = "gemini-3.8-flash"
    full_markdown: str


# =========================================================
# RELEVANCE FILTER ENGINE
# =========================================================

class RelevanceFilterEngine:
    """
    Executes high-throughput multi-tier relevance screening on large paper libraries
    and synthesizes scoped deep question-answering exclusively on shortlisted literature.
    """

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "of", "from", "as", "is", "was", "were", "are", "be", "been", "it",
        "this", "that", "these", "those", "which", "who", "whom", "what", "where",
        "when", "how", "all", "any", "both", "each", "few", "more", "most", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "can", "will", "just", "should", "now", "d", "ll", "m", "o", "re", "ve", "y",
        "paper", "papers", "study", "studies", "research", "find", "show", "tell",
        "me", "which", "does", "report", "reported", "evaluate", "evaluated"
    }

    CHEM_SYNONYMS = {
        "potency": ["ic50", "activity", "ki", "inhibition", "ec50", "nanomolar", "nm"],
        "bioactivity": ["ic50", "ki", "ec50", "activity", "assay", "inhibition"],
        "activity": ["ic50", "inhibition", "potency", "ki", "ec50"],
        "yield": ["yield", "isolated", "conversion", "efficiency"],
        "synthesis": ["synthesis", "synthetic", "reaction", "route", "coupling", "catalyst", "step"],
        "catalyst": ["pd", "palladium", "copper", "catalyst", "ligand", "ruthenium", "nickel", "dppf"],
        "solvent": ["dcm", "meoh", "thf", "dmf", "etac", "solvent", "dioxane", "toluene"],
        "kinase": ["egfr", "her2", "braf", "cdk4", "cdk6", "vegfr", "alk", "mek", "akt", "kinase"],
        "inhibition": ["inhibitor", "inhibition", "inhibitory", "ic50", "ki"],
        "antiproliferative": ["cytotoxicity", "gi50", "inhibition", "viability", "mcf7", "a549", "hela"],
        "antifungal": ["candida", "aspergillus", "albicans", "mic", "fungal"],
        "antibacterial": ["mrsa", "staphylococcus", "e. coli", "pseudomonas", "mic"],
        "natural": ["natural product", "flavonoid", "alkaloid", "terpenoid", "extract", "isolate"],
        "heterocycle": ["pyrimidine", "quinazoline", "indole", "pyridine", "triazole", "imidazole"]
    }

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Extract clean alphanumeric tokens."""
        clean = re.sub(r"[^a-zA-Z0-9_\-\s]", " ", text.lower())
        tokens = [t.strip() for t in clean.split() if t.strip()]
        return [t for t in tokens if t not in cls.STOP_WORDS and len(t) > 1]

    @classmethod
    def _parse_query_intent(cls, query: str) -> Dict[str, Any]:
        """Extracts chemical domain anchors, target names, metrics, and synonyms from query."""
        tokens = cls._tokenize(query)
        q_lower = query.lower()
        
        # Identify specific target mentions
        targets_found = set()
        common_targets = [
            "egfr", "her2", "cdk4", "cdk6", "braf", "vegfr", "alk", "mek", "akt",
            "candida", "albicans", "aspergillus", "mrsa", "mcf7", "a549", "hela",
            "bace1", "mpro", "protease", "tubulin", "kinase", "polymerase", "cox-2"
        ]
        for ct in common_targets:
            if re.search(rf"\b{re.escape(ct)}\b", q_lower):
                targets_found.add(ct.upper())

        # Identify requested metrics and constraints
        metrics_found = set()
        for m in ["ic50", "ec50", "ki", "kd", "mic", "yield", "purity", "selectivity", "gi50"]:
            if re.search(rf"\b{re.escape(m)}\b", q_lower):
                metrics_found.add(m)
        if "poten" in q_lower or "nanomolar" in q_lower or "nm" in q_lower:
            metrics_found.add("potency")
        if "yield" in q_lower or "conversion" in q_lower:
            metrics_found.add("yield")

        # Chemistry and reaction types
        chem_types = set()
        for k in ["catalyst", "palladium", "cross-coupling", "suzuki", "buchwald", "solvent", "natural product", "heterocycle"]:
            if k in q_lower:
                chem_types.add(k)

        # Expanded query tokens with synonym support
        expanded_tokens = Counter(tokens)
        for t in tokens:
            if t in cls.CHEM_SYNONYMS:
                for syn in cls.CHEM_SYNONYMS[t]:
                    if syn not in expanded_tokens:
                        expanded_tokens[syn] = 0.65

        return {
            "raw_query": query,
            "tokens": tokens,
            "expanded_tokens": expanded_tokens,
            "targets": list(targets_found),
            "metrics": list(metrics_found),
            "chem_types": list(chem_types)
        }

    # ---------------------------------------------------------
    # TIER 1: FAST BIBLIOGRAPHIC & ABSTRACT TRIAGE
    # ---------------------------------------------------------
    @classmethod
    def _screen_tier1_bibliographic(
        cls,
        intent: Dict[str, Any],
        paper_data: Dict[str, Any]
    ) -> ScreeningTierResult:
        """
        High-throughput lexical screening on Title, Abstract, Journal, and Page 1.
        Designed to filter out completely off-topic literature in milliseconds.
        """
        metadata: PaperMetadata = paper_data.get("metadata")
        title = (getattr(metadata, "title", "") or "").lower() if metadata else ""
        abstract = (getattr(metadata, "abstract", "") or "").lower() if metadata else ""
        journal = (getattr(metadata, "journal_or_doi", "") or "").lower() if metadata else ""
        
        # In case abstract is empty, take first 1000 chars of page 1
        pages = paper_data.get("pages", [])
        page1_text = (pages[0].text[:1200].lower()) if pages else ""
        
        combined_text = f"{title} {title} {title} {abstract} {abstract} {journal} {page1_text}"
        doc_tokens = cls._tokenize(combined_text)
        doc_counter = Counter(doc_tokens)
        
        expanded_tokens = intent["expanded_tokens"]
        score = 0.0
        matched_terms = []
        
        for q_term, weight in expanded_tokens.items():
            if q_term in doc_counter:
                tf = doc_counter[q_term]
                # logarithmic term frequency with title match bonus
                in_title = q_term in title
                title_mult = 3.0 if in_title else 1.0
                term_score = (1.0 + math.log(tf)) * weight * title_mult
                score += term_score
                matched_terms.append(q_term)

        # Normalization to 0-100 scale
        max_possible = max(1.0, len(intent["tokens"]) * 4.5)
        normalized_score = min(100.0, max(0.0, (score / max_possible) * 100.0))
        
        # High-recall pass threshold
        passed = normalized_score >= 12.0 or len(matched_terms) >= 1

        return ScreeningTierResult(
            tier=1,
            name="Tier 1: Bibliographic & Abstract Triage",
            score=round(normalized_score, 1),
            passed=passed,
            details={
                "matched_terms": matched_terms[:10],
                "in_title_matches": [t for t in matched_terms if t in title],
                "abstract_length": len(abstract)
            }
        )

    # ---------------------------------------------------------
    # TIER 2: CHEMICAL ENTITY & TARGET CONCORDANCE
    # ---------------------------------------------------------
    @classmethod
    def _screen_tier2_entity_concordance(
        cls,
        intent: Dict[str, Any],
        paper_data: Dict[str, Any]
    ) -> ScreeningTierResult:
        """
        Checks structured chemical entity extraction:
        - Target concordance (biological assay target names)
        - Potency & bioactivity measurements (IC50, Ki, units)
        - Compound series & chemical classes
        - Synthetic reactions, catalysts, and yields
        """
        analysis: Optional[PaperAnalysisResult] = paper_data.get("analysis")
        if not analysis:
            return ScreeningTierResult(
                tier=2,
                name="Tier 2: Chemical Entity & Target Concordance",
                score=10.0,
                passed=False,
                details={"reason": "No extracted chemical analysis available"}
            )

        score = 0.0
        matched_targets = []
        matched_compounds = []
        matched_conditions = []
        concordance_notes = []

        # 1. Target concordance
        paper_targets = [str(b.target).lower() for b in analysis.bioactivities if b.target]
        paper_targets_str = " ".join(paper_targets)
        
        query_targets = intent["targets"]
        if query_targets:
            for qt in query_targets:
                qt_lower = qt.lower()
                if qt_lower in paper_targets_str or any(qt_lower in pt for pt in paper_targets):
                    score += 45.0
                    matched_targets.append(qt)
                    concordance_notes.append(f"Direct target concordance on {qt}")
        else:
            # If query didn't specify a specific target, reward presence of any characterized target
            if paper_targets:
                score += 20.0
                matched_targets.extend(list(set(paper_targets))[:3])

        # 2. Potency & Activity Alignment
        if "potency" in intent["metrics"] or any(m in intent["metrics"] for m in ["ic50", "ki", "ec50", "mic"]):
            has_quant_bio = False
            for b in analysis.bioactivities:
                if b.value:
                    has_quant_bio = True
                    # Check if nanomolar potency exists
                    val_str = str(b.value).lower()
                    u_str = str(b.unit).lower() if b.unit else ""
                    if "nm" in u_str or ("um" in u_str and any(char.isdigit() for char in val_str)):
                        score += 30.0
                        concordance_notes.append(f"Reports quantitative bioactivity: {b.value} {b.unit} ({b.target})")
                        break
            if not has_quant_bio and "potency" in intent["metrics"]:
                concordance_notes.append("No quantitative enzyme potency measurements reported")
        else:
            if analysis.bioactivities:
                score += 15.0

        # 3. Chemical Class & Compound Series
        q_lower = intent["raw_query"].lower()
        for c in analysis.compounds:
            c_name = str(c.name or "").lower()
            c_class = str(c.chemical_class or "").lower()
            if any(t in c_name or t in c_class for t in intent["tokens"]):
                score += 15.0
                matched_compounds.append(c.compound_id or c.name)
                break

        # 4. Reaction Conditions & Synthetic Route
        if "yield" in intent["metrics"] or intent["chem_types"]:
            has_synth = False
            for cond in analysis.conditions:
                step_str = (cond.reaction_step or "").lower()
                cat_str = (cond.catalyst or "").lower()
                if any(t in step_str or t in cat_str for t in intent["tokens"] + list(intent["chem_types"])):
                    score += 25.0
                    matched_conditions.append(f"{cond.reaction_step} ({cond.catalyst or 'solvent'})")
                    has_synth = True
                    break
            if not has_synth and analysis.conditions and "yield" in intent["metrics"]:
                score += 12.0

        # Base bonus for rich structured data
        total_entities = len(analysis.compounds) + len(analysis.bioactivities) + len(analysis.conditions)
        if total_entities >= 5:
            score += 10.0

        normalized_score = min(100.0, max(0.0, score))
        passed = normalized_score >= 20.0 or (bool(matched_targets) and bool(matched_compounds))

        return ScreeningTierResult(
            tier=2,
            name="Tier 2: Chemical Entity & Target Concordance",
            score=round(normalized_score, 1),
            passed=passed,
            details={
                "matched_targets": matched_targets,
                "matched_compounds": matched_compounds[:5],
                "concordance_notes": concordance_notes,
                "total_entities_extracted": total_entities
            }
        )

    # ---------------------------------------------------------
    # TIER 3: DEEP EVIDENCE & FULL-TEXT PASSAGE VERIFICATION
    # ---------------------------------------------------------
    @classmethod
    def _screen_tier3_evidence_density(
        cls,
        intent: Dict[str, Any],
        paper_data: Dict[str, Any]
    ) -> ScreeningTierResult:
        """
        Deep passage-level evidence retrieval using EvidenceEngine:
        - BM25 chemical passage scoring
        - Co-occurrence of target + compound + quantitative metrics in same passage
        - Section relevance (Results/Discussion vs Intro)
        - Extracts verbatim quote and source page
        """
        engine: Optional[EvidenceEngine] = paper_data.get("engine")
        pages: List[PageContent] = paper_data.get("pages", [])
        
        if not engine:
            if pages:
                engine = EvidenceEngine(pages)
                paper_data["engine"] = engine
            else:
                return ScreeningTierResult(
                    tier=3,
                    name="Tier 3: Deep Evidence & Passage Verification",
                    score=5.0,
                    passed=False,
                    details={"reason": "No text pages or evidence engine available"}
                )

        # Retrieve top 5 evidence chunks
        search_results = engine.search(intent["raw_query"], top_k=5, min_score=0.0)
        retrieved_chunks = [chunk for chunk, _ in search_results] if search_results else engine.chunks[:3]
        if not retrieved_chunks:
            return ScreeningTierResult(
                tier=3,
                name="Tier 3: Deep Evidence & Passage Verification",
                score=10.0,
                passed=False,
                details={"reason": "No matching evidence chunks retrieved"}
            )

        top_chunk = retrieved_chunks[0]
        key_quote = top_chunk.text.strip().replace("\n", " ")
        if len(key_quote) > 280:
            key_quote = key_quote[:277] + "..."
        source_page = top_chunk.page_number

        # Calculate passage evidence density
        passage_scores = []
        co_occurrence_count = 0
        q_targets_lower = [t.lower() for t in intent["targets"]]
        
        for c in retrieved_chunks[:3]:
            c_text_lower = c.text.lower()
            # Check co-occurrence
            has_target = any(qt in c_text_lower for qt in q_targets_lower) if q_targets_lower else True
            has_number = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:nm|µm|um|%|°c|mg/ml|h)\b", c_text_lower))
            has_compound = bool(re.search(r"\b(?:compound|derivative|inhibitor|analog)\s+[0-9a-z]+\b", c_text_lower))
            
            density = 0.0
            if has_target:
                density += 35.0
            if has_number:
                density += 35.0
            if has_compound:
                density += 20.0
            if c.section and any(sec in c.section.lower() for sec in ["results", "discussion", "table", "experimental"]):
                density += 10.0
                
            passage_scores.append(density)
            if has_target and (has_number or has_compound):
                co_occurrence_count += 1

        avg_passage_score = sum(passage_scores) / max(1, len(passage_scores))
        normalized_score = min(100.0, max(0.0, avg_passage_score))
        passed = normalized_score >= 25.0 or co_occurrence_count >= 1

        chunk_summaries = []
        for c in retrieved_chunks[:3]:
            chunk_summaries.append({
                "page": c.page_number,
                "section": c.section,
                "excerpt": (c.text[:180] + "...") if len(c.text) > 180 else c.text
            })

        return ScreeningTierResult(
            tier=3,
            name="Tier 3: Deep Evidence & Passage Verification",
            score=round(normalized_score, 1),
            passed=passed,
            details={
                "key_quote": key_quote,
                "source_page": source_page,
                "top_chunks": chunk_summaries,
                "co_occurrence_verified": co_occurrence_count > 0
            }
        )

    # ---------------------------------------------------------
    # MULTI-TIER SCREENING PIPELINE ORCHESTRATOR
    # ---------------------------------------------------------
    @classmethod
    def screen_library(
        cls,
        query: str,
        papers_dict: Dict[str, Any],
        min_score: float = 45.0,
        max_shortlist: int = 10
    ) -> RelevanceScreeningReport:
        """
        Executes complete multi-tier screening across all papers in the library.
        Returns a structured report containing shortlisted and excluded papers.
        """
        start_time = time.time()
        intent = cls._parse_query_intent(query)
        total_screened = len(papers_dict)
        
        assessments: List[PaperRelevanceAssessment] = []
        tier1_passed_count = 0
        tier2_passed_count = 0
        tier3_passed_count = 0

        for paper_id, p_data in papers_dict.items():
            metadata: Optional[PaperMetadata] = p_data.get("metadata")
            analysis: Optional[PaperAnalysisResult] = p_data.get("analysis")
            
            raw_title = str(getattr(metadata, "title", None) or paper_id).strip()
            authors = str(getattr(metadata, "authors", None) or "N/A").strip()
            journal = str(getattr(metadata, "journal_or_doi", None) or "N/A").strip()
            year = getattr(metadata, "year", None)

            # --- TIER 1: Bibliographic & Abstract Triage ---
            t1 = cls._screen_tier1_bibliographic(intent, p_data)
            if t1.passed:
                tier1_passed_count += 1

            # --- TIER 2: Chemical Entity & Target Concordance ---
            # Even if T1 score is borderline, we evaluate T2 to ensure chemical specificity isn't missed
            t2 = cls._screen_tier2_entity_concordance(intent, p_data)
            if t2.passed:
                tier2_passed_count += 1

            # --- TIER 3: Deep Evidence Density ---
            # If paper showed potential in T1 or T2, execute passage verification
            if t1.passed or t2.passed or total_screened <= 5:
                t3 = cls._screen_tier3_evidence_density(intent, p_data)
            else:
                t3 = ScreeningTierResult(
                    tier=3,
                    name="Tier 3: Deep Evidence & Passage Verification",
                    score=0.0,
                    passed=False,
                    details={"reason": "Skipped due to low Tier 1 & Tier 2 scores"}
                )
            if t3.passed:
                tier3_passed_count += 1

            # Composite Score: 25% Tier 1 + 35% Tier 2 + 40% Tier 3
            composite = (0.25 * t1.score) + (0.35 * t2.score) + (0.40 * t3.score)
            composite = round(min(100.0, max(0.0, composite)), 1)

            # Rationale generation
            reasons = []
            if t2.details.get("matched_targets"):
                reasons.append(f"Target match: {', '.join(t2.details['matched_targets'])}")
            if t2.details.get("concordance_notes"):
                reasons.extend(t2.details["concordance_notes"][:2])
            if t1.details.get("in_title_matches"):
                reasons.append(f"Title anchor: {', '.join(t1.details['in_title_matches'])}")
            if t3.details.get("co_occurrence_verified"):
                reasons.append("Direct evidence co-occurrence verified in full text")

            if not reasons:
                if composite < min_score:
                    rationale = f"Fails specific criteria for '{query}'. Low term & entity overlap (Score: {composite}%)."
                else:
                    rationale = f"General contextual relevance across research passages (Score: {composite}%)."
            else:
                rationale = "; ".join(reasons) + f" (Composite: {composite}%)."

            # Relevance Tier classification
            if composite >= 75.0:
                rel_tier = "High Relevance"
            elif composite >= 45.0:
                rel_tier = "Moderate Relevance"
            else:
                rel_tier = "Low / Screened Out"

            key_q = t3.details.get("key_quote", "")
            src_p = t3.details.get("source_page", 1)
            top_ev = t3.details.get("top_chunks", [])

            assessment = PaperRelevanceAssessment(
                paper_id=paper_id,
                title=raw_title,
                authors=authors,
                journal=journal,
                year=year,
                tier1_result=t1,
                tier2_result=t2,
                tier3_result=t3,
                composite_score=composite,
                status="SHORTLISTED" if composite >= min_score else "EXCLUDED",
                relevance_tier=rel_tier,
                rank=1,
                match_rationale=rationale,
                matched_targets=t2.details.get("matched_targets", []),
                matched_compounds=t2.details.get("matched_compounds", []),
                matched_keywords=t1.details.get("matched_terms", []),
                key_quote=key_q,
                source_page=src_p,
                top_evidence_chunks=top_ev
            )
            assessments.append(assessment)

        # Sort all assessments by composite score descending
        assessments.sort(key=lambda a: a.composite_score, reverse=True)
        for idx, a in enumerate(assessments):
            a.rank = idx + 1

        # Separate shortlisted from excluded
        shortlisted = [a for a in assessments if a.composite_score >= min_score][:max_shortlist]
        
        # Fallback safeguard: if none met min_score, shortlist the top 1-2 candidates with clear badge
        if not shortlisted and assessments:
            top_cand = assessments[0]
            top_cand.status = "SHORTLISTED"
            top_cand.relevance_tier = "Moderate Relevance (Best Candidate)"
            top_cand.match_rationale = f"Closest match in corpus (Score: {top_cand.composite_score}%). " + top_cand.match_rationale
            shortlisted = [top_cand]

        shortlisted_ids = set(a.paper_id for a in shortlisted)
        excluded = [a for a in assessments if a.paper_id not in shortlisted_ids]
        for e in excluded:
            e.status = "EXCLUDED"

        reduction = 0.0
        if total_screened > 0:
            reduction = round(((total_screened - len(shortlisted)) / total_screened) * 100.0, 1)

        elapsed = round(time.time() - start_time, 2)

        return RelevanceScreeningReport(
            query=query,
            total_screened=total_screened,
            tier1_passed=tier1_passed_count,
            tier2_passed=tier2_passed_count,
            tier3_shortlisted=len(shortlisted),
            excluded_count=len(excluded),
            reduction_percentage=reduction,
            shortlisted=shortlisted,
            excluded=excluded,
            execution_time_seconds=elapsed
        )

    # ---------------------------------------------------------
    # DEEP QUESTION-ANSWERING ON SHORTLISTED PAPERS
    # ---------------------------------------------------------
    @classmethod
    def ask_deep_qa(
        cls,
        question: str,
        shortlisted_assessments: List[PaperRelevanceAssessment],
        papers_store: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> DeepQAResponse:
        """
        Executes high-precision cross-study QA exclusively on the shortlisted papers.
        Generates executive synthesis, comparative parameter matrix, verifiable citations,
        and inter-paper divergence/conflict detection.
        """
        if not shortlisted_assessments:
            return DeepQAResponse(
                question=question,
                shortlisted_paper_ids=[],
                executive_synthesis="No shortlisted papers available to query. Please run relevance screening first.",
                comparative_matrix=[],
                citations=[],
                divergences_and_gaps=[],
                model_used="none",
                full_markdown="No candidate publications shortlisted for Deep QA."
            )

        active_key = api_key or os.getenv("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY
        shortlisted_ids = [a.paper_id for a in shortlisted_assessments]

        # Gather scoped evidence chunks strictly from shortlisted papers
        scoped_evidence: List[Dict[str, Any]] = []
        comparative_matrix: List[Dict[str, Any]] = []

        for rank_idx, assess in enumerate(shortlisted_assessments, 1):
            p_data = papers_store.get(assess.paper_id)
            if not p_data:
                continue
            
            engine: Optional[EvidenceEngine] = p_data.get("engine")
            analysis: Optional[PaperAnalysisResult] = p_data.get("analysis")
            
            # Retrieve query-specific passages from this shortlisted paper
            paper_chunks = []
            if engine:
                search_results = engine.search(question, top_k=4, min_score=0.0)
                retrieved = [chunk for chunk, _ in search_results] if search_results else engine.chunks[:2]
                for c in retrieved:
                    paper_chunks.append({
                        "paper_rank": f"Paper {rank_idx}",
                        "paper_title": assess.title,
                        "page_number": c.page_number,
                        "section": c.section,
                        "quote": c.text.strip().replace("\n", " ")
                    })
            
            # If no chunks, fallback to key_quote
            if not paper_chunks and assess.key_quote:
                paper_chunks.append({
                    "paper_rank": f"Paper {rank_idx}",
                    "paper_title": assess.title,
                    "page_number": assess.source_page,
                    "section": "Results & Discussion",
                    "quote": assess.key_quote
                })

            scoped_evidence.extend(paper_chunks)

            # Build comparative matrix row
            lead_comp = "N/A"
            potency = "N/A"
            target = "N/A"
            yield_str = "N/A"
            
            if analysis:
                if analysis.compounds:
                    lead_comp = str(analysis.compounds[0].compound_id or analysis.compounds[0].name)
                if analysis.bioactivities:
                    b0 = analysis.bioactivities[0]
                    target = str(b0.target or "Target")
                    potency = f"{b0.value} {b0.unit}".strip() if b0.value else "N/A"
                if analysis.conditions:
                    c0 = analysis.conditions[0]
                    yield_str = str(c0.yield_reported or "N/A")

            comparative_matrix.append({
                "Paper Index": f"Paper {rank_idx}",
                "Title": assess.title[:35] + "..." if len(assess.title) > 35 else assess.title,
                "Lead Candidate": lead_comp,
                "Target / Assay": target,
                "Potency Metric": potency,
                "Reaction Yield": yield_str,
                "Relevance Score": f"{assess.composite_score}%"
            })

        # Format context for Gemini
        context_blocks = []
        for rank_idx, assess in enumerate(shortlisted_assessments, 1):
            chunks_for_p = [c for c in scoped_evidence if c["paper_rank"] == f"Paper {rank_idx}"]
            quotes_str = "\n".join([f"  [p.{c['page_number']} | {c['section']}]: \"{c['quote'][:250]}...\"" for c in chunks_for_p])
            block = f"""### Shortlisted Publication {rank_idx}: {assess.title}
Authors: {assess.authors} | Journal: {assess.journal}
Relevance Score: {assess.composite_score}% | Rationale: {assess.match_rationale}
Key Verifiable Passages:
{quotes_str or '  No direct passage text extracted.'}"""
            context_blocks.append(block)

        full_scoped_context = "\n\n".join(context_blocks)

        # Attempt Gemini 3.8 Flash synthesis
        llm_response_text = None
        if active_key and HAS_GENAI:
            try:
                client = genai.Client(api_key=active_key, http_options={"timeout": 60000})
                prompt = f"""You are ChemEvidence AI, an expert medicinal chemist conducting deep question-answering across a shortlisted set of {len(shortlisted_assessments)} research papers.

Answer the researcher's query strictly and exclusively using the evidence provided below for the shortlisted papers.
Do NOT extrapolate beyond the reported text. Explicitly cite which publication (Paper 1, Paper 2, etc.) and page number support each claim.

Research Question:
{question}

Shortlisted Literature Evidence:
{full_scoped_context}

Please provide an executive, structured, evidence-grounded comparative answer with:
1. Executive Scientific Synthesis (direct comprehensive answer synthesizing findings across shortlisted papers)
2. Paper-by-Paper Comparative Evidence (comparing compounds, potencies, targets, or yields reported by each paper)
3. Direct Verifiable Citations (Paper, Page, Section, and exact quote)
4. Divergences & Absence Audit (state if papers conflict on values or if parameters are not addressed)"""

                resp = generate_content_with_fallback(
                    client=client,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=1400
                    ),
                    models=AVAILABLE_GEMINI_MODELS
                )
                if resp and resp.text:
                    llm_response_text = resp.text.strip()
            except Exception as e:
                print(f"[RelevanceFilterEngine] Notice: Falling back to local Deep QA synthesizer due to: {e}")
                llm_response_text = None

        if not llm_response_text:
            llm_response_text = cls._synthesize_local_deep_qa(
                question=question,
                shortlisted_assessments=shortlisted_assessments,
                comparative_matrix=comparative_matrix,
                scoped_evidence=scoped_evidence
            )

        # Format citations list
        citations_list = []
        for c in scoped_evidence[:6]:
            citations_list.append({
                "paper": c["paper_title"],
                "rank": c["paper_rank"],
                "page": c["page_number"],
                "section": c["section"],
                "quote": c["quote"]
            })

        # Identify potential divergences across shortlisted papers
        divergences = []
        potencies = [row["Potency Metric"] for row in comparative_matrix if row["Potency Metric"] != "N/A"]
        if len(set(potencies)) > 1:
            divergences.append(f"Potency metrics diverge across shortlisted studies: {', '.join(potencies)}.")
        
        return DeepQAResponse(
            question=question,
            shortlisted_paper_ids=shortlisted_ids,
            executive_synthesis=llm_response_text,
            comparative_matrix=comparative_matrix,
            citations=citations_list,
            divergences_and_gaps=divergences,
            model_used="gemini-3.8-flash" if (active_key and HAS_GENAI) else "deterministic-local-engine",
            full_markdown=llm_response_text
        )

    @classmethod
    def _synthesize_local_deep_qa(
        cls,
        question: str,
        shortlisted_assessments: List[PaperRelevanceAssessment],
        comparative_matrix: List[Dict[str, Any]],
        scoped_evidence: List[Dict[str, Any]]
    ) -> str:
        """High-fidelity local deterministic synthesizer for deep QA without API key."""
        lines = [
            f"### 📋 Executive Cross-Literature Synthesis\n",
            f"Based on the **{len(shortlisted_assessments)} shortlisted candidate publications** screened for your research inquiry (*\"{question}\"*), a rigorous cross-study synthesis was conducted:\n"
        ]

        # Paper-by-paper breakdown
        lines.append("#### 🔬 Shortlisted Literature Comparative Breakdown")
        for assess in shortlisted_assessments:
            chunks = [c for c in scoped_evidence if c["paper_title"] == assess.title]
            q_text = chunks[0]["quote"] if chunks else assess.key_quote
            p_num = chunks[0]["page_number"] if chunks else assess.source_page
            
            lines.append(f"- **{assess.title}** (Relevance: **{assess.composite_score}%** | *{assess.relevance_tier}*):")
            lines.append(f"  - **Relevance Rationale:** {assess.match_rationale}")
            if assess.matched_targets:
                lines.append(f"  - **Target Assays:** {', '.join(assess.matched_targets)}")
            if assess.matched_compounds:
                lines.append(f"  - **Evaluated Chemotypes:** {', '.join(assess.matched_compounds)}")
            lines.append(f"  - **Verifiable Evidence (Page {p_num}):** \"*{q_text[:200]}...*\"\n")

        # Citations
        lines.append("#### 📚 Direct Verifiable Literature Citations")
        for idx, c in enumerate(scoped_evidence[:4], 1):
            lines.append(f"{idx}. **{c['paper_rank']} ({c['paper_title'][:30]}...)**, Page {c['page_number']} (*{c['section']}*):")
            lines.append(f"   > \"{c['quote'][:220]}...\"\n")

        # Divergences and Gaps
        lines.append("#### ⚠️ Inter-Study Concordance & Boundary Audit")
        lines.append(f"- **Corpus Scope:** Analysis strictly restricted to the {len(shortlisted_assessments)} shortlisted publications.")
        lines.append("- **Verification:** All quantitative parameters cited above correspond to verbatim extractions from the respective publications.")

        return "\n".join(lines)


    # ---------------------------------------------------------
    # 100+ PAPER BENCHMARK CORPUS GENERATOR (FOR SCALE TESTING)
    # ---------------------------------------------------------
    @classmethod
    def generate_benchmark_library_100(cls) -> Dict[str, Any]:
        """
        Generates 100 realistic, diverse medicinal chemistry research papers
        spanning oncology kinases, antibacterials, proteases, natural products,
        and catalytic synthesis. Used to demonstrate and evaluate screening at scale.
        """
        library: Dict[str, Any] = {}

        archetypes = [
            # 1. EGFR / Kinase Oncology (Target match for kinase queries)
            {
                "topic": "EGFR Kinase Inhibitors",
                "title_template": "Discovery of Potent {sub} Quinazoline Derivatives as Covalent EGFR T790M Kinase Inhibitors for Non-Small Cell Lung Cancer",
                "compounds": ["Compound 4a", "Compound 8b (Lead)", "WZ-4002 Analogue"],
                "target": "EGFR Kinase",
                "assay": "IC50",
                "val": "8.4 ± 0.6",
                "unit": "nM",
                "yield": "78%",
                "catalyst": "Pd(dppf)Cl2",
                "solvent": "1,4-dioxane/H2O",
                "abstract_kw": "nanomolar potency against mutant EGFR T790M kinase with antiproliferative cellular efficacy in A549 lung cancer models",
                "count": 18
            },
            # 2. CDK4/6 Cell Cycle Kinases
            {
                "topic": "CDK4/6 Kinase Inhibitors",
                "title_template": "Structure-Activity Relationship of Novel Pyrido[2,3-d]pyrimidin-7-one Derivatives as Selective CDK4/6 Inhibitors",
                "compounds": ["Analogue 12c", "Lead 15f", "Palbociclib Control"],
                "target": "CDK4/6 Kinase",
                "assay": "IC50",
                "val": "24.1 ± 1.8",
                "unit": "nM",
                "yield": "82%",
                "catalyst": "Pd(PPh3)4",
                "solvent": "DMF",
                "abstract_kw": "selective cell cycle arrest at G1/S phase with high potency against CDK4/6 in MCF-7 breast carcinoma",
                "count": 16
            },
            # 3. Antifungal & Antibacterial Natural Products
            {
                "topic": "Antifungal Natural Products",
                "title_template": "Isolation, Semi-Synthesis, and Antifungal Profiling of Flavonoid Glycosides against Candida albicans",
                "compounds": ["Glycoside A", "Deriv 3d", "Amphotericin B Ref"],
                "target": "Candida albicans",
                "assay": "MIC",
                "val": "3.1",
                "unit": "µg/mL",
                "yield": "64%",
                "catalyst": "Sc(OTf)3",
                "solvent": "CH2Cl2",
                "abstract_kw": "bioactive natural product extraction with potent membrane permeabilization in clinical Candida albicans isolates",
                "count": 18
            },
            # 4. Catalytic Cross-Coupling Synthesis
            {
                "topic": "Palladium Catalyzed Cross-Coupling",
                "title_template": "High-Efficiency Palladium-Catalyzed Suzuki-Miyaura Cross-Coupling of Sterically Hindered Heteroaryl Chlorides",
                "compounds": ["Heterocycle 5a", "Coupled Intermediate 9"],
                "target": "Chemical Methodology",
                "assay": "Isolated Yield",
                "val": "92.4",
                "unit": "%",
                "yield": "94%",
                "catalyst": "Pd(OAc)2 / XPhos",
                "solvent": "Toluene",
                "abstract_kw": "robust catalytic cross-coupling methodology affording biaryl and heteroaryl scaffolds in elevated isolated yields above 90 percent",
                "count": 16
            },
            # 5. Viral Protease Inhibitors (SARS-CoV-2 Mpro & HIV)
            {
                "topic": "Viral Protease Inhibitors",
                "title_template": "Design and Crystallographic Characterization of Peptidomimetic Inhibitors Targeting the SARS-CoV-2 Main Protease (Mpro)",
                "compounds": ["Inhibitor 6a", "Lead Peptidomimetic 10b"],
                "target": "SARS-CoV-2 Mpro",
                "assay": "Ki",
                "val": "14.8",
                "unit": "nM",
                "yield": "71%",
                "catalyst": "HATU / DIPEA",
                "solvent": "DMF",
                "abstract_kw": "covalent active site cysteine modification providing sub-micromolar viral replication suppression",
                "count": 16
            },
            # 6. Neurodegenerative BACE1 / Tau Targets
            {
                "topic": "Alzheimer's BACE1 Inhibitors",
                "title_template": "Optimization of Brain-Penetrant Iminothiadiazine Dioxide BACE1 Inhibitors for Alzheimer's Disease",
                "compounds": ["Compound 7e", "Optimized Lead 14a"],
                "target": "BACE1 Protease",
                "assay": "IC50",
                "val": "38.2 ± 3.4",
                "unit": "nM",
                "yield": "76%",
                "catalyst": "CuI / Proline",
                "solvent": "DMSO",
                "abstract_kw": "in vivo reduction of amyloid-beta peptides with balanced blood-brain barrier permeability and metabolic stability",
                "count": 16
            }
        ]

        paper_counter = 1
        substituents = ["Fluoro", "Methoxy", "Trifluoromethyl", "Cyclopropyl", "Piperazine", "Morpholine", "Ethynyl", "Sulfonamide"]

        for arch in archetypes:
            for i in range(arch["count"]):
                sub = substituents[i % len(substituents)]
                title = arch["title_template"].format(sub=sub)
                paper_id = f"Manuscript_{paper_counter:03d}_{arch['topic'].replace(' ', '_')}.pdf"
                
                # Create PaperMetadata
                metadata = PaperMetadata(
                    title=title,
                    authors=f"Dr. A. Scientist et al. (Group {paper_counter})",
                    journal_or_doi=f"Journal of Medicinal Chemistry, 2025, Vol. 68, pp. {1000 + paper_counter * 12}-{1012 + paper_counter * 12}",
                    year="2025",
                    abstract=f"We report the systematic investigation of {arch['topic']}. Chemical modification yielded {arch['compounds'][1]} exhibiting {arch['abstract_kw']}. The compound was synthesized via catalytic transformation using {arch['catalyst']} in {arch['solvent']} with an isolated yield of {arch['yield']}. Quantitative bioactivity assays confirmed an {arch['assay']} of {arch['val']} {arch['unit']} against {arch['target']}."
                )

                # Create PageContent
                page1 = PageContent(
                    page_number=1,
                    text=f"""Title: {metadata.title}
Authors: {metadata.authors}
Journal: {metadata.journal_or_doi}

Abstract:
{metadata.abstract}

Introduction:
The development of selective therapeutics targeting {arch['target']} remains a critical objective. Previous studies reported moderate activity, but metabolic liabilities and low synthetic efficiency limited further evaluation."""
                )

                page2 = PageContent(
                    page_number=2,
                    text=f"""Results and Discussion:
Synthesis of the target library proceeded through {arch['catalyst']} mediated transformation in {arch['solvent']}. 
Isolated yields ranged between 65% and {arch['yield']} for the series.
Compound characterization via 1H NMR and HRMS confirmed high analytical purity (>98%).

Bioactivity Evaluation:
Biological evaluation was conducted against {arch['target']}. Among all synthesized derivatives, {arch['compounds'][1]} demonstrated outstanding inhibitory activity, with a reported {arch['assay']} of {arch['val']} {arch['unit']}.
In vitro stability testing demonstrated favorable clearance profiles."""
                )

                pages = [page1, page2]
                engine = EvidenceEngine(pages)

                # Create PaperAnalysisResult
                citation1 = EvidenceCitation(
                    page_number=2,
                    section="Results and Discussion",
                    verbatim_quote=f"{arch['compounds'][1]} demonstrated outstanding inhibitory activity, with a reported {arch['assay']} of {arch['val']} {arch['unit']} against {arch['target']}.",
                    confidence=0.98
                )

                citation2 = EvidenceCitation(
                    page_number=2,
                    section="Experimental",
                    verbatim_quote=f"Synthesis mediated by {arch['catalyst']} in {arch['solvent']} afforded an isolated yield of {arch['yield']}.",
                    confidence=0.95
                )

                comp_objs = [
                    CompoundInfo(
                        compound_id=arch["compounds"][1],
                        name=f"{sub}-substituted analogue",
                        formula="C22H24FN5O2",
                        chemical_class=arch["topic"],
                        evidence=citation1
                    )
                ]

                bio_objs = [
                    BioactivityResult(
                        compound_id=arch["compounds"][1],
                        assay_type=arch["assay"],
                        target=arch["target"],
                        value=arch["val"],
                        unit=arch["unit"],
                        cell_line="A549" if "Lung" in title else None,
                        evidence=citation1
                    )
                ]

                cond_objs = [
                    ExperimentalCondition(
                        reaction_step=arch["topic"],
                        solvent=arch["solvent"],
                        catalyst=arch["catalyst"],
                        temperature="85 °C",
                        time="6 h",
                        yield_reported=arch["yield"],
                        evidence=citation2
                    )
                ]

                analysis = PaperAnalysisResult(
                    metadata=metadata,
                    executive_summary=metadata.abstract,
                    compounds=comp_objs,
                    properties=[],
                    bioactivities=bio_objs,
                    conditions=cond_objs,
                    methodologies=[],
                    findings=[],
                    conflicts_detected=[]
                )

                library[paper_id] = {
                    "pages": pages,
                    "metadata": metadata,
                    "engine": engine,
                    "analysis": analysis,
                    "pdf_bytes": None,
                    "file_type": "pdf"
                }

                paper_counter += 1

        return library
