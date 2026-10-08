"""
Reaction Pathway & Multi-Step Synthetic Route Extraction Engine.
Reconstructs synthetic sequences (Reactant -> Intermediate -> Product),
step order, catalysts, reagents, solvents, temperatures, and overall yields.
"""

import re
from typing import List, Optional, Dict, Any
from core.pdf_parser import PageContent
from core.evidence_engine import EvidenceEngine
from core.models import ReactionPathway, ReactionStep, EvidenceCitation, ExperimentalCondition


class ReactionPathwayExtractor:
    """Extracts sequential synthetic routes and Schemes from scientific chemistry literature."""

    @staticmethod
    def extract_pathways(
        pages: List[PageContent],
        conditions: List[ExperimentalCondition],
        evidence_engine: EvidenceEngine
    ) -> List[ReactionPathway]:
        pathways: List[ReactionPathway] = []
        full_text = "\n".join([f"[Page {p.page_number}] {p.text}" for p in pages])

        # Look for Scheme mentions (e.g., Scheme 1, Scheme 2, Synthetic Route)
        scheme_matches = list(re.finditer(r"(Scheme\s+(\d+|[A-Z]))[:\.\-\s]+([^\.\n]+)", full_text, re.IGNORECASE))
        
        # Determine target compounds
        target_leads = []
        for cond in conditions:
            if cond.reaction_step and "lead" in cond.reaction_step.lower() or "compound" in cond.reaction_step.lower():
                target_leads.append(cond.reaction_step)

        # Detect multi-step sequences from chemistry synthesis text
        steps: List[ReactionStep] = []
        step_num = 1

        for cond in conditions:
            # Derive starting material and product from step name or context
            step_name = cond.reaction_step or "Synthetic Transformation"
            reagents = cond.catalyst or "Reagents not specified"
            prod_match = re.search(r"\b(Compound\s+[0-9]+[a-z]?|Derivative\s+[0-9]+[a-z]?|Intermediate\s+[0-9]+[a-z]?)\b", cond.evidence.verbatim_quote, re.IGNORECASE)
            product_name = prod_match.group(0).title() if prod_match else f"Product of Step {step_num}"

            # Guess starting material
            start_mat = None
            if "suzuki" in cond.evidence.verbatim_quote.lower():
                start_mat = "Aryl halide / Chloro intermediate & Boronic acid"
                step_name = "Suzuki-Miyaura Cross-Coupling"
            elif "snar" in cond.evidence.verbatim_quote.lower() or "nucleophilic" in cond.evidence.verbatim_quote.lower():
                start_mat = "Dichloropyrimidine / Heteroaryl core"
                step_name = "Nucleophilic Aromatic Substitution (SNAr)"
            elif "c-h" in cond.evidence.verbatim_quote.lower() or "arylation" in cond.evidence.verbatim_quote.lower():
                start_mat = "Heterocycle / Arene core & Aryl halide"
                step_name = "Palladium-Catalyzed C-H Activation"
            elif "alkylation" in cond.evidence.verbatim_quote.lower():
                start_mat = "Amine / Phenol intermediate"
                step_name = "N-/O-Alkylation"

            step = ReactionStep(
                step_number=step_num,
                reaction_name=step_name,
                starting_material=start_mat or "Synthetic Precursor",
                reagents=reagents,
                solvent=cond.solvent,
                temperature=cond.temperature,
                time=cond.time,
                product=product_name,
                yield_percent=cond.yield_reported,
                evidence=cond.evidence
            )
            steps.append(step)
            step_num += 1

        if not steps and pages:
            # Extract basic synthetic steps from pages directly
            for page in pages:
                if any(k in page.text.lower() for k in ["synthesis", "scheme", "reaction", "prepared"]):
                    matches = re.finditer(r"([A-Za-z0-9\s\-]+)\s+(?:was|were)\s+prepared\s+(?:via|by|using)\s+([^.]+)\.", page.text, re.IGNORECASE)
                    for m in matches:
                        p_name = m.group(1).strip()
                        proc = m.group(2).strip()
                        cit = EvidenceCitation(
                            page_number=page.page_number,
                            section="Chemistry",
                            verbatim_quote=m.group(0),
                            confidence=0.88
                        )
                        steps.append(ReactionStep(
                            step_number=step_num,
                            reaction_name="Preparation / Coupling",
                            starting_material="Reagent precursor",
                            reagents=proc[:100],
                            product=p_name[:50],
                            evidence=cit
                        ))
                        step_num += 1

        if steps:
            # Calculate overall yield if individual yields exist
            numeric_yields = []
            for s in steps:
                if s.yield_percent:
                    y_match = re.search(r"(\d+(?:\.\d+)?)", s.yield_percent)
                    if y_match:
                        numeric_yields.append(float(y_match.group(1)) / 100.0)

            overall_str = None
            if numeric_yields:
                prod_yield = 1.0
                for y in numeric_yields:
                    prod_yield *= y
                overall_str = f"{round(prod_yield * 100, 1)}%"

            scheme_title = scheme_matches[0].group(0) if scheme_matches else "Scheme 1: Synthetic Route to Target Analogues"
            lead_target = steps[-1].product if steps else "Lead Target Molecule"

            pathways.append(ReactionPathway(
                scheme_id="Scheme 1",
                title=scheme_title,
                target_compound=lead_target,
                steps=steps,
                overall_yield=overall_str or "Variable / Not explicitly totaled",
                evidence=steps[0].evidence
            ))

        return pathways
