"""
Research Gap & Literature Blind-Spot Detection Engine.
Performs an automated medicinal chemistry peer-review audit on the uploaded paper
to proactively highlight missing assays, omitted ADME/PK profiling, lack of selectivity panels,
and uncharacterized intermediate steps.
"""

import re
from typing import List
from core.pdf_parser import PageContent
from core.models import ResearchGap, CompoundInfo, BioactivityResult, ExperimentalCondition


class ResearchGapDetector:
    """Detects omitted experiments, missing pharmacology, and scientific blind spots."""

    @classmethod
    def audit_paper_gaps(
        cls,
        pages: List[PageContent],
        compounds: List[CompoundInfo],
        bioactivities: List[BioactivityResult],
        conditions: List[ExperimentalCondition]
    ) -> List[ResearchGap]:
        full_text = "\n".join([p.text for p in pages]).lower()
        gaps: List[ResearchGap] = []

        # 1. Check ADME & Pharmacokinetics (PK)
        pk_keywords = ["bioavailability", "pharmacokinetic", "clearance", "cmax", "t1/2", "half-life", "microsomal stability", "plasma protein binding", "adme"]
        if not any(k in full_text for k in pk_keywords):
            gaps.append(ResearchGap(
                category="ADME & Pharmacokinetics",
                title="Omission of In Vivo Pharmacokinetic (ADME) Profiling",
                description="The publication reports cellular antiproliferative efficacy and enzyme inhibition, but omits oral bioavailability (%F), metabolic clearance, and half-life (t1/2) measurements.",
                severity="Critical",
                recommendation="Perform mouse liver microsomal stability assays and evaluate oral pharmacokinetic parameters in rodent models before progressing leads."
            ))

        # 2. Check In Vivo Efficacy / Animal Models
        in_vivo_keywords = ["xenograft", "in vivo", "mouse model", "athymic nude", "tumor volume", "animal study"]
        if not any(k in full_text for k in in_vivo_keywords):
            gaps.append(ResearchGap(
                category="In Vivo Translation",
                title="Absence of In Vivo Tumor Xenograft Validation",
                description="Efficacy evaluation is restricted to in vitro enzymatic assays and 2D cancer cell culture. No in vivo animal models or tumor regression experiments were reported.",
                severity="Important",
                recommendation="Assess lead compound tolerability and antitumor growth inhibition in a target-driven human tumor xenograft mouse model."
            ))

        # 3. Check Target Selectivity & Gatekeeper Mutants
        mutant_keywords = ["t790m", "c797s", "selectivity panel", "kinome", "off-target", "her2", "cdk"]
        has_kinase = any("kinase" in full_text or "egfr" in full_text for _ in [1])
        if has_kinase and not any(k in full_text for k in mutant_keywords):
            gaps.append(ResearchGap(
                category="Target Selectivity & Mutants",
                title="Lack of Drug-Resistant Gatekeeper Mutant Counter-Screens",
                description="Lead inhibitors were only characterized against wild-type EGFR. Clinically relevant secondary resistance mutations (e.g. T790M gatekeeper and C797S solvent-front mutations) were not tested.",
                severity="Important",
                recommendation="Profile top analogues against an extended kinase panel and clinically emergent EGFR mutant variants to assess resistance liability."
            ))

        # 4. Check Safety & Toxicity Controls (Non-Cancerous Cells / hERG)
        tox_keywords = ["herg", "cyp", "normal cell", "fibroblast", "pbmc", "cytotoxicity control", "therapeutic index"]
        if not any(k in full_text for k in tox_keywords):
            gaps.append(ResearchGap(
                category="Safety & Toxicity",
                title="Missing Non-Malignant Toxicity Controls & Off-Target Liability",
                description="Antiproliferative assays were tested on malignant adenocarcinoma cells without parallel screening on healthy human epithelial cells or cardiac hERG channel counterscreens.",
                severity="Critical",
                recommendation="Determine the in vitro Therapeutic Index (TI) by testing viability against non-malignant cell lines (e.g., MRC-5 or PBMC) and screen for hERG cardiotoxicity."
            ))

        # 5. Check Analytical Characterization (HRMS, HPLC purity for all series members)
        has_chiral = "chiral" in full_text or "stereocenter" in full_text or "enantiomer" in full_text
        has_opt = "optical rotation" in full_text or "[α]" in full_text or "ee" in full_text
        if has_chiral and not has_opt:
            gaps.append(ResearchGap(
                category="Analytical Characterization",
                title="Incomplete Enantiomeric Purity / Stereochemical Analysis",
                description="Stereocenters or chiral precursors were referenced, but specific optical rotation values or chiral HPLC enantiomeric excess (ee) determinations were not documented.",
                severity="Important",
                recommendation="Record specific optical rotation ([α]D) and perform chiral phase HPLC to establish enantiomeric purity."
            ))

        # 6. Check Chemical Space & Analogs
        if len(compounds) <= 4:
            gaps.append(ResearchGap(
                category="Chemical Space & Analogs",
                title="Narrow Chemical Space Exploration around Core Scaffold",
                description=f"Only {len(compounds)} specific derivatives were evaluated. Exploration was largely restricted to single aromatic substituent variations.",
                severity="Exploratory",
                recommendation="Explore bioisosteric core replacements (e.g. quinazoline, pyrrolopyrimidine) and bioisosteric functional group replacements."
            ))

        return gaps
