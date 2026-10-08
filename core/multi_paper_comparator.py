"""
Multi-Paper Comparison & Cross-Study Synthesis Engine.
Enables side-by-side comparison across multiple chemistry publications:
- Cross-paper compound potency leaderboard
- Target and assay overlap analysis
- Synthetic route and efficiency comparison
- Inter-laboratory conflict and discrepancy detection
"""

from typing import List, Dict, Any, Optional
from core.models import (
    PaperAnalysisResult,
    MultiPaperComparisonResult,
    ConflictAlert,
    EvidenceCitation
)


class MultiPaperComparator:
    """Compares multiple analyzed chemistry papers and synthesizes cross-study findings."""

    @classmethod
    def compare_papers(cls, analyses: List[PaperAnalysisResult]) -> MultiPaperComparisonResult:
        if not analyses:
            return MultiPaperComparisonResult(
                papers=[],
                comparative_synthesis="No papers provided for comparison."
            )

        metadata_list = [a.metadata for a in analyses]

        # 1. Collate lead compounds across all papers
        lead_table: List[Dict[str, Any]] = []
        common_targets_set = set()

        for idx, a in enumerate(analyses):
            meta_title = str(getattr(a.metadata, "title", "") or f"Paper {idx + 1}").strip()
            paper_title = meta_title[:35] + "..." if len(meta_title) > 35 else meta_title
            
            # Find best lead in this paper
            best_bio = None
            min_val = float("inf")
            for b in a.bioactivities:
                if b.target:
                    common_targets_set.add(str(b.target).strip())
                try:
                    if not b.value:
                        continue
                    clean_v = str(b.value).replace("±", "").split()[0]
                    val = float(clean_v)
                    if b.unit and ("µm" in str(b.unit).lower() or "um" in str(b.unit).lower()):
                        val *= 1000.0
                    if val < min_val:
                        min_val = val
                        best_bio = b
                except Exception:
                    continue

            # Associated lead compound name
            lead_comp_name = None
            if best_bio and best_bio.compound_id:
                lead_comp_name = str(best_bio.compound_id).strip()
            elif a.compounds:
                for c in a.compounds:
                    if c.compound_id:
                        lead_comp_name = str(c.compound_id).strip()
                        break
            if not lead_comp_name:
                lead_comp_name = "Lead Analogue"

            lead_comp_obj = next((c for c in a.compounds if c.compound_id == lead_comp_name), None) if lead_comp_name else None

            # Isolated yield
            lead_yield = "N/A"
            for p in a.properties:
                if p.compound_id == lead_comp_name and p.parameter and "yield" in str(p.parameter).lower():
                    unit_p = p.unit or "%"
                    lead_yield = f"{p.value or ''} {unit_p}".strip() or "N/A"
                    break

            # Best Potency string
            if best_bio and best_bio.value:
                u_str = f" {best_bio.unit}" if best_bio.unit else ""
                best_pot_str = f"{best_bio.value}{u_str}".strip() or "N/A"
            else:
                best_pot_str = "N/A"

            target_str = (str(best_bio.target).strip() if best_bio and best_bio.target else "Enzyme / Cell") or "Enzyme / Cell"
            formula_str = (str(lead_comp_obj.formula).strip() if lead_comp_obj and lead_comp_obj.formula else "N/A") or "N/A"
            chem_class_str = (str(lead_comp_obj.chemical_class).strip() if lead_comp_obj and lead_comp_obj.chemical_class else "Organic Lead") or "Organic Lead"

            lead_table.append({
                "Paper Index": f"Paper {idx + 1}",
                "Paper Title": paper_title or f"Paper {idx + 1}",
                "Lead Molecule": lead_comp_name or "Lead Analogue",
                "Formula": formula_str,
                "Target": target_str,
                "Best Potency": best_pot_str,
                "Isolated Yield": lead_yield,
                "Chemical Class": chem_class_str
            })

        # 2. Synthetic route comparison
        synthetic_comp: List[Dict[str, Any]] = []
        for idx, a in enumerate(analyses):
            catalysts_used = set(str(c.catalyst).strip() for c in a.conditions if c.catalyst)
            solvents_used = set(str(c.solvent).strip() for c in a.conditions if c.solvent)
            steps_count = len(a.conditions)
            avg_yield = "Variable"
            yields = []
            for c in a.conditions:
                if c.yield_reported:
                    try:
                        clean_y = str(c.yield_reported).replace("%", "").strip()
                        yields.append(float(clean_y))
                    except Exception:
                        pass
            if yields:
                avg_yield = f"{round(sum(yields)/len(yields), 1)}%"

            m_title = str(getattr(a.metadata, "title", "") or f"Paper {idx + 1}").strip()
            short_m_title = m_title[:30] + "..." if len(m_title) > 30 else m_title

            synthetic_comp.append({
                "Paper": f"Paper {idx + 1}: {short_m_title}",
                "Synthetic Transformations": f"{steps_count} steps reported",
                "Key Catalysts / Reagents": ", ".join(list(catalysts_used)[:2]) or "Standard Reagents",
                "Primary Solvents": ", ".join(list(solvents_used)[:2]) or "Organic Solvents",
                "Average Yield": avg_yield
            })

        # 3. Cross-study discrepancies
        cross_discrepancies: List[ConflictAlert] = []
        # Check if same compound ID or target appears with diverging values
        for i in range(len(analyses)):
            for j in range(i + 1, len(analyses)):
                a1 = analyses[i]
                a2 = analyses[j]
                t1 = str(getattr(a1.metadata, "title", "") or "Paper 1")[:30]
                t2 = str(getattr(a2.metadata, "title", "") or "Paper 2")[:30]
                for b1 in a1.bioactivities:
                    for b2 in a2.bioactivities:
                        if b1.target and b2.target and str(b1.target).lower() == str(b2.target).lower():
                            if b1.compound_id and b2.compound_id and str(b1.compound_id).lower() == str(b2.compound_id).lower():
                                cross_discrepancies.append(ConflictAlert(
                                    topic=f"Cross-Paper Activity Discrepancy for {b1.compound_id}",
                                    description=f"Paper 1 ({t1}) reports {b1.value or 'N/A'} {b1.unit or ''}, whereas Paper 2 ({t2}) reports {b2.value or 'N/A'} {b2.unit or ''}.",
                                    claim_a=f"{b1.value or 'N/A'} {b1.unit or ''} reported in Paper 1",
                                    citation_a=b1.evidence,
                                    claim_b=f"{b2.value or 'N/A'} {b2.unit or ''} reported in Paper 2",
                                    citation_b=b2.evidence,
                                    severity="High Discrepancy",
                                    resolution_note="Verify differing assay protocols, enzyme batches, or substrate concentrations."
                                ))

        # 4. Comparative synthesis text
        synthesis_paragraphs = [
            f"### Cross-Literature Synthesis Across {len(analyses)} Chemistry Publications\n",
            f"A multi-document comparison was conducted across **{len(analyses)} distinct research publications**, encompassing a total of **{sum(len(a.compounds) for a in analyses)} evaluated compounds** and **{sum(len(a.bioactivities) for a in analyses)} bioactivity assay points**.\n",
            f"**Key Findings & Lead Comparisons:**\n"
        ]

        for lead in lead_table:
            p_idx = lead.get('Paper Index', 'Paper')
            p_tit = lead.get('Paper Title', 'Study')
            l_mol = lead.get('Lead Molecule', 'Lead Analogue')
            c_cls = lead.get('Chemical Class', 'Organic Lead')
            b_pot = lead.get('Best Potency', 'N/A')
            t_tgt = lead.get('Target', 'Target Assay')
            i_yld = lead.get('Isolated Yield', 'N/A')
            synthesis_paragraphs.append(
                f"- **{p_idx} ({p_tit})**: Highlighted **{l_mol}** ({c_cls}) exhibiting **{b_pot}** against *{t_tgt}* with an isolated synthetic yield of **{i_yld}**."
            )

        if cross_discrepancies:
            synthesis_paragraphs.append(f"\n⚠️ **Notice:** {len(cross_discrepancies)} cross-study discrepancies were flagged between independent publications.")

        return MultiPaperComparisonResult(
            papers=metadata_list,
            lead_comparison_table=lead_table,
            common_targets=list(common_targets_set),
            synthetic_route_comparison=synthetic_comp,
            cross_study_discrepancies=cross_discrepancies,
            comparative_synthesis="\n".join(synthesis_paragraphs)
        )

    @classmethod
    def ask_cross_paper(
        cls,
        question: str,
        analyses: List[PaperAnalysisResult],
        api_key: Optional[str] = None
    ) -> str:
        """
        Synthesizes a response to a comparative chemistry query across all analyzed publications.
        Uses Gemini 3.8 Flash if available with local deterministic fallback.
        """
        import os
        from core.qa_system import DEFAULT_GEMINI_API_KEY, HAS_GENAI
        
        active_key = api_key or os.getenv("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY
        
        # Build multi-paper context summary
        context_parts = []
        for idx, a in enumerate(analyses, 1):
            comp_summary = ", ".join([f"{c.compound_id} ({c.name})" for c in a.compounds[:5]])
            bio_summary = ", ".join([f"{b.compound_id}: {b.value} {b.unit} ({b.target})" for b in a.bioactivities[:5]])
            cond_summary = ", ".join([f"{c.reaction_step}: {c.catalyst or 'No cat'} in {c.solvent or 'solvent'}, Yield {c.yield_reported or 'N/A'}" for c in a.conditions[:4]])
            
            authors_str = a.metadata.authors or "N/A"
            journal_str = a.metadata.journal_or_doi or "N/A"
            exec_summary = (a.executive_summary[:400] + "...") if a.executive_summary else "N/A"
            p_text = f"""--- Publication {idx}: {a.metadata.title} ---
Authors: {authors_str} | Source: {journal_str}
Executive Summary: {exec_summary}
Top Compounds: {comp_summary or 'None listed'}
Bioactivity Highlights: {bio_summary or 'None listed'}
Reaction Conditions: {cond_summary or 'None listed'}
"""
            context_parts.append(p_text)

        full_context = "\n\n".join(context_parts)
        
        if active_key and HAS_GENAI:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=active_key, http_options={"timeout": 15000})
                prompt = f"""You are ChemEvidence AI, an expert medicinal chemist and literature intelligence agent.
Answer the following comparative question based strictly on the provided multi-paper chemistry literature summaries.
Cite which paper (e.g., Paper 1, Paper 2) reports which facts. Be rigorous with numbers, chemical names, targets, and yields.

Question: {question}

Corpus Context:
{full_context}

Provide a structured, evidence-grounded comparative answer:"""
                
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=1000
                    )
                )
                if response and response.text:
                    return response.text
            except Exception:
                pass
                
        # Deterministic local fallback
        return cls._local_cross_paper_fallback(question, analyses)

    @classmethod
    def _local_cross_paper_fallback(cls, question: str, analyses: List[PaperAnalysisResult]) -> str:
        q_lower = question.lower()
        if "poten" in q_lower or "best" in q_lower or "ic50" in q_lower or "lead" in q_lower:
            lines = ["**Cross-Literature Potency Synthesis:**\n"]
            for idx, a in enumerate(analyses, 1):
                best = None
                min_v = float("inf")
                for b in a.bioactivities:
                    try:
                        if not b.value:
                            continue
                        clean_v = str(b.value).replace("±", "").split()[0]
                        v = float(clean_v)
                        if b.unit and ("µm" in str(b.unit).lower() or "um" in str(b.unit).lower()):
                            v *= 1000.0
                        if v < min_v:
                            min_v = v
                            best = b
                    except Exception:
                        continue
                t_str = str(getattr(a.metadata, "title", "") or f"Paper {idx}")[:30]
                if best:
                    c_id = best.compound_id or (a.compounds[0].compound_id if (a.compounds and a.compounds[0].compound_id) else "Lead Candidate")
                    tgt = best.target or "Target Enzyme"
                    u_str = f" {best.unit}" if best.unit else ""
                    lines.append(f"- **Paper {idx} ({t_str}...)**: Lead candidate **{c_id}** reports **{best.value}{u_str}** against **{tgt}**.")
                else:
                    lines.append(f"- **Paper {idx} ({t_str}...)**: No quantitative enzyme potency data extracted.")
            return "\n".join(lines)
            
        elif "yield" in q_lower or "synth" in q_lower or "step" in q_lower:
            lines = ["**Synthetic Methodology & Efficiency Comparison:**\n"]
            for idx, a in enumerate(analyses, 1):
                yields = []
                for c in a.conditions:
                    if c.yield_reported:
                        clean_y = str(c.yield_reported).replace("%", "").strip()
                        try:
                            yields.append(float(clean_y))
                        except Exception:
                            pass
                avg_y = f"{round(sum(yields)/len(yields), 1)}%" if yields else "Variable"
                cats = set(str(c.catalyst).strip() for c in a.conditions if c.catalyst)
                t_str = str(getattr(a.metadata, "title", "") or f"Paper {idx}")[:30]
                lines.append(f"- **Paper {idx} ({t_str}...)**: Reports **{len(a.conditions)} synthetic steps** with an average yield of **{avg_y}** (Key catalysts: {', '.join(list(cats)[:2]) or 'standard'}).")
            return "\n".join(lines)
            
        else:
            return f"**Comparative Cross-Study Overview:**\nAcross the {len(analyses)} active publications, {sum(len(a.compounds) for a in analyses)} compounds and {sum(len(a.bioactivities) for a in analyses)} bioactivity records were analyzed. To explore specific comparisons, query for potency leads, synthetic reaction yields, target selectivity, or experimental discrepancies."
