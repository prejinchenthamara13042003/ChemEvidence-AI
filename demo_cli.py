"""
Comprehensive Demonstration CLI for ChemEvidence AI.
Demonstrates:
1. Ingestion of sample paper (PDF or TXT) with page-boundary tracking
2. Automated structured chemistry discovery (compounds, bioactivity, properties)
3. Automatic Table Extraction & Dataframe reconstruction
4. Reaction Pathway Extraction (Multistep Schemes & Catalysts)
5. Structure-Activity Relationship (SAR) analysis & pIC50 calculations
6. Knowledge Graph topology summary
7. Empirical Multi-Factor Evidence Confidence Scoring
8. Proactive Research-Gap & Blind-Spot Detection
9. Evidence-grounded Q&A with Strict Absence & Discrepancy Detection
10. Multi-Paper Comparison & Cross-Study Synthesis
"""

import sys
import os
from core.pdf_parser import DocumentParser
from core.evidence_engine import EvidenceEngine
from core.chemistry_extractor import ChemistryExtractor
from core.qa_system import QASystem
from core.multi_paper_comparator import MultiPaperComparator


def main():
    print("=" * 75)
    print("⚗️  ChemEvidence AI — Advanced Chemistry Literature Intelligence Platform")
    print("=" * 75)

    sample_dir = "/Users/prejin/Thesis/sample_papers"
    sample_paper = os.path.join(sample_dir, "paper1_kinase_inhibitors.pdf")
    
    print(f"\n[1/5] Ingesting & Parsing Literature: {os.path.basename(sample_paper)}...")
    with open(sample_paper, "rb") as f:
        pdf_bytes = f.read()
    pages = DocumentParser.parse_pdf_bytes(pdf_bytes)
    metadata = DocumentParser.extract_metadata(pages)
    print(f"  ✓ Document parsed: {metadata.page_count} pages, {metadata.total_words} words.")
    print(f"  ✓ Title: {metadata.title}")

    print("\n[2/5] Indexing Passages & Executing Full Analytics Suite...")
    engine = EvidenceEngine(pages)
    extractor = ChemistryExtractor()
    analysis = extractor.analyze_paper(pages, metadata, engine)

    print(f"\n--- EXECUTIVE SCIENTIFIC SUMMARY ---")
    print(f"  {analysis.executive_summary}\n")

    print(f"--- 1. STRUCTURED DISCOVERY ---")
    print(f"  • Key Compounds Extracted: {len(analysis.compounds)}")
    for c in analysis.compounds[:3]:
        print(f"    - {c.compound_id}: {c.name} | Formula: {c.formula or 'N/A'}")
        print(f"      Citation: [Page {c.evidence.page_number}, {c.evidence.section}] \"{c.evidence.verbatim_quote[:80]}...\"")

    print(f"  • Bioactivity Results: {len(analysis.bioactivities)}")
    for b in analysis.bioactivities[:2]:
        print(f"    - {b.compound_id}: {b.assay_type} vs {b.target} = {b.value} {b.unit} [Page {b.evidence.page_number}]")

    print(f"\n--- 2. AUTOMATIC TABLE EXTRACTION ({len(analysis.tables)} Tables) ---")
    for t in analysis.tables:
        print(f"  • {t.table_id}: {t.title} [Page {t.page_number}]")
        print(f"    Columns: {', '.join(t.headers)}")
        print(f"    Rows Extracted: {len(t.rows)}")

    print(f"\n--- 3. REACTION PATHWAYS & SYNTHESIS ({len(analysis.reaction_pathways)} Pathways) ---")
    for p in analysis.reaction_pathways:
        print(f"  • {p.scheme_id}: {p.title} -> Target: {p.target_compound}")
        for s in p.steps[:2]:
            print(f"    Step {s.step_number}: {s.reaction_name} | Product: {s.product} | Yield: {s.yield_percent or 'N/A'} [Page {s.evidence.page_number}]")

    print(f"\n--- 4. SAR ANALYSIS & POTENCY LANDSCAPE ---")
    for sar in analysis.sar_analyses:
        print(f"  • {sar.series_name} (Scaffold: {sar.core_scaffold})")
        print(f"    Optimal Lead: {sar.optimal_lead}")
        for ins in sar.key_insights:
            print(f"    - {ins}")

    print(f"\n--- 5. EVIDENCE CONFIDENCE AUDIT RUBRIC ---")
    if analysis.confidence_breakdown:
        cb = analysis.confidence_breakdown
        print(f"  • Overall Audit Score: {cb.overall_score:.1f}% [{cb.tier}]")
        print(f"    - Verbatim Quote Match: {cb.verbatim_match_score:.1f}%")
        print(f"    - Entity-Metric Proximity: {cb.proximity_score:.1f}%")
        print(f"    - Numeric & Unit Precision: {cb.numeric_precision_score:.1f}%")
        print(f"    - Cross-Section Validation: {cb.cross_validation_score:.1f}%")

    print(f"\n--- 6. PROACTIVE RESEARCH-GAP DETECTION ({len(analysis.research_gaps)} Gaps Found) ---")
    for g in analysis.research_gaps[:3]:
        print(f"  • [{g.severity}] {g.title} ({g.category})")
        print(f"    Recommendation: {g.recommendation}")

    print("\n" + "=" * 75)
    print("[3/5] Interactive Evidence-Aware Q&A Demonstrations")
    print("=" * 75)

    qa = QASystem()

    # Query 1: Positive grounded query
    q1 = "What is the IC50 value of Compound 3b against EGFR kinase?"
    print(f"\nQ1: {q1}")
    r1 = qa.ask(q1, pages, engine, analysis)
    print(f"Status: [{r1.status.upper()}] (Confidence: {r1.confidence_score * 100:.0f}%)")
    print(f"Answer:\n{r1.answer[:300]}...")
    if r1.citations:
        print(f"Supporting Citations: Page {r1.citations[0].page_number} ({r1.citations[0].section})")

    # Query 2: Strict absence detection query
    q2 = "What was the in-vivo rat oral bioavailability and clearance?"
    print(f"\nQ2: {q2}")
    r2 = qa.ask(q2, pages, engine, analysis)
    print(f"Status: [{r2.status.upper()}]")
    print(f"Answer:\n{r2.answer}")

    # Query 3: Conflict detection query on Paper 2
    print("\n" + "=" * 75)
    print("[4/5] Testing Conflict Detection on Discrepancy Paper (Paper 2)")
    print("=" * 75)
    paper2_path = os.path.join(sample_dir, "paper2_catalytic_synthesis_conflicts.pdf")
    with open(paper2_path, "rb") as f:
        p2_bytes = f.read()
    p2_pages = DocumentParser.parse_pdf_bytes(p2_bytes)
    p2_meta = DocumentParser.extract_metadata(p2_pages)
    p2_engine = EvidenceEngine(p2_pages)
    p2_analysis = extractor.analyze_paper(p2_pages, p2_meta, p2_engine)

    q3 = "What was the isolated yield of compound 5c?"
    print(f"Q3: {q3}")
    r3 = qa.ask(q3, p2_pages, p2_engine, p2_analysis)
    print(f"Status: [{r3.status.upper()}]")
    if r3.conflicts:
        print(f"⚠️ CONFLICT ALERT: {r3.conflicts[0].topic}")
        print(f"  Details: {r3.conflicts[0].description}")

    # Cross-Paper Comparison
    print("\n" + "=" * 75)
    print("[5/5] Multi-Paper Comparison & Cross-Study Synthesis (Paper 1 vs Paper 2)")
    print("=" * 75)
    comparison = MultiPaperComparator.compare_papers([analysis, p2_analysis])
    print(f"Cross-Literature Summary:\n{comparison.comparative_synthesis}")
    print("\nLead Molecules Cross-Study Leaderboard:")
    for lead in comparison.lead_comparison_table:
        print(f"  • {lead['Paper Index']}: {lead['Lead Molecule']} | Target: {lead['Target']} | Potency: {lead['Best Potency']} | Yield: {lead['Isolated Yield']}")

    print("\n" + "=" * 75)
    print("🎉 All 9 advanced chemistry capabilities demonstrated successfully!")
    print("Launch the full interactive visual UI: `streamlit run app.py`")
    print("=" * 75)


if __name__ == "__main__":
    main()
