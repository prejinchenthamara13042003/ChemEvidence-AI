"""
Structure-Activity Relationship (SAR) Engine & Analytics.
Extracts substituent-activity series, computes pIC50 values, analyzes fold potency improvements,
and builds interactive Plotly visualization figures for medicinal chemistry analysis.
"""

import math
import re
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px

from core.models import (
    SARAnalysis,
    SARDataPoint,
    CompoundInfo,
    BioactivityResult,
    ChemicalProperty,
    EvidenceCitation
)


class SAREngine:
    """Extracts, models, and visualizes Structure-Activity Relationships from chemical literature."""

    @classmethod
    def extract_sar_series(
        cls,
        compounds: List[CompoundInfo],
        bioactivities: List[BioactivityResult],
        properties: List[ChemicalProperty]
    ) -> List[SARAnalysis]:
        """Extracts SAR datapoints by correlating compounds with their bioactivities and substituents."""
        analyses: List[SARAnalysis] = []
        if not bioactivities:
            return analyses

        # Group bioactivities by target
        target_groups: Dict[str, List[BioactivityResult]] = {}
        for b in bioactivities:
            t = b.target or "Kinase Assay"
            target_groups.setdefault(t, []).append(b)

        # Build yield map
        yield_map: Dict[str, float] = {}
        for p in properties:
            if "yield" in p.parameter.lower() and p.compound_id:
                m = re.search(r"(\d+(?:\.\d+)?)", p.value)
                if m:
                    yield_map[p.compound_id.lower()] = float(m.group(1))

        for target_name, b_list in target_groups.items():
            datapoints: List[SARDataPoint] = []
            
            # Find baseline/parent compound for fold-change calculation
            baseline_ic50: Optional[float] = None
            parent_id = None

            for b in b_list:
                comp_id = b.compound_id or "Compound"
                val_str = b.value.replace("±", "").split()[0] if b.value else "0"
                ic50_num = None
                try:
                    ic50_num = float(val_str)
                    # Convert µM to nM if necessary
                    if b.unit and ("µm" in b.unit.lower() or "um" in b.unit.lower()):
                        ic50_num *= 1000.0
                except ValueError:
                    continue

                if ic50_num is not None and ic50_num > 0:
                    # Calculate pIC50: -log10(M) = 9 - log10(nM)
                    pic50 = round(9.0 - math.log10(ic50_num), 2)

                    # Determine substituent from quote or compound ID
                    sub_pos, sub_grp = cls._infer_substituent(comp_id, b.evidence.verbatim_quote)

                    y_val = yield_map.get(comp_id.lower())

                    dp = SARDataPoint(
                        compound_id=comp_id,
                        substituent_position=sub_pos,
                        substituent_group=sub_grp,
                        ic50_nm=round(ic50_num, 2),
                        pic50=pic50,
                        yield_percent=y_val,
                        evidence=b.evidence
                    )
                    datapoints.append(dp)

                    if baseline_ic50 is None or "3a" in comp_id.lower() or "parent" in b.evidence.verbatim_quote.lower():
                        baseline_ic50 = ic50_num
                        parent_id = comp_id

            if not datapoints:
                continue

            # Calculate fold improvement relative to baseline
            optimal_lead = None
            min_ic50 = float("inf")
            for dp in datapoints:
                if dp.ic50_nm and dp.ic50_nm > 0:
                    if baseline_ic50 and baseline_ic50 > 0:
                        fold = round(baseline_ic50 / dp.ic50_nm, 2)
                        dp.fold_improvement = fold
                    if dp.ic50_nm < min_ic50:
                        min_ic50 = dp.ic50_nm
                        optimal_lead = dp.compound_id

            # Synthesize key SAR insights
            insights = []
            if optimal_lead:
                lead_dp = next(d for d in datapoints if d.compound_id == optimal_lead)
                insights.append(f"**Lead Optimization**: {optimal_lead} emerged as the most potent derivative with IC50 = {lead_dp.ic50_nm} nM (pIC50 = {lead_dp.pic50}).")
                if lead_dp.fold_improvement and lead_dp.fold_improvement > 1.0:
                    insights.append(f"**Potency Gain**: Displayed a {lead_dp.fold_improvement}x potency enhancement compared to parent {parent_id or 'baseline'}.")
                if lead_dp.substituent_group:
                    insights.append(f"**Steric/Electronic Effect**: Introducing {lead_dp.substituent_group} at {lead_dp.substituent_position or 'core scaffold'} was key to the activity boost.")

            analyses.append(SARAnalysis(
                series_name=f"SAR Series against {target_name}",
                core_scaffold="Heterocyclic Kinase Inhibitor / Small Molecule",
                target=target_name,
                datapoints=datapoints,
                key_insights=insights,
                optimal_lead=optimal_lead
            ))

        return analyses

    @staticmethod
    def _infer_substituent(comp_id: str, quote: str) -> Tuple[Optional[str], Optional[str]]:
        """Infers substituent position and functional group from context."""
        quote_l = quote.lower()
        
        # Check explicit R-group table entries (e.g., 4-OMe, 4-CF3, 4-Me, 3-CN)
        r_match = re.search(r"\b([23456]\-(?:OMe|CF3|Me|CN|Cl|F|Br|OEt|OH|NH2|iPr|tBu))\b", quote, re.IGNORECASE)
        if r_match:
            grp = r_match.group(1)
            pos = grp[:2]
            return pos, grp

        # Check substituent words in text
        if "ethoxy" in quote_l:
            return "C-5", "5-Ethoxy (-OEt)"
        elif "methoxy" in quote_l:
            return "C-4'", "4-Methoxy (-OMe)"
        elif "pyridine" in quote_l or "pyridyl" in quote_l:
            return "Heteroaryl", "3-Pyridyl"
        elif "phenyl" in quote_l:
            return "Aryl", "Phenyl (-Ph)"
        elif "tert-butyl" in quote_l:
            return "para", "4-(tert-butyl)"

        return "Scaffold Position", comp_id

    # -------------------------------------------------------------
    # Interactive Plotly Visualization Creators
    # -------------------------------------------------------------

    @staticmethod
    def create_sar_potency_chart(sar: SARAnalysis) -> go.Figure:
        """Creates an interactive Plotly bar chart showing potency (IC50 in nM) and pIC50 values."""
        if not sar.datapoints:
            return go.Figure()

        df = pd.DataFrame([
            {
                "Compound": dp.compound_id,
                "Substituent": f"{dp.substituent_group or dp.compound_id}",
                "IC50 (nM)": dp.ic50_nm,
                "pIC50": dp.pic50,
                "Fold Improvement": dp.fold_improvement or 1.0,
                "Yield (%)": dp.yield_percent or 0
            }
            for dp in sar.datapoints if dp.ic50_nm is not None
        ])

        if df.empty:
            return go.Figure()

        # Sort by potency (lowest IC50 first)
        df = df.sort_values(by="IC50 (nM)", ascending=True)

        fig = go.Figure()

        # IC50 Bar
        fig.add_trace(go.Bar(
            x=df["Compound"],
            y=df["IC50 (nM)"],
            name="Enzyme IC50 (nM) [Lower = More Potent]",
            marker=dict(
                color=df["IC50 (nM)"],
                colorscale="Viridis_r",
                showscale=True,
                colorbar=dict(title="IC50 (nM)", len=0.7)
            ),
            text=[f"{v} nM<br>({f}x fold)" for v, f in zip(df["IC50 (nM)"], df["Fold Improvement"])],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Substituent: %{customdata[0]}<br>IC50: %{y} nM<br>pIC50: %{customdata[1]}<br>Fold Gain: %{customdata[2]}x<extra></extra>",
            customdata=df[["Substituent", "pIC50", "Fold Improvement"]]
        ))

        fig.update_layout(
            title=f"<b>{sar.series_name} — Compound Potency Landscape</b>",
            xaxis_title="<b>Synthesized Analogues</b>",
            yaxis_title="<b>IC50 (nM) — Logarithmic Potency Scale</b>",
            yaxis_type="log",
            template="plotly_white",
            paper_bgcolor="rgba(255, 255, 255, 0.85)",
            plot_bgcolor="rgba(253, 242, 240, 0.5)",
            font=dict(family="Inter, sans-serif", size=12, color="#2D1D22"),
            margin=dict(l=40, r=40, t=50, b=40),
            height=420
        )

        return fig

    @staticmethod
    def create_yield_vs_potency_scatter(sar: SARAnalysis) -> go.Figure:
        """Creates an interactive scatter plot of Synthetic Isolated Yield (%) vs Biological Potency (pIC50)."""
        data = [
            {
                "Compound": dp.compound_id,
                "Substituent": dp.substituent_group or dp.compound_id,
                "IC50 (nM)": dp.ic50_nm,
                "pIC50": dp.pic50,
                "Yield (%)": dp.yield_percent
            }
            for dp in sar.datapoints if dp.ic50_nm is not None and dp.yield_percent is not None
        ]

        if not data:
            return go.Figure()

        df = pd.DataFrame(data)

        fig = px.scatter(
            df,
            x="Yield (%)",
            y="pIC50",
            text="Compound",
            size=[18] * len(df),
            color="pIC50",
            color_continuous_scale="Viridis",
            hover_data=["Substituent", "IC50 (nM)", "Yield (%)"],
            title="<b>Synthetic Feasibility vs. Biological Potency (pIC50 vs. Isolated Yield %)</b>"
        )

        fig.update_traces(
            textposition="top center",
            marker=dict(line=dict(width=2, color="#FFFFFF"))
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(255, 255, 255, 0.85)",
            plot_bgcolor="rgba(253, 242, 240, 0.5)",
            xaxis_title="<b>Synthetic Isolated Yield (%)</b>",
            yaxis_title="<b>pIC50 Potency (-log10 M) [Higher = More Potent]</b>",
            font=dict(family="Inter, sans-serif", size=12, color="#2D1D22"),
            margin=dict(l=40, r=40, t=50, b=40),
            height=420
        )

        return fig

    @staticmethod
    def compare_compounds_matrix(
        compounds: List[CompoundInfo],
        bioactivities: List[BioactivityResult],
        properties: List[ChemicalProperty],
        selected_ids: List[str]
    ) -> pd.DataFrame:
        """Constructs a side-by-side comparison matrix for selected compounds."""
        records = []
        for cid in selected_ids:
            # Match compound info
            c_info = next((c for c in compounds if c.compound_id.lower() == cid.lower()), None)
            
            # Find bioactivities
            c_bios = [b for b in bioactivities if b.compound_id and b.compound_id.lower() == cid.lower()]
            bio_str = "; ".join([f"{b.assay_type} ({b.target}) = {b.value} {b.unit}" for b in c_bios]) or "Not Tested"

            # Find properties
            c_props = [p for p in properties if p.compound_id and p.compound_id.lower() == cid.lower()]
            yield_str = next((f"{p.value} {p.unit or '%'}" for p in c_props if "yield" in p.parameter.lower()), "N/A")
            mp_str = next((f"{p.value} {p.unit or '°C'}" for p in c_props if "melting" in p.parameter.lower()), "N/A")
            purity_str = next((f"{p.value} {p.unit or '%'}" for p in c_props if "purity" in p.parameter.lower()), "N/A")

            records.append({
                "Compound Identifier": cid,
                "Chemical Name": c_info.name if c_info else "N/A",
                "Molecular Formula": c_info.formula if c_info else "N/A",
                "Molecular Weight": f"{c_info.molecular_weight:.1f} g/mol" if (c_info and c_info.molecular_weight) else "N/A",
                "PubChem CID": c_info.pubchem_cid if (c_info and c_info.pubchem_cid) else "N/A",
                "Biological Activity": bio_str,
                "Isolated Yield": yield_str,
                "Melting Point": mp_str,
                "Purity (HPLC)": purity_str
            })

        return pd.DataFrame(records).set_index("Compound Identifier").T
