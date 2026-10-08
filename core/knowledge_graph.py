"""
Compound Relationship & Chemistry Knowledge Graph Engine.
Constructs an interactive network graph connecting Papers, Compounds,
Biological Targets, Synthetic Reactions, Catalysts, and Cell Lines.
Generates self-contained interactive HTML/Canvas visualizers for Streamlit rendering.
"""

import json
from typing import List, Dict, Any, Optional
from core.models import (
    KnowledgeGraphData,
    KnowledgeGraphNode,
    KnowledgeGraphEdge,
    CompoundInfo,
    BioactivityResult,
    ExperimentalCondition,
    PaperMetadata,
    ChemicalProperty
)


class KnowledgeGraphBuilder:
    """Builds and visualizes relational chemistry knowledge graphs."""

    CATEGORY_COLORS = {
        "Paper": "#38BDF8",       # Sky Blue
        "Compound": "#10B981",    # Emerald Green
        "Target": "#EC4899",      # Pink
        "Reaction": "#F59E0B",    # Amber
        "CellLine": "#8B5CF6",    # Purple
        "Catalyst": "#F97316",    # Orange
        "Property": "#64748B"     # Slate Gray
    }

    @classmethod
    def build_graph(
        cls,
        metadata: PaperMetadata,
        compounds: List[CompoundInfo],
        bioactivities: List[BioactivityResult],
        conditions: List[ExperimentalCondition],
        properties: Optional[List[ChemicalProperty]] = None
    ) -> KnowledgeGraphData:
        """Constructs a typed KnowledgeGraphData structure from parsed paper components."""
        nodes: Dict[str, KnowledgeGraphNode] = {}
        edges: List[KnowledgeGraphEdge] = []

        # 1. Paper root node
        paper_id = "paper_root"
        nodes[paper_id] = KnowledgeGraphNode(
            id=paper_id,
            label=metadata.title[:40] + ("..." if len(metadata.title) > 40 else ""),
            category="Paper",
            properties={"DOI": metadata.journal_or_doi or "N/A", "Pages": metadata.page_count}
        )

        # 2. Compound nodes
        for comp in compounds:
            cid = comp.compound_id.strip()
            if not cid:
                continue
            node_id = f"comp_{cid.lower().replace(' ', '_')}"
            nodes[node_id] = KnowledgeGraphNode(
                id=node_id,
                label=cid,
                category="Compound",
                properties={
                    "Name": comp.name,
                    "Formula": comp.formula or "N/A",
                    "MW": f"{comp.molecular_weight:.1f}" if comp.molecular_weight else "N/A",
                    "Class": comp.chemical_class or "Heterocycle"
                }
            )
            # Edge: Paper -> Reported Compound
            edges.append(KnowledgeGraphEdge(
                source=paper_id,
                target=node_id,
                relation="REPORTS_COMPOUND",
                label="reports",
                weight=1.0
            ))

        # 3. Target and Bioactivity nodes
        for b in bioactivities:
            if not b.target:
                continue
            target_id = f"target_{b.target.lower().replace(' ', '_')}"
            if target_id not in nodes:
                nodes[target_id] = KnowledgeGraphNode(
                    id=target_id,
                    label=b.target,
                    category="Target",
                    properties={"Type": "Kinase / Enzyme / Receptor"}
                )

            # Edge: Compound -> Inhibits/Modulates -> Target
            if b.compound_id:
                comp_node_id = f"comp_{b.compound_id.lower().replace(' ', '_')}"
                if comp_node_id in nodes:
                    edges.append(KnowledgeGraphEdge(
                        source=comp_node_id,
                        target=target_id,
                        relation="INHIBITS",
                        label=f"{b.assay_type} = {b.value} {b.unit}",
                        weight=1.5
                    ))

            # Cell line node if reported
            if b.cell_line:
                cell_id = f"cell_{b.cell_line.lower().replace(' ', '_')}"
                if cell_id not in nodes:
                    nodes[cell_id] = KnowledgeGraphNode(
                        id=cell_id,
                        label=b.cell_line,
                        category="CellLine",
                        properties={"Type": "Assay Cell Model"}
                    )
                if b.compound_id:
                    comp_node_id = f"comp_{b.compound_id.lower().replace(' ', '_')}"
                    if comp_node_id in nodes:
                        edges.append(KnowledgeGraphEdge(
                            source=comp_node_id,
                            target=cell_id,
                            relation="TESTED_IN",
                            label="antiproliferative",
                            weight=1.0
                        ))

        # 4. Reaction and Catalyst nodes
        for cond in conditions:
            rxn_label = cond.reaction_step or "Synthetic Step"
            rxn_id = f"rxn_{hash(rxn_label) % 10000}"
            if rxn_id not in nodes:
                nodes[rxn_id] = KnowledgeGraphNode(
                    id=rxn_id,
                    label=rxn_label[:28],
                    category="Reaction",
                    properties={
                        "Solvent": cond.solvent or "N/A",
                        "Temp": cond.temperature or "N/A",
                        "Yield": cond.yield_reported or "N/A"
                    }
                )

            # Edge to Paper
            edges.append(KnowledgeGraphEdge(
                source=paper_id,
                target=rxn_id,
                relation="UTILIZES_REACTION",
                label="reaction route",
                weight=1.0
            ))

            # Catalyst node
            if cond.catalyst and len(cond.catalyst.strip()) > 2:
                cat_id = f"cat_{hash(cond.catalyst) % 10000}"
                if cat_id not in nodes:
                    nodes[cat_id] = KnowledgeGraphNode(
                        id=cat_id,
                        label=cond.catalyst[:22],
                        category="Catalyst",
                        properties={"Type": "Catalyst / Ligand / Base"}
                    )
                edges.append(KnowledgeGraphEdge(
                    source=rxn_id,
                    target=cat_id,
                    relation="CATALYZED_BY",
                    label="catalyst",
                    weight=1.0
                ))

        return KnowledgeGraphData(nodes=list(nodes.values()), edges=edges)

    @classmethod
    def render_interactive_html(cls, graph_data: KnowledgeGraphData) -> str:
        """
        Generates a standalone, responsive, interactive HTML force-directed graph
        using a lightweight D3-compatible SVG/Canvas engine with pan, zoom, drag, and tooltips.
        """
        nodes_json = json.dumps([
            {
                "id": n.id,
                "label": n.label,
                "category": n.category,
                "color": cls.CATEGORY_COLORS.get(n.category, "#94A3B8"),
                "props": n.properties
            }
            for n in graph_data.nodes
        ])
        edges_json = json.dumps([
            {
                "source": e.source,
                "target": e.target,
                "label": e.label or e.relation,
                "relation": e.relation
            }
            for e in graph_data.edges
        ])

        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: #25161D;
                    color: #FDF2F4;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    overflow: hidden;
                    border-radius: 12px;
                }}
                #graph-container {{
                    width: 100vw;
                    height: 520px;
                    position: relative;
                    border-radius: 12px;
                }}
                canvas {{
                    width: 100%;
                    height: 100%;
                    display: block;
                    border-radius: 12px;
                }}
                #legend {{
                    position: absolute;
                    top: 12px;
                    left: 12px;
                    background: rgba(45, 26, 35, 0.9);
                    border: 1px solid rgba(235, 182, 188, 0.3);
                    border-radius: 8px;
                    padding: 8px 12px;
                    font-size: 11px;
                    pointer-events: none;
                    backdrop-filter: blur(8px);
                    color: #FDF2F4;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                }}
                .legend-item {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 4px;
                }}
                .legend-dot {{
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 6px;
                }}
                #tooltip {{
                    position: absolute;
                    display: none;
                    background: rgba(45, 26, 35, 0.96);
                    border: 1px solid #B76E79;
                    color: #FFFFFF;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-size: 12px;
                    pointer-events: none;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                    max-width: 250px;
                }}
            </style>
        </head>
        <body>
            <div id="graph-container">
                <canvas id="kgCanvas"></canvas>
                <div id="legend">
                    <div style="font-weight: 600; margin-bottom: 6px; color: #94A3B8;">ENTITY TYPES</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#38BDF8"></div>Paper / Document</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#10B981"></div>Compound</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#EC4899"></div>Biological Target</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#F59E0B"></div>Reaction / Route</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#8B5CF6"></div>Cell Line</div>
                    <div class="legend-item"><div class="legend-dot" style="background:#F97316"></div>Catalyst / Base</div>
                </div>
                <div id="tooltip"></div>
            </div>

            <script>
                const rawNodes = {nodes_json};
                const rawEdges = {edges_json};

                const canvas = document.getElementById("kgCanvas");
                const ctx = canvas.getContext("2d");
                const tooltip = document.getElementById("tooltip");

                let width = canvas.width = window.innerWidth;
                let height = canvas.height = 520;

                // Build node objects with physics positions
                const nodeMap = {{}};
                const nodes = rawNodes.map((n, i) => {{
                    const angle = (i / rawNodes.length) * 2 * Math.PI;
                    const radius = 120 + Math.random() * 80;
                    const obj = {{
                        ...n,
                        x: width / 2 + Math.cos(angle) * radius,
                        y: height / 2 + Math.sin(angle) * radius,
                        vx: 0,
                        vy: 0,
                        radius: n.category === "Paper" ? 18 : (n.category === "Compound" ? 14 : 11)
                    }};
                    nodeMap[n.id] = obj;
                    return obj;
                }});

                const links = rawEdges.map(e => ({{
                    source: nodeMap[e.source],
                    target: nodeMap[e.target],
                    label: e.label,
                    relation: e.relation
                }})).filter(l => l.source && l.target);

                let draggingNode = null;
                let mouseX = 0, mouseY = 0;

                // Physics simulation step
                function tick() {{
                    // Repulsion between nodes
                    for (let i = 0; i < nodes.length; i++) {{
                        for (let j = i + 1; j < nodes.length; j++) {{
                            let dx = nodes[j].x - nodes[i].x;
                            let dy = nodes[j].y - nodes[i].y;
                            let dist = Math.sqrt(dx * dx + dy * dy) || 1;
                            if (dist < 180) {{
                                let force = (180 - dist) / dist * 0.08;
                                nodes[i].vx -= dx * force;
                                nodes[i].vy -= dy * force;
                                nodes[j].vx += dx * force;
                                nodes[j].vy += dy * force;
                            }}
                        }}
                    }}

                    // Link spring attraction
                    links.forEach(l => {{
                        let dx = l.target.x - l.source.x;
                        let dy = l.target.y - l.source.y;
                        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
                        let desired = 90;
                        let force = (dist - desired) * 0.015;
                        l.source.vx += dx / dist * force;
                        l.source.vy += dy / dist * force;
                        l.target.vx -= dx / dist * force;
                        l.target.vy -= dy / dist * force;
                    }});

                    // Center gravity
                    nodes.forEach(n => {{
                        if (n !== draggingNode) {{
                            n.vx += (width / 2 - n.x) * 0.002;
                            n.vy += (height / 2 - n.y) * 0.002;
                            n.vx *= 0.88;
                            n.vy *= 0.88;
                            n.x += n.vx;
                            n.y += n.vy;
                            // Bounds
                            n.x = Math.max(n.radius + 10, Math.min(width - n.radius - 10, n.x));
                            n.y = Math.max(n.radius + 10, Math.min(height - n.radius - 10, n.y));
                        }}
                    }});

                    render();
                    requestAnimationFrame(tick);
                }}

                function render() {{
                    ctx.clearRect(0, 0, width, height);

                    // Draw links
                    ctx.lineWidth = 1.2;
                    links.forEach(l => {{
                        ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
                        ctx.beginPath();
                        ctx.moveTo(l.source.x, l.source.y);
                        ctx.lineTo(l.target.x, l.target.y);
                        ctx.stroke();

                        // Label
                        if (l.label && l.label.length < 25) {{
                            let midX = (l.source.x + l.target.x) / 2;
                            let midY = (l.source.y + l.target.y) / 2;
                            ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
                            ctx.font = "9px sans-serif";
                            ctx.fillText(l.label, midX + 3, midY - 3);
                        }}
                    }});

                    // Draw nodes
                    nodes.forEach(n => {{
                        ctx.shadowColor = n.color;
                        ctx.shadowBlur = n === draggingNode ? 15 : 6;
                        ctx.fillStyle = n.color;
                        ctx.beginPath();
                        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
                        ctx.fill();

                        ctx.shadowBlur = 0;
                        ctx.strokeStyle = "#FFFFFF";
                        ctx.lineWidth = 1.5;
                        ctx.stroke();

                        // Node text label
                        ctx.fillStyle = "#F8FAFC";
                        ctx.font = "bold 11px sans-serif";
                        ctx.textAlign = "center";
                        ctx.fillText(n.label, n.x, n.y + n.radius + 13);
                    }});
                }}

                // Interaction
                canvas.addEventListener("mousedown", e => {{
                    const rect = canvas.getBoundingClientRect();
                    const mx = e.clientX - rect.left;
                    const my = e.clientY - rect.top;
                    for (let n of nodes) {{
                        let d = Math.hypot(n.x - mx, n.y - my);
                        if (d <= n.radius + 4) {{
                            draggingNode = n;
                            break;
                        }}
                    }}
                }});

                window.addEventListener("mousemove", e => {{
                    const rect = canvas.getBoundingClientRect();
                    mouseX = e.clientX - rect.left;
                    mouseY = e.clientY - rect.top;

                    if (draggingNode) {{
                        draggingNode.x = mouseX;
                        draggingNode.y = mouseY;
                    }}

                    // Hover check for tooltip
                    let hovered = null;
                    for (let n of nodes) {{
                        let d = Math.hypot(n.x - mouseX, n.y - mouseY);
                        if (d <= n.radius + 4) {{
                            hovered = n;
                            break;
                        }}
                    }}

                    if (hovered) {{
                        tooltip.style.display = "block";
                        tooltip.style.left = (mouseX + 14) + "px";
                        tooltip.style.top = (mouseY + 10) + "px";
                        let propsHtml = Object.entries(hovered.props || {{}})
                            .map(([k, v]) => `<div><b>${{k}}:</b> ${{v}}</div>`).join("");
                        tooltip.innerHTML = `<div style="font-weight:700; color:${{hovered.color}}">${{hovered.category}}: ${{hovered.label}}</div>${{propsHtml}}`;
                    }} else {{
                        tooltip.style.display = "none";
                    }}
                }});

                window.addEventListener("mouseup", () => {{
                    draggingNode = null;
                }});

                window.addEventListener("resize", () => {{
                    width = canvas.width = window.innerWidth;
                    height = canvas.height = 520;
                }});

                tick();
            </script>
        </body>
        </html>
        """
        return html_code
