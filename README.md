<p align="center">
  <img src="assets/logo.png" width="300" alt="ChemEvidence AI Logo" style="border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);" />
</p>

# ⚗️ ChemEvidence AI: Evidence-Aware Intelligence Platform for Scientific Chemistry Literature

> **Applying LLM-based Agents to Automate Chemical Literature Extraction and Data Synthesis**

**ChemEvidence AI** is an intelligent, evidence-grounded AI system designed to accelerate the analysis of scientific chemistry publications (PDFs and manuscripts). It automates the discovery of structured chemical data, reconstructs synthetic reaction pathways, extracts data tables, models SAR trends with interactive Plotly analytics, builds relational knowledge graphs, audits evidence confidence, proactively detects literature research gaps, and powers an interactive, hallucination-resistant question-answering workflow with exact page and quotation citations.

---

## 🌟 Advanced System Capabilities

### 1. Multi-Paper Comparison & Cross-Study Synthesis
- Batch ingest and compare multiple publications side-by-side.
- Cross-paper **Lead Compounds Leaderboard** comparing top chemotypes, potency against shared targets, and synthetic efficiencies.
- Automatic **Cross-Study Discrepancy Detection** flagging inter-laboratory conflicts (e.g. differing IC50 values or yields reported across groups).

### 2. Automatic Table Extraction (Table Studio)
- Identifies and reconstructs structured scientific tables (SAR matrices, reaction condition optimization tables, physical properties).
- Robust parser supporting markdown pipe tables, tab/whitespace alignment, and vertical PDF-wrapped tables.
- Interactive DataFrames with one-click **Export to CSV** and **Export to Excel (.xlsx)**.

### 3. Reaction Pathway & Multistep Synthesis Extraction
- Reconstructs sequential synthetic routes (**Scheme 1: Starting Materials $\rightarrow$ Intermediates $\rightarrow$ Target Molecule**).
- Captures reaction transformations, catalysts, ligands, bases, solvents, temperatures, times, and isolated yields.
- Computes estimated overall synthetic sequence yields.

### 4. Chemical Entity Normalization
- Standardizes compound names, codes, and biological targets against standard nomenclature (PubChem CIDs, SMILES, InChIKeys, molecular weights).
- Resolves colloquial synonyms and paper-specific abbreviations (e.g., *Compound 3b* $\leftrightarrow$ *lead diaminopyrimidine*, *Erlotinib* $\leftrightarrow$ *Tarceva* $\leftrightarrow$ *OSI-774*).

### 5. Interactive Chemistry Knowledge Graph
- Interactive physics-based network graph linking **Publication $\leftrightarrow$ Compounds $\leftrightarrow$ Targets $\leftrightarrow$ Reaction Routes $\leftrightarrow$ Cell Lines $\leftrightarrow$ Catalysts**.
- Draggable nodes, live force-directed simulation, category color coding, and property tooltips.

### 6. Side-by-Side Similar Compound Comparator
- Select any 2 to 4 synthesized derivatives to view an instant side-by-side comparison matrix.
- Directly contrasts molecular formulas, weights, bioactivity assays, isolated yields, melting points, and HPLC purities.

### 7. SAR (Structure-Activity Relationship) Studio & Analytics
- Automated extraction of substituent series (e.g., $4\text{-OMe}$, $4\text{-CF}_3$, $4\text{-Me}$, $3\text{-CN}$, $\text{C-5}$ ethoxy).
- Calculates $-\log_{10}(\text{IC50 in M})$ as $\text{pIC}_{50}$ and computes fold-potency improvement over parent molecules.
- Interactive Plotly visualizations:
  - **Substituent Potency Bar Chart** ($\text{IC}_{50}\text{ in nM}$ logarithmic scale + fold gains).
  - **Synthetic Feasibility vs. Potency Scatter Plot** (Yield % vs. $\text{pIC}_{50}$).

### 8. Empirical Multi-Factor Evidence Confidence Scoring
- Replaces static confidence estimates with an empirical audit across four dimensions:
  1. **Verbatim Quote Match Score** (exact substring fidelity in source document).
  2. **Entity-Metric Proximity Score** (character and token distance between molecule and assay outcome).
  3. **Numeric & Unit Precision Score** (presence of standard units and error margins $\pm \text{SD}$).
  4. **Multi-Section Cross-Validation Score** (corroboration across Abstract, Results, and Tables).
- Assigns transparent audit tiers (`High (A+)`, `Strong (A)`, `Moderate (B)`, `Tentative (C)`).

### 9. Proactive Research-Gap & Blind-Spot Detection
- Automated peer-review audit identifying what was omitted or uninvestigated:
  - **Omitted In-Vivo Pharmacokinetics (ADME)**: Clearance, oral bioavailability, half-life.
  - **Lack of Gatekeeper Mutant Selectivity Panels**: Missing T790M/C797S resistance screening.
  - **Absence of In-Vivo Tumor Xenograft Models**: 2D in-vitro cell culture only.
  - **Missing Non-Malignant Toxicity Controls**: Counter-screens in healthy cells and hERG liabilities.
  - **Incomplete Stereochemical Characterization**: Missing optical rotation or chiral HPLC ee.
  - **Narrow Chemical Space**: Recommendations for bioisosteric diversification.

### 10. Strict Absence Detection & Interactive Q&A
- Rigorous anti-hallucination engine: queries regarding missing data explicitly return:
  > **"No supporting evidence was found in the uploaded paper."**
- Dual-mode intelligence: Powered by **Gemini 3.8 Flash** (`gemini-3.8-flash`) with an automatic offline deterministic fallback.

### 11. High-Resolution PDF Page Viewer (PyMuPDF / fitz)
- Direct high-resolution rendering of research paper PDFs (~150-200 DPI crisp previews) without external PDF plug-ins.
- Responsive page navigation controls: `◄ Prev Page | Page X of Y | Next Page ►` with direct page selector.
- PyMuPDF amber highlight box annotations (`stroke=(1.0, 0.72, 0.0)` with translucent amber fill) dynamically mapped to exact coordinates of verbatim citations.
- Synchronized text inspection with glowing yellow `<mark class="glowing-citation-mark">` tags for visual quote grounding.

### 12. "📄 Show in Paper & Highlight Quote" Action & Side-by-Side Split View
- Every Grounded Evidence Card across the entire platform features a prominent action:
  `📄 Show in Paper & Highlight Quote (Page X)`
- Clicking this action instantly:
  1. Switches to the **Side-by-Side Split View**.
  2. Jumps the left document reader to the exact citation page (`Page X`).
  3. Highlights the verbatim quote snippet directly on the PDF page image with PyMuPDF amber annotations and glowing yellow tags.
- **Side-by-Side Split View Layout**:
  - **Left Column (55% width)**: Live Document Viewer with page navigation, zoom controls (150%, 200%, 250%), and real-time amber/yellow quote highlights.
  - **Right Column (45% width)**: Chemistry Intelligence Suite featuring Q&A Assistant, Gemini AI reasoning engine, dynamic prompt chips, Grounded Evidence Cards, and seamless view switching across: **Structured Discovery • Table Extraction • Reaction Pathways • SAR Analytics • Knowledge Graph • Anti-Hallucination Audit**.

---

## 🏗️ Architecture & Project Structure

```
Thesis/
├── app.py                     # Modernized Streamlit Web Application (Side-by-Side Split View & 8 Tabs)
├── requirements.txt           # Python package dependencies (Streamlit, PyMuPDF, Plotly, PyPDF, GenAI)
├── .env.example               # Environment variables template
├── README.md                  # Documentation and usage guide
├── demo_cli.py                # Comprehensive 10-step CLI demonstration
├── generate_sample_pdfs.py    # Utility to render sample benchmark PDFs
│
├── src/                       # User-facing and UI components
│   ├── __init__.py
│   ├── pdf_parser.py          # High-resolution PDF renderer and PyMuPDF text/quote highlighter
│   └── ui_components.py       # Grounded Evidence Cards, page navigation, and SMILES inspector
│
├── core/
│   ├── __init__.py            # Module exports
│   ├── models.py              # Pydantic data schemas for extraction, SAR, graph & evidence
│   ├── pdf_parser.py          # Page-aware PDF and text document extractor (PyMuPDF powered)
│   ├── ui_components.py       # Re-exported UI components
│   ├── evidence_engine.py     # BM25 chunk indexer & exact quote verifier
│   ├── chemistry_extractor.py # 6-category structured chemistry extractor + post-enrichment
│   ├── table_extractor.py     # Grid, markdown, and vertical PDF table parser
│   ├── reaction_engine.py     # Multistep reaction pathway and Scheme reconstructor
│   ├── entity_normalizer.py   # Chemical synonym and reference database canonicalizer
│   ├── sar_engine.py          # SAR series analyzer & Plotly chart generator
│   ├── knowledge_graph.py     # Multi-entity relational knowledge graph & HTML5 physics visualizer
│   ├── confidence_scorer.py   # Empirical multi-factor confidence audit rubric
│   ├── gap_detector.py        # Proactive peer-review research gap detector
│   ├── multi_paper_comparator.py # Cross-study multi-paper comparison engine
│   ├── qa_system.py           # Evidence-grounded Q&A, absence & conflict engine
│   └── pubchem_service.py     # PubChem REST API 2D structure integration
│
├── sample_papers/             # Benchmark chemistry evaluation literature
│   ├── paper1_kinase_inhibitors.txt & .pdf
│   ├── paper2_catalytic_synthesis_conflicts.txt & .pdf
│   └── paper3_natural_product_sar.txt & .pdf
│
├── evaluation/                # Quantitative Research-Grade Evaluation Framework
│   ├── __init__.py            # Package exports
│   ├── evaluator.py           # Unified multi-axial evaluation engine
│   ├── benchmark_loader.py    # Benchmark case schema loader & validator
│   ├── benchmark_runner.py    # CLI and programmatic benchmark execution runner
│   ├── retrieval_metrics.py   # Precision@K, Recall@K, Hit Rate@K (K=1,3,5,10), MRR
│   ├── answer_metrics.py      # Exact Match, Token F1, Normalized Accuracy
│   ├── chemistry_metrics.py   # Entity, Target, Metric-Type, Numeric & Unit Accuracy
│   ├── citation_metrics.py    # Citation Validity, Accuracy, Completeness, Faithfulness
│   ├── reliability_metrics.py # Hallucination-Free, Unsupported Claim, Absence & Conflict F1
│   ├── ground_truth/          # Benchmark ground-truth datasets
│   │   ├── single_paper_benchmark.json # Real sample paper benchmark questions
│   │   └── README.md          # Ground-truth contribution guide
│   └── results/               # Empirical evaluation outputs (JSON & CSV)
│       ├── evaluation_summary.json
│       ├── per_question_results.csv
│       └── metrics.csv
│
└── tests/                     # Automated test suite (57 unit tests across 7 test modules)
    ├── test_parser.py         # PDF parsing and metadata tests
    ├── test_evidence.py       # BM25 retrieval and quote verification tests
    ├── test_qa.py             # Q&A, absence detection, and conflict detection tests
    ├── test_advanced_features.py # Tables, reactions, SAR, graph, gaps, and multi-paper tests
    ├── test_chem_calc.py      # Chemical calculator tests
    ├── test_pdf_viewer_and_split_view.py # High-res PDF viewer, quote highlighting & split view tests
    └── test_evaluation_metrics.py # Evaluation framework metrics unit tests
```

---

## 📊 Quantitative Evaluation Framework

The project includes an empirical, research-grade evaluation framework designed to rigorously benchmark RAG retrieval, question-answering accuracy, evidence grounding, and chemical factual consistency without relying on hard-coded or fabricated metrics.

### 1. Evaluated Dimensions & Metrics

| Dimension | Metric | Formula / Definition |
|---|---|---|
| **Retrieval Quality** | **Precision@K** | $\frac{\text{Relevant Chunks in Top K}}{K}$ (evaluated at $K \in \{1, 3, 5, 10\}$) |
| | **Recall@K** | $\frac{\text{Relevant Chunks in Top K}}{\text{Total Relevant Chunks}}$ |
| | **Hit Rate@K** | $1$ if at least one relevant chunk is retrieved in top $K$, else $0$ |
| | **MRR** | Mean Reciprocal Rank: $\frac{1}{\text{rank of first relevant chunk}}$ |
| **Answer Correctness** | **Exact Match (EM)** | $1.0$ if prediction exactly equals ground-truth answer after normalization |
| | **Token F1** | Harmonic mean of token-level precision and recall |
| | **Normalized Accuracy** | Factual string match with chemical Unicode subscript (`IC₅₀` $\rightarrow$ `IC50`), micro symbol (`µM` $\rightarrow$ `uM`), and whitespace normalization |
| **Chemistry Precision** | **Entity Accuracy** | Evaluates compound identity match (strictly distinguishes e.g. `5a` vs `5d`) |
| | **Target Accuracy** | Evaluates biological macromolecule target alignment (e.g. `EGFR` vs `VEGFR2`) |
| | **Metric-Type Accuracy** | Enforces assay type distinction: $\text{IC}_{50} \neq \text{GI}_{50} \neq \text{EC}_{50} \neq K_i \neq K_d \neq \text{MIC}$ |
| | **Numeric Accuracy** | Numerical equivalence accounting for unit conversions ($87\text{ nM} \equiv 0.087\ \mu\text{M}$) |
| | **Unit Accuracy** | Unit scale match or valid convertible dimension |
| **Citation & Grounding** | **Citation Validity** | Fraction of citations whose quoted excerpts verifiably exist in the source text |
| | **Citation Accuracy** | Verification that cited excerpt directly corroborates the factual claim |
| | **Citation Completeness**| Proportion of extracted factual assertions accompanied by citations |
| | **Faithfulness** | $\frac{\text{Supported Claims}}{\text{Total Extracted Claims}}$ |
| **Reliability & Hallucination** | **Unsupported Claim Rate** | $\frac{\text{Unsupported Claims}}{\text{Total Claims}}$ |
| | **Hallucination-Free Rate**| $1.0 - \text{Unsupported Claim Rate}$ |
| | **Absence F1** | Precision, Recall, and F1 for unmentioned or negative questions |
| | **Conflict F1** | Precision, Recall, and F1 for contradictory/discrepant literature findings |
| | **Confusion Matrix** | $3 \times 3$ classification matrix for `found`, `absent`, and `conflicting` |
| **Latency & Performance**| **Latency Metrics** | Mean, Median, Min, Max, and P95 latency (ms) for retrieval and generation |
| **Composite Score** | **Composite Benchmark** | $0.25 \times \text{Chemistry} + 0.25 \times \text{Answer F1} + 0.20 \times \text{Hit@5} + 0.15 \times \text{Citation Acc} + 0.15 \times \text{Hallucination-Free}$ |

---

### 2. Benchmark Ground-Truth Schema

Benchmark datasets are defined as structured JSON files under `evaluation/ground_truth/`.

Each test case contains:
```json
{
  "id": "Q001",
  "category": "bioactivity",
  "question": "What is the IC50 value of Compound 3b against EGFR kinase?",
  "expected_answer": "8.4 ± 0.6 nM",
  "expected_status": "found",
  "compound": "Compound 3b",
  "target": "EGFR",
  "metric": "IC50",
  "value": 8.4,
  "unit": "nM",
  "expected_page": 3,
  "expected_section": "Biological Evaluation",
  "evidence_keywords": ["Compound 3b", "8.4", "EGFR", "nM"],
  "relevant_chunk_ids": []
}
```

Negative/Absence cases:
```json
{
  "id": "Q007",
  "category": "absence",
  "question": "What was the Phase III human clinical trial dose of Compound 3b?",
  "expected_answer": "No supporting evidence was found in the uploaded manuscript.",
  "expected_status": "absent"
}
```

Contradiction/Conflict cases:
```json
{
  "id": "Q009",
  "category": "conflict",
  "question": "What was the isolated yield of compound 4b?",
  "expected_answer": "Discrepancy reported: 45% in Scheme 2 vs 88% in Table 2.",
  "expected_status": "conflicting"
}
```

---

### 3. How to Run an Evaluation Benchmark

#### Via CLI:
```bash
# Run against a paper with local deterministic offline engine
python3 -m evaluation.benchmark_runner \
    --paper sample_papers/paper1_kinase_inhibitors.txt \
    --benchmark evaluation/ground_truth/single_paper_benchmark.json

# Run with Gemini LLM reasoning (requires GEMINI_API_KEY)
python3 -m evaluation.benchmark_runner \
    --paper sample_papers/paper1_kinase_inhibitors.txt \
    --benchmark evaluation/ground_truth/single_paper_benchmark.json \
    --use-gemini
```

#### Via Streamlit Web Interface:
1. Launch the application: `streamlit run app.py`
2. Open the **📊 Evaluation** tab in the Studio ribbon (or select "📊 Evaluation" in the Side-by-Side split view).
3. Inspect live metric cards, top-K retrieval charts, chemistry accuracy breakdowns, reliability gauges, status confusion matrix, and per-question diagnostics.
4. Click **🚀 Run Benchmark Evaluation** to execute a benchmark run directly on any uploaded or sample paper.

---

### 4. Evaluation Output Files

Every benchmark run exports persistent empirical results into `evaluation/results/`:
- **`evaluation_summary.json`**: Complete metrics, confusion matrix, and composite scores.
- **`per_question_results.csv`**: Granular question-level predictions, latencies, and validation flags.
- **`metrics.csv`**: Flattened metric key-value table for plotting and automated CI regression tracking.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Gemini API Key
To enable Gemini 3.8 Flash semantic reasoning:
```bash
cp .env.example .env
# Set GEMINI_API_KEY=your_key_here
```
*(Alternatively, enter your API key directly in the web UI sidebar at runtime. If omitted, the app runs 100% offline using the built-in local deterministic engine).*

### 4. Run the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run the End-to-End CLI Demo
```bash
python3 demo_cli.py
```

### 6. Run the Automated Test Suite (57 Tests)
```bash
python3 -m unittest discover tests/ -v
```

