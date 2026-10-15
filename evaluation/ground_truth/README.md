# ChemEvidence AI Benchmark Ground Truth Guide

This directory contains curated, verifiable ground-truth datasets used to evaluate the RAG chatbot and chemical extraction pipeline.

## Benchmark Schema

Each test case in the JSON benchmark file follows this schema:

```json
{
    "id": "Q001",
    "category": "bioactivity | synthesis | property | author | summary | absence | conflict",
    "question": "The scientific question to evaluate",
    "expected_answer": "Verifiable reference answer string",
    "expected_status": "found | absent | conflicting",
    "compound": "Optional compound identifier (e.g. '3b', '5a')",
    "target": "Optional biological target or cell line (e.g. 'EGFR', 'A549')",
    "metric": "Optional metric type ('IC50', 'GI50', 'EC50', 'Ki', 'yield', etc.)",
    "value": 8.4,
    "unit": "nM | µM | % | mol % | etc.",
    "expected_page": 3,
    "expected_section": "Name of manuscript section containing the evidence",
    "evidence_keywords": ["3b", "EGFR", "8.4", "nM"],
    "relevant_chunk_ids": []
}
```

## Supported Categories

1. **`bioactivity`**: Questions testing potency, inhibition, binding affinity, or cell cytotoxicity (`IC50`, `Ki`, `GI50`, `MIC`).
2. **`synthesis`**: Questions testing reaction conditions, catalysts, solvents, temperatures, and isolated yields.
3. **`property`**: Questions evaluating molecular formula, molecular weight, melting point, or characterization data.
4. **`author` / `summary`**: Executive document overview and researcher attribution inquiries.
5. **`absence`**: Negative control questions testing strict anti-hallucination. The expected status is `"absent"`.
6. **`conflict`**: Contradictory claims within the paper (e.g., Abstract vs. Table discrepancies). The expected status is `"conflicting"`.

## How to Add New Test Cases

1. Open `single_paper_benchmark.json` or create a new benchmark file (e.g., `my_paper_benchmark.json`).
2. Add a new JSON object adhering to the schema above.
3. Ensure numeric values and units match verbatim manuscript text.
4. If testing absence, set `"expected_status": "absent"`.
5. Run the evaluation CLI:
   ```bash
   python -m evaluation.benchmark_runner --paper <path_to_paper> --benchmark evaluation/ground_truth/single_paper_benchmark.json
   ```
