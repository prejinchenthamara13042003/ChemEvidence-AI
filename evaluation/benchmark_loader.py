"""
Benchmark Dataset Loader and Validator.
Loads benchmark test cases from JSON, validates schema fields,
and structures evaluation cases for the RAG evaluation framework.
"""

import json
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    """A structured benchmark question with ground-truth scientific evidence."""
    id: str = Field(description="Unique question identifier, e.g. Q001")
    category: str = Field(default="general", description="Category: bioactivity, synthesis, property, author, absence, conflict")
    question: str = Field(description="Scientific inquiry evaluated against the paper")
    expected_answer: str = Field(description="Expected ground-truth answer text")
    expected_status: str = Field(default="found", description="Expected status: found, absent, conflicting")
    compound: Optional[str] = Field(default=None, description="Target compound identifier, e.g. 3b, 5a")
    target: Optional[str] = Field(default=None, description="Biological target or cell line, e.g. EGFR, A549")
    metric: Optional[str] = Field(default=None, description="Chemical metric type, e.g. IC50, GI50, Ki, yield")
    value: Optional[float] = Field(default=None, description="Numeric ground-truth value")
    unit: Optional[str] = Field(default=None, description="Measurement unit, e.g. nM, µM, %")
    expected_page: Optional[int] = Field(default=None, description="Expected page number containing the evidence")
    expected_section: Optional[str] = Field(default=None, description="Expected manuscript section containing the evidence")
    evidence_keywords: List[str] = Field(default_factory=list, description="Key verbatim tokens required for evidence grounding")
    relevant_chunk_ids: List[str] = Field(default_factory=list, description="IDs of ground-truth matching chunks if available")


def load_benchmark(filepath: str) -> List[BenchmarkCase]:
    """
    Loads and validates benchmark test cases from a JSON file.
    
    Args:
        filepath: Absolute or relative path to the benchmark JSON file.
        
    Returns:
        List of validated BenchmarkCase instances.
        
    Raises:
        FileNotFoundError: If the benchmark JSON file cannot be found.
        ValueError: If JSON syntax is invalid or schema validation fails.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Benchmark file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in benchmark file {filepath}: {e}")

    if not isinstance(raw_data, list):
        raise ValueError(f"Benchmark file {filepath} must contain a JSON array of test cases.")

    cases: List[BenchmarkCase] = []
    for idx, item in enumerate(raw_data):
        try:
            case = BenchmarkCase(**item)
            cases.append(case)
        except Exception as e:
            raise ValueError(f"Validation failed for benchmark item index {idx} (ID: {item.get('id', 'unknown')}): {e}")

    return cases
