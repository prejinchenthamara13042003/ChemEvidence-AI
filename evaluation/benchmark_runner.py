"""
CLI and Programmatic Benchmark Runner for RAG Evaluation.
Executes benchmark questions against loaded manuscripts, measures latency,
computes research-grade evaluation metrics, and persists results to JSON and CSV.
"""

import os
import sys
import csv
import json
import time
import argparse
from typing import List, Dict, Any, Optional

from core.pdf_parser import DocumentParser, PageContent
from core.evidence_engine import EvidenceEngine
from core.chemistry_extractor import ChemistryExtractor
from core.qa_system import QASystem
from core.models import QAResponse

from evaluation.benchmark_loader import load_benchmark, BenchmarkCase
from evaluation.evaluator import Evaluator, EvaluationResult, QuestionEvaluationResult


class BenchmarkRunner:
    """Executes benchmark evaluations against papers and exports results."""

    def __init__(
        self,
        paper_path: str,
        benchmark_path: str,
        output_dir: str = "evaluation/results",
        use_local_only: bool = False,
        use_gemini: Optional[bool] = None,
        api_key: Optional[str] = None
    ):
        self.paper_path = paper_path
        self.benchmark_path = benchmark_path
        self.output_dir = output_dir
        if use_gemini is not None:
            self.use_local_only = not use_gemini
        else:
            self.use_local_only = use_local_only
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        os.makedirs(self.output_dir, exist_ok=True)

    def load_document(self) -> List[PageContent]:
        """Loads and parses the target research paper (PDF or TXT)."""
        if not os.path.exists(self.paper_path):
            raise FileNotFoundError(f"Manuscript not found at: {self.paper_path}")

        if self.paper_path.lower().endswith(".txt"):
            with open(self.paper_path, "r", encoding="utf-8") as f:
                content = f.read()
            return DocumentParser.parse_text_content(content)
        else:
            return DocumentParser.parse_pdf(self.paper_path)

    def run(self) -> EvaluationResult:
        """
        Executes all benchmark questions against the document and evaluates results.
        """
        print(f"\n🔬 Starting ChemEvidence AI Benchmark Evaluation...")
        print(f"📄 Paper: {self.paper_path}")
        print(f"📋 Benchmark: {self.benchmark_path}")
        print(f"⚙️ Mode: {'Local Engine Only' if self.use_local_only else 'Gemini 3.8 Flash (with Local Fallback)'}")

        # 1. Load document and benchmark
        pages = self.load_document()
        meta = DocumentParser.extract_metadata(pages)
        engine = EvidenceEngine(pages)
        benchmark_cases = load_benchmark(self.benchmark_path)

        # 2. Extract structured chemistry context
        print(f"🧪 Analyzing document structure and chemical entities...")
        extractor = ChemistryExtractor(api_key=self.api_key, use_local_only=self.use_local_only)
        analysis_result = extractor.analyze_paper(pages, meta, engine)

        # 3. Initialize QA System
        qa_system = QASystem(api_key=self.api_key, use_local_only=self.use_local_only)
        evaluator = Evaluator()

        q_results: List[QuestionEvaluationResult] = []

        print(f"\n🚀 Running {len(benchmark_cases)} benchmark inquiries...\n" + "-" * 70)

        for idx, case in enumerate(benchmark_cases, 1):
            t_start = time.perf_counter()

            # Measure retrieval latency
            t_ret_start = time.perf_counter()
            retrieved = engine.search(case.question, top_k=5, min_score=0.10)
            ret_chunks = [c for c, _ in retrieved]
            ret_latency_ms = (time.perf_counter() - t_ret_start) * 1000.0

            # Measure answering latency
            t_gen_start = time.perf_counter()
            engine_used = "local" if self.use_local_only or not qa_system.client else "gemini"
            
            try:
                qa_resp: QAResponse = qa_system.ask(
                    question=case.question,
                    pages=pages,
                    evidence_engine=engine,
                    analysis_result=analysis_result
                )
            except Exception as e:
                print(f"⚠️ Error during QA generation ({e}), utilizing local fallback.")
                engine_used = "local"
                qa_resp = qa_system._synthesize_locally(
                    question=case.question,
                    search_results=retrieved,
                    evidence_engine=engine,
                    conflicts=[],
                    analysis_result=analysis_result
                )

            gen_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0

            # Evaluate against ground truth
            q_res = evaluator.evaluate_question(
                case=case,
                qa_response=qa_resp,
                retrieved_chunks=ret_chunks,
                pages=pages,
                retrieval_latency_ms=ret_latency_ms,
                generation_latency_ms=gen_latency_ms,
                engine_used=engine_used
            )
            q_results.append(q_res)

            status_icon = "🟢" if q_res.is_status_correct else "🔴"
            chem_icon = "🧪✓" if q_res.overall_chemistry_correct else "🧪✗"
            print(f"[{idx:02d}/{len(benchmark_cases):02d}] {status_icon} ID: {case.id} | Cat: {case.category:<12} | {chem_icon} | Latency: {q_res.total_latency_ms:.1f}ms")

        # 4. Aggregate results
        eval_result = evaluator.aggregate_results(q_results)

        # 5. Persist Results
        self.save_results(eval_result)

        print("\n" + "=" * 70)
        print(f"🎯 BENCHMARK COMPLETE — Composite Score: {eval_result.composite_benchmark_score:.1f}%")
        print(f"📊 Overall Chemistry Accuracy: {eval_result.chemistry_metrics['overall_chemistry_accuracy'] * 100:.1f}%")
        print(f"🔎 Retrieval Hit@5: {eval_result.retrieval_metrics.get('hit_rate@5', 0) * 100:.1f}%")
        print(f"📝 Answer Token F1: {eval_result.answer_metrics['token_f1'] * 100:.1f}%")
        print(f"🛡️ Hallucination-Free Rate: {eval_result.reliability_metrics['hallucination_free_rate'] * 100:.1f}%")
        print(f"⏱️ Mean Latency: {eval_result.latency_metrics['mean_total_latency_ms']:.1f} ms")
        print(f"💾 Files saved to: {self.output_dir}/")
        print("=" * 70 + "\n")

        return eval_result

    def save_results(self, eval_result: EvaluationResult):
        """Saves evaluation results to JSON and CSV formats."""
        summary_path = os.path.join(self.output_dir, "evaluation_summary.json")
        per_q_path = os.path.join(self.output_dir, "per_question_results.csv")
        metrics_path = os.path.join(self.output_dir, "metrics.csv")

        # 1. Summary JSON
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(eval_result.model_dump(), f, indent=2, ensure_ascii=False)

        # 2. Per-Question CSV
        fieldnames = [
            "question_id", "category", "question", "expected_answer", "predicted_answer",
            "expected_status", "predicted_status", "is_status_correct",
            "exact_match", "token_f1", "normalized_accuracy",
            "entity_correct", "target_correct", "metric_type_correct", "numeric_correct", "unit_correct", "overall_chemistry_correct",
            "citation_validity", "citation_accuracy", "citation_completeness", "faithfulness",
            "unsupported_claim_rate", "hallucination_free_rate",
            "retrieval_latency_ms", "generation_latency_ms", "total_latency_ms",
            "engine_used", "notes"
        ]
        with open(per_q_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for q in eval_result.per_question_results:
                row = q.model_dump()
                # filter down to fieldnames
                filtered_row = {k: row[k] for k in fieldnames if k in row}
                writer.writerow(filtered_row)

        # 3. Metrics Summary CSV
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric Category", "Metric Name", "Value"])
            
            for k, v in eval_result.retrieval_metrics.items():
                writer.writerow(["Retrieval", k, v])
            for k, v in eval_result.answer_metrics.items():
                writer.writerow(["Answer Quality", k, v])
            for k, v in eval_result.chemistry_metrics.items():
                writer.writerow(["Chemistry Factual", k, v])
            for k, v in eval_result.citation_metrics.items():
                writer.writerow(["Citations", k, v])
            for k, v in eval_result.reliability_metrics.items():
                writer.writerow(["Reliability & Detection", k, v])
            for k, v in eval_result.latency_metrics.items():
                writer.writerow(["Latency", k, v])
            writer.writerow(["Overall", "composite_benchmark_score", eval_result.composite_benchmark_score])


def main():
    parser = argparse.ArgumentParser(description="ChemEvidence AI Benchmark Evaluation Runner")
    parser.add_argument("--paper", "-p", required=True, help="Path to research paper (.pdf or .txt)")
    parser.add_argument("--benchmark", "-b", default="evaluation/ground_truth/single_paper_benchmark.json", help="Path to benchmark JSON file")
    parser.add_argument("--output-dir", "-o", default="evaluation/results", help="Directory where evaluation outputs will be stored")
    parser.add_argument("--use-local-only", action="store_true", help="Force local deterministic offline engine")
    parser.add_argument("--api-key", default=None, help="Optional Gemini API key")

    args = parser.parse_args()

    runner = BenchmarkRunner(
        paper_path=args.paper,
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        use_local_only=args.use_local_only,
        api_key=args.api_key
    )
    runner.run()


if __name__ == "__main__":
    main()
