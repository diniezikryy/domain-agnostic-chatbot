#!/usr/bin/env python3
"""
Test different combinations of RAG pipeline parameters to find optimal configuration.

This script systematically tests combinations of:
- Experiment types (baseline, reranking, grounded_hyde, combined_best)
- Candidate pool sizes (retrieval breadth)
- Rerank keep-top-N (precision control)
- Generation top-K (context budget)

Goal: Find the best balance of performance and cost.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd


class CombinationTester:
    """Orchestrates testing of different parameter combinations."""
    
    def __init__(
        self,
        batch_id: str = "my_policies",
        dataset: str = "test_data/minimal_test_dataset.json",
        output_dir: str = "evaluation/results/combinations"
    ):
        self.batch_id = batch_id
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        
    def run_single_test(
        self,
        experiment: str,
        candidate_pool: int,
        rerank_keep_n: Optional[int],
        generation_top_k: int,
        skip_ragas: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Run a single test configuration and return results."""
        
        # Build environment variables for this test
        env = os.environ.copy()
        env["RETRIEVAL_CANDIDATE_POOL"] = str(candidate_pool)
        env["GENERATION_TOP_K"] = str(generation_top_k)
        
        if rerank_keep_n is not None:
            env["RERANK_KEEP_TOP_N"] = str(rerank_keep_n)
        
        # Build command
        cmd = [
            sys.executable,
            "run_evaluation.py",
            "--experiment", experiment,
            "--batch_id", self.batch_id,
            "--dataset", self.dataset,
            "--no_show_table",  # Suppress verbose output
        ]
        
        if skip_ragas:
            cmd.append("--skip_ragas")
        
        test_id = f"{experiment}_pool{candidate_pool}_gen{generation_top_k}"
        if rerank_keep_n is not None:
            test_id += f"_rerank{rerank_keep_n}"
        
        print(f"\n{'='*80}")
        print(f"Running Test: {test_id}")
        print(f"  Experiment: {experiment}")
        print(f"  Candidate Pool: {candidate_pool}")
        print(f"  Generation Top-K: {generation_top_k}")
        if rerank_keep_n is not None:
            print(f"  Rerank Keep-N: {rerank_keep_n}")
        print(f"{'='*80}\n")
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Find the most recent results file for this experiment
            results_pattern = f"ragas_{experiment}_{self.batch_id}_*.json"
            results_files = list(Path("evaluation/results").glob(results_pattern))
            
            if not results_files:
                print(f"[WARN] No results file found for {test_id}")
                return None
            
            latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
            
            # Load and extract metrics
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            ragas_summary = data.get("ragas_metrics", {}).get("summary", {})
            metadata = data.get("metadata", {})
            
            # Extract mean scores for key metrics
            metrics = {
                "test_id": test_id,
                "experiment": experiment,
                "candidate_pool": candidate_pool,
                "generation_top_k": generation_top_k,
                "rerank_keep_n": rerank_keep_n,
                "timestamp": metadata.get("timestamp", ""),
                "num_questions": metadata.get("num_questions", 0),
            }
            
            # Add RAGAS metrics
            for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]:
                metric_data = ragas_summary.get(metric_name, {})
                metrics[f"{metric_name}_mean"] = metric_data.get("mean", 0.0)
                metrics[f"{metric_name}_std"] = metric_data.get("std", 0.0)
            
            # Add latency info
            latency_summary = metadata.get("latency_summary", {})
            metrics["avg_latency_s"] = latency_summary.get("average_seconds", 0.0)
            metrics["total_latency_s"] = latency_summary.get("total_seconds", 0.0)
            
            print(f"[OK] Test {test_id} completed successfully")
            print(f"  Faithfulness: {metrics['faithfulness_mean']:.3f}")
            print(f"  Answer Relevancy: {metrics['answer_relevancy_mean']:.3f}")
            print(f"  Context Precision: {metrics['context_precision_mean']:.3f}")
            print(f"  Context Recall: {metrics['context_recall_mean']:.3f}")
            print(f"  Answer Correctness: {metrics['answer_correctness_mean']:.3f}")
            
            return metrics
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Test {test_id} failed with exit code {e.returncode}")
            if e.stderr:
                print(f"Error output: {e.stderr[:500]}")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error in test {test_id}: {e}")
            return None
    
    def run_baseline_tests(self, skip_ragas: bool = False) -> None:
        """Run baseline tests with different parameter settings."""
        print("\n" + "="*80)
        print("PHASE 1: BASELINE PARAMETER SWEEP")
        print("="*80)
        
        # Test different candidate pool sizes
        for pool_size in [20, 50, 100]:
            result = self.run_single_test(
                experiment="baseline",
                candidate_pool=pool_size,
                rerank_keep_n=None,
                generation_top_k=8,  # Fixed for baseline
                skip_ragas=skip_ragas
            )
            if result:
                self.results.append(result)
        
        # Test different generation top-k values with optimal pool
        for gen_k in [3, 5, 8, 10]:
            result = self.run_single_test(
                experiment="baseline",
                candidate_pool=50,  # Optimal from previous tests
                rerank_keep_n=None,
                generation_top_k=gen_k,
                skip_ragas=skip_ragas
            )
            if result:
                self.results.append(result)
    
    def run_reranking_tests(self, skip_ragas: bool = False) -> None:
        """Run reranking tests with different keep-N values."""
        print("\n" + "="*80)
        print("PHASE 2: RERANKING OPTIMIZATION")
        print("="*80)
        
        # Test different rerank keep-N values
        for rerank_n in [3, 5, 8, 10]:
            result = self.run_single_test(
                experiment="reranking",
                candidate_pool=50,
                rerank_keep_n=rerank_n,
                generation_top_k=rerank_n,  # Match to ensure consistency
                skip_ragas=skip_ragas
            )
            if result:
                self.results.append(result)
    
    def run_advanced_experiments(self, skip_ragas: bool = False) -> None:
        """Run advanced experiments (HyDE, combined_best)."""
        print("\n" + "="*80)
        print("PHASE 3: ADVANCED EXPERIMENTS")
        print("="*80)
        
        configs = [
            ("grounded_hyde", 50, None, 8),
            ("combined_best", 50, 5, 5),
            ("combined_best", 50, 8, 8),
        ]
        
        for exp, pool, rerank_n, gen_k in configs:
            result = self.run_single_test(
                experiment=exp,
                candidate_pool=pool,
                rerank_keep_n=rerank_n,
                generation_top_k=gen_k,
                skip_ragas=skip_ragas
            )
            if result:
                self.results.append(result)
    
    def run_semantic_chunking_test(self, skip_ragas: bool = False) -> None:
        """Test semantic chunking experiment."""
        print("\n" + "="*80)
        print("PHASE 4: SEMANTIC CHUNKING")
        print("="*80)
        
        result = self.run_single_test(
            experiment="semantic_chunking",
            candidate_pool=50,
            rerank_keep_n=None,
            generation_top_k=8,
            skip_ragas=skip_ragas
        )
        if result:
            self.results.append(result)
    
    def generate_comparison_report(self) -> str:
        """Generate a comparison report of all tested configurations."""
        if not self.results:
            return "No results to compare."
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(self.results)
        
        # Sort by a composite score (you can adjust weights)
        df["composite_score"] = (
            0.25 * df["faithfulness_mean"] +
            0.20 * df["answer_relevancy_mean"] +
            0.20 * df["context_precision_mean"] +
            0.20 * df["context_recall_mean"] +
            0.15 * df["answer_correctness_mean"]
        )
        
        df = df.sort_values("composite_score", ascending=False)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.output_dir / f"combination_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        json_file = self.output_dir / f"combination_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary report
        report = []
        report.append("\n" + "="*80)
        report.append("COMBINATION TEST RESULTS SUMMARY")
        report.append("="*80 + "\n")
        
        report.append(f"Total configurations tested: {len(df)}")
        report.append(f"Dataset: {self.dataset}")
        report.append(f"Batch: {self.batch_id}\n")
        
        report.append("TOP 5 CONFIGURATIONS (by composite score):")
        report.append("-" * 80)
        
        top_5 = df.head(5)
        for idx, row in top_5.iterrows():
            report.append(f"\n{row['test_id']}")
            report.append(f"  Composite Score: {row['composite_score']:.3f}")
            report.append(f"  Faithfulness: {row['faithfulness_mean']:.3f} (±{row['faithfulness_std']:.3f})")
            report.append(f"  Answer Relevancy: {row['answer_relevancy_mean']:.3f} (±{row['answer_relevancy_std']:.3f})")
            report.append(f"  Context Precision: {row['context_precision_mean']:.3f} (±{row['context_precision_std']:.3f})")
            report.append(f"  Context Recall: {row['context_recall_mean']:.3f} (±{row['context_recall_std']:.3f})")
            report.append(f"  Answer Correctness: {row['answer_correctness_mean']:.3f} (±{row['answer_correctness_std']:.3f})")
            report.append(f"  Avg Latency: {row['avg_latency_s']:.2f}s")
        
        report.append("\n" + "="*80)
        report.append("BEST CONFIGURATION PER METRIC:")
        report.append("="*80)
        
        for metric in ["faithfulness_mean", "answer_relevancy_mean", "context_precision_mean", 
                      "context_recall_mean", "answer_correctness_mean"]:
            best_idx = df[metric].idxmax()
            best_row = df.loc[best_idx]
            report.append(f"\n{metric.replace('_mean', '').replace('_', ' ').title()}:")
            report.append(f"  {best_row['test_id']}: {best_row[metric]:.3f}")
        
        report.append("\n" + "="*80)
        report.append(f"Detailed results saved to:")
        report.append(f"  CSV: {csv_file}")
        report.append(f"  JSON: {json_file}")
        report.append("="*80 + "\n")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / f"combination_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Test different combinations of RAG pipeline parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--batch_id",
        default="my_policies",
        help="Batch ID to test against (default: my_policies)"
    )
    
    parser.add_argument(
        "--dataset",
        default="test_data/minimal_test_dataset.json",
        help="Test dataset to use (default: minimal_test_dataset.json)"
    )
    
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["baseline", "reranking", "advanced", "semantic", "all"],
        default=["all"],
        help="Which test phases to run (default: all)"
    )
    
    parser.add_argument(
        "--skip_ragas",
        action="store_true",
        help="Skip RAGAS evaluation (faster, for pipeline testing only)"
    )
    
    args = parser.parse_args()
    
    tester = CombinationTester(
        batch_id=args.batch_id,
        dataset=args.dataset
    )
    
    phases = args.phases
    if "all" in phases:
        phases = ["baseline", "reranking", "advanced", "semantic"]
    
    print("="*80)
    print("RAG PIPELINE COMBINATION TESTING")
    print("="*80)
    print(f"Batch: {args.batch_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Phases: {', '.join(phases)}")
    print(f"Skip RAGAS: {args.skip_ragas}")
    print("="*80)
    
    # Run selected test phases
    if "baseline" in phases:
        tester.run_baseline_tests(skip_ragas=args.skip_ragas)
    
    if "reranking" in phases:
        tester.run_reranking_tests(skip_ragas=args.skip_ragas)
    
    if "advanced" in phases:
        tester.run_advanced_experiments(skip_ragas=args.skip_ragas)
    
    if "semantic" in phases:
        tester.run_semantic_chunking_test(skip_ragas=args.skip_ragas)
    
    # Generate comparison report
    tester.generate_comparison_report()


if __name__ == "__main__":
    main()
