#!/usr/bin/env python3
"""
Quick combination tester - tests only the most promising configurations.

This is a streamlined version for rapid iteration on key parameter combinations.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd


def run_test(experiment, candidate_pool, rerank_keep_n, generation_top_k, batch_id, dataset, skip_ragas=False):
    """Run a single test and return the results file path."""
    
    env = os.environ.copy()
    env["RETRIEVAL_CANDIDATE_POOL"] = str(candidate_pool)
    env["GENERATION_TOP_K"] = str(generation_top_k)
    
    if rerank_keep_n is not None:
        env["RERANK_KEEP_TOP_N"] = str(rerank_keep_n)
    
    cmd = [
        sys.executable,
        "run_evaluation.py",
        "--experiment", experiment,
        "--batch_id", batch_id,
        "--dataset", dataset,
        "--no_show_table",
    ]
    
    if skip_ragas:
        cmd.append("--skip_ragas")
    
    test_id = f"{experiment}_p{candidate_pool}_g{generation_top_k}"
    if rerank_keep_n:
        test_id += f"_r{rerank_keep_n}"
    
    print(f"\n{'='*60}")
    print(f"Test: {test_id}")
    print(f"{'='*60}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
        
        # Find results file
        pattern = f"ragas_{experiment}_{batch_id}_*.json"
        results_files = list(Path("evaluation/results").glob(pattern))
        if results_files:
            return max(results_files, key=lambda p: p.stat().st_mtime)
    except Exception as e:
        print(f"ERROR: {e}")
    
    return None


def extract_metrics(results_file):
    """Extract key metrics from a results file."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    ragas = data.get("ragas_metrics", {}).get("summary", {})
    meta = data.get("metadata", {})
    
    return {
        "experiment": meta.get("experiment", ""),
        "faithfulness": ragas.get("faithfulness", {}).get("mean", 0),
        "answer_relevancy": ragas.get("answer_relevancy", {}).get("mean", 0),
        "context_precision": ragas.get("context_precision", {}).get("mean", 0),
        "context_recall": ragas.get("context_recall", {}).get("mean", 0),
        "answer_correctness": ragas.get("answer_correctness", {}).get("mean", 0),
        "pool": meta.get("retrieval_candidate_pool", ""),
        "gen_k": meta.get("generation_top_k", ""),
        "rerank_n": meta.get("rerank_keep_top_n", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Quick combination tester")
    parser.add_argument("--batch_id", default="my_policies")
    parser.add_argument("--dataset", default="test_data/minimal_test_dataset.json")
    parser.add_argument("--skip_ragas", action="store_true")
    args = parser.parse_args()
    
    print("="*60)
    print("QUICK COMBINATION TEST")
    print("="*60)
    
    # Define the most promising configurations to test
    configs = [
        # (experiment, candidate_pool, rerank_keep_n, generation_top_k)
        ("baseline", 50, None, 8),           # Baseline reference
        ("reranking", 50, 5, 5),            # Reranking with moderate K
        ("reranking", 50, 8, 8),            # Reranking with higher K
        ("grounded_hyde", 50, None, 8),     # HyDE alone
        ("combined_best", 50, 5, 5),        # HyDE + Reranking (conservative)
        ("combined_best", 50, 8, 8),        # HyDE + Reranking (aggressive)
    ]
    
    results = []
    
    for exp, pool, rerank, gen_k in configs:
        result_file = run_test(exp, pool, rerank, gen_k, args.batch_id, args.dataset, args.skip_ragas)
        if result_file:
            metrics = extract_metrics(result_file)
            results.append(metrics)
    
    if not results:
        print("\nNo results collected!")
        return
    
    # Create comparison table
    df = pd.DataFrame(results)
    
    # Add composite score
    df["composite"] = (
        0.25 * df["faithfulness"] +
        0.20 * df["answer_relevancy"] +
        0.20 * df["context_precision"] +
        0.20 * df["context_recall"] +
        0.15 * df["answer_correctness"]
    )
    
    df = df.sort_values("composite", ascending=False)
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    # Display table
    display_cols = ["experiment", "pool", "gen_k", "rerank_n", 
                   "faithfulness", "answer_relevancy", "context_precision", 
                   "context_recall", "answer_correctness", "composite"]
    
    print(df[display_cols].to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("evaluation/results/combinations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / f"quick_test_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\n[OK] Results saved to: {csv_file}")
    
    # Show winner
    winner = df.iloc[0]
    print(f"\n{'='*60}")
    print("BEST CONFIGURATION:")
    print(f"{'='*60}")
    print(f"  Experiment: {winner['experiment']}")
    print(f"  Pool: {winner['pool']}, Gen K: {winner['gen_k']}, Rerank N: {winner['rerank_n']}")
    print(f"  Composite Score: {winner['composite']:.3f}")
    print(f"  Faithfulness: {winner['faithfulness']:.3f}")
    print(f"  Answer Relevancy: {winner['answer_relevancy']:.3f}")
    print(f"  Context Precision: {winner['context_precision']:.3f}")
    print(f"  Context Recall: {winner['context_recall']:.3f}")
    print(f"  Answer Correctness: {winner['answer_correctness']:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
