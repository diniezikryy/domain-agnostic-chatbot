#!/usr/bin/env python3
"""
Analyze and visualize combination testing results.

This script loads result CSV files and generates insights.
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def load_latest_results(results_dir="evaluation/results/combinations"):
    """Load the most recent combination test results."""
    results_path = Path(results_dir)
    
    csv_files = list(results_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return None
    
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest}")
    
    return pd.read_csv(latest)


def analyze_results(df):
    """Analyze the results and generate insights."""
    
    print("\n" + "="*80)
    print("COMBINATION TESTING ANALYSIS")
    print("="*80)
    
    # Calculate composite score if not present
    if "composite_score" not in df.columns and "composite" not in df.columns:
        df["composite"] = (
            0.25 * df.get("faithfulness_mean", df.get("faithfulness", 0)) +
            0.20 * df.get("answer_relevancy_mean", df.get("answer_relevancy", 0)) +
            0.20 * df.get("context_precision_mean", df.get("context_precision", 0)) +
            0.20 * df.get("context_recall_mean", df.get("context_recall", 0)) +
            0.15 * df.get("answer_correctness_mean", df.get("answer_correctness", 0))
        )
    elif "composite_score" in df.columns:
        df["composite"] = df["composite_score"]
    
    # Sort by composite score
    df = df.sort_values("composite", ascending=False)
    
    print(f"\nTotal configurations tested: {len(df)}")
    print(f"Best composite score: {df['composite'].max():.3f}")
    print(f"Worst composite score: {df['composite'].min():.3f}")
    print(f"Average composite score: {df['composite'].mean():.3f}")
    
    # Top 3 configurations
    print("\n" + "="*80)
    print("TOP 3 CONFIGURATIONS")
    print("="*80)
    
    for i, (idx, row) in enumerate(df.head(3).iterrows(), 1):
        exp = row.get("experiment", "unknown")
        test_id = row.get("test_id", f"{exp}_config{i}")
        
        print(f"\n#{i}: {test_id}")
        print(f"  Experiment: {exp}")
        
        if "candidate_pool" in row:
            print(f"  Candidate Pool: {row['candidate_pool']}")
        if "generation_top_k" in row:
            print(f"  Generation Top-K: {row['generation_top_k']}")
        if "rerank_keep_n" in row and pd.notna(row["rerank_keep_n"]):
            print(f"  Rerank Keep-N: {int(row['rerank_keep_n'])}")
        
        print(f"  Composite Score: {row['composite']:.3f}")
        
        # Show metrics
        for metric in ["faithfulness", "answer_relevancy", "context_precision", 
                      "context_recall", "answer_correctness"]:
            col = f"{metric}_mean" if f"{metric}_mean" in df.columns else metric
            if col in df.columns:
                print(f"    {metric.replace('_', ' ').title()}: {row[col]:.3f}")
        
        if "avg_latency_s" in row:
            print(f"  Avg Latency: {row['avg_latency_s']:.2f}s")
    
    # Analysis by experiment type
    if "experiment" in df.columns:
        print("\n" + "="*80)
        print("PERFORMANCE BY EXPERIMENT TYPE")
        print("="*80)
        
        exp_stats = df.groupby("experiment")["composite"].agg(["mean", "std", "max", "count"])
        exp_stats = exp_stats.sort_values("mean", ascending=False)
        
        print("\nExperiment Type Comparison:")
        print(exp_stats.to_string())
    
    # Parameter sensitivity analysis
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY")
    print("="*80)
    
    if "candidate_pool" in df.columns:
        pool_stats = df.groupby("candidate_pool")["composite"].agg(["mean", "std", "count"])
        print("\nCandidate Pool Impact:")
        print(pool_stats.to_string())
    
    if "generation_top_k" in df.columns:
        gen_k_stats = df.groupby("generation_top_k")["composite"].agg(["mean", "std", "count"])
        print("\nGeneration Top-K Impact:")
        print(gen_k_stats.to_string())
    
    if "rerank_keep_n" in df.columns:
        rerank_data = df[df["rerank_keep_n"].notna()]
        if len(rerank_data) > 0:
            rerank_stats = rerank_data.groupby("rerank_keep_n")["composite"].agg(["mean", "std", "count"])
            print("\nRerank Keep-N Impact:")
            print(rerank_stats.to_string())
    
    # Metric correlations
    print("\n" + "="*80)
    print("METRIC INSIGHTS")
    print("="*80)
    
    metric_cols = []
    for base in ["faithfulness", "answer_relevancy", "context_precision", 
                "context_recall", "answer_correctness"]:
        if f"{base}_mean" in df.columns:
            metric_cols.append(f"{base}_mean")
        elif base in df.columns:
            metric_cols.append(base)
    
    if metric_cols:
        print("\nMetric Averages Across All Configs:")
        for col in metric_cols:
            metric_name = col.replace("_mean", "").replace("_", " ").title()
            print(f"  {metric_name}: {df[col].mean():.3f} (±{df[col].std():.3f})")
    
    return df


def generate_recommendations(df):
    """Generate configuration recommendations based on results."""
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    best = df.iloc[0]
    
    print("\n🏆 OPTIMAL CONFIGURATION (Overall Best):")
    print(f"   Experiment: {best.get('experiment', 'N/A')}")
    if "candidate_pool" in best:
        print(f"   export RETRIEVAL_CANDIDATE_POOL={int(best['candidate_pool'])}")
    if "generation_top_k" in best:
        print(f"   export GENERATION_TOP_K={int(best['generation_top_k'])}")
    if "rerank_keep_n" in best and pd.notna(best["rerank_keep_n"]):
        print(f"   export RERANK_KEEP_TOP_N={int(best['rerank_keep_n'])}")
    print(f"   Expected Composite Score: {best['composite']:.3f}")
    
    # Find best by specific criteria
    if "experiment" in df.columns:
        baseline_configs = df[df["experiment"] == "baseline"]
        if len(baseline_configs) > 0:
            best_baseline = baseline_configs.iloc[0]
            print("\n💰 COST-EFFICIENT CONFIGURATION (Best Baseline):")
            print(f"   Experiment: baseline")
            if "candidate_pool" in best_baseline:
                print(f"   export RETRIEVAL_CANDIDATE_POOL={int(best_baseline['candidate_pool'])}")
            if "generation_top_k" in best_baseline:
                print(f"   export GENERATION_TOP_K={int(best_baseline['generation_top_k'])}")
            print(f"   Expected Composite Score: {best_baseline['composite']:.3f}")
        
        rerank_configs = df[df["experiment"] == "reranking"]
        if len(rerank_configs) > 0:
            best_rerank = rerank_configs.iloc[0]
            print("\n⚡ BALANCED CONFIGURATION (Best Reranking):")
            print(f"   Experiment: reranking")
            if "candidate_pool" in best_rerank:
                print(f"   export RETRIEVAL_CANDIDATE_POOL={int(best_rerank['candidate_pool'])}")
            if "generation_top_k" in best_rerank:
                print(f"   export GENERATION_TOP_K={int(best_rerank['generation_top_k'])}")
            if "rerank_keep_n" in best_rerank and pd.notna(best_rerank["rerank_keep_n"]):
                print(f"   export RERANK_KEEP_TOP_N={int(best_rerank['rerank_keep_n'])}")
            print(f"   Expected Composite Score: {best_rerank['composite']:.3f}")
    
    # Performance gains
    baseline_score = df[df["experiment"] == "baseline"]["composite"].max() if "experiment" in df.columns else None
    best_score = df["composite"].max()
    
    if baseline_score is not None and best_score > baseline_score:
        improvement = ((best_score - baseline_score) / baseline_score) * 100
        print(f"\n📈 Improvement over best baseline: +{improvement:.1f}%")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze combination testing results")
    parser.add_argument(
        "--results_dir",
        default="evaluation/results/combinations",
        help="Directory containing result CSV files"
    )
    parser.add_argument(
        "--export_config",
        action="store_true",
        help="Export winning configuration as shell script"
    )
    args = parser.parse_args()
    
    df = load_latest_results(args.results_dir)
    if df is None:
        return
    
    df = analyze_results(df)
    generate_recommendations(df)
    
    if args.export_config and len(df) > 0:
        best = df.iloc[0]
        config_script = []
        config_script.append("#!/bin/bash")
        config_script.append("# Optimal RAG pipeline configuration")
        config_script.append(f"# Generated from combination testing")
        config_script.append("")
        
        if "candidate_pool" in best:
            config_script.append(f"export RETRIEVAL_CANDIDATE_POOL={int(best['candidate_pool'])}")
        if "generation_top_k" in best:
            config_script.append(f"export GENERATION_TOP_K={int(best['generation_top_k'])}")
        if "rerank_keep_n" in best and pd.notna(best["rerank_keep_n"]):
            config_script.append(f"export RERANK_KEEP_TOP_N={int(best['rerank_keep_n'])}")
        
        config_script.append("")
        config_script.append(f"# To use: source this file before running your application")
        config_script.append(f"# Expected composite score: {best['composite']:.3f}")
        
        config_file = Path(args.results_dir) / "optimal_config.sh"
        with open(config_file, 'w') as f:
            f.write('\n'.join(config_script))
        
        print(f"\n[OK] Optimal configuration exported to: {config_file}")
        print(f"     Use: source {config_file}")


if __name__ == "__main__":
    main()
