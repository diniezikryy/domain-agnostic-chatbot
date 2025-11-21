"""
scripts/evaluate_ablation.py

Ablation study: Test individual contributions of RRF and Cross-Encoder reranking.
Runs 4 configurations:
1. Baseline (weighted sum, no Cross-Encoder)
2. RRF Only (RRF fusion, no Cross-Encoder)
3. Cross-Encoder Only (weighted sum, with Cross-Encoder)
4. Both (RRF fusion + Cross-Encoder)
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from scripts.evaluate_rag import evaluate_dataset


async def run_ablation_study(
    dataset_path: str,
    batch_id: str,
    output_dir: str = "evaluation/results/ablation",
    top_k: int = 10,
    evaluator_model: str = "gpt-4o-mini",
    profile_path: str = None,
):
    """
    Run ablation study with 4 configurations.
    
    Args:
        dataset_path: Path to golden dataset JSON
        batch_id: Batch ID to use for retrieval
        output_dir: Directory for ablation outputs
        top_k: Number of retrieved chunks
        evaluator_model: LLM model for Ragas evaluation
        profile_path: Optional user profile path
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ablation_dir = Path(output_dir) / timestamp
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"ABLATION STUDY: RRF vs Cross-Encoder Contributions")
    print(f"Timestamp: {timestamp}")
    print(f"Dataset: {dataset_path}")
    print(f"Batch: {batch_id}")
    print(f"Output: {ablation_dir}")
    print("=" * 80)
    
    # Define configurations to test
    configurations = [
        {
            "name": "baseline",
            "label": "Baseline (Weighted Sum, No Cross-Encoder)",
            "pipeline": "baseline",
            "use_rrf": False,
            "use_cross_encoder": False,
        },
        {
            "name": "rrf_only",
            "label": "RRF Only (RRF Fusion, No Cross-Encoder)",
            "pipeline": "optimized",
            "use_rrf": True,
            "use_cross_encoder": False,
        },
        {
            "name": "cross_encoder_only",
            "label": "Cross-Encoder Only (Weighted Sum, With Cross-Encoder)",
            "pipeline": "optimized",
            "use_rrf": False,
            "use_cross_encoder": True,
        },
        {
            "name": "both",
            "label": "Both (RRF Fusion + Cross-Encoder)",
            "pipeline": "optimized",
            "use_rrf": True,
            "use_cross_encoder": True,
        },
    ]
    
    results = {}
    
    # Run each configuration
    for i, config in enumerate(configurations, 1):
        print(f"\n{'=' * 80}")
        print(f"CONFIGURATION {i}/4: {config['label']}")
        print(f"{'=' * 80}")
        
        # Set environment variables for feature flags
        os.environ["USE_RRF_FUSION"] = "true" if config["use_rrf"] else "false"
        os.environ["USE_CROSS_ENCODER"] = "true" if config["use_cross_encoder"] else "false"
        
        # Run evaluation
        output_csv = ablation_dir / f"{config['name']}_results.csv"
        
        try:
            await evaluate_dataset(
                dataset_path=dataset_path,
                batch_id=batch_id,
                output_csv=str(output_csv),
                top_k=top_k,
                evaluator_model=evaluator_model,
                profile_path=profile_path,
                pipeline=config["pipeline"],
            )
            
            # Load results
            df = pd.read_csv(output_csv)
            # Only aggregate the metrics we expect to be numeric
            numeric_metrics = [
                "faithfulness",
                "context_precision",
                "context_recall",
                "answer_relevancy",
                "semantic_similarity",
                "context_utilization",
            ]
            metrics_summary = {}
            for m in numeric_metrics:
                if m in df.columns:
                    try:
                        metrics_summary[m] = float(df[m].mean())
                    except Exception:
                        metrics_summary[m] = None

            results[config["name"]] = {
                "config": config,
                "csv_path": str(output_csv),
                "metrics": metrics_summary,
            }
            
            print(f"✓ {config['name']} complete")
            
        except Exception as e:
            print(f"✗ {config['name']} failed: {e}")
            results[config["name"]] = {"config": config, "error": str(e)}
    
    # Generate comparison analysis
    print(f"\n{'=' * 80}")
    print("GENERATING ABLATION ANALYSIS")
    print(f"{'=' * 80}")
    
    # Create comparison DataFrame
    comparison_data = []
    
    for config_name, result in results.items():
        if "metrics" in result:
            row = {"configuration": config_name}
            row.update(result["metrics"])
            comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv = ablation_dir / "ablation_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"✓ Comparison saved to: {comparison_csv}")
        
        # Calculate deltas relative to baseline
        if "baseline" in results and "metrics" in results["baseline"]:
            baseline_metrics = results["baseline"]["metrics"]
            delta_data = []
            
            for config_name, result in results.items():
                if config_name != "baseline" and "metrics" in result:
                    row = {"configuration": config_name}
                    for metric, value in result["metrics"].items():
                        if metric in baseline_metrics:
                            baseline_value = baseline_metrics[metric]
                            delta = value - baseline_value
                            delta_pct = (delta / baseline_value * 100) if baseline_value != 0 else 0
                            row[f"{metric}_delta"] = delta
                            row[f"{metric}_improvement_pct"] = delta_pct
                    delta_data.append(row)
            
            if delta_data:
                delta_df = pd.DataFrame(delta_data)
                delta_csv = ablation_dir / "ablation_deltas.csv"
                delta_df.to_csv(delta_csv, index=False)
                print(f"✓ Deltas saved to: {delta_csv}")
    
    # Save summary JSON
    summary = {
        "timestamp": timestamp,
        "dataset": dataset_path,
        "batch": batch_id,
        "configurations": len(configurations),
        "results": results,
    }
    
    summary_json = ablation_dir / "ablation_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_json}")
    
    # Print summary table
    print(f"\n{'=' * 80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'=' * 80}\n")
    
    if comparison_data:
        # Show key metrics
        key_metrics = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
        
        print("Configuration Comparison (Key Metrics):")
        print("-" * 80)
        
        for metric in key_metrics:
            if metric in comparison_df.columns:
                print(f"\n{metric.upper()}:")
                for _, row in comparison_df.iterrows():
                    config_name = row["configuration"]
                    value = row[metric]
                    
                    # Calculate delta from baseline
                    if config_name != "baseline" and "baseline" in comparison_df["configuration"].values:
                        baseline_value = comparison_df[comparison_df["configuration"] == "baseline"][metric].values[0]
                        delta = value - baseline_value
                        delta_pct = (delta / baseline_value * 100) if baseline_value != 0 else 0
                        delta_str = f" ({delta_pct:+.1f}%)" if config_name != "baseline" else ""
                    else:
                        delta_str = ""
                    
                    print(f"  {config_name:25s}: {value:.4f}{delta_str}")
    
    print(f"\n{'=' * 80}")
    print(f"Ablation study complete! Results in: {ablation_dir}")
    print(f"{'=' * 80}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ablation study: Test RRF and Cross-Encoder contributions separately"
    )
    parser.add_argument("--dataset", required=True, help="Path to golden dataset JSON")
    parser.add_argument("--batch", required=True, help="Batch ID to evaluate")
    parser.add_argument("--output", default="evaluation/results/ablation", help="Output directory")
    parser.add_argument("--top-k", type=int, default=10, help="Number of retrieved chunks")
    parser.add_argument("--evaluator-model", default="gpt-4o-mini", help="LLM model for evaluation")
    parser.add_argument("--profile", default=None, help="Path to user profile JSON (optional)")
    
    args = parser.parse_args()
    
    asyncio.run(
        run_ablation_study(
            dataset_path=args.dataset,
            batch_id=args.batch,
            output_dir=args.output,
            top_k=args.top_k,
            evaluator_model=args.evaluator_model,
            profile_path=args.profile,
        )
    )


if __name__ == "__main__":
    main()
