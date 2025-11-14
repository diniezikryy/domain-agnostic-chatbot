#!/usr/bin/env python3
"""
Tune combined_best parameters by running a small grid search over key knobs.
Runs each parameter combo against the semantic batch and collects metrics.
Picks the best combo based on average answer_correctness.

Usage:
    python scripts/tune_combined_best.py --batch_id my_policies --dataset test_data/evaluation_dataset_auto_ragas.json

Requires:
- OPENAI_API_KEY set
- Semantic batch exists (run semantic_chunking experiment first if needed)
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

# Parameter grid: small but meaningful variations
PARAM_GRID = [
    {
        "RERANK_HYDE_WEIGHT": "0.4",
        "RERANK_KEEP_TOP_N": "3",
        "RETRIEVAL_CANDIDATE_POOL": "30"
    },
    {
        "RERANK_HYDE_WEIGHT": "0.6",
        "RERANK_KEEP_TOP_N": "5",
        "RETRIEVAL_CANDIDATE_POOL": "50"
    },
    {
        "RERANK_HYDE_WEIGHT": "0.8",
        "RERANK_KEEP_TOP_N": "7",
        "RETRIEVAL_CANDIDATE_POOL": "70"
    }
]

def run_experiment_combo(
    batch_id: str,
    dataset_path: str,
    params: Dict[str, str],
    output_dir: Path
) -> Dict[str, Any]:
    """Run one parameter combo and return the results."""
    print(f"\n--- Running combo: {params} ---")

    # Build environment variables
    env = os.environ.copy()
    env.update(params)
    env["USE_WEB_RESEARCH"] = "false"  # Keep consistent for tuning
    env["GENERATION_TOP_K"] = "8"      # Fixed for fair comparison

    # Run the experiment
    cmd = [
        sys.executable, "run_evaluation.py",
        "--experiment", "combined_best",
        "--batch_id", batch_id,
        "--dataset", dataset_path,
        "--no-cache",    # Fresh runs for tuning
    ]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=3600  # 60 min timeout per combo (RAGAS runs can be slow)
        )
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return {"params": params, "error": result.stderr, "success": False}

        output_lines = result.stdout.split('\n')
        metrics = {}

        # Look for the local metrics summary
        in_local_metrics = False
        for line in output_lines:
            if "Local Support Metrics" in line:
                in_local_metrics = True
            if in_local_metrics and "Questions analysed:" in line:
                # Extract average metrics
                parts = line.split()
                if len(parts) >= 3:
                    metrics["questions_analysed"] = int(parts[2])
                continue
            if in_local_metrics and "Context support" in line:
                parts = line.split()
                if len(parts) >= 4:
                    metrics["context_support_pct"] = float(parts[3].rstrip('%'))
                continue
            if in_local_metrics and "Answer support" in line:
                parts = line.split()
                if len(parts) >= 4:
                    metrics["answer_support_pct"] = float(parts[3].rstrip('%'))
                continue

        # Look for the output file path
        output_file = None
        for line in output_lines:
            if "Output:" in line:
                output_file = line.split("Output:")[-1].strip()
                break

        return {
            "params": params,
            "success": True,
            "metrics": metrics,
            "output_file": output_file,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        return {"params": params, "error": "Timeout", "success": False}
    except Exception as e:
        return {"params": params, "error": str(e), "success": False}

def main():
    parser = argparse.ArgumentParser(description="Tune combined_best parameters")
    parser.add_argument(
        "--batch_id",
        type=str,
        default="my_policies",
        help="Batch ID to use (page-chunked batch, default: my_policies)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="test_data/evaluation_dataset_auto_ragas.json",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/tuning_results",
        help="Directory to save tuning results"
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Please set it before running.")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Check if batch exists
    batch_meta = Path("batches") / args.batch_id / "metadata.json"
    if not batch_meta.exists():
        print(f"ERROR: Batch '{args.batch_id}' not found. Run semantic_chunking experiment first.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting parameter tuning for combined_best")
    print(f"Batch: {args.batch_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Output dir: {output_dir}")
    print(f"Parameter combinations: {len(PARAM_GRID)}")

    results = []
    best_result = None
    best_score = -1

    for i, params in enumerate(PARAM_GRID, 1):
        print(f"\n[{i}/{len(PARAM_GRID)}] Testing parameters: {params}")

        result = run_experiment_combo(args.batch_id, str(dataset_path), params, output_dir)
        results.append(result)

        if result["success"]:
            # Use context support as proxy for answer correctness (since we skipped RAGAS)
            score = result["metrics"].get("context_support_pct", 0)
            if score > best_score:
                best_score = score
                best_result = result
        else:
            print(f"  FAILED: {result.get('error', 'Unknown error')}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"tuning_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "batch_id": args.batch_id,
            "dataset": str(dataset_path),
            "parameter_grid": PARAM_GRID,
            "results": results,
            "best_result": best_result,
            "best_score": best_score
        }, f, indent=2)

    print("\n=== TUNING COMPLETE ===")
    print(f"Results saved to: {results_file}")

    if best_result:
        print("\nBEST PARAMETERS:")
        print(f"  Params: {best_result['params']}")
        print(f"  Score: {best_score:.1f}")
        print(f"  Output file: {best_result.get('output_file', 'N/A')}")
    else:
        print("\nNo successful runs. Check errors above.")

if __name__ == "__main__":
    main()