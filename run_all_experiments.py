#!/usr/bin/env python3
"""
Script to run all RAGAS evaluation experiments sequentially.
This will execute baseline, no_rag, reranking, hyde, and semantic_chunking experiments
one after another, using the my_policies batch.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_experiment(experiment_name, batch_id="my_policies"):
    """Run a single experiment using subprocess."""
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "run_evaluation.py",
        "--experiment", experiment_name,
        "--batch_id", batch_id
    ]
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name.upper()}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"Experiment {experiment_name} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment {experiment_name} failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Run all experiments sequentially."""
    experiments = ["baseline", "no_rag", "reranking", "hyde", "semantic_chunking"]
    batch_id = "my_policies"
    
    print("Starting sequential execution of all RAGAS evaluation experiments...")
    print(f"Batch ID: {batch_id}")
    print(f"Experiments: {', '.join(experiments)}")
    
    failed_experiments = []
    
    for exp in experiments:
        success = run_experiment(exp, batch_id)
        if not success:
            failed_experiments.append(exp)
            # Continue to next experiment even if one fails
    
    print(f"\n{'='*80}")
    print("All experiments completed.")
    if failed_experiments:
        print(f"Failed experiments: {', '.join(failed_experiments)}")
    else:
        print("All experiments ran successfully.")
    print("Check evaluation/results/ for output files.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()