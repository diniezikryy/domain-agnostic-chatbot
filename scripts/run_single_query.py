"""Run all RAG pipeline experiments for a single ad-hoc query.

This helper avoids creating a full evaluation dataset by constructing a
single-question payload and calling the existing `run_pipeline` and
`run_ragas_evaluation` functions from `run_evaluation.py`.

Usage examples (PowerShell):

# Run baseline + reranking cold (no cache) for an ad-hoc query:
# Note: replace PROFILE_ID with your profile file id in test_data (e.g., profile_1)
$qt = 'What is the annual benefit limit for GREAT SupremeHealth?'
python .\scripts\run_single_query.py --query "$qt" --ground_truth "S$1,500,000" --profile_id profile_1 --batch_id my_policies --experiments baseline,reranking --no-cache

# Run all experiments (may take longer and hit API limits)
python .\scripts\run_single_query.py --query "How long is post-hospitalisation treatment covered?" --profile_id profile_1 --batch_id my_policies

"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Make repo importable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

# Import functions from the evaluation harness
from run_evaluation import (
    run_pipeline,
    run_ragas_evaluation,
    compute_local_support_metrics,
    save_results,
    DEFAULT_TEST_BATCH_ID,
)
from utils.cache_manager import CacheManager


def load_profile(profile_id: str):
    p = Path('test_data') / f"{profile_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run experiments for a single ad-hoc query.")
    parser.add_argument("--query", type=str, required=True, help="The user query to evaluate")
    parser.add_argument("--ground_truth", type=str, default="", help="Optional ground-truth text for evaluation")
    parser.add_argument("--profile_id", type=str, default="profile_1", help="User profile id (file in test_data/<profile_id>.json)")
    parser.add_argument("--batch_id", type=str, default=DEFAULT_TEST_BATCH_ID, help="Batch id containing the documents (default from run_evaluation)")
    parser.add_argument("--experiments", type=str, default="baseline,reranking,no_rag,grounded_hyde,combined_best,semantic_chunking",
                        help="Comma-separated experiment names to run (defaults to many).")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache for these runs (force fresh processing)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache for each experiment before running")
    parser.add_argument("--skip_ragas", action="store_true", help="Skip running RAGAS evaluation (fast mode)")
    args = parser.parse_args()

    load_dotenv()

    experiments = [e.strip() for e in args.experiments.split(',') if e.strip()]
    if not experiments:
        print("No experiments selected. Exiting.")
        return

    profile = load_profile(args.profile_id)
    profiles = {args.profile_id: profile}

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    question_id = f"SINGLE_{timestamp}"

    questions = [{
        "question_id": question_id,
        "question": args.query,
        "ground_truth": args.ground_truth,
        "user_profile_id": args.profile_id,
    }]

    # For each experiment run the pipeline and optionally RAGAS evaluation
    for exp in experiments:
        print(f"\n--- Running experiment: {exp} ---")
        cache_manager = None if args.no_cache else CacheManager()
        if args.no_cache:
            print("[Cache] Disabled for this run (--no-cache)")
        else:
            print("[Cache] Enabled")

        if args.clear_cache and cache_manager:
            print(f"[Cache] Clearing cache for experiment={exp}, batch={args.batch_id}...")
            try:
                cache_manager.clear(experiment_name=exp, batch_id=args.batch_id)
            except Exception as e:
                print(f"Failed to clear cache: {e}")

        try:
            pipeline_results = run_pipeline(questions, profiles, exp, args.batch_id, cache_manager)
        except Exception as e:
            print(f"Error running pipeline for {exp}: {e}")
            continue

        local_metrics = compute_local_support_metrics(pipeline_results)

        if args.skip_ragas:
            evaluation_result = {}
            evaluation_seconds = 0.0
            print("[Info] Skipping RAGAS evaluation as requested (--skip_ragas)")
        else:
            evaluation_result, evaluation_seconds = run_ragas_evaluation(pipeline_results, exp, args.batch_id, cache_manager)

        out_path = save_results(evaluation_result, pipeline_results, exp, args.batch_id, local_metrics, evaluation_seconds)
        print(f"Saved experiment output: {out_path}")

    print("\nAll requested experiments completed.")


if __name__ == "__main__":
    main()
