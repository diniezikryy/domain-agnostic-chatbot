#!/usr/bin/env python3
"""
Serial RAGAS re-evaluation helper
Runs RAGAS.evaluate per-item serially to avoid parallel worker timeouts and recover missing metrics like context_precision.

This script uses the validated configuration (ChatOpenAI with timeouts + RAGAS RunConfig)
to eliminate NaN values and timeout issues discovered during debugging.

Usage: run this inside the project venv where OPENAI_API_KEY is available in the environment (or in .env).
    python evaluation/rerun_ragas_serial.py
"""

import json
import os
import traceback
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from langchain_openai import ChatOpenAI


def safe_evaluate_single(item, metrics, llm, run_config):
    """Evaluate a single QA pair and return a pandas DataFrame row (or None on failure)."""
    try:
        ds = Dataset.from_list([
            {
                "question": item.get("question"),
                "answer": item.get("answer"),
                "contexts": item.get("contexts", []),
                "ground_truth": item.get("ground_truth", ""),
            }
        ])
        ev = evaluate(ds, metrics=metrics, llm=llm, run_config=run_config)

        # ev is usually a RAGAS EvaluationResult with to_pandas()
        if hasattr(ev, "to_pandas"):
            df = ev.to_pandas()
        elif isinstance(ev, pd.DataFrame):
            df = ev
        elif hasattr(ev, "to_dict"):
            df = pd.DataFrame([ev.to_dict()])
        else:
            # Fallback: try to coerce to dict
            df = pd.DataFrame([dict(ev)])

        return df

    except Exception:
        traceback.print_exc()
        return None


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set in the environment or .env before running this script")

    base_input = Path("evaluation/results/ragas_baseline_my_policies_20251112_112727.json")
    if not base_input.exists():
        print(f"Input file not found: {base_input}")
        return

    with open(base_input, "r", encoding="utf-8") as f:
        data = json.load(f)

    pipeline_results = data.get("pipeline_results", [])
    if not pipeline_results:
        print("No pipeline_results found in the input file.")
        return

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    # Configure LLM with explicit settings to avoid NaN and timeout issues
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        request_timeout=180,
        max_retries=3,
    )

    # Configure RAGAS RunConfig for serial execution with extended timeout
    run_config = RunConfig(
        max_workers=1,
        timeout=300,
    )

    rows = []
    failures = []

    print(f"Re-evaluating {len(pipeline_results)} items serially using gpt-4o-mini...")

    for i, item in enumerate(pipeline_results, start=1):
        qid = item.get("question_id", f"item_{i}")
        print(f"\n[{i}/{len(pipeline_results)}] Evaluating {qid}: {item.get('question')[:120]}...")
        df = safe_evaluate_single(item, metrics=metrics, llm=llm, run_config=run_config)
        if df is None or df.empty:
            print(f"[WARN] Evaluation failed for {qid}")
            failures.append({"index": i - 1, "question_id": qid})
        else:
            # attach question_id for traceability
            df["question_id"] = qid
            rows.append(df)
            print(f"[OK] Completed {qid}")

    if rows:
        full_df = pd.concat(rows, ignore_index=True)
    else:
        full_df = pd.DataFrame()

    out_dir = base_input.parent
    out_base = base_input.stem + "_with_metrics_retry"
    out_json = out_dir / f"{out_base}.json"
    out_csv = out_dir / f"{out_base}.csv"

    ragas_metrics = {
        "per_item": full_df.to_dict(orient="records"),
        "failed": failures,
    }

    output_data = {
        "metadata": data.get("metadata", {}),
        "ragas_metrics": ragas_metrics,
        "pipeline_results": pipeline_results,
    }

    # Write output
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Save CSV if we have results
    if not full_df.empty:
        full_df.to_csv(out_csv, index=False, encoding="utf-8")

    print("\nDone.")
    print(f"Saved JSON: {out_json}")
    if not full_df.empty:
        print(f"Saved CSV:  {out_csv}")
    if failures:
        print(f"Failures: {len(failures)} items failed. See output JSON 'failed' list.")


if __name__ == "__main__":
    main()
