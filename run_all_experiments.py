#!/usr/bin/env python3
"""Sequential runner for RAGAS experiments with per-experiment metric summary."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

METRIC_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def find_latest_results_file(experiment: str, batch_id: str) -> Optional[Path]:
    """Return the most recent RAGAS results JSON for the experiment/batch."""
    results_dir = Path("evaluation/results")
    pattern = f"ragas_{experiment}_{batch_id}_*.json"
    matches = list(results_dir.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def extract_metric_summary(results_path: Path) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Load the saved RAGAS JSON and extract metric means and metadata."""
    with results_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    metadata = payload.get("metadata", {})
    ragas_summary = (payload.get("ragas_metrics", {}) or {}).get("summary", {})

    metrics = {}
    for key in METRIC_KEYS:
        value = ragas_summary.get(key, {}).get("mean")
        metrics[key] = float(value) if isinstance(value, (int, float)) else None

    return metrics, {
        "timestamp": metadata.get("timestamp", ""),
        "retrieval_candidate_pool": str(metadata.get("retrieval_candidate_pool", "")),
        "use_web_research": str(metadata.get("use_web_research", "")),
        "rerank_keep_top_n": str(metadata.get("rerank_keep_top_n", "")),
        "generation_top_k": str(metadata.get("generation_top_k", "")),
    }


def run_experiment(experiment_name: str, batch_id: str, show_table: bool) -> Tuple[bool, Optional[Dict[str, float]], Optional[Dict[str, str]], Optional[Path]]:
    """Run a single experiment and return success flag with metric summary."""
    cmd = [
        sys.executable,
        "run_evaluation.py",
        "--experiment",
        experiment_name,
        "--batch_id",
        batch_id,
    ]
    if show_table:
        cmd.append("--show_table")
    else:
        cmd.append("--no_show_table")

    print(f"\n{'=' * 90}")
    print(f"Running experiment: {experiment_name.upper()}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 90}\n")

    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"Experiment {experiment_name} completed successfully.")
    except subprocess.CalledProcessError as exc:
        print(f"Experiment {experiment_name} failed with exit code {exc.returncode}")
        if exc.stderr:
            print(exc.stderr)
        return False, None, None, None

    latest_file = find_latest_results_file(experiment_name, batch_id)
    if not latest_file:
        print(f"[WARN] Could not locate results JSON for {experiment_name} ({batch_id}).")
        return True, None, None, None

    metrics, extra_meta = extract_metric_summary(latest_file)
    return True, metrics, extra_meta, latest_file


def format_summary_table(rows: List[Dict[str, Any]]) -> str:
    """Create a pretty-printed table summarizing metric means per experiment."""
    if not rows:
        return "(no metrics captured)"

    headers = [
        "experiment",
        "timestamp",
        "cand_pool",
        "web?",
        "rerank_k",
        "gen_k",
    ] + METRIC_KEYS

    # Prepare string rows with formatting
    formatted_rows: List[List[str]] = []
    for row in rows:
        formatted = [
            row.get("experiment", ""),
            row.get("timestamp", ""),
            row.get("retrieval_candidate_pool", ""),
            row.get("use_web_research", ""),
            row.get("rerank_keep_top_n", ""),
            row.get("generation_top_k", ""),
        ]
        for metric in METRIC_KEYS:
            value = row.get(metric)
            formatted.append(f"{value:.4f}" if isinstance(value, float) else "--")
        formatted_rows.append(formatted)

    # Compute column widths
    widths = [len(h) for h in headers]
    for formatted in formatted_rows:
        for idx, cell in enumerate(formatted):
            widths[idx] = max(widths[idx], len(cell))

    def render_line(parts: List[str]) -> str:
        return " | ".join(part.ljust(widths[idx]) for idx, part in enumerate(parts))

    divider = "-+-".join("-" * width for width in widths)
    table_lines = [render_line(headers), divider]
    for formatted in formatted_rows:
        table_lines.append(render_line(formatted))

    return "\n".join(table_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS experiments sequentially and show metric summaries.")
    parser.add_argument(
        "--batch_id",
        default="my_policies",
        help="Batch identifier to evaluate (default: my_policies)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "baseline",
            "reranking",
            "rrf",
            "grounded_hyde",
            "combined_best",
            "combined_best_rrf",
            "semantic_chunking",
        ],
        help="List of experiments to run (space separated).",
    )
    parser.add_argument(
        "--show_table",
        action="store_true",
        help="Pass --show_table to each run_evaluation invocation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Starting sequential execution of RAGAS experiments...")
    print(f"Batch ID: {args.batch_id}")
    print(f"Experiments: {', '.join(args.experiments)}")

    failed: List[str] = []
    summary_rows: List[Dict[str, Any]] = []

    for experiment in args.experiments:
        success, metrics, extra_meta, results_file = run_experiment(experiment, args.batch_id, args.show_table)
        if not success:
            failed.append(experiment)
            continue

        row: Dict[str, Any] = {"experiment": experiment}
        if extra_meta:
            row.update(extra_meta)
        if metrics:
            row.update(metrics)
        if results_file:
            row["results_file"] = str(results_file)
        summary_rows.append(row)

    print(f"\n{'=' * 90}")
    print("All experiments completed.")
    if failed:
        print(f"Failed experiments: {', '.join(failed)}")
    else:
        print("All experiments ran successfully.")

    print("\nRAGAS metric summary (means per experiment):")
    print(format_summary_table(summary_rows))

    if summary_rows:
        print("\nLatest result artifacts:")
        for row in summary_rows:
            file_path = row.get("results_file")
            if file_path:
                print(f"  • {row['experiment']}: {file_path}")

    print("\nCheck evaluation/results/ for detailed outputs.")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()