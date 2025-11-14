"""Utility to inspect per-question latency metadata across RAGAS artifacts."""
import glob
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.throttler import compute_safe_delay

def format_seconds(value):
    return f"{value:.6f}" if value is not None else "NA"


def main():
    files = sorted(glob.glob(os.path.join("evaluation", "results", "ragas_*.json")))
    if not files:
        print("No RAGAS results found under evaluation/results.")
        return

    print(f"Found {len(files)} ragas result files.")
    throttle_delay = compute_safe_delay()
    print(f"Assuming a constant throttle delay of {throttle_delay:.2f}s between questions (not included in latency_summary)." )
    print(",".join([
        "file",
        "experiment",
        "batch",
        "num_questions",
        "total_seconds",
        "average_seconds",
        "min_seconds",
        "max_seconds",
        "wall_clock_seconds",
        "per_question_seconds",
    ]))

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Skipping {os.path.basename(path)} because it couldn't be parsed: {exc}")
            continue
        meta = data.get("metadata", {})
        ls = meta.get("latency_summary", {})
        per_question = ls.get("per_question_seconds", [])
        per_question_str = "|".join(f"{value:.4f}" for value in per_question) if per_question else ""
        total_latency = sum(per_question) if per_question else 0.0
        wall_clock_seconds = (total_latency + throttle_delay * len(per_question)) if per_question else None
        print(
            ",".join([
                os.path.basename(path),
                meta.get("experiment", ""),
                meta.get("batch_id", ""),
                str(meta.get("num_questions", "")),
                format_seconds(ls.get("total_seconds")),
                format_seconds(ls.get("average_seconds")),
                format_seconds(ls.get("min_seconds")),
                format_seconds(ls.get("max_seconds")),
                format_seconds(wall_clock_seconds) if wall_clock_seconds is not None else "",
                per_question_str,
            ])
        )


if __name__ == "__main__":
    main()
