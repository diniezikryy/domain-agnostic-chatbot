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
        "cached_total_seconds",
        "cold_total_seconds",
        "cached_count",
        "cold_count",
        "wall_clock_seconds",
        "per_question_cold_seconds",
        "per_question_cached_seconds",
    ]))

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Skipping {os.path.basename(path)} because it couldn't be parsed: {exc}")
            continue
        meta = data.get("metadata", {})
        # Prefer per-question breakdown from the pipeline results so we can split
        # cached vs cold runs using the `from_cache` flag. Fall back to latency_summary
        # in metadata if detailed pipeline entries are not present.
        pipeline_results = data.get("pipeline_results", []) or []

        cold_per_question = []
        cached_per_question = []
        for entry in pipeline_results:
            # Derive a sensible per-question latency: use latency_seconds if present,
            # otherwise sum retrieval_seconds+generation_seconds or fallback to 0.0
            latency = entry.get("latency_seconds")
            if latency is None:
                r = entry.get("retrieval_seconds") or 0.0
                g = entry.get("generation_seconds") or 0.0
                latency = float(r) + float(g)

            if entry.get("from_cache"):
                cached_per_question.append(float(latency))
            else:
                cold_per_question.append(float(latency))

        # Fallback: if no pipeline entries present, use metadata.latency_summary
        if not pipeline_results:
            ls = meta.get("latency_summary", {})
            per_question = ls.get("per_question_seconds", [])
            # We cannot determine which were cached, assume all are cold
            cold_per_question = [float(v) for v in per_question]
            cached_per_question = []

        total_latency = sum(cold_per_question) + sum(cached_per_question)
        cached_total = sum(cached_per_question)
        cold_total = sum(cold_per_question)
        cached_count = len(cached_per_question)
        cold_count = len(cold_per_question)

        wall_clock_seconds = (cold_total + throttle_delay * cold_count) if (cold_count or cold_total) else 0.0

        per_cold_str = "|".join(f"{v:.4f}" for v in cold_per_question) if cold_per_question else ""
        per_cached_str = "|".join(f"{v:.4f}" for v in cached_per_question) if cached_per_question else ""

        print(",".join([
            os.path.basename(path),
            meta.get("experiment", ""),
            meta.get("batch_id", ""),
            str(meta.get("num_questions", "")),
            format_seconds(total_latency),
            format_seconds(cached_total),
            format_seconds(cold_total),
            str(cached_count),
            str(cold_count),
            format_seconds(wall_clock_seconds),
            per_cold_str,
            per_cached_str,
        ]))


if __name__ == "__main__":
    main()
