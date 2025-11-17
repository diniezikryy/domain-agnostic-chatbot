#!/usr/bin/env python3
"""
Generate a comparison CSV and HTML for RAGAS metric summaries across experiments.
Writes outputs to the evaluation/results/ directory with a timestamp.
Also produces a small cache report from evaluation/cache/_cache_index.json if present.
"""
from pathlib import Path
import json
import re
import csv
import datetime
import sys

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_INDEX = Path(__file__).resolve().parent / "cache" / "_cache_index.json"

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

PATTERN = re.compile(r"ragas_([a-zA-Z0-9_]+)_my_policies_(\d{8}_\d{6})\.json$")
REPORT_PATTERN = re.compile(r"ragas_report_my_policies_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_.*\.json$")


def find_latest_ragas_files(results_dir: Path):
    files = {}
    for p in results_dir.glob("ragas_*_my_policies_*.json"):
        m = PATTERN.search(p.name)
        if not m:
            continue
        exp = m.group(1)
        ts_str = m.group(2)
        try:
            ts = datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        except Exception:
            ts = datetime.datetime.fromtimestamp(p.stat().st_mtime)
        prev = files.get(exp)
        if prev is None or ts > prev["ts"]:
            files[exp] = {"path": p, "ts": ts}
    return files


def find_latest_ragas_report(results_dir: Path):
    """Find the latest ragas_report for this batch and return its path.

    This is a fallback for aggregated runs that create a single report file
    containing the results for multiple experiments (e.g., baseline+semantic_reranking).
    """
    reports = list(results_dir.glob("ragas_report_my_policies_*.json"))
    if not reports:
        return None
    # Pick newest by modification time
    reports_sorted = sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True)
    return reports_sorted[0]


def load_ragas_summary(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}", file=sys.stderr)
        return None
    ragas = data.get("ragas_metrics", {})
    summary = ragas.get("summary", {}) if isinstance(ragas, dict) else {}
    metadata = data.get("metadata", {})
    return summary, metadata


def format_float(v):
    if v is None:
        return ""
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def write_csv(rows, out_path: Path):
    headers = [
        "experiment",
        "file",
        "timestamp",
        "num_questions",
        "retrieval_candidate_pool",
        "latency_total_seconds",
        "latency_avg_seconds",
    ] + METRICS

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_html(rows, cache_summary, out_path: Path):
    title = "RAGAS Evaluation Comparison - my_policies"
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    # Simple color map for metrics
    color_map = {
        "faithfulness": "#1f77b4",
        "answer_relevancy": "#ff7f0e",
        "context_precision": "#2ca02c",
        "context_recall": "#d62728",
        "answer_correctness": "#9467bd",
    }

    def bar_html(value, metric):
        if value == "":
            val = 0.0
        else:
            try:
                val = float(value)
            except Exception:
                val = 0.0
        pct = max(0.0, min(1.0, val)) * 100
        color = color_map.get(metric, "#4CAF50")
        return (
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<div style='flex:1;background:#eee;border-radius:6px;height:14px;overflow:hidden;'>"
            f"<div style='width:{pct}%;height:100%;background:{color};'></div>"
            f"</div><div style='width:64px;text-align:right;font-weight:600;'>{val:.3f}</div>"
            f"</div>"
        )

    html_parts = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        f"<meta charset=\"utf-8\"><title>{title}</title>",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "<style>",
        "body{font-family:Inter, Roboto, Arial, sans-serif;margin:24px;color:#111}",
        "table{border-collapse:collapse;width:100%;margin-bottom:20px}",
        "th,td{border:1px solid #e6e6e6;padding:8px;text-align:left}",
        "th{background:#fafafa;font-weight:700}",
        "h1{font-size:18px;margin-bottom:6px}",
        "small{color:#666}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<small>Generated: {now}</small>",
        "<p>Mean RAGAS metrics per experiment (latest outputs).</p>",
        "<table>",
    ]

    # header
    header_row = (
        "<tr>"
        "<th>Experiment</th>"
        "<th>Timestamp</th>"
        "<th>#Q</th>"
        "<th>RetrievalPool</th>"
        "<th>Latency total (s)</th>"
        "<th>Latency avg (s)</th>"
    )
    for m in METRICS:
        header_row += f"<th>{m.replace('_',' ').title()}</th>"
    header_row += "</tr>"
    html_parts.append(header_row)

    for r in rows:
        row_html = f"<tr><td style='font-weight:700'>{r['experiment']}</td>"
        row_html += f"<td>{r['timestamp']}</td>"
        row_html += f"<td>{r.get('num_questions','')}</td>"
        row_html += f"<td>{r.get('retrieval_candidate_pool','')}</td>"
        row_html += f"<td>{r.get('latency_total_seconds','')}</td>"
        row_html += f"<td>{r.get('latency_avg_seconds','')}</td>"
        for m in METRICS:
            row_html += f"<td>{bar_html(r.get(m, ''), m)}</td>"
        row_html += "</tr>"
        html_parts.append(row_html)

    html_parts.append("</table>")

    # Cache summary
    if cache_summary is not None:
        html_parts.append("<h2>Cache summary</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Total entries</th><th>Total size (KB)</th><th>Last modified</th></tr>")
        html_parts.append(
            f"<tr><td>{cache_summary['total_entries']}</td><td>{cache_summary['total_size_kb']:.1f}</td><td>{cache_summary['last_modified']}</td></tr>"
        )
        html_parts.append("</table>")

        # breakdown by experiment
        html_parts.append("<h3>Entries by experiment</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Experiment</th><th>Count</th><th>Avg size (KB)</th></tr>")
        for exp,info in sorted(cache_summary['by_experiment'].items(), key=lambda x: x[0]):
            html_parts.append(f"<tr><td>{exp}</td><td>{info['count']}</td><td>{info['avg_kb']:.1f}</td></tr>")
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


def build_cache_summary(cache_index_path: Path):
    if not cache_index_path.exists():
        return None
    try:
        with open(cache_index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
    except Exception as e:
        print(f"Failed to read cache index: {e}", file=sys.stderr)
        return None
    entries = idx.get("entries", {})
    total_entries = len(entries)
    total_size = 0
    by_exp = {}
    last_modified = idx.get("metadata", {}).get("last_modified")
    for k,v in entries.items():
        sz = v.get("file_size_bytes", 0)
        total_size += sz
        exp = v.get("experiment", "unknown")
        by_exp.setdefault(exp, []).append(sz)
    by_experiment = {}
    for exp,sizes in by_exp.items():
        by_experiment[exp] = {
            "count": len(sizes),
            "avg_kb": (sum(sizes)/len(sizes)/1024) if sizes else 0,
        }

    return {
        "total_entries": total_entries,
        "total_size_kb": total_size/1024.0,
        "last_modified": last_modified,
        "by_experiment": by_experiment,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = find_latest_ragas_files(RESULTS_DIR)
    if not files:
        # Try aggregated ragas report (single JSON containing multiple experiments)
        report_path = find_latest_ragas_report(RESULTS_DIR)
        if report_path is None:
            print(f"No ragas JSON files found in {RESULTS_DIR}")
            return 1
        # Load aggregated report and extract experiments
        print(f"Found aggregated ragas report: {report_path.name}. Building comparison from it.")
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Each experiment payload is in data['experiments'] list
        exp_payloads = data.get('experiments', [])
        rows = []
        for payload in exp_payloads:
            metadata = payload.get('metadata', {})
            ragas = payload.get('ragas_metrics', {})
            summary = ragas.get('summary', {}) if isinstance(ragas, dict) else {}
            latency_meta = metadata.get('latency_summary') or {}
            row = {
                'experiment': metadata.get('experiment', 'unknown'),
                'file': report_path.name,
                'timestamp': metadata.get('timestamp') or data.get('metadata', {}).get('timestamp') or '',
                'num_questions': metadata.get('num_questions',''),
                'retrieval_candidate_pool': metadata.get('retrieval_candidate_pool',''),
                'latency_total_seconds': format_float(latency_meta.get('total_seconds')),
                'latency_avg_seconds': format_float(latency_meta.get('average_seconds')),
            }
            for m in METRICS:
                val = ''
                if isinstance(summary, dict) and m in summary:
                    v = summary.get(m)
                    if isinstance(v, dict) and 'mean' in v:
                        val = v['mean']
                    else:
                        try:
                            val = float(v)
                        except Exception:
                            val = ''
                row[m] = format_float(val) if val != '' else ''
            rows.append(row)

    if files:
        rows = []
        for exp, info in sorted(files.items()):
            path = info['path']
            summary, metadata = load_ragas_summary(path)
            if summary is None:
                continue
        latency_meta = metadata.get('latency_summary') or {}
        row = {
            'experiment': exp,
            'file': path.name,
            'timestamp': (metadata.get('timestamp') or info['ts'].strftime('%Y%m%d_%H%M%S')),
            'num_questions': metadata.get('num_questions',''),
            'retrieval_candidate_pool': metadata.get('retrieval_candidate_pool',''),
            'latency_total_seconds': format_float(latency_meta.get('total_seconds')),
            'latency_avg_seconds': format_float(latency_meta.get('average_seconds')),
        }
        for m in METRICS:
            val = ""
            if isinstance(summary, dict) and m in summary:
                # summary[m] might be dict with mean
                if isinstance(summary[m], dict) and 'mean' in summary[m]:
                    val = summary[m]['mean']
                else:
                    # sometimes summary may store arrays only or scalar
                    try:
                        val = float(summary[m])
                    except Exception:
                        val = ""
            row[m] = format_float(val) if val != "" else ""
        rows.append(row)

    now_ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_csv = RESULTS_DIR / f"ragas_comparison_my_policies_{now_ts}.csv"
    out_html = RESULTS_DIR / f"ragas_comparison_my_policies_{now_ts}.html"

    write_csv(rows, out_csv)

    cache_summary = build_cache_summary(CACHE_INDEX)
    make_html(rows, cache_summary, out_html)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote HTML: {out_html}")
    if cache_summary is not None:
        print("Cache entries:", cache_summary['total_entries'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
