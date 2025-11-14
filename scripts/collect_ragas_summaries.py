import json
from pathlib import Path
import re

RESULTS_DIR = Path('evaluation/results')
EXPS = ['baseline', 'no_rag', 'reranking', 'grounded_hyde', 'combined_best', 'semantic_chunking']
METRIC_KEYS = ['faithfulness', 'answer_relevancy', 'answer_correctness', 'context_precision', 'context_recall']

pattern = re.compile(r"ragas_(?P<exp>.+?)_my_policies_(?P<ts>\d{8}_\d{6})\.json$")

summaries = {}
for f in RESULTS_DIR.iterdir():
    m = pattern.match(f.name)
    if not m:
        continue
    exp = m.group('exp')
    ts = m.group('ts')
    if exp not in EXPS:
        continue
    # keep latest
    if exp not in summaries or ts > summaries[exp]['ts']:
        summaries[exp] = {'path': f, 'ts': ts}

metric_store = {}
rows = []
print('Experiment,Timestamp,retrieval_candidate_pool,use_web_research,faithfulness_mean,answer_relevancy_mean,answer_correctness_mean,context_precision_mean,context_recall_mean,latency_total_seconds,latency_avg_seconds')
for exp in EXPS:
    if exp not in summaries:
        print(f"{exp},MISSING,,,,,,,,")
        continue
    ts = summaries[exp]['ts']
    p = summaries[exp]['path']
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        print(f"{exp},{ts},ERROR_READING,,,,,,,,")
        continue
    r = data.get('ragas_metrics', {})
    summary = r.get('summary', {})
    metadata = data.get('metadata', {})
    pool = metadata.get('retrieval_candidate_pool')
    web = metadata.get('use_web_research')
    def get_mean(k):
        v = summary.get(k, {})
        return v.get('mean') if isinstance(v, dict) else None
    metrics = {k: get_mean(k) for k in METRIC_KEYS}
    metric_store[exp] = metrics
    latency_meta = metadata.get('latency_summary', {}) or {}
    total_latency = latency_meta.get('total_seconds')
    avg_latency = latency_meta.get('average_seconds')
    print(f"{exp},{ts},{pool},{web},{metrics['faithfulness']},{metrics['answer_relevancy']},{metrics['answer_correctness']},{metrics['context_precision']},{metrics['context_recall']},{total_latency},{avg_latency}")

    rows.append({
        'experiment': exp,
        'timestamp': ts,
        'pool': pool,
        'web': web,
        'metrics': metrics,
        'latency_total': total_latency,
        'latency_avg': avg_latency,
    })

if rows:
    print('\nSummary table:')
    headers = ['Experiment', 'Timestamp', 'RetrievalPool', 'WebResearch', 'Latency total (s)', 'Latency avg (s)'] + [m.replace('_',' ').title() for m in METRIC_KEYS]
    col_widths = {h: len(h) for h in headers}
    table_rows = []
    for row in rows:
        values = [
            row['experiment'],
            row['timestamp'],
            str(row.get('pool','')),
            str(row.get('web','')),
            f"{row.get('latency_total','')}",
            f"{row.get('latency_avg','')}",
        ]
        metric_values = []
        for m in METRIC_KEYS:
            val = row['metrics'].get(m)
            if isinstance(val, (int, float)):
                metric_values.append(f"{val:.4f}")
            elif val is None:
                metric_values.append("")
            else:
                metric_values.append(str(val))
        values.extend(metric_values)
        for header, val in zip(headers, values):
            col_widths[header] = max(col_widths[header], len(str(val)))
        table_rows.append(values)

    header_line = ' | '.join(h.ljust(col_widths[h]) for h in headers)
    divider = '-+-'.join('-' * col_widths[h] for h in headers)
    print(header_line)
    print(divider)
    for values in table_rows:
        print(' | '.join(str(val).ljust(col_widths[h]) for val, h in zip(values, headers)))

baseline_metrics = metric_store.get('baseline')
if baseline_metrics:
    print('\nComparison versus baseline (delta = experiment - baseline):')
    for exp in EXPS:
        if exp == 'baseline' or exp not in metric_store:
            continue
        metrics = metric_store[exp]
        delta_parts = []
        for key in METRIC_KEYS:
            value = metrics.get(key)
            base_value = baseline_metrics.get(key)
            if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
                delta = value - base_value
                delta_parts.append(f"{key} {delta:+.3f}")
        if delta_parts:
            print(f"  {exp}: {', '.join(delta_parts)}")
