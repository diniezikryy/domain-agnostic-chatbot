import json
import sys
import math
from pathlib import Path
from collections import defaultdict


def safe_mean(seq):
    seq = [x for x in seq if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not seq:
        return float('nan')
    return sum(seq) / len(seq)


def print_metric_table(data):
    print('\nMetric means per experiment (from ragas_metrics.summary):\n')
    rows = []
    for exp in data['experiments']:
        name = exp['metadata']['experiment']
        summ = exp['ragas_metrics']['summary'] if 'ragas_metrics' in exp and 'summary' in exp['ragas_metrics'] else {}
        rows.append(
            (
                name,
                summ.get('faithfulness', {}).get('mean', float('nan')),
                summ.get('answer_relevancy', {}).get('mean', float('nan')),
                summ.get('context_precision', {}).get('mean', float('nan')),
                summ.get('answer_correctness', {}).get('mean', float('nan')),
            )
        )
    # Sort by faithfulness desc as default
    rows = sorted(rows, key=lambda r: (r[1] is not None and not (isinstance(r[1], float) and math.isnan(r[1])), r[1]), reverse=True)
    print('| Experiment | faithfulness | answer_relevancy | context_precision | answer_correctness |')
    print('|---|---:|---:|---:|---:|')
    for r in rows:
        print(f'| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} |')


def per_question_winners(data):
    # Build ragas_metrics arrays map
    exp_map = {e['metadata']['experiment']: e for e in data['experiments']}
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'answer_correctness']
    n_questions = len(data['response_table'])

    winners = defaultdict(lambda: defaultdict(list))
    for i in range(n_questions):
        qid = data['response_table'][i]['question_id']
        for metric in metrics:
            best = None
            for name, exp in exp_map.items():
                arr = exp['ragas_metrics'].get(metric)
                if not arr:
                    continue
                val = arr[i]
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                if best is None or val > best[0]:
                    best = (val, name)
            if best:
                winners[qid][metric].append((best[1], best[0]))
            else:
                winners[qid][metric].append((None, None))
    # Print winners
    print('\nPer-question winners (top-by-metric):\n')
    for i, row in enumerate(data['response_table']):
        qid = row['question_id']
        print(f"Question {qid}: {row['question'][:120]}...")
        for metric in metrics:
            if winners[qid][metric] and winners[qid][metric][0][0] is not None:
                print(f" - {metric} top: {winners[qid][metric][0][0]} (score={winners[qid][metric][0][1]:.3f})")
            else:
                print(f" - {metric} top: None")
        print()


def unique_answer_counts(data):
    print('\nUnique answer counts across experiments per question:')
    for row in data['response_table']:
        qid = row['question_id']
        answers = []
        exps = []
        for exp in data['metadata']['experiments']:
            ans = row.get(exp, '')
            answers.append(ans.strip())
            exps.append(exp)
        unique = {}  # hash->[expnames]
        for name, ans in zip(exps, answers):
            h = ans
            if h not in unique:
                unique[h] = []
            unique[h].append(name)
        distinct = len(unique)
        print(f" - {qid}: {distinct} distinct answers across {len(exps)} experiments")
        # print small mapping
        for ans, exps in unique.items():
            expstr = ', '.join(exps)
            print(f"    - {expstr} -> '{ans[:100].replace('\n',' ')}{'…' if len(ans)>100 else ''}'")


def context_usage_analysis(data):
    print('\nContext usage statistics: average #rag_chunks + sizes per experiment')
    for exp in data['experiments']:
        name = exp['metadata']['experiment']
        r = exp.get('pipeline_results', [])
        if not r:
            print(f' - {name}: no pipeline results')
            continue
        counts = []
        sizes = []
        for q in r:
            rag_chunks = q.get('rag_chunks', [])
            counts.append(len(rag_chunks))
            sizes.append(sum(len(chunk.get('content','')) for chunk in rag_chunks))
        if counts:
            print(f' - {name}: avg_chunks={sum(counts)/len(counts):.2f}, avg_chunk_chars={sum(sizes)/len(sizes):.1f}')
        else:
            print(f' - {name}: no chunks')


def top_experiments_by_metric(data):
    # Build simple ranking for metric means from comparison_table in top level
    comp = {m['metric']: m for m in data['comparison_table']}
    print('\nTop experiments by metric (comparison_table):')
    for metric, d in comp.items():
        print(f' - {metric}')
        # sort experiments by value
        values = []
        for k, v in d.items():
            if k == 'metric':
                continue
            values.append((k, v))
        values = sorted(values, key=lambda kv: (kv[1] is not None and not (isinstance(kv[1], float) and math.isnan(kv[1])), kv[1]), reverse=True)
        for name, val in values[:5]:
            print(f'    {name}: {val}')


def main():
    if len(sys.argv) < 2:
        print('Usage: analyze_ragas_experiments.py <path>')
        sys.exit(2)
    p = Path(sys.argv[1])
    if not p.exists():
        print('File not found:', p)
        sys.exit(2)
    data = json.loads(p.read_text(encoding='utf-8'))
    print('File:', p)
    print('Dataset:', data['metadata'].get('dataset'))

    # 1. metrics table
    print_metric_table(data)
    # 2. winners per question
    per_question_winners(data)
    # 3. unique answers
    unique_answer_counts(data)
    # 4. context usage
    context_usage_analysis(data)
    # 5. top experiments by metric
    top_experiments_by_metric(data)


if __name__ == '__main__':
    main()
