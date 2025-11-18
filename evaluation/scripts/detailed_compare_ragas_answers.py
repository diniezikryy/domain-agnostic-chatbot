import json, sys
from pathlib import Path
from collections import defaultdict
import hashlib


def md5(s):
    if not s:
        return ''
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def main():
    if len(sys.argv) < 2:
        print('Usage: detailed_compare_ragas_answers.py <path-to-ragas-json>')
        sys.exit(2)
    p = Path(sys.argv[1])
    data = json.loads(p.read_text(encoding='utf-8'))
    experiments = data['metadata'].get('experiments', [])
    response_table = data.get('response_table', [])
    # Build mapping of index in response_table
    # Extract experiment metrics
    exp_metrics = {}
    for exp in data.get('experiments', []):
        name = exp['metadata']['experiment']
        exp_metrics[name] = exp['ragas_metrics']

    rows = []
    for idx, q in enumerate(response_table):
        qid = q['question_id']
        question = q['question']
        answers = {e: q.get(e, '') for e in experiments}
        # compute unique answers by hash
        unique_map = defaultdict(list)
        for e, ans in answers.items():
            unique_map[md5(ans)].append(e)
        distinct = len(unique_map)
        # get metrics for this question for each experiment
        metrics_for_q = {}
        for e in experiments:
            if e in exp_metrics:
                # find the index of this question in the experiment's ragas_metrics (index = same ordering)
                faith = exp_metrics[e].get('faithfulness', [None])[idx] if exp_metrics[e].get('faithfulness') else None
                answer_rel = exp_metrics[e].get('answer_relevancy', [None])[idx] if exp_metrics[e].get('answer_relevancy') else None
                metrics_for_q[e] = {'faithfulness': faith, 'answer_relevancy': answer_rel}
            else:
                metrics_for_q[e] = {'faithfulness': None, 'answer_relevancy': None}
        rows.append({'qid': qid, 'question': question, 'answers': answers, 'distinct_answers': distinct, 'metrics': metrics_for_q})

    # Print a condensed summary
    out_lines = []
    out_lines.append('| question_id | #distinct_answers | experiments (top faithfulness) |')
    out_lines.append('| --- | ---: | --- |')

    for r in rows:
        qid = r['qid']
        distinct = r['distinct_answers']
        # choose top few experiments by faithfulness
        sorted_by_faith = sorted(r['metrics'].items(), key=lambda kv: (kv[1]['faithfulness'] is not None, kv[1]['faithfulness']), reverse=True)
        top = ', '.join([f"{k}({v['faithfulness']:.2f})" if v['faithfulness'] is not None else f"{k}(NA)" for k,v in sorted_by_faith[:3]])
        out_lines.append(f'| {qid} | {distinct} | {top} |')

    out = '\n'.join(out_lines)
    print(out)
    # Save to file
    out_file = p.with_suffix('.details.md')
    out_file.write_text(out, encoding='utf-8')
    print('\nSaved to', out_file)

if __name__ == '__main__':
    main()
