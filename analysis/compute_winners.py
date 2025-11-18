import pandas as pd

p = 'evaluation/results/per_question_analysis.csv'
df = pd.read_csv(p)

metric_suffixes = ['answer_correctness','context_recall','context_precision']

for qid in df['question_id']:
    row = df[df['question_id'] == qid].iloc[0]
    print('\nQuestion:', qid)
    for metric in metric_suffixes:
        cols = [c for c in df.columns if c.endswith('_'+metric)]
        values = {c.split('_'+metric)[0]: float(row[c]) if pd.notna(row[c]) else float('-inf') for c in cols}
        top = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
        print('  Top for', metric, '->', top[:3])
