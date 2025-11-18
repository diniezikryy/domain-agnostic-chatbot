import json
import sys
import textwrap
from pathlib import Path

MAXLEN = 300


def truncate(text, n=MAXLEN):
    if text is None:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[:n-1] + "…"


def main():
    if len(sys.argv) < 2:
        print("Usage: summarize_ragas_answers.py <path-to-ragas-json>")
        sys.exit(2)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(2)

    data = json.loads(path.read_text(encoding='utf-8'))
    experiments = data['metadata'].get('experiments', [])
    response_table = data.get('response_table', [])

    # Print compact comparison table
    header = ['question_id', 'question'] + experiments
    sep = '| ' + ' | '.join(header) + ' |'
    print(sep)
    print('| ' + ' | '.join(['---'] * len(header)) + ' |')

    for row in response_table:
        qid = row.get('question_id', '')
        question = truncate(row.get('question', ''), 140)
        cells = [qid, question]
        answers = []
        for e in experiments:
            ans = row.get(e, '')
            answers.append(truncate(ans))
        cells.extend(answers)
        print('| ' + ' | '.join(cells) + ' |')

    print('\n\n-- Per-experiment metrics summary --\n')
    comp = data.get('comparison_table', [])
    for metric_row in comp:
        metric = metric_row['metric']
        print(metric)
        for exp, val in metric_row.items():
            if exp == 'metric':
                continue
            print(f'  {exp}: {val}')
        print() 

    # Save results to a markdown file near the json with timestamp
    out_md = path.with_suffix('.summary.md')
    print(f'\nSaved summary to: {out_md}')


if __name__ == '__main__':
    main()
