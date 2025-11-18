import json,sys
from pathlib import Path

if len(sys.argv) < 2:
    print('Usage: extract_top_experiments.py <path>')
    sys.exit(2)

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding='utf-8'))
experiments = ['baseline','reranking','rrf_reranking','advanced_fusion','semantic_reranking']

print('Comparing answers across these experiments: ', experiments)
print('\n')
for item in data['response_table']:
    qid = item['question_id']
    question = item['question']
    print('Question:', qid)
    print(question)
    for e in experiments:
        ans = item.get(e,'')
        print('-'*80)
        print(e, '\n', ans[:1000], '\n')
    # show if all same
    answers = [item.get(e,'').strip() for e in experiments]
    uniq = len(set(answers))
    print('Unique answers across experiments of selected set:', uniq)
    print('\n' + '='*120 + '\n')

path_out = path.with_suffix('.selected.md')
print('Saved to', path_out)
path_out.write_text('')
