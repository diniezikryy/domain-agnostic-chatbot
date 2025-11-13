import json
from pathlib import Path
import re

RESULTS_DIR = Path('evaluation/results')
EXPS = ['baseline', 'no_rag', 'reranking', 'hyde', 'semantic_chunking']

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

print('Experiment,Timestamp,retrieval_candidate_pool,use_web_research,faithfulness_mean,answer_relevancy_mean,answer_correctness_mean,context_precision_mean,context_recall_mean')
for exp in EXPS:
    if exp not in summaries:
        print(f"{exp},MISSING,,,,,")
        continue
    p = summaries[exp]['path']
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"{exp},{summaries[exp]['ts']},ERROR_READING,,,,,")
        continue
    r = data.get('ragas_metrics', {})
    summary = r.get('summary', {})
    metadata = data.get('metadata', {})
    pool = metadata.get('retrieval_candidate_pool')
    web = metadata.get('use_web_research')
    def get_mean(k):
        v = summary.get(k, {})
        return v.get('mean') if isinstance(v, dict) else None
    fm = get_mean('faithfulness')
    arm = get_mean('answer_relevancy')
    acm = get_mean('answer_correctness')
    cpm = get_mean('context_precision')
    crm = get_mean('context_recall')
    print(f"{exp},{summaries[exp]['ts']},{pool},{web},{fm},{arm},{acm},{cpm},{crm}")
