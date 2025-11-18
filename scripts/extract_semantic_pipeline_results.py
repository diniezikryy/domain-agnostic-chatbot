import json
from pathlib import Path

report_path = Path('evaluation') / 'results' / 'ragas_report_my_policies_2025-11-17_18-16-47_advanced_fusion+baseline+combined_best+combined_best_rrf+grounded_hyde+no_rag+reranking+rrf+semantic_chunking+semantic_reranking.json'
if not report_path.exists():
    print('report file not found:', report_path)
    raise SystemExit(1)

with open(report_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

experiments = data.get('experiments', [])

sem = None
for exp in experiments:
    meta = exp.get('metadata', {})
    if meta.get('experiment') == 'semantic_chunking':
        sem = exp
        break

if not sem:
    print('semantic_chunking experiment not found in report')
    raise SystemExit(1)

out = {
    'metadata': sem.get('metadata', {}),
    'ragas_metrics_summary': sem.get('ragas_metrics', {}).get('summary', {}),
    'pipeline_results': []
}

for pr in sem.get('pipeline_results', []):
    qid = pr.get('question_id') or pr.get('question_id') or pr.get('question', '')[:30]
    entry = {
        'question_id': pr.get('question_id'),
        'question': pr.get('question'),
        'answer': pr.get('answer'),
        'rag_chunks': []
    }
    for ch in pr.get('rag_chunks', []):
        content = ch.get('content','')
        meta = ch.get('metadata', {})
        entry['rag_chunks'].append({
            'content_preview': content[:300].replace('\n','\\n'),
            'score': ch.get('score'),
            'source': ch.get('source'),
            'rank': ch.get('rank'),
            'metadata': {
                'chunk_id': meta.get('chunk_id'),
                'filename': meta.get('filename'),
                'page_number': meta.get('page_number'),
                'chunking_strategy': meta.get('chunking_strategy'),
                'extraction_method': meta.get('extraction_method'),
                'page_heading': meta.get('page_heading'),
            }
        })
    out['pipeline_results'].append(entry)

# Save a diagnostics file
out_path = Path('evaluation') / 'results' / 'semantic_chunking_diagnostics_2025-11-17_18-16-47.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print('Wrote diagnostics to', out_path)
