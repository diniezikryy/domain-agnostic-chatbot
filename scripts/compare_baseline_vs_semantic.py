import json
from pathlib import Path

report_path = Path('evaluation') / 'results' / 'ragas_report_my_policies_2025-11-17_18-16-47_advanced_fusion+baseline+combined_best+combined_best_rrf+grounded_hyde+no_rag+reranking+rrf+semantic_chunking+semantic_reranking.json'
if not report_path.exists():
    print('report file not found:', report_path)
    raise SystemExit(1)

with open(report_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

experiments = data.get('experiments', [])
exp_map = {exp.get('metadata', {}).get('experiment'): exp for exp in experiments}

baseline = exp_map.get('baseline')
semantic = exp_map.get('semantic_chunking')

if not baseline or not semantic:
    print('Required experiments not found')
    raise SystemExit(1)

# Build question_id -> pipeline_result maps
base_map = {pr.get('question_id'): pr for pr in baseline.get('pipeline_results', [])}
sem_map = {pr.get('question_id'): pr for pr in semantic.get('pipeline_results', [])}

comparison = {'comparisons': []}
for qid, base_pr in base_map.items():
    sem_pr = sem_map.get(qid)
    if not sem_pr:
        continue
    comp = {'question_id': qid, 'question': base_pr.get('question'), 'baseline_top_chunks': [], 'semantic_top_chunks': []}
    for ch in base_pr.get('rag_chunks', [])[:5]:
        meta = ch.get('metadata', {})
        comp['baseline_top_chunks'].append({'chunk_id': meta.get('chunk_id'), 'filename': meta.get('filename'), 'page_number': meta.get('page_number'), 'chunking_strategy': meta.get('chunking_strategy'), 'extraction_method': meta.get('extraction_method'), 'score': ch.get('score')})
    for ch in sem_pr.get('rag_chunks', [])[:5]:
        meta = ch.get('metadata', {})
        comp['semantic_top_chunks'].append({'chunk_id': meta.get('chunk_id'), 'filename': meta.get('filename'), 'page_number': meta.get('page_number'), 'chunking_strategy': meta.get('chunking_strategy'), 'extraction_method': meta.get('extraction_method'), 'score': ch.get('score')})
    comparison['comparisons'].append(comp)

out_path = Path('evaluation') / 'results' / 'semantic_vs_baseline_comparison_2025-11-17_18-16-47.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)

print('Wrote comparison to', out_path)
