"""
Analyze the custom dataset evaluation results
"""
import json
import numpy as np
import csv
import textwrap
from pathlib import Path

def main():
    report_file = Path('evaluation/results/ragas_report_my_policies_2025-11-18_15-55-58_advanced_fusion+baseline+combined_best_rrf+rrf_reranking.json')
    
    with open(report_file) as f:
        data = json.load(f)
    
    experiments = data.get('experiments', [])
    
    print('='*100)
    print('EVALUATION RESULTS: CUSTOM USER QUESTIONS DATASET')
    print('Dataset: test_data/evaluation_dataset_custom_user_questions.json')
    print('Questions: 6 (AUTO_013 to AUTO_018)')
    print('Experiments: baseline, advanced_fusion, rrf_reranking, combined_best_rrf')
    print('='*100)
    
    # Map experiment order to names
    exp_names = ['baseline', 'advanced_fusion', 'rrf_reranking', 'combined_best_rrf']
    
    results = []
    for i, exp in enumerate(experiments):
        metrics = exp.get('ragas_metrics', {})
        
        exp_name = exp_names[i] if i < len(exp_names) else f'experiment_{i}'
        
        result = {
            'Experiment': exp_name,
            'Faithfulness': np.mean(metrics.get('faithfulness', [0])),
            'Answer_Relevancy': np.mean(metrics.get('answer_relevancy', [0])),
            'Context_Precision': np.mean(metrics.get('context_precision', [0])),
            'Context_Recall': np.mean(metrics.get('context_recall', [0])),
            'Answer_Correctness': np.mean(metrics.get('answer_correctness', [0]))
        }
        results.append(result)
    
    # Print formatted table
    print('\nMETRIC MEANS (Averaged across 6 questions):')
    print('-'*100)
    print(f'{"Experiment":<20} {"Faithful":<12} {"Ans_Relev":<12} {"Ctx_Prec":<12} {"Ctx_Recall":<12} {"Ans_Correct":<12}')
    print('-'*100)
    for r in results:
        print(f'{r["Experiment"]:<20} {r["Faithfulness"]:<12.4f} {r["Answer_Relevancy"]:<12.4f} {r["Context_Precision"]:<12.4f} {r["Context_Recall"]:<12.4f} {r["Answer_Correctness"]:<12.4f}')
    
    # Rankings
    print('\n' + '='*100)
    print('RANKINGS BY KEY METRICS (Higher is Better):')
    print('='*100)
    
    for metric_key, metric_name in [
        ('Faithfulness', 'Faithfulness (Answer is grounded in contexts)'),
        ('Answer_Relevancy', 'Answer Relevancy (Answer addresses the question)'),
        ('Context_Precision', 'Context Precision (Retrieved contexts are relevant)'),
        ('Answer_Correctness', 'Answer Correctness (Matches ground truth)')
    ]:
        print(f'\n{metric_name}:')
        ranked = sorted(results, key=lambda x: x[metric_key], reverse=True)
        for i, r in enumerate(ranked, 1):
            emoji = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '  '
            print(f'  {emoji} {i}. {r["Experiment"]:<20} = {r[metric_key]:.4f}')
    
    # Analysis
    print('\n' + '='*100)
    print('CRITICAL FINDINGS: (computed from experiment means)')
    print('='*100)

    # Compute averages across experiments for a quick, aggregated insight
    metric_keys = ['Faithfulness', 'Answer_Relevancy', 'Context_Precision', 'Context_Recall', 'Answer_Correctness']
    metric_avgs = {k: np.mean([r[k] for r in results if r.get(k) is not None]) for k in metric_keys}

    print('\n📊 AVERAGE METRICS (across experiments):')
    for k, v in metric_avgs.items():
        print(f"  - {k}: {v:.3f}")

    # Interpret the context precision and recall
    cp = metric_avgs['Context_Precision']
    cr = metric_avgs['Context_Recall']
    ac = metric_avgs['Answer_Correctness']

    if cp < 0.5:
        print('\n🔴 CONTEXT PRECISION IS LOW: retrieved contexts are noisy; add reranking to improve precision')
    elif cp < 0.8:
        print('\n🟡 CONTEXT PRECISION is moderate; check reranker or fusion weighting to reduce noise')
    else:
        print('\n✅ CONTEXT PRECISION is high; retrieved contexts are very relevant')

    if cr < 0.5:
        print('🔴 CONTEXT RECALL IS LOW: the pipeline is not finding the gold evidence chunks — fix indexing or chunk mapping')
    else:
        print('✅ CONTEXT RECALL is acceptable; relevant chunks are being retrieved')

    if ac < 0.5:
        print('🔴 ANSWER CORRECTNESS is low — generator is producing incorrect answers, likely due to retrieval errors or hallucinations')
    else:
        print('✅ ANSWER CORRECTNESS is moderate-to-good in aggregate')
    
    print('\n' + '='*100)
    print('WHAT THIS MEANS:')
    print('='*100)
    
    print('\n1. RETRIEVAL HEALTH (context recall, precision)')
    print(f"   → Averaged context precision: {metric_avgs['Context_Precision']:.3f}")
    print(f"   → Averaged context recall: {metric_avgs['Context_Recall']:.3f}")
    print('   → If recall is low for a question, the gold context was not returned by the retriever; this is often caused by mismatched chunk_ids or chunking differences')
    print('   → Fix: Verify chunk_ids in reference_contexts match actual batch chunks (script: scripts/populate_reference_contexts.py)')
    
    print('\n2. THE LLM GENERATES PLAUSIBLE BUT WRONG ANSWERS')
    print('   → Answer_Relevancy is high (questions are addressed)')
    print('   → Faithfulness is moderate (answers cite the contexts given)')
    print('   → BUT Answer_Correctness is terrible because the contexts were wrong')
    print('   → This is the "Lost in the Middle" problem - LLM given 8 chunks, only 1-2 relevant')
    
    print('\n3. NO EXPERIMENT WINS ACROSS THE BOARD')
    print('   → Baseline: Best faithfulness (0.69) but worst answer correctness (0.18)')
    print('   → Advanced_fusion: Best answer correctness (0.30) but mid-tier on others')
    print('   → RRF_reranking/Combined_best_rrf: Best answer relevancy (0.76) but lower faithfulness')
    
    print('\n' + '='*100)
    print('RECOMMENDED FIXES (Priority Order):')
    print('='*100)
    
    print('\n🔴 PRIORITY 1: Fix Context Recall (Currently 0.0)')
    print('   Action: Verify reference_contexts chunk_ids are correct')
    print('   - Run: python -c "import pickle; data=pickle.load(open(\'batches/my_policies/faiss_index/index.pkl\',\'rb\')); print([m for m in data[\'metadata\'][:5]])"')
    print('   - Compare chunk_ids in dataset vs actual batch metadata')
    print('   - If mismatch, regenerate reference_contexts with correct chunk identifiers')
    
    print('\n🔴 PRIORITY 2: Fix Context Precision (Currently 0.22)')
    print('   Action: Add cross-encoder reranking to ALL experiments')
    print('   - Baseline should use reranking by default')
    print('   - Set RERANK_KEEP_TOP_N=3 (not 8) to reduce noise')
    print('   - Expected improvement: Context_Precision should rise to 0.60-0.80')
    
    print('\n🟡 PRIORITY 3: Improve Answer Correctness')
    print('   Action: After fixing retrieval, strengthen generation prompt')
    print('   - Set temperature=0.0 for deterministic output')
    print('   - Add explicit instruction: "If documents don\'t support an answer, say \'I don\'t know\'"')
    print('   - Force source citations: "You MUST cite [Source N] for every claim"')
    
    print('\n🟢 PRIORITY 4: Validate with Small Test')
    print('   Action: Pick 1 question (AUTO_015 - flight delay) and debug end-to-end')
    print('   - Manually verify the TravelCare page 4 chunk is in the batch')
    print('   - Run retrieval and check if it appears in top-8')
    print('   - If not, investigate: embedding quality? BM25 tokenization? Query expansion?')
    
    print('\n' + '='*100)
    print('NEXT STEPS:')
    print('='*100)
    
    # --- Per-question Debugging Output ---
    print('\n' + '='*100)
    print('PER-QUESTION DETAILS (per experiment)')
    print('='*100)

    # Load dataset to read canonical reference_chunk_ids (if present)
    dataset_path = Path(data.get('metadata', {}).get('dataset') or 'test_data/evaluation_dataset_custom_user_questions.refined.json')
    dataset_items = {}
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            ds = json.load(f)
            for q in ds:
                dataset_items[q.get('question_id')] = q
    except Exception:
        dataset_items = {}

    # CSV export optional, helpful for quick offline inspection
    csv_out = Path('evaluation/results/per_question_summary.csv')
    rows = []

    for exp_idx, exp in enumerate(experiments):
        exp_name = exp_names[exp_idx] if exp_idx < len(exp_names) else f'exp_{exp_idx}'
        ragas_metrics = exp.get('ragas_metrics', {})
        pipeline = exp.get('pipeline_results', [])
        local_metrics = {m.get('question_id'): m for m in exp.get('local_metrics', [])}

        # per-question values: each metric is a list of numbers in the same order
        metric_names = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']

        for i, p in enumerate(pipeline):
            qid = p.get('question_id') or f'Q{i}'
            qtext = p.get('question', '')
            answer = p.get('answer', '')
            gt = p.get('ground_truth', '')

            # Pull ragas per-question values from lists safely
            def get_metric(name):
                arr = ragas_metrics.get(name, [])
                if i < len(arr):
                    return float(arr[i]) if arr[i] is not None else None
                return None

            faith = get_metric('faithfulness')
            ans_rel = get_metric('answer_relevancy')
            cprec = get_metric('context_precision')
            crecall = get_metric('context_recall')
            acert = get_metric('answer_correctness')

            lm = local_metrics.get(qid, {})
            context_token_recall = lm.get('context_token_recall')
            answer_token_recall = lm.get('answer_token_recall')

            # Top contexts (from rag_chunks if present)
            rag_chunks = p.get('rag_chunks', []) or []
            rag_ctxs = p.get('rag_contexts', []) or []

            top_sources = []
            if rag_chunks:
                for c in rag_chunks[:5]:
                    md = c.get('metadata', {})
                    src = md.get('filename', 'unknown')
                    page = md.get('page_number', md.get('page', 'N/A'))
                    top_sources.append(f"{src}:page_{page}")
            else:
                # use the rag_contexts strings to attempt to find filename:page markers
                for s in rag_ctxs[:5]:
                    # try to find 'Source X: filename, Page Y' pattern first
                    cnt = textwrap.shorten(s.replace('\n',' '), width=200)
                    top_sources.append(cnt)

            print(f"\nExperiment: {exp_name} | Question: {qid}")
            print(f"  Q: {qtext}")
            print(f"  GT: {gt[:240].replace('\n',' ')}")
            print(f"  Answer (truncated): {answer[:240].replace('\n',' ')}")
            print(f"  Metrics - faith={faith}, relev={ans_rel}, ctx_prec={cprec}, ctx_rec={crecall}, ans_corr={acert}")
            print(f"  Local token recall - ctx: {context_token_recall}, answer: {answer_token_recall}")
            print(f"  Top contexts: {', '.join(top_sources[:5])}")

            rows.append({
                'experiment': exp_name,
                'question_id': qid,
                'faithfulness': faith,
                'answer_relevancy': ans_rel,
                'context_precision': cprec,
                'context_recall': crecall,
                'answer_correctness': acert,
                'context_token_recall': context_token_recall,
                'answer_token_recall': answer_token_recall,
                'num_contexts': len(rag_chunks) or len(rag_ctxs),
                'top_sources': ';'.join(top_sources[:5]),
                'question': qtext,
                'ground_truth': gt,
                'answer': answer,
            })

            # Add chunk-id-based recall: did we retrieve any of the canonical
            # reference chunks by metadata.chunk_id? This helps when the
            # dataset uses chunk_id as the stable reference instead of raw text.
            ds_q = dataset_items.get(qid, {})
            ref_chunk_ids = ds_q.get('reference_chunk_ids', []) if isinstance(ds_q, dict) else []
            retrieved_chunk_ids = []
            if rag_chunks:
                for c in rag_chunks:
                    md = c.get('metadata', {}) or {}
                    cid = md.get('chunk_id')
                    if cid:
                        retrieved_chunk_ids.append(cid)

            ref_hit = False
            if ref_chunk_ids and retrieved_chunk_ids:
                ref_hit = bool(set(ref_chunk_ids) & set(retrieved_chunk_ids))

            # Add diagnostics to the CSV row
            rows[-1].update({
                'reference_chunk_ids': ';'.join(ref_chunk_ids) if ref_chunk_ids else '',
                'retrieved_chunk_ids': ';'.join(retrieved_chunk_ids[:10]) if retrieved_chunk_ids else '',
                'reference_chunk_id_retrieved': ref_hit
            })

    # write CSV
    if rows:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0].keys())
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"\nWrote per-question CSV to: {csv_out}")
        # Compute aggregated chunk-id recall metrics
        per_exp = {}
        for r in rows:
            exp = r['experiment']
            if exp not in per_exp:
                per_exp[exp] = {'total': 0, 'hits': 0}
            per_exp[exp]['total'] += 1
            if r.get('reference_chunk_id_retrieved'):
                per_exp[exp]['hits'] += 1

        print('\nAGGREGATED REFERENCE CHUNK-ID RECALL:')
        for exp, v in per_exp.items():
            rate = 0.0
            if v['total'] > 0:
                rate = v['hits'] / v['total']
            print(f"  {exp}: {v['hits']}/{v['total']} ({rate:.2%})")
    print('1. Fix reference_contexts chunk_ids (see PRIORITY 1 above)')
    print('2. Re-run evaluation: python run_evaluation.py --experiments baseline,rrf_reranking --batch_id my_policies --dataset test_data/evaluation_dataset_custom_user_questions.json --no-cache')
    print('3. Check if Context_Recall improves from 0.0')
    print('4. If still 0.0, the chunking strategy or batch creation is fundamentally broken')
    print('='*100)

if __name__ == '__main__':
    main()


