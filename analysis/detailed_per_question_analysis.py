"""
Detailed per-question analysis of RAGAS aggregated results.

This script parses an aggregated `ragas_report_...json` file (the multi-experiment report) and
produces:
 - A CSV with per-question metrics for each experiment
 - A short printed summary indicating which experiment was the best for each metric
 - For each question, a list of top-k retrieved chunks with metadata (filename, page_number, chunk_id),
   allowing you to inspect where the pipeline retrieved context from.

Usage:
  python analysis/detailed_per_question_analysis.py --input evaluation/results/ragas_report_...json --output evaluation/results/per_question_analysis.csv

"""
from pathlib import Path
import json
import argparse
from typing import List, Dict, Any
import csv

DEFAULT_INPUT = Path("evaluation/results")
# Please update this filename if you generate a new aggregated run
DEFAULT_INPUT_FILE = DEFAULT_INPUT / "ragas_report_my_policies_2025-11-18_15-55-58_advanced_fusion+baseline+combined_best_rrf+rrf_reranking.json"


def flatten_per_question(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a list of per-question dicts with metrics for each experiment.

    Each returned dict contains: question_id, question, ground_truth, user_profile_id
    plus metrics and a column per experiment for the key ragas metrics (faithfulness, answer_relevancy,
    context_precision, context_recall, answer_correctness).
    """
    # Determine question count from the first experiment
    if not experiments:
        return []

    first = experiments[0]
    pipeline = first.get('pipeline_results', [])
    num_questions = len(pipeline)

    # Initialize records for each base question
    records = []
    for i in range(num_questions):
        rec = {
            'index': i,
            'question_id': pipeline[i].get('question_id'),
            'question': pipeline[i].get('question'),
            'ground_truth': pipeline[i].get('ground_truth'),
            'user_profile_id': pipeline[i].get('user_profile_id')
        }
        records.append(rec)

    # For each experiment, fill metric columns
    for exp in experiments:
        exp_name = exp.get('metadata', {}).get('experiment', 'unknown')
        ragas = exp.get('ragas_metrics', {})
        # Each metric is an array of values matching question order
        for i, rec in enumerate(records):
            # safe index extraction
            def safe_get(arr, idx):
                try:
                    return arr[idx]
                except Exception:
                    return None
            rec[f'{exp_name}_faithfulness'] = safe_get(ragas.get('faithfulness', []), i)
            rec[f'{exp_name}_answer_relevancy'] = safe_get(ragas.get('answer_relevancy', []), i)
            rec[f'{exp_name}_context_precision'] = safe_get(ragas.get('context_precision', []), i)
            rec[f'{exp_name}_context_recall'] = safe_get(ragas.get('context_recall', []), i)
            rec[f'{exp_name}_answer_correctness'] = safe_get(ragas.get('answer_correctness', []), i)
            # Also include the top-k contexts and rag chunks for deeper inspection
            pipeline_res = exp.get('pipeline_results', [])
            if i < len(pipeline_res):
                pr = pipeline_res[i]
                # rag_chunks is a list of chunk objects with metadata
                chunks = pr.get('rag_chunks', []) or pr.get('rag_chunks', []) or pr.get('rag_chunks', [])
                # Normalize to filename:page and chunk_id if metadata exists
                chunk_summaries = []
                for ch in pr.get('rag_chunks', [])[:10]:
                    m = ch.get('metadata', {})
                    filename = m.get('filename') or m.get('source')
                    page = m.get('page_number')
                    chunk_id = m.get('chunk_id')
                    # use snippet
                    snippet = ch.get('content','').strip().replace('\n',' ')[:150]
                    chunk_summaries.append(f"{filename or 'unknown'}:p{page} ({chunk_id}) score={ch.get('combined_score') or ch.get('score') or ''} snippet={snippet}")
                rec[f'{exp_name}_top_chunks'] = ' ||| '.join(chunk_summaries)
                # Also store the contexts (rag_contexts) array as an indicator
                rec[f'{exp_name}_contexts_count'] = len(pr.get('rag_contexts', []))
                rec[f'{exp_name}_from_cache'] = pr.get('from_cache', False)
                rec[f'{exp_name}_cache_key'] = pr.get('cache_key')
                rec[f'{exp_name}_full_gt_in_context'] = exp.get('local_metrics', [])[i].get('full_ground_truth_in_context') if exp.get('local_metrics') and i < len(exp.get('local_metrics')) else None
            else:
                rec[f'{exp_name}_top_chunks'] = ''
                rec[f'{exp_name}_contexts_count'] = 0
                rec[f'{exp_name}_from_cache'] = False
                rec[f'{exp_name}_cache_key'] = None
                rec[f'{exp_name}_full_gt_in_context'] = None
    return records


def save_csv(records: List[Dict[str, Any]], out_path: Path):
    if not records:
        print("No records")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(records[0].keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            # convert any lists to strings
            for k, v in r.items():
                if isinstance(v, list):
                    r[k] = ' | '.join(str(x) for x in v)
            writer.writerow(r)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', type=Path, default=DEFAULT_INPUT_FILE)
    p.add_argument('--output', '-o', type=Path, default=Path('evaluation/results/per_question_analysis.csv'))
    args = p.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}")
        raise SystemExit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    experiments = data.get('experiments') or []
    flattened = flatten_per_question(experiments)
    save_csv(flattened, args.output)

    print(f"Wrote per-question CSV to {args.output}")
    # Print a short human-readable summary for each question
    for rec in flattened:
        print('\n' + '='*85)
        print(f"Question: {rec['question_id']} => {rec['question']}")
        # show whether the ground truth was found in the retrieved contexts
        first_exp_name = next(iter(experiments))['metadata']['experiment']
        print(f"Ground Truth present? (per first experiment) => {rec.get(first_exp_name + '_full_gt_in_context', None)}")
        # show top chunks from baseline and combined_best_rrf
        baseline = rec.get('baseline_top_chunks', '')
        combined = rec.get('combined_best_rrf_top_chunks', '')
        print('\nTop baseline chunks:')
        for ch in baseline.split(' ||| ')[:5]:
            if ch: print('  - ' + ch[:200])
        print('\nTop combined_best_rrf chunks:')
        for ch in combined.split(' ||| ')[:5]:
            if ch: print('  - ' + ch[:200])

    print('\nAnalysis complete. Refer to CSV for full per-question metrics (per experiment).')
