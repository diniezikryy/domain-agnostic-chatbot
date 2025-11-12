import pandas as pd
import json
import os
from pathlib import Path

# Load and compare recent evaluation results
results_dir = Path('evaluation/results')

experiments = ['baseline', 'no_rag', 'reranking', 'hyde', 'semantic_chunking']
batch_ids = ['my_policies', 'my_policies_semantic', 'my_policies_large']

print('=== EXPERIMENT COMPARISON SUMMARY ===\n')

# Find the most recent results for each experiment
for exp in experiments:
    print(f'## {exp.upper()} EXPERIMENT')

    # Find CSV files for this experiment
    csv_files = list(results_dir.glob(f'local_metrics_{exp}_*.csv'))
    if csv_files:
        # Get the most recent file
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_csv)

        # Calculate averages
        avg_context_recall = df['context_token_recall'].mean()
        avg_answer_recall = df['answer_token_recall'].mean()
        full_ground_truth_answers = df['full_ground_truth_in_answer'].sum()

        print(f'  File: {latest_csv.name}')
        print(f'  Avg Context Recall: {avg_context_recall:.3f}')
        print(f'  Avg Answer Recall: {avg_answer_recall:.3f}')
        print(f'  Perfect Answers: {full_ground_truth_answers}/{len(df)}')

    # Check for RAGAS JSON results
    json_files = list(results_dir.glob(f'ragas_{exp}_*.json'))
    if json_files:
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        try:
            with open(latest_json, 'r') as f:
                ragas_data = json.load(f)
                if 'ragas_metrics' in ragas_data and ragas_data['ragas_metrics']:
                    metrics = ragas_data['ragas_metrics']
                    print(f'  Faithfulness: {metrics.get("faithfulness", "N/A")}')
                    print(f'  Answer Correctness: {metrics.get("answer_correctness", "N/A")}')
                    print(f'  Context Precision: {metrics.get("context_precision", "N/A")}')
                    print(f'  Context Recall: {metrics.get("context_recall", "N/A")}')
        except Exception as e:
            print(f'  Error reading RAGAS metrics: {e}')

    print()