"""
RAG Pipeline Experiment Harness (emoji-free)

This script runs configurable experiments for RAG pipelines and prints
plain-text summaries suitable for CI/logging. It intentionally avoids
emoji characters to keep outputs machine- and human-friendly.
"""

import json
import time
import sys
import os
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_manager import BatchManager
from experimental_query_processor import ExperimentalQueryProcessor
from evaluation.industry_standard_evaluator import IndustryStandardEvaluator
from evaluation.trustworthiness_evaluator import TrustworthinessEvaluator
from evaluation.llm_baseline_comparator import BaselineLLMComparator
from evaluation.llm_judge_evaluator import LLMJudgeEvaluator


# Experiment configuration (kept minimal and emoji-free)
EXPERIMENT_CONFIGS = {
    "Control_1_LLM_Baseline": {
        "description": "gpt-4o-mini with no retrieval. Establishes 'no-RAG' floor performance.",
        "type": "baseline",
        "processor_class": BaselineLLMComparator,
        "batch_id": None,
        "params": {}
    },
    "Control_2_RAG_Baseline": {
        "description": "text-embedding-3-small + Hybrid Search + gpt-4o-mini.",
        "type": "rag",
        "processor_class": ExperimentalQueryProcessor,
        "batch_id": "my_policies",
        "params": {
            "generation_model": "gpt-4o-mini",
            "retrieval_strategy": "hybrid",
            "use_hyde": False,
            "use_reranking": False,
            "top_k": 5
        }
    },

    # Exp 1: Embedding model change (requires creating a batch with text-embedding-3-large)
    "Exp_1_Embedding_Large": {
        "description": "Compare embeddings: text-embedding-3-large (requires my_policies_large batch).",
        "type": "rag",
        "processor_class": ExperimentalQueryProcessor,
        "batch_id": "my_policies_large",
        "params": {
            "generation_model": "gpt-4o-mini",
            "retrieval_strategy": "hybrid",
            "use_hyde": False,
            "use_reranking": False,
            "top_k": 5,
            "embedding_model": "text-embedding-3-large"
        }
    },

    # Exp 2: HyDE query transformation
    "Exp_2_HyDE": {
        "description": "Use HyDE (hypothetical document) query transformation before retrieval.",
        "type": "rag",
        "processor_class": ExperimentalQueryProcessor,
        "batch_id": "my_policies",
        "params": {
            "generation_model": "gpt-4o-mini",
            "retrieval_strategy": "hybrid",
            "use_hyde": True,
            "use_reranking": False,
            "top_k": 5
        }
    },

    # Exp 3: Reranking enabled
    "Exp_3_Reranking": {
        "description": "Retrieve larger set and rerank results prior to generation.",
        "type": "rag",
        "processor_class": ExperimentalQueryProcessor,
        "batch_id": "my_policies",
        "params": {
            "generation_model": "gpt-4o-mini",
            "retrieval_strategy": "hybrid",
            "use_hyde": False,
            "use_reranking": True,
            "top_k": 5
        }
    },

    # Exp 5: Vector-only (FAISS) retrieval
    "Exp_5_Vector_Only": {
        "description": "FAISS-only semantic search (no BM25).",
        "type": "rag",
        "processor_class": ExperimentalQueryProcessor,
        "batch_id": "my_policies",
        "params": {
            "generation_model": "gpt-4o-mini",
            "retrieval_strategy": "vector_only",
            "use_hyde": False,
            "use_reranking": False,
            "top_k": 5
        }
    },

    # Control: LLM model upgrade test (baseline using gpt-5-mini)
    "Control_3_LLM_GPT5_Mini": {
        "description": "LLM-only baseline using gpt-5-mini to test generation model impact.",
        "type": "baseline",
        "processor_class": BaselineLLMComparator,
        "batch_id": None,
        "params": {"model": "gpt-5-mini"}
    }
}


def load_queries() -> List[Dict[str, Any]]:
    """Load test cases from the custom queries file and the 'my_policies' test file.

    This replaces the previous benchmark which included `ground_truth.json`.
    The function returns a flat list of query dicts with at least 'id' and 'query'.
    """
    queries: List[Dict[str, Any]] = []

    

    # personal policy tests (my_policies)
    try:
        with open('evaluation/test_queries_my_policies.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        policy_queries = data.get('test_cases', [])
        queries.extend(policy_queries)
        print(f"Loaded {len(policy_queries)} queries from test_queries_my_policies.json")
    except FileNotFoundError:
        print('evaluation/test_queries_my_policies.json not found.')

    if not queries:
        print('No test cases found. Exiting.')
        return []

    # Allow limiting number of queries via environment variable to reduce token usage
    import os
    max_q = os.getenv('EXPERIMENT_MAX_QUERIES')
    if max_q:
        try:
            m = int(max_q)
            if m > 0 and m < len(queries):
                # Sample m questions spread across the full set to get varied coverage
                def sample_queries(qs, k):
                    n = len(qs)
                    if k >= n:
                        return qs
                    if k == 1:
                        return [qs[n // 2]]
                    # compute evenly spaced indices over [0, n-1]
                    indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
                    # Ensure indices unique and in order
                    seen = set()
                    picked = []
                    for idx in indices:
                        if idx < 0:
                            idx = 0
                        if idx >= n:
                            idx = n - 1
                        if idx not in seen:
                            seen.add(idx)
                            picked.append(qs[idx])
                    return picked

                sampled = sample_queries(queries, m)
                print(f"Sampling {m} queries from {len(queries)} total (EXPERIMENT_MAX_QUERIES={m})")
                queries = sampled
        except ValueError:
            print(f"Invalid EXPERIMENT_MAX_QUERIES value: {max_q}")

    print(f"Total queries to run: {len(queries)}")
    return queries


def run_experiment(
    exp_name: str,
    config: dict,
    queries: List[Dict[str, Any]],
    ragas_evaluator: IndustryStandardEvaluator,
    trust_evaluator: TrustworthinessEvaluator,
    batch_manager: BatchManager,
    user_profile: dict | None = None,
    reference_batch_id: str = "my_policies",  # Default reference batch for faithfulness checks
) -> Dict[str, Any]:
    """Run a single experiment and return a summary dict."""

    print('\n' + '=' * 80)
    print(f"RUNNING: {exp_name}")
    print('=' * 80)

    # init
    processor = None
    if config['type'] == 'rag':
        # Check batch exists before attempting to run the experiment
        batch_id = config.get('batch_id')
        if batch_id and batch_id not in batch_manager.list_batches():
            print(f"SKIPPING: batch '{batch_id}' not found for experiment '{exp_name}'. Create it with setup_batch.py and re-run.")
            return {}

        # Some params (like embedding_model) are batch-level and not constructor args for the processor.
        params = dict(config.get('params', {}))
        if 'embedding_model' in params:
            print(f"Note: experiment requests embedding_model={params['embedding_model']}. Ensure the batch '{batch_id}' is created with that embedding model.")
            params.pop('embedding_model')
        processor = config['processor_class'](batch_manager, user_profile=user_profile, **params)
    elif config['type'] == 'baseline':
        # Allow baseline processors to accept params (e.g., model override)
        processor = config['processor_class'](user_profile=user_profile, **config.get('params', {}))

    if not processor:
        print(f"Could not initialize processor for {exp_name}")
        return {}

    all_query_results = []
    start_ts = time.time()

    for i, test_case in enumerate(queries, start=1):
        query = test_case.get('query', '')
        ground_truth_answer = test_case.get('ground_truth_answer')
        ground_truth_contexts = test_case.get('ground_truth_contexts')
        
        print(f"\nQuery {i}/{len(queries)}: {query[:80]}")

        result = {"query_id": test_case.get('id', f'query_{i}'), "query": query, "answer": None, "metrics": {}, "error": None}

        try:
            if config['type'] == 'rag':
                answer, tokens, latency = processor.process_query(query, config['batch_id'])
                contexts = processor.get_last_contexts()
            else:
                # Pure LLM baseline - get answer without retrieval
                baseline = processor.get_baseline_response(query)
                answer = baseline.get('response', '')
                tokens = baseline.get('tokens_used', 0)
                latency = baseline.get('time_seconds', 0.0)
                
                # Baseline gets no contexts - it must answer without retrieval
                contexts = []

            ragas_metrics = ragas_evaluator.evaluate_rag_response(
                query=query,
                answer=answer,
                retrieved_contexts=contexts,
                ground_truth_contexts=ground_truth_contexts,
                ground_truth_answer=ground_truth_answer
            )
            trust_metrics = trust_evaluator.evaluate_single_response(answer, test_case)

            result['answer'] = answer
            result['metrics'] = {**asdict(ragas_metrics), **trust_metrics, 'generation_tokens': tokens, 'latency_seconds': latency}

            # Print individual metrics instead of just RAGAS score
            print(f"Faithful: {ragas_metrics.faithfulness:.3f} | Relevance: {ragas_metrics.answer_relevance:.3f} | Precision: {ragas_metrics.context_precision:.3f} | Recall: {ragas_metrics.context_recall:.3f}")
            if ragas_metrics.answer_correctness is not None:
                print(f"Correctness: {ragas_metrics.answer_correctness:.3f} | RAGAS: {ragas_metrics.ragas_score:.3f} | Tokens: {tokens} | Time: {latency:.2f}s")
            else:
                print(f"RAGAS: {ragas_metrics.ragas_score:.3f} | Tokens: {tokens} | Time: {latency:.2f}s")

        except Exception as e:
            err = str(e)
            print(f"ERROR processing query: {err}")
            result['error'] = err

        all_query_results.append(result)

    total_time = time.time() - start_ts

    # aggregate
    valid = [r for r in all_query_results if r.get('error') is None]
    if not valid:
        print(f"No valid results for {exp_name}")
        return {}

    def avg(name: str) -> float:
        values = [r['metrics'].get(name) for r in valid if r['metrics'].get(name) is not None]
        return sum(values) / len(values) if values else 0.0

    summary = {
        'experiment_name': exp_name,
        'description': config.get('description', ''),
        'batch_id': config.get('batch_id'),
        'averages': {
            'avg_ragas_score': avg('ragas_score'),
            'avg_faithfulness': avg('faithfulness'),
            'avg_answer_relevance': avg('answer_relevance'),
            'avg_context_precision': avg('context_precision'),
            'avg_context_recall': avg('context_recall'),
            'avg_answer_correctness': avg('answer_correctness'),
            'avg_hallucination_score': avg('hallucination_score'),
            'avg_latency_seconds': avg('latency_seconds'),
            'avg_generation_tokens': avg('generation_tokens')
        },
        'total_time_seconds': total_time,
        'queries_run': len(valid),
        'queries_failed': len(all_query_results) - len(valid),
        'individual_query_results': all_query_results,
        'params': config.get('params', {})
    }

    print('\n' + '-' * 80)
    print(f"COMPLETE - {exp_name}")
    print(f"Faithfulness: {summary['averages']['avg_faithfulness']:.4f} | Relevance: {summary['averages']['avg_answer_relevance']:.4f}")
    print(f"Precision: {summary['averages']['avg_context_precision']:.4f} | Recall: {summary['averages']['avg_context_recall']:.4f}")
    if summary['averages']['avg_answer_correctness'] > 0:
        print(f"Answer Correctness: {summary['averages']['avg_answer_correctness']:.4f}")
    print(f"RAGAS: {summary['averages']['avg_ragas_score']:.4f} | Avg Latency: {summary['averages']['avg_latency_seconds']:.2f}s | Avg Tokens: {summary['averages']['avg_generation_tokens']:.0f}")
    print('-' * 80)

    return summary


def print_final_report(all_results: List[Dict[str, Any]]):
    if not all_results:
        print('No experiment results to report.')
        return

    # sort by ragas
    all_results.sort(key=lambda r: r['averages']['avg_ragas_score'], reverse=True)

    print('\n' + '=' * 160)
    print(f"{'Rank':<6}{'Experiment':<30}{'Faithful':<10}{'Relevance':<10}{'Precision':<10}{'Recall':<10}{'Correct.':<10}{'RAGAS':<10}{'Halluc.':<10}{'Latency':<10}{'Tokens':<10}")
    print('-' * 160)

    for i, res in enumerate(all_results, start=1):
        avg = res['averages']
        rank_label = f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th"
        correctness_str = f"{avg.get('avg_answer_correctness', 0.0):.4f}" if avg.get('avg_answer_correctness', 0) > 0 else "N/A"
        print(f"{rank_label:<6}{res['experiment_name']:<30}{avg['avg_faithfulness']:<10.4f}{avg['avg_answer_relevance']:<10.4f}{avg['avg_context_precision']:<10.4f}{avg['avg_context_recall']:<10.4f}{correctness_str:<10}{avg['avg_ragas_score']:<10.4f}{avg.get('avg_hallucination_score', 0):<10.4f}{avg['avg_latency_seconds']:<10.2f}{avg['avg_generation_tokens']:<10.0f}")

    print('=' * 160)

    winner = all_results[0]
    print(f"\nBEST CONFIGURATION: {winner['experiment_name']}")
    print(f"Description: {winner.get('description','')}")
    print(f"\nKey Metrics:")
    print(f"  Faithfulness: {winner['averages']['avg_faithfulness']:.4f}")
    print(f"  Answer Relevance: {winner['averages']['avg_answer_relevance']:.4f}")
    print(f"  Context Precision: {winner['averages']['avg_context_precision']:.4f}")
    print(f"  Context Recall: {winner['averages']['avg_context_recall']:.4f}")
    if winner['averages'].get('avg_answer_correctness', 0) > 0:
        print(f"  Answer Correctness: {winner['averages']['avg_answer_correctness']:.4f}")
    print(f"  Overall RAGAS Score: {winner['averages']['avg_ragas_score']:.4f}")


def main():
    print('\n' + '=' * 120)
    print('RAG PIPELINE EXPERIMENT HARNESS')
    print('=' * 120)

    # Load user profile for personalization
    user_profile = None
    profile_path = Path('user_profile.json')
    if profile_path.exists():
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                user_profile = json.load(f)
            print(f"Loaded user profile: {user_profile.get('name', 'Unknown')}")
        except Exception as e:
            print(f"Warning: Could not load user profile: {e}")
    else:
        print("Warning: user_profile.json not found. Running without personalization context.")

    queries = load_queries()
    if not queries:
        return

    print('Initializing evaluators...')
    batch_manager = BatchManager()
    ragas_evaluator = IndustryStandardEvaluator(use_llm_judge=True)
    trust_evaluator = TrustworthinessEvaluator(None, None)
    judge = LLMJudgeEvaluator(user_profile=user_profile)
    print('Evaluators ready')

    if 'my_policies' not in batch_manager.list_batches():
        print("ERROR: 'my_policies' batch not found. Create it with: python setup_batch.py my_policies --source documents/my_policies")
        return
    
    # Reference batch for faithfulness evaluation of pure LLM baselines
    reference_batch_id = "my_policies"

    all_results = []
    all_query_answers = {}  # Collect all experiment answers per query for judge evaluation

    for name, cfg in EXPERIMENT_CONFIGS.items():
        s = run_experiment(
            name, 
            cfg, 
            queries, 
            ragas_evaluator, 
            trust_evaluator, 
            batch_manager, 
            user_profile=user_profile,
            reference_batch_id=reference_batch_id
        )
        if s:
            all_results.append(s)
            
            # Collect answers for LLM judge
            for q_result in s.get('individual_query_results', []):
                query_id = q_result.get('query_id', '')
                if query_id not in all_query_answers:
                    all_query_answers[query_id] = {
                        'query': q_result.get('query', ''),
                        'experiments': {}
                    }
                all_query_answers[query_id]['experiments'][name] = {
                    'answer': q_result.get('answer', ''),
                    'metrics': q_result.get('metrics', {})
                }
        time.sleep(0.5)

    if not all_results:
        print('No experiments completed successfully.')
        return

    print_final_report(all_results)

    # Run LLM judge on collected answers
    print('\n' + '=' * 120)
    print('LLM JUDGE EVALUATION')
    print('=' * 120)
    
    judge_report = {
        'user_profile': user_profile,
        'query_rankings': []
    }
    
    for query_id, query_data in all_query_answers.items():
        query_text = query_data['query']
        answers_dict = {exp_name: exp_data['answer'] for exp_name, exp_data in query_data['experiments'].items()}
        
        try:
            # Judge the answers for this query
            judge_result = judge.judge_answers(
                query=query_text,
                answers=answers_dict,
                retrieved_contexts=[]  # Could be enhanced with actual contexts per experiment
            )
            
            judge_report['query_rankings'].append({
                'query_id': query_id,
                'query': query_text,
                'judge_result': judge_result
            })
            
            print(f"\nQuery: {query_text[:100]}...")
            if judge_result and 'winner' in judge_result:
                print(f"LLM Judge Winner: {judge_result['winner']}")
                if 'reasoning' in judge_result:
                    print(f"Reasoning: {judge_result['reasoning'][:200]}...")
        except Exception as e:
            print(f"Error judging query {query_id}: {e}")

    Path('evaluation/results').mkdir(parents=True, exist_ok=True)
    out = f"evaluation/results/experiment_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    judge_out = f"evaluation/results/judge_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(judge_out, 'w', encoding='utf-8') as f:
        json.dump(judge_report, f, indent=2)

    print(f"\nDetailed results saved to: {out}")
    print(f"LLM Judge report saved to: {judge_out}")
    print('Experiment harness complete')


if __name__ == '__main__':
    main()
