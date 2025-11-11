"""
RAGAS Evaluation Harness
Orchestrates evaluation of RAG pipeline with 4 experiments: baseline, no_rag, reranking, HyDE.
Uses gpt-4o-mini for RAGAS metrics to reduce costs 10x.

Usage:
  python run_evaluation.py --experiment baseline --batch_id my_policies
  python run_evaluation.py --experiment reranking --batch_id my_policies
  python run_evaluation.py --experiment hyde --batch_id my_policies
  python run_evaluation.py --experiment no_rag --batch_id my_policies
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from flashrank import Ranker
from langchain_openai import ChatOpenAI

# Import your application classes
from query_processor import QueryProcessor
from batch_manager import BatchManager


# =========================================================================
# CONFIGURATION
# =========================================================================

EVAL_DATASET_PATH = "test_data/evaluation_dataset.json"
TEST_PROFILES_DIR = "test_data"

# IMPORTANT: Update this to match your test user's batch_id
# First, register a user in the UI, upload the 3 PDFs, then find the batch_id in batches/
DEFAULT_TEST_BATCH_ID = "my_policies"


# =========================================================================
# DATA LOADING
# =========================================================================

def load_data() -> tuple[List[Dict], Dict]:
    """Loads the golden set and test profiles."""
    with open(EVAL_DATASET_PATH, 'r') as f:
        golden_set = json.load(f)
    
    profiles = {}
    for item in golden_set:
        profile_id = item.get("user_profile_id")
        if profile_id and profile_id not in profiles:
            profile_path = Path(TEST_PROFILES_DIR) / f"{profile_id}.json"
            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    profiles[profile_id] = json.load(f)
    
    print(f"Loaded {len(golden_set)} test questions and {len(profiles)} user profiles.")
    return golden_set, profiles


# =========================================================================
# EXPERIMENTAL PIPELINE FUNCTIONS
# =========================================================================

def run_retrieval_baseline(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Runs the default retrieval pipeline."""
    return query_processor.run_retrieval(query, batch_id, user_profile)


def run_generation_baseline(
    query_processor: QueryProcessor,
    query: str,
    retrieval_data: Dict,
    user_profile: Optional[Dict]
) -> str:
    """Runs the default generation pipeline."""
    return query_processor.run_generation(
        query=query,
        rag_chunks=retrieval_data["rag_chunks_details"],
        research_results=retrieval_data["web_research_raw"],
        user_profile=user_profile
    )


def run_generation_no_rag(
    query_processor: QueryProcessor,
    query: str,
    retrieval_data: Dict,
    user_profile: Optional[Dict]
) -> str:
    """Experiment 1: Runs generation with NO RAG context."""
    print("--- [EVAL] Running Generation (NO_RAG) ---")
    # Pass empty contexts to force LLM-only response
    return query_processor.run_generation(
        query=query,
        rag_chunks=[],
        research_results={},
        user_profile=user_profile
    )


def run_retrieval_rerank(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Experiment 2: Runs retrieval and adds a re-ranking step using FlashRank."""
    print("--- [EVAL] Running Retrieval (RE-RANKING) ---")
    retrieval_data = query_processor.run_retrieval(query, batch_id, user_profile)
    
    # Initialize re-ranker (cached after first use)
    if not hasattr(query_processor, 'reranker'):
        print("Initializing FlashRank re-ranker (first run)...")
        query_processor.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".flashrank_cache")
    
    # Re-rank the document chunks
    rag_chunks = retrieval_data["rag_chunks_details"]
    if rag_chunks:
        passages = [{"id": i, "text": chunk["content"]} for i, chunk in enumerate(rag_chunks)]
        reranked = query_processor.reranker.rerank(query=query, passages=passages)
        
        # Keep top 5 reranked chunks
        reranked_indices = {r['id'] for r in reranked[:5]}
        final_rag_chunks = [c for i, c in enumerate(rag_chunks) if i in reranked_indices]
        
        # Update retrieval data with reranked chunks
        retrieval_data["rag_chunks_details"] = final_rag_chunks
        retrieval_data["rag_contexts_list"] = [c["content"] for c in final_rag_chunks]
        print(f"Re-ranked from {len(rag_chunks)} to {len(final_rag_chunks)} chunks.")
    
    return retrieval_data


def run_retrieval_hyde(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Experiment 3: Runs retrieval using HyDE (Hypothetical Document Embeddings)."""
    print("--- [EVAL] Running Retrieval (HyDE) ---")
    
    # 1. Generate Hypothetical Answer
    hyde_prompt = f"""Write a short, hypothetical answer to the following question. 
Do not say you don't know. Be specific and detailed.

Question: {query}"""
    
    hyde_response = query_processor.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": hyde_prompt}],
        max_tokens=150,
        temperature=0.0
    )
    hyde_answer = hyde_response.choices[0].message.content
    print(f"HyDE Answer: {hyde_answer[:100]}...")
    
    # 2. Run retrieval using the hypothetical answer as query
    # This leverages semantic search to find documents that match the hypothetical response
    retrieval_data = query_processor.run_retrieval(hyde_answer, batch_id, user_profile)
    
    return retrieval_data


# =========================================================================
# MAIN EVALUATION ORCHESTRATOR
# =========================================================================

def run_pipeline(
    questions_data: List[Dict],
    profiles_data: Dict,
    experiment_name: str,
    test_batch_id: str
) -> List[Dict]:
    """
    Runs the RAG pipeline for all test questions and collects results
    based on the specified experiment.
    """
    print(f"\n{'='*70}")
    print(f"Initializing RAG pipeline for experiment: {experiment_name.upper()}")
    print(f"Test batch: {test_batch_id}")
    print(f"{'='*70}\n")
    
    batch_manager = BatchManager()
    query_processor = QueryProcessor(batch_manager)
    
    # Verify the batch exists
    if not query_processor._ensure_batch_loaded(test_batch_id):
        raise Exception(f"FATAL: Could not load batch '{test_batch_id}'. "
                       f"Please ensure the batch exists in batches/{test_batch_id}/")
    
    print(f"✓ Successfully loaded test batch '{test_batch_id}'.\n")
    
    # Select the functions to run based on the experiment
    if experiment_name == "baseline":
        retrieval_func = run_retrieval_baseline
        generation_func = run_generation_baseline
    elif experiment_name == "no_rag":
        retrieval_func = run_retrieval_baseline
        generation_func = run_generation_no_rag
    elif experiment_name == "reranking":
        retrieval_func = run_retrieval_rerank
        generation_func = run_generation_baseline
    elif experiment_name == "hyde":
        retrieval_func = run_retrieval_hyde
        generation_func = run_generation_baseline
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    results = []
    
    for item in questions_data:
        question = item["question"]
        ground_truth = item["ground_truth"]
        profile_id = item.get("user_profile_id")
        user_profile = profiles_data.get(profile_id, {})
        question_id = item.get("question_id", "unknown")
        
        print(f"\n--- Processing Question: {question_id} ---")
        print(f"Q: {question}")
        
        try:
            # 1. Run Retrieval
            retrieval_data = retrieval_func(query_processor, question, test_batch_id, user_profile)
            
            # Collate all contexts for RAGAS
            all_contexts = retrieval_data["rag_contexts_list"] + retrieval_data["web_contexts_list"]
            
            # For no_rag experiment, explicitly set contexts to empty
            if experiment_name == "no_rag":
                all_contexts = []
            
            if not all_contexts:
                print("⚠ Warning: No context was retrieved.")
            else:
                print(f"✓ Retrieved {len(all_contexts)} context chunks.")
            
            # 2. Run Generation
            generated_answer = generation_func(query_processor, question, retrieval_data, user_profile)
            print(f"A: {generated_answer[:100]}...")
            
            # 3. Store results for RAGAS
            results.append({
                "question": question,
                "answer": generated_answer,
                "contexts": all_contexts,
                "ground_truth": ground_truth,
                "question_id": question_id,
            })
        
        except Exception as e:
            print(f"✗ Error processing question: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next question
            continue
    
    return results


# =========================================================================
# RAGAS EVALUATION & REPORTING
# =========================================================================

def run_ragas_evaluation(results: List[Dict], experiment_name: str, batch_id: str) -> Dict:
    """Runs RAGAS metrics on the collected results."""
    print(f"\n{'='*70}")
    print("Running RAGAS Evaluation Metrics...")
    print(f"{'='*70}\n")
    
    if not results:
        print("ERROR: No results to evaluate!")
        return {}
    
    # Create dataset in RAGAS format
    dataset = Dataset.from_list(results)
    
    # Define metrics
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]
    
    print(f"Evaluating {len(results)} Q&A pairs with {len(metrics)} metrics...")
    print("(This will make LLM calls to gpt-4o-mini for cost-effective evaluation)\n")
    
    # Configure gpt-4o-mini for cost efficiency
    # Note: RAGAS will use this model for metric evaluation
    try:
        evaluation_result = evaluate(
            dataset,
            metrics=metrics,
            llm=ChatOpenAI(model="gpt-4o-mini"),  # Cost-effective evaluator
        )
        return evaluation_result
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {}


def save_results(
    evaluation_result: Dict,
    pipeline_results: List[Dict],
    experiment_name: str,
    batch_id: str
) -> str:
    """Saves evaluation results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"evaluation/results/ragas_{experiment_name}_{batch_id}_{timestamp}.json"
    
    # Ensure output directory exists
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare output data
    output_data = {
        "metadata": {
            "experiment": experiment_name,
            "batch_id": batch_id,
            "timestamp": timestamp,
            "num_questions": len(pipeline_results),
        },
        "ragas_metrics": evaluation_result.to_dict() if hasattr(evaluation_result, 'to_dict') else {},
        "pipeline_results": pipeline_results,
    }
    
    # Save to JSON
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_filename}")
    return output_filename


def print_metrics_summary(evaluation_result: Dict, experiment_name: str):
    """Prints a summary of RAGAS metrics."""
    print(f"\n{'='*70}")
    print(f"RAGAS Evaluation Results: {experiment_name.upper()}")
    print(f"{'='*70}\n")
    
    if not evaluation_result or not hasattr(evaluation_result, 'to_pandas'):
        print("No metrics to display.")
        return
    
    df = evaluation_result.to_pandas()
    
    # Print column names
    print("Metrics Columns:")
    print(df.columns.tolist())
    print()
    
    # Print summary statistics for each metric
    print("Summary Statistics:")
    print(df.describe().to_string())
    print()
    
    # Print detailed results
    print("Detailed Results:")
    print(df.to_string())


# =========================================================================
# MAIN ENTRY POINT
# =========================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation Harness for RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --experiment baseline
  python run_evaluation.py --experiment reranking --batch_id my_policies
  python run_evaluation.py --experiment hyde --batch_id my_policies_large
  python run_evaluation.py --experiment no_rag --batch_id my_policies
        """
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=["baseline", "no_rag", "reranking", "hyde"],
        help="The experiment to run (default: baseline)"
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        default=DEFAULT_TEST_BATCH_ID,
        help=f"The test batch ID to use (default: {DEFAULT_TEST_BATCH_ID})"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set in .env file")
    
    print(f"\n🚀 RAGAS Evaluation Harness")
    print(f"Experiment: {args.experiment.upper()}")
    print(f"Batch ID: {args.batch_id}")
    
    # Load data
    questions, profiles = load_data()
    
    # Run pipeline
    pipeline_results = run_pipeline(questions, profiles, args.experiment, args.batch_id)
    
    # Run RAGAS evaluation
    evaluation_result = run_ragas_evaluation(pipeline_results, args.experiment, args.batch_id)
    
    # Print and save results
    print_metrics_summary(evaluation_result, args.experiment)
    output_file = save_results(evaluation_result, pipeline_results, args.experiment, args.batch_id)
    
    print(f"\n✓ Evaluation complete!")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
