"""
RAGAS Evaluation Harness
Orchestrates evaluation of RAG pipeline with experiments including baseline, no_rag, reranking, HyDE and semantic_chunking.
Uses gpt-4o-mini for RAGAS metrics to reduce costs.

Usage:
  python run_evaluation.py --experiment baseline --batch_id my_policies
  python run_evaluation.py --experiment reranking --batch_id my_policies
  python run_evaluation.py --experiment hyde --batch_id my_policies_large
  python run_evaluation.py --experiment no_rag --batch_id my_policies
  python run_evaluation.py --experiment semantic_chunking --batch_id my_policies
"""

import json
import os
import sys
import argparse
import time
import re
import csv
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
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI

# Import your application classes
from query_processor import QueryProcessor
from batch_manager import BatchManager


MAX_CONTEXTS_FOR_RAGAS = 8
# Reranker configuration: default number of top-ranked chunks to keep
RERANK_KEEP_TOP_N = int(os.getenv("RERANK_KEEP_TOP_N", "5"))

# Number of top retrieval chunks to keep when generating answers (prevents confounding
# improvements due to different candidate set sizes between baseline and re-ranking)
GENERATION_TOP_K = 5


# Simple manager to encapsulate FlashRank initialization and calls.
# This replaces the previous pattern of attaching the reranker to the function
# object and centralizes error handling.
class RerankerManager:
    def __init__(self):
        self._initialized = False
        self.ranker = None
        self.RerankRequest = None

    def init(self):
        if self._initialized:
            return
        try:
            from flashrank import Ranker, RerankRequest
            # keep a small local cache dir to avoid re-downloading models repeatedly
            self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".flashrank_cache")
            self.RerankRequest = RerankRequest
            self._initialized = True
        except Exception as e:
            # bubble up so callers can handle fallback
            raise

    def rerank(self, query, passages):
        self.init()
        request = self.RerankRequest(query=query, passages=passages)
        result = self.ranker.rerank(request)
        try:
            return list(result)
        except Exception:
            return result


reranker_manager = RerankerManager()

# =========================================================================
# CONFIGURATION
# =========================================================================

EVAL_DATASET_PATH = "test_data/evaluation_dataset.json"
TEST_PROFILES_DIR = "test_data"

# IMPORTANT: Update this to match your test user's batch_id
# First, register a user in the UI, upload the 3 PDFs, then find the batch_id in batches/
DEFAULT_TEST_BATCH_ID = "my_policies"  # Actual batch ID


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
    retrieval_data = query_processor.run_retrieval(query, batch_id, user_profile)

    # Keep only the top-K retrieved chunks for generation to avoid a confounding
    # variable where the baseline uses many more candidates than the reranker.
    try:
        retrieval_data["rag_chunks_details"] = retrieval_data.get("rag_chunks_details", [])[:GENERATION_TOP_K]
        retrieval_data["rag_contexts_list"] = retrieval_data.get("rag_contexts_list", [])[:GENERATION_TOP_K]
    except Exception:
        # Be robust to unexpected retrieval_data shapes
        pass

    return retrieval_data


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
    
    # Re-rank the document chunks (using the centralized RerankerManager)
    rag_chunks = retrieval_data.get("rag_chunks_details", [])
    if not rag_chunks:
        return retrieval_data

    passages = [{"id": i, "text": chunk.get("content", "")} for i, chunk in enumerate(rag_chunks)]

    try:
        reranked = reranker_manager.rerank(query, passages)
    except Exception as e:
        print(f"Warning: re-ranker initialization/execute failed ({e}). Skipping re-ranking step.")
        return retrieval_data

    # Ensure list
    try:
        reranked_list = list(reranked)
    except Exception:
        reranked_list = reranked if isinstance(reranked, (list, tuple)) else []

    # Map rerank info (score and rank position) back to original chunks
    rerank_info = {}
    for rank_pos, item in enumerate(reranked_list, start=1):
        # item expected to contain at least 'id' and optionally 'score'
        idx = item.get('id') if isinstance(item, dict) else None
        score = item.get('score') if isinstance(item, dict) and 'score' in item else None
        if idx is None:
            continue
        rerank_info[int(idx)] = {"rerank_score": score, "rerank_rank": rank_pos}

    # Attach rerank metadata to each chunk
    for i, chunk in enumerate(rag_chunks):
        info = rerank_info.get(i, {})
        chunk['rerank_score'] = info.get('rerank_score')
        chunk['rerank_rank'] = info.get('rerank_rank')

    # Decide how many top-ranked chunks to keep (use GENERATION_TOP_K to ensure
    # the baseline and reranking experiments operate over the same candidate set size)
    keep_n = GENERATION_TOP_K if isinstance(GENERATION_TOP_K, int) and GENERATION_TOP_K > 0 else 5

    # Build set of top ids to keep
    top_ids = [item.get('id') for item in reranked_list[:keep_n] if isinstance(item, dict) and item.get('id') is not None]
    reranked_indices = {int(i) for i in top_ids}

    final_rag_chunks = [c for i, c in enumerate(rag_chunks) if i in reranked_indices]

    # Update retrieval data with reranked chunks and provide rerank_info for debugging/analysis
    retrieval_data["rag_chunks_details"] = final_rag_chunks
    retrieval_data["rag_contexts_list"] = [c.get("content", "") for c in final_rag_chunks]
    retrieval_data["rerank_info"] = rerank_info

    print(f"Re-ranked from {len(rag_chunks)} to {len(final_rag_chunks)} chunks. (keep_n={keep_n})")
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
    
    print(f"[OK] Successfully loaded test batch '{test_batch_id}'.\n")
    
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
    elif experiment_name == "semantic_chunking":
        retrieval_func = run_retrieval_semantic_chunking
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

            # Trim contexts to avoid oversized evaluation prompts
            rag_contexts = retrieval_data["rag_contexts_list"][:MAX_CONTEXTS_FOR_RAGAS]
            web_contexts = retrieval_data["web_contexts_list"][:max(0, MAX_CONTEXTS_FOR_RAGAS - len(rag_contexts))]

            retrieval_data["rag_contexts_list"] = rag_contexts
            retrieval_data["rag_chunks_details"] = retrieval_data["rag_chunks_details"][:MAX_CONTEXTS_FOR_RAGAS]
            retrieval_data["web_contexts_list"] = web_contexts

            # Collate all contexts for RAGAS
            all_contexts = rag_contexts + web_contexts

            # For no_rag experiment, explicitly set contexts to empty
            if experiment_name == "no_rag":
                all_contexts = []

            if not all_contexts:
                print("[Warning] No context was retrieved.")
                # RAGAS requires at least one context (even if empty string) to avoid validation errors
                all_contexts = [""]
            else:
                print(f"[OK] Retrieved {len(all_contexts)} context chunks.")
            
            # 2. Run Generation
            generated_answer = generation_func(query_processor, question, retrieval_data, user_profile)
            print(f"A: {generated_answer[:100]}...")
            
            # 3. Store results for RAGAS and include retrieval metadata (rerank scores etc.)
            results.append({
                "question": question,
                "answer": generated_answer,
                "contexts": all_contexts,
                "ground_truth": ground_truth,
                "question_id": question_id,
                # include detailed retrieval info for debugging/analysis
                "rag_chunks": retrieval_data.get("rag_chunks_details", []),
                "rag_contexts": retrieval_data.get("rag_contexts_list", []),
                "web_research": retrieval_data.get("web_research_raw", {}),
                "rerank_info": retrieval_data.get("rerank_info", {}),
            })
        
        except Exception as e:
            print(f"[Error] Error processing question: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next question
            continue
    
    return results


def run_retrieval_semantic_chunking(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Experiment: Create/load a semantic-chunked batch (batch_id + '_semantic') and run retrieval against it.

    This function will create a new batch using semantic chunking if it doesn't exist yet. It is non-destructive to the original baseline batch.
    """
    print("--- [EVAL] Running Retrieval (SEMANTIC CHUNKING) ---")
    semantic_batch_id = f"{batch_id}_semantic"
    semantic_meta_path = Path("batches") / semantic_batch_id / "metadata.json"
    batch_meta_path = Path("batches") / batch_id / "metadata.json"

    # If semantic batch doesn't exist, attempt to create it using the same documents as the original batch
    if not semantic_meta_path.exists():
        if not batch_meta_path.exists():
            print(f"Original batch metadata not found: {batch_meta_path}. Falling back to baseline retrieval.")
            return run_retrieval_baseline(query_processor, query, batch_id, user_profile)

        try:
            with open(batch_meta_path, 'r') as f:
                meta = json.load(f)
            doc_entries = meta.get('documents', [])
            doc_paths = [d.get('file_path') for d in doc_entries if d.get('file_path')]

            from document_processor import DocumentProcessor
            dp = DocumentProcessor()

            print(f"Creating semantic-chunked batch '{semantic_batch_id}' from original documents...")
            success = dp.create_batch(
                batch_id=semantic_batch_id,
                document_paths=doc_paths,
                batch_name=f"{meta.get('name', semantic_batch_id)} (Semantic)",
                description="Semantic-chunked batch created for evaluation (header-aware chunking)",
                embedding_model_name="text-embedding-3-small",
                embedding_dimension=1536,
                chunking_strategy="semantic"
            )

            if not success:
                print("Failed to create semantic batch. Falling back to baseline retrieval.")
                return run_retrieval_baseline(query_processor, query, batch_id, user_profile)

            print(f"Semantic batch '{semantic_batch_id}' created.")

        except Exception as e:
            print(f"Error creating semantic batch: {e}")
            return run_retrieval_baseline(query_processor, query, batch_id, user_profile)

    # Ensure the query processor loads the semantic batch
    if not query_processor._ensure_batch_loaded(semantic_batch_id):
        print(f"Failed to load semantic batch '{semantic_batch_id}'. Falling back to baseline.")
        return run_retrieval_baseline(query_processor, query, batch_id, user_profile)

    # Run retrieval against the semantic batch
    return query_processor.run_retrieval(query, semantic_batch_id, user_profile)


# =========================================================================
# RAGAS EVALUATION & REPORTING
# =========================================================================


def compute_local_support_metrics(pipeline_results: List[Dict]) -> List[Dict]:
    """Compute lightweight lexical overlap metrics as a fast, local fallback.

    These metrics give a quick signal about whether ground-truth tokens appear in
    the retrieved contexts and generated answer without making additional LLM calls.
    """
    metrics: List[Dict[str, Any]] = []
    token_pattern = re.compile(r"\w+")

    for entry in pipeline_results:
        ground_truth = entry.get("ground_truth", "") or ""
        contexts = entry.get("contexts", []) or []
        answer = entry.get("answer", "") or ""

        tokens = [t.lower() for t in token_pattern.findall(ground_truth) if len(t) > 2]
        unique_tokens = list(dict.fromkeys(tokens))  # preserve order for reproducibility
        token_count = len(unique_tokens)

        context_text = " ".join(contexts).lower()
        answer_text = answer.lower()

        if token_count:
            context_hits = sum(1 for token in unique_tokens if token in context_text)
            answer_hits = sum(1 for token in unique_tokens if token in answer_text)
            context_recall = context_hits / token_count
            answer_recall = answer_hits / token_count
        else:
            context_hits = answer_hits = 0
            context_recall = answer_recall = 0.0

        metrics.append({
            "question_id": entry.get("question_id") or "unknown",
            "ground_truth_token_count": token_count,
            "context_token_hits": context_hits,
            "answer_token_hits": answer_hits,
            "context_token_recall": round(context_recall, 4),
            "answer_token_recall": round(answer_recall, 4),
            "full_ground_truth_in_context": any(
                ground_truth.lower() in c.lower() for c in contexts if ground_truth
            ),
            "full_ground_truth_in_answer": bool(ground_truth and ground_truth.lower() in answer_text),
        })

    return metrics


def print_local_metrics_summary(local_metrics: List[Dict[str, Any]]):
    """Print a compact summary of the local support metrics."""
    if not local_metrics:
        print("[Local Metrics] No pipeline results available for analysis.")
        return

    total = len(local_metrics)
    context_support = sum(1 for m in local_metrics if m["context_token_recall"] >= 0.3)
    answer_support = sum(1 for m in local_metrics if m["answer_token_recall"] >= 0.3)

    print(f"\n{'='*70}")
    print("Local Support Metrics (lexical overlap)")
    print(f"{'='*70}\n")
    print(f"Questions analysed: {total}")
    print(f"Context support (>=30% token recall): {context_support}/{total}")
    print(f"Answer support (>=30% token recall): {answer_support}/{total}")
    print("\nPer-question breakdown:")
    for metric in local_metrics:
        print(
            f"  - {metric['question_id']}: context_recall={metric['context_token_recall']:.2f}, "
            f"answer_recall={metric['answer_token_recall']:.2f}, "
            f"full_match_ctx={metric['full_ground_truth_in_context']}, "
            f"full_match_ans={metric['full_ground_truth_in_answer']}"
        )


def save_local_metrics_csv(
    local_metrics: List[Dict[str, Any]],
    experiment_name: str,
    batch_id: str
) -> Optional[str]:
    """Persist the local metrics to CSV for offline analysis."""
    if not local_metrics:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("evaluation/results") / f"local_metrics_{experiment_name}_{batch_id}_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(local_metrics[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(local_metrics)

    print(f"[OK] Local metrics saved to: {output_path}")
    return str(output_path)

def run_ragas_evaluation(results: List[Dict], experiment_name: str, batch_id: str) -> Any:
    """Runs RAGAS metrics on the collected results with validation and retries.

    This wrapper will attempt evaluate() up to 3 times and validate the returned
    DataFrame to ensure expected metric columns exist and are not all-NaN.
    On repeated failures it writes a debug artifact to `evaluation/results/` and returns {}.
    """
    print(f"\n{'='*70}")
    print("Running RAGAS Evaluation Metrics (safe wrapper)...")
    print(f"{'='*70}\n")

    if not results:
        print("ERROR: No results to evaluate!")
        return {}

    dataset = Dataset.from_list(results)
    # Temporarily disable expensive metrics to avoid rate limits
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    required_cols = ['faithfulness', 'answer_relevancy', 'answer_correctness']

    attempts = 3
    last_exception = None
    
    # Configure RAGAS RunConfig: reduce parallelism (max_workers) to avoid hitting rate limits
    # and keep reasonable timeouts/retries. Setting max_workers low reduces concurrent LLM calls.
    run_config = RunConfig(
        timeout=120,     # seconds for a single operation (keep current value)
        max_retries=3,   # retry attempts on transient failures
        max_wait=60,     # maximum backoff wait between retries
        max_workers=2,   # REDUCED: limit concurrent workers to 2 to lower parallel LLM load
    )
    
    for attempt in range(1, attempts + 1):
        try:
            print(f"RAGAS evaluate() attempt {attempt}/{attempts}...")
            # Use a more cost- and rate-friendly model for RAGAS evaluation to avoid TPM limits
            # Switched to gpt-4o-mini to reduce token-per-minute usage during large evaluations
            # and explicitly request a single generation (n=1) to avoid multiple-completion requests
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, n=1)
            evaluation_result = evaluate(
                dataset, 
                metrics=metrics, 
                llm=llm,
                run_config=run_config  # Pass RunConfig to control timeout and concurrency
            )

            # Basic validation: must be convertible to pandas and contain required columns
            if not hasattr(evaluation_result, 'to_pandas'):
                raise ValueError("evaluate() returned unexpected type (missing to_pandas)")

            df = evaluation_result.to_pandas()
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing metric columns in evaluation result: {missing}")

            # Check for all-NaN columns which indicate evaluator failure
            all_nan = [c for c in required_cols if df[c].isnull().all()]
            if all_nan:
                raise ValueError(f"Evaluator returned all-NaN for columns: {all_nan}")

            # Passed validation
            print("RAGAS evaluation completed and validated.")
            return evaluation_result

        except Exception as e:
            print(f"RAGAS evaluation attempt {attempt} failed: {e}")
            import traceback
            traceback.print_exc()
            last_exception = e
            # Exponential backoff before retry (5s, 10s, 20s)
            if attempt < attempts:
                backoff_time = 5 * (2 ** (attempt - 1))
                print(f"Waiting {backoff_time}s before retry...")
                time.sleep(backoff_time)
            continue

    # If we reach here, all attempts failed — write debug artifact
    debug_dir = Path("evaluation/results")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir / f"ragas_evaluator_debug_{experiment_name}_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    debug_payload = {
        "error": str(last_exception),
        "attempts": attempts,
        "num_results": len(results),
    }
    with open(debug_file, 'w') as f:
        json.dump(debug_payload, f, indent=2)

    print(f"[ERROR] RAGAS evaluation failed after {attempts} attempts. Debug saved to {debug_file}")
    return {}


def save_results(
    evaluation_result: Any,
    pipeline_results: List[Dict],
    experiment_name: str,
    batch_id: str,
    local_metrics: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Saves evaluation results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"evaluation/results/ragas_{experiment_name}_{batch_id}_{timestamp}.json"
    
    # Ensure output directory exists
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Extract RAGAS metrics properly from the Result object
    ragas_metrics = {}
    if hasattr(evaluation_result, 'to_pandas'):
        df = evaluation_result.to_pandas()
        # Convert DataFrame to dictionary with metrics as keys and lists of values
        ragas_metrics = {col: df[col].tolist() for col in df.columns if col in ['faithfulness', 'answer_relevancy', 'answer_correctness', 'context_precision', 'context_recall']}
        # Also include summary statistics
        ragas_metrics['summary'] = {col: {'mean': df[col].mean(), 'std': df[col].std(), 'min': df[col].min(), 'max': df[col].max()} 
                                   for col in ragas_metrics.keys() if col != 'summary'}
    
    # Prepare output data
    output_data = {
        "metadata": {
            "experiment": experiment_name,
            "batch_id": batch_id,
            "timestamp": timestamp,
            "num_questions": len(pipeline_results),
        },
        "ragas_metrics": ragas_metrics,
        "local_metrics": local_metrics or [],
        "pipeline_results": pipeline_results,
    }
    
    # Save to JSON
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[OK] Results saved to: {output_filename}")
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
        choices=["baseline", "no_rag", "reranking", "hyde", "semantic_chunking"],
        help="The experiment to run (default: baseline)"
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        default=DEFAULT_TEST_BATCH_ID,
        help=f"The test batch ID to use (default: {DEFAULT_TEST_BATCH_ID})"
    )
    parser.add_argument(
        "--rerank_keep_n",
        type=int,
        default=None,
        help="Override the default number of top reranked chunks to keep (env RERANK_KEEP_TOP_N or default 5)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set in .env file")
    
    print(f"\n[RAGAS] Evaluation Harness")
    print(f"Experiment: {args.experiment.upper()}")
    print(f"Batch ID: {args.batch_id}")
    # Allow CLI override of rerank keep-n
    global RERANK_KEEP_TOP_N
    if getattr(args, 'rerank_keep_n', None) is not None:
        try:
            RERANK_KEEP_TOP_N = int(args.rerank_keep_n)
            print(f"Using rerank_keep_n={RERANK_KEEP_TOP_N}")
        except Exception:
            print(f"Invalid --rerank_keep_n value: {args.rerank_keep_n}; using default {RERANK_KEEP_TOP_N}")
    
    # Load data
    questions, profiles = load_data()
    
    # Run pipeline
    pipeline_results = run_pipeline(questions, profiles, args.experiment, args.batch_id)

    # Compute lightweight local metrics before making evaluator calls
    local_metrics = compute_local_support_metrics(pipeline_results)
    
    # Run RAGAS evaluation
    evaluation_result = run_ragas_evaluation(pipeline_results, args.experiment, args.batch_id)
    
    # Print and save results
    print_metrics_summary(evaluation_result, args.experiment)
    print_local_metrics_summary(local_metrics)
    output_file = save_results(evaluation_result, pipeline_results, args.experiment, args.batch_id, local_metrics)
    save_local_metrics_csv(local_metrics, args.experiment, args.batch_id)
    
    print(f"\n[OK] Evaluation complete!")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
