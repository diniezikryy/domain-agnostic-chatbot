"""
Interactive Query Comparison with Industry-Standard Metrics
Uses RAGAS and academic research metrics instead of custom heuristics.
"""

import sys
import time
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from query_processor import QueryProcessor
from batch_manager import BatchManager
from evaluation.llm_baseline_comparator import BaselineLLMComparator
from evaluation.industry_standard_evaluator import IndustryStandardEvaluator, StandardRAGMetrics
from config.settings import Settings

# Initialize settings
settings = Settings()
OPENAI_API_KEY = settings.openai_api_key


def print_side_by_side(left_text: str, right_text: str, left_title: str, right_title: str, width: int = 60):
    """Print two texts side by side."""
    # Split into lines
    left_lines = left_text.split('\n')
    right_lines = right_text.split('\n')
    
    # Wrap long lines
    def wrap_text(text, width):
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > width:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    lines.append(word[:width])
            else:
                current_line.append(word)
                current_length += word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    # Wrap all lines
    wrapped_left = []
    for line in left_lines:
        if line.strip():
            wrapped_left.extend(wrap_text(line, width))
        else:
            wrapped_left.append('')
    
    wrapped_right = []
    for line in right_lines:
        if line.strip():
            wrapped_right.extend(wrap_text(line, width))
        else:
            wrapped_right.append('')
    
    # Print headers
    print("\n" + "=" * (width * 2 + 5))
    print(f"{left_title:^{width}} | {right_title:^{width}}")
    print("=" * (width * 2 + 5))
    
    # Print side by side
    max_lines = max(len(wrapped_left), len(wrapped_right))
    for i in range(max_lines):
        left_line = wrapped_left[i] if i < len(wrapped_left) else ''
        right_line = wrapped_right[i] if i < len(wrapped_right) else ''
        print(f"{left_line:<{width}} | {right_line:<{width}}")
    
    print("=" * (width * 2 + 5) + "\n")


def print_standard_metrics(baseline_metrics: StandardRAGMetrics, rag_metrics: StandardRAGMetrics):
    """Print industry-standard metrics comparison."""
    print("\n" + "=" * 100)
    print(" INDUSTRY-STANDARD RAG EVALUATION METRICS (RAGAS Framework) ".center(100))
    print("=" * 100)
    
    print(f"\n{'Metric':<35} {'Baseline (No RAG)':<20} {'RAG Pipeline':<20} {'Winner':<15}")
    print("-" * 100)
    
    # RAGAS Score (overall)
    if abs(rag_metrics.ragas_score - baseline_metrics.ragas_score) < 0.05:  # Within 5% is tie
        ragas_winner = "Tie"
    elif rag_metrics.ragas_score > baseline_metrics.ragas_score:
        ragas_winner = "[RAG]"
    else:
        ragas_winner = "[Baseline]"
    print(f"{'RAGAS Score (Overall)':<35} {baseline_metrics.ragas_score:<20.3f} {rag_metrics.ragas_score:<20.3f} {ragas_winner:<15}")
    
    print("\n" + "-" * 100)
    print(" Core RAGAS Metrics ".center(100))
    print("-" * 100)
    
    # Faithfulness
    if abs(rag_metrics.faithfulness - baseline_metrics.faithfulness) < 0.05:
        faith_winner = "Tie"
    elif rag_metrics.faithfulness > baseline_metrics.faithfulness:
        faith_winner = "RAG"
    else:
        faith_winner = "Baseline"
    print(f"{'Faithfulness':<35} {baseline_metrics.faithfulness:<20.3f} {rag_metrics.faithfulness:<20.3f} {faith_winner:<15}")
    print(f"{'  (factual consistency)':<35}")
    
    # Answer Relevance
    if abs(rag_metrics.answer_relevance - baseline_metrics.answer_relevance) < 0.05:
        rel_winner = "Tie"
    elif rag_metrics.answer_relevance > baseline_metrics.answer_relevance:
        rel_winner = "RAG"
    else:
        rel_winner = "Baseline"
    print(f"{'Answer Relevance':<35} {baseline_metrics.answer_relevance:<20.3f} {rag_metrics.answer_relevance:<20.3f} {rel_winner:<15}")
    print(f"{'  (addresses the query)':<35}")
    
    # Context Precision
    if abs(rag_metrics.context_precision - baseline_metrics.context_precision) < 0.05:
        prec_winner = "Tie"
    elif rag_metrics.context_precision > baseline_metrics.context_precision:
        prec_winner = "RAG"
    else:
        prec_winner = "Baseline"
    print(f"{'Context Precision':<35} {baseline_metrics.context_precision:<20.3f} {rag_metrics.context_precision:<20.3f} {prec_winner:<15}")
    print(f"{'  (retrieval quality)':<35}")
    
    # Context Recall
    if abs(rag_metrics.context_recall - baseline_metrics.context_recall) < 0.05:
        rec_winner = "Tie"
    elif rag_metrics.context_recall > baseline_metrics.context_recall:
        rec_winner = "RAG"
    else:
        rec_winner = "Baseline"
    print(f"{'Context Recall':<35} {baseline_metrics.context_recall:<20.3f} {rag_metrics.context_recall:<20.3f} {rec_winner:<15}")
    print(f"{'  (info coverage)':<35}")
    
    print("\n" + "-" * 100)
    print(" Additional Metrics ".center(100))
    print("-" * 100)
    
    # Hallucination (lower is better)
    if abs(rag_metrics.hallucination_score - baseline_metrics.hallucination_score) < 0.05:
        hall_winner = "Tie"
    elif rag_metrics.hallucination_score < baseline_metrics.hallucination_score:
        hall_winner = "RAG"
    else:
        hall_winner = "Baseline"
    print(f"{'Hallucination Score':<35} {baseline_metrics.hallucination_score:<20.3f} {rag_metrics.hallucination_score:<20.3f} {hall_winner:<15}")
    print(f"{'  (lower = better)':<35}")
    
    # Citation Quality
    if abs(rag_metrics.citation_quality - baseline_metrics.citation_quality) < 0.05:
        cite_winner = "Tie"
    elif rag_metrics.citation_quality > baseline_metrics.citation_quality:
        cite_winner = "RAG"
    else:
        cite_winner = "Baseline"
    print(f"{'Citation Quality':<35} {baseline_metrics.citation_quality:<20.3f} {rag_metrics.citation_quality:<20.3f} {cite_winner:<15}")
    
    print("\n" + "=" * 100)
    
    # Improvement summary
    ragas_improvement = rag_metrics.ragas_score - baseline_metrics.ragas_score
    ragas_improvement_pct = (ragas_improvement / baseline_metrics.ragas_score * 100) if baseline_metrics.ragas_score > 0 else None

    print(f"\nEvaluation Method: {rag_metrics.evaluation_method.upper()}")

    if ragas_improvement > 0:
        print(f"\n[RESULT] RAG system demonstrates a +{ragas_improvement:.3f} point improvement in RAGAS score")
        if ragas_improvement_pct is not None:
            print(f"         Relative improvement: {ragas_improvement_pct:+.1f}% over baseline performance")
        else:
            print(f"         Relative improvement: N/A (baseline score is zero)")
        if ragas_improvement > 0.2:
            print(f"         Assessment: Statistically significant quality enhancement observed")
        else:
            print(f"         Assessment: Measurable quality improvement detected")
    elif ragas_improvement < 0:
        if ragas_improvement_pct is not None:
            print(f"\n[WARNING] Baseline system outperforms RAG by {abs(ragas_improvement):.3f} points ({abs(ragas_improvement_pct):.1f}%)")
        else:
            print(f"\n[WARNING] Baseline system outperforms RAG by {abs(ragas_improvement):.3f} points (relative % N/A, baseline score is zero)")
        print(f"          Recommendation: Review RAG configuration and document corpus quality")
    else:
        print(f"\n[INFO] Performance parity observed between systems (difference within statistical margin)")
    
    # Dimension breakdown
    print(f"\n[ANALYSIS] Performance Breakdown by Dimension:")
    improvements = {
        "Faithfulness": rag_metrics.faithfulness - baseline_metrics.faithfulness,
        "Relevance": rag_metrics.answer_relevance - baseline_metrics.answer_relevance,
        "Precision": rag_metrics.context_precision - baseline_metrics.context_precision,
        "Recall": rag_metrics.context_recall - baseline_metrics.context_recall
    }
    
    for metric, improvement in improvements.items():
        if improvement > 0.1:
            symbol = "+"
        elif improvement > -0.1:
            symbol = "="
        else:
            symbol = "-"
        print(f"   [{symbol}] {metric}: {improvement:+.3f}")
    
    print()


def compare_query_standard(query: str, batch_name: str = "critical_illness", use_llm_judge: bool = True):
    """
    Run query comparison using industry-standard metrics.
    
    Args:
        query: The query to compare
        batch_name: Batch to use for RAG retrieval
        use_llm_judge: Use LLM-as-judge for evaluation (more accurate but uses tokens)
    """
    print(f"\n{'='*100}")
    print(f" QUERY: {query} ".center(100))
    print(f"{'='*100}\n")
    
    # Initialize components
    print("[INIT] Initializing components...")
    batch_manager = BatchManager()
    query_processor = QueryProcessor(batch_manager)
    baseline_comparator = BaselineLLMComparator()
    evaluator = IndustryStandardEvaluator(use_llm_judge=use_llm_judge)
    
    # Check batch exists
    if batch_name not in batch_manager.list_batches():
        print(f"[ERROR] Batch '{batch_name}' not found")
        print(f"Available batches: {', '.join(batch_manager.list_batches())}")
        return
    
    # Get baseline response
    print("\n[BASELINE] Getting baseline LLM response (no RAG)...")
    baseline_start = time.time()
    baseline_result = baseline_comparator.get_baseline_response(query)
    baseline_response = baseline_result['response']
    baseline_time = baseline_result['time_seconds']
    print(f"[BASELINE] Response received in {baseline_time:.2f}s")
    
    # Get RAG response with retrieved contexts
    print("\n[RAG] Getting RAG pipeline response...")
    rag_start = time.time()
    
    # Get query processor response (which internally retrieves contexts)
    rag_response = query_processor.process_query(query, batch_name)
    rag_time = time.time() - rag_start
    
    # Retrieve contexts separately to pass to evaluator
    from utils.search import HybridSearchEngine
    
    paths = batch_manager.get_batch_paths(batch_name)
    if not paths:
        print(f"[ERROR] Could not load batch paths for '{batch_name}'")
        return
    
    searcher = HybridSearchEngine()
    searcher.load_indexes(paths['faiss_index'], paths['bm25_index'])
    top_chunks = searcher.hybrid_search(query, top_k=5)
    retrieved_contexts = [chunk['content'] for chunk in top_chunks]  # Use 'content' key
    
    print(f"[RAG] Response received in {rag_time:.2f}s")
    print(f"[RAG] Retrieved {len(retrieved_contexts)} context chunks")
    
    # Performance comparison
    print("\n" + "=" * 100)
    print(" PERFORMANCE COMPARISON ".center(100))
    print("=" * 100)
    print(f"Baseline Time: {baseline_time:.2f}s")
    print(f"RAG Time:      {rag_time:.2f}s")
    time_diff = rag_time - baseline_time
    time_diff_pct = (time_diff / baseline_time * 100)
    time_diff_str = f"+{time_diff:.2f}s" if time_diff >= 0 else f"{time_diff:.2f}s"
    print(f"Overhead:      {time_diff_str} ({time_diff_pct:+.1f}%)")
    print("=" * 100)
    
    # Evaluate baseline (treat as RAG with no contexts for fair comparison)
    print("\n[EVAL] Evaluating baseline response with industry-standard metrics...")
    baseline_metrics = evaluator.evaluate_rag_response(
        query=query,
        answer=baseline_response,
        retrieved_contexts=[]  # No retrieval for baseline
    )
    
    # Evaluate RAG
    print("\n[EVAL] Evaluating RAG response with industry-standard metrics...")
    rag_metrics = evaluator.evaluate_rag_response(
        query=query,
        answer=rag_response,
        retrieved_contexts=retrieved_contexts
    )
    
    # Print side-by-side responses
    print_side_by_side(
        left_text=baseline_response,
        right_text=rag_response,
        left_title="BASELINE LLM (No RAG)",
        right_title="RAG PIPELINE",
        width=60
    )
    
    # Print metrics comparison
    print_standard_metrics(baseline_metrics, rag_metrics)
    
    # Token usage summary
    if evaluator.use_llm_judge:
        print("\n[TOKEN USAGE] Evaluation Cost:")
        if baseline_metrics.tokens_used:
            print(f"   Baseline evaluation: {baseline_metrics.tokens_used} tokens")
        if rag_metrics.tokens_used:
            print(f"   RAG evaluation: {rag_metrics.tokens_used} tokens")
        total_eval_tokens = (baseline_metrics.tokens_used or 0) + (rag_metrics.tokens_used or 0)
        print(f"   Total evaluation cost: ~${total_eval_tokens * 0.00000015:.4f} (at $0.15/1M tokens)")


def interactive_mode():
    """Run in interactive mode with industry-standard metrics."""
    print("\n" + "="*100)
    print(" INTERACTIVE QUERY COMPARISON (Industry-Standard Metrics) ".center(100))
    print("="*100)
    print("\nEnter queries to compare baseline LLM vs RAG pipeline responses.")
    print("Using RAGAS framework and academic research metrics.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Check API key
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not found in environment")
        print("Please set it in your .env file")
        return
    
    # Get batch name
    batch_manager = BatchManager()
    batches = list(batch_manager.list_batches())
    
    if not batches:
        print("[ERROR] No batches found")
        print("Please run setup_batch.py first to create a batch")
        return
    
    print(f"Available batches: {', '.join(batches)}")
    batch_name = input(f"Enter batch name to use (default: {batches[0]}): ").strip()
    
    if not batch_name:
        batch_name = batches[0]
    
    if batch_name not in batches:
        print(f"[ERROR] Batch '{batch_name}' not found")
        return
    
    print(f"\n[INFO] Using batch: {batch_name}")
    
    # Ask about LLM judge
    use_llm = input("\nUse LLM-as-judge for evaluation? (more accurate, uses tokens) [Y/n]: ").strip().lower()
    use_llm_judge = use_llm != 'n'
    
    if use_llm_judge:
        print("[INFO] Using LLM-as-judge (GPT-4o-mini) for nuanced evaluation")
    else:
        print("[INFO] Using heuristic evaluation (free, but less accurate)")
    
    print()
    
    while True:
        try:
            # Get query
            query = input("\n[INPUT] Enter your query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                print("[WARNING] Please enter a query")
                continue
            
            # Compare
            compare_query_standard(query, batch_name, use_llm_judge)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()


def single_query_mode(query: str, batch_name: str, use_llm_judge: bool = True):
    """Run a single query comparison with standard metrics."""
    # Check API key
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not found in environment")
        print("Please set it in your .env file")
        return
    
    compare_query_standard(query, batch_name, use_llm_judge)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query mode
        query = sys.argv[1]
        batch_name = sys.argv[2] if len(sys.argv) > 2 else "critical_illness"
        use_llm_judge = sys.argv[3].lower() != 'false' if len(sys.argv) > 3 else True
        single_query_mode(query, batch_name, use_llm_judge)
    else:
        # Interactive mode
        interactive_mode()
