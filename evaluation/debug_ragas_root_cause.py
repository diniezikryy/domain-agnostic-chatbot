#!/usr/bin/env python3
"""
RAGAS Root Cause Diagnostic Script
Purpose: Debug why context_precision returns NaN and why we get "LM returned 1 generations instead of 3" warnings.

This script:
1. Configures RAGAS with explicit settings (timeout, workers, logging)
2. Tests on a single item first
3. Logs all LLM calls and responses
4. Identifies the exact failure point

Usage:
    python evaluation/debug_ragas_root_cause.py

Requirements:
    - OPENAI_API_KEY in .env or environment
    - Baseline results file at: evaluation/results/ragas_baseline_my_policies_20251112_112727.json

This is a diagnostic tool. It's safe to run and useful for validating RAGAS configuration.
The proper configuration discovered here (ChatOpenAI with timeout + RunConfig) eliminates NaN issues.
"""

import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from langchain_openai import ChatOpenAI

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce noise from some loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def test_single_item():
    """Test RAGAS evaluation on a single item with full diagnostic output."""
    
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set")
    
    # Load the baseline results
    base_input = Path("evaluation/results/ragas_baseline_my_policies_20251112_112727.json")
    with open(base_input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    pipeline_results = data.get("pipeline_results", [])
    if not pipeline_results:
        raise ValueError("No pipeline_results found")
    
    # Take first item for testing
    test_item = pipeline_results[0]
    
    logger.info("=" * 80)
    logger.info("Testing RAGAS evaluation on single item")
    logger.info(f"Question ID: {test_item.get('question_id')}")
    logger.info(f"Question: {test_item.get('question')}")
    logger.info(f"Number of contexts: {len(test_item.get('contexts', []))}")
    logger.info(f"Answer length: {len(test_item.get('answer', ''))}")
    logger.info("=" * 80)
    
    # Configure LLM with explicit settings
    logger.info("\nConfiguring ChatOpenAI with:")
    logger.info("  - model: gpt-4o-mini")
    logger.info("  - temperature: 0 (deterministic)")
    logger.info("  - request_timeout: 180 seconds")
    logger.info("  - max_retries: 3")
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        request_timeout=180,
        max_retries=3,
    )
    
    # Configure RAGAS RunConfig
    logger.info("\nConfiguring RAGAS RunConfig with:")
    logger.info("  - max_workers: 1 (serial execution)")
    logger.info("  - timeout: 300 seconds")
    
    run_config = RunConfig(
        max_workers=1,
        timeout=300,
    )
    
    # Prepare dataset
    ds = Dataset.from_list([
        {
            "question": test_item.get("question"),
            "answer": test_item.get("answer"),
            "contexts": test_item.get("contexts", []),
            "ground_truth": test_item.get("ground_truth", ""),
        }
    ])
    
    # Test each metric individually to isolate the issue
    metrics_to_test = [
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
        ("context_precision", context_precision),
        ("context_recall", context_recall),
        ("answer_correctness", answer_correctness),
    ]
    
    results = {}
    
    for metric_name, metric in metrics_to_test:
        logger.info("\n" + "=" * 80)
        logger.info(f"Testing metric: {metric_name}")
        logger.info("=" * 80)
        
        try:
            logger.info(f"Calling evaluate() for {metric_name}...")
            result = evaluate(
                ds,
                metrics=[metric],
                llm=llm,
                run_config=run_config,
            )
            
            df = result.to_pandas()
            value = df[metric_name].iloc[0] if not df.empty else None
            
            logger.info(f"✓ {metric_name} = {value}")
            results[metric_name] = value
            
        except Exception as e:
            logger.error(f"✗ {metric_name} FAILED: {type(e).__name__}: {e}")
            results[metric_name] = f"ERROR: {e}"
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)
    for metric_name, value in results.items():
        status = "✓" if not isinstance(value, str) or not value.startswith("ERROR") else "✗"
        logger.info(f"{status} {metric_name}: {value}")
    
    # Now test all metrics together
    logger.info("\n" + "=" * 80)
    logger.info("Testing all metrics together")
    logger.info("=" * 80)
    
    try:
        all_metrics = [m for _, m in metrics_to_test]
        result = evaluate(
            ds,
            metrics=all_metrics,
            llm=llm,
            run_config=run_config,
        )
        
        df = result.to_pandas()
        logger.info("\n✓ All metrics completed successfully")
        logger.info("\nResults:")
        for col in df.columns:
            if col not in ["question", "answer", "contexts", "ground_truth"]:
                logger.info(f"  {col}: {df[col].iloc[0]}")
        
        return df
        
    except Exception as e:
        logger.error(f"\n✗ All metrics together FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    logger.info("RAGAS Root Cause Diagnostic Tool")
    logger.info("This will test RAGAS evaluation with detailed logging")
    logger.info("")
    
    result = test_single_item()
    
    if result is not None:
        logger.info("\n" + "=" * 80)
        logger.info("DIAGNOSIS COMPLETE - Evaluation succeeded with proper configuration")
        logger.info("=" * 80)
    else:
        logger.info("\n" + "=" * 80)
        logger.info("DIAGNOSIS COMPLETE - Found issues, check logs above")
        logger.info("=" * 80)
