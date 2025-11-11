# RAGAS Evaluation Guide - Phase 4-5: Run Experiments & Analyze

## Overview

The RAGAS evaluation infrastructure is now set up! This guide walks you through running the 4 experiments and interpreting the results.

## What's Ready

✅ **Phase 1: Query Processor Refactored**
- `run_retrieval()` method: Separates retrieval logic
- `run_generation()` method: Accepts pre-retrieved contexts
- Creates testable "seams" for RAGAS metric calculation

✅ **Phase 2: Test Data Created**
- `test_data/profile_1.json`: Test user profile
- `test_data/evaluation_dataset.json`: 5 golden test questions

✅ **Phase 3: Evaluation Harness Built**
- `run_evaluation.py`: Orchestrates 4 experiments
- Uses gpt-4o-mini for 10x cost savings
- Implements FlashRank re-ranking and HyDE query transformation

## Quick Start

### Prerequisites

1. **Ensure your batch exists**: The script uses batch_id `my_policies` by default
   - If you haven't uploaded documents, do that first in the UI
   - Or update `DEFAULT_TEST_BATCH_ID` in `run_evaluation.py`

2. **Verify .env file has API keys**:
   ```bash
   echo $env:OPENAI_API_KEY  # Should print your API key
   ```

### Run Baseline Experiment (5-10 minutes)

```bash
python run_evaluation.py --experiment baseline --batch_id my_policies
```

**What it does:**
1. Loads 5 test questions from `test_data/evaluation_dataset.json`
2. Runs your current retrieval pipeline (intent analysis, query expansion, RAG, web research)
3. Generates answers from retrieved contexts
4. Evaluates with 5 RAGAS metrics:
   - **context_precision**: Are retrieved chunks relevant? (0-1, higher is better)
   - **context_recall**: Did we retrieve all necessary chunks? (0-1, higher is better)
   - **faithfulness**: Is the answer supported by context? (0-1, higher is better)
   - **answer_relevancy**: Is the answer relevant to the question? (0-1, higher is better)
   - **answer_correctness**: Is the answer factually correct? (0-1, higher is better)

**Expected output:**
```
Results saved to: evaluation/results/ragas_baseline_my_policies_20251111_143022.json
```

### Run All 4 Experiments

```bash
# Experiment 1: Baseline (as above)
python run_evaluation.py --experiment baseline

# Experiment 2: No-RAG (LLM-only, should score low)
python run_evaluation.py --experiment no_rag

# Experiment 3: Re-ranking (with FlashRank, should improve context_precision)
python run_evaluation.py --experiment reranking

# Experiment 4: HyDE (with hypothetical answer expansion, may improve context_recall)
python run_evaluation.py --experiment hyde
```

Each experiment takes 5-10 minutes depending on API latency.

## Interpreting Results

Results are saved to `evaluation/results/ragas_*.json`. Each file contains:

```json
{
  "metadata": {
    "experiment": "baseline",
    "batch_id": "my_policies",
    "timestamp": "20251111_143022",
    "num_questions": 5
  },
  "ragas_metrics": {
    "faithfulness": [...],
    "answer_relevancy": [...],
    "context_precision": [...],
    "context_recall": [...],
    "answer_correctness": [...]
  },
  "pipeline_results": [
    {
      "question": "...",
      "answer": "...",
      "contexts": [...],
      "ground_truth": "...",
      "question_id": "TEST_001_BASELINE_RAG"
    }
  ]
}
```

### Diagnostic Guide

| Metric | If LOW... | Diagnosis | Fix |
|--------|-----------|-----------|-----|
| **context_precision** | Noisy, irrelevant chunks | No re-ranker (Lost in the Middle problem) | Run `reranking` experiment |
| **context_recall** | Missing necessary chunks | Naive chunking breaks tables/structure | Implement semantic chunking (Phase 5) |
| **faithfulness** | Hallucinations in answers | Symptom of low context_precision | Fix re-ranking first |
| **answer_correctness** | Factually wrong | Symptom of low context_recall | Fix chunking first |
| **answer_relevancy** | Off-topic answers | Query expansion too aggressive | Fine-tune expansion prompt |

### Expected Results

**Test Question 1 (Baseline):**
- Q: "What is the annual benefit limit for GREAT SupremeHealth?"
- Expected: Should retrieve directly from GREAT_SupremeHealth_Benefits.pdf
- Metric: context_recall and context_precision should both be 1.0

**Test Question 2 (Personalization):**
- Q: "What is my co-insurance for the GREAT SupremeHealth plan?"
- Expected: Answer should use policy_tiers from profile_1.json to give tier-specific info
- Metric: If profile not used → low correctness

**Test Question 3 (Web Fallback):**
- Q: "What are market alternatives to Manulife ManuProtect Term?"
- Expected: Should trigger web search (document doesn't have this)
- Metric: Low context_recall from docs, but web research fills the gap

**Test Question 4 (Chunking Flaw):**
- Q: "What is the benefit for 'Post Hospitalisation Treatment' for the 'P PLUS' plan?"
- Expected: This is a table row in GREAT SupremeHealth. If naive chunking breaks tables:
  - context_recall will be LOW
  - This proves the chunking issue!
  
**Test Question 5 (Comparison):**
- Q: "Compare medical evacuation in GREAT TravelCare vs GREAT SupremeHealth"
- Expected: Should retrieve from both documents
- Metric: If context_precision is low, web search triggered unnecessarily

## Comparing Experiments

Once you've run all 4, compare the CSV files:

```bash
# Open all results in a comparison view
Get-ChildItem evaluation/results/ragas_*.json | % { 
  $content = Get-Content $_; 
  $json = $content | ConvertFrom-Json;
  Write-Output "File: $($_.Name)"
  Write-Output "Average Metrics:"
  # Extract and display average scores
}
```

**What to expect:**

| Experiment | context_precision | context_recall | faithfulness | When Better |
|------------|------------------|-----------------|--------------|----------|
| baseline | ~0.6-0.7 | ~0.7-0.8 | ~0.5-0.7 | Control |
| no_rag | N/A (no context) | N/A | ~0.3 | Proves value of RAG |
| reranking | ~0.85-0.95 ↑ | ~0.7-0.8 | ~0.75-0.85 ↑ | Better context quality |
| hyde | ~0.6-0.7 | ~0.8-0.9 ↑ | ~0.5-0.7 | Complex queries |

## Next Steps (Phase 5: Advanced)

After analyzing baseline results:

1. **If context_precision is LOW**: Integrate re-ranking into production
   ```python
   # In query_processor.run_retrieval(), add:
   reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
   reranked = reranker.rerank(query, rag_chunks[:50])
   final_chunks = reranked[:10]  # Keep only top 10
   ```

2. **If context_recall is LOW**: Implement semantic chunking
   ```python
   # In utils/file_handlers.py, replace _create_chunks() with:
   from langchain.text_splitter import MarkdownHeaderTextSplitter
   splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "header1"), ("##", "header2")])
   ```

3. **If faithfulness is LOW**: Ensure re-ranking is working
   - Re-ranking directly improves faithfulness by filtering noise

4. **Embedding Model Experiment**: 
   ```bash
   # Test text-embedding-3-small vs text-embedding-3-large
   python run_evaluation.py --experiment baseline --batch_id my_policies_small
   python run_evaluation.py --experiment baseline --batch_id my_policies_large
   # Compare cost vs accuracy tradeoff
   ```

## Troubleshooting

### "Could not load batch 'my_policies'"
- Update `DEFAULT_TEST_BATCH_ID` in `run_evaluation.py` to match your actual batch
- Check `batches/` directory for available batch names

### "OPENAI_API_KEY must be set in .env file"
- Create/update `.env` in the project root:
  ```
  OPENAI_API_KEY=sk-...
  TAVILY_API_KEY=tvly-...  (optional, for web research)
  ```

### "FlashRank model download failed"
- First run will download the model (~100MB)
- Cached in `.flashrank_cache/`
- Requires internet connection

### Low scores on all metrics
- Verify documents are indexed correctly: `python check_status.py`
- Verify test questions are actually in your documents
- Check that batch_id points to documents with the expected policies

## Cost Estimate

- **Baseline experiment (5 questions)**: ~$0.10-0.20 (using gpt-4o-mini)
- **All 4 experiments**: ~$0.50-0.80 total
- **Re-ranking**: One-time download of FlashRank model (~100MB)

For comparison, using gpt-4-turbo would cost 10x more: $1.00-2.00 per experiment.

## Files Generated

```
evaluation/results/
├── ragas_baseline_my_policies_20251111_143022.json
├── ragas_no_rag_my_policies_20251111_143456.json
├── ragas_reranking_my_policies_20251111_144102.json
└── ragas_hyde_my_policies_20251111_144721.json
```

Each file contains full results, metrics, and individual question performance.

---

**Need Help?** Check the main RAGAS_Evaluation_Plan.instructions.md in .github/instructions/ for full details on the evaluation strategy.
