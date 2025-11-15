# Combination Testing for RAG Pipeline Optimization

This directory contains tools for systematically testing different combinations of RAG pipeline parameters to find the optimal configuration that balances performance and cost.

## Overview

The RAG pipeline has several tunable parameters that affect performance:

1. **Experiment Type**: Different retrieval/generation strategies
   - `baseline`: Standard hybrid search (BM25 + FAISS)
   - `reranking`: Adds FlashRank re-ranking to improve precision
   - `grounded_hyde`: Uses grounded hypothetical document embeddings
   - `combined_best`: Combines grounded HyDE + re-ranking
   - `semantic_chunking`: Uses header-aware semantic chunking

2. **Retrieval Parameters**:
   - `RETRIEVAL_CANDIDATE_POOL`: Number of chunks to retrieve (20-100)
   - `GENERATION_TOP_K`: Number of chunks to send to LLM (3-10)
   - `RERANK_KEEP_TOP_N`: Number of chunks to keep after re-ranking (3-10)

## Quick Start

### 1. Prepare Your Environment

```bash
# Ensure you have required dependencies
pip install -r requirements.txt

# Set up your API keys in .env
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and TAVILY_API_KEY
```

### 2. Run Quick Test (Recommended)

Test the most promising configurations quickly:

```bash
python quick_test_combinations.py \
  --batch_id my_policies \
  --dataset test_data/minimal_test_dataset.json
```

This will test 6 key configurations:
- Baseline (reference)
- Reranking with K=5 and K=8
- Grounded HyDE
- Combined Best with K=5 and K=8

**Output**: CSV and comparison table showing which configuration performs best.

### 3. Run Full Combination Sweep

For exhaustive testing:

```bash
python test_combinations.py \
  --batch_id my_policies \
  --dataset test_data/minimal_test_dataset.json \
  --phases all
```

This tests:
- **Baseline phase**: Different pool sizes (20, 50, 100) and generation K values (3, 5, 8, 10)
- **Reranking phase**: Different rerank K values (3, 5, 8, 10)
- **Advanced phase**: HyDE and combined experiments
- **Semantic phase**: Semantic chunking

### 4. Run Selective Phases

```bash
# Test only reranking optimizations
python test_combinations.py --phases reranking

# Test baseline and advanced experiments only
python test_combinations.py --phases baseline advanced
```

## Test Datasets

### Minimal Dataset (Recommended for iteration)
**File**: `test_data/minimal_test_dataset.json`

Contains 3 carefully selected questions covering:
- Simple fact retrieval
- Table extraction from complex markdown
- Multi-document comparison

**Use case**: Fast iteration and parameter tuning (completes in minutes)

### Full Dataset
**File**: `test_data/evaluation_dataset_auto_ragas.json`

Contains 12 auto-generated questions covering comprehensive scenarios.

**Use case**: Final validation of winning configuration

## Understanding Results

### Metrics Explained

Each configuration is evaluated using RAGAS metrics:

1. **Faithfulness** (0-1): Is the answer grounded in the retrieved context?
   - Higher is better
   - Critical for avoiding hallucinations

2. **Answer Relevancy** (0-1): Does the answer address the question?
   - Higher is better
   - Measures if the LLM understood the query

3. **Context Precision** (0-1): Are the retrieved chunks relevant?
   - Higher is better
   - Low score means too much noise in retrieval

4. **Context Recall** (0-1): Did retrieval find all necessary information?
   - Higher is better
   - Low score means missing key chunks

5. **Answer Correctness** (0-1): Is the answer factually correct vs. ground truth?
   - Higher is better
   - Overall quality measure

### Composite Score

The scripts calculate a weighted composite score:
```
composite = 0.25 × faithfulness + 0.20 × answer_relevancy + 
            0.20 × context_precision + 0.20 × context_recall + 
            0.15 × answer_correctness
```

This balances faithfulness (avoiding hallucinations) with other quality metrics.

## Cost vs. Performance Tradeoffs

### Understanding the Tradeoffs

1. **Larger Candidate Pool** (e.g., 100 vs. 20):
   - ✅ Better recall (less likely to miss relevant info)
   - ❌ More computation, slower retrieval
   - ❌ More irrelevant chunks (noise)

2. **Higher Generation Top-K** (e.g., 10 vs. 3):
   - ✅ More context for LLM
   - ❌ Higher token costs
   - ❌ Possible "lost in the middle" effect

3. **Re-ranking**:
   - ✅ Dramatically improves precision
   - ✅ Reduces "lost in the middle" by prioritizing best chunks
   - ❌ Adds ~100-200ms latency
   - ✅ Actually reduces cost (fewer tokens to LLM)

4. **Grounded HyDE**:
   - ✅ Better recall for complex queries
   - ✅ Grounded (no hallucinations unlike regular HyDE)
   - ❌ Extra LLM call (adds cost and latency)
   - ❌ Requires 2 retrieval passes

5. **Combined Best** (HyDE + Re-ranking):
   - ✅ Best overall performance
   - ❌ Highest cost and latency
   - 💡 Use only for complex queries or when accuracy is critical

## Recommended Configurations

Based on typical results:

### For Cost-Sensitive Applications
```bash
Experiment: reranking
Candidate Pool: 50
Generation Top-K: 5
Rerank Keep-N: 5
```
**Rationale**: Re-ranking dramatically improves precision without expensive LLM calls. K=5 is sweet spot for cost/quality.

### For Performance-Critical Applications
```bash
Experiment: combined_best
Candidate Pool: 50
Generation Top-K: 8
Rerank Keep-N: 8
```
**Rationale**: Grounded HyDE + re-ranking maximizes all metrics at the cost of extra latency and tokens.

### For Balanced Performance
```bash
Experiment: grounded_hyde
Candidate Pool: 50
Generation Top-K: 8
Rerank Keep-N: N/A
```
**Rationale**: Better recall than baseline without re-ranking overhead. Good middle ground.

## Output Files

All results are saved to `evaluation/results/combinations/`:

- `combination_results_YYYYMMDD_HHMMSS.csv`: Full metrics table
- `combination_results_YYYYMMDD_HHMMSS.json`: Detailed JSON results
- `combination_report_YYYYMMDD_HHMMSS.txt`: Human-readable summary
- `quick_test_YYYYMMDD_HHMMSS.csv`: Quick test results

## Advanced Usage

### Skip RAGAS Evaluation (Testing Only)

For rapid pipeline testing without metric computation:

```bash
python quick_test_combinations.py --skip_ragas
```

**Warning**: This skips all quality metrics. Use only for debugging pipeline issues.

### Use Custom Dataset

```bash
python test_combinations.py \
  --dataset path/to/your/dataset.json \
  --batch_id your_batch_id
```

### Environment Variables

You can override defaults via environment variables:

```bash
# Set before running tests
export RETRIEVAL_CANDIDATE_POOL=100
export GENERATION_TOP_K=10
export RERANK_KEEP_TOP_N=8
export USE_WEB_RESEARCH=false

python quick_test_combinations.py
```

## Troubleshooting

### "Batch not found" error
- Ensure the batch exists: check `batches/batch_registry.json`
- Create a batch: see main README for `setup_batch.py` instructions

### "Rate limit exceeded" errors
- Add delays between tests by editing the scripts
- Use smaller dataset (minimal_test_dataset.json)
- Set `ENABLE_TPM_THROTTLE=true` in environment

### Missing RAGAS metrics
- Ensure OPENAI_API_KEY is set in .env
- RAGAS uses gpt-4o-mini by default (check `utils/model_config.py`)
- Check API quota

### Semantic chunking batch creation fails
- First run may create semantic batch automatically
- Check `batches/my_policies_semantic/` exists
- Semantic chunking requires source documents in batch metadata

## Next Steps

After identifying optimal configuration:

1. **Validate on full dataset**:
   ```bash
   python run_evaluation.py \
     --experiment [winning_experiment] \
     --batch_id my_policies \
     --dataset test_data/evaluation_dataset_auto_ragas.json
   ```

2. **Update production config**:
   - Set environment variables in production
   - Update `query_processor.py` defaults if needed

3. **Monitor in production**:
   - Track latency and cost metrics
   - Validate quality on real user queries
   - Consider A/B testing if switching experiments

## Example Workflow

```bash
# 1. Quick test to find promising configs
python quick_test_combinations.py

# 2. Review results
cat evaluation/results/combinations/quick_test_*.csv

# 3. If needed, do full sweep on specific phases
python test_combinations.py --phases reranking advanced

# 4. Validate winner on full dataset
export BEST_EXPERIMENT=combined_best  # example
python run_evaluation.py \
  --experiment $BEST_EXPERIMENT \
  --dataset test_data/evaluation_dataset_auto_ragas.json

# 5. Compare with baseline
python evaluation/generate_comparison.py  # if available
```

## Contributing

When adding new experiments or parameters:

1. Update `test_combinations.py` with new test configurations
2. Add to `quick_test_combinations.py` if it's promising
3. Document the tradeoffs in this README
4. Run full validation before recommending

## References

- **RAGAS**: https://docs.ragas.io/
- **FlashRank**: https://github.com/PrithivirajDamodaran/FlashRank
- **HyDE**: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)
