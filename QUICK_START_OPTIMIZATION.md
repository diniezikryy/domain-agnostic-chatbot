# Quick Start Guide: Finding Your Optimal RAG Configuration

This guide will help you quickly find the best RAG pipeline configuration for your use case.

## Prerequisites

1. **Set up API keys**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY and TAVILY_API_KEY
   ```

2. **Verify your batch exists**:
   ```bash
   python -c "import json; print(json.load(open('batches/batch_registry.json'))['batches'].keys())"
   ```
   You should see `my_policies` or your batch ID.

## Option 1: Quick Test (Recommended - 15 minutes)

Test the 6 most promising configurations:

```bash
python quick_test_combinations.py
```

This will test:
- ✅ Baseline (reference)
- ✅ Reranking with K=5 (cost-efficient)
- ✅ Reranking with K=8 (balanced)
- ✅ Grounded HyDE (better recall)
- ✅ Combined Best K=5 (aggressive optimization, lower cost)
- ✅ Combined Best K=8 (maximum performance)

**Results**: CSV file in `evaluation/results/combinations/`

**Analyze**:
```bash
python analyze_combinations.py --export_config
```

**Apply winning config**:
```bash
source evaluation/results/combinations/optimal_config.sh
```

## Option 2: Full Sweep (Comprehensive - 1-2 hours)

Test all parameter combinations systematically:

```bash
python test_combinations.py --phases all
```

This tests:
- **Baseline phase**: Pool sizes (20, 50, 100) and Gen K (3, 5, 8, 10) = ~7 tests
- **Reranking phase**: Rerank K (3, 5, 8, 10) = 4 tests
- **Advanced phase**: HyDE and Combined = 3 tests
- **Semantic phase**: Semantic chunking = 1 test

**Total**: ~15 configurations

**Analyze**:
```bash
python analyze_combinations.py --export_config
```

## Option 3: Selective Testing

Test only specific phases:

```bash
# Just reranking optimizations
python test_combinations.py --phases reranking

# Baseline + advanced only
python test_combinations.py --phases baseline advanced
```

## Understanding Your Results

After running tests, the analysis script shows:

1. **Top 3 Configurations**: Best performers by composite score
2. **Performance by Experiment Type**: Which experiment class works best
3. **Parameter Sensitivity**: Impact of pool size, top-K, etc.
4. **Recommendations**: 
   - 🏆 Overall best
   - 💰 Cost-efficient
   - ⚡ Balanced

### Interpreting Metrics

- **Faithfulness** (0-1): Prevents hallucinations ← Most important
- **Answer Relevancy** (0-1): Answers the actual question
- **Context Precision** (0-1): Retrieved chunks are relevant
- **Context Recall** (0-1): Found all necessary information
- **Answer Correctness** (0-1): Factually accurate vs. ground truth

**Composite Score** = Weighted average (emphasizes faithfulness)

### Typical Results

Based on similar evaluations:

| Experiment | Composite | Speed | Cost | Use Case |
|-----------|-----------|-------|------|----------|
| Baseline | 0.70-0.75 | Fast | Low | Simple queries, cost-sensitive |
| Reranking | 0.80-0.85 | Fast | Low | Balanced performance |
| Grounded HyDE | 0.80-0.85 | Slow | Medium | Complex queries needing better recall |
| Combined Best | 0.85-0.90 | Slow | High | Maximum accuracy needed |

## Applying Your Results

### Method 1: Environment Variables (Recommended)

```bash
# Use the exported config
source evaluation/results/combinations/optimal_config.sh

# Then run your application
python run_query.py "What is my coverage limit?"
```

### Method 2: Update Code Defaults

Edit `run_evaluation.py` or your application to set:

```python
RETRIEVAL_CANDIDATE_POOL = 50  # from your results
GENERATION_TOP_K = 8           # from your results
RERANK_KEEP_TOP_N = 8          # if using reranking

# Use the winning experiment in your app:
experiment = "combined_best"  # or "reranking", etc.
```

### Method 3: Per-Query Configuration

For advanced use cases, set parameters per query type:

```python
# For simple factual queries
if is_simple_fact_query(query):
    experiment = "baseline"
    top_k = 5

# For complex comparison queries
elif is_comparison_query(query):
    experiment = "combined_best"
    top_k = 8
```

## Validating on Full Dataset

After finding your optimal configuration, validate it:

```bash
# Set your winning experiment
export BEST_EXPERIMENT=combined_best  # example from your results

# Run on full test set
python run_evaluation.py \
  --experiment $BEST_EXPERIMENT \
  --batch_id my_policies \
  --dataset test_data/evaluation_dataset_auto_ragas.json
```

Compare the full dataset results with your combination test results to ensure consistency.

## Cost Optimization

If your optimal config is too expensive:

1. **Try the "Balanced" recommendation** (usually reranking)
2. **Reduce generation top-K**: 8 → 5 saves ~40% tokens
3. **Use baseline for simple queries**: Create a query classifier
4. **Disable web research**: Set `USE_WEB_RESEARCH=false`

Example of cost-aware configuration:

```bash
export RETRIEVAL_CANDIDATE_POOL=50
export GENERATION_TOP_K=5        # Reduced from 8
export RERANK_KEEP_TOP_N=5       # Reduced from 8
export USE_WEB_RESEARCH=false    # Disabled
```

## Troubleshooting

### Tests are too slow
- Use `--skip_ragas` flag for pipeline testing only (no metrics)
- Use minimal dataset (3 questions) instead of full (12 questions)
- Run selective phases instead of full sweep

### Rate limit errors
- Reduce test frequency (edit scripts to add sleep between tests)
- Use gpt-4o-mini instead of gpt-4o (set in `utils/model_config.py`)
- Enable throttling: `export ENABLE_TPM_THROTTLE=true`

### Results don't match expectations
- Check batch is correct: `cat batches/batch_registry.json`
- Verify dataset questions match your use case
- Check cache: may be using cached results (`--clear-cache` to reset)

### Missing metrics in output
- Ensure OPENAI_API_KEY is set and valid
- Check RAGAS didn't fail (look for error messages in output)
- Verify you're not using `--skip_ragas` flag

## Next Steps

1. **Run quick test** (15 min)
2. **Analyze results** (1 min)
3. **Apply winning config** (instant)
4. **Validate on full dataset** (30 min)
5. **Deploy to production** with monitoring

## Example Full Workflow

```bash
# 1. Quick test
python quick_test_combinations.py

# 2. Analyze and export config
python analyze_combinations.py --export_config

# 3. Review recommendations
cat evaluation/results/combinations/optimal_config.sh

# 4. Apply and validate
source evaluation/results/combinations/optimal_config.sh
python run_evaluation.py \
  --experiment combined_best \
  --dataset test_data/evaluation_dataset_auto_ragas.json

# 5. Compare with baseline
python evaluation/generate_comparison.py  # if available

# 6. Deploy
# Use the exported config in production environment
```

## Support

- Full documentation: `COMBINATION_TESTING_README.md`
- Analysis help: `python analyze_combinations.py --help`
- Testing help: `python test_combinations.py --help`
- Quick test help: `python quick_test_combinations.py --help`

## Success Metrics

You've successfully optimized your RAG pipeline when:

- ✅ You have a composite score > 0.80
- ✅ Faithfulness > 0.85 (critical for avoiding hallucinations)
- ✅ Your winning config beats baseline by >10%
- ✅ Cost/latency is acceptable for your use case
- ✅ Results validated on full test dataset

Good luck finding your optimal configuration! 🚀
