# RAG Pipeline Optimization - Combination Testing

**Status**: ✅ Complete and Ready to Use

## What This Is

A comprehensive framework for systematically testing different RAG pipeline configurations to find the optimal balance between performance (quality metrics) and cost (latency/tokens).

## Quick Start

### 1. Prerequisites
```bash
# Set up API keys
cp .env.example .env
# Edit .env: add OPENAI_API_KEY and TAVILY_API_KEY

# Install dependencies (if needed)
pip install pandas
```

### 2. Run Quick Test (15 minutes)
```bash
python quick_test_combinations.py
```

### 3. Analyze Results
```bash
python analyze_combinations.py --export_config
```

### 4. Apply Winning Configuration
```bash
source evaluation/results/combinations/optimal_config.sh
```

## What Gets Tested

### Experiments
- **baseline**: Standard hybrid search (BM25 + FAISS)
- **reranking**: + FlashRank re-ranking for precision
- **grounded_hyde**: + Hypothetical doc embeddings for recall
- **combined_best**: + HyDE + Re-ranking (maximum performance)

### Parameters
- **Candidate Pool**: 20, 50, 100 (retrieval breadth)
- **Generation Top-K**: 3, 5, 8, 10 (context budget)
- **Rerank Keep-N**: 3, 5, 8, 10 (precision control)

### Metrics (RAGAS)
- Faithfulness (avoid hallucinations) - 25% weight
- Answer Relevancy - 20% weight
- Context Precision - 20% weight
- Context Recall - 20% weight
- Answer Correctness - 15% weight

## Tools Included

### Testing Scripts
1. **quick_test_combinations.py** - Fast testing (6 configs, ~15 min)
   ```bash
   python quick_test_combinations.py
   ```

2. **test_combinations.py** - Full sweep (~15 configs, 1-2 hours)
   ```bash
   python test_combinations.py --phases all
   ```

3. **analyze_combinations.py** - Analysis & recommendations
   ```bash
   python analyze_combinations.py --export_config
   ```

### Data
- **test_data/minimal_test_dataset.json** - 3 questions for fast iteration
- **test_data/evaluation_dataset_auto_ragas.json** - 12 questions for validation

### Documentation
- **QUICK_START_OPTIMIZATION.md** - Step-by-step guide
- **COMBINATION_TESTING_README.md** - Full documentation
- **IMPLEMENTATION_SUMMARY.md** - Technical details

## Sample Results

Based on demonstration data:

| Configuration | Score | vs Baseline | Use Case |
|--------------|-------|-------------|----------|
| combined_best (K=8) | 0.851 | +14% | Max accuracy |
| reranking (K=8) | 0.829 | +11% | Balanced |
| baseline (K=8) | 0.747 | — | Reference |

## Typical Outcomes

After running tests, you'll get:

1. **Ranked configurations** by composite score
2. **Top 3 recommendations**:
   - 🏆 Overall best (maximum performance)
   - 💰 Cost-efficient (best baseline variant)
   - ⚡ Balanced (best reranking variant)
3. **Parameter sensitivity analysis**
4. **Exported shell script** with optimal settings

## Usage Patterns

### For Development
```bash
# Quick iteration with minimal dataset
python quick_test_combinations.py \
  --dataset test_data/minimal_test_dataset.json
```

### For Production Validation
```bash
# Full validation with complete dataset
python quick_test_combinations.py \
  --dataset test_data/evaluation_dataset_auto_ragas.json
```

### For Cost-Sensitive Use Cases
```bash
# Disable expensive features
export USE_WEB_RESEARCH=false
python quick_test_combinations.py
# Then select the "Cost-Efficient" recommendation
```

## Integration

The framework integrates with existing code:

```python
# Apply optimal config in your code
import os

# From exported config
os.environ["RETRIEVAL_CANDIDATE_POOL"] = "50"
os.environ["GENERATION_TOP_K"] = "8"
os.environ["RERANK_KEEP_TOP_N"] = "8"

# Or source the shell script before running
# source evaluation/results/combinations/optimal_config.sh
```

## Architecture

```
User runs test script
  ├─> For each configuration:
  │   ├─> Set environment variables
  │   ├─> Run run_evaluation.py
  │   ├─> Collect RAGAS metrics
  │   └─> Save results
  └─> Generate comparison CSV

User runs analysis script
  ├─> Load all results
  ├─> Calculate composite scores
  ├─> Rank configurations
  └─> Export optimal config

User applies configuration
  └─> Source shell script or update code
```

## File Structure

```
domain-agnostic-chatbot/
├── test_combinations.py              # Full parameter sweep
├── quick_test_combinations.py        # Fast testing
├── analyze_combinations.py           # Analysis tool
├── QUICK_START_OPTIMIZATION.md       # Quick start guide
├── COMBINATION_TESTING_README.md     # Full documentation
├── IMPLEMENTATION_SUMMARY.md         # Technical summary
├── test_data/
│   ├── minimal_test_dataset.json     # 3 questions (fast)
│   └── evaluation_dataset_auto_ragas.json  # 12 questions (full)
└── evaluation/results/combinations/
    ├── *_results_*.csv               # Test results
    ├── *_report_*.txt                # Analysis reports
    └── optimal_config.sh             # Exported config
```

## Next Steps

1. **Read**: QUICK_START_OPTIMIZATION.md
2. **Run**: `python quick_test_combinations.py`
3. **Analyze**: `python analyze_combinations.py --export_config`
4. **Apply**: `source evaluation/results/combinations/optimal_config.sh`
5. **Validate**: Run on full dataset
6. **Deploy**: Use in production

## Troubleshooting

- **No API keys**: Set OPENAI_API_KEY in .env
- **Batch not found**: Check batches/batch_registry.json
- **Rate limits**: Use --skip_ragas or add delays
- **Slow tests**: Use minimal dataset

See QUICK_START_OPTIMIZATION.md for detailed troubleshooting.

## Support

- Full guide: `QUICK_START_OPTIMIZATION.md`
- Documentation: `COMBINATION_TESTING_README.md`
- Technical: `IMPLEMENTATION_SUMMARY.md`
- Help: `python [script].py --help`

## Status

✅ All scripts validated
✅ Documentation complete
✅ Sample data provided
✅ Ready for production use

Just configure API keys and run!

---

**Quick Command Reference**

```bash
# Quick test
python quick_test_combinations.py

# Full sweep
python test_combinations.py --phases all

# Analyze results
python analyze_combinations.py --export_config

# Apply config
source evaluation/results/combinations/optimal_config.sh

# Validate
python run_evaluation.py --experiment combined_best
```
