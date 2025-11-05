# RAG Evaluation Plan

## Overview

This document outlines the configurable RAG (Retrieval-Augmented Generation) experimentation framework designed to enable systematic testing and optimization of different retrieval and generation strategies.

## Architecture

### Configurable Components

The system supports parameterization of the following core components:

1. **Embedding Models**
   - `text-embedding-3-small` (default, 1536 dimensions)
   - `text-embedding-3-large` (3072 dimensions)
   - Custom embedding models (extensible)

2. **Retrieval Strategies**
   - `hybrid` (default): FAISS (60%) + BM25 (40%)
   - `vector_only`: Pure FAISS semantic search
   - `keyword_only`: Pure BM25 keyword search

3. **HyDE (Hypothetical Document Embeddings)**
   - Enabled/Disabled
   - Generates hypothetical answers and embeds them for improved retrieval
   - Lazy-loaded with graceful fallback if unavailable

4. **Reranking**
   - Enabled/Disabled
   - Uses CrossEncoder models for semantic reranking of retrieved chunks
   - Lazy-loaded with graceful fallback if unavailable
   - Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

5. **Generation Models**
   - `gpt-4o` (high-quality, comprehensive reasoning)
   - `gpt-4o-mini` (default, fast and cost-effective)
   - `gpt-3.5-turbo` (legacy support)

## Evaluation Harness

### Metrics

The evaluation framework uses RAGAS (Retrieval-Augmented Generation Assessment) metrics:

1. **Context Precision**: How relevant are the retrieved contexts?
2. **Context Recall**: Are all necessary contexts retrieved?
3. **Faithfulness**: Is the answer grounded in the retrieved context?
4. **Answer Relevancy**: Does the answer address the question?

Additional custom metrics:
- **Retrieval Balance**: Fair representation across sources (for comparison queries)
- **Citation Accuracy**: Proper source attribution
- **Response Time**: End-to-end latency

### Experiment Configuration

Experiments are defined in `evaluation/experiments.json`:

```json
{
  "experiments": [
    {
      "name": "baseline",
      "description": "Default hybrid search with gpt-4o-mini",
      "config": {
        "embedding_model": "text-embedding-3-small",
        "retrieval_strategy": "hybrid",
        "use_hyde": false,
        "use_reranking": false,
        "generation_model": "gpt-4o-mini"
      }
    },
    {
      "name": "hyde_enabled",
      "description": "Baseline + HyDE query synthesis",
      "config": {
        "embedding_model": "text-embedding-3-small",
        "retrieval_strategy": "hybrid",
        "use_hyde": true,
        "use_reranking": false,
        "generation_model": "gpt-4o-mini"
      }
    },
    {
      "name": "reranking_enabled",
      "description": "Baseline + CrossEncoder reranking",
      "config": {
        "embedding_model": "text-embedding-3-small",
        "retrieval_strategy": "hybrid",
        "use_hyde": false,
        "use_reranking": true,
        "generation_model": "gpt-4o-mini"
      }
    },
    {
      "name": "full_pipeline",
      "description": "HyDE + Reranking + GPT-4o",
      "config": {
        "embedding_model": "text-embedding-3-small",
        "retrieval_strategy": "hybrid",
        "use_hyde": true,
        "use_reranking": true,
        "generation_model": "gpt-4o"
      }
    }
  ]
}
```

### Ground Truth Queries

Test queries with expected behaviors are defined in `evaluation/test_queries.json`:

```json
{
  "queries": [
    {
      "query": "What medical conditions are covered?",
      "expected_sources": ["policy1.pdf", "policy2.pdf"],
      "expected_behavior": "comprehensive_coverage_list"
    },
    {
      "query": "Compare coverage between SingLife and FWD",
      "expected_sources": ["SingLife.pdf", "FWD.pdf"],
      "expected_behavior": "balanced_comparison"
    }
  ]
}
```

## Usage

### Prerequisites

```bash
# Install evaluation dependencies
pip install ragas datasets sentence-transformers
```

### Running Experiments

```bash
# Run all experiments
python evaluation/run_experiments.py --batch insurance

# Run specific experiment
python evaluation/run_experiments.py --batch insurance --experiment hyde_enabled

# Run with custom queries
python evaluation/run_experiments.py --batch insurance --queries evaluation/custom_queries.json

# Output results to custom directory
python evaluation/run_experiments.py --batch insurance --output-dir results/
```

### Output

The evaluation harness produces:

1. **JSON Summary** (`results/experiment_results.json`):
   ```json
   {
     "timestamp": "2024-11-05T04:45:00Z",
     "batch": "insurance",
     "experiments": [
       {
         "name": "baseline",
         "metrics": {
           "context_precision": 0.85,
           "context_recall": 0.78,
           "faithfulness": 0.92,
           "answer_relevancy": 0.88,
           "avg_response_time": 7.2
         }
       }
     ]
   }
   ```

2. **Console Leaderboard**:
   ```
   ================================================================================
   RAG EXPERIMENT LEADERBOARD
   ================================================================================
   Batch: insurance | Queries: 10 | Date: 2024-11-05
   
   Rank | Experiment        | C.Prec | C.Recall | Faith | A.Relev | Avg Time
   --------------------------------------------------------------------------------
   1    | full_pipeline     | 0.91   | 0.86     | 0.95  | 0.92    | 9.3s
   2    | reranking_enabled | 0.88   | 0.82     | 0.93  | 0.90    | 8.1s
   3    | hyde_enabled      | 0.87   | 0.80     | 0.92  | 0.89    | 7.8s
   4    | baseline          | 0.85   | 0.78     | 0.92  | 0.88    | 7.2s
   ```

## Rebuilding Indexes with Different Embeddings

To test different embedding models, rebuild batches:

```bash
# Rebuild with text-embedding-3-large
python setup_batch.py insurance --rebuild --embedding-model text-embedding-3-large

# Rebuild with default model
python setup_batch.py insurance --rebuild
```

## Best Practices

1. **Start with Baseline**: Always run baseline experiments first to establish performance floor
2. **Isolate Variables**: Change one component at a time to understand impact
3. **Use Representative Queries**: Ensure test queries cover diverse use cases
4. **Monitor Costs**: Track API usage, especially for expensive models (GPT-4o, large embeddings)
5. **Document Findings**: Update this document with insights from experiments

## Trustworthiness Considerations

Beyond RAGAS metrics, consider:

- **Hallucination Detection**: Does the system invent facts not in documents?
- **Source Attribution**: Are all claims properly cited?
- **Uncertainty Acknowledgment**: Does it admit when information is unavailable?
- **Retrieval Balance**: Fair representation in comparisons?

These can be incorporated as custom evaluation rules in the harness.

## Future Enhancements

- [ ] A/B testing framework for production queries
- [ ] Real-time monitoring dashboard
- [ ] Automatic hyperparameter tuning
- [ ] Multi-turn conversation evaluation
- [ ] User feedback integration
- [ ] Cost-quality tradeoff analysis

## References

- RAGAS Framework: https://github.com/explodinggradients/ragas
- HyDE Paper: https://arxiv.org/abs/2212.10496
- CrossEncoder Models: https://www.sbert.net/examples/applications/cross-encoder/README.html
