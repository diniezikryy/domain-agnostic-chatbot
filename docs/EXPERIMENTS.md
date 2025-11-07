# RAG Experiments — Reference and Runbook

This document consolidates the experiment setup, implementation notes, evaluation metrics, quick start steps, and customization examples for controlled RAG (Retrieval-Augmented Generation) experiments. It is factual, self-contained, and intended for reproducible experiments on the `my_policies` batch.

## Scope
- Purpose: Compare retrieval and generation configurations (hybrid, vector-only, HyDE, reranking, and LLM baseline) under controlled conditions.
- Safety: Production code (`query_processor.py`) is unchanged. The experimental processor is `experimental_query_processor.py` and is isolated from production logic.

## Experiments (configured by default)
- Control 1 — LLM Baseline: generation-only (no retrieval).
- Control 2 — RAG Baseline: hybrid search + generation model.
- Exp 2 — HyDE: use hypothetical-document (HyDE) transformation before retrieval.
- Exp 3 — Reranking: retrieve a larger set then rerank top results prior to generation.
- Exp 5 — Vector Only: FAISS-only semantic search (no BM25).

## Metrics collected

Quality (RAGAS):
- Faithfulness (0–1): factual consistency with retrieved context.
- Answer Relevance (0–1): relevance to the query.
- Context Precision (0–1): relevance of retrieved documents.
- Context Recall (0–1): coverage of necessary information.
- RAGAS score: harmonic mean of the four metrics.

Trustworthiness:
- Hallucination score (0–1, lower is better).
- Accuracy score (0–1): expected keywords present.
- Completeness score (0–1): coverage of expected items.

Operational:
- Latency (seconds).
- Tokens used (API cost proxy).

## Quick start

1. Ensure Python environment and `.env` with `OPENAI_API_KEY`.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Ensure the `my_policies` batch exists (create if needed):

```powershell
python setup_batch.py my_policies --source documents/my_policies
```

4. Run a quick test:

```powershell
python test_experimental_processor.py
```

5. Run full experiments:

```powershell
python evaluation/run_experiments.py
```

By default the harness loads queries from `evaluation/ground_truth.json` and `evaluation/custom_queries.json` (your experiment configuration may load `evaluation/test_queries_my_policies.json` if configured).

## What the harness does
- Loads configured test queries.
- Iterates configured experiment definitions.
- For each query+experiment: retrieves contexts, generates an answer, evaluates with RAGAS and trustworthiness metrics, records tokens and latency.
- Produces an aggregate comparison and writes detailed JSON results to `evaluation/results/`.

## Results and report format

Saved JSON structure (per-run):
- experiment configuration and parameters.
- per-query results (response, retrieved contexts, metrics, tokens, latency).
- aggregate statistics and ranked comparison table (best config by RAGAS).

Filename pattern: `evaluation/results/experiment_report_YYYYMMDD_HHMMSS.json`.

Console output (per-experiment sample):

```
RUNNING: Control_2_RAG_Baseline
  Query 1/15: <short description>
    RAGAS: 0.823 | Faithful: 0.889 | Tokens: 1659 | Time: 7.73s
COMPLETE - Control_2_RAG_Baseline
  RAGAS Score: 0.8234 | Faithfulness: 0.8891 | Avg Latency: 7.45s | Avg Tokens: 1650
```

## Key files (purpose)
- `experimental_query_processor.py` — experiment-ready RAG processor; parameterized (model, retrieval strategy, HyDE, reranking, top_k) and returns (response, tokens, latency). Exposes retrieved contexts via `get_last_contexts()`.
- `evaluation/run_experiments.py` — main experiment harness that enumerates experiment configs, runs queries, evaluates, and saves results.
- `utils/search.py` — Hybrid search engine with `vector_only_search()` method.
- `evaluation/industry_standard_evaluator.py` — RAGAS metrics calculation and LLM-judge wrapper.
- `evaluation/trustworthiness_evaluator.py` — hallucination/accuracy/completeness checks.
- `test_experimental_processor.py` — lightweight smoke test for experiment plumbing.

## How to add or modify experiments

Edit `evaluation/run_experiments.py` and add an entry to `EXPERIMENT_CONFIGS`. Example:

```python
EXPERIMENT_CONFIGS["Exp_Custom"] = {
    "description": "HyDE + reranking",
    "type": "rag",
    "processor_class": ExperimentalQueryProcessor,
    "batch_id": "my_policies",
    "params": {
        "generation_model": "gpt-4o-mini",
        "retrieval_strategy": "hybrid",
        "use_hyde": True,
        "use_reranking": True,
        "top_k": 10
    }
}
```

To add a test query, append to `evaluation/custom_queries.json` using the same schema as existing entries (id, query, category, expected_keywords, notes).

## Evaluator modes

LLM-as-judge (recommended): accurate, uses API tokens. Use for final evaluations and production readiness checks.

Heuristic mode: deterministic, fast, and free (term overlap, simple rules). Use for rapid iteration and development.

## Design notes (concise)
- Experimental processor omits user profile integration to ensure fair comparison across experiments.
- The experimental harness prioritizes reproducibility (low-temp deterministic generation) and consistent metric collection.
- Reranking may require additional dependencies; if native extensions cause environment issues, use an LLM-based reranker or disable reranking.

## Troubleshooting (common issues)
- OpenAI API key missing: ensure `.env` contains `OPENAI_API_KEY=...`.
- Batch not found: run `python setup_batch.py my_policies --source documents/my_policies`.
- Import errors: `pip install -r requirements.txt` and verify environment.
- Slow runs: reduce query count, use `gpt-4o-mini`, or disable reranking experiments.

## Cost and runtime guidance
- Per-query time: ~5–10s (varies by model and retrieval).
- Cost proxy: monitor `tokens_used` in results. Example estimate: 15 queries × 5 experiments ≈ 75 API calls; using `gpt-4o-mini` this is cost-effective for iterative runs.

## Next steps after running experiments
1. Analyze ranked comparison (sort by RAGAS score and inspect trade-offs: latency, tokens, hallucination).
2. Identify candidate configuration for production testing (balance of quality, speed, cost).
3. Optionally run targeted A/B evaluation with a subset of real users before full deployment.

## References
- RAGAS metrics: harmonic mean of faithfulness, relevance, precision, and recall.
- Implementation files listed above for quick access.

---

File created by merging experiment documentation: setup, implementation summary, evaluation guide, and quick-reference information. Keep this file up-to-date when experiment configs or metrics change.
