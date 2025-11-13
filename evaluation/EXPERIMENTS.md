## Experiments & Settings — RAGAS evaluation (domain-agnostic-chatbot)

This file documents the evaluation experiments available in this repository, their runtime settings, where results are written, and quick commands to reproduce runs.

Files & locations
- Experiment harness: `run_evaluation.py`
- Results directory: `evaluation/results/` (JSON per-run + comparison CSV/HTML)
- Comparison generator: `evaluation/generate_comparison.py`
- Cache manager: `utils/cache_manager.py` (persistent gzipped cache in `evaluation/cache/`)

Quick commands (from repository root)

```powershell
# Run a single experiment (change --experiment and --batch_id as needed)
python run_evaluation.py --experiment combined_best --batch_id my_policies

# Regenerate the comparison CSV/HTML (after one or more runs)
cd evaluation; python generate_comparison.py

# Clear combined_best cache (helper script)
python scripts\clear_cache_combined_best.py
```

Default environment variables (used by the harness)
- `RETRIEVAL_CANDIDATE_POOL` (int) — default: 50. How many candidate chunks to retrieve before reranking.
- `MAX_CONTEXTS_FOR_RAGAS` (int) — default: 8. Budget passed to RAGAS evaluation (and generation by default).
- `GENERATION_TOP_K` (int) — default: 8 (falls back to `MAX_CONTEXTS_FOR_RAGAS`). Number of chunks used by the generator.
- `RERANK_KEEP_TOP_N` (int) — default: equal to `GENERATION_TOP_K` (so reranker doesn't change generation budget unless configured).
- `USE_WEB_RESEARCH` (bool) — default: false. Controls whether DeepResearch/web fallback is allowed during retrieval.
- `RERANK_HYDE_WEIGHT` (float) — default: 0.6. Used by the hybrid reranker to weight HyDE vs original-query scores.

Models and components used
- Embeddings: `text-embedding-3-small` (per-batch; some experimental batches may use `text-embedding-3-large`).
- HyDE / grounding generation: `gpt-4o-mini` (used by `run_retrieval_grounded_hyde`).
- Reranker: FlashRank `ms-marco-MiniLM-L-12-v2` via the local `flashrank.Ranker`.
- Indexing: FAISS (vector index) + BM25 (sparse) via `utils/search.py` and `document_processor.py`.

Experiments (summary)

Below are the experiments currently implemented in `run_evaluation.py`. Each entry lists the core flow, key config, and the most recent aggregate RAGAS metrics (see `evaluation/results/ragas_comparison_*.csv`).

1) baseline
- Flow: standard retrieval (expansion → hybrid search), no reranking. Generator receives top `GENERATION_TOP_K` chunks.
- Settings: `RETRIEVAL_CANDIDATE_POOL=50`, `GENERATION_TOP_K=8`, `USE_WEB_RESEARCH=false` (unless overridden).
- Purpose: control group (RAG with default retrieval).
- Recent metrics (from comparison CSV): faithfulness=0.5427, answer_relevancy=0.9560, context_precision=0.3494, context_recall=0.3000, answer_correctness=0.4403

2) no_rag
- Flow: LLM-only (generation with no document context). Retrieval is still executed but generation is forced to use zero contexts.
- Purpose: show RAG value baseline.
- Recent metrics: faithfulness=0.3913, answer_relevancy=0.3872, context_precision=0.0000, context_recall=0.0000, answer_correctness=0.3808

3) hyde
- Flow: HyDE (Hypothetical Document Expansion) — generate hypothetical answer from LLM and use it directly for retrieval.
- Settings: typically `RETRIEVAL_CANDIDATE_POOL=30` for these runs (was used historically).
- Purpose: improve semantic matching / recall for tricky questions.
- Recent metrics: faithfulness=0.4119, answer_relevancy=0.7626, context_precision=0.2371, context_recall=0.2000, answer_correctness=0.3964

4) grounded_hyde
- Flow: Grounded HyDE (two-pass): initial retrieval → generate a grounded hypothetical answer using top chunks → second retrieval using the grounded answer.
- Purpose: HyDE but constrained (grounded) to retrieved documents to reduce hallucination.
- Recent metrics: faithfulness=0.7371, answer_relevancy=0.9574, context_precision=0.5169, context_recall=0.8000, answer_correctness=0.4720

5) reranking
- Flow: baseline retrieval → rerank the candidate passages with FlashRank (single-query rerank using the original user query) → generator uses top-k reranked chunks.
- Purpose: increase context precision by selecting higher-quality passages.
- Recent metrics: faithfulness=0.6231, answer_relevancy=0.7675, context_precision=0.5420, context_recall=0.4000, answer_correctness=0.3802

6) semantic_chunking
- Flow: baseline retrieval but using a semantic/structure-aware chunker (experimental). Aims to preserve tables/headers and improve chunk boundaries.
- Purpose: reduce chunking losses that harm context_recall (fix table breaks, etc.).
- Recent metrics: faithfulness=0.3365, answer_relevancy=0.5274, context_precision=0.3700, context_recall=0.4000, answer_correctness=0.2850

7) combined_best (current)
- Flow (implemented): Grounded HyDE (two-pass) → Reranking. The reranker previously used either the original query or the HyDE answer; the code now supports a hybrid reranking strategy (both signals combined) to avoid mismatch.
- Hybrid rerank details:
  - If a HyDE answer exists, the harness calls the reranker twice (HyDE answer and original query), produces two score lists, and combines them with a weighted average: combined_score = alpha * score_hyde + (1-alpha) * score_orig, where `alpha` is `RERANK_HYDE_WEIGHT` (default 0.6).
  - The top `keep_n` IDs (min of `GENERATION_TOP_K` and `RERANK_KEEP_TOP_N`) are kept for generation.
- Purpose: capture benefits of HyDE recall/faithfulness and reranker precision.
- Recent metrics (after hybrid rerank): faithfulness=0.5667, answer_relevancy=0.7659, context_precision=0.5600, context_recall=0.6000, answer_correctness=0.4941

Notes on metrics and interpretation
- faithfulness: degree to which the generated answer is supported by provided contexts (higher = less hallucination).
- answer_relevancy: how relevant the answer is to the question.
- context_precision: percent of retrieved contexts that are relevant (precision at chunk level).
- context_recall: percent of required contexts that were retrieved (recall at chunk level).
- answer_correctness: human-evaluable correctness (approx.).

Practical tips & next steps
- If you change retrieval / rerank behavior, clear relevant cache entries (see `scripts/clear_cache_combined_best.py` or `CacheManager.clear`) before re-running — otherwise prior cached pipeline results may hide code changes.
- To tune combined_best, run a small grid search over `RERANK_HYDE_WEIGHT` (recommended values: 0.0, 0.25, 0.5, 0.75, 1.0) and `RERANK_KEEP_TOP_N` (eg 8, 12, 16). Use the CSV comparison to quickly compare experiments.
- To debug individual failures, enable per-question diff logging (I can add a small debug flag to write the top-10 pre-rerank chunks and top-K post-rerank chunks to `evaluation/results/debug/<question_id>.json`).

Where to look in the code
- `run_evaluation.py` — experiment implementations and flow orchestration.
- `query_processor.py` — retrieval and generation glue code used by the harness.
- `utils/cache_manager.py` — caching logic (how to invalidate and re-run experiments).
- `scripts/clear_cache_combined_best.py` — helper to clear combined_best cache entries during development.

If you'd like, I can:
- Add a small script to run a parameter sweep and summarize results (CSV + plots).
- Add per-question debug diffs to help pin down which chunks the reranker drops/keeps for failing questions.

---
Generated on 2025-11-13 (automatically by the evaluation helper).
