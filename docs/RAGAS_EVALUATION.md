## RAGAS Evaluation — Quick Guide

This document explains how to run the RAGAS evaluation harness included in this repository. It gives step-by-step, reproducible instructions (PowerShell examples), describes the key environment knobs used to make experiments fair and comparable, and explains how to interpret the main RAGAS metrics.

---

## Goal

Run reproducible, fair RAG (retrieve-and-generate) experiments and collect RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness). This guide covers:

- Prerequisites and quick setup
- How to run a single experiment (powerful knobs shown)
- How to run the full experiment suite
- Where results are saved and how to aggregate them
- How to interpret metrics and next-step recommendations

## Prerequisites

- Python 3.10+ (use the project's virtual environment)
- A valid OpenAI API key in `.env` as `OPENAI_API_KEY`
- (Optional) `TAVILY_API_KEY` if you want to enable the web research integration
- FAISS and other optional system dependencies are installed via `requirements.txt`

Activate the project's venv (PowerShell example):

```powershell
& '.venv\Scripts\Activate.ps1'
```

Create or update `.env` with your keys (PowerShell example):

```powershell
$env:OPENAI_API_KEY = 'sk-...'
# Optionally set the web-research key
$env:TAVILY_API_KEY = 'tavily-key-if-you-have-it'
```

Important: make sure the batch you want to evaluate (example: `my_policies`) exists under `batches/` and that you've processed documents with `setup_batch.py` if needed.

## Key scripts

- `run_evaluation.py` — the main RAGAS evaluation harness (per-experiment)
- `run_all_experiments.py` — convenience runner that calls the per-experiment harness sequentially
- `scripts/collect_ragas_summaries.py` — helper to aggregate saved RAGAS JSON outputs into a compact CSV-like summary

## Environment knobs (fairness controls)

These environment variables are used by the harness to make experiments deterministic and comparable:

- `RETRIEVAL_CANDIDATE_POOL` — how many candidate chunks the retriever returns (e.g. 30 or 50)
- `USE_WEB_RESEARCH` — `true` or `false`, controls whether external web research is allowed
- `RERANK_KEEP_TOP_N` — when re-ranking is enabled, how many top chunks to keep for generation
- `GENERATION_TOP_K` — how many contexts the generator receives (defaults to the RAG budget)
- `MAX_CONTEXTS_FOR_RAGAS` — number of contexts passed to RAGAS for evaluation (keeps the evaluator stable)

Set these in PowerShell (example):

```powershell
$env:RETRIEVAL_CANDIDATE_POOL = '30'
$env:USE_WEB_RESEARCH = 'false'
$env:RERANK_KEEP_TOP_N = '5'
$env:GENERATION_TOP_K = '5'
```

## Quick: Run a single experiment (baseline)

This will run the baseline experiment for `my_policies` and print a compact per-question table.

PowerShell example (single-line):

```powershell
$env:RETRIEVAL_CANDIDATE_POOL='30'; $env:USE_WEB_RESEARCH='false'; $env:RERANK_KEEP_TOP_N='5'; & '.venv\Scripts\python.exe' run_evaluation.py --experiment baseline --batch_id my_policies --show_table
```

Notes:
- `--show_table` prints the readable per-question table in the terminal.
- The run creates two artifacts under `evaluation/results/`: a RAGAS JSON (detailed evaluator output) and a `local_metrics_*.csv` summarizing per-question results.

## Run the full experiment suite

The repository includes `run_all_experiments.py` to run experiments sequentially (baseline, no_rag, reranking, hyde, semantic_chunking). Use the same environment knobs for every run to keep comparisons fair.

Example PowerShell (run all experiments sequentially):

```powershell
$env:RETRIEVAL_CANDIDATE_POOL='30'; $env:USE_WEB_RESEARCH='false'; $env:RERANK_KEEP_TOP_N='5'; & '.venv\Scripts\python.exe' run_all_experiments.py --batch_id my_policies
```

After the run, aggregate the outputs using the helper script:

```powershell
& '.venv\Scripts\python.exe' scripts\collect_ragas_summaries.py evaluation/results/ --out evaluation/results/ragas_summary_latest.csv
```

## Typical workflow — rebuild and verify a batch, then evaluate

1. Recreate the batch (if you changed documents or chunking):

```powershell
& '.venv\Scripts\python.exe' setup_batch.py my_policies --rebuild
```

2. Inspect the batch to sanity-check chunking and metadata:

```powershell
& '.venv\Scripts\python.exe' utils\inspect_batch.py my_policies
```

3. Run the semantic-chunking experiment (example):

```powershell
$env:RETRIEVAL_CANDIDATE_POOL='30'; & '.venv\Scripts\python.exe' run_evaluation.py --experiment semantic_chunking --batch_id my_policies --show_table
```

## Output files & where to look

- `evaluation/results/ragas_<experiment>_<batch_id>_<timestamp>.json` — full RAGAS evaluation JSON with per-example annotations
- `evaluation/results/local_metrics_<experiment>_<batch_id>_<timestamp>.csv` — a compact CSV with per-question answers, contexts count, and local support metrics

Tip: Keep an experiment naming convention and always record the `RETRIEVAL_CANDIDATE_POOL` and `USE_WEB_RESEARCH` values when comparing runs.

## Interpreting the main metrics (quick guide)

- context_precision (low): retrieved contexts are noisy. Likely fix: apply re-ranking (experiment `reranking`) or lower the retrieval pool.
- context_recall (low): the retriever is missing the necessary chunks. Likely fix: switch to HyDE or improve chunking (semantic chunking experiment).
- faithfulness (low): the answer is not supported by contexts. Likely fix: improve context_precision (re-rank) or reduce noisy contexts.
- answer_correctness (low): generator gave a factually wrong answer. Likely fix: retrieval/recall improvement (semantic chunking or HyDE).

## Troubleshooting

- "Batch not found" — run `setup_batch.py` and ensure the folder exists in `batches/`.
- "No OPENAI_API_KEY" — create a `.env` file or export `OPENAI_API_KEY` before running.
- Web research missing: set `TAVILY_API_KEY` (if you want web research) or set `USE_WEB_RESEARCH=false`.
- FAISS errors: make sure `faiss-cpu` (or `faiss-gpu`) is installed per `requirements.txt` and the batch indexes were built successfully.

## Next steps & experiments

- Use `run_all_experiments.py` to produce side-by-side artifacts.
- Run light hyperparameter sweeps across `RETRIEVAL_CANDIDATE_POOL` and `GENERATION_TOP_K` and compare CSV outputs using `scripts/collect_ragas_summaries.py`.
- If `context_recall` is low for targeted questions, try `HyDE` or create a specialized semantic batch with a different embedding model.

## Where to find more docs

- Short pointer in the root `README.md` to this file.
- Implementation notes and improvement plans are under `.github/instructions/` (RAGAS_Evaluation_Plan.instructions.md and RAGAS_Improvement_Plan.instructions.md).

---

If you want, I can also add a short `examples/` notebook that runs one question through `run_retrieval()` and `run_generation()` so you can inspect retrieved chunks interactively.
