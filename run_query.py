#!/usr/bin/env python3
"""
run_query.py

Utility to run the RAG pipeline for a single query without the UI and with an option
to skip the "deep research" step. Two main modes:

1) retrieval  - only run hybrid retrieval and print top chunks (no OpenAI calls)
2) rag-no-research - run retrieval and then generate a final answer using the RAG prompt
                     (this will call OpenAI for the final answer but will NOT invoke the
                     DeepResearch module)

Examples:
  .\.venv\Scripts\python run_query.py "Do I have inpatient cataract coverage?" --batch my_policies --mode retrieval
  .\.venv\Scripts\python run_query.py "Do I have inpatient cataract coverage?" --batch my_policies --mode rag-no-research

"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional

from batch_manager import BatchManager
from utils.search import HybridSearchEngine

try:
    from query_processor import QueryProcessor
except Exception:
    # QueryProcessor imports heavier modules (OpenAI/research). We only need it for
    # generating a final answer in rag-no-research mode; if import fails we'll error later.
    QueryProcessor = None


def load_profile(path: Optional[str]):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"Profile file not found: {path}")
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"Failed to load profile JSON: {e}")
        return None


def retrieval_only(query: str, batch_id: Optional[str], top_k: int):
    bm = BatchManager()
    # Allow overriding index paths via environment (useful when the registry doesn't list the batch)
    faiss_override = os.environ.get("RUN_QUERY_FAISS")
    bm25_override = os.environ.get("RUN_QUERY_BM25")

    engine = HybridSearchEngine()
    if faiss_override and bm25_override:
        if not engine.load_indexes(faiss_path=faiss_override, bm25_path=bm25_override):
            print("Failed to load search indexes from provided paths. Ensure the files exist.")
            return
    else:
        target_batch = batch_id or bm.get_default_batch()
        if not target_batch:
            print("No batch specified and no default batch available.")
            return

        paths = bm.get_batch_paths(target_batch)
        if not paths:
            print(f"Batch not found in registry: {target_batch}")
            return

        if not engine.load_indexes(faiss_path=paths["faiss_index"], bm25_path=paths["bm25_index"]):
            print("Failed to load search indexes. Ensure the batch has been processed.")
            return

    results = engine.hybrid_search(query=query, top_k=top_k)
    # Simple dedupe
    seen = set()
    unique = []
    for r in results:
        c = (r.get("content") or "").strip()
        if not c or c in seen:
            continue
        seen.add(c)
        unique.append(r)

    print(f"Retrieved {len(unique)} chunks (showing up to {top_k}):")
    for i, r in enumerate(unique[:top_k], start=1):
        md = r.get("metadata", {})
        fn = md.get("filename", "Unknown")
        pg = md.get("page_number", "N/A")
        score = r.get("score")
        print("\n---")
        print(f"Chunk {i}  |  file={fn}  page={pg}  score={score}")
        content = r.get("content", "").strip()
        print(content[:300] + ("..." if len(content) > 300 else ""))


def rag_no_research(query: str, batch_id: Optional[str], top_k: int, profile_path: Optional[str]):
    if QueryProcessor is None:
        raise RuntimeError("QueryProcessor import failed; ensure project dependencies are installed.")

    bm = BatchManager()
    qp = QueryProcessor(bm)

    target_batch = batch_id or bm.get_default_batch()
    if not target_batch:
        print("No batch specified and no default batch available.")
        return

    if not qp._ensure_batch_loaded(target_batch):
        print(f"Failed to load batch indexes for: {target_batch}")
        return

    # Use the raw user query (skip _expand_query if you want to avoid extra LLM call here)
    raw_results = qp.search_engine.hybrid_search(query=query, top_k=top_k)

    unique = []
    seen = set()
    for r in raw_results:
        c = (r.get("content") or "").strip()
        if not c or c in seen:
            continue
        seen.add(c)
        unique.append(r)

    if not unique:
        print("No relevant documents found for the query.")
        return

    user_profile = load_profile(profile_path)

    # Call the internal response generator (this will call OpenAI for the final answer)
    answer = qp._generate_response(query, unique, target_batch.startswith("user_"), user_profile)
    print("\n--- FINAL ANSWER ---\n")
    print(answer)


def main():
    parser = argparse.ArgumentParser(description="Run a single RAG query without the UI.")
    parser.add_argument("query", help="The user query text (wrap in quotes)")
    parser.add_argument("--batch", help="Batch id to use (e.g., my_policies)")
    parser.add_argument("--faiss-path", help="Direct path to a FAISS index dir (overrides batch)")
    parser.add_argument("--bm25-path", help="Direct path to a BM25 index file (overrides batch)")
    parser.add_argument("--mode", choices=["retrieval", "rag-no-research", "process_query"], default="retrieval", help="Mode of operation")
    parser.add_argument("--profile", help="Optional path to user profile JSON")
    parser.add_argument("--topk", type=int, default=10, help="Number of top chunks to retrieve")

    args = parser.parse_args()

    # If direct index paths were provided, export them so helper functions can use them
    if args.faiss_path and args.bm25_path:
        os.environ["RUN_QUERY_FAISS"] = args.faiss_path
        os.environ["RUN_QUERY_BM25"] = args.bm25_path

    if args.mode == "retrieval":
        retrieval_only(args.query, args.batch, args.topk)
    elif args.mode == "rag-no-research":
        rag_no_research(args.query, args.batch, args.topk, args.profile)
    else:
        # process_query uses the full pipeline (may trigger DeepResearch depending on intent)
        if QueryProcessor is None:
            raise RuntimeError("QueryProcessor import failed; ensure project dependencies are installed.")
        bm = BatchManager()
        qp = QueryProcessor(bm)
        resp = qp.process_query(args.query, batch_id=args.batch)
        print(resp)


if __name__ == "__main__":
    main()
