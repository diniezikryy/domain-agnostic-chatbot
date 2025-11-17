"""
RAGAS Evaluation Harness
Orchestrates evaluation of RAG pipeline with experiments including baseline, no_rag, reranking, HyDE and semantic_chunking.
Uses gpt-4o-mini for RAGAS metrics to reduce costs.

Output Naming Scheme:
- Individual experiments: ragas_{experiment}_{batch_id}_{readable_time}.json
- Multi-experiment reports: ragas_report_{batch_id}_{readable_time}_{experiments}.json
- Readable time format: YYYY-MM-DD_HH-MM-SS
- Experiments sorted alphabetically and joined with '+'

Usage:
  python run_evaluation.py --experiment baseline --batch_id my_policies
  python run_evaluation.py --experiment reranking --batch_id my_policies
  python run_evaluation.py --experiment hyde --batch_id my_policies_large
  python run_evaluation.py --experiment no_rag --batch_id my_policies
  python run_evaluation.py --experiment semantic_chunking --batch_id my_policies
"""

import json
import os
import sys
import argparse
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI

# Import your application classes
from query_processor import QueryProcessor
from batch_manager import BatchManager
from utils.cache_manager import CacheManager
from utils.model_config import get_model_name
from utils.throttler import compute_safe_delay


MAX_CONTEXTS_FOR_RAGAS = int(os.getenv("MAX_CONTEXTS_FOR_RAGAS", "8"))

# Number of top retrieval chunks to keep when generating answers.
# Default: use the same value as MAX_CONTEXTS_FOR_RAGAS to ensure
# evaluation (RAGAS) and generation receive the same context budget.
GENERATION_TOP_K = int(os.getenv("GENERATION_TOP_K", str(MAX_CONTEXTS_FOR_RAGAS)))

# Reranker configuration: default number of top-ranked chunks to keep.
# By default this will fall back to the generation top-k so reranking does not
# accidentally return a different-sized candidate set to the generator.
RERANK_KEEP_TOP_N = int(os.getenv("RERANK_KEEP_TOP_N", str(GENERATION_TOP_K)))

# Evaluation tuning: candidate pool size and whether to allow web research during experiments
RETRIEVAL_CANDIDATE_POOL = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "50"))
USE_WEB_RESEARCH = os.getenv("USE_WEB_RESEARCH", "false").lower() in ("1", "true", "yes")

# RRF fusion aggressiveness: lower values weight the very top-ranked chunks more heavily.
# First recommended fix for regressions is to dial this down from the theoretical 60 default.
RRF_FUSION_K = int(os.getenv("RRF_FUSION_K", "20"))

# Number of RRF-fused chunks to append before reranking when running the
# combined_best_rrf experiment. Keep modest to avoid overwhelming the reranker.
COMBINED_RRF_TOP_M = int(os.getenv("COMBINED_RRF_TOP_M", "8"))

# Optional override to force RRF regardless of gating
RRF_FORCE = os.getenv("RRF_FORCE", "false").lower() in ("1", "true", "yes")
# FAISS confidence threshold for gating RRF (applies if force==False)
try:
    RRF_FAISS_CONFIDENCE_THRESHOLD = float(os.getenv("RRF_FAISS_CONFIDENCE_THRESHOLD", "0.5"))
except Exception:
    RRF_FAISS_CONFIDENCE_THRESHOLD = 0.5

# Allow configuration of which sources to promote into the RRF top-k (string list). Example: 'faiss,bm25'
RRF_PROMOTE_ORDER = os.getenv("RRF_PROMOTE_ORDER", "faiss,bm25")

ENABLE_TPM_THROTTLE = os.getenv("ENABLE_TPM_THROTTLE", "false").lower() in ("1", "true", "yes")

EXPERIMENT_CHOICES = [
    "baseline",
    "no_rag",
    "reranking",
    "rrf",
    "grounded_hyde",
    "combined_best",
    "combined_best_rrf",
    "semantic_chunking",
    "semantic_reranking",
    "advanced_fusion",
]


def _chunk_unique_key(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return f"id:{chunk_id}"
    filename = metadata.get("filename")
    page = metadata.get("page_number")
    if filename or page is not None:
        return f"file:{filename or 'unknown'}|page:{page}"
    content = chunk.get("content", "")
    if content:
        return content[:256]
    return str(id(chunk))


def _merge_chunk_lists(
    primary: List[Dict[str, Any]],
    fallback: List[Dict[str, Any]],
    max_chunks: Optional[int] = None
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()

    def _add_chunk(chunk: Dict[str, Any]) -> None:
        if not chunk:
            return
        key = _chunk_unique_key(chunk)
        if key in seen:
            return
        seen.add(key)
        merged.append(chunk)

    for chunk in primary or []:
        if max_chunks is not None and len(merged) >= max_chunks:
            return merged
        _add_chunk(chunk)

    for chunk in fallback or []:
        if max_chunks is not None and len(merged) >= max_chunks:
            break
        _add_chunk(chunk)

    return merged


# Simple manager to encapsulate FlashRank initialization and calls.
# This replaces the previous pattern of attaching the reranker to the function
# object and centralizes error handling.
class RerankerManager:
    def __init__(self):
        self._initialized = False
        self.ranker = None
        self.RerankRequest = None

    def init(self):
        if self._initialized:
            return
        try:
            from flashrank import Ranker, RerankRequest
            # keep a small local cache dir to avoid re-downloading models repeatedly
            self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".flashrank_cache")
            self.RerankRequest = RerankRequest
            self._initialized = True
        except Exception as e:
            # bubble up so callers can handle fallback
            raise

    def rerank(self, query, passages):
        self.init()
        request = self.RerankRequest(query=query, passages=passages)
        result = self.ranker.rerank(request)
        try:
            return list(result)
        except Exception:
            return result


reranker_manager = RerankerManager()

# =========================================================================
# CONFIGURATION
# =========================================================================

EVAL_DATASET_PATH = "test_data/evaluation_dataset_minimal.json"
TEST_PROFILES_DIR = "test_data"

# IMPORTANT: Update this to match your test user's batch_id
# First, register a user in the UI, upload the 3 PDFs, then find the batch_id in batches/
DEFAULT_TEST_BATCH_ID = "my_policies"  # Actual batch ID - change to match your setup


# =========================================================================
# DATA LOADING
# =========================================================================

def load_data() -> tuple[List[Dict], Dict]:
    """Loads the golden set and test profiles."""
    with open(EVAL_DATASET_PATH, 'r') as f:
        golden_set = json.load(f)
    
    profiles = {}
    for item in golden_set:
        profile_id = item.get("user_profile_id")
        if profile_id and profile_id not in profiles:
            profile_path = Path(TEST_PROFILES_DIR) / f"{profile_id}.json"
            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    profiles[profile_id] = json.load(f)
    
    print(f"Loaded {len(golden_set)} test questions and {len(profiles)} user profiles.")
    return golden_set, profiles


# =========================================================================
# EXPERIMENTAL PIPELINE FUNCTIONS
# =========================================================================

def run_retrieval_baseline(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Runs the default retrieval pipeline."""
    # Ensure the retrieval candidate pool size and web-research policy are applied
    retrieval_data = query_processor.run_retrieval(
        query=query,
        batch_id=batch_id,
        user_profile=user_profile,
        skip_expansion=False,
        top_k=RETRIEVAL_CANDIDATE_POOL,
        allow_web_research=USE_WEB_RESEARCH,
    )

    # Keep only the top-K retrieved chunks for generation to avoid a confounding
    # variable where the baseline uses many more candidates than the reranker.
    try:
        retrieval_data["rag_chunks_details"] = retrieval_data.get("rag_chunks_details", [])[:GENERATION_TOP_K]
        retrieval_data["rag_contexts_list"] = retrieval_data.get("rag_contexts_list", [])[:GENERATION_TOP_K]
    except Exception:
        # Be robust to unexpected retrieval_data shapes
        pass

    return retrieval_data


def run_generation_baseline(
    query_processor: QueryProcessor,
    query: str,
    retrieval_data: Dict,
    user_profile: Optional[Dict]
) -> str:
    """Runs the default generation pipeline."""
    return query_processor.run_generation(
        query=query,
        rag_chunks=retrieval_data["rag_chunks_details"],
        research_results=retrieval_data["web_research_raw"],
        user_profile=user_profile
    )


def run_generation_no_rag(
    query_processor: QueryProcessor,
    query: str,
    retrieval_data: Dict,
    user_profile: Optional[Dict]
) -> str:
    """Experiment 1: Runs generation with NO RAG context."""
    print("--- [EVAL] Running Generation (NO_RAG) ---")
    # Provide the generator with the full-document contexts returned by
    # the retrieval stage (if any). This keeps the behavior "no RAG"
    # (no FAISS/BM25/re-ranking), but still allows the LLM to read the
    # user's documents and profile for grounded answers.
    rag_chunks = []
    research_results = {}
    if retrieval_data:
        rag_chunks = retrieval_data.get("rag_chunks_details", []) or []
        research_results = retrieval_data.get("web_research_raw", {}) or {}

    return query_processor.run_generation(
        query=query,
        rag_chunks=rag_chunks,
        research_results=research_results,
        user_profile=user_profile
    )


def run_retrieval_rerank(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Experiment 2: Runs retrieval and adds a re-ranking step using FlashRank."""
    print("--- [EVAL] Running Retrieval (RE-RANKING) ---")
    # Use same candidate pool and web research setting as baseline for fairness
    retrieval_data = query_processor.run_retrieval(
        query=query,
        batch_id=batch_id,
        user_profile=user_profile,
        skip_expansion=False,
        top_k=RETRIEVAL_CANDIDATE_POOL,
        allow_web_research=USE_WEB_RESEARCH,
    )
    
    # Re-rank the document chunks (using the centralized RerankerManager)
    rag_chunks = retrieval_data.get("rag_chunks_details", [])
    if not rag_chunks:
        return retrieval_data

    passages = [{"id": i, "text": chunk.get("content", "")} for i, chunk in enumerate(rag_chunks)]

    try:
        reranked = reranker_manager.rerank(query, passages)
    except Exception as e:
        print(f"Warning: re-ranker initialization/execute failed ({e}). Skipping re-ranking step.")
        return retrieval_data

    # Ensure list
    try:
        reranked_list = list(reranked)
    except Exception:
        reranked_list = reranked if isinstance(reranked, (list, tuple)) else []

    # Map rerank info (score and rank position) back to original chunks
    rerank_info = {}
    for rank_pos, item in enumerate(reranked_list, start=1):
        # item expected to contain at least 'id' and optionally 'score'
        idx = item.get('id') if isinstance(item, dict) else None
        score = item.get('score') if isinstance(item, dict) and 'score' in item else None
        if idx is None:
            continue
        # Convert numpy float32 to Python float for JSON serialization
        score_value = float(score) if score is not None and hasattr(score, '__float__') else score
        rerank_info[int(idx)] = {"rerank_score": score_value, "rerank_rank": rank_pos}

    # Attach rerank metadata to each chunk
    for i, chunk in enumerate(rag_chunks):
        info = rerank_info.get(i, {})
        chunk['rerank_score'] = info.get('rerank_score')
        chunk['rerank_rank'] = info.get('rerank_rank')

    # Decide how many top-ranked chunks to keep.
    # Base this on GENERATION_TOP_K (the number that will be used for generation),
    # but allow an explicit RERANK_KEEP_TOP_N to further restrict it.
    keep_n = GENERATION_TOP_K if isinstance(GENERATION_TOP_K, int) and GENERATION_TOP_K > 0 else 5
    if isinstance(RERANK_KEEP_TOP_N, int) and RERANK_KEEP_TOP_N > 0:
        keep_n = min(keep_n, RERANK_KEEP_TOP_N)

    # Build set of top ids to keep (filter None values defensively)
    top_ids = [item.get('id') for item in reranked_list[:keep_n] if isinstance(item, dict) and item.get('id') is not None]
    reranked_indices = {int(i) for i in top_ids if i is not None}

    final_rag_chunks = [c for i, c in enumerate(rag_chunks) if i in reranked_indices]

    generation_limit = GENERATION_TOP_K if isinstance(GENERATION_TOP_K, int) and GENERATION_TOP_K > 0 else None
    final_rag_chunks = _merge_chunk_lists(final_rag_chunks, rag_chunks, max_chunks=generation_limit)

    # Update retrieval data with reranked chunks and provide rerank_info for debugging/analysis
    retrieval_data["rag_chunks_details"] = final_rag_chunks
    retrieval_data["rag_contexts_list"] = [c.get("content", "") for c in final_rag_chunks]
    retrieval_data["rerank_info"] = rerank_info

    print(f"Re-ranked from {len(rag_chunks)} to {len(final_rag_chunks)} chunks. (keep_n={keep_n})")
    return retrieval_data


def run_retrieval_rrf(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Experiment: Runs retrieval using Reciprocal Rank Fusion (RRF) to combine FAISS and BM25.
    
    RRF is a principled, parameter-free fusion method that combines ranked lists from
    multiple retrieval systems by using ranks instead of raw scores. This avoids the
    need for ad-hoc weighting between FAISS and BM25 scores.
    """
    print(f"--- [EVAL] Running Retrieval (RRF Fusion, k={RRF_FUSION_K}) ---")
    
    # First, run the standard retrieval to get rag_chunks_details and trigger web research
    retrieval_data = query_processor.run_retrieval(
        query=query,
        batch_id=batch_id,
        user_profile=user_profile,
        skip_expansion=False,
        top_k=RETRIEVAL_CANDIDATE_POOL,
        allow_web_research=USE_WEB_RESEARCH,
    )
    
    # Now apply RRF-based fusion at the search engine level
    try:
        search_engine = query_processor.search_engine
        if search_engine and hasattr(search_engine, 'hybrid_search_rrf'):
            print("  Using RRF fusion instead of weighted combination")
            # Apply the configured promotion order (if any) so callers can control
            # whether Faiss or BM25 gets preference when promoting items into the
            # top-k. Default is 'faiss,bm25'.
            if hasattr(search_engine.hybrid_search_rrf, '__call__'):
                try:
                    rrf_results = search_engine.hybrid_search_rrf(
                        query=query,
                        top_k=RETRIEVAL_CANDIDATE_POOL,
                        k=RRF_FUSION_K,
                        ensure_top_sources=True,
                        promote_order=[p.strip() for p in RRF_PROMOTE_ORDER.split(',') if p.strip()],
                        force=RRF_FORCE,
                        faiss_confidence_threshold=RRF_FAISS_CONFIDENCE_THRESHOLD,
                    )
                except TypeError:
                    # Older versions of the method might not accept 'promote_order'
                    # so fall back to calling without it (backwards compatible).
                    rrf_results = search_engine.hybrid_search_rrf(
                        query=query,
                        top_k=RETRIEVAL_CANDIDATE_POOL,
                        k=RRF_FUSION_K,
                        ensure_top_sources=True,
                        force=RRF_FORCE,
                        faiss_confidence_threshold=RRF_FAISS_CONFIDENCE_THRESHOLD,
                    )
            
            # Convert RRF results to rag_chunks_details format
            rag_chunks = []
            for rrf_result in rrf_results:
                chunk = {
                    "content": rrf_result.get("content", ""),
                    "metadata": rrf_result.get("metadata", {}),
                    "score": rrf_result.get("score", 0.0),
                    "source": "rrf",
                    "rrf_score": rrf_result.get("score", 0.0),
                }
                rag_chunks.append(chunk)
            
            # Limit to generation budget
            rag_chunks = rag_chunks[:GENERATION_TOP_K]
            retrieval_data["rag_chunks_details"] = rag_chunks
            retrieval_data["rag_contexts_list"] = [c.get("content", "") for c in rag_chunks]
            
            print(f"RRF fusion produced {len(rag_chunks)} final chunks for generation")
        else:
            print("  Warning: RRF method not available, falling back to baseline")
            
    except Exception as e:
        print(f"  Warning: RRF fusion failed ({e}), using baseline results")
        # Fallback to baseline: just apply generation_top_k limit
        retrieval_data["rag_chunks_details"] = retrieval_data.get("rag_chunks_details", [])[:GENERATION_TOP_K]
        retrieval_data["rag_contexts_list"] = retrieval_data.get("rag_contexts_list", [])[:GENERATION_TOP_K]
    
    return retrieval_data


def run_retrieval_docs_full(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Return whole-document texts (no chunking/retrieval) so generation can use raw docs.

    This is intended for the `no_rag` experiment where the generator should be
    able to read the documents but we do NOT perform indexing, chunking,
    BM25/FAISS retrieval or re-ranking. Each document is provided as a single
    context entry (one string per document).
    """
    print("--- [EVAL] Running Retrieval (WHOLE DOCUMENTS, NO RAG) ---")

    # Locate batch metadata to find the source document paths
    try:
        paths = query_processor.batch_manager.get_batch_paths(batch_id)
        meta_path = paths.get("metadata") if paths else None
    except Exception:
        meta_path = None

    docs = []
    if meta_path and Path(meta_path).exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            doc_entries = meta.get('documents', [])
        except Exception:
            doc_entries = []
    else:
        doc_entries = []

    # If no metadata, try a few fallbacks: batches/<batch_id>/ documents or documents/<batch_id>
    if not doc_entries:
        fallback_dir = Path('batches') / batch_id
        if fallback_dir.exists():
            # collect any files in the batch dir
            for p in fallback_dir.glob('*'):
                if p.is_file() and p.suffix.lower() in ['.pdf', '.docx', '.txt', '.md']:
                    doc_entries.append({'filename': p.name, 'file_path': str(p)})

    if not doc_entries:
        # Last resort: look in documents/<batch_id> or documents
        docs_dir = Path('documents') / batch_id
        if docs_dir.exists():
            for p in docs_dir.glob('*'):
                if p.is_file() and p.suffix.lower() in ['.pdf', '.docx', '.txt', '.md']:
                    doc_entries.append({'filename': p.name, 'file_path': str(p)})

    file_handler = None
    try:
        from utils.file_handlers import FileHandler
        file_handler = FileHandler()
    except Exception:
        file_handler = None

    rag_chunks_details = []
    rag_contexts_list = []

    for entry in doc_entries:
        fp = entry.get('file_path') or entry.get('filename')
        if not fp:
            continue
        p = Path(fp)
        # Try a few candidate locations if the path is not absolute / not found
        candidates = [p]
        if not p.exists():
            candidates.insert(0, Path('batches') / batch_id / p.name)
            candidates.insert(0, Path('documents') / p.name)

        doc_path = None
        for c in candidates:
            if c.exists():
                doc_path = c
                break

        if not doc_path:
            print(f"  [WARN] Document not found for entry: {entry}")
            continue

        # Extract text
        text = None
        try:
            if file_handler:
                chunks, _ = file_handler.process_document(str(doc_path), chunking_strategy='page')
                if chunks:
                    text = "\n\n".join(chunks)
            if not text:
                # Fallback to simple read for text files
                if doc_path.suffix.lower() in ['.txt', '.md']:
                    text = doc_path.read_text(encoding='utf-8')
                else:
                    # As a last resort, set a placeholder
                    text = f"[Full document not extractable: {doc_path.name}]"
        except Exception as e:
            print(f"  [WARN] Failed to extract {doc_path}: {e}")
            text = f"[Error extracting document: {doc_path.name}]"

        # Prepare single-entry chunk for the whole document
        metadata = {
            'source': str(doc_path),
            'filename': doc_path.name,
            'file_type': doc_path.suffix.lower(),
            'chunking_strategy': 'whole_document'
        }
        rag_chunks_details.append({'content': text, 'metadata': metadata})
        rag_contexts_list.append(text)

    return {
        'rag_chunks_details': rag_chunks_details,
        'rag_contexts_list': rag_contexts_list,
        'web_contexts_list': [],
        'web_research_raw': {},
        'rerank_info': {}
    }


def run_retrieval_grounded_hyde(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Experiment 3: Runs retrieval using Grounded HyDE.
    
    Grounded HyDE fixes the hallucination problem of standard HyDE by:
    1. First retrieving documents using the original query
    2. Generating a hypothetical answer GROUNDED in those retrieved documents
    3. Using that grounded answer for a second retrieval pass
    
    This prevents the LLM from inventing incorrect facts.
    """
    print("--- [EVAL] Running Retrieval (GROUNDED HyDE) ---")
    
    # STEP 1: Initial retrieval using original query
    print("  Step 1: Initial retrieval with original query...")
    initial_retrieval = query_processor.run_retrieval(
        query=query,
        batch_id=batch_id,
        user_profile=user_profile,
        skip_expansion=False,
        top_k=RETRIEVAL_CANDIDATE_POOL,
        allow_web_research=False,  # No web research for initial pass
    )
    
    # Extract the top 5 chunks to ground the hypothetical answer
    initial_chunks = initial_retrieval.get("rag_chunks_details", [])[:5]
    if not initial_chunks:
        print("  No initial chunks retrieved. Falling back to baseline retrieval.")
        return initial_retrieval
    
    # Format the context from initial retrieval
    context_parts = []
    for i, chunk in enumerate(initial_chunks, 1):
        content = chunk.get("content", "").strip()
        metadata = chunk.get("metadata", {})
        filename = metadata.get("filename", "Unknown")
        page = metadata.get("page_number", "N/A")
        context_parts.append(f"[Source {i}: {filename}, Page {page}]\n{content}")
    
    grounding_context = "\n\n---\n\n".join(context_parts)
    
    # STEP 2: Generate GROUNDED hypothetical answer
    print("  Step 2: Generating grounded hypothetical answer...")
    grounded_hyde_prompt = f"""Using ONLY the information provided in the documents below, write a short, factual answer to the question.

DO NOT invent facts. DO NOT make assumptions. If the documents don't contain the answer, say so.

DOCUMENTS:
{grounding_context}

QUESTION: {query}

ANSWER:"""
    
    try:
        hyde_response = query_processor.client.chat.completions.create(
            model=get_model_name("hyde"),
            messages=[{"role": "user", "content": grounded_hyde_prompt}],
            max_tokens=200,
            temperature=0.0
        )
        hyde_answer = None
        if hasattr(hyde_response, 'choices') and hyde_response.choices:
            first_choice = hyde_response.choices[0]
            if hasattr(first_choice, 'message') and getattr(first_choice.message, 'content', None) is not None:
                hyde_answer = first_choice.message.content
            elif getattr(first_choice, 'text', None) is not None:
                hyde_answer = first_choice.text
    except Exception as e:
        print(f"  Warning: Grounded HyDE generation failed: {e}")
        hyde_answer = None
    
    if not hyde_answer or not isinstance(hyde_answer, str) or hyde_answer.strip() == "":
        print("  Grounded HyDE returned no answer. Using initial retrieval.")
        # Attach hyde_answer (None or empty) for downstream callers to inspect
        try:
            initial_retrieval["hyde_answer"] = hyde_answer
        except Exception:
            pass
        return initial_retrieval
    
    print(f"  Grounded HyDE Answer: {hyde_answer[:100]}...")
    
    # STEP 3: Second retrieval pass using the grounded hypothetical answer
    print("  Step 3: Second retrieval pass with grounded answer...")
    final_retrieval = query_processor.run_retrieval(
        query=str(hyde_answer),
        batch_id=batch_id,
        user_profile=user_profile,
        skip_expansion=True,  # No expansion needed, HyDE answer is already detailed
        top_k=RETRIEVAL_CANDIDATE_POOL,
        allow_web_research=USE_WEB_RESEARCH,
    )

    # Include the grounded HyDE answer in the returned retrieval data so callers
    # (e.g., combined experiments) can re-use it for downstream steps like
    # re-ranking or debugging.
    try:
        final_retrieval["hyde_answer"] = hyde_answer
    except Exception:
        pass

    # Keep any high-quality chunks from the initial retrieval so we don't lose signals
    merged_chunks = _merge_chunk_lists(
        final_retrieval.get("rag_chunks_details", []),
        initial_retrieval.get("rag_chunks_details", []),
        max_chunks=RETRIEVAL_CANDIDATE_POOL if isinstance(RETRIEVAL_CANDIDATE_POOL, int) and RETRIEVAL_CANDIDATE_POOL > 0 else None
    )
    final_retrieval["rag_chunks_details"] = merged_chunks
    final_retrieval["rag_contexts_list"] = [chunk.get("content", "") for chunk in merged_chunks]

    return final_retrieval


def run_retrieval_advanced_fusion(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Advanced multi-source fusion: HyDE + 3-source RRF + cross-encoder reranking.
    
    Pipeline:
    1. Generate grounded HyDE answer
    2. Retrieve from 3 sources: FAISS(original), FAISS(HyDE), BM25(original)
    3. Apply RRF fusion across all 3 sources
    4. Cross-encoder rerank the fused candidates
    5. Return top-k for generation
    """
    print("--- [EVAL] Running Retrieval (ADVANCED FUSION) ---")
    
    # Step 1: Generate grounded HyDE answer
    print("  Step 1: Generating grounded HyDE answer...")
    hyde_data = run_retrieval_grounded_hyde(query_processor, query, batch_id, user_profile)
    hyde_answer = hyde_data.get("hyde_answer")
    
    if not hyde_answer:
        print("  No HyDE answer generated. Falling back to combined_best.")
        return run_retrieval_combined_best(query_processor, query, batch_id, user_profile)
    
    print(f"  HyDE answer: {hyde_answer[:80]}...")
    
    # Step 2: Multi-source retrieval
    print("  Step 2: Retrieving from 3 sources (FAISS-orig, FAISS-HyDE, BM25-orig)...")
    se = query_processor.search_engine
    if not se or not se.faiss_index or not se.bm25_index:
        print("  Search engine not ready. Falling back.")
        return hyde_data
    
    candidate_k = RETRIEVAL_CANDIDATE_POOL
    
    # Get results from each source
    faiss_orig = se._faiss_search(query, candidate_k)
    faiss_hyde = se._faiss_search(hyde_answer, candidate_k)
    bm25_orig = se._bm25_search(query, candidate_k)
    
    # Step 3: RRF fusion across 3 sources
    print("  Step 3: Applying 3-source RRF fusion...")
    k = RRF_FUSION_K
    
    # Build rank maps for each source
    def build_rank_map(results):
        rank_map = {}
        content_map = {}
        for i, r in enumerate(results):
            content = r.get("content")
            if content:
                rank_map[content] = i
                content_map[content] = r
        return rank_map, content_map
    
    faiss_orig_ranks, faiss_orig_map = build_rank_map(faiss_orig)
    faiss_hyde_ranks, faiss_hyde_map = build_rank_map(faiss_hyde)
    bm25_orig_ranks, bm25_orig_map = build_rank_map(bm25_orig)
    
    # Collect all unique content
    all_content = set(faiss_orig_ranks.keys()) | set(faiss_hyde_ranks.keys()) | set(bm25_orig_ranks.keys())
    
    # Calculate RRF scores
    rrf_scores = {}
    for content in all_content:
        rank_fo = faiss_orig_ranks.get(content, len(faiss_orig))
        rank_fh = faiss_hyde_ranks.get(content, len(faiss_hyde))
        rank_bo = bm25_orig_ranks.get(content, len(bm25_orig))
        
        rrf_score = (
            (1.0 / (k + rank_fo + 1)) +
            (1.0 / (k + rank_fh + 1)) +
            (1.0 / (k + rank_bo + 1))
        )
        rrf_scores[content] = rrf_score
    
    # Sort by RRF score and build result list
    sorted_content = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    fused_results = []
    for content, rrf_score in sorted_content:
        # Get metadata from any source that has this content
        if content in faiss_orig_map:
            metadata = faiss_orig_map[content].get("metadata", {})
            source = "faiss_orig"
        elif content in faiss_hyde_map:
            metadata = faiss_hyde_map[content].get("metadata", {})
            source = "faiss_hyde"
        else:
            metadata = bm25_orig_map[content].get("metadata", {})
            source = "bm25_orig"
        
        fused_results.append({
            "content": content,
            "score": rrf_score,
            "source": source,
            "metadata": metadata,
        })
    
    # Take top candidates for reranking
    top_m = min(20, len(fused_results))
    fused_top = fused_results[:top_m]
    print(f"  RRF fusion: {len(fused_results)} total → keeping top {top_m} for reranking")
    
    # Step 4: Cross-encoder reranking
    print("  Step 4: Cross-encoder reranking...")
    passages = [{"id": i, "text": r["content"]} for i, r in enumerate(fused_top)]
    
    try:
        reranked = reranker_manager.rerank(query, passages)
        reranked_list = list(reranked) if not isinstance(reranked, list) else reranked
        
        # Map back to original chunks
        final_chunks = []
        for item in reranked_list[:RERANK_KEEP_TOP_N]:
            idx = int(item.get("id")) if item.get("id") is not None else None
            if idx is not None and idx < len(fused_top):
                chunk = fused_top[idx].copy()
                chunk["rerank_score"] = item.get("score")
                final_chunks.append(chunk)
        
        print(f"  Reranked to {len(final_chunks)} chunks")
        
    except Exception as e:
        print(f"  Warning: Reranking failed ({e}). Using RRF results.")
        final_chunks = fused_top[:GENERATION_TOP_K]
    
    # Return standard retrieval data format
    return {
        "rag_chunks_details": final_chunks,
        "rag_contexts_list": [c.get("content", "") for c in final_chunks],
        "web_contexts_list": [],
        "web_research_raw": {},
        "hyde_answer": hyde_answer,
    }


def _apply_hyde_rerank(
    query_processor: QueryProcessor,
    query: str,
    hyde_retrieval: Dict[str, Any],
    rag_chunks_override: Optional[List[Dict[str, Any]]] = None,
    label: str = "Combined Best"
) -> Dict[str, Any]:
    """Apply FlashRank reranking to HyDE retrieval results, optionally overriding chunks."""

    rag_chunks = rag_chunks_override if rag_chunks_override is not None else hyde_retrieval.get("rag_chunks_details", [])

    if not rag_chunks:
        print("  No chunks from HyDE retrieval. Returning results as-is.")
        return hyde_retrieval

    hyde_answer = hyde_retrieval.get("hyde_answer") if isinstance(hyde_retrieval, dict) else None
    rerank_query = hyde_answer if hyde_answer else query
    print(f"  Using rerank query: {'HyDE answer' if hyde_answer else 'original query'}")

    passages = [{"id": i, "text": chunk.get("content", "")} for i, chunk in enumerate(rag_chunks)]

    try:
        if hyde_answer:
            try:
                reranked_hyde = reranker_manager.rerank(hyde_answer, passages)
                reranked_orig = reranker_manager.rerank(query, passages)
                reranked_hyde_list = list(reranked_hyde)
                reranked_orig_list = list(reranked_orig)
            except Exception as e:
                print(f"  Warning: Dual re-ranker call failed ({e}). Falling back to single rerank.")
                reranked = reranker_manager.rerank(rerank_query, passages)
                try:
                    reranked_list = list(reranked)
                except Exception:
                    reranked_list = reranked if isinstance(reranked, (list, tuple)) else []
            else:
                score_h = {
                    int(item.get('id')): float(item.get('score')) if isinstance(item, dict) and item.get('score') is not None else 0.0
                    for item in reranked_hyde_list if isinstance(item, dict) and item.get('id') is not None
                }
                score_o = {
                    int(item.get('id')): float(item.get('score')) if isinstance(item, dict) and item.get('score') is not None else 0.0
                    for item in reranked_orig_list if isinstance(item, dict) and item.get('id') is not None
                }

                try:
                    alpha = float(os.getenv("RERANK_HYDE_WEIGHT", "0.6"))
                except Exception:
                    alpha = 0.6

                combined_scores = {}
                for pid in set(list(score_h.keys()) + list(score_o.keys())):
                    combined_scores[pid] = alpha * score_h.get(pid, 0.0) + (1.0 - alpha) * score_o.get(pid, 0.0)

                reranked_list = [
                    {"id": pid, "score": combined_scores[pid]}
                    for pid in sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
                ]
        else:
            reranked = reranker_manager.rerank(rerank_query, passages)
            try:
                reranked_list = list(reranked)
            except Exception:
                reranked_list = reranked if isinstance(reranked, (list, tuple)) else []
    except Exception as e:
        print(f"  Warning: Re-ranker failed ({e}). Skipping re-ranking.")
        return hyde_retrieval

    rerank_info = {}
    for rank_pos, item in enumerate(reranked_list, start=1):
        idx = item.get('id') if isinstance(item, dict) else None
        score = item.get('score') if isinstance(item, dict) and 'score' in item else None
        if idx is None:
            continue
        score_value = float(score) if score is not None and hasattr(score, '__float__') else score
        rerank_info[int(idx)] = {"rerank_score": score_value, "rerank_rank": rank_pos}

    for i, chunk in enumerate(rag_chunks):
        info = rerank_info.get(i, {})
        chunk['rerank_score'] = info.get('rerank_score')
        chunk['rerank_rank'] = info.get('rerank_rank')

    keep_n = GENERATION_TOP_K if isinstance(GENERATION_TOP_K, int) and GENERATION_TOP_K > 0 else 5
    if isinstance(RERANK_KEEP_TOP_N, int) and RERANK_KEEP_TOP_N > 0:
        keep_n = min(keep_n, RERANK_KEEP_TOP_N)

    top_ids = [item.get('id') for item in reranked_list[:keep_n] if isinstance(item, dict) and item.get('id') is not None]
    reranked_indices = {int(i) for i in top_ids if i is not None}

    final_rag_chunks = [c for i, c in enumerate(rag_chunks) if i in reranked_indices]
    generation_limit = GENERATION_TOP_K if isinstance(GENERATION_TOP_K, int) and GENERATION_TOP_K > 0 else None
    final_rag_chunks = _merge_chunk_lists(final_rag_chunks, rag_chunks, max_chunks=generation_limit)

    hyde_retrieval["rag_chunks_details"] = final_rag_chunks
    hyde_retrieval["rag_contexts_list"] = [c.get("content", "") for c in final_rag_chunks]
    hyde_retrieval["rerank_info"] = rerank_info

    print(f"  {label}: Grounded HyDE supplied {len(rag_chunks)} chunks, re-ranked to {len(final_rag_chunks)} chunks.")
    return hyde_retrieval


def run_retrieval_combined_best(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Experiment: Combined Best - Uses Grounded HyDE followed by Re-ranking.
    
    This combines the two most successful techniques:
    - Grounded HyDE: Improves context recall and faithfulness
    - Re-ranking: Improves context precision
    
    Flow:
    1. Initial retrieval with original query
    2. Generate grounded hypothetical answer from top chunks
    3. Second retrieval pass using grounded answer (HyDE)
    4. Re-rank the results from step 3
    """
    print("--- [EVAL] Running Retrieval (COMBINED BEST: Grounded HyDE + Re-ranking) ---")
    
    # STEP 1-3: Run Grounded HyDE to get improved retrieval results
    print("  Phase 1: Running Grounded HyDE...")
    hyde_retrieval = run_retrieval_grounded_hyde(query_processor, query, batch_id, user_profile)

    # STEP 4: Apply re-ranking to the HyDE results
    print("  Phase 2: Applying re-ranking to HyDE results...")
    return _apply_hyde_rerank(query_processor, query, hyde_retrieval)


def run_retrieval_combined_best_rrf(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Experiment: Combined Best + RRF augmentation before reranking."""

    print("--- [EVAL] Running Retrieval (COMBINED BEST + RRF) ---")
    print("  Phase 1: Running Grounded HyDE...")
    hyde_retrieval = run_retrieval_grounded_hyde(query_processor, query, batch_id, user_profile)

    base_chunks = hyde_retrieval.get("rag_chunks_details", []) or []
    augmented_chunks = list(base_chunks)
    rrf_chunks: List[Dict[str, Any]] = []

    try:
        search_engine = getattr(query_processor, "search_engine", None)
        if search_engine and hasattr(search_engine, "hybrid_search_rrf"):
            print(f"  Augmenting with RRF fusion results (top {COMBINED_RRF_TOP_M})...")
            # Prefer the grounded HyDE answer for the RRF query so RRF can
            # enrich the HyDE-derived context set only when needed.
            query_for_rrf = hyde_retrieval.get("hyde_answer") or query
            rrf_results = search_engine.hybrid_search_rrf(
                query=query_for_rrf,
                top_k=RETRIEVAL_CANDIDATE_POOL,
                k=RRF_FUSION_K,
                ensure_top_sources=True,
                promote_order=[p.strip() for p in RRF_PROMOTE_ORDER.split(',') if p.strip()],
                force=RRF_FORCE,
                faiss_confidence_threshold=RRF_FAISS_CONFIDENCE_THRESHOLD,
            )
            for entry in rrf_results[:COMBINED_RRF_TOP_M]:
                rrf_chunks.append({
                    "content": entry.get("content", ""),
                    "metadata": entry.get("metadata", {}),
                    "score": entry.get("score"),
                    "source": entry.get("source", "rrf"),
                    "rrf_score": entry.get("score"),
                    "fusion_rank": entry.get("rank"),
                })
            if rrf_chunks:
                augmented_chunks = _merge_chunk_lists(base_chunks, rrf_chunks, max_chunks=RETRIEVAL_CANDIDATE_POOL)
                print(f"  Appended {len(rrf_chunks)} RRF chunks (candidate pool now {len(augmented_chunks)}).")
        else:
            print("  Warning: Search engine missing RRF support; skipping augmentation.")
    except Exception as e:
        print(f"  Warning: Failed to augment with RRF ({e}). Proceeding with HyDE chunks only.")
        augmented_chunks = base_chunks

    hyde_retrieval["rag_chunks_details"] = augmented_chunks
    hyde_retrieval["rag_contexts_list"] = [c.get("content", "") for c in augmented_chunks]

    print("  Phase 2: Applying reranking to augmented HyDE results...")
    return _apply_hyde_rerank(query_processor, query, hyde_retrieval, rag_chunks_override=augmented_chunks, label="Combined Best + RRF")


# =========================================================================
# MAIN EVALUATION ORCHESTRATOR
# =========================================================================

def run_pipeline(
    questions_data: List[Dict],
    profiles_data: Dict,
    experiment_name: str,
    test_batch_id: str,
    cache_manager: Optional[CacheManager] = None
) -> List[Dict]:
    """
    Runs the RAG pipeline for all test questions and collects results
    based on the specified experiment.
    
    Args:
        cache_manager: Optional cache manager for caching pipeline results
    """
    print(f"\n{'='*70}")
    print(f"Initializing RAG pipeline for experiment: {experiment_name.upper()}")
    print(f"Test batch: {test_batch_id}")
    if cache_manager:
        print("Cache: ENABLED")
    else:
        print("Cache: DISABLED")
    print(f"{'='*70}\n")
    evaluation_start = time.perf_counter()
    
    batch_manager = BatchManager()
    query_processor = QueryProcessor(batch_manager)
    
    # Verify the batch exists
    # For the `no_rag` experiment we intentionally avoid loading the FAISS/BM25
    # indexes because we will read the raw documents directly. Skip the
    # heavy index load in that case to reduce startup cost.
    if experiment_name != "no_rag":
        if not query_processor._ensure_batch_loaded(test_batch_id):
            raise Exception(f"FATAL: Could not load batch '{test_batch_id}'. "
                           f"Please ensure the batch exists in batches/{test_batch_id}/")
        print(f"[OK] Successfully loaded test batch '{test_batch_id}'.\n")
    else:
        print(f"[NO_RAG] Skipping FAISS/BM25 index load for batch '{test_batch_id}' (using full docs).")

    if ENABLE_TPM_THROTTLE:
        throttle_delay = compute_safe_delay()
        if throttle_delay > 0:
            print(f"Throttling between questions by {throttle_delay:.2f}s to stay within the TPM budget.")
    else:
        throttle_delay = 0
        print("[TPM] Throttling disabled by default. Set ENABLE_TPM_THROTTLE=1 to re-enable fixed delays.")
    
    # Select the functions to run based on the experiment
    if experiment_name == "baseline":
        retrieval_func = run_retrieval_baseline
        generation_func = run_generation_baseline
    elif experiment_name == "no_rag":
        # For no_rag we provide the generator with the full documents (no chunking
        # / indexing). The generator will still run, but retrieval is a simple
        # document loader rather than a search over FAISS/BM25.
        retrieval_func = run_retrieval_docs_full
        generation_func = run_generation_no_rag
    elif experiment_name == "reranking":
        retrieval_func = run_retrieval_rerank
        generation_func = run_generation_baseline
    elif experiment_name == "rrf":
        retrieval_func = run_retrieval_rrf
        generation_func = run_generation_baseline
    elif experiment_name == "grounded_hyde":
        retrieval_func = run_retrieval_grounded_hyde
        generation_func = run_generation_baseline
    elif experiment_name == "combined_best":
        retrieval_func = run_retrieval_combined_best
        generation_func = run_generation_baseline
    elif experiment_name == "combined_best_rrf":
        retrieval_func = run_retrieval_combined_best_rrf
        generation_func = run_generation_baseline
    elif experiment_name == "advanced_fusion":
        retrieval_func = run_retrieval_advanced_fusion
        generation_func = run_generation_baseline
    elif experiment_name == "semantic_chunking":
        retrieval_func = run_retrieval_semantic_chunking
        generation_func = run_generation_baseline
    elif experiment_name == "semantic_reranking":
        retrieval_func = run_retrieval_semantic_rerank
        generation_func = run_generation_baseline
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    results = []
    
    for item in questions_data:
        question = item["question"]
        ground_truth = item["ground_truth"]
        profile_id = item.get("user_profile_id")
        user_profile = profiles_data.get(profile_id, {})
        question_id = item.get("question_id", "unknown")
        
        print(f"\n--- Processing Question: {question_id} ---")
        print(f"Q: {question}")
        question_start = time.perf_counter()
        
        # Check cache first
        cached_result = None
        if cache_manager:
            cached_result = cache_manager.get(
                question=question,
                batch_id=test_batch_id,
                experiment_name=experiment_name,
                user_profile_id=profile_id
            )
        
        retrieval_seconds = 0.0
        generation_seconds = 0.0
        cache_key = None
        from_cache = False
        if cached_result:
            # Use cached data
            retrieval_data = cached_result["retrieval_data"]
            generated_answer = cached_result["generated_answer"]
            retrieval_seconds = cached_result.get("retrieval_seconds", 0.0)
            generation_seconds = cached_result.get("generation_seconds", 0.0)
            cache_key = cached_result.get("_cache_key")
            from_cache = True
            print(f"[Using cached result] key={cache_key[:8] if cache_key else 'NA'}")
        else:
            # Run the pipeline
            try:
                # 1. Run Retrieval
                retrieval_start = time.perf_counter()
                retrieval_data = retrieval_func(query_processor, question, test_batch_id, user_profile)
                retrieval_seconds = time.perf_counter() - retrieval_start
                
                # 2. Run Generation
                generation_start = time.perf_counter()
                generated_answer = generation_func(query_processor, question, retrieval_data, user_profile)
                generation_seconds = time.perf_counter() - generation_start
                print(f"A: {generated_answer[:100]}...")
                
                # Save to cache
                if cache_manager:
                    cache_manager.put(
                        question=question,
                        batch_id=test_batch_id,
                        experiment_name=experiment_name,
                        retrieval_data=retrieval_data,
                        generated_answer=generated_answer,
                        user_profile_id=profile_id,
                        metadata={"question_id": question_id},
                        retrieval_seconds=retrieval_seconds,
                        generation_seconds=generation_seconds,
                    )
            
            except Exception as e:
                print(f"[Error] Error processing question: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next question
                continue
        
        # Trim contexts to avoid oversized evaluation prompts
        rag_contexts = retrieval_data["rag_contexts_list"][:MAX_CONTEXTS_FOR_RAGAS]
        web_contexts = retrieval_data["web_contexts_list"][:max(0, MAX_CONTEXTS_FOR_RAGAS - len(rag_contexts))]

        retrieval_data["rag_contexts_list"] = rag_contexts
        retrieval_data["rag_chunks_details"] = retrieval_data["rag_chunks_details"][:MAX_CONTEXTS_FOR_RAGAS]
        retrieval_data["web_contexts_list"] = web_contexts

        # Collate all contexts for RAGAS
        all_contexts = rag_contexts + web_contexts

        # For no_rag experiment, explicitly set contexts to empty
        if experiment_name == "no_rag":
            all_contexts = []

        if not all_contexts:
            print("[Warning] No context was retrieved.")
            # RAGAS requires at least one context (even if empty string) to avoid validation errors
            all_contexts = [""]
        else:
            print(f"[OK] Retrieved {len(all_contexts)} context chunks.")
        
        # 3. Store results for RAGAS and include retrieval metadata (rerank scores etc.)
        # Ensure any mapping-like fields use string keys so pyarrow/datasets can
        # serialize them reliably (pyarrow requires dict keys to be str/bytes).
        raw_rerank_info = retrieval_data.get("rerank_info", {}) or {}
        safe_rerank_info = {str(k): v for k, v in raw_rerank_info.items()}

        question_latency = time.perf_counter() - question_start
        results.append({
            "question": question,
            "answer": generated_answer,
            "contexts": all_contexts,
            "ground_truth": ground_truth,
            "question_id": question_id,
            "user_profile_id": profile_id,  # Add user_profile_id for RAGAS caching
            # include detailed retrieval info for debugging/analysis
            "rag_chunks": retrieval_data.get("rag_chunks_details", []),
            "rag_contexts": retrieval_data.get("rag_contexts_list", []),
            "web_research": retrieval_data.get("web_research_raw", {}),
            "rerank_info": safe_rerank_info,
            "latency_seconds": question_latency,
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": generation_seconds,
            "from_cache": from_cache,
            "cache_key": cache_key,
        })

        if throttle_delay > 0:
            print(f"Respecting the TPM delay budget: sleeping {throttle_delay:.2f}s before the next question.")
            time.sleep(throttle_delay)
    
    return results


def run_retrieval_semantic_chunking(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Experiment: Create/load a semantic-chunked batch (batch_id + '_semantic') and run retrieval against it.

    This function will create a new batch using semantic chunking if it doesn't exist yet. It is non-destructive to the original baseline batch.
    """
    print("--- [EVAL] Running Retrieval (SEMANTIC CHUNKING) ---")
    semantic_batch_id = f"{batch_id}_semantic"
    semantic_meta_path = Path("batches") / semantic_batch_id / "metadata.json"
    batch_meta_path = Path("batches") / batch_id / "metadata.json"

    # If semantic batch doesn't exist, do not create one at runtime.
    # This experiment relies on a precomputed semantic batch. Please run
    # `python create_semantic_batch.py` before running this experiment.
    if not semantic_meta_path.exists():
        if not batch_meta_path.exists():
            print(f"Original batch metadata not found: {batch_meta_path}. Falling back to baseline retrieval.")
            return run_retrieval_baseline(query_processor, query, batch_id, user_profile)

        # Do not attempt to create the semantic batch at runtime. Instead return
        # baseline retrieval and instruct the operator to run the pre-run tool.
        print(f"Semantic batch '{semantic_batch_id}' not found. Please create it using 'python create_semantic_batch.py' or use 'setup_batch.py' to precompute semantic chunks.")
        return run_retrieval_baseline(query_processor, query, batch_id, user_profile)

    # Ensure the query processor loads the semantic batch
    if not query_processor._ensure_batch_loaded(semantic_batch_id):
        print(f"Failed to load semantic batch '{semantic_batch_id}'. Falling back to baseline.")
        return run_retrieval_baseline(query_processor, query, batch_id, user_profile)

    # Run retrieval against the semantic batch (keep candidate pool and web research consistent)
    return query_processor.run_retrieval(
        query=query,
        batch_id=semantic_batch_id,
        user_profile=user_profile,
        skip_expansion=False,
        top_k=RETRIEVAL_CANDIDATE_POOL,
        allow_web_research=USE_WEB_RESEARCH,
    )


def run_retrieval_semantic_rerank(
    query_processor: QueryProcessor,
    query: str,
    batch_id: str,
    user_profile: Optional[Dict]
) -> Dict[str, Any]:
    """Experiment: semantic-chunking + reranking.

    Creates (if needed) a semantic-chunked batch and then runs the FlashRank
    re-ranker over the top candidates. This combines both strategies.
    """
    print("--- [EVAL] Running Retrieval (SEMANTIC + RE-RANKING) ---")

    # Run retrieval using semantic chunking (creates the semantic batch if needed)
    retrieval_data = run_retrieval_semantic_chunking(query_processor, query, batch_id, user_profile)

    # If no chunks, just return
    rag_chunks = retrieval_data.get("rag_chunks_details", [])
    if not rag_chunks:
        return retrieval_data

    passages = [{"id": i, "text": chunk.get("content", "")} for i, chunk in enumerate(rag_chunks)]

    try:
        reranked = reranker_manager.rerank(query, passages)
    except Exception as e:
        print(f"Warning: semantic rerank failed ({e}). Returning semantic retrieval only.")
        return retrieval_data

    try:
        reranked_list = list(reranked)
    except Exception:
        reranked_list = reranked if isinstance(reranked, (list, tuple)) else []

    rerank_info = {}
    for rank_pos, item in enumerate(reranked_list, start=1):
        idx = item.get('id') if isinstance(item, dict) else None
        score = item.get('score') if isinstance(item, dict) and 'score' in item else None
        if idx is None:
            continue
        score_value = float(score) if score is not None and hasattr(score, '__float__') else score
        rerank_info[int(idx)] = {"rerank_score": score_value, "rerank_rank": rank_pos}

    for i, chunk in enumerate(rag_chunks):
        info = rerank_info.get(i, {})
        chunk['rerank_score'] = info.get('rerank_score')
        chunk['rerank_rank'] = info.get('rerank_rank')

    keep_n = GENERATION_TOP_K if isinstance(GENERATION_TOP_K, int) and GENERATION_TOP_K > 0 else 5
    if isinstance(RERANK_KEEP_TOP_N, int) and RERANK_KEEP_TOP_N > 0:
        keep_n = min(keep_n, RERANK_KEEP_TOP_N)

    top_ids = [item.get('id') for item in reranked_list[:keep_n] if isinstance(item, dict) and item.get('id') is not None]
    reranked_indices = {int(i) for i in top_ids if i is not None}

    final_rag_chunks = [c for i, c in enumerate(rag_chunks) if i in reranked_indices]

    generation_limit = GENERATION_TOP_K if isinstance(GENERATION_TOP_K, int) and GENERATION_TOP_K > 0 else None
    final_rag_chunks = _merge_chunk_lists(final_rag_chunks, rag_chunks, max_chunks=generation_limit)

    retrieval_data["rag_chunks_details"] = final_rag_chunks
    retrieval_data["rag_contexts_list"] = [c.get("content", "") for c in final_rag_chunks]
    retrieval_data["rerank_info"] = rerank_info

    print(f"Semantic retrieval returned {len(rag_chunks)} chunks; after rerank keep={len(final_rag_chunks)}")
    return retrieval_data


# =========================================================================
# RAGAS EVALUATION & REPORTING
# =========================================================================


def compute_local_support_metrics(pipeline_results: List[Dict]) -> List[Dict]:
    """Compute lightweight lexical overlap metrics as a fast, local fallback.

    These metrics give a quick signal about whether ground-truth tokens appear in
    the retrieved contexts and generated answer without making additional LLM calls.
    The work is CPU-bound and safe to parallelize because it never touches external
    APIs, so we opportunistically fan it out using a thread pool when multiple
    questions are present.
    """

    if not pipeline_results:
        return []

    token_pattern = re.compile(r"\w+")

    def _compute(entry: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = entry.get("ground_truth", "") or ""
        contexts = entry.get("contexts", []) or []
        answer = entry.get("answer", "") or ""

        tokens = [t.lower() for t in token_pattern.findall(ground_truth) if len(t) > 2]
        unique_tokens = list(dict.fromkeys(tokens))  # preserve order for reproducibility
        token_count = len(unique_tokens)

        context_text = " ".join(contexts).lower()
        answer_text = answer.lower()

        if token_count:
            context_hits = sum(1 for token in unique_tokens if token in context_text)
            answer_hits = sum(1 for token in unique_tokens if token in answer_text)
            context_recall = context_hits / token_count
            answer_recall = answer_hits / token_count
        else:
            context_hits = answer_hits = 0
            context_recall = answer_recall = 0.0

        return {
            "question_id": entry.get("question_id") or "unknown",
            "ground_truth_token_count": token_count,
            "context_token_hits": context_hits,
            "answer_token_hits": answer_hits,
            "context_token_recall": round(context_recall, 4),
            "answer_token_recall": round(answer_recall, 4),
            "full_ground_truth_in_context": any(
                ground_truth.lower() in c.lower() for c in contexts if ground_truth
            ),
            "full_ground_truth_in_answer": bool(ground_truth and ground_truth.lower() in answer_text),
        }

    # ThreadPoolExecutor preserves ordering with map(), so table rendering remains deterministic
    max_workers = min(8, len(pipeline_results)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        metrics = list(executor.map(_compute, pipeline_results))

    return metrics


def print_local_metrics_summary(local_metrics: List[Dict[str, Any]]):
    """Print a compact summary of the local support metrics."""
    if not local_metrics:
        print("[Local Metrics] No pipeline results available for analysis.")
        return

    total = len(local_metrics)
    context_support = sum(1 for m in local_metrics if m["context_token_recall"] >= 0.3)
    answer_support = sum(1 for m in local_metrics if m["answer_token_recall"] >= 0.3)

    print(f"\n{'='*70}")
    print("Local Support Metrics (lexical overlap)")
    print(f"{'='*70}\n")
    print(f"Questions analysed: {total}")
    print(f"Context support (>=30% token recall): {context_support}/{total}")
    print(f"Answer support (>=30% token recall): {answer_support}/{total}")
    print("\nPer-question breakdown:")
    for metric in local_metrics:
        print(
            f"  - {metric['question_id']}: context_recall={metric['context_token_recall']:.2f}, "
            f"answer_recall={metric['answer_token_recall']:.2f}, "
            f"full_match_ctx={metric['full_ground_truth_in_context']}, "
            f"full_match_ans={metric['full_ground_truth_in_answer']}"
        )


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects into standard primitives."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]

    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode('utf-8')
        except Exception:
            import base64
            return base64.b64encode(obj).decode('ascii')

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, datetime):
        return obj.isoformat()

    try:
        return str(obj)
    except Exception:
        return None


def build_results_payload(
    experiment_name: str,
    batch_id: str,
    pipeline_results: List[Dict[str, Any]],
    local_metrics: List[Dict[str, Any]],
    evaluation_result: Any,
    evaluation_seconds: Optional[float]
) -> Dict[str, Any]:
    """Builds and sanitizes the JSON-serializable payload for a single experiment run."""
    ragas_metrics: Dict[str, Any] = {}
    if evaluation_result and hasattr(evaluation_result, 'to_pandas'):
        df = evaluation_result.to_pandas()
        ragas_metrics = {col: df[col].tolist() for col in df.columns if col in ['faithfulness', 'answer_relevancy', 'answer_correctness', 'context_precision', 'context_recall']}
        metric_cols = list(ragas_metrics.keys())
        ragas_metrics['summary'] = {
            col: {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
            }
            for col in metric_cols
        }

    latencies = []
    for entry in pipeline_results:
        latency_value = entry.get('latency_seconds')
        if latency_value is None:
            continue
        try:
            latencies.append(float(latency_value))
        except Exception:
            continue
    latency_summary = None
    if latencies:
        latency_summary = {
            'total_seconds': sum(latencies),
            'average_seconds': sum(latencies) / len(latencies),
            'min_seconds': min(latencies),
            'max_seconds': max(latencies),
            'per_question_seconds': latencies,
        }

    output_data = {
        'metadata': {
            'experiment': experiment_name,
            'batch_id': batch_id,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'num_questions': len(pipeline_results),
            'retrieval_candidate_pool': RETRIEVAL_CANDIDATE_POOL,
            'use_web_research': bool(USE_WEB_RESEARCH),
            'rerank_keep_top_n': RERANK_KEEP_TOP_N,
            'generation_top_k': GENERATION_TOP_K,
            'evaluation_seconds': evaluation_seconds,
            'latency_summary': latency_summary,
        },
        'ragas_metrics': ragas_metrics,
        'local_metrics': local_metrics or [],
        'pipeline_results': pipeline_results,
    }

    return sanitize_for_json(output_data)


def save_results(
    evaluation_result: Any,
    pipeline_results: List[Dict[str, Any]],
    experiment_name: str,
    batch_id: str,
    local_metrics: Optional[List[Dict[str, Any]]] = None,
    evaluation_seconds: Optional[float] = None,
    payload: Optional[Dict[str, Any]] = None
) -> str:
    """Saves the evaluation payload to JSON (single experiment)."""
    if payload is None:
        payload = build_results_payload(
            experiment_name,
            batch_id,
            pipeline_results,
            local_metrics or [],
            evaluation_result,
            evaluation_seconds
        )

    # Ensure output directory exists
    timestamp = payload['metadata']['timestamp']
    readable_timestamp = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d_%H-%M-%S')
    output_filename = f"evaluation/results/ragas_{experiment_name}_{batch_id}_{readable_timestamp}.json"
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {output_filename}")
    return output_filename


def _build_metric_comparison_table(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a table comparing key metrics across experiments."""
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']
    table: List[Dict[str, Any]] = []
    for metric in metrics:
        row: Dict[str, Any] = {'metric': metric}
        for payload in payloads:
            exp = payload.get('metadata', {}).get('experiment') or 'unknown'
            summary = payload.get('ragas_metrics', {}).get('summary', {}) or {}
            metric_stats = summary.get(metric) or {}
            row[exp] = metric_stats.get('mean')
        table.append(row)
    return table


def _build_response_comparison_table(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a table that aligns answers per question across experiments."""
    response_map: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        exp = payload.get('metadata', {}).get('experiment') or 'unknown'
        for entry in payload.get('pipeline_results', []):
            question_id = entry.get('question_id') or entry.get('question') or "unknown"
            row = response_map.setdefault(question_id, {
                'question_id': question_id,
                'question': entry.get('question'),
                'ground_truth': entry.get('ground_truth')
            })
            row[exp] = entry.get('answer')
    # Preserve deterministic order by sorting on question_id
    return [response_map[qid] for qid in sorted(response_map.keys())]


def save_aggregated_results(
    payloads: List[Dict[str, Any]],
    batch_id: str,
    runner_args: Dict[str, Any]
) -> str:
    """Saves a single JSON containing multiple experiments."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    readable_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiments = [payload['metadata'].get('experiment') for payload in payloads]
    experiments_str = '+'.join(sorted(experiments))  # Sort for consistent ordering
    
    output_filename = f"evaluation/results/ragas_report_{batch_id}_{readable_time}_{experiments_str}.json"
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    comparison_table = _build_metric_comparison_table(payloads)
    response_comparison = _build_response_comparison_table(payloads)

    aggregate = {
        'metadata': {
            'batch_id': batch_id,
            'timestamp': readable_time,  # Use readable format in metadata too
            'experiments': experiments,
            'dataset': EVAL_DATASET_PATH,
            'runner_args': runner_args,
        },
        'comparison_table': comparison_table,
        'response_table': response_comparison,
        'experiments': payloads,
    }

    sanitized = sanitize_for_json(aggregate)
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Aggregated results saved to: {output_filename}")
    return output_filename


def run_single_experiment(
    experiment_name: str,
    batch_id: str,
    questions: List[Dict[str, Any]],
    profiles: Dict[str, Any],
    cache_manager: Optional[CacheManager],
    skip_ragas: bool,
    clear_cache: bool
) -> Dict[str, Any]:
    """Runs the pipeline, evaluation, and local metrics for a single experiment."""
    if cache_manager and clear_cache:
        cache_manager.clear(experiment_name=experiment_name, batch_id=batch_id)

    pipeline_results = run_pipeline(questions, profiles, experiment_name, batch_id, cache_manager)
    local_metrics = compute_local_support_metrics(pipeline_results)

    if skip_ragas:
        evaluation_result = {}
        evaluation_seconds = 0.0
    else:
        evaluation_result, evaluation_seconds = run_ragas_evaluation(
            pipeline_results, experiment_name, batch_id, cache_manager
        )

    payload = build_results_payload(
        experiment_name,
        batch_id,
        pipeline_results,
        local_metrics,
        evaluation_result,
        evaluation_seconds
    )

    return {
        'experiment_name': experiment_name,
        'pipeline_results': pipeline_results,
        'local_metrics': local_metrics,
        'evaluation_result': evaluation_result,
        'evaluation_seconds': evaluation_seconds,
        'payload': payload,
    }

def check_cached_ragas_metrics(
    results: List[Dict],
    cache_manager: Optional[CacheManager],
    experiment_name: str,
    batch_id: str
) -> Optional[Dict[str, List[float]]]:
    """Check if all questions have cached RAGAS metrics.
    
    Returns:
        Dict mapping metric names to lists of scores if all cached, None otherwise
    """
    if not cache_manager:
        return None
    
    # Check if all questions have cached RAGAS metrics
    all_cached = True
    cached_metrics = {
        'faithfulness': [],
        'answer_relevancy': [],
        'context_precision': [],
        'context_recall': [],
        'answer_correctness': []
    }
    
    for result in results:
        question = result.get('question') or result.get('user_input') or ''
        question_id = result.get('question_id')
        user_profile_id = result.get('user_profile_id')  # May not exist in result dict
        
        # Try to get RAGAS metrics from cache
        metrics = cache_manager.get_ragas_metrics(
            question=question,
            batch_id=batch_id,
            experiment_name=experiment_name,
            user_profile_id=user_profile_id
        )
        
        if metrics:
            # Add metrics to our collection
            for metric_name in cached_metrics.keys():
                cached_metrics[metric_name].append(metrics.get(metric_name, 0.0))
        else:
            all_cached = False
            break
    
    if all_cached:
        print(f"[Cache HIT] All {len(results)} questions have cached RAGAS metrics!")
        return cached_metrics
    else:
        print(f"[Cache MISS] RAGAS metrics not fully cached, will compute...")
        return None


def run_ragas_evaluation(
    results: List[Dict],
    experiment_name: str,
    batch_id: str,
    cache_manager: Optional[CacheManager] = None
) -> tuple[Any, float]:
    """Runs RAGAS metrics on the collected results with validation and retries.

    This wrapper will attempt evaluate() up to 3 times and validate the returned
    DataFrame to ensure expected metric columns exist and are not all-NaN.
    On repeated failures it writes a debug artifact to `evaluation/results/` and returns {}.
    
    Args:
        cache_manager: Optional cache manager for caching RAGAS metrics
    """
    print(f"\n{'='*70}")
    print("Running RAGAS Evaluation Metrics (safe wrapper)...")
    print(f"{'='*70}\n")
    evaluation_start = time.perf_counter()
    
    # Check if all RAGAS metrics are cached
    cached_metrics = check_cached_ragas_metrics(results, cache_manager, experiment_name, batch_id)
    if cached_metrics:
        # Build a mock evaluation result using cached metrics
        print("RAGAS evaluation completed (from cache) and validated.\n")
        
        # Create a DataFrame-like structure that matches what evaluate() returns
        import pandas as pd
        df_data = {
            'user_input': [r['question'] for r in results],
            'retrieved_contexts': [r['contexts'] for r in results],
            'response': [r['answer'] for r in results],
            'reference': [r['ground_truth'] for r in results],
            'faithfulness': cached_metrics['faithfulness'],
            'answer_relevancy': cached_metrics['answer_relevancy'],
            'context_precision': cached_metrics['context_precision'],
            'context_recall': cached_metrics['context_recall'],
            'answer_correctness': cached_metrics['answer_correctness']
        }
        
        # Create a mock evaluation result object
        class MockEvaluationResult:
            def __init__(self, df):
                self._df = df
            
            def to_pandas(self):
                return self._df
        
        return MockEvaluationResult(pd.DataFrame(df_data)), time.perf_counter() - evaluation_start

    if not results:
        print("ERROR: No results to evaluate!")
        return {}, time.perf_counter() - evaluation_start

    dataset = Dataset.from_list(results)
    # Temporarily disable expensive metrics to avoid rate limits
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    required_cols = ['faithfulness', 'answer_relevancy', 'answer_correctness']

    attempts = 3
    last_exception = None
    
    # Configure RAGAS RunConfig: reduce parallelism (max_workers) to avoid hitting rate limits
    # and keep reasonable timeouts/retries. Setting max_workers low reduces concurrent LLM calls.
    run_config = RunConfig(
        timeout=120,     # seconds for a single operation (keep current value)
        max_retries=3,   # retry attempts on transient failures
        max_wait=60,     # maximum backoff wait between retries
        max_workers=1,   # REDUCED: force sequential execution to avoid rate limits and ensure stability
    )
    
    for attempt in range(1, attempts + 1):
        try:
            print(f"RAGAS evaluate() attempt {attempt}/{attempts}...")
            # Use a more cost- and rate-friendly model for RAGAS evaluation to avoid TPM limits
            # Switched to gpt-4o-mini to reduce token-per-minute usage during large evaluations
            # and explicitly request a single generation (n=1) to avoid multiple-completion requests
            llm = ChatOpenAI(model=get_model_name("evaluation"), temperature=0.0, n=1)
            evaluation_result = evaluate(
                dataset, 
                metrics=metrics, 
                llm=llm,
                run_config=run_config,  # Pass RunConfig to control timeout and concurrency
                raise_exceptions=False  # Critical: prevents 1 bad question from crashing the whole script
            )

            # Basic validation: must be convertible to pandas and contain required columns
            if not hasattr(evaluation_result, 'to_pandas'):
                raise ValueError("evaluate() returned unexpected type (missing to_pandas)")

            df = getattr(evaluation_result, 'to_pandas')()
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing metric columns in evaluation result: {missing}")

            # Check for all-NaN columns which indicate evaluator failure
            all_nan = [c for c in required_cols if df[c].isnull().all()]
            if all_nan:
                raise ValueError(f"Evaluator returned all-NaN for columns: {all_nan}")

            # Passed validation
            print("RAGAS evaluation completed and validated.")
            
            # Save RAGAS metrics to cache for each question
            if cache_manager:
                print("Saving RAGAS metrics to cache...")
                import pandas as pd
                for i, result in enumerate(results):
                    question = result.get('question') or result.get('user_input') or ''
                    # Try to get user_profile_id from different possible locations
                    user_profile_id = None
                    # First check if it's in the question data (from test_data)
                    for q_data in [r for r in results if r.get('question') == question]:
                        if 'user_profile_id' in q_data:
                            user_profile_id = q_data['user_profile_id']
                            break
                    
                    # Extract metrics for this question from the DataFrame
                    ragas_metrics = {
                        'faithfulness': float(df.iloc[i]['faithfulness']) if pd.notna(df.iloc[i]['faithfulness']) else 0.0,
                        'answer_relevancy': float(df.iloc[i]['answer_relevancy']) if pd.notna(df.iloc[i]['answer_relevancy']) else 0.0,
                        'context_precision': float(df.iloc[i]['context_precision']) if pd.notna(df.iloc[i]['context_precision']) else 0.0,
                        'context_recall': float(df.iloc[i]['context_recall']) if pd.notna(df.iloc[i]['context_recall']) else 0.0,
                        'answer_correctness': float(df.iloc[i]['answer_correctness']) if pd.notna(df.iloc[i]['answer_correctness']) else 0.0
                    }
                    
                    # Update cache entry with RAGAS metrics
                    cache_manager.update_ragas_metrics(
                        question=question,
                        batch_id=batch_id,
                        experiment_name=experiment_name,
                        ragas_metrics=ragas_metrics,
                        user_profile_id=user_profile_id
                    )
                print(f"[Cache] Saved RAGAS metrics for {len(results)} questions.")
            
            return evaluation_result, time.perf_counter() - evaluation_start

        except Exception as e:
            print(f"RAGAS evaluation attempt {attempt} failed: {e}")
            import traceback
            traceback.print_exc()
            last_exception = e
            # Exponential backoff before retry (5s, 10s, 20s)
            if attempt < attempts:
                backoff_time = 5 * (2 ** (attempt - 1))
                print(f"Waiting {backoff_time}s before retry...")
                time.sleep(backoff_time)
            continue

    # If we reach here, all attempts failed — write debug artifact
    debug_dir = Path("evaluation/results")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir / f"ragas_evaluator_debug_{experiment_name}_{batch_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    debug_payload = {
        "error": str(last_exception),
        "attempts": attempts,
        "num_results": len(results),
    }
    with open(debug_file, 'w') as f:
        json.dump(debug_payload, f, indent=2)

    print(f"[ERROR] RAGAS evaluation failed after {attempts} attempts. Debug saved to {debug_file}")
    return {}, time.perf_counter() - evaluation_start


def print_metrics_summary(
    evaluation_result: Any,
    pipeline_results: List[Dict[str, Any]],
    experiment_name: str,
    show_table: bool = True,
    evaluation_seconds: Optional[float] = None
):
    """Prints a combined table of pipeline results and RAGAS metrics.

    The function attempts to merge the per-question RAGAS metrics (if available)
    with the original pipeline results (question, generated answer, ground truth,
    contexts) and prints a readable table to the console followed by a compact
    metrics summary (means).
    """
    print(f"\n{'='*70}")
    print(f"RAGAS Evaluation Results: {experiment_name.upper()}")
    print(f"{'='*70}\n")

    # Build a dataframe from pipeline results (safe fallback if evaluator failed)
    pipeline_df = pd.DataFrame(pipeline_results)

    # Normalize some common column names
    # pipeline_df expected keys: question, answer, ground_truth, question_id, contexts
    if 'question' not in pipeline_df.columns and 'user_input' in pipeline_df.columns:
        pipeline_df = pipeline_df.rename(columns={'user_input': 'question'})

    # Shortening helpers
    def shorten_text(t, max_len=300):
        if t is None:
            return ''
        s = str(t)
        return (s[:max_len] + '...') if len(s) > max_len else s

    def contexts_count_preview(ctxs):
        try:
            if not ctxs:
                return '0'
            if isinstance(ctxs, (list, tuple)):
                preview = shorten_text(ctxs[0], 200)
                return f"{len(ctxs)} [{preview}]"
            # If it's a string, just shorten it
            return shorten_text(ctxs, 200)
        except Exception:
            return 'N/A'

    metrics_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']

    # If evaluation_result is valid, extract metrics df
    metrics_df = None
    if evaluation_result and hasattr(evaluation_result, 'to_pandas'):
        try:
            metrics_df = evaluation_result.to_pandas()
        except Exception:
            metrics_df = None

    # Build combined rows (preserve order)
    combined_rows = []
    num_rows = max(len(pipeline_df), 0)
    for i in range(num_rows):
        row: Dict[str, Any] = {}
        pr = pipeline_df.iloc[i].to_dict() if i < len(pipeline_df) else {}

        row['question_id'] = pr.get('question_id', pr.get('question_id', f'Q{i}'))
        row['question'] = shorten_text(pr.get('question') or pr.get('user_input') or '')
        row['answer'] = shorten_text(pr.get('answer') or pr.get('response') or '')
        row['ground_truth'] = shorten_text(pr.get('ground_truth') or pr.get('reference') or '')
        row['contexts'] = contexts_count_preview(pr.get('contexts') or pr.get('retrieved_contexts') or pr.get('rag_contexts') or [])

        # Merge metrics if available
        if metrics_df is not None and i < len(metrics_df):
            for col in metrics_cols:
                row[col] = metrics_df.iloc[i][col] if col in metrics_df.columns else None
        else:
            for col in metrics_cols:
                row[col] = None

        combined_rows.append(row)

    combined_df = pd.DataFrame(combined_rows)

    # Configure pandas display options for readability
    if show_table:
        with pd.option_context('display.max_colwidth', 200, 'display.width', 200):
            if combined_df.empty:
                print("No pipeline rows to display.")
            else:
                print("Full per-question results:")
                print(combined_df.to_string(index=False))
    else:
        print("[Info] Per-question table suppressed (show_table=False).")

    # Print compact metrics summary if we have metrics
    if metrics_df is not None and not metrics_df.empty:
        summary = {col: {'mean': float(metrics_df[col].mean()), 'std': float(metrics_df[col].std())} for col in metrics_cols if col in metrics_df.columns}
        print(f"\nMetric means (per-question):")
        for k, v in summary.items():
            print(f"  - {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")
    else:
        print("\nNo RAGAS metrics available to summarize.")

    if 'latency_seconds' in pipeline_df.columns:
        latency_series = pd.to_numeric(pipeline_df['latency_seconds'], errors='coerce').dropna()
        if not latency_series.empty:
            latency_total = float(latency_series.sum())
            latency_mean = float(latency_series.mean())
            latency_min = float(latency_series.min())
            latency_max = float(latency_series.max())
            print(f"\nLatency per question: total={latency_total:.2f}s, avg={latency_mean:.2f}s, min={latency_min:.2f}s, max={latency_max:.2f}s")

        if evaluation_seconds is not None:
            print(f"\nRAGAS evaluation wall-clock time: {evaluation_seconds:.2f}s")


# =========================================================================
# MAIN ENTRY POINT
# =========================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation Harness for RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --experiment baseline
  python run_evaluation.py --experiment reranking --batch_id my_policies
  python run_evaluation.py --experiment grounded_hyde --batch_id my_policies_large
  python run_evaluation.py --experiment no_rag --batch_id my_policies
        """
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=EXPERIMENT_CHOICES,
        help="The experiment to run (default: baseline)"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated list of experiments to run sequentially (overrides --experiment)"
    )
    parser.add_argument(
        "--run_all_experiments",
        action="store_true",
        help="Run every available experiment in a single execution (overrides --experiment)"
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        default=DEFAULT_TEST_BATCH_ID,
        help=f"The test batch ID to use (default: {DEFAULT_TEST_BATCH_ID})"
    )
    parser.add_argument(
        "--rerank_keep_n",
        type=int,
        default=None,
        help="Override the default number of top reranked chunks to keep (env RERANK_KEEP_TOP_N or default 5)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (force fresh pipeline runs)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before running evaluation"
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=None,
        help="Cache TTL in hours (default: never expire)"
    )
    parser.add_argument(
        "--skip_ragas",
        dest="skip_ragas",
        action="store_true",
        help="Skip the RAGAS evaluation step (fast mode) and only run pipeline + local metrics."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to evaluation dataset JSON (default: test_data/evaluation_dataset_auto_ragas.json)"
    )
    # Toggle to show or hide the per-question table output
    table_group = parser.add_mutually_exclusive_group()
    table_group.add_argument(
        "--show_table",
        dest="show_table",
        action="store_true",
        help="Show combined per-question table in output (default)"
    )
    table_group.add_argument(
        "--no_show_table",
        dest="show_table",
        action="store_false",
        help="Do not show the combined per-question table"
    )
    parser.set_defaults(show_table=True)
    
    args = parser.parse_args()
    # If user provided a custom dataset path, override the module-level EVAL_DATASET_PATH
    if getattr(args, 'dataset', None):
        from pathlib import Path
        candidate = Path(args.dataset)
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset file not found: {candidate}")
        # update the module-level constant to point to the provided dataset
        global EVAL_DATASET_PATH
        EVAL_DATASET_PATH = str(candidate)
        print(f"Using evaluation dataset: {EVAL_DATASET_PATH}")

    experiments_to_run: List[str]
    if args.run_all_experiments:
        experiments_to_run = EXPERIMENT_CHOICES
    elif getattr(args, 'experiments', None):
        experiments_to_run = [exp.strip() for exp in args.experiments.split(',') if exp.strip()]
        if not experiments_to_run:
            raise ValueError("--experiments requires at least one experiment name")
        invalid = [exp for exp in experiments_to_run if exp not in EXPERIMENT_CHOICES]
        if invalid:
            raise ValueError(f"Unknown experiments requested: {invalid}. Valid choices: {EXPERIMENT_CHOICES}")
    else:
        experiments_to_run = [args.experiment]
    
    # Load environment variables
    load_dotenv()
    # If the user requests a fast run that skips RAGAS evaluation, allow running
    # without an OPENAI_API_KEY (generation will likely fail but the per-question
    # table will still be printed). Otherwise require the key.
    if not os.getenv("OPENAI_API_KEY") and not getattr(args, 'skip_ragas', False):
        raise ValueError("OPENAI_API_KEY must be set in .env file")
    
    print(f"\n[RAGAS] Evaluation Harness")
    print(f"Batch ID: {args.batch_id}")
    print(f"Experiments: {', '.join(experiments_to_run)}")
    # Allow CLI override of rerank keep-n
    global RERANK_KEEP_TOP_N
    if getattr(args, 'rerank_keep_n', None) is not None:
        try:
            RERANK_KEEP_TOP_N = int(args.rerank_keep_n)
            print(f"Using rerank_keep_n={RERANK_KEEP_TOP_N}")
        except Exception:
            print(f"Invalid --rerank_keep_n value: {args.rerank_keep_n}; using default {RERANK_KEEP_TOP_N}")
    
    # Initialize cache manager
    cache_manager = None
    if not args.no_cache:
        cache_manager = CacheManager(ttl_hours=args.cache_ttl)
        print(f"[Cache] Initialized (TTL: {args.cache_ttl or 'never expire'})")
    else:
        print("[Cache] Disabled by --no-cache flag")
    
    # Load data
    questions, profiles = load_data()
    
    aggregated_payloads: List[Dict[str, Any]] = []

    for experiment_name in experiments_to_run:
        print(f"\n{'='*70}\nRunning experiment: {experiment_name.upper()}\n{'='*70}")
        result = run_single_experiment(
            experiment_name,
            args.batch_id,
            questions,
            profiles,
            cache_manager,
            args.skip_ragas,
            args.clear_cache
        )

        aggregated_payloads.append(result['payload'])

        print_metrics_summary(
            result['evaluation_result'],
            result['pipeline_results'],
            experiment_name,
            show_table=args.show_table,
            evaluation_seconds=result['evaluation_seconds']
        )
        print_local_metrics_summary(result['local_metrics'])

    runner_args = {
        'skip_ragas': args.skip_ragas,
        'no_cache': args.no_cache,
        'clear_cache': args.clear_cache,
        'cache_ttl': args.cache_ttl,
        'show_table': args.show_table,
        'rerank_keep_n': args.rerank_keep_n,
        'requested_experiments': args.experiments,
        'experiments_run': experiments_to_run,
        'run_all_experiments': args.run_all_experiments,
    }
    output_file = save_aggregated_results(aggregated_payloads, args.batch_id, runner_args)

    # Print cache statistics
    if cache_manager:
        cache_manager.print_stats()

    print(f"\n[OK] Evaluation complete!")
    if output_file:
        print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
