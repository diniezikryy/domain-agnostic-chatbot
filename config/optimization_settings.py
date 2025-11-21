# config/optimization_settings.py

"""
Optimization Settings for Advanced RAG Pipeline
Configurations for Cross-Encoder reranking and Reciprocal Rank Fusion (RRF).
"""

import os

class OptimizationSettings:
    """Settings for optimized RAG pipeline components."""

    def __init__(self):
        # ============================================
        # CROSS-ENCODER RERANKING SETTINGS
        # ============================================
        
        # Model selection: Fast, balanced, or high-accuracy
        # Options (based on 2025 benchmarks):
        #   - cross-encoder/ms-marco-MiniLM-L-6-v2 (90MB, excellent speed/accuracy balance - DEFAULT)
        #   - bge-reranker-base (strong accuracy, moderate compute)
        #   - cross-encoder/ms-marco-MiniLM-L-12-v2 (120MB, higher accuracy)
        #   - zerank-1 (best-in-class accuracy, slower)
        # Research shows ms-marco-MiniLM-L-6-v2 provides optimal prod performance with NDCG@10 ~0.82+
        self.CROSS_ENCODER_MODEL = os.getenv(
            "CROSS_ENCODER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        # Batch size for cross-encoder processing
        self.CROSS_ENCODER_BATCH_SIZE = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "16"))
        
        # Maximum token length for cross-encoder input
        self.CROSS_ENCODER_MAX_LENGTH = int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512"))
        
        # Device: "cpu" or "cuda" (if GPU available)
        self.CROSS_ENCODER_DEVICE = os.getenv("CROSS_ENCODER_DEVICE", "cpu")
        
        # ============================================
        # RECIPROCAL RANK FUSION (RRF) SETTINGS
        # ============================================
        
        # RRF k constant (research-backed optimal value: 60)
        # Lower k (20-40): More weight to top-ranked items
        # Higher k (60-100): More democratic fusion, better for diverse sources
        # Default 60 is robust across diverse retrieval pipelines per 2025 benchmarks
        self.RRF_K_CONSTANT = int(os.getenv("RRF_K_CONSTANT", "60"))
        # Weighted RRF: give BM25 (keyword) more weight for insurance domain
        # Values should sum to 1.0 (default: bm25 dominates)
        self.RRF_BM25_WEIGHT = float(os.getenv("RRF_BM25_WEIGHT", "0.3"))
        self.RRF_VECTOR_WEIGHT = float(os.getenv("RRF_VECTOR_WEIGHT", "0.7"))
        
        # ============================================
        # RERANKING PIPELINE SETTINGS
        # ============================================
        
        # Size of candidate pool to retrieve before reranking
        # Research shows larger pools increase recall when paired with tighter final context windows
        # Default raised to 70 based on Nov 2025 ablations (Both pipeline) using heuristic sweep results
        self.RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "100"))
        
        # Final number of results to return after all reranking stages
        # Default lowered to 7 to reduce distractors in the generation context
        self.RERANK_OUTPUT_K = int(os.getenv("RERANK_OUTPUT_K", "10"))
        
        # Minimum cross-encoder score threshold to filter low-quality chunks
        # Chunks scoring below this threshold are discarded after cross-encoder reranking
        # Range: 0.0-1.0, default 0.0 (disabled) to 0.3 for moderate filtering
        # NOTE: Cross-encoder scores are often negative or low positive, so use conservative thresholds
        self.MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-10.0"))
        
        # ============================================
        # FEATURE FLAGS
        # ============================================
        
        # Enable/disable cross-encoder reranking
        self.USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "true").lower() == "true"
        
        # Enable/disable RRF fusion (if False, uses weighted sum from baseline)
        self.USE_RRF_FUSION = os.getenv("USE_RRF_FUSION", "true").lower() == "true"
        
        # Enable optimized pipeline globally
        self.ENABLE_OPTIMIZED_PIPELINE = os.getenv("ENABLE_OPTIMIZED_PIPELINE", "false").lower() == "true"

        # Fine-grained index controls
        # Control whether parent section text is appended to the search index
        self.INDEX_PARENT_SECTION_TEXT = os.getenv("INDEX_PARENT_SECTION_TEXT", "true").lower() == "true"
        # Control whether document-level summary is appended to the search index
        self.INDEX_DOCUMENT_SUMMARY = os.getenv("INDEX_DOCUMENT_SUMMARY", "true").lower() == "true"

        # (MIN_CHUNK_SCORE and post-generation support-check settings removed)

    def get_config_summary(self):
        """Return a dictionary of current optimization settings."""
        return {
            "cross_encoder_model": self.CROSS_ENCODER_MODEL,
            "cross_encoder_batch_size": self.CROSS_ENCODER_BATCH_SIZE,
            "cross_encoder_max_length": self.CROSS_ENCODER_MAX_LENGTH,
            "cross_encoder_device": self.CROSS_ENCODER_DEVICE,
            "rrf_k_constant": self.RRF_K_CONSTANT,
            "rerank_top_k": self.RERANK_TOP_K,
            "rerank_output_k": self.RERANK_OUTPUT_K,
            "use_cross_encoder": self.USE_CROSS_ENCODER,
            "use_rrf_fusion": self.USE_RRF_FUSION,
            "optimized_pipeline_enabled": self.ENABLE_OPTIMIZED_PIPELINE,
            "index_parent_section_text": self.INDEX_PARENT_SECTION_TEXT,
            "index_document_summary": self.INDEX_DOCUMENT_SUMMARY,
        }


# Create a global settings instance
optimization_settings = OptimizationSettings()
