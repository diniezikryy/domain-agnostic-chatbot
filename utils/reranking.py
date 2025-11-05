"""
CrossEncoder Reranking Module
Uses semantic reranking to improve retrieval quality.
"""

from typing import List, Dict, Any, Optional


class CrossEncoderReranker:
    """Reranks search results using CrossEncoder models."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Singleton pattern with lazy initialization."""
        if cls._instance is None:
            cls._instance = super(CrossEncoderReranker, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker (lazy loaded)."""
        if not CrossEncoderReranker._initialized:
            self.model = None
            self.model_name = model_name
            self.available = False
            
            try:
                from sentence_transformers import CrossEncoder
                print(f"Loading CrossEncoder model: {model_name}...")
                self.model = CrossEncoder(model_name, max_length=512)
                self.available = True
                print("CrossEncoder reranker initialized successfully")
            except ImportError:
                print("Warning: sentence-transformers not installed. Reranking disabled.")
                print("Install with: pip install sentence-transformers")
            except Exception as e:
                print(f"Warning: CrossEncoder initialization failed - {e}")
                self.available = False
            
            CrossEncoderReranker._initialized = True
    
    def rerank(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results based on semantic similarity.
        
        Args:
            query: Original query
            results: List of search results with 'content' field
            top_k: Number of top results to return (None = all)
            
        Returns:
            Reranked list of results with added 'rerank_score' field
        """
        if not self.available or not self.model:
            print("Reranker not available, returning original order")
            return results
        
        if not results:
            return results
        
        try:
            # Prepare query-document pairs for reranking
            pairs = [[query, result.get('content', '')] for result in results]
            
            # Get reranking scores
            scores = self.model.predict(pairs)
            
            # Add rerank scores to results
            for result, score in zip(results, scores):
                result['rerank_score'] = float(score)
            
            # Sort by rerank score (descending)
            reranked = sorted(results, key=lambda x: x.get('rerank_score', -float('inf')), reverse=True)
            
            # Return top_k if specified
            if top_k:
                reranked = reranked[:top_k]
            
            print(f"Reranked {len(results)} results, returning top {len(reranked)}")
            return reranked
            
        except Exception as e:
            print(f"Warning: Reranking failed - {e}. Returning original order.")
            return results


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    """Get singleton instance of reranker."""
    return CrossEncoderReranker(model_name)
