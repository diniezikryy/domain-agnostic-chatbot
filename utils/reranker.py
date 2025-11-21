# utils/reranker.py

"""
Cross-Encoder Reranking
Implements semantic reranking using pre-trained cross-encoder models.
"""

from typing import List, Dict, Any, Optional


class CrossEncoderReranker:
    """
    Semantic reranking using Cross-Encoder models.
    
    Cross-encoders compute relevance scores by encoding query-document pairs
    together, providing more accurate relevance scores than bi-encoders but
    at higher computational cost.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Initialize cross-encoder model for reranking.
        
        Args:
            model_name: Hugging Face model identifier
                Recommended options:
                - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fast, 17MB)
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (balanced, 90MB) - DEFAULT
                - cross-encoder/ms-marco-MiniLM-L-12-v2 (accurate, 120MB)
                - cross-encoder/qnli-distilroberta-base (most accurate, 420MB)
            device: "cpu" or "cuda"
            batch_size: Batch size for encoding (higher = faster but more memory)
            max_length: Maximum token length for input sequences
        """
        print(f"[CrossEncoderReranker] Loading model: {model_name}")
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = device

        # Try to load the CrossEncoder (preferred). If unavailable or fails,
        # fall back to a bi-encoder based cosine-similarity scorer using
        # sentence-transformers' SentenceTransformer (less accurate but robust).
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name, max_length=max_length, device=device)
            self.mode = "cross"
            print(f"[CrossEncoderReranker] Cross-Encoder model loaded on {device}")
        except Exception as e:
            print(f"[CrossEncoderReranker] Cross-Encoder unavailable or failed to load: {e}")
            print("[CrossEncoderReranker] Falling back to bi-encoder scoring (SentenceTransformer)")
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as _np

                # Use a compact, fast bi-encoder for fallback
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.mode = "bi"
                # store a small helper for numpy operations
                self._np = _np
                print("[CrossEncoderReranker] Bi-encoder fallback loaded: all-MiniLM-L6-v2")
            except Exception as e2:
                # If even the fallback can't load, raise so caller can handle
                raise RuntimeError(f"Failed to initialize any reranker model: {e2}")
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        score_field: str = "rerank_score"
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder semantic similarity.
        
        Args:
            query: Original search query
            candidates: List of candidate documents
                Each dict must have 'content' field
            top_k: Number of top results to return (None = return all)
            score_field: Name for the new score field to add
        
        Returns:
            Reranked candidates sorted by cross-encoder score (highest first)
            Original candidates are modified in-place with new score field
        """
        if not candidates:
            return []

        # Build a list of candidates with content
        valid_candidates = [c for c in candidates if (c.get("content") or "").strip()]
        if not valid_candidates:
            print("[CrossEncoderReranker] Warning: No valid candidates with content")
            return []

        # Cross-encoder scoring (preferred)
        if getattr(self, "mode", None) == "cross":
            pairs = [[query, c.get("content", "")] for c in valid_candidates]
            print(f"[CrossEncoderReranker] Reranking {len(pairs)} candidates with Cross-Encoder...")
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            scores = [float(s) for s in scores]

        else:
            # Bi-encoder fallback: cosine similarity between query embedding and doc embeddings
            print(f"[CrossEncoderReranker] Reranking {len(valid_candidates)} candidates with bi-encoder fallback...")
            import numpy as np

            # Encode query and documents
            query_emb = self.model.encode([query], convert_to_numpy=True)
            docs = [c.get("content", "") for c in valid_candidates]
            doc_embs = self.model.encode(docs, convert_to_numpy=True, show_progress_bar=False)

            # Normalize and compute cosine similarities
            try:
                # Ensure arrays are 2D
                query_emb = np.asarray(query_emb).reshape(1, -1)
                doc_embs = np.asarray(doc_embs)
                # normalize
                qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
                dn = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
                scores = (dn @ qn.T).squeeze().tolist()
            except Exception:
                # fallback naive dot
                scores = [float((np.asarray(d) @ np.asarray(query_emb).T).squeeze()) for d in doc_embs]

        # Attach scores and sort
        for idx, candidate in enumerate(valid_candidates):
            candidate[score_field] = float(scores[idx])

        valid_candidates.sort(key=lambda x: x[score_field], reverse=True)

        if top_k is not None:
            return valid_candidates[:top_k]

        return valid_candidates
    
    def rank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rank a list of document strings (simpler interface).
        
        Args:
            query: Search query
            documents: List of document strings
            top_k: Number of top results to return
            return_scores: Whether to include scores in output
        
        Returns:
            List of dicts with keys: 'text', 'score' (if return_scores=True), 'rank'
        """
        if not documents:
            return []

        if getattr(self, "mode", None) == "cross":
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            results = []
            for idx, (doc, score) in enumerate(zip(documents, scores)):
                result = {"text": doc, "rank": idx}
                if return_scores:
                    result["score"] = float(score)
                results.append(result)

            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            for idx, result in enumerate(results):
                result["rank"] = idx

            if top_k is not None:
                return results[:top_k]
            return results

        # Bi-encoder fallback
        import numpy as np
        query_emb = self.model.encode([query], convert_to_numpy=True)
        docs_emb = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=False)
        query_emb = np.asarray(query_emb).reshape(1, -1)
        docs_emb = np.asarray(docs_emb)
        qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        dn = docs_emb / np.linalg.norm(docs_emb, axis=1, keepdims=True)
        scores = (dn @ qn.T).squeeze().tolist()

        results = []
        for idx, (doc, score) in enumerate(zip(documents, scores)):
            result = {"text": doc, "rank": idx}
            if return_scores:
                result["score"] = float(score)
            results.append(result)

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        for idx, result in enumerate(results):
            result["rank"] = idx

        if top_k is not None:
            return results[:top_k]

        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """Return information about the loaded model."""
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
        }
        if getattr(self, "mode", None) == "cross":
            info["max_length"] = getattr(self.model, "max_length", None)
            info["type"] = "cross-encoder"
        else:
            info["type"] = "bi-encoder-fallback"
        return info


class TieredReranker:
    """
    Tiered reranker for insurance domain.

    Sorts documents using domain-aware tiers (promotion) instead of applying
    large ad-hoc heuristic boosts. This helps prevent numeric score mismatch
    issues while emphasizing high-signal domain content.
    """
    def __init__(self):
        import re
        from config.optimization_settings import optimization_settings
        self.currency_re = re.compile(r"(\$\s?\d|sgd|sum insured|payout|benefit)", re.I)
        self.health_re = re.compile(r"(deductible|co-?insurance|out[- ]of[- ]pocket)", re.I)
        self.critical_terms_re = re.compile(r"(exclusion|pre-existing|waiting period|eligibility)", re.I)
        self.optimization_settings = optimization_settings

    def rerank(self, query: str, results: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
        if not results:
            return []

        q = query.lower()
        wants_ci = any(k in q for k in ["cancer", "ci", "critical illness", "major cancer"]) or "critical illness" in q
        wants_health = any(k in q for k in ["hospital", "treatment", "surgery", "warded", "deductible", "co-insurance"]) or "hospital" in q
        wants_amount = any(k in q for k in ["how much", "amount", "cost", "price", "pay", "$", "sum insured"]) or "$" in q

        tiered = []

        for idx, res in enumerate(results):
            meta = res.get("metadata", {}) or {}
            content_parts = [res.get("content") or ""]

            if getattr(self.optimization_settings, "INDEX_PARENT_SECTION_TEXT", True):
                parent_section = (res.get("metadata", {}) or {}).get("parent_section_text")
                if parent_section:
                    content_parts.append(parent_section)
            if getattr(self.optimization_settings, "INDEX_DOCUMENT_SUMMARY", True):
                doc_summary = (res.get("metadata", {}) or {}).get("document_summary")
                if doc_summary:
                    content_parts.append(doc_summary)

            content = "\n\n".join(x for x in content_parts if x).lower()
            plan_context = meta.get("plan_context") or []

            retrieval_score = (
                res.get("rrf_score")
                or res.get("cross_encoder_score")
                or res.get("combined_score")
                or res.get("score", 0)
                or 0.0
            )
            retrieval_score -= idx * 0.0001

            has_amount = bool(self.currency_re.search(content))
            has_health_terms = bool(self.health_re.search(content))
            has_critical_terms = bool(self.critical_terms_re.search(content))
            is_ci_policy = any("critical care enhancer" in pc.lower() for pc in plan_context)
            is_health_policy = any("supremehealth" in pc.lower() for pc in plan_context)
            has_specific_illness = any(word in content for word in ["major cancer", "critical illness", "bypass", "heart attack", "stroke"]) or False

            tier = 3
            if wants_ci and is_ci_policy and (has_amount or has_specific_illness or has_critical_terms):
                tier = 1
            elif wants_health and is_health_policy and (has_health_terms or has_amount):
                tier = 1
            elif wants_amount and has_amount and (is_ci_policy or is_health_policy):
                tier = 1
            elif (wants_ci and is_ci_policy) or (wants_health and is_health_policy):
                tier = 2
            elif has_amount or has_health_terms or has_critical_terms:
                tier = 2

            tiered.append((tier, retrieval_score, res))

        tiered.sort(key=lambda x: (x[0], -x[1]))
        return [r for _, _, r in tiered][:max_results]


# Export default instance for convenience
tiered_reranker = TieredReranker()
