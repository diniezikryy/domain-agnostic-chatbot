"""
Query Processor
Handles domain-agnostic query processing using hybrid FAISS + BM25 search.
This is adapted from the original query_test.py but with SIT-specific logic removed.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

# Import utilities (to be created)
from utils.search import HybridSearchEngine
from utils.embeddings import EmbeddingGenerator
from batch_manager import BatchManager

class QueryProcessor:
    def __init__(self, batch_manager: BatchManager):
        self.batch_manager = batch_manager
        self.search_engine = None
        self.current_batch_id = None

        # Generic prompts (SIT-specific removed)
        self.rewrite_prompt_template = """
Rewrite this question to improve document retrieval from a knowledge base.
Make the query more specific and include relevant terminology.
Focus on the key concepts and important terms.

Original: {query}
Rewritten:
"""

        self.system_prompt = """
You are a knowledgeable assistant helping users find information from documents.
Answer questions clearly and accurately based on the provided context.
Always cite specific information from the documents when possible.
If you cannot find relevant information, say so clearly.
Keep your answers concise but comprehensive.
"""

    def _ensure_batch_loaded(self, batch_id: str) -> bool:
        """Ensure the specified batch is loaded in the search engine."""
        if self.current_batch_id == batch_id and self.search_engine:
            return True

        # Get batch paths
        paths = self.batch_manager.get_batch_paths(batch_id)
        if not paths:
            print(f"Batch '{batch_id}' not found")
            return False

        # Initialize search engine with batch data
        try:
            self.search_engine = HybridSearchEngine()
            success = self.search_engine.load_indexes(
                faiss_path=paths["faiss_index"],
                bm25_path=paths["bm25_index"]
            )

            if success:
                self.current_batch_id = batch_id
                return True
            else:
                print(f"Failed to load indexes for batch '{batch_id}'")
                return False

        except Exception as e:
            print(f"Error loading batch '{batch_id}': {e}")
            return False

    def process_query(self, query: str, batch_id: str = None) -> str:
        """Process a query and return the response."""
        try:
            # Determine batch to use
            target_batch = batch_id or self.batch_manager.get_current_batch()
            if not target_batch:
                return "No batch specified and no active batch found."

            # Ensure batch is loaded
            if not self._ensure_batch_loaded(target_batch):
                return f"Failed to load batch '{target_batch}'"

            start_time = time.time()

            # Step 1: Rewrite query for better retrieval
            rewritten_query = self._rewrite_query(query)
            print(f"Rewritten query: {rewritten_query}")

            # Step 2: Perform hybrid search
            search_results = self.search_engine.hybrid_search(
                query=rewritten_query,
                top_k=10  # Get top 10 results
            )

            if not search_results:
                return "No relevant documents found for your question."

            # Step 3: Generate response using retrieved context
            response = self._generate_response(query, search_results)

            processing_time = time.time() - start_time
            print(f"Processing time: {processing_time:.2f}s")

            return response

        except Exception as e:
            return f"Error processing query: {e}"

    def _rewrite_query(self, query: str) -> str:
        """Rewrite query for better retrieval (placeholder - implement with LangChain)."""
        # TODO: Implement with OpenAI/LangChain
        # For now, return original query with basic preprocessing
        return self._preprocess_query(query)

    def _preprocess_query(self, query: str) -> str:
        """Basic query preprocessing."""
        # Remove common stopwords for better keyword matching
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.lower().split()
        filtered_words = [word for word in words if word not in stopwords]
        return ' '.join(filtered_words) if filtered_words else query

    def _generate_response(self, original_query: str, search_results: List[Dict]) -> str:
        """Generate conversational response from search results using OpenAI."""
        if not search_results:
            return "I couldn't find any relevant information in the documents to answer your question."

        # Build context from top search results
        context_parts = []
        for result in search_results[:5]:  # Top 5 most relevant
            content = result.get('content', '').strip()
            if content:
                context_parts.append(content)

        if not context_parts:
            return "I found some documents but couldn't extract relevant information to answer your question."

        context = "\n\n".join(context_parts)

        # Create prompt for OpenAI
        prompt = f"""You are an expert insurance advisor helping colleagues understand policy details.

Based on the following policy documents, answer this question in a conversational, helpful way:

Question: {original_query}

Policy Information:
{context}

Instructions:
- Give a clear, direct answer (Yes/No/Partially, etc.)
- Explain the reasoning based on specific policy terms
- Quote relevant policy language when helpful
- Be conversational but professional
- If the information is unclear or contradictory, say so
- Keep the response focused and practical for an insurance worker

Answer:"""

        try:
            from openai import OpenAI
            from dotenv import load_dotenv
            load_dotenv()

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful insurance expert providing clear, accurate policy guidance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3  # Lower temperature for more consistent, factual responses
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            # Fallback to basic response if OpenAI fails
            print(f"OpenAI generation failed: {e}")
            return f"Based on the policy documents, I found relevant information but couldn't generate a complete response. Key details: {context[:300]}..."

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.search_engine:
            return self.search_engine.get_stats()
        return {}