"""
Query Processor
Handles domain-agnostic query processing using hybrid FAISS + BM25 search.
This is adapted from the original query_test.py but with SIT-specific logic removed.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

from openai import OpenAI

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

        # TODO - using insurance as a benchmark for now, need to generalize and make it domain-agnostic
        self.decomposition_prompt_template = """
        You are an expert system that decomposes a user's complex question into several simpler self-contained sub-questions.
        The goal is to generate questions that can be answered by a vector search against a knowledge base of insurance policy documents.
        
        - Do not answer the question.
        - Generate between 1 and 5 sub-questions.
        - Each sub-question should be a complete, standalone question.
        - The output MUST be a JSON list of strings.
        
        Example:
        User Question: "What are the pros and cons of SingLife's policy compared to FWD's, especially regarding pre-existing conditions?"
        Output: [
            "What are the pros of the SingLife Essential Critical Illness II policy?",
            "What are the cons of the SingLife Essential Critical Illness II policy?",
            "What are the pros of the FWD Critical Illness Plus policy?",
            "What are the cons of the FWD Critical Illness Plus policy?",
            "How does the SingLife policy handle pre-existing conditions?",
            "How does the FWD policy handle pre-existing conditions?"
        ]
        
        User Question: {query}
        Output:
        """

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-questions using a language model.
        """
        try:
            prompt = self.decomposition_prompt_template.format(query=query)
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)

            # finds list of qns within the json obj
            for key, value in result.items():
                if isinstance(value, list):
                    return value
            return []  # return empty if no list is found
        except Exception as e:
            print(f"Failed to decompose query: {e}")
            # Fallback to just using the original query in a list
            return [query]

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

            # Step 1: decomp query into sub-queries
            sub_queries = self._decompose_query(query)
            print(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")

            # Step 2: perform hybrid search for each sub-query and collect results
            all_results = []
            for sub_q in sub_queries:
                # preprocess each sub-query
                processed_sub_q = self._preprocess_query(sub_q)
                results = self.search_engine.hybrid_search(
                    query=processed_sub_q,
                    top_k=5
                )
                all_results.extend(results)

            # de-dup the results based on content to avoid redundancy
            unique_results = []
            seen_content = set()
            for result in all_results:
                content = result.get('content', '')
                if content not in seen_content:
                    unique_results.append(result)
                    seen_content.add(content)

            print(f"Retrieved {len(unique_results)} unique chunks from {len(all_results)} total.")

            if not unique_results:
                return "No relevant documents found for your question."

            # Step 3: Generate response using retrieved context
            response = self._generate_response(query, unique_results)

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

        # Build context from top search results WITH METADATA
        context_parts = []
        for i, result in enumerate(search_results[:5], 1):  # Top 5 most relevant
            content = result.get('content', '').strip()
            metadata = result.get('metadata', {})

            if content:
                # Include source information with each chunk
                filename = metadata.get('filename', 'Unknown')
                page = metadata.get('page_number', 'N/A')
                year = metadata.get('year', 'N/A')

                # Format: [Source 1: filename, Page X, Year YYYY]
                source_header = f"[Source {i}: {filename}, Page {page}"
                if year != 'N/A':
                    source_header += f", Year {year}"
                source_header += "]"

                context_parts.append(f"{source_header}\n{content}")

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
        - **Always cite your sources using the [Source X] references provided in the context above**
        - Format citations like: "According to Source 1 (filename, Page X)..." or "Source 2 indicates that..."
        - Include the specific source number, filename, and page number when making claims
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
                    {"role": "system", "content": "You are a helpful insurance expert providing clear, accurate policy guidance with proper citations."},
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