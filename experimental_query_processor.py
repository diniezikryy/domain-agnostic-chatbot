"""
Experimental Query Processor
Simplified RAG processor for controlled experiments without user profile complexity.
Designed for systematic evaluation of different RAG pipeline configurations.
"""

import os
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from openai import OpenAI
from utils.search import HybridSearchEngine
from batch_manager import BatchManager
import re


class ExperimentalQueryProcessor:
    """
    Simplified query processor for RAG experiments.
    
    Key differences from production QueryProcessor:
    - No user profile integration
    - Parameterized components (model, strategy, etc.)
    - Returns (response, tokens, latency) tuple
    - Exposes retrieved contexts for evaluation
    - Simpler generation prompts for fair comparison
    """

    def __init__(
        self,
        batch_manager: BatchManager,
        generation_model: str = "gpt-4o-mini",
        retrieval_strategy: str = "hybrid",
        use_hyde: bool = False,
        use_reranking: bool = False,
        top_k: int = 5,
        user_profile: dict | None = None
    ):
        """
        Initialize experimental processor.
        
        Args:
            batch_manager: BatchManager instance
            generation_model: OpenAI model for generation (e.g., "gpt-4o-mini")
            retrieval_strategy: "hybrid" or "vector_only"
            use_hyde: Whether to use HyDE query transformation
            use_reranking: Whether to rerank results with CrossEncoder
            top_k: Number of top results to use for generation
            user_profile: Optional dict with user profile for personalization
        """
        self.batch_manager = batch_manager
        self.search_engine = None
        self.current_batch_id = None
        
        # Experiment parameters
        self.generation_model = generation_model
        self.retrieval_strategy = retrieval_strategy
        self.use_hyde = use_hyde
        self.use_reranking = use_reranking
        self.top_k = top_k
        
        # Optional user profile for personalization
        self.user_profile = user_profile or {}
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Store last retrieved contexts for evaluation
        self.last_retrieved_contexts = []

    def _ensure_batch_loaded(self, batch_id: str) -> bool:
        """Ensure the specified batch is loaded in the search engine."""
        if self.current_batch_id == batch_id and self.search_engine:
            return True

        paths = self.batch_manager.get_batch_paths(batch_id)
        if not paths:
            print(f"Error: Batch '{batch_id}' not found in registry.")
            return False

        print(f"Loading indexes for batch '{batch_id}'...")
        try:
            self.search_engine = HybridSearchEngine()
            success = self.search_engine.load_indexes(
                faiss_path=paths["faiss_index"],
                bm25_path=paths["bm25_index"]
            )

            if success:
                self.current_batch_id = batch_id
                print(f"Successfully loaded batch '{batch_id}'.")
                return True
            else:
                print(f"Failed to load indexes for batch '{batch_id}'.")
                self.search_engine = None
                self.current_batch_id = None
                return False

        except Exception as e:
            print(f"Error loading batch '{batch_id}': {e}")
            self.search_engine = None
            self.current_batch_id = None
            return False

    def _generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical answer using HyDE (Hypothetical Document Embeddings).
        This creates a dense representation closer to potential answer documents.
        """
        print("  Generating HyDE query...")
        prompt = f"""Generate a hypothetical document that would perfectly answer this question. 
Write it as if it's an excerpt from a policy document.

Question: {query}

Hypothetical Answer:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for fast HyDE generation
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0
            )
            hyde_answer = response.choices[0].message.content.strip()
            print(f"  HyDE query generated: {hyde_answer[:100]}...")
            return hyde_answer
        except Exception as e:
            print(f"  HyDE generation failed: {e}. Using original query.")
            return query

    def _rerank_results(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """
        Rerank search results using LLM-based relevance scoring (no external ML dependencies).
        
        Args:
            query: Original user query
            search_results: Initial search results
            
        Returns:
            Reranked and filtered results (top_k)
        """
        if not self.use_reranking or not search_results:
            return search_results[:self.top_k]

        print(f"  Reranking {len(search_results)} results with LLM...")
        
        # Limit to top 20 for scoring (balance quality vs. cost)
        candidates = search_results[:min(20, len(search_results))]
        
        try:
            scores = []
            for i, result in enumerate(candidates, 1):
                content = result.get('content', '').strip()[:500]  # Truncate to 500 chars to save tokens
                
                # Ask GPT to score relevance
                prompt = f"""Rate how relevant this document excerpt is to answering the user's question.
Respond with ONLY a number from 0-10 (0=irrelevant, 10=highly relevant).

QUESTION: {query}

DOCUMENT: {content}

RELEVANCE SCORE:"""

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0
                )
                
                # Extract numeric score
                response_text = response.choices[0].message.content.strip()
                match = re.search(r'\d+', response_text)
                score = float(match.group()) if match else 0.0
                
                result['rerank_score'] = score
                scores.append(score)
                
                if i == 1 or i % 5 == 0:
                    print(f"    Scored {i}/{len(candidates)}: {score:.1f}")
            
            # Sort by score descending
            sorted_results = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

            # Safely format top scores
            top1 = sorted_results[0]['rerank_score'] if len(sorted_results) > 0 else None
            top2 = sorted_results[1]['rerank_score'] if len(sorted_results) > 1 else None
            if top1 is not None and top2 is not None:
                print(f"  Reranking complete. Top scores: {top1:.1f}, {top2:.1f}")
            elif top1 is not None:
                print(f"  Reranking complete. Top scores: {top1:.1f}, N/A")
            else:
                print("  Reranking complete. No scores available.")

            # Return top-k with reranked results
            return sorted_results[:self.top_k]

        except Exception as e:
            print(f"  LLM reranking failed: {e}. Using original ranking.")
            return candidates[:self.top_k]

    def _retrieve_contexts(self, query: str, batch_id: str) -> List[Dict]:
        """
        Retrieve relevant contexts for a query.
        
        Returns:
            List of context dictionaries with 'content' and 'metadata'
        """
        # Determine retrieval query (original or HyDE)
        retrieval_query = query
        if self.use_hyde:
            retrieval_query = self._generate_hypothetical_answer(query)

        # Determine how many results to retrieve
        # Get more if we're going to rerank
        initial_k = 20 if self.use_reranking else self.top_k

        # Retrieve using specified strategy
        search_results = []
        if self.retrieval_strategy == "hybrid":
            search_results = self.search_engine.hybrid_search(
                query=retrieval_query,
                top_k=initial_k
            )
        elif self.retrieval_strategy == "vector_only":
            search_results = self.search_engine.vector_only_search(
                query=retrieval_query,
                top_k=initial_k
            )
        else:
            raise ValueError(f"Unknown retrieval strategy: {self.retrieval_strategy}")

        # Rerank if enabled
        final_results = self._rerank_results(query, search_results)
        
        return final_results

    def _generate_response(self, query: str, contexts: List[Dict]) -> Tuple[str, int]:
        """
        Generate response using retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved context chunks
            
        Returns:
            (response_text, tokens_used)
        """
        if not contexts:
            return "I couldn't find relevant information to answer your question.", 0

        # Build context string
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            content = ctx.get('content', '').strip()
            metadata = ctx.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            page = metadata.get('page_number', 'N/A')
            
            context_parts.append(
                f"[Context {i} - {filename}, Page {page}]\n{content}"
            )

        context_text = "\n\n".join(context_parts)

        # Simple, standard RAG prompt (no user profile complexity)
        prompt = f"""You are a helpful insurance advisor. Answer the user's question based on the provided policy document excerpts.

IMPORTANT RULES:
1. Only use information from the provided contexts
2. Cite sources using [Context X] notation
3. If the answer isn't in the contexts, say so clearly
4. Be specific and accurate

CONTEXT FROM POLICY DOCUMENTS:
{context_text}

USER QUESTION:
{query}

ANSWER:"""

        # Build personalized system message if profile provided
        system_message = "You are a precise insurance advisor. Base answers only on provided context."
        if self.user_profile:
            profile_summary = self._format_profile_context()
            system_message = f"You are a precise insurance advisor assisting {profile_summary}. Base answers only on provided context and consider the user's profile when relevant."

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )

            answer = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0

            return answer, tokens_used

        except Exception as e:
            print(f"Generation failed: {e}")
            return f"Error generating response: {e}", 0
    
    def _format_profile_context(self) -> str:
        """Format user profile into a concise context string."""
        if not self.user_profile:
            return ""
        
        profile = self.user_profile
        name = profile.get("name", "User")
        age = profile.get("age", "Unknown")
        location = profile.get("location", "")
        policies = profile.get("policies", [])
        
        policy_names = [p.get("name", "") for p in policies if p.get("name")]
        policy_str = ", ".join(policy_names) if policy_names else "no policies"
        
        context = f"{name}, age {age}"
        if location:
            context += f", based in {location}"
        context += f", with {policy_str}"
        
        return context

    def process_query(self, query: str, batch_id: str) -> Tuple[str, int, float]:
        """
        Process a query and return response with metrics.
        
        Args:
            query: User query
            batch_id: Batch to query
            
        Returns:
            (response, tokens_used, latency_seconds)
        """
        start_time = time.time()

        try:
            # Load batch
            if not self._ensure_batch_loaded(batch_id):
                return f"Error: Could not load batch '{batch_id}'", 0, 0

            # Retrieve contexts
            print(f"Processing query with {self.retrieval_strategy} retrieval...")
            contexts = self._retrieve_contexts(query, batch_id)

            if not contexts:
                self.last_retrieved_contexts = []
                return "No relevant documents found.", 0, time.time() - start_time

            # Store contexts for evaluation
            self.last_retrieved_contexts = contexts

            # Generate response
            response, tokens = self._generate_response(query, contexts)

            latency = time.time() - start_time
            print(f"  Completed in {latency:.2f}s, {tokens} tokens")

            return response, tokens, latency

        except Exception as e:
            import traceback
            print(f"Error processing query: {e}")
            traceback.print_exc()
            return f"Error: {e}", 0, time.time() - start_time

    def get_last_contexts(self) -> List[str]:
        """
        Get the contexts retrieved in the last query.
        Useful for evaluation metrics that need retrieved contexts.
        
        Returns:
            List of context strings (just the content, not metadata)
        """
        return [ctx.get('content', '') for ctx in self.last_retrieved_contexts]

    def get_last_contexts_full(self) -> List[Dict]:
        """
        Get full context objects (with metadata) from the last query.
        
        Returns:
            List of context dictionaries
        """
        return self.last_retrieved_contexts
