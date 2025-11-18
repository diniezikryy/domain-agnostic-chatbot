"""
Query Processor
Handles domain-agnostic query processing using hybrid FAISS + BM25 search.
Loads user profile for personalized responses within specific batches (e.g., 'my_policies').

            intent_response = self.client.chat.completions.create(
                model=self.streaming_model,
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI
from utils.openai_retry import wrap_openai_client

from batch_manager import BatchManager
from utils.search import HybridSearchEngine
from research import DeepResearch  
from utils.model_config import get_model_name


class QueryProcessor:
    # Static Keyword Dictionary for high-priority, known semantic gaps. Not sure if we want to keep.
    STATIC_KEYWORD_MAP = {
        "collision damage waiver": "rental vehicle excess",
        "cdw": "rental vehicle excess",
        "rental car insurance": "rental vehicle excess",
        "accident in singapore": "medical expenses while in Singapore",
        "motorcycle accident in singapore": "medical expenses while in Singapore",
        "collision damage waiver": "rental vehicle excess rental car insurance",
        "cdw": "rental vehicle excess rental car insurance",
        "rental car insurance": "rental vehicle excess collision damage waiver",
        "car rental excess": "rental vehicle excess",
        "rental vehicle": "rental vehicle excess",
    }

    def __init__(self, batch_manager: BatchManager):
        self.batch_manager = batch_manager
        self.search_engine = None
        self.current_batch_id = None
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Wrap the OpenAI client methods with retry/backoff to handle rate limits
        try:
            wrap_openai_client(self.client)
        except Exception as e:
            print(f"[openai_retry] Could not enable retry wrapper: {e}")
        self.intent_model = get_model_name("intent")
        self.expansion_model = get_model_name("expansion")
        self.generation_model = get_model_name("generation")
        self.streaming_model = get_model_name("stream")
        # self.user_profile = self._load_user_profile()  # Load profile on initialization

    # def _load_user_profile(self) -> Optional[Dict[str, Any]]:
    #     """Loads the user profile from user_profile.json in the project root."""
    #     profile_path = Path("user_profile_basic.json")
    #     if profile_path.exists():
    #         try:
    #             with open(profile_path, "r") as f:
    #                 profile = json.load(f)
    #                 print("User profile loaded successfully.")
    #                 return profile
    #         except json.JSONDecodeError:
    #             print("Error: user_profile.json is not valid JSON.")
    #             return None
    #         except Exception as e:
    #             print(f"Error loading user profile: {e}")
    #             return None
    #     else:
    #         # it's okay if the profile doesn't exist, just means no personalization
    #         print(
    #             "Info: user_profile.json not found. Proceeding without personalization."
    #         )
    #         return None

    def _ensure_batch_loaded(self, batch_id: str) -> bool:
        """Ensure the specified batch is loaded in the search engine."""
        # If already loaded, do nothing
        if self.current_batch_id == batch_id and self.search_engine:
            return True

        paths = self.batch_manager.get_batch_paths(batch_id)
        if not paths:
            print(f"Error: Batch '{batch_id}' configuration not found in registry.")
            return False

        print(f"Loading indexes for batch '{batch_id}'...")
        try:
            # Create a new search engine instance for the specified batch
            self.search_engine = HybridSearchEngine()
            success = self.search_engine.load_indexes(
                faiss_path=paths["faiss_index"], bm25_path=paths["bm25_index"]
            )

            if success:
                self.current_batch_id = batch_id
                print(f"Successfully loaded indexes for batch '{batch_id}'.")
                return True
            else:
                print(f"Error: Failed to load indexes for batch '{batch_id}'.")
                self.search_engine = None  # Ensure it's None if loading failed
                self.current_batch_id = None
                return False

        except Exception as e:
            print(f"Error initializing search engine for batch '{batch_id}': {e}")
            self.search_engine = None
            self.current_batch_id = None
            return False

    # =========================================================================
    # == NEW FUNCTIONS FOR RAGAS EVALUATION: Testable Retrieval & Generation ==
    # =========================================================================

    def run_retrieval(self, query: str, batch_id: str, user_profile: Optional[Dict] = None, skip_expansion: bool = False, top_k: int = 50, allow_web_research: Optional[bool] = None, faiss_weight: Optional[float] = None, bm25_weight: Optional[float] = None) -> Dict[str, Any]:
        """
        Runs the full retrieval pipeline (intent, expansion, RAG, and web research).
        Returns a dictionary containing all retrieved contexts.
        
        This is a NON-STREAMING, testable method designed for RAGAS evaluation.
        It separates the retrieval phase from generation, enabling metric calculation.
        """
        print(f"--- [EVAL] Running Retrieval for Query: {query} ---")
        
        # Ensure batch is loaded
        if not self._ensure_batch_loaded(batch_id):
            raise Exception(f"Could not load batch '{batch_id}'")
        
        # 1. Analyze Intent
        try:
            intent_response = self.client.chat.completions.create(
                model=self.intent_model,
                messages=[{"role": "user", "content": self._get_intent_prompt(query)}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            intent = json.loads(intent_response.choices[0].message.content)
        except Exception as e:
            print(f"Error analyzing intent: {e}")
            intent = {
                "needs_comparison": False,
                "asks_about_uncovered_features": False,
                "requires_external_info": False
            }

        # 2. Expand and Search Documents (RAG)
        if skip_expansion:
            print("Skipping query expansion (using raw input)...")
            expanded_query = query
        else:
            expanded_query = self._expand_query(query)

        # Use configurable top_k so the evaluation harness can control the
        # candidate pool size (important for fair experiments).
        raw_search_results = self.search_engine.hybrid_search(
            query=expanded_query, top_k=top_k, faiss_weight=faiss_weight, bm25_weight=bm25_weight
        )
        unique_results = self._deduplicate_results(raw_search_results)
        
        # 3. Determine if Web Research is Needed
        needs_research = (
            intent["needs_comparison"] or
            intent["asks_about_uncovered_features"] or
            intent["requires_external_info"] or
            len(unique_results) == 0
        )

        # 4. Run Web Research if Needed (but only if allowed/available)
        research_results = {"answer": "", "sources": []}
        tavily_available = os.getenv("TAVILY_API_KEY") is not None

        # allow_web_research parameter overrides environment availability and
        # lets the evaluation harness disable web calls to keep experiments fair
        # and reproducible.
        allow_web = allow_web_research if allow_web_research is not None else tavily_available

        if needs_research and allow_web:
            print("--- [EVAL] Triggering DeepResearch (Web Search) ---")
            try:
                researcher = DeepResearch()
                enhanced_query = self._format_enhanced_query(query, unique_results, intent)
                research_results = researcher.research(enhanced_query)
            except Exception as e:
                print(f"Error during deep research: {e}")
                research_results = {"answer": "", "sources": []}
        elif needs_research and not allow_web:
            print("--- [EVAL] Web research needed but disabled for this experiment, skipping ---")
        
        # 5. Collate all contexts for RAGAS
        rag_contexts = [chunk.get("content", "") for chunk in unique_results]
        web_contexts = [research_results["answer"]] if research_results.get("answer") else []

        return {
            "rag_chunks_details": unique_results,    # For generation (with metadata)
            "rag_contexts_list": rag_contexts,        # For RAGAS metrics
            "web_contexts_list": web_contexts,        # For RAGAS metrics
            "web_research_raw": research_results,     # For generation
            "intent": intent,                          # For diagnostics
        }

    def run_generation(self, query: str, rag_chunks: List[Dict], research_results: Dict, user_profile: Optional[Dict] = None) -> str:
        """
        Runs the generation step given a query and retrieved contexts.
        Returns a single, complete answer string.
        
        This is a NON-STREAMING version designed for RAGAS evaluation.
        It accepts already-retrieved contexts instead of doing retrieval internally.
        """
        print(f"--- [EVAL] Running Generation for Query: {query} ---")
        
        # 1. Format Document Context
        context_from_docs = self._format_rag_context_for_prompt(rag_chunks)

        # 2. Format Profile Context
        profile_info, salutation = self._format_profile_for_prompt(user_profile)

        # 3. Combine Web Research into the prompt
        research_context = ""
        if research_results and research_results.get("answer"):
            research_context = f"\n--- EXTERNAL WEB RESEARCH ---\n{research_results['answer']}\n--- END OF WEB RESEARCH ---"

        # 4. Create the Final Prompt
        prompt_instructions = f"""You are an expert financial advisor specializing in insurance policy analysis.
Your task is to answer the user's question using the provided documents AND external research if available.

User Question: {query}
{profile_info}

--- POLICY DOCUMENT CHUNKS ---
{context_from_docs if context_from_docs else "No relevant information found in policy documents."}
--- END OF DOCUMENTS ---
{research_context}

CRITICAL RESPONSE RULES:
1. Base your answer on BOTH document chunks and external research.
2. Prioritize document information if available.
3. Use the user's specific policy tier (e.g., "P PLUS") when citing benefits.
4. Cite document facts with [Source X: filename.pdf, Page Y].
5. If using web research, state that (e.g., "External research shows...").
6. Keep response concise and focused on the user's specific question."""

        # 5. Call OpenAI API (non-streaming)
        try:
            response = self.client.chat.completions.create(
                model=self.expansion_model,
                messages=[
                    {"role": "system", "content": "You are an expert financial advisor specializing in insurance policy analysis and benefits explanation."},
                    {"role": "user", "content": prompt_instructions},
                ],
                max_tokens=1500,
                temperature=0.0,
            )
            final_answer = response.choices[0].message.content.strip()
            return final_answer
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Error generating response: {e}"

    def _format_rag_context_for_prompt(self, rag_chunks: List[Dict]) -> str:
        """Formats RAG chunks into a prompt-friendly string with citations."""
        context_parts = []
        max_chunks = 30  # Your original limit
        for i, result in enumerate(rag_chunks[:max_chunks], 1):
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            if content:
                filename = metadata.get("filename", "Unknown")
                page = metadata.get("page_number", "N/A")
                source_ref = f"[Source {i}: {filename}, Page {page}]"
                context_parts.append(f"{source_ref}\n{content}")
        return "\n\n---\n\n".join(context_parts)

    def _format_profile_for_prompt(self, user_profile: Optional[Dict]) -> tuple:
        """Formats user profile into prompt-friendly string and salutation."""
        if user_profile:
            user_name = user_profile.get("name", "User")
            policy_tiers = user_profile.get("policy_tiers", {})
            profile_info = f"\n\nUSER PROFILE:\n- User Name: {user_name}"
            if policy_tiers:
                profile_info += "\n- Policy Tiers:"
                for policy, tier in policy_tiers.items():
                    profile_info += f"\n  - {policy}: {tier} plan"
        else:
            user_name = "User"
            profile_info = ""
        
        salutation = f"Hi {user_name.split()[0] if user_name != 'User' else 'there'},"
        return profile_info, salutation

    # =========================================================================

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content."""
        unique_results = []
        seen_content = set()
        for result in results:
            content = result.get("content", "")
            # Check for non-empty content before adding
            if content and content not in seen_content:
                unique_results.append(result)
                seen_content.add(content)
        return unique_results
    
    def _filter_results_by_profile(self, results: List[Dict], user_profile: Optional[Dict]) -> List[Dict]:
        """Filters search results to only include documents listed in the user profile."""
        # Only filter if a profile with policies_owned exists
        if not user_profile or "policies_owned" not in user_profile:
            print("Info: No 'policies_owned' found in profile. Returning all results.")
            return results

        owned_policies = set(user_profile["policies_owned"])
        if not owned_policies:
            print(
                "Warning: 'policies_owned' list is empty in profile. Returning all results."
            )
            return results

        filtered_results = []
        for result in results:
            # Ensure metadata and filename exist before checking
            metadata = result.get("metadata", {})
            filename = metadata.get("filename")
            if filename and filename in owned_policies:
                filtered_results.append(result)

        print(
            f"Filtered results to {len(filtered_results)} chunks based on {len(owned_policies)} owned policies."
        )
        return filtered_results

    def _expand_query(self, query: str) -> str:
        """
        Expands the user query using intelligent LLM rewriting for insurance domain.
        """
        try:
            expansion_prompt = f"""
            You are an insurance domain expert. A user is asking: "{query}"

            Generate a comprehensive search query that includes:
            1. The original terms from the user's question
            2. Official insurance terminology that means the same thing
            3. Common abbreviations and synonyms used in insurance policies
            4. Related concepts that might appear in policy documents

            For example:
            - "collision damage waiver" should include "rental vehicle excess", "CDW", "car rental insurance"
            - "medical coverage" should include "medical expenses", "healthcare benefits", "treatment costs"

            Focus on terms that would actually appear in insurance policy documents.
            Output only the search terms separated by spaces (no explanations):
            """

            response = self.client.chat.completions.create(
                # Use the expansion model for query expansion (gpt-4o-mini by default)
                model=self.expansion_model,
                messages=[{"role": "user", "content": expansion_prompt}],
                max_tokens=150,  # Increased for more comprehensive expansion
                temperature=0.1,
            )

            keywords = response.choices[0].message.content.strip()
            expanded_query = f"{query} {keywords}"

            print(f"Query intelligently expanded to: {expanded_query}")
            return expanded_query

        except Exception as e:
            print(f"Error during query expansion: {e}")
            return query  # Fallback to original query
        
    def process_query(self, query: str, batch_id: str = None, user_profile: Optional[Dict] = None) -> str:
        """Process a query and return the response."""
        try:
            # Determine the target batch
            target_batch = batch_id or self.batch_manager.get_default_batch()
            if not target_batch:
                return "Error: No batch specified and no default batch set."

            # Ensure the correct batch's indexes are loaded
            if not self._ensure_batch_loaded(target_batch):
                return f"Error: Failed to load or switch to batch '{target_batch}'."

            start_time = time.time()
            print(f"\nProcessing query for batch: {target_batch}")
            print(f"Query: {query}")

            # Expand query 
            expanded_query = self._expand_query(query)
            # expanded_query = ' '.join(expanded_query.split()[:20])  # Limit to 20 terms

            raw_search_results = self.search_engine.hybrid_search(
                query=expanded_query, top_k=50  # Use top_k = 20 to limit context size
            )
            print(
                f"Retrieved {len(raw_search_results)} raw results from hybrid search."
            )

            # is_personal_batch = (target_batch == "my_policies")
            is_personal_batch = target_batch.startswith("user_")

            # relevant_results = raw_search_results
            # if is_personal_batch:
            #     if user_profile:
            #         relevant_results = self._filter_results_by_profile(
            #             raw_search_results, user_profile
            #         )
            #     else:
            #         print(
            #             "Warning: Operating in personal batch mode but no user profile loaded."
            #         )

            # unique_results = self._deduplicate_results(relevant_results)
            unique_results = self._deduplicate_results(raw_search_results)
            print(
                f"Retained {len(unique_results)} unique relevant chunks after filtering/deduplication."
            )

            if not unique_results:
                # if is_personal_batch and self.user_profile:
                #     return f"Based on your profile, I couldn't find relevant information in your specific policy documents ('{', '.join(self.user_profile.get('policies_owned', []))}') for the question: '{query}'."
                if is_personal_batch:
                    return f"I couldn't find relevant information in your uploaded documents for the question: '{query}'."
                else:
                    return f"No relevant information found in the documents of batch '{target_batch}' for the question: '{query}'."

            response = self._generate_response(query, unique_results, is_personal_batch, user_profile)

            processing_time = time.time() - start_time
            print(f"Total processing time: {processing_time:.2f}s")

            return response

        except Exception as e:
            # Log the full error for debugging
            import traceback

            print(f"An unexpected error occurred in process_query: {e}")
            traceback.print_exc()
            return f"An error occurred while processing your query. Please check logs. Error: {e}"

    def _generate_response(
        self, original_query: str, search_results: List[Dict], is_personal_batch: bool, user_profile: Optional[Dict]
    ) -> str:
        """Generate comprehensive response using retrieved chunks and potentially user profile."""
        if not search_results:
            return "I couldn't find any relevant information in the documents to answer your question."

        context_parts = []
        max_chunks_for_context = 30
        cited_filenames = set()  # keeps track of which documents we found

        print(
            f"Building context from top {min(len(search_results), max_chunks_for_context)} chunks..."
        )
        for i, result in enumerate(search_results[:max_chunks_for_context], 1):
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            if content:
                filename = metadata.get("filename", "Unknown Document")
                page = metadata.get("page_number", "N/A")
                source_ref = f"[Source {i}: {filename}, Page {page}]"
                context_parts.append(f"{source_ref}\n{content}")
                cited_filenames.add(filename)

        if not context_parts:
            return "Error: Found relevant documents but failed to extract content for context."

        context_from_docs = "\n\n---\n\n".join(context_parts)

        # Enhanced profile handling with policy tiers
        if is_personal_batch and user_profile:
            user_name = user_profile.get("name", "User")
            policy_tiers = user_profile.get("policy_tiers", {})

            profile_info = f"\n\nUSER PROFILE:\n- User Name: {user_name}"
            if policy_tiers:
                profile_info += "\n- Policy Tiers:"
                for policy, tier in policy_tiers.items():
                    profile_info += f"\n  - {policy}: {tier} plan"
        else:
            user_name = "User"
            profile_info = ""

        salutation = f"Hi {user_name.split()[0] if user_name != 'User' else 'Hi'},"

        # Enhanced prompt with insurance terminology awareness and tier personalization
        prompt_instructions = f"""You are an expert financial advisor specializing in insurance policy analysis.
    Your task is to answer the user's question about their insurance coverage.

    IMPORTANT INSURANCE TERMINOLOGY EQUIVALENCE:
    - "Rental vehicle excess" = "Collision damage waiver (CDW)" = "Car rental insurance"
    - "Loss damage waiver (LDW)" = "Collision damage waiver (CDW)"
    - When a user asks about CDW and you find "rental vehicle excess" coverage, treat them as the same thing

    User Question: {original_query}
    {profile_info}

    POLICY DOCUMENT CHUNKS:
    --- START OF DOCUMENTS ---
    {context_from_docs}
    --- END OF DOCUMENTS ---

    CRITICAL RESPONSE RULES:
    1. **Base your answer on the document chunks above**
    2. **Use the user's specific policy tier** - if the profile shows they have a specific plan tier (e.g., "Prestige", "Platinum"), focus on that tier's benefits rather than listing all options
    3. **Understand insurance terminology equivalence** - connect related terms confidently
    4. **If you find relevant coverage information, explain it clearly with specific dollar amounts for their tier (e.g., "up to S$2,500")**
    5. **Cite every fact with [Source X: filename.pdf, Page Y]**
    6. **Be specific about coverage amounts, plan types, and conditions**
    7. **Start your response with: "{salutation}"**
    8. **Provide actionable advice based on the coverage found**
    9. **When multiple plan tiers are shown, focus on the user's specific tier from their profile**

    Generate a helpful response now:
    """

        # Call OpenAI API
        try:
            print("Sending request to OpenAI API...")
            if not self.client:
                raise ValueError("OpenAI client is not initialized.")

            response = self.client.chat.completions.create(
                model=self.generation_model,  # use gpt-3.5-turbo for cheaper model
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial advisor. Answer insurance questions using the provided document chunks, with proper citations. Always include specific coverage amounts when available in the documents. Understand that insurance terminology can have equivalent meanings (e.g., 'rental vehicle excess' = 'collision damage waiver'). Always personalize responses based on the user's specific policy tiers when available.",
                    },
                    {"role": "user", "content": prompt_instructions},
                ],
                max_tokens=1500,
                temperature=0.05,
            )

            final_answer = response.choices[0].message.content.strip()
            print("Received response from OpenAI API.")
            return final_answer

        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return "Sorry, I encountered an error while generating the response. Please try again later or check the system logs."

    def process_query_stream_refactored(self, query: str, batch_id: str = None, user_profile: Optional[Dict] = None):
        """Process a query using refactored methods and yield response chunks for streaming."""
        try:
            # Determine the target batch
            target_batch = batch_id or self.batch_manager.get_default_batch()
            if not target_batch:
                yield "data: " + json.dumps({"error": "No batch specified and no default batch set."}) + "\n\n"
                return

            # Ensure the correct batch's indexes are loaded
            if not self._ensure_batch_loaded(target_batch):
                yield "data: " + json.dumps({"error": f"Failed to load or switch to batch '{target_batch}'."}) + "\n\n"
                return

            print(f"Streaming query for batch: {target_batch} using refactored methods")

            # 1. Run retrieval to get contexts
            retrieval_data = self.run_retrieval(query, target_batch, user_profile)

            # 2. Stream the generation using the retrieved contexts
            yield from self._generate_response_stream_refactored(
                query=query,
                rag_chunks=retrieval_data["rag_chunks_details"],
                research_results=retrieval_data["web_research_raw"],
                user_profile=user_profile
            )

        except Exception as e:
            import traceback
            print(f"Error in process_query_stream_refactored: {e}")
            traceback.print_exc()
            yield "data: " + json.dumps({"error": f"An error occurred: {str(e)}"}) + "\n\n"

    def _generate_response_stream_refactored(self, query: str, rag_chunks: List[Dict], research_results: Dict, user_profile: Optional[Dict] = None):
        """Stream generation response using retrieved contexts."""
        try:
            # 1. Format Document Context
            context_from_docs = self._format_rag_context_for_prompt(rag_chunks)

            # 2. Format Profile Context
            profile_info, salutation = self._format_profile_for_prompt(user_profile)

            # 3. Combine Web Research into the prompt
            research_context = ""
            if research_results and research_results.get("answer"):
                research_context = f"\n--- EXTERNAL WEB RESEARCH ---\n{research_results['answer']}\n--- END OF WEB RESEARCH ---"

            # 4. Create the Final Prompt
            prompt_instructions = f"""
{salutation}
Your task is to answer the user's question using the provided documents AND external research.

User Question: {query}
{profile_info}

--- POLICY DOCUMENT CHUNKS ---
{context_from_docs if context_from_docs else "No relevant information found in policy documents."}
--- END OF DOCUMENTS ---
{research_context}

CRITICAL RESPONSE RULES:
1. Base your answer on BOTH document chunks and external research.
2. Prioritize document information if available.
3. Use the user's specific policy tier (e.g., "P PLUS") when citing benefits.
4. Cite document facts with [Source X: filename.pdf, Page Y].
5. If using web research, state that (e.g., "External research shows...").
"""

            # 5. Call OpenAI API with streaming
            try:
                response = self.client.chat.completions.create(
                    model=self.streaming_model,
                    messages=[
                        {"role": "system", "content": "You are an expert financial advisor..."},
                        {"role": "user", "content": prompt_instructions},
                    ],
                    max_tokens=1500,
                    temperature=0.0,
                    stream=True  # Enable streaming
                )

                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield "data: " + json.dumps({"content": content}) + "\n\n"

                # Signal completion
                yield "data: " + json.dumps({"done": True}) + "\n\n"

            except Exception as e:
                print(f"Error in OpenAI streaming call: {e}")
                yield "data: " + json.dumps({"error": f"OpenAI API error: {str(e)}"}) + "\n\n"

        except Exception as e:
            import traceback
            print(f"Error in _generate_response_stream_refactored: {e}")
            traceback.print_exc()
            yield "data: " + json.dumps({"error": f"An error occurred: {str(e)}"}) + "\n\n"

    def process_query_stream(self, query: str, batch_id: str = None, user_profile: Optional[Dict] = None):
        """Process a query and yield response chunks for streaming."""
        try:
            # Determine the target batch
            target_batch = batch_id or self.batch_manager.get_default_batch()
            if not target_batch:
                yield "data: " + json.dumps({"error": "No batch specified and no default batch set."}) + "\n\n"
                return

            # Ensure the correct batch's indexes are loaded
            if not self._ensure_batch_loaded(target_batch):
                yield "data: " + json.dumps({"error": f"Failed to load or switch to batch '{target_batch}'."}) + "\n\n"
                return

            start_time = time.time()
            print(f"\nProcessing query for batch: {target_batch}")
            print(f"Query: {query}")

            # Analyze query intent ---- for deep research part
            try:
                intent_response = self.client.chat.completions.create(
                    model=self.intent_model,  # gpt-3.5-turbo for cheaper model
                    messages=[{"role": "user", "content": self._get_intent_prompt(query)}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                intent = json.loads(intent_response.choices[0].message.content)
            except Exception as e:
                print(f"Error analyzing intent: {e}")
                intent = {
                    "needs_comparison": False,
                    "asks_about_uncovered_features": False,
                    "requires_external_info": False
                }

            expanded_query = self._expand_query(query)
            raw_search_results = self.search_engine.hybrid_search(
                query=expanded_query, top_k=50
            )
            print(f"Retrieved {len(raw_search_results)} raw results from hybrid search.")

            is_personal_batch = target_batch.startswith("user_")

            # relevant_results = raw_search_results
            # if is_personal_batch:
            #     if self.user_profile:
            #         relevant_results = self._filter_results_by_profile(
            #             raw_search_results
            #         )
            #     else:
            #         print(
            #             "Warning: Operating in personal batch mode but no user profile loaded."
            #         )

            unique_results = self._deduplicate_results(raw_search_results)
            print(
                f"Retained {len(unique_results)} unique relevant chunks after filtering/deduplication."
            )

            # Determine if we need deep research
            needs_research = (
                intent["needs_comparison"] or
                intent["asks_about_uncovered_features"] or
                intent["requires_external_info"] or
                len(unique_results) == 0
            )

            if not unique_results and not needs_research:
                if is_personal_batch:
                    error_msg = f"Based on your profile, I couldn't find relevant information in your specific policy documents ('{', '.join(self.user_profile.get('policies_owned', []))}') for the question: '{query}'."
                else:
                    error_msg = f"No relevant information found in the documents of batch '{target_batch}' for the question: '{query}'."
                yield "data: " + json.dumps({"content": error_msg, "done": True}) + "\n\n"
                return

            # initialize research results 
            research_results = {}
            if needs_research:
                try:
                    researcher = DeepResearch()
                    enhanced_query = self._format_enhanced_query(query, unique_results, intent)
                    research_results = researcher.research(enhanced_query)
                    # Stream research results first if we have them
                    if research_results.get('answer'):
                        yield "data: " + json.dumps({"content": "\nResearching additional information...\n"}) + "\n\n"
                        yield "data: " + json.dumps({"content": research_results['answer']}) + "\n\n"
                except Exception as e:
                    print(f"Error during deep research: {e}")
                    research_results = {"answer": "", "sources": []}

            # Stream RAG results if available
            if unique_results:
                for chunk in self._generate_response_stream(query, unique_results, is_personal_batch, user_profile):
                    yield chunk

            processing_time = time.time() - start_time
            print(f"Total processing time: {processing_time:.2f}s")
            yield "data: " + json.dumps({"done": True}) + "\n\n"

        except Exception as e:
            import traceback
            print(f"An unexpected error occurred in process_query_stream: {e}")
            traceback.print_exc()
            yield "data: " + json.dumps({"error": f"An error occurred while processing your query: {e}"}) + "\n\n"

    def _generate_response_stream(
        self, original_query: str, search_results: List[Dict], is_personal_batch: bool, user_profile: Optional[Dict]
    ):
        """Generate streaming response using retrieved chunks and potentially user profile."""
        if not search_results:
            yield "data: " + json.dumps({"content": "I couldn't find any relevant information in the documents to answer your question.", "done": True}) + "\n\n"
            return

        context_parts = []
        max_chunks_for_context = 30
        cited_filenames = set()

        print(
            f"Building context from top {min(len(search_results), max_chunks_for_context)} chunks..."
        )
        for i, result in enumerate(search_results[:max_chunks_for_context], 1):
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            if content:
                filename = metadata.get("filename", "Unknown Document")
                page = metadata.get("page_number", "N/A")
                source_ref = f"[Source {i}: {filename}, Page {page}]"
                context_parts.append(f"{source_ref}\n{content}")
                cited_filenames.add(filename)

        if not context_parts:
            yield "data: " + json.dumps({"content": "Error: Found relevant documents but failed to extract content for context.", "done": True}) + "\n\n"
            return

        context_from_docs = "\n\n---\n\n".join(context_parts)

        # Enhanced profile handling with policy tiers
        if is_personal_batch and user_profile:
            user_name = user_profile.get("name", "User")
            policy_tiers = user_profile.get("policy_tiers", {})

            profile_info = f"\n\nUSER PROFILE:\n- User Name: {user_name}"
            if policy_tiers:
                profile_info += "\n- Policy Tiers:"
                for policy, tier in policy_tiers.items():
                    profile_info += f"\n  - {policy}: {tier} plan"
        else:
            user_name = "User"
            profile_info = ""

        salutation = f"Hi {user_name.split()[0] if user_name != 'User' else 'Hi'},"

        # Enhanced prompt with insurance terminology awareness and tier personalization
        prompt_instructions = f"""You are an expert financial advisor specializing in insurance policy analysis.
    Your task is to answer the user's question about their insurance coverage.

    IMPORTANT INSURANCE TERMINOLOGY EQUIVALENCE:
    - "Rental vehicle excess" = "Collision damage waiver (CDW)" = "Car rental insurance"
    - "Loss damage waiver (LDW)" = "Collision damage waiver (CDW)"
    - When a user asks about CDW and you find "rental vehicle excess" coverage, treat them as the same thing

    User Question: {original_query}
    {profile_info}

    POLICY DOCUMENT CHUNKS:
    --- START OF DOCUMENTS ---
    {context_from_docs}
    --- END OF DOCUMENTS ---

    CRITICAL RESPONSE RULES:
    1. **Base your answer on the document chunks above**
    2. **Use the user's specific policy tier** - if the profile shows they have a specific plan tier (e.g., "Prestige", "Platinum"), focus on that tier's benefits rather than listing all options
    3. **Understand insurance terminology equivalence** - connect related terms confidently
    4. **If you find relevant coverage information, explain it clearly with specific dollar amounts for their tier (e.g., "up to S$2,500")**
    5. **Cite every fact with [Source X: filename.pdf, Page Y]**
    6. **Be specific about coverage amounts, plan types, and conditions**
    7. **Start your response with: "{salutation}"**
    8. **Provide actionable advice based on the coverage found**
    9. **When multiple plan tiers are shown, focus on the user's specific tier from their profile**

    Generate a helpful response now:
    """

        # Call OpenAI API with streaming
        try:
            print("Sending streaming request to OpenAI API...")
            if not self.client:
                raise ValueError("OpenAI client is not initialized.")

            stream = self.client.chat.completions.create(
                model=self.streaming_model,  # Using GPT-4 Turbo for good balance
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert financial advisor who gives clear, practical insurance advice. 
                        Focus on the most important information from policy documents.
                        Highlight key coverage amounts, benefits, and limitations.
                        Compare policies when relevant and explain terms simply.
                        Make clear recommendations based on the user's situation.
                        Always cite your sources from the documents."""
                    },
                    {"role": "user", "content": prompt_instructions},
                ],
                max_tokens=1500,  # Balanced token limit
                temperature=0.1,  # Lower temperature for more focused responses
                stream=True  # Enable streaming
            )

            print("Streaming response from OpenAI API...")
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield "data: " + json.dumps({"content": content}) + "\n\n"

            # Send final done message
            yield "data: " + json.dumps({"done": True}) + "\n\n"
            print("Finished streaming response from OpenAI API.")

        except Exception as e:
            print(f"Error during OpenAI API streaming call: {e}")
            yield "data: " + json.dumps({"error": "Sorry, I encountered an error while generating the response. Please try again later or check the system logs."}) + "\n\n"

    def _get_intent_prompt(self, query: str) -> str:
        """Get the prompt for analyzing query intent"""
        prompt_text = """Analyze this insurance-related query to determine its intent.
        
        Consider carefully:
        1. Does it ask for comparison with other policies or insurers?
           - Looking for alternatives
           - Asking about other companies
           - Wanting to compare benefits/prices
           
        2. Does it ask about features that might not be in user's policy?
           - New or additional coverage types
           - Optional riders or add-ons
           - Coverage upgrades or enhancements
           
        3. Does it need external information beyond policy documents?
           - General insurance concepts
           - Market rates or trends
           - Regulatory requirements
           - Industry standards
           - Medical/health information
        
        Respond with JSON containing these boolean fields:
        - needs_comparison
        - asks_about_uncovered_features
        - requires_external_info"""
        return f"{prompt_text}\n\nQuery: {query}"

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the search engine state."""
        if self.search_engine:
            return self.search_engine.get_stats()
        return {"error": "Search engine not initialized."}

    def _format_rag_context(self, results: List[Dict]) -> str:
        """Format RAG results as context for deep research while managing token limits"""
        from utils.embeddings import trim_prompt  # Import the trim function
        
        # Sort results by relevance score if available
        if results and "score" in results[0]:
            results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        
        max_chunks = min(len(results), getattr(self.settings, 'max_chunks_per_query', 20))
        context_parts = []
        total_length = 0
        max_length = 12000  # Increased limit for better context while still safe
        
        # First pass: Include high-priority chunks (policy details, coverage amounts)
        priority_keywords = ["coverage", "benefit", "limit", "sum assured", "premium", "claim"]
        
        def is_priority_chunk(content):
            return any(keyword in content.lower() for keyword in priority_keywords)
        
        # Process priority chunks first
        for result in results[:max_chunks]:
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            if content and is_priority_chunk(content):
                policy_name = metadata.get("filename", "Unknown Policy")
                page = metadata.get("page_number", "N/A")
                chunk = f"From {policy_name} (Page {page}):\n{content}"
                
                if total_length + len(chunk) < max_length:
                    context_parts.append(chunk)
                    total_length += len(chunk)
        
        # Second pass: Include remaining chunks if space allows
        for result in results[:max_chunks]:
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            if content and not is_priority_chunk(content):
                policy_name = metadata.get("filename", "Unknown Policy")
                page = metadata.get("page_number", "N/A")
                chunk = f"From {policy_name} (Page {page}):\n{content}"
                
                if total_length + len(chunk) > max_length:
                    # Try to include a trimmed version if it's an important chunk
                    if "policy" in content.lower() or "coverage" in content.lower():
                        chunk = trim_prompt(chunk, (max_length - total_length))
                        if chunk:
                            context_parts.append(chunk)
                    break
                
                context_parts.append(chunk)
                total_length += len(chunk)
        if not context_parts:
            return ""
            
        return "\n\n".join([
            "Information from your current policies:",
            "---",
            "\n\n".join(context_parts),
            "---"
        ])

    def _format_enhanced_query(self, query: str, rag_results: List[Dict], intent: Dict[str, bool]) -> str:
        """Format the query for deep research with RAG context"""
        rag_context = self._format_rag_context(rag_results)
        objectives = []

        if intent["needs_comparison"]:
            objectives.append("- Compare with similar policies from other insurers")
        if intent["asks_about_uncovered_features"]:
            objectives.append("- Find alternative policies that might cover these features")
        if intent["requires_external_info"]:
            objectives.append("- Research general information about this topic")

        if not objectives:
            objectives.append("- Provide relevant insurance information")

        return f"""User Query: {query}

Current Policy Information:
{rag_context}

Research Objectives:
{chr(10).join(objectives)}"""
    def _format_rag_context(self, results: List[Dict]) -> str:
        """Format RAG results as context for deep research"""
        context_parts = []
        for result in results:
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            if content:
                policy_name = metadata.get("filename", "Unknown Policy")
                page = metadata.get("page_number", "N/A")
                context_parts.append(f"From {policy_name} (Page {page}):\n{content}")
        
        if not context_parts:
            return ""
            
        return "\n\n".join([
            "Information from your current policies:",
            "---",
            "\n\n".join(context_parts),
            "---"
        ])
    # new mtd to perform enhanced deep research with RAG context
    async def _enhanced_deep_research(
        self, 
        query: str, 
        rag_results: List[Dict],
        intent: Dict[str, bool]
    ) -> Dict[str, Any]:
        """Perform deep research with RAG context"""
        researcher = DeepResearch()
        
        rag_context = self._format_rag_context(rag_results)
        
        enhanced_query = f"""
        User Query: {query}
        
        Current Policy Information:
        {rag_context}
        
        Research Objectives:
        {self._get_research_objectives(intent)}
        """
        
        return researcher.research(enhanced_query)

    def _get_research_objectives(self, intent: Dict[str, bool]) -> str:
        objectives = []
        if intent["needs_comparison"]:
            objectives.append("- Compare with similar policies from other insurers")
        if intent["asks_about_uncovered_features"]:
            objectives.append("- Find alternative policies that might cover these features")
        if intent["requires_external_info"]:
            objectives.append("- Research general information about this topic")
        
        return "\n".join(objectives) if objectives else "- Provide relevant insurance information"

    # new mtd to format combined RAG + research response
    def _format_combined_response(
        self, 
        query: str,
        rag_results: List[Dict],
        research_results: Dict[str, Any],
        intent: Dict[str, bool],
        is_personal_batch: bool
    ) -> str:
        """Format the combined RAG and research results"""
        if not rag_results and not research_results.get('answer'):
            return "I couldn't find any relevant information to answer your question."
            
        response_parts = []
        
        # Generate RAG response if we have results
        if rag_results:
            rag_response = self._generate_response(query, rag_results, is_personal_batch)
            if rag_response:
                response_parts.append("Based on your policy documents:\n" + rag_response)
            
        # Add research results if available
        if research_results and research_results.get('answer'):
            research_answer = research_results['answer'].strip()
            if research_answer:
                if intent["needs_comparison"] or intent["asks_about_uncovered_features"]:
                    response_parts.append("\nAdditional options to consider:\n" + research_answer)
                else:
                    response_parts.append("\nAdditional information from research:\n" + research_answer)
                
        # Format the final response
        if not response_parts:
            return "I apologize, but I couldn't find enough information to answer your question comprehensively."
            
        return "\n\n".join(response_parts)

    def process_query(self, query: str, batch_id: str = None) -> str:
        """Process a query and return the response."""
        try:
            # Determine the target batch
            target_batch = batch_id or self.batch_manager.get_default_batch()
            if not target_batch:
                return "Error: No batch specified and no default batch set."

            # Ensure the correct batch's indexes are loaded
            if not self._ensure_batch_loaded(target_batch):
                return f"Error: Failed to load or switch to batch '{target_batch}'."

            start_time = time.time()
            print(f"\nProcessing query for batch: {target_batch}")
            print(f"Query: {query}")

            # Analyze query intent synchronously for both standard and streaming responses
            try:
                intent_response = self.client.chat.completions.create(
                    model=self.intent_model,
                    messages=[{"role": "user", "content": self._get_intent_prompt(query)}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                intent = json.loads(intent_response.choices[0].message.content)
            except Exception as e:
                print(f"Error analyzing intent: {e}")
                intent = {
                    "needs_comparison": False,
                    "asks_about_uncovered_features": False,
                    "requires_external_info": False
                }

            expanded_query = self._expand_query(query)
            raw_search_results = self.search_engine.hybrid_search(
                query=expanded_query, top_k=50
            )
            print(f"Retrieved {len(raw_search_results)} raw results from hybrid search.")

            is_personal_batch = target_batch.startswith("user_")
            relevant_results = raw_search_results
            if is_personal_batch:
                if self.user_profile:
                    relevant_results = self._filter_results_by_profile(raw_search_results)
                else:
                    print("Warning: Operating in personal batch mode but no user profile loaded.")

            unique_results = self._deduplicate_results(relevant_results)
            print(f"Retained {len(unique_results)} unique relevant chunks after filtering/deduplication.")

            # Determine if we need deep research
            needs_research = (
                intent["needs_comparison"] or
                intent["asks_about_uncovered_features"] or
                intent["requires_external_info"] or
                len(unique_results) == 0
            )

            if not unique_results and not needs_research:
                if is_personal_batch and self.user_profile:
                    return f"Based on your profile, I couldn't find relevant information in your specific policy documents ('{', '.join(self.user_profile.get('policies_owned', []))}') for the question: '{query}'."
                else:
                    return f"No relevant information found in the documents of batch '{target_batch}' for the question: '{query}'."

            # Initialize research results
            research_results = {}
            if needs_research:
                try:
                    researcher = DeepResearch()
                    enhanced_query = self._format_enhanced_query(query, unique_results, intent)
                    research_results = researcher.research(enhanced_query)
                except Exception as e:
                    print(f"Error during deep research: {e}")
                    research_results = {"answer": "", "sources": []}

            # Generate response using both RAG and research results
            response = self._format_combined_response(
                query=query,
                rag_results=unique_results,
                research_results=research_results,
                intent=intent,
                is_personal_batch=is_personal_batch
            )

            processing_time = time.time() - start_time
            print(f"Total processing time: {processing_time:.2f}s")

            return response

        except Exception as e:
            import traceback
            print(f"An unexpected error occurred in process_query: {e}")
            traceback.print_exc()
            return f"An error occurred while processing your query. Please check logs. Error: {e}"
