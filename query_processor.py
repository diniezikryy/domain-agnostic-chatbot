"""
Query Processor
Handles domain-agnostic query processing using hybrid FAISS + BM25 search.
Loads user profile for personalized responses within specific batches (e.g., 'my_policies').

* This is the "Comprehensive" version that relies on the detailed user_profile.json
* to provide facts and uses the document chunks only for citation.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from openai import OpenAI

from utils.search import HybridSearchEngine
from batch_manager import BatchManager

class QueryProcessor:

    # Static Keyword Dictionary for high-priority, known semantic gaps. Not sure if we want to keep.
    STATIC_KEYWORD_MAP = {
        "collision damage waiver": "rental vehicle excess",
        "cdw": "rental vehicle excess",
        "rental car insurance": "rental vehicle excess",
        "accident in singapore": "medical expenses while in Singapore",
        "motorcycle accident in singapore": "medical expenses while in Singapore",
    }

    def __init__(self, batch_manager: BatchManager):
        self.batch_manager = batch_manager
        self.search_engine = None
        self.current_batch_id = None
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.user_profile = self._load_user_profile() # Load profile on initialization

    def _load_user_profile(self) -> Optional[Dict[str, Any]]:
        """Loads the user profile from user_profile.json in the project root."""
        profile_path = Path("user_profile.json")
        if profile_path.exists():
            try:
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                    print("User profile loaded successfully.")
                    return profile
            except json.JSONDecodeError:
                print("Error: user_profile.json is not valid JSON.")
                return None
            except Exception as e:
                print(f"Error loading user profile: {e}")
                return None
        else:
            # it's okay if the profile doesn't exist, just means no personalization
            print("Info: user_profile.json not found. Proceeding without personalization.")
            return None

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
                faiss_path=paths["faiss_index"],
                bm25_path=paths["bm25_index"]
            )

            if success:
                self.current_batch_id = batch_id
                print(f"Successfully loaded indexes for batch '{batch_id}'.")
                return True
            else:
                print(f"Error: Failed to load indexes for batch '{batch_id}'.")
                self.search_engine = None # Ensure it's None if loading failed
                self.current_batch_id = None
                return False

        except Exception as e:
            print(f"Error initializing search engine for batch '{batch_id}': {e}")
            self.search_engine = None
            self.current_batch_id = None
            return False

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content."""
        unique_results = []
        seen_content = set()
        for result in results:
            content = result.get('content', '')
            # Check for non-empty content before adding
            if content and content not in seen_content:
                unique_results.append(result)
                seen_content.add(content)
        return unique_results

    def _filter_results_by_profile(self, results: List[Dict]) -> List[Dict]:
        """Filters search results to only include documents listed in the user profile."""
        # Only filter if a profile with policies_owned exists
        if not self.user_profile or "policies_owned" not in self.user_profile:
            print("Info: No 'policies_owned' found in profile. Returning all results.")
            return results

        owned_policies = set(self.user_profile["policies_owned"])
        if not owned_policies:
            print("Warning: 'policies_owned' list is empty in profile. Returning all results.")
            return results

        filtered_results = []
        for result in results:
            # Ensure metadata and filename exist before checking
            metadata = result.get('metadata', {})
            filename = metadata.get('filename')
            if filename and filename in owned_policies:
                filtered_results.append(result)

        print(f"Filtered results to {len(filtered_results)} chunks based on {len(owned_policies)} owned policies.")
        return filtered_results

    def _expand_query(self, query: str) -> str:
        """
        Expands the user query using a hybrid approach:
        1. Check a static map for known, high-value synonyms.
        2. If no static match, fall back to an LLM for dynamic expansion.
        """

        original_query_lower = query.lower()
        static_keywords_to_add = set()

        for key, value in self.STATIC_KEYWORD_MAP.items():
            if key in original_query_lower:
                static_keywords_to_add.add(value)

        if static_keywords_to_add:
            expanded_query = f"{query} {' '.join(static_keywords_to_add)}"
            print(f"Query expanded (STATIC) to: {expanded_query}")
            return expanded_query

        print("No static keywords found, using dynamic LLM expansion...")
        try:
            expansion_prompt = f"""
            You are an insurance policy expert. A user is asking a question.
            Your task is to list 5 to 7 technical synonyms or related insurance policy terms
            for the concepts in the user's query to improve search.
            
            Focus on *policy benefit language*, not general chatter.
            Do not answer the question. Only output a list of 5-7 related keywords, separated by spaces.
            
            User Query: "{query}"
            
            Related Keywords:
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini", # Keep mini here, it's cheap and fast for this task
                messages=[{"role": "user", "content": expansion_prompt}],
                max_tokens=50,
                temperature=0.1,
            )

            keywords = response.choices[0].message.content.strip().replace(",", " ").replace("\n", " ")
            expanded_query = f"{query} {keywords}"

            print(f"Query expanded (DYNAMIC) to: {expanded_query}")
            return expanded_query

        except Exception as e:
            print(f"Error during query expansion: {e}")
            return query # Fallback to original query on error

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

            expanded_query = self._expand_query(query)

            raw_search_results = self.search_engine.hybrid_search(
                query=expanded_query, # Use the expanded query
                top_k=50
            )
            print(f"Retrieved {len(raw_search_results)} raw results from hybrid search.")

            is_personal_batch = (target_batch == "my_policies")

            relevant_results = raw_search_results
            if is_personal_batch:
                if self.user_profile:
                    relevant_results = self._filter_results_by_profile(raw_search_results)
                else:
                    print("Warning: Operating in personal batch mode but no user profile loaded.")

            unique_results = self._deduplicate_results(relevant_results)
            print(f"Retained {len(unique_results)} unique relevant chunks after filtering/deduplication.")

            if not unique_results:
                if is_personal_batch and self.user_profile:
                    return f"Based on your profile, I couldn't find relevant information in your specific policy documents ('{', '.join(self.user_profile.get('policies_owned',[]))}') for the question: '{query}'."
                else:
                    return f"No relevant information found in the documents of batch '{target_batch}' for the question: '{query}'."

            response = self._generate_response(query, unique_results, is_personal_batch)

            processing_time = time.time() - start_time
            print(f"Total processing time: {processing_time:.2f}s")

            return response

        except Exception as e:
            # Log the full error for debugging
            import traceback
            print(f"An unexpected error occurred in process_query: {e}")
            traceback.print_exc()
            return f"An error occurred while processing your query. Please check logs. Error: {e}"

    def _generate_response(self, original_query: str, search_results: List[Dict], is_personal_batch: bool) -> str:
        """Generate comprehensive response using retrieved chunks and potentially user profile."""
        if not search_results:
            return "I couldn't find any relevant information in the documents to answer your question."

        context_parts = []
        max_chunks_for_context = 15
        cited_filenames = set() # keeps track of which documents we found

        print(f"Building context from top {min(len(search_results), max_chunks_for_context)} chunks...")
        for i, result in enumerate(search_results[:max_chunks_for_context], 1):
            content = result.get('content', '').strip()
            metadata = result.get('metadata', {})
            if content:
                filename = metadata.get('filename', 'Unknown Document')
                page = metadata.get('page_number', 'N/A')
                source_ref = f"[Source {i}: {filename}, Page {page}]"
                context_parts.append(f"{source_ref}\n{content}")
                cited_filenames.add(filename)

        if not context_parts:
            return "Error: Found relevant documents but failed to extract content for context."

        context_from_docs = "\n\n---\n\n".join(context_parts)

        policy_data_string = ""
        user_name = "User" # Default fallback
        if is_personal_batch and self.user_profile:
            print("Including user profile and ALL structured policy data in the prompt.")
            profile_items = []
            policy_data_items = []

            user_name = self.user_profile.get('name', 'User')
            if user_name:
                profile_items.append(f"- User Name: {user_name}")

            # Inject ALL policy details from the comprehensive profile
            for policy in self.user_profile.get("policy_details", []):
                print(f"Injecting structured data for: {policy.get('filename')}")
                policy_data_items.append(json.dumps(policy, indent=2))

            # Add general profile info
            profile_info_string = "\n\nUSER PROFILE:\n" + "\n".join(profile_items)

            # Add the structured policy data
            if policy_data_items:
                policy_data_string = "\n\nSTRUCTURED USER POLICY DATA (FOR REASONING):\n" + "\n---\n".join(policy_data_items)

        salutation = f"Hi {user_name},"

        # --- Construct the Final Prompt ---
        prompt_instructions = f"""You are an expert financial advisor with STRICT EVIDENCE REQUIREMENTS.
Your task is to answer the user's question about their insurance portfolio.

You are given two types of context:
1.  **STRUCTURED USER POLICY DATA:** Clean JSON data for ALL of the user's policies. This is your primary source of truth for coverage details, benefit amounts, and what policies exist.
2.  **AVAILABLE DOCUMENTS:** A small set of messy, raw text chunks from the original PDF policy files. These are *only* for finding citations. They may not be complete and may be missing policies.

Your Task: Answer the following question:
>>> {original_query} <<<
{profile_info_string}
{policy_data_string}

AVAILABLE DOCUMENTS (FOR CITATION ONLY):
--- START OF DOCUMENTS ---
{context_from_docs}
--- END OF DOCUMENTS ---

CRITICAL RESPONSE RULES:
1.  **Trust Structured Data First:** Base your answer on the `STRUCTURED USER POLICY DATA`. This is the complete, correct information.
2.  **Find Citation in Documents:** After finding the answer in the structured data, you MUST try to locate supporting evidence for it in the `AVAILABLE DOCUMENTS`.
3.  **Cite Everything:** If you find a supporting citation, end the fact with [Source X: filename.pdf, Page Y].
4.  **Handle Missing Citations:** If you find information in the `STRUCTURED USER POLICY DATA` (e.g., 'rental_vehicle_excess: 1500') but CANNOT find a matching citation in the `AVAILABLE DOCUMENTS` (the raw text), you MUST state the fact and cite the `filename` from the structured data. Example: `...S$1,500 [Source: GREAT_TravelCare.pdf, from your policy profile]`.
5.  **Handle Missing Information:** If the information is not in the `STRUCTURED USER POLICY DATA` or the `AVAILABLE DOCUMENTS`, state that the information is not available.
6.  **Perform Calculations:** If the user provides numbers and the policy data provides coverage amounts, perform simple calculations to help the user.
7.  **Address the User By Name:** You MUST start the response with the exact salutation: "{salutation}". Do not invent a different name.
8.  **Be Comprehensive:** Check ALL policies in the `STRUCTURED USER POLICY DATA` for relevance to the user's question.
9.  **Add Sources Section:** After your complete answer, add a horizontal rule (---)

PROHIBITED:
- Answering without citations (must use [Source X] or [Source: filename.pdf, from your policy profile]).
- Using information *only* from the messy `AVAILABLE DOCUMENTS` if it contradicts the `STRUCTURED USER POLICY DATA`.
- Inventing a user name or a persona for yourself.
- Starting the response with any text other than the exact salutation: "{salutation}"

Generate the answer now following all rules:
"""

        # --- Call OpenAI API ---
        try:
            print("Sending request to OpenAI API...")
            if not self.client:
                 raise ValueError("OpenAI client is not initialized.")

            response = self.client.chat.completions.create(
                model="gpt-4o", # Keep gpt-4o for this complex reasoning task
                messages=[
                    {"role": "system", "content": "You are a precise, expert financial advisor. You answer questions by combining structured JSON data with citable text snippets from policy documents."},
                    {"role": "user", "content": prompt_instructions}
                ],
                max_tokens=1500,
                temperature=0.05,
                stop=None
            )

            final_answer = response.choices[0].message.content.strip()
            print("Received response from OpenAI API.")
            return final_answer

        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return "Sorry, I encountered an error while generating the response. Please try again later or check the system logs."

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the search engine state."""
        if self.search_engine:
            return self.search_engine.get_stats()
        return {"error": "Search engine not initialized."}