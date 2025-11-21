"""
Query Processor
Handles domain-agnostic query processing using hybrid FAISS + BM25 search.
Loads user profile for personalized responses within specific batches (e.g., 'my_policies').

All hardcoded values moved to config.py
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from openai import OpenAI
from openai import AsyncOpenAI
from functools import partial
try:
    # Prefer the LangChain Google wrapper for Gemini - keeps us compatible with installed langchain-core
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    # Fall back to raw google.generativeai if the LangChain wrapper is not available
    import google.generativeai as genai

from batch_manager import BatchManager
from utils.search import HybridSearchEngine
from src.intent_analyzer import IntentAnalyzer
from src.response_analyzer import ResponseAnalyzer
from config.settings import settings as config


class QueryProcessor:
    # def __init__(self, batch_manager: BatchManager):
    #     self.batch_manager = batch_manager
    #     self.search_engine = None
    #     self.current_batch_id = None
    #     self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #     self.deep_research_enabled = config.DEEP_RESEARCH_ENABLED

    # using gemini 2.5 flash
    def __init__(self, batch_manager: BatchManager):
        self.batch_manager = batch_manager
        self.search_engine = None
        self.current_batch_id = None

        # Accept both 'GOOGLE_API_KEY' and 'GEMINI_API_KEY' environment variables for flexibility
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        # Allow tests and scripts to force the generation provider via env var
        # GENERATION_PROVIDER='openai' will force OpenAI for generation, even
        # if Gemini keys are present.
        self.generation_provider = os.getenv("GENERATION_PROVIDER") or os.getenv("EVAL_LLM_PROVIDER", "gemini")
        self.model_name = "gemini-2.5-flash"
        self.deep_research_enabled = config.DEEP_RESEARCH_ENABLED
        # If we have the LangChain Google wrapper installed, create a Geminai LLM instance
        try:
            # Only initialize Gemini LLM if generation provider permits it
            if self.generation_provider.lower() != "openai" and ChatGoogleGenerativeAI is not None:
                self.gemini_llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=google_api_key,
                    temperature=config.RESPONSE_TEMPERATURE,
                )
            else:
                self.gemini_llm = None
        except Exception:
            self.gemini_llm = None

    async def run_retrieval(
        self,
        query: str,
        batch_id: str,
        user_profile: Optional[Dict] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Async method to run retrieval pipeline.

        Args:
            query: User query
            batch_id: Batch ID to search within
            user_profile: Optional user profile for personalization
            top_k: Number of top results to return

        Returns:
            List of retrieved documents with metadata
        """
        # The underlying search and expansion code is synchronous (uses
        # blocking OpenAI client and local FAISS/BM25 calls). To avoid
        # blocking the asyncio event loop, run the synchronous retrieval
        # pipeline in a thread executor.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._run_retrieval_sync, query, batch_id, user_profile, top_k
        )

    def _is_out_of_scope(self, query: str) -> bool:
        """
        Lightweight guard to block prompts that are clearly unrelated to insurance/policy Q&A.
        This avoids wasting retrieval/generation on code/math/general chit-chat.
        """
        q = (query or "").lower()

        insurance_signals = [
            "policy",
            "coverage",
            "covered",
            "cover",
            "claim",
            "claims",
            "premium",
            "benefit",
            "deductible",
            "co-insurance",
            "coinsurance",
            "exclusion",
            "rider",
            "sum insured",
            "health plan",
            "critical illness",
            "insurer",
            "travel",
            "trip",
            "accident",
            "hospital",
            # Plan/insurer names so short queries like "singlife vs fwd" are allowed
            "singlife",
            "fwd",
            "aia",
            "prudential",
            "manulife",
            "great eastern",
        ]

        # If it mentions any insurance-related term, allow it
        if any(sig in q for sig in insurance_signals):
            return False

        # General knowledge or non-insurance domains
        if any(
            phrase in q
            for phrase in [
                "who is",
                "president",
                "prime minister",
                "capital of",
                "population of",
                "weather",
                "forecast",
                "temperature",
                "stock price",
                "share price",
                "bitcoin",
                "crypto",
                "football",
                "soccer",
                "nba",
                "movie",
                "song",
                "lyrics",
                "restaurant",
                "visit",
                "travel to",
            ]
        ):
            return True

        out_of_scope_signals = [
            "python",
            "javascript",
            "code",
            "snippet",
            "algorithm",
            "sql",
            "database",
            "react",
            "typescript",
            "api design",
            "server",
            "docker",
            "kubernetes",
            "math",
            "equation",
            "poem",
            "story",
            "song",
            "lyrics",
        ]

        if any(sig in q for sig in out_of_scope_signals):
            return True

        # Default behavior: if it's not clearly insurance, treat it as out of scope
        return True

    def _run_retrieval_sync(
        self,
        query: str,
        batch_id: str,
        user_profile: Optional[Dict] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Synchronous retrieval pipeline (intended to be run in a thread).

        This mirrors the previous implementation of run_retrieval but stays
        synchronous so it can safely call blocking libraries (OpenAI sync
        client, FAISS, BM25) without hanging the event loop.
        """
        target_batch = batch_id or self.batch_manager.get_default_batch()
        if not target_batch:
            raise ValueError("No batch specified and no default batch set.")

        if not self._ensure_batch_loaded(target_batch):
            raise RuntimeError(f"Failed to load batch '{target_batch}'.")

        expanded_query = self._expand_query(query)
        is_personal_batch = target_batch.startswith("user_")

        # Use a larger pool for retrieval to allow reranker to work effectively
        # We want to retrieve enough candidates (e.g. 60) and then rerank/filter down to top_k
        search_pool_size = max(top_k, config.SEARCH_TOP_K)

        if is_personal_batch and user_profile:
            # For multi-policy, we want to ensure we get enough chunks per policy
            num_policies = len(
                user_profile.get("insurance_policies", {}) or {"default": None}
            )
            # Ensure at least 10 chunks per policy to give reranker enough material
            chunks_per_policy = max(search_pool_size // max(num_policies, 1), 10)

            raw_results = self._multi_policy_search(
                query=query,
                expanded_query=expanded_query,
                user_profile=user_profile,
                chunks_per_policy=chunks_per_policy,
            )
        else:
            raw_results = self.search_engine.hybrid_search(
                query=expanded_query, top_k=search_pool_size
            )

        unique_results = self._deduplicate_results(raw_results)
        reranked_results = self._rerank_insurance_results(
            query, unique_results, max_results=top_k
        )

        return reranked_results

    # async def run_generation(
    #     self,
    #     query: str,
    #     search_results: List[Dict],
    #     is_personal_batch: bool = False,
    #     user_profile: Optional[Dict] = None,
    # ) -> str:
    #     """
    #     Async method to run generation pipeline.

    #     Args:
    #         query: Original user query
    #         search_results: Retrieved search results
    #         is_personal_batch: Whether this is a personal batch
    #         user_profile: Optional user profile

    #     Returns:
    #         Generated response string
    #     """
    #     if not search_results:
    #         return "I couldn't find any relevant information to answer your question."

    #     context_parts = []
    #     for i, result in enumerate(search_results, 1):
    #         content = result.get("content", "").strip()
    #         metadata = result.get("metadata", {})

    #         if content:
    #             filename = metadata.get("filename", "Unknown Document")
    #             page = metadata.get("page_number", "N/A")
    #             heading = metadata.get("page_heading", "General Information")

    #             source_ref = f"[Source {i}: {filename}, Page {page}]"
    #             context_parts.append(
    #                 f"{source_ref}\nPAGE HEADING: {heading}\n\n{content}"
    #             )

    #     if not context_parts:
    #         return "Error: Found documents but failed to extract content."

    #     context_from_docs = "\n\n---\n\n".join(context_parts)

    #     if is_personal_batch and user_profile:
    #         user_name = user_profile.get("name", "User")
    #         insurance_policies = user_profile.get("insurance_policies", {})

    #         profile_info = f"\n\n--- USER PROFILE (YOUR SOURCE OF TRUTH) ---\n"
    #         profile_info += f"- User Name: {user_name}\n"

    #         if insurance_policies:
    #             profile_info += f"- User's Policies:\n"
    #             for filename, policy_data in insurance_policies.items():
    #                 plan = policy_data.get("plan_name", "Unknown Plan")
    #                 tier = policy_data.get("tier", "N/A")
    #                 profile_info += f"  - Policy: {plan} (Tier: {tier})\n"
    #     else:
    #         user_name = "User"
    #         profile_info = "\n\n--- USER PROFILE (YOUR SOURCE OF TRUTH) ---\n- No user profile provided.\n"

    #     salutation = f"Hi {user_name.split()[0] if user_name != 'User' else 'Hi'},"

    #     prompt_instructions = config.INSURANCE_SYSTEM_PROMPT.format(
    #         profile_info=profile_info,
    #         context_from_docs=context_from_docs,
    #         salutation=salutation,
    #         original_query=query,
    #     )

    #     try:
    #         # Use the async OpenAI client for non-streaming generation inside
    #         # an async method (avoids blocking the event loop in evaluation).
    #         async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    #         response = await async_client.chat.completions.create(
    #             model=config.RESPONSE_MODEL,
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": "You are an expert financial advisor. Answer insurance questions using provided documents. If the user asks for anything outside insurance/policies/coverage/claims, respond that you can only answer insurance questions and stop. Be concise and accurate.",
    #                 },
    #                 {"role": "user", "content": prompt_instructions},
    #             ],
    #             max_tokens=config.RESPONSE_MAX_TOKENS,
    #             temperature=config.RESPONSE_TEMPERATURE,
    #         )

    #         # Workaround for object wrappers in openai client
    #         return response.choices[0].message.content

    #     except Exception as e:
    #         print(f"Error during generation: {e}")
    #         raise

    async def run_generation(
        self,
        query: str,
        search_results: List[Dict],
        is_personal_batch: bool = False,
        user_profile: Optional[Dict] = None,
    ) -> str:
        """
        Async method to run generation pipeline using Gemini.
        """
        if not search_results:
            return "I couldn't find any relevant information to answer your question."

        context_parts = []
        for i, result in enumerate(search_results, 1):
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            if content:
                filename = metadata.get("filename", "Unknown Document")
                page = metadata.get("page_number", "N/A")
                heading = metadata.get("page_heading", "General Information")
                source_ref = f"[Source {i}: {filename}, Page {page}]"
                context_parts.append(
                    f"{source_ref}\nPAGE HEADING: {heading}\n\n{content}"
                )

        if not context_parts:
            return "Error: Found documents but failed to extract content."

        context_from_docs = "\n\n---\n\n".join(context_parts)

        if is_personal_batch and user_profile:
            user_name = user_profile.get("name", "User")
            profile_info = f"\n\n--- USER PROFILE (YOUR SOURCE OF TRUTH) ---\n- User Name: {user_name}\n"
            # (Add more profile logic here if needed, matching your stream method)
        else:
            user_name = "User"
            profile_info = "\n\n--- USER PROFILE (YOUR SOURCE OF TRUTH) ---\n- No user profile provided.\n"

        salutation = f"Hi {user_name.split()[0] if user_name != 'User' else 'Hi'},"

        prompt_instructions = config.INSURANCE_SYSTEM_PROMPT.format(
            profile_info=profile_info,
            context_from_docs=context_from_docs,
            salutation=salutation,
            original_query=query,
        )

        try:
            # If we are forced to use OpenAI for generation, use AsyncOpenAI
            if (self.generation_provider or "").lower() == "openai":
                async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = await async_client.chat.completions.create(
                    model=config.RESPONSE_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert financial advisor. Answer insurance questions using provided documents. If the user asks for anything outside insurance/policies/coverage/claims, respond that you can only answer insurance questions and stop. Be concise and accurate.",
                        },
                        {"role": "user", "content": prompt_instructions},
                    ],
                    max_tokens=config.RESPONSE_MAX_TOKENS,
                    temperature=config.RESPONSE_TEMPERATURE,
                )

                return response.choices[0].message.content

            # Otherwise, prefer Gemini via LangChain wrapper where possible.
            # Prefer using the LangChain wrapper where possible (it avoids raw
            # genai dependency conflicts). If unavailable, fall back to the
            # raw google.generativeai library.
            system_instruction = "You are an expert financial advisor. Answer insurance questions using provided documents. Be concise and accurate."
            # Messages: system + user
            messages = [SystemMessage(content=system_instruction), HumanMessage(content=prompt_instructions)]

            if self.gemini_llm is not None:
                # ChatGoogleGenerativeAI.invoke is synchronous - run it in the default thread pool
                loop = asyncio.get_running_loop()
                invoker = partial(self.gemini_llm.invoke, messages)
                response = await loop.run_in_executor(None, invoker)
                # Response object from LangChain generator has 'content'
                return getattr(response, "content", str(response))
            else:
                # Fallback: raw google.generativeai usage
                model = genai.GenerativeModel(
                    model_name=self.model_name, system_instruction=system_instruction
                )

                response = await model.generate_content_async(
                    prompt_instructions,
                    generation_config=genai.types.GenerationConfig(
                        temperature=config.RESPONSE_TEMPERATURE,
                        max_output_tokens=config.RESPONSE_MAX_TOKENS,
                    ),
                )

                return response.text
        except Exception as e:
            print(f"Error during generation: {e}")
            raise

    def _ensure_batch_loaded(self, batch_id: str) -> bool:
        """Ensure the specified batch is loaded in the search engine."""
        if self.current_batch_id == batch_id and self.search_engine:
            return True

        paths = self.batch_manager.get_batch_paths(batch_id)
        if not paths:
            print(f"Error: Batch '{batch_id}' configuration not found in registry.")
            return False

        print(f"Loading indexes for batch '{batch_id}'...")
        try:
            # Initialize with weights from settings
            self.search_engine = HybridSearchEngine(
                faiss_weight=config.faiss_weight, bm25_weight=config.bm25_weight
            )
            success = self.search_engine.load_indexes(
                faiss_path=paths["faiss_index"], bm25_path=paths["bm25_index"]
            )

            if success:
                self.current_batch_id = batch_id
                print(f"Successfully loaded indexes for batch '{batch_id}'.")
                return True
            else:
                print(f"Error: Failed to load indexes for batch '{batch_id}'.")
                self.search_engine = None
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
            content = result.get("content", "")
            if content and content not in seen_content:
                unique_results.append(result)
                seen_content.add(content)
        return unique_results

    def _rerank_insurance_results(
        self, query: str, results: List[Dict], max_results: int
    ) -> List[Dict]:
        """
        Apply simple insurance-domain heuristics:
        - Boost chunks that match the policy/rider likely relevant to the query.
        - Force-include a high-signal CI chunk (sum insured) for cancer/CI intent.
        - Force-include a high-signal health chunk (deductible/co-insurance) for hospital/treatment intent.
        """
        if not results:
            return []

        query_lower = query.lower()
        wants_ci = any(
            k in query_lower
            for k in [
                "cancer",
                "ci",
                "critical illness",
                "major cancer",
                "bypass",
                "cabg",
                "coronary artery",
                "heart attack",
                "stroke",
                "angioplasty",
            ]
        )
        wants_health = any(
            k in query_lower
            for k in [
                "hospital",
                "treatment",
                "surgery",
                "warded",
                "deductible",
                "co-insurance",
            ]
        )
        wants_travel_incident = any(
            k in query_lower
            for k in [
                "travel",
                "trip",
                "overseas",
                "abroad",
                "rental car",
                "car rental",
                "road accident",
                "accident",
                "baggage",
                "luggage",
                "flight delay",
                "trip cancellation",
                "trip interruption",
                "evacuation",
            ]
        )

        boosted = []
        currency_re = re.compile(r"(\$\s?\d|sgd|sum insured|payout|benefit)", re.I)
        health_re = re.compile(r"(deductible|co-?insurance|out[- ]of[- ]pocket)", re.I)
        travel_med_re = re.compile(
            r"(medical|medical expense|hospital|clinic|treatment|emergency evacuation|repatriation|overseas medical)",
            re.I,
        )
        travel_pa_re = re.compile(
            r"(personal accident|accidental death|permanent disablement|injury)",
            re.I,
        )

        forced_ci_candidate = None
        forced_ci_score = float("-inf")
        forced_health_candidate = None
        forced_health_score = float("-inf")
        forced_travel_med_candidate = None
        forced_travel_med_score = float("-inf")
        forced_travel_pa_candidate = None
        forced_travel_pa_score = float("-inf")

        for idx, res in enumerate(results):
            meta = res.get("metadata", {}) or {}
            content = (res.get("content") or "").lower()
            plan_context = meta.get("plan_context") or []
            filename = (meta.get("filename") or "").lower()
            travel_match = any(
                "travel" in pc.lower() or "travelcare" in pc.lower()
                for pc in plan_context
            ) or ("travelcare" in filename)

            # Use combined_score from hybrid search as the base, falling back to score
            score = res.get("combined_score") or res.get("score", 0) or 0
            # Preserve some original ordering
            score -= idx * 0.001

            if wants_ci and any(
                "critical care enhancer" in pc.lower() for pc in plan_context
            ):
                score += 0.5
            if wants_health and any(
                "supremehealth" in pc.lower() for pc in plan_context
            ):
                score += 0.3
            if wants_travel_incident and travel_match:
                score += 0.4

            if "sum insured" in content or "$" in content or "benefit" in content:
                score += 0.1  # slight boost for amount-bearing chunks
            if "deductible" in content or "co-insurance" in content:
                score += 0.1
            if "major cancer" in content or "critical illness" in content:
                score += 0.1
            if wants_travel_incident and (
                "evacuation" in content
                or "repatriation" in content
                or "accident" in content
                or "medical expense" in content
            ):
                score += 0.2

            # Track best CI chunk with explicit amount words
            if wants_ci and (
                any("critical care enhancer" in pc.lower() for pc in plan_context)
                or "critical care enhancer" in content
            ):
                has_amount_signal = bool(currency_re.search(content))
                bonus = 20
                if has_amount_signal:
                    bonus = 50  # force top placement when amount present
                ci_score = score + bonus
                if ci_score > forced_ci_score:
                    forced_ci_candidate = res
                    forced_ci_score = ci_score

            # Track best health chunk with deductible/co-insurance signals
            if wants_health and (
                any("supremehealth" in pc.lower() for pc in plan_context)
                or "supremehealth" in content
            ):
                has_health_signal = bool(health_re.search(content))
                bonus = 10
                if has_health_signal:
                    bonus = (
                        30  # force top placement when deductible/co-insurance present
                    )
                health_score = score + bonus
                if health_score > forced_health_score:
                    forced_health_candidate = res
                    forced_health_score = health_score

            # Track travel medical/evacuation chunks
            if wants_travel_incident and travel_match:
                if travel_med_re.search(content):
                    bonus = 25
                    travel_score = score + bonus
                    if travel_score > forced_travel_med_score:
                        forced_travel_med_candidate = res
                        forced_travel_med_score = travel_score
                if travel_pa_re.search(content):
                    bonus = 20
                    travel_pa_score = score + bonus
                    if travel_pa_score > forced_travel_pa_score:
                        forced_travel_pa_candidate = res
                        forced_travel_pa_score = travel_pa_score

            boosted.append((score, res))

        # Sort by boosted score descending
        boosted.sort(key=lambda x: x[0], reverse=True)
        reranked = [r for _, r in boosted]

        # Force-include the best CI chunk if relevant and not already in top slots
        if wants_ci and forced_ci_candidate:
            if forced_ci_candidate not in reranked[:max_results]:
                reranked = [forced_ci_candidate] + reranked

        # Force-include the best health chunk if relevant and not already in top slots
        if wants_health and forced_health_candidate:
            if forced_health_candidate not in reranked[:max_results]:
                reranked = [forced_health_candidate] + reranked

        # Force-include travel medical/evacuation and personal accident chunks
        if wants_travel_incident and forced_travel_med_candidate:
            if forced_travel_med_candidate not in reranked[:max_results]:
                reranked = [forced_travel_med_candidate] + reranked
        if wants_travel_incident and forced_travel_pa_candidate:
            if forced_travel_pa_candidate not in reranked[:max_results]:
                reranked = [forced_travel_pa_candidate] + reranked

        return reranked[:max_results]

    def _inject_top_plan_chunks(
        self, plan_substr: str, signals: List[str], max_hits: int = 1
    ) -> List[Dict]:
        """
        Deterministically pull top chunks from loaded indexes for a given plan substring,
        prioritizing amount-bearing signals. Use this when critical plan facts are needed
        even if search/rerank missed them.
        """
        injected: List[Dict] = []
        try:
            chunks = getattr(self.search_engine, "bm25_chunks", []) or []
            metadata = getattr(self.search_engine, "bm25_metadata", []) or []
        except Exception:
            return injected

        scored = []
        for content, meta in zip(chunks, metadata):
            text = content or ""
            text_lower = text.lower()
            plan_ctx = meta.get("plan_context") or []
            if not any(plan_substr in pc.lower() for pc in plan_ctx):
                continue

            score = 0
            for s in signals:
                if s in text_lower:
                    score += 3
            if re.search(r"\$\s?\d", text):
                score += 6
            if "sum insured" in text_lower:
                score += 6
            if "deductible" in text_lower or "co-insurance" in text_lower:
                score += 4

            if score > 0:
                scored.append((score, content, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, content, meta in scored[:max_hits]:
            injected.append(
                {
                    "content": content,
                    "metadata": meta,
                    "score": score + 1000,  # large to keep near top pre-rerank
                    "source": "forced_plan_chunk",
                }
            )

        return injected

    # def _multi_policy_search(
    #     self,
    #     query: str,
    #     expanded_query: str,
    #     user_profile: Dict,
    #     chunks_per_policy: int = 10,
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Perform separate searches for each policy the user owns, then combine results.
    #     This ensures balanced representation across all policies.

    #     Args:
    #         query: Original user query
    #         expanded_query: Expanded query with additional keywords
    #         user_profile: User profile containing insurance_policies
    #         chunks_per_policy: Number of chunks to retrieve per policy (default: 10)

    #     Returns:
    #         Combined list of search results from all policies
    #     """
    #     insurance_policies = user_profile.get("insurance_policies", {})
    #     if not insurance_policies:
    #         print("No policies found in user profile, falling back to standard search")
    #         return self.search_engine.hybrid_search(
    #             query=expanded_query, top_k=config.SEARCH_TOP_K
    #         )

    #     all_results = []
    #     policy_filenames = list(insurance_policies.keys())

    #     # Detect if query is asking about amounts/coverage
    #     query_lower = query.lower()
    #     wants_amount = any(
    #         kw in query_lower
    #         for kw in [
    #             "how much",
    #             "amount",
    #             "sum insured",
    #             "coverage",
    #             "covered for",
    #             "$",
    #             "money",
    #             "payout",
    #         ]
    #     )

    #     print(f"\n=== Multi-Policy Search ===")
    #     print(f"Searching across {len(policy_filenames)} policies: {policy_filenames}")

    #     for filename in policy_filenames:
    #         print(f"\nSearching policy: {filename}")
    #         policy_results = self.search_engine.hybrid_search(
    #             query=expanded_query, top_k=chunks_per_policy, filename_filter=filename
    #         )
    #         print(f"  → Retrieved {len(policy_results)} chunks from {filename}")

    #         # For amount queries on Life/CI policies, do an additional targeted search
    #         if wants_amount and "Manulife" in filename:
    #             print(
    #                 f"  → Amount query detected - performing additional targeted search"
    #             )
    #             amount_query = "sum insured benefit amount payout table premium"
    #             extra_results = self.search_engine.hybrid_search(
    #                 query=amount_query, top_k=5, filename_filter=filename
    #             )
    #             print(
    #                 f"  → Retrieved {len(extra_results)} additional chunks with amount focus"
    #             )

    #             # Add extra results, avoiding duplicates
    #             existing_content = {r.get("content", "") for r in policy_results}
    #             for extra in extra_results:
    #                 if extra.get("content", "") not in existing_content:
    #                     policy_results.append(extra)
    #                     print(
    #                         f"     + Added extra chunk from page {extra.get('metadata', {}).get('page_number', '?')}"
    #                     )

    #         # Debug: Show page numbers and scores for Manulife
    #         if "Manulife" in filename:
    #             print(f"  → Manulife chunk details:")
    #             for i, result in enumerate(policy_results[:15], 1):
    #                 metadata = result.get("metadata", {})
    #                 page = metadata.get("page_number", "N/A")
    #                 score = result.get("combined_score", 0)
    #                 content_preview = result.get("content", "")[:60].replace("\n", " ")
    #                 has_dollar = "$" in result.get("content", "")
    #                 dollar_marker = " [$]" if has_dollar else ""
    #                 print(
    #                     f"     #{i}: Page {page} (score: {score:.3f}){dollar_marker} - {content_preview}..."
    #                 )

    #         # Tag each result with its policy for tracking
    #         for result in policy_results:
    #             result["source_policy"] = filename
    #             all_results.append(result)

    #     print(
    #         f"\nTotal chunks retrieved: {len(all_results)} ({chunks_per_policy} per policy)"
    #     )
    #     print(f"=== End Multi-Policy Search ===\n")

    #     return all_results

    def _multi_policy_search(
        self,
        query: str,
        expanded_query: str,
        user_profile: Dict,
        chunks_per_policy: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform separate searches for each policy, PLUS force-include the first few pages
        and dollar-heavy tables to ensure the "Schedule/Benefit Table" is always present.
        """
        insurance_policies = user_profile.get("insurance_policies", {})
        if not insurance_policies:
            print("No policies found in user profile, falling back to standard search")
            return self.search_engine.hybrid_search(
                query=expanded_query, top_k=config.SEARCH_TOP_K
            )

        all_results = []
        policy_filenames = list(insurance_policies.keys())

        print(f"\n=== Multi-Policy Search (Enhanced) ===")
        print(f"Searching across {len(policy_filenames)} policies: {policy_filenames}")

        for filename in policy_filenames:
            print(f"\nSearching policy: {filename}")

            # 1. Standard Semantic/Keyword Search (Finds definitions/medical terms)
            policy_results = self.search_engine.hybrid_search(
                query=expanded_query, top_k=chunks_per_policy, filename_filter=filename
            )

            # 2. Force-Include Schedule/Benefit Tables (Finds dollar amounts)
            # We run a targeted search for table-like keywords
            print(f"  → Injecting Schedule/Benefit pages for {filename}...")

            schedule_results = self.search_engine.hybrid_search(
                query="Policy Schedule Sum Insured Benefit Table Deductible Limit Premium",
                top_k=10,  # Look at top 10 candidates for tables
                filename_filter=filename,
            )

            # Filter to keep only the most high-value "Money Pages"
            injected_count = 0
            for res in schedule_results:
                page = res.get("metadata", {}).get("page_number", 999)
                content = res.get("content", "")

                # CONDITIONAL: Keep chunk IF:
                # A. It is within the first 7 pages (Covers Manulife Rider Summary on Pg 6)
                # B. OR it contains a dollar sign "$" (Covers SupremeHealth tables on later pages)
                if page <= 7 or "$" in content:

                    # Dedup: Don't add if we already found it in the semantic search
                    existing_contents = [r.get("content") for r in policy_results]
                    if content not in existing_contents:
                        policy_results.append(res)
                        injected_count += 1
                        print(f"     + Injected Schedule Chunk (Page {page})")

            print(f"  → Total chunks for this policy: {len(policy_results)}")

            # Tag and collect
            for result in policy_results:
                result["source_policy"] = filename
                all_results.append(result)

        print(f"\nTotal chunks retrieved: {len(all_results)}")
        print(f"=== End Multi-Policy Search ===\n")

        return all_results

    # def _expand_query(self, query: str) -> str:
    #     """
    #     Expands the user query using intelligent LLM rewriting and a hardcoded
    #     critical term map for the insurance domain.
    #     """
    #     added_keywords = set()
    #     query_lower = query.lower()

    #     # Add keywords from the critical term map (from config)
    #     for term, expansion in config.CRITICAL_TERM_MAP.items():
    #         if term in query_lower:
    #             added_keywords.add(expansion)

    #     # Add keywords from the policy type map (from config)
    #     for term, expansion in config.POLICY_TYPE_KEYWORDS.items():
    #         if term in query_lower:
    #             added_keywords.update(expansion.split())

    #     # Add keywords for specific scenarios
    #     if (
    #         "warded" in query_lower
    #         or "surgery" in query_lower
    #         or "hospital" in query_lower
    #     ):
    #         added_keywords.add("deductible")
    #         added_keywords.add("co-insurance")

    #     manual_expansion = " ".join(added_keywords)

    #     try:
    #         expansion_prompt = config.QUERY_EXPANSION_PROMPT.format(query=query)

    #         response = self.client.chat.completions.create(
    #             model=config.EXPANSION_MODEL,
    #             messages=[{"role": "user", "content": expansion_prompt}],
    #             max_tokens=config.EXPANSION_MAX_TOKENS,
    #             temperature=config.EXPANSION_TEMPERATURE,
    #         )

    #         llm_keywords = response.choices[0].message.content.strip()

    #         # Combine all three: Original Query + Manual Keywords + LLM Keywords
    #         expanded_query = f"{query} {manual_expansion} {llm_keywords}"

    #         print(f"Query intelligently expanded to: {expanded_query}")
    #         return expanded_query

    #     except Exception as e:
    #         print(f"Error during query expansion: {e}")
    #         # Fallback to original query + manual expansion
    #         return f"{query} {manual_expansion}"

    def _expand_query(self, query: str) -> str:
        """
        Expands the user query using OpenAI (reverted from Gemini) and hardcoded maps.
        """
        added_keywords = set()
        query_lower = query.lower()

        # Add keywords from the critical term map (from config)
        for term, expansion in config.CRITICAL_TERM_MAP.items():
            if term in query_lower:
                added_keywords.add(expansion)

        # Add keywords from the policy type map (from config)
        for term, expansion in config.POLICY_TYPE_KEYWORDS.items():
            if term in query_lower:
                added_keywords.update(expansion.split())

        # Add keywords for specific scenarios
        if (
            "warded" in query_lower
            or "surgery" in query_lower
            or "hospital" in query_lower
        ):
            added_keywords.add("deductible")
            added_keywords.add("co-insurance")

        manual_expansion = " ".join(added_keywords)

        try:
            expansion_prompt = config.QUERY_EXPANSION_PROMPT.format(query=query)

            # --- REVERTED: Use OpenAI for expansion ---
            # We need a synchronous OpenAI client here since this runs in a thread executor
            # and we don't want to mess with async loops inside it if possible,
            # or we can just use the standard OpenAI client.
            # The original code used self.client which was OpenAI(api_key=...)
            # But in __init__ we removed self.client. Let's re-instantiate it locally or in __init__.
            # For safety, let's instantiate locally.
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model=config.EXPANSION_MODEL,
                messages=[{"role": "user", "content": expansion_prompt}],
                max_tokens=config.EXPANSION_MAX_TOKENS,
                temperature=config.EXPANSION_TEMPERATURE,
            )

            llm_keywords = response.choices[0].message.content.strip()

            # Combine all three: Original Query + Manual Keywords + LLM Keywords
            expanded_query = f"{query} {manual_expansion} {llm_keywords}"

            print(f"Query intelligently expanded to: {expanded_query}")
            return expanded_query

        except Exception as e:
            print(f"Error during query expansion: {e}")
            # Fallback to original query + manual expansion
            return f"{query} {manual_expansion}"

    def _prepare_context(
        self,
        search_results: List[Dict],
        is_personal_batch: bool,
        user_profile: Optional[Dict],
        max_chunks: int = 30,
        max_context_length: int = 100000,  # Token limit
    ) -> tuple[str, str]:
        """
        Prepares the context for AI with token management.
        Returns: (profile_info, context_from_docs)
        """

        # ===== PART 1: Build Profile Info =====
        if is_personal_batch and user_profile:
            user_name = user_profile.get("name", "User")
            user_dob = user_profile.get("date_of_birth", "N/A")
            insurance_policies = user_profile.get("insurance_policies", {})

            profile_info = f"\n\n--- USER PROFILE (YOUR SOURCE OF TRUTH) ---\n"
            profile_info += f"- User Name: {user_name}\n"
            profile_info += f"- User DOB: {user_dob}\n"

            if insurance_policies:
                profile_info += f"- User's Policies:\n"
                for filename, policy_data in insurance_policies.items():
                    plan = policy_data.get("plan_name", "Unknown Plan")
                    tier = policy_data.get("tier", "N/A")
                    riders = policy_data.get("riders", [])
                    underwriting = policy_data.get("underwriting", {})
                    exclusions = underwriting.get("exclusions")

                    profile_info += f"  - Policy: {plan} (Tier: {tier})\n"

                    if riders:
                        rider_names = [
                            r.get("plan_name", r) if isinstance(r, dict) else r
                            for r in riders
                        ]
                        profile_info += f"    - Riders: {', '.join(rider_names)}\n"
                    else:
                        profile_info += f"    - Riders: None listed\n"

                    if exclusions:
                        profile_info += (
                            f"    - !! IMPORTANT EXCLUSION: {exclusions} !!\n"
                        )
        else:
            profile_info = "\n\n--- USER PROFILE (YOUR SOURCE OF TRUTH) ---\n- No user profile provided.\n"

        # ===== PART 2: Build Document Context (enhanced with token management) =====

        # Priority keywords
        priority_keywords = [
            "coverage",
            "benefit",
            "limit",
            "sum assured",
            "sum insured",
            "premium",
            "claim",
            "deductible",
            "exclusion",
        ]

        def is_priority_chunk(content: str) -> bool:
            content_lower = content.lower()
            return any(keyword in content_lower for keyword in priority_keywords)

        # Sort results: priority chunks first
        priority_results = [
            r
            for r in search_results[:max_chunks]
            if is_priority_chunk(r.get("content", ""))
        ]
        other_results = [
            r
            for r in search_results[:max_chunks]
            if not is_priority_chunk(r.get("content", ""))
        ]
        sorted_results = priority_results + other_results

        context_parts = []
        total_length = 0

        print(
            f"Building context from top {len(sorted_results)} chunks (priority-sorted)..."
        )

        for i, result in enumerate(sorted_results, 1):
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})

            if not content:
                continue

            filename = metadata.get("filename", "Unknown Document")
            page = metadata.get("page_number", "N/A")
            heading = metadata.get("page_heading", "General Information")

            # Format with citation
            source_ref = f"[Source {i}: {filename}, Page {page}]"
            chunk_text = f"{source_ref}\nPAGE HEADING: {heading}\n\n{content}"

            # Token management
            if total_length + len(chunk_text) > max_context_length:
                print(
                    f"⚠️ Reached token limit ({max_context_length}). Stopping at {i} chunks."
                )

                # Try to fit a trimmed version if it's important
                if is_priority_chunk(content):
                    remaining_space = max_context_length - total_length
                    if remaining_space > 500:  # Only trim if we have reasonable space
                        trimmed_content = content[: remaining_space - 200] + "..."
                        chunk_text = f"{source_ref}\nPAGE HEADING: {heading}\n\n{trimmed_content}"
                        context_parts.append(chunk_text)
                        print(f"  ✂️ Trimmed priority chunk from {filename}")
                break

            context_parts.append(chunk_text)
            total_length += len(chunk_text)

            # Log priority chunks
            if is_priority_chunk(content):
                print(f"  ⭐ Priority chunk: {filename} Page {page}")

        if not context_parts:
            context_from_docs = "No relevant document chunks found."
        else:
            context_from_docs = "\n\n---\n\n".join(context_parts)
            print(
                f"✅ Built context: {len(context_parts)} chunks, ~{total_length} chars"
            )

        return profile_info, context_from_docs

    # def process_query_stream(
    #     self, query: str, batch_id: str = None, user_profile: Optional[Dict] = None
    # ):
    #     """Process a query and yield response chunks for streaming."""
    #     try:
    #         # === Guard 1: Out-of-scope filter before any retrieval/LLM work ===
    #         if self._is_out_of_scope(query):
    #             refusal = "I can only help with insurance and policy questions using the provided documents."
    #             yield "data: " + json.dumps({"content": refusal}) + "\n\n"
    #             yield "data: " + json.dumps({"done": True}) + "\n\n"
    #             return

    #         # Determine the target batch
    #         target_batch = batch_id or self.batch_manager.get_default_batch()
    #         if not target_batch:
    #             yield "data: " + json.dumps(
    #                 {"error": "No batch specified and no default batch set."}
    #             ) + "\n\n"
    #             return

    #         # Ensure the correct batch's indexes are loaded
    #         if not self._ensure_batch_loaded(target_batch):
    #             yield "data: " + json.dumps(
    #                 {"error": f"Failed to load or switch to batch '{target_batch}'."}
    #             ) + "\n\n"
    #             return

    #         start_time = time.time()
    #         print(f"\nProcessing query for batch: {target_batch}")
    #         print(f"Query: {query}")

    #         # Analyze intent
    #         intent_analyzer = IntentAnalyzer()
    #         intent = intent_analyzer.analyze(query)
    #         print(f"Intent analysis: {intent}")

    #         # Use the new adapter-style retrieval method (async) from a
    #         # synchronous context so the streaming UI remains unchanged.
    #         is_personal_batch = target_batch.startswith("user_")

    #         loop = asyncio.new_event_loop()
    #         asyncio.set_event_loop(loop)
    #         try:
    #             raw_search_results = loop.run_until_complete(
    #                 self.run_retrieval(
    #                     query=query,
    #                     batch_id=target_batch,
    #                     user_profile=user_profile,
    #                     top_k=config.SEARCH_TOP_K,
    #                 )
    #             )
    #         finally:
    #             loop.close()

    #         print(
    #             f"Retrieved {len(raw_search_results)} raw results from hybrid search."
    #         )

    #         # Quick intent flags for targeted boosts
    #         query_lower = query.lower()
    #         wants_ci = any(
    #             k in query_lower
    #             for k in [
    #                 "cancer",
    #                 "ci",
    #                 "critical illness",
    #                 "major cancer",
    #                 "bypass",
    #                 "cabg",
    #                 "coronary artery",
    #                 "heart attack",
    #                 "stroke",
    #                 "angioplasty",
    #             ]
    #         )
    #         wants_health = any(
    #             k in query_lower
    #             for k in [
    #                 "hospital",
    #                 "treatment",
    #                 "surgery",
    #                 "warded",
    #                 "deductible",
    #                 "co-insurance",
    #             ]
    #         )

    #         # Targeted second-pass search to force in key policy chunks if missing
    #         def _has_plan(results, name_substr: str) -> bool:
    #             for r in results:
    #                 meta = r.get("metadata") or {}
    #                 plan_context = meta.get("plan_context") or []
    #                 content = (r.get("content") or "").lower()
    #                 if any(name_substr in pc.lower() for pc in plan_context):
    #                     return True
    #                 if name_substr in content:
    #                     return True
    #             return False

    #         if wants_ci and not _has_plan(raw_search_results, "critical care enhancer"):
    #             extra_ci = self.search_engine.hybrid_search(
    #                 query="Critical Care Enhancer Rider sum insured payout benefit CCE",
    #                 top_k=5,
    #             )
    #             raw_search_results.extend(extra_ci)
    #             # Heuristic forced inclusion from index if amounts still missing
    #             forced_cce = self._inject_top_plan_chunks(
    #                 plan_substr="critical care enhancer",
    #                 signals=[
    #                     "sum insured",
    #                     "$",
    #                     "benefit",
    #                     "payout",
    #                     "500,000",
    #                     "1,000,000",
    #                 ],
    #                 max_hits=2,
    #             )
    #             raw_search_results.extend(forced_cce)

    #         if wants_health and not _has_plan(raw_search_results, "supremehealth"):
    #             extra_health = self.search_engine.hybrid_search(
    #                 query="GREAT SupremeHealth P PLUS deductible co-insurance ward A",
    #                 top_k=5,
    #             )
    #             raw_search_results.extend(extra_health)
    #             forced_supreme = self._inject_top_plan_chunks(
    #                 plan_substr="supremehealth",
    #                 signals=[
    #                     "deductible",
    #                     "co-insurance",
    #                     "$3,500",
    #                     "ward",
    #                     "co payment",
    #                     "co-payment",
    #                 ],
    #                 max_hits=2,
    #             )
    #             raw_search_results.extend(forced_supreme)

    #         unique_results = self._deduplicate_results(raw_search_results)
    #         # Domain-aware rerank/force-includes to get higher-signal chunks
    #         unique_results = self._rerank_insurance_results(
    #             query, unique_results, max_results=config.MAX_CONTEXT_CHUNKS + 3
    #         )

    #         # === Guard 2: Retrieval sufficiency gate ===
    #         if not unique_results:
    #             error_msg = "I could not find relevant insurance information in your documents for this question."
    #             yield "data: " + json.dumps(
    #                 {"content": error_msg, "done": True}
    #             ) + "\n\n"
    #             return

    #         # Debug: write reranked chunks to a log file for inspection
    #         try:
    #             from pathlib import Path

    #             debug_dir = Path("logs")
    #             debug_dir.mkdir(exist_ok=True)
    #             log_path = debug_dir / "reranked_chunks.txt"
    #             with open(log_path, "w", encoding="utf-8") as log_file:
    #                 log_file.write(f"Query: {query}\n")
    #                 log_file.write(
    #                     f"Total reranked: {len(unique_results)} (showing top {min(50, len(unique_results))})\n\n"
    #                 )
    #                 for i, r in enumerate(
    #                     unique_results[: min(50, len(unique_results))], 1
    #                 ):
    #                     meta = r.get("metadata", {}) or {}
    #                     filename = meta.get("filename", "Unknown")
    #                     page = meta.get("page_number", "N/A")
    #                     plan_ctx = ", ".join(meta.get("plan_context") or [])
    #                     snippet = (r.get("content") or "").replace("\n", " ")[:400]
    #                     log_file.write(
    #                         f"#{i} {filename} p{page} | plans: {plan_ctx}\n{snippet}\n\n"
    #                     )
    #         except Exception as debug_err:
    #             print(f"[DEBUG] Failed to write reranked chunk log: {debug_err}")

    #         # Prepare context early for potential deep research use
    #         profile_info, context_from_docs = self._prepare_context(
    #             unique_results, is_personal_batch, user_profile
    #         )

    #         # Check if no results found
    #         if not unique_results:
    #             if is_personal_batch:
    #                 error_msg = f"I couldn't find relevant information in your uploaded documents for the question: '{query}'."
    #             else:
    #                 error_msg = f"No relevant information found in the documents of batch '{target_batch}' for the question: '{query}'."
    #             yield "data: " + json.dumps(
    #                 {"content": error_msg, "done": True}
    #             ) + "\n\n"
    #             return

    #         # ===== NEW: ANALYZE → RETRY LOGIC =====
    #         # Step 1: Try normal RAG first (stream AND collect, but don't show to user yet)
    #         print("🔄 Attempting normal RAG response...")
    #         normal_response_chunks = []
    #         stored_rag_chunks = []

    #         for chunk in self._generate_response_stream(
    #             query, unique_results, is_personal_batch, user_profile
    #         ):
    #             stored_rag_chunks.append(chunk)  # Store for later
    #             # Collect for analysis
    #             try:
    #                 chunk_data = json.loads(chunk.replace("data: ", "").strip())
    #                 if "content" in chunk_data:
    #                     normal_response_chunks.append(chunk_data["content"])
    #             except:
    #                 pass  # Ignore parsing errors for "done" chunks

    #         # Step 2: Analyze the collected response quality
    #         full_response = "".join(normal_response_chunks)
    #         response_analyzer = ResponseAnalyzer()
    #         is_satisfactory = response_analyzer._analyze_response_quality(
    #             query, full_response, user_profile
    #         )

    #         # Step 3: If bad → trigger deep research
    #         print(
    #             f"🔍 DEBUG: is_satisfactory={is_satisfactory}, deep_research_enabled={self.deep_research_enabled}"
    #         )
    #         if not is_satisfactory and self.deep_research_enabled:
    #             print("⚠️ Normal RAG response insufficient. Triggering deep research...")
    #             # Notify user
    #             yield "data: " + json.dumps(
    #                 {
    #                     "status": "enhancing_response",
    #                     "message": "Let me search for more comprehensive information...",
    #                 }
    #             ) + "\n\n"

    #             # Run deep research
    #             from src.run_ui import run_ui

    #             loop = asyncio.new_event_loop()
    #             asyncio.set_event_loop(loop)

    #             try:
    #                 async_gen = run_ui(
    #                     query=query,
    #                     intent=intent,
    #                     profile_info=profile_info,
    #                     context_from_docs=context_from_docs,
    #                 ).__aiter__()

    #                 while True:
    #                     try:
    #                         chunk = loop.run_until_complete(async_gen.__anext__())
    #                         yield "data: " + json.dumps(chunk) + "\n\n"
    #                     except StopAsyncIteration:
    #                         break
    #             finally:
    #                 loop.close()

    #             print("✅ Deep research completed.")
    #         else:
    #             print("✅ Normal RAG response satisfactory. Streaming to user...")
    #             # Yield ALL stored RAG chunks
    #             for chunk in stored_rag_chunks:
    #                 yield chunk

    #         # Signal completion
    #         yield "data: " + json.dumps({"done": True}) + "\n\n"

    #         processing_time = time.time() - start_time
    #         print(f"Total processing time: {processing_time:.2f}s")

    #     except Exception as e:
    #         import traceback

    #         print(f"An unexpected error occurred in process_query_stream: {e}")
    #         traceback.print_exc()
    #         yield "data: " + json.dumps(
    #             {"error": f"An error occurred while processing your query: {e}"}
    #         ) + "\n\n"

    def process_query_stream(
        self, query: str, batch_id: str = None, user_profile: Optional[Dict] = None
    ):
        """
        Two-Phase Streaming Flow:
        1. Immediate RAG Response (Fast)
        2. Post-Response Analysis -> Optional Deep Research (Slow)
        """
        try:
            # === Guard: Out-of-scope ===
            if self._is_out_of_scope(query):
                yield "data: " + json.dumps(
                    {"content": "I can only help with insurance questions."}
                ) + "\n\n"
                yield "data: " + json.dumps({"done": True}) + "\n\n"
                return

            # Setup Batch
            target_batch = batch_id or self.batch_manager.get_default_batch()
            if not target_batch or not self._ensure_batch_loaded(target_batch):
                yield "data: " + json.dumps({"error": "Batch loading failed."}) + "\n\n"
                return

            print(f"\nProcessing query: {query}")

            # 1. Intent Analysis
            intent_analyzer = IntentAnalyzer()
            intent = intent_analyzer.analyze(query)

            # 2. Retrieval (Async)
            is_personal_batch = target_batch.startswith("user_")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                raw_search_results = loop.run_until_complete(
                    self.run_retrieval(
                        query, target_batch, user_profile, config.SEARCH_TOP_K
                    )
                )
            finally:
                loop.close()

            # 3. Deduplicate/Rerank
            unique_results = self._deduplicate_results(raw_search_results)
            unique_results = self._rerank_insurance_results(
                query, unique_results, max_results=config.MAX_CONTEXT_CHUNKS + 3
            )

            # Buffer to capture the first response for analysis
            rag_response_buffer = ""

            # =====================================================
            # PHASE 1: STANDARD RAG (Immediate Answer)
            # =====================================================
            if not unique_results:
                # Handle no docs case
                msg = "I couldn't find any relevant documents regarding your query."
                yield "data: " + json.dumps({"content": msg}) + "\n\n"
                rag_response_buffer = msg
            else:
                # Stream the RAG response chunks
                for chunk in self._generate_response_stream(
                    query,
                    unique_results,
                    is_personal_batch,
                    user_profile,
                    include_final_done=False,
                ):
                    # Send chunk to UI
                    yield chunk

                    # Accumulate for analysis
                    try:
                        data = json.loads(chunk.replace("data: ", "").strip())
                        if data.get("done"):
                            # We control the final done message outside this loop.
                            continue
                        if "content" in data:
                            rag_response_buffer += data["content"]
                    except:
                        pass

            # =====================================================
            # PHASE 2: ANALYZE & TRANSITION
            # =====================================================

            # Note: We have NOT sent {"done": True} yet. The stream is still open.

            should_deep_research = False

            if self.deep_research_enabled:
                response_analyzer = ResponseAnalyzer()

                # Analyze the buffer we just collected
                is_satisfactory = response_analyzer._analyze_response_quality(
                    query, rag_response_buffer, user_profile
                )

                if not is_satisfactory:
                    should_deep_research = True

            # =====================================================
            # PHASE 3: DEEP RESEARCH (If needed)
            # =====================================================

            if should_deep_research:
                print("⚠️ Initial response insufficient. Triggering Deep Research...")

                # --- THE VISUAL SEPARATOR ---
                # This creates a visual break in the UI to look like a "new section"
                # You can also use **Bold Text** to announce the status change.
                transition_msg = (
                    "\n\n"
                    "---\n"
                    "**🕵️ Deep Research Triggered**\n"
                    "I am now performing a deeper analysis of your documents and external sources. Please wait...\n\n"
                )

                yield "data: " + json.dumps({"content": transition_msg}) + "\n\n"

                # Prepare fresh context
                profile_info, context_from_docs = self._prepare_context(
                    unique_results, is_personal_batch, user_profile
                )

                # Run Deep Research Stream
                from src.run_ui import run_ui

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    async_gen = run_ui(
                        query=query,
                        intent=intent,
                        profile_info=profile_info,
                        context_from_docs=context_from_docs,
                    ).__aiter__()

                    while True:
                        try:
                            chunk = loop.run_until_complete(async_gen.__anext__())
                            yield "data: " + json.dumps(chunk) + "\n\n"
                        except StopAsyncIteration:
                            break
                finally:
                    loop.close()

            # =====================================================
            # DONE
            # =====================================================
            yield "data: " + json.dumps({"done": True}) + "\n\n"

        except Exception as e:
            import traceback

            traceback.print_exc()
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"

    # def _generate_response_stream(
    #     self,
    #     original_query: str,
    #     search_results: List[Dict],
    #     is_personal_batch: bool,
    #     user_profile: Optional[Dict],
    # ):
    #     """Generate streaming response using retrieved chunks and potentially user profile."""
    #     if not search_results:
    #         yield "data: " + json.dumps(
    #             {
    #                 "content": "I couldn't find any relevant information in the documents to answer your question.",
    #                 "done": True,
    #             }
    #         ) + "\n\n"
    #         return

    #     context_parts = []
    #     max_chunks_for_context = config.MAX_CONTEXT_CHUNKS
    #     cited_filenames = set()

    #     print(
    #         f"Building context from top {min(len(search_results), max_chunks_for_context)} chunks..."
    #     )
    #     for i, result in enumerate(search_results[:max_chunks_for_context], 1):
    #         content = result.get("content", "").strip()
    #         metadata = result.get("metadata", {})
    #         if content:
    #             filename = metadata.get("filename", "Unknown Document")
    #             page = metadata.get("page_number", "N/A")

    #             heading = metadata.get("page_heading", "General Information")
    #             source_ref = f"[Source {i}: {filename}, Page {page}]"
    #             context_parts.append(
    #                 f"{source_ref}\nPAGE HEADING: {heading}\n\n{content}"
    #             )

    #             cited_filenames.add(filename)

    #     if not context_parts:
    #         yield "data: " + json.dumps(
    #             {
    #                 "content": "Error: Found relevant documents but failed to extract content for context.",
    #                 "done": True,
    #             }
    #         ) + "\n\n"
    #         return

    #     context_from_docs = "\n\n---\n\n".join(context_parts)

    #     if is_personal_batch and user_profile:
    #         user_name = user_profile.get("name", "User")
    #         user_dob = user_profile.get("date_of_birth", "N/A")
    #         insurance_policies = user_profile.get("insurance_policies", {})

    #         profile_info = f"\n\n--- USER PROFILE (YOUR SOURCE OF TRUTH) ---\n"
    #         profile_info += f"- User Name: {user_name}\n"
    #         profile_info += f"- User DOB: {user_dob}\n"

    #         if insurance_policies:
    #             profile_info += f"- User's Policies:\n"
    #             for filename, policy_data in insurance_policies.items():
    #                 plan = policy_data.get("plan_name", "Unknown Plan")
    #                 tier = policy_data.get("tier", "N/A")
    #                 riders = policy_data.get("riders", [])
    #                 underwriting = policy_data.get("underwriting", {})
    #                 exclusions = underwriting.get("exclusions")

    #                 # Add policy and tier info
    #                 profile_info += f"  - Policy: {plan} (Tier: {tier})\n"

    #                 # Add rider info
    #                 if riders:
    #                     # Handle list of strings or list of dicts
    #                     rider_names = [
    #                         r.get("plan_name", r) if isinstance(r, dict) else r
    #                         for r in riders
    #                     ]
    #                     profile_info += f"    - Riders: {', '.join(rider_names)}\n"
    #                 else:
    #                     profile_info += f"    - Riders: None listed\n"

    #                 # Add Exclusions to prompt
    #                 if exclusions:
    #                     profile_info += (
    #                         f"    - !! IMPORTANT EXCLUSION: {exclusions} !!\n"
    #                     )
    #     else:
    #         user_name = "User"
    #         profile_info = "\n\n--- USER PROFILE (YOUR SOURCE OF TRUTH) ---\n- No user profile provided.\n"

    #     salutation = f"Hi {user_name.split()[0] if user_name != 'User' else 'Hi'},"

    #     # Use the system prompt from config with formatting
    #     prompt_instructions = config.INSURANCE_SYSTEM_PROMPT.format(
    #         profile_info=profile_info,
    #         context_from_docs=context_from_docs,
    #         salutation=salutation,
    #         original_query=original_query,
    #     )

    #     # Call OpenAI API with streaming
    #     try:
    #         print("Sending streaming request to OpenAI API...")
    #         if not self.client:
    #             raise ValueError("OpenAI client is not initialized.")

    #         stream = self.client.chat.completions.create(
    #             model=config.RESPONSE_MODEL,
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": "You are an expert financial advisor. Answer insurance questions using the provided document chunks AND the user profile. The user profile, especially underwriting exclusions, is the absolute source of truth and overrides all document chunks. Exclusions are policy-scoped (a health plan exclusion cannot cancel CI/Life benefits) — evaluate each relevant policy independently. If the user question is not about insurance/policies/claims/coverage, politely respond that you can only answer insurance questions using the provided documents and stop. If no relevant context is available, state that plainly instead of inventing details. State exact amounts from documents; if an amount is not found, use EXACTLY: 'Amount not found in provided documents.' Never say 'depends' or 'the exact amount is not specified.' Do NOT output the literal token '<USER PROFILE>'. Use ONLY plain text - absolutely NO decorative characters, arrows (⬇, ↓, →), boxes, or emoji anywhere in your response. IMPORTANT: After each citation [Source X: filename, Page Y], show a brief excerpt (1-2 sentences) from that source in quotes to prove where the information came from. End cleanly after the last factual statement—do not add offers to help or follow-up questions.",
    #                 },
    #                 {"role": "user", "content": prompt_instructions},
    #             ],
    #             max_tokens=config.RESPONSE_MAX_TOKENS,
    #             temperature=config.RESPONSE_TEMPERATURE,
    #             stream=True,
    #         )

    #         print("Streaming response from OpenAI API...")

    #         for chunk in stream:
    #             if chunk.choices[0].delta.content is not None:
    #                 content = chunk.choices[0].delta.content
    #                 yield "data: " + json.dumps({"content": content}) + "\n\n"

    #         # Send final done message
    #         yield "data: " + json.dumps({"done": True}) + "\n\n"
    #         print("Finished streaming response from OpenAI API.")

    #     except Exception as e:
    #         print(f"Error during OpenAI API streaming call: {e}")
    #         yield "data: " + json.dumps(
    #             {
    #                 "error": "Sorry, I encountered an error while generating the response. Please try again later or check the system logs."
    #             }
    #         ) + "\n\n"

    def _generate_response_stream(
        self,
        original_query: str,
        search_results: List[Dict],
        is_personal_batch: bool,
        user_profile: Optional[Dict],
        include_final_done: bool = True,
    ):
        """
        Generate streaming response using Gemini Flash (Synchronous Version).
        """
        if not search_results:
            yield "data: " + json.dumps(
                {
                    "content": "I couldn't find any relevant information in the documents to answer your question.",
                }
            ) + "\n\n"
            if include_final_done:
                yield "data: " + json.dumps({"done": True}) + "\n\n"
            return

        context_parts = []
        max_chunks_for_context = config.MAX_CONTEXT_CHUNKS

        print(
            f"Building context from top {min(len(search_results), max_chunks_for_context)} chunks..."
        )
        for i, result in enumerate(search_results[:max_chunks_for_context], 1):
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            if content:
                filename = metadata.get("filename", "Unknown Document")
                page = metadata.get("page_number", "N/A")

                heading = metadata.get("page_heading", "General Information")
                source_ref = f"[Source {i}: {filename}, Page {page}]"
                context_parts.append(
                    f"{source_ref}\nPAGE HEADING: {heading}\n\n{content}"
                )

        if not context_parts:
            yield "data: " + json.dumps(
                {
                    "content": "Error: Found relevant documents but failed to extract content for context.",
                }
            ) + "\n\n"
            if include_final_done:
                yield "data: " + json.dumps({"done": True}) + "\n\n"
            return

        context_from_docs = "\n\n---\n\n".join(context_parts)

        # --- FIX: Build FULL User Profile String ---
        if is_personal_batch and user_profile:
            user_name = user_profile.get("name", "User")
            user_dob = user_profile.get("date_of_birth", "N/A")
            insurance_policies = user_profile.get("insurance_policies", {})

            profile_info = f"\n\n--- USER PROFILE (YOUR SOURCE OF TRUTH) ---\n"
            profile_info += f"- User Name: {user_name}\n"
            profile_info += f"- User DOB: {user_dob}\n"

            if insurance_policies:
                profile_info += f"- User's Policies:\n"
                for filename, policy_data in insurance_policies.items():
                    plan = policy_data.get("plan_name", "Unknown Plan")
                    tier = policy_data.get("tier", "N/A")
                    riders = policy_data.get("riders", [])
                    underwriting = policy_data.get("underwriting", {})
                    exclusions = underwriting.get("exclusions")

                    # Add policy and tier info
                    profile_info += f"  - Policy: {plan} (Tier: {tier})\n"

                    # Add rider info
                    if riders:
                        # Handle list of strings or list of dicts
                        rider_names = [
                            r.get("plan_name", r) if isinstance(r, dict) else r
                            for r in riders
                        ]
                        profile_info += f"    - Riders: {', '.join(rider_names)}\n"
                    else:
                        profile_info += f"    - Riders: None listed\n"

                    # Add Exclusions to prompt (CRITICAL for your query)
                    if exclusions:
                        profile_info += (
                            f"    - !! IMPORTANT EXCLUSION: {exclusions} !!\n"
                        )
        else:
            user_name = "User"
            profile_info = "\n\n--- USER PROFILE (YOUR SOURCE OF TRUTH) ---\n- No user profile provided.\n"

        salutation = f"Hi {user_name.split()[0] if user_name != 'User' else 'Hi'},"

        # Prepare the prompts
        system_instruction = (
            "You are an expert financial advisor. Answer insurance questions using the provided document chunks AND the user profile. "
            "The user profile, especially underwriting exclusions, is the absolute source of truth. "
            "State exact amounts from documents. Use ONLY plain text."
        )

        prompt_instructions = config.INSURANCE_SYSTEM_PROMPT.format(
            profile_info=profile_info,
            context_from_docs=context_from_docs,
            salutation=salutation,
            original_query=original_query,
        )

        # Call Gemini API with streaming (SYNCHRONOUSLY)
        try:
            print("Sending streaming request to Gemini API...")
            if self.gemini_llm is not None:
                # Use LangChain LLM streaming
                messages = [SystemMessage(content=system_instruction), HumanMessage(content=prompt_instructions)]
                for chunk in self.gemini_llm.stream(messages):
                    # chunk has .content property on each partial response
                    if getattr(chunk, "content", None):
                        yield "data: " + json.dumps({"content": chunk.content}) + "\n\n"
                # Send final done message
                yield "data: " + json.dumps({"done": True}) + "\n\n"
                print("Finished streaming response from Gemini API.")
            else:
                # Fallback: raw genai Api
                model = genai.GenerativeModel(
                    model_name=self.model_name, system_instruction=system_instruction
                )
                from google.generativeai.types import HarmCategory, HarmBlockThreshold
                response = model.generate_content(
                    prompt_instructions,
                    stream=True,
                    generation_config=genai.types.GenerationConfig(
                        temperature=config.RESPONSE_TEMPERATURE,
                        max_output_tokens=config.RESPONSE_MAX_TOKENS,
                    ),
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    },
                )
                for chunk in response:
                    if chunk.text:
                        yield "data: " + json.dumps({"content": chunk.text}) + "\n\n"
                # Send final done message
                yield "data: " + json.dumps({"done": True}) + "\n\n"
                print("Finished streaming response from Gemini API.")
        except Exception as e:
            print(f"Error during Gemini API streaming call: {e}")
            yield "data: " + json.dumps(
                {
                    "error": "Sorry, I encountered an error while generating the response."
                }
            ) + "\n\n"

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the search engine state."""
        if self.search_engine:
            return self.search_engine.get_stats()
        return {"error": "Search engine not initialized."}
