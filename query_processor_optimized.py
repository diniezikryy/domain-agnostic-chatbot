"""
Optimized Query Processor
Extends baseline QueryProcessor with Cross-Encoder reranking and RRF fusion.
"""

import asyncio
from typing import List, Dict, Any, Optional

from query_processor import QueryProcessor
from batch_manager import BatchManager
from utils.fusion import reciprocal_rank_fusion, weighted_reciprocal_rank_fusion
from config.optimization_settings import optimization_settings
import os
from openai import AsyncOpenAI
from functools import partial
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    # Fallback to raw genai if LangChain wrapper not available
    try:
        import google.generativeai as genai
    except Exception:
        genai = None
from config.settings import settings as config
from utils.reranker import tiered_reranker


class OptimizedQueryProcessor(QueryProcessor):
    """
    Optimized RAG pipeline with advanced retrieval techniques.
    
    Enhancements over baseline:
    1. Reciprocal Rank Fusion (RRF) for combining FAISS + BM25
    2. Cross-Encoder reranking for semantic relevance scoring
    3. Domain-specific heuristic reranking (inherited from baseline)
    
    Pipeline Flow:
    1. Query expansion (inherited)
    2. FAISS search (large pool, e.g., 100 candidates)
    3. BM25 search (large pool, e.g., 100 candidates)
    4. RRF fusion (combine FAISS + BM25)
    5. Cross-Encoder reranking (semantic scoring)
    6. Domain heuristic reranking (insurance-specific, inherited)
    7. Return top_k results
    """
    
    def __init__(self, batch_manager: BatchManager):
        """
        Initialize optimized pipeline.
        
        Args:
            batch_manager: Batch manager instance (same as baseline)
        """
        super().__init__(batch_manager)

        # Ensure optimized pipeline metadata composition is enabled
        # This gate controls whether parent-section text and document summaries
        # are appended to search/index texts. If you enable the optimized
        # pipeline dynamically, rebuild your batches to include parent
        # metadata (FAISS/BM25 indexes must be recreated).
        optimization_settings.ENABLE_OPTIMIZED_PIPELINE = True
        
        # Initialize cross-encoder reranker if enabled
        self.reranker = None
        if optimization_settings.USE_CROSS_ENCODER:
            print("[OptimizedQueryProcessor] Initializing Cross-Encoder...")
            try:
                from utils.reranker import CrossEncoderReranker

                self.reranker = CrossEncoderReranker(
                    model_name=optimization_settings.CROSS_ENCODER_MODEL,
                    device=optimization_settings.CROSS_ENCODER_DEVICE,
                    batch_size=optimization_settings.CROSS_ENCODER_BATCH_SIZE,
                    max_length=optimization_settings.CROSS_ENCODER_MAX_LENGTH,
                )
                print(f"[OptimizedQueryProcessor] Cross-Encoder ready: {self.reranker.get_model_info()}")
            except Exception as e:
                print(f"[OptimizedQueryProcessor] Failed to initialize Cross-Encoder: {e}")
                self.reranker = None
        else:
            print("[OptimizedQueryProcessor] Cross-Encoder disabled (USE_CROSS_ENCODER=false)")
    
    def _run_retrieval_sync(
        self,
        query: str,
        batch_id: str,
        user_profile: Optional[Dict] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Optimized synchronous retrieval pipeline.
        
        Overrides baseline to integrate RRF fusion and cross-encoder reranking.
        
        Args:
            query: User query
            batch_id: Batch ID to search within
            user_profile: Optional user profile for personalization
            top_k: Number of final results to return
        
        Returns:
            List of reranked documents
        """
        target_batch = batch_id or self.batch_manager.get_default_batch()
        if not target_batch:
            raise ValueError("No batch specified and no default batch set.")

        if not self._ensure_batch_loaded(target_batch):
            raise RuntimeError(f"Failed to load batch '{target_batch}'.")

        # Step 1: Query expansion (inherited from baseline)
        expanded_query = self._expand_query(query)
        is_personal_batch = target_batch.startswith("user_")

        # Step 2 & 3: Retrieve large candidate pool from FAISS and BM25
        # Use larger pool size to give reranker more candidates
        search_pool_size = max(
            optimization_settings.RERANK_TOP_K,  # Optimization setting (default 100)
            config.SEARCH_TOP_K,  # Baseline setting (default 60)
            top_k * 10  # Dynamic: at least 10x final output
        )
        
        print(f"[OptimizedQueryProcessor] Retrieving {search_pool_size} candidates...")

        if is_personal_batch and user_profile:
            # For multi-policy searches, use baseline's multi-policy logic
            num_policies = len(user_profile.get("insurance_policies", {}) or {"default": None})
            chunks_per_policy = max(search_pool_size // max(num_policies, 1), 10)
            
            raw_results = self._multi_policy_search(
                query=query,
                expanded_query=expanded_query,
                user_profile=user_profile,
                chunks_per_policy=chunks_per_policy,
            )
        else:
            # Standard search: get separate FAISS and BM25 results for RRF fusion
            if optimization_settings.USE_RRF_FUSION:
                raw_results = self._hybrid_search_with_rrf(
                    expanded_query,
                    search_pool_size
                )
            else:
                # Fallback to baseline weighted sum fusion
                raw_results = self.search_engine.hybrid_search(
                    query=expanded_query,
                    top_k=search_pool_size
                )

        # Step 4: Deduplicate (inherited from baseline)
        unique_results = self._deduplicate_results(raw_results)
        
        print(f"[OptimizedQueryProcessor] After deduplication: {len(unique_results)} candidates")

        # Step 5: Cross-Encoder reranking (NEW - semantic scoring)
        if self.reranker and optimization_settings.USE_CROSS_ENCODER:
            print(f"[OptimizedQueryProcessor] Applying Cross-Encoder reranking...")
            # Limit cross-encoder reranking to a manageable top-K to reduce
            # the chance of broad summary chunks being over-promoted and to
            # reduce computation. The RERANK_TOP_K setting controls this.
            unique_results = self.reranker.rerank(
                query=query,
                candidates=unique_results,
                top_k=optimization_settings.RERANK_TOP_K,
                score_field="cross_encoder_score",
            )
            print(f"[OptimizedQueryProcessor] Cross-Encoder reranking complete")
            
            # Step 5a: Filter by minimum cross-encoder score threshold
            # We allow negative thresholds now (e.g. -10.0) to filter only very irrelevant chunks
            if optimization_settings.MIN_RERANK_SCORE > -999:
                original_count = len(unique_results)
                unique_results = [
                    result for result in unique_results
                    if result.get("cross_encoder_score", 0) >= optimization_settings.MIN_RERANK_SCORE
                ]
                filtered_count = original_count - len(unique_results)
                if filtered_count > 0:
                    print(f"[OptimizedQueryProcessor] Filtered {filtered_count} low-scoring chunks (threshold={optimization_settings.MIN_RERANK_SCORE})")

        # Step 6: Domain heuristic reranking (inherited from baseline)
        # This applies insurance-specific boosting on top of cross-encoder scores
        reranked_results = self._rerank_insurance_results(
            query,
            unique_results,
            max_results=top_k
        )
        
        print(f"[OptimizedQueryProcessor] Final results: {len(reranked_results)}")

        return reranked_results

    def _hybrid_search_with_rrf(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using RRF fusion instead of weighted sum.
        
        Args:
            query: Expanded search query
            top_k: Number of candidates to retrieve
        
        Returns:
            Combined results using RRF
        """
        # Get separate FAISS and BM25 results
        faiss_results = self.search_engine._faiss_search(query, top_k)
        bm25_results = self.search_engine._bm25_search(query, top_k)
        
        print(f"[OptimizedQueryProcessor] RRF Fusion: {len(faiss_results)} FAISS + {len(bm25_results)} BM25")
        
        # Apply RRF fusion
        if optimization_settings.RRF_BM25_WEIGHT is not None:
            combined_results = weighted_reciprocal_rank_fusion(
                faiss_results=faiss_results,
                bm25_results=bm25_results,
                k=optimization_settings.RRF_K_CONSTANT,
                top_k=top_k,
                faiss_weight=optimization_settings.RRF_VECTOR_WEIGHT,
                bm25_weight=optimization_settings.RRF_BM25_WEIGHT,
            )
        else:
            combined_results = reciprocal_rank_fusion(
                faiss_results=faiss_results,
                bm25_results=bm25_results,
                k=optimization_settings.RRF_K_CONSTANT,
                top_k=top_k,
            )
        
        return combined_results

    def _rerank_insurance_results(self, query: str, results: List[Dict], max_results: int):
        """
        Use the tiered reranker for the optimized pipeline. If it fails, fall back
        to the baseline heuristic reranker inherited from QueryProcessor.
        """
        try:
            return tiered_reranker.rerank(query, results, max_results)
        except Exception as e:
            print(f"[OptimizedQueryProcessor] Tiered reranker failed, falling back to baseline heuristics: {e}")
            return super()._rerank_insurance_results(query, results, max_results)

    async def run_generation(
        self,
        query: str,
        search_results: List[Dict],
        is_personal_batch: bool = False,
        user_profile: Optional[Dict] = None,
    ) -> str:
        """
        Override to include parent section text snippet in the optimized pipeline only.
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
                parent_section_heading = metadata.get("parent_section_heading")
                parent_section_text = metadata.get("parent_section_text")

                source_ref = f"[Source {i}: {filename}, Page {page}]"

                # Build a context using parent section for the optimized pipeline
                context_display = f"{source_ref}\nPAGE HEADING: {heading}"
                if parent_section_heading:
                    context_display += f" (Section: {parent_section_heading})"

                if parent_section_text:
                    excerpt = (
                        parent_section_text[:300] + "..."
                        if len(parent_section_text) > 300
                        else parent_section_text
                    )
                    context_display += f"\n\n{excerpt}"

                context_display += f"\n\n{content}"
                context_parts.append(context_display)

        if not context_parts:
            return "Error: Found documents but failed to extract content."

        context_from_docs = "\n\n---\n\n".join(context_parts)

        # Now reuse baseline generation code to create the prompt and call the LLM
        if is_personal_batch and user_profile:
            user_name = user_profile.get("name", "User")
            insurance_policies = user_profile.get("insurance_policies", {})

            profile_info = f"\n\n--- USER PROFILE (YOUR SOURCE OF TRUTH) ---\n"
            profile_info += f"- User Name: {user_name}\n"

            if insurance_policies:
                profile_info += f"- User's Policies:\n"
                for filename, policy_data in insurance_policies.items():
                    plan = policy_data.get("plan_name", "Unknown Plan")
                    tier = policy_data.get("tier", "N/A")
                    profile_info += f"  - Policy: {plan} (Tier: {tier})\n"
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
            # If generation provider has been forced to OpenAI, use AsyncOpenAI
            if getattr(self, "generation_provider", "").lower() == "openai":
                async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = await async_client.chat.completions.create(
                    model=config.RESPONSE_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert financial advisor. Answer insurance questions using provided documents. Be concise and accurate.",
                        },
                        {"role": "user", "content": prompt_instructions},
                    ],
                    max_tokens=config.RESPONSE_MAX_TOKENS,
                    temperature=config.RESPONSE_TEMPERATURE,
                )

                return response.choices[0].message.content

            # Otherwise, proceed with Gemini/OpenAI fallback as before
            # Use Gemini via LangChain wrapper for the optimized pipeline as well
            system_instruction = "You are an expert financial advisor. Answer insurance questions using provided documents. Be concise and accurate."
            messages = [SystemMessage(content=system_instruction), HumanMessage(content=prompt_instructions)]

            # Prefer using the LangChain Google wrapper for Gemini
            if getattr(self, "gemini_llm", None) is not None:
                loop = asyncio.get_running_loop()
                invoker = partial(self.gemini_llm.invoke, messages)
                response = await loop.run_in_executor(None, invoker)
                return getattr(response, "content", str(response))

            # Fallback to raw google.generativeai if available
            if genai is not None:
                model = genai.GenerativeModel(model_name=getattr(self, "model_name", "gemini-2.5-flash"), system_instruction=system_instruction)
                response = await model.generate_content_async(
                    prompt_instructions,
                    generation_config=genai.types.GenerationConfig(
                        temperature=config.RESPONSE_TEMPERATURE,
                        max_output_tokens=config.RESPONSE_MAX_TOKENS,
                    ),
                )
                return response.text

            # If no Gemini integration is available, fall back to OpenAI for generation
            async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = await async_client.chat.completions.create(
                model=config.RESPONSE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_instruction,
                    },
                    {"role": "user", "content": prompt_instructions},
                ],
                max_tokens=config.RESPONSE_MAX_TOKENS,
                temperature=config.RESPONSE_TEMPERATURE,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error during optimized generation: {e}")
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Return information about the optimized pipeline configuration.
        
        Returns:
            Dictionary with pipeline settings and status
        """
        info = {
            "pipeline_type": "optimized",
            "cross_encoder_enabled": optimization_settings.USE_CROSS_ENCODER,
            "rrf_fusion_enabled": optimization_settings.USE_RRF_FUSION,
            "rerank_pool_size": optimization_settings.RERANK_TOP_K,
        }
        
        if self.reranker:
            info["cross_encoder_model"] = self.reranker.get_model_info()
        
        if optimization_settings.USE_RRF_FUSION:
            info["rrf_k_constant"] = optimization_settings.RRF_K_CONSTANT
        
        return info
