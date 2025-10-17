"""
Query Processor
Handles domain-agnostic query processing using hybrid FAISS + BM25 search.
Enhanced for comprehensive retrieval and responses.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

from utils.search import HybridSearchEngine
from utils.embeddings import EmbeddingGenerator
from batch_manager import BatchManager

class QueryProcessor:
    def __init__(self, batch_manager: BatchManager):
        self.batch_manager = batch_manager
        self.search_engine = None
        self.current_batch_id = None

        # Enhanced decomposition prompt for comprehensive coverage
        self.decomposition_prompt_template = """
You are an expert system that decomposes complex insurance policy comparison questions into focused sub-questions.

CRITICAL RULES:
1. For comparison questions, generate BALANCED sub-questions for BOTH policies
2. Each sub-question must target ONE specific aspect of ONE policy
3. Always generate equal sub-questions for each policy being compared
4. Be specific about which policy each sub-question refers to
5. Generate 4-10 sub-questions total (MORE is better for comprehensive coverage)
6. **ALWAYS include feature-focused questions like "What are ALL unique features/benefits/riders in X policy?"**

**IMPORTANT: Your response MUST be in JSON format with a "questions" key containing an array of question strings.**

Examples:

User Question: "What are the pros and cons of SingLife versus FWD?"
Output (JSON format):
{{
    "questions": [
        "What are ALL the unique benefits and features of SingLife Essential Critical Illness II?",
        "What are ALL the unique benefits and features of FWD Critical Illness Plus?",
        "What are ALL the optional riders available in SingLife's policy?",
        "What are ALL the optional riders available in FWD's policy?",
        "What are the limitations or disadvantages of SingLife's policy?",
        "What are the limitations or disadvantages of FWD's policy?",
        "What special benefits does SingLife offer (like rewards, pre-existing conditions)?",
        "What special benefits does FWD offer (like premium waivers, auto-reload)?",
        "What makes SingLife's application process unique?",
        "What makes FWD's claim process unique?"
    ]
}}

User Question: "What exclusions are present in SingLife's policy that are not in FWD's policy?"
Output (JSON format):
{{
    "questions": [
        "What are ALL the exclusions listed in the SingLife Essential Critical Illness II policy?",
        "What are ALL the exclusions listed in the FWD Critical Illness Plus policy?",
        "What conditions or situations are excluded in SingLife's policy documents?",
        "What conditions or situations are excluded in FWD's policy documents?"
    ]
}}

User Question: {query}
Output (JSON format):
"""

        self.system_prompt = """
You are a knowledgeable assistant helping users find information from documents.
Answer questions clearly and accurately based on the provided context.
Always cite specific information from the documents when possible.
If you cannot find relevant information, say so clearly.
Be comprehensive and thorough in your responses.
"""

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into simpler sub-questions using a language model."""
        try:
            prompt = self.decomposition_prompt_template.format(query=query)
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)

            # Look for 'questions' key specifically
            if "questions" in result and isinstance(result["questions"], list):
                return result["questions"]

            # Fallback: find any list in the JSON object
            for key, value in result.items():
                if isinstance(value, list) and len(value) > 0:
                    return value

            print("Warning: No question list found in decomposition response")
            return []

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from decomposition: {e}")
            return [query]
        except Exception as e:
            print(f"Failed to decompose query: {e}")
            return [query]

    def _ensure_batch_loaded(self, batch_id: str) -> bool:
        """Ensure the specified batch is loaded in the search engine."""
        if self.current_batch_id == batch_id and self.search_engine:
            return True

        paths = self.batch_manager.get_batch_paths(batch_id)
        if not paths:
            print(f"Batch '{batch_id}' not found")
            return False

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

    def _is_comparison_query(self, query: str) -> bool:
        """Detect if query is asking for a comparison between policies."""
        comparison_keywords = [
            'compare', 'versus', 'vs', 'vs.', 'difference', 'different',
            'better than', 'worse than', 'between', 'both policies',
            'pros and cons', 'advantages', 'disadvantages'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in comparison_keywords)

    def _detect_policy_names(self, query: str) -> List[str]:
        """Detect which policies are mentioned in the query."""
        policies = []
        query_lower = query.lower()

        if 'singlife' in query_lower or 'sing life' in query_lower:
            policies.append('singlife')
        if 'fwd' in query_lower:
            policies.append('fwd')

        return policies

    def _balanced_search(self, sub_queries: List[str], is_comparison: bool,
                        mentioned_policies: List[str]) -> List[Dict]:
        """Perform balanced retrieval with increased coverage for comprehensive responses."""

        all_results = []

        # Perform search for each sub-query
        for sub_q in sub_queries:
            processed_sub_q = self._preprocess_query(sub_q)
            results = self.search_engine.hybrid_search(
                query=processed_sub_q,
                top_k=5
            )
            all_results.extend(results)

        # If not a comparison, just deduplicate and return
        if not is_comparison:
            return self._deduplicate_results(all_results)

        # For comparisons, ensure balanced retrieval from each policy
        policy_results = {policy: [] for policy in mentioned_policies}
        other_results = []

        for result in all_results:
            content = result.get('content', '').lower()
            filename = result.get('metadata', {}).get('filename', '').lower()

            categorized = False
            for policy in mentioned_policies:
                if policy in content or policy in filename:
                    policy_results[policy].append(result)
                    categorized = True
                    break

            if not categorized:
                other_results.append(result)

        # Balance results - INCREASED to 10 chunks per policy for comprehensiveness
        balanced = []
        max_per_policy = 10  # INCREASED from 6 to 10

        for policy in mentioned_policies:
            policy_chunks = self._deduplicate_results(policy_results[policy])
            balanced.extend(policy_chunks[:max_per_policy])

        # Add some other results
        balanced.extend(self._deduplicate_results(other_results)[:5])

        # Final deduplication
        balanced = self._deduplicate_results(balanced)

        # Print balance for debugging
        print(f"\n=== Retrieval Balance ===")
        for policy in mentioned_policies:
            count = sum(1 for r in balanced
                       if policy in r.get('content', '').lower() or
                       policy in r.get('metadata', {}).get('filename', '').lower())
            print(f"{policy.upper()}: {count} chunks")
        print(f"========================\n")

        return balanced

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content."""
        unique_results = []
        seen_content = set()

        for result in results:
            content = result.get('content', '')
            if content not in seen_content:
                unique_results.append(result)
                seen_content.add(content)

        return unique_results

    def _verify_answer_confidence(self, query: str, search_results: List[Dict],
                                  is_comparison: bool, mentioned_policies: List[str]) -> Tuple[bool, str]:
        """Verify if we have enough evidence to answer the query confidently."""

        if not search_results:
            return False, "No relevant documents found for your question."

        # For comparison queries, check if we have evidence from all mentioned policies
        if is_comparison and len(mentioned_policies) >= 2:
            found_policies = set()

            for result in search_results:
                content = result.get('content', '').lower()
                filename = result.get('metadata', {}).get('filename', '').lower()

                for policy in mentioned_policies:
                    if policy in content or policy in filename:
                        found_policies.add(policy)

            missing_policies = set(mentioned_policies) - found_policies

            if missing_policies:
                missing_str = ', '.join(missing_policies)
                found_str = ', '.join(found_policies) if found_policies else 'none'
                return False, (f"I can only find information about {found_str}. "
                             f"Cannot make a fair comparison without information about {missing_str}.")

        return True, ""

    def process_query(self, query: str, batch_id: str = None) -> str:
        """Process a query and return the response."""
        try:
            target_batch = batch_id or self.batch_manager.get_current_batch()
            if not target_batch:
                return "No batch specified and no active batch found."

            if not self._ensure_batch_loaded(target_batch):
                return f"Failed to load batch '{target_batch}'"

            start_time = time.time()

            is_comparison = self._is_comparison_query(query)
            mentioned_policies = self._detect_policy_names(query)

            print(f"Query type: {'Comparison' if is_comparison else 'Single-topic'}")
            print(f"Policies mentioned: {mentioned_policies}")

            # Step 1: Decompose query
            sub_queries = self._decompose_query(query)

            if not sub_queries:
                sub_queries = [query]
                print("Decomposition returned no questions, using original query")

            print(f"Decomposed into {len(sub_queries)} sub-queries:")
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq}")

            # Step 2: Perform balanced search
            unique_results = self._balanced_search(sub_queries, is_comparison, mentioned_policies)
            print(f"Retrieved {len(unique_results)} unique chunks")

            # Step 3: Verify evidence
            has_evidence, reason = self._verify_answer_confidence(
                query, unique_results, is_comparison, mentioned_policies
            )

            if not has_evidence:
                return f"I cannot confidently answer this question. {reason}"

            if not unique_results:
                return "No relevant documents found for your question."

            # Step 4: Generate response
            response = self._generate_response(query, unique_results, is_comparison)

            processing_time = time.time() - start_time
            print(f"Processing time: {processing_time:.2f}s")

            return response

        except Exception as e:
            return f"Error processing query: {e}"

    def _preprocess_query(self, query: str) -> str:
        """Basic query preprocessing."""
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.lower().split()
        filtered_words = [word for word in words if word not in stopwords]
        return ' '.join(filtered_words) if filtered_words else query

    def _generate_response(self, original_query: str, search_results: List[Dict],
                          is_comparison: bool) -> str:
        """Generate comprehensive response with strict evidence requirements."""
        if not search_results:
            return "I couldn't find any relevant information in the documents to answer your question."

        # Build context - INCREASED to 20 chunks for comparisons
        context_parts = []
        max_chunks = 20 if is_comparison else 15

        for i, result in enumerate(search_results[:max_chunks], 1):
            content = result.get('content', '').strip()
            metadata = result.get('metadata', {})

            if content:
                filename = metadata.get('filename', 'Unknown')
                page = metadata.get('page_number', 'N/A')
                year = metadata.get('year', 'N/A')

                source_header = f"[Source {i}: {filename}, Page {page}"
                if year != 'N/A':
                    source_header += f", Year {year}"
                source_header += "]"

                context_parts.append(f"{source_header}\n{content}")

        if not context_parts:
            return "I found some documents but couldn't extract relevant information to answer your question."

        context = "\n\n".join(context_parts)


        comparison_instructions = ""
        if is_comparison:
            comparison_instructions = """
COMPARISON-SPECIFIC RULES:
- **You MUST have evidence from ALL policies being compared**
- **To say "Policy A has X but Policy B doesn't", you need EXPLICIT evidence that Policy B excludes X**
- **If Policy B simply doesn't mention X, say: "Policy A explicitly covers X. Policy B's documents don't discuss this feature."**
- **NEVER assume silence means exclusion**
- **Only compare pricing examples with IDENTICAL customer profiles**
- **If profiles differ, explicitly state: "Cannot directly compare prices - different customer profiles"**
- **Never claim one policy is "better" or "cheaper" without specific comparable evidence**
"""

        prompt = f"""You are an insurance policy expert with STRICT EVIDENCE REQUIREMENTS.

Question: {original_query}

Policy Documents:
{context}

CRITICAL INSTRUCTIONS:
1. **ONLY state facts directly supported by the documents above**
2. **Every claim MUST be cited with [Source X] reference**
3. **If you don't have information, say "The documents don't provide information about X" and cite the sources you checked**
4. **Be COMPREHENSIVE - include ALL features, benefits, riders, and unique selling points mentioned in the sources**
5. **Don't just list obvious features - dig into special benefits, optional riders, and unique advantages**
6. **Be precise about what documents say vs. what they don't mention**

{comparison_instructions}

**COMPREHENSIVENESS REQUIREMENT:**
- For "pros and cons" questions: List AT LEAST 4-6 pros and 3-4 cons per policy if available in sources
- For "coverage" questions: List ALL conditions, benefits, and riders mentioned
- For "differences" questions: Identify EVERY difference mentioned, not just the most obvious ones
- Look for special benefits like premium waivers, rewards, ICU benefits, diabetic conditions coverage, etc.

CITATION FORMAT:
- Use [Source X] or [Sources X, Y] after each factual claim
- Example: "The policy covers diabetes-related conditions [Source 2, Page 15]"
- Example: "SingLife mentions pre-existing condition coverage [Source 1], but FWD's documents don't discuss this aspect [no mention in Sources 3-6]"
- **IMPORTANT: Even when saying "no information found", cite the sources you checked**

PROHIBITED BEHAVIORS:
❌ Making claims without source citations
❌ Assuming missing information means exclusion
❌ Comparing non-comparable customer profiles
❌ Stating general insurance principles not in the documents
❌ Hallucinating features not explicitly mentioned
❌ Being superficial - dig deep into ALL features mentioned in sources

Answer the question comprehensively with proper citations:
"""

        try:
            from openai import OpenAI
            from dotenv import load_dotenv
            load_dotenv()

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise insurance expert who provides comprehensive, well-cited answers from documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"OpenAI generation failed: {e}")
            return f"Based on the policy documents, I found relevant information but couldn't generate a complete response. Key details: {context[:300]}..."

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.search_engine:
            return self.search_engine.get_stats()
        return {}