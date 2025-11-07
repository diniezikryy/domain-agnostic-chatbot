"""
Industry-Standard RAG Evaluation Metrics
Implements RAGAS-style and academic research metrics for proper RAG evaluation.
"""

# Ideal number of citations for normalization (domain-specific, adjust as needed)
IDEAL_CITATION_COUNT = 3.0

import re
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
from config.settings import Settings

# Regex patterns for extracting words of different lengths
WORD_PATTERN_5 = r'\b\w{5,}\b'  # Words with 5 or more characters
WORD_PATTERN_4 = r'\b\w{4,}\b'  # Words with 4 or more characters
WORD_PATTERN_3 = r'\b\w{3,}\b'  # Words with 3 or more characters

settings = Settings()

# Threshold constants for evaluation heuristics
FAITHFULNESS_SUPPORT_THRESHOLD = 0.5  # 50% of key terms in a sentence must be found in context to be considered supported
CONTEXT_RELEVANCE_THRESHOLD = 0.3     # 30% of query terms must be found in context to be considered relevant
RECALL_SUPPORT_THRESHOLD = 0.5        # 50% of key terms in a fact must be found in context to be considered supported


@dataclass
class StandardRAGMetrics:
    """Industry-standard RAG evaluation metrics."""
    # RAGAS-style metrics (0-1, higher is better)
    faithfulness: float  # Factual consistency with retrieved context
    answer_relevance: float  # How well answer addresses the query
    context_precision: float  # Precision of retrieved documents
    context_recall: float  # Recall of retrieved documents
    
    # Additional standard metrics
    hallucination_score: float  # 0-1, lower is better
    citation_quality: float  # Quality of citations (0-1)
    
    # Overall score
    ragas_score: float  # Harmonic mean of faithfulness, relevance, precision, recall
    
    # Metadata
    evaluation_method: str  # "llm_judge" or "heuristic"
    
    # Optional fields (must come after required fields)
    answer_correctness: Optional[float] = None  # Correctness vs ground truth (0-1)
    tokens_used: Optional[int] = None


class IndustryStandardEvaluator:
    """
    Evaluates RAG systems using industry-standard metrics.
    Based on RAGAS framework and academic research best practices.
    """
    
    def __init__(self, use_llm_judge: bool = True, judge_model: str = "gpt-4o-mini"):
        """
        Initialize evaluator.
        
        Args:
            use_llm_judge: Use LLM-as-judge for nuanced evaluation
            judge_model: Model to use for LLM-as-judge
        """
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model
        
        if use_llm_judge and settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.client = None
    
    def evaluate_faithfulness(
        self, 
        answer: str, 
        retrieved_contexts: List[str],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate faithfulness: degree to which answer is supported by retrieved context.
        
        RAGAS definition: Measures factual consistency with source documents.
        
        Args:
            answer: Generated answer
            retrieved_contexts: List of retrieved document chunks
            query: Original query (optional, for context)
            
        Returns:
            Dict with faithfulness score and breakdown
        """
        if self.use_llm_judge and self.client:
            return self._evaluate_faithfulness_llm(answer, retrieved_contexts, query)
        else:
            return self._evaluate_faithfulness_heuristic(answer, retrieved_contexts)
    
    def _evaluate_faithfulness_llm(
        self, 
        answer: str, 
        contexts: List[str],
        query: Optional[str]
    ) -> Dict[str, Any]:
        """Use LLM-as-judge to evaluate faithfulness."""
        context_text = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are evaluating the factual consistency of an AI-generated answer with its source documents.

QUERY: {query if query else "N/A"}

RETRIEVED CONTEXTS:
{context_text}

GENERATED ANSWER:
{answer}

Task: Rate the faithfulness of the answer (0.0 to 1.0) where:
- 1.0 = Every claim in the answer is directly supported by the contexts, OR the answer correctly states that the contexts don't contain the needed information
- 0.8 = Most claims supported, minor unsupported details
- 0.6 = Significant claims supported, but some unsupported information
- 0.4 = Mix of supported and unsupported claims
- 0.2 = Few claims supported by contexts
- 0.0 = Answer makes factual claims that contradict or are not supported by the contexts

IMPORTANT: If the answer honestly states that the contexts don't contain the requested information (e.g., "The provided contexts do not contain..."), this should score 1.0 for faithfulness, as it's accurately representing the limitations of the retrieved contexts rather than hallucinating information.

Identify:
1. Supported claims (directly from contexts)
2. Unsupported claims (not in contexts or inferred)
3. Contradictions (conflicts with contexts)

Respond in JSON format:
{{
    "faithfulness_score": <float 0-1>,
    "supported_claims": ["claim1", "claim2"],
    "unsupported_claims": ["claim1", "claim2"],
    "contradictions": ["claim1"],
    "reasoning": "Brief explanation"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for RAG systems. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return {
                "score": result.get("faithfulness_score", 0.0),
                "supported_claims": result.get("supported_claims", []),
                "unsupported_claims": result.get("unsupported_claims", []),
                "contradictions": result.get("contradictions", []),
                "reasoning": result.get("reasoning", ""),
                "method": "llm_judge",
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            print(f"Warning: LLM judge failed, falling back to heuristic: {e}")
            return self._evaluate_faithfulness_heuristic(answer, contexts)
    
    def _evaluate_faithfulness_heuristic(
        self, 
        answer: str, 
        contexts: List[str]
    ) -> Dict[str, Any]:
        """Heuristic faithfulness evaluation based on text overlap."""
        # Split answer into sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return {
                "score": 0.0,
                "method": "heuristic",
                "reasoning": "No substantial sentences in answer"
            }
        
        # Combine contexts
        combined_context = " ".join(contexts).lower()
        
        # Check each sentence for support
        supported_count = 0
        for sentence in sentences:
            # Extract key terms (words > 4 chars)
            words = [w.lower() for w in re.findall(WORD_PATTERN_5, sentence)]
            if not words:
                continue
            
            # Check if majority of key terms appear in context
            found_count = sum(1 for word in words if word in combined_context)
            if found_count / len(words) >= FAITHFULNESS_SUPPORT_THRESHOLD:  # 50% threshold
                supported_count += 1
        
        score = supported_count / len(sentences) if sentences else 0.0
        
        return {
            "score": score,
            "method": "heuristic",
            "reasoning": f"{supported_count}/{len(sentences)} sentences supported by context"
        }
    
    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate answer relevance: how well answer addresses the query.
        
        RAGAS definition: Measures if the answer is pertinent to the question asked.
        """
        if self.use_llm_judge and self.client:
            return self._evaluate_relevance_llm(query, answer)
        else:
            return self._evaluate_relevance_heuristic(query, answer)
    
    def _evaluate_relevance_llm(self, query: str, answer: str) -> Dict[str, Any]:
        """Use LLM-as-judge to evaluate answer relevance."""
        prompt = f"""You are evaluating how well an answer addresses a user's query.

QUERY: {query}

ANSWER: {answer}

Task: Rate the relevance (0.0 to 1.0) where:
- 1.0 = Directly and completely answers the query
- 0.8 = Answers the query well with minor tangents
- 0.6 = Partially answers, missing some aspects
- 0.4 = Loosely related but incomplete
- 0.2 = Barely addresses the query
- 0.0 = Completely irrelevant

Respond in JSON format:
{{
    "relevance_score": <float 0-1>,
    "query_aspects_addressed": ["aspect1", "aspect2"],
    "query_aspects_missed": ["aspect1"],
    "reasoning": "Brief explanation"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return {
                "score": result.get("relevance_score", 0.0),
                "addressed": result.get("query_aspects_addressed", []),
                "missed": result.get("query_aspects_missed", []),
                "reasoning": result.get("reasoning", ""),
                "method": "llm_judge",
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            print(f"Warning: LLM judge failed, falling back to heuristic: {e}")
            return self._evaluate_relevance_heuristic(query, answer)
    
    def _evaluate_relevance_heuristic(self, query: str, answer: str) -> Dict[str, Any]:
        """Heuristic relevance evaluation based on term overlap."""
        # Extract key terms from query (words > 3 chars)
        query_terms = set(re.findall(WORD_PATTERN_4, query.lower()))
        
        if not query_terms:
            return {"score": 0.5, "method": "heuristic", "reasoning": "No key terms in query"}
        
        # Count how many query terms appear in answer
        answer_lower = answer.lower()
        found_terms = sum(1 for term in query_terms if term in answer_lower)
        
        score = found_terms / len(query_terms)
        
        return {
            "score": score,
            "method": "heuristic",
            "reasoning": f"{found_terms}/{len(query_terms)} query terms found in answer"
        }
    
    def evaluate_context_precision(
        self,
        retrieved_contexts: List[str],
        query: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate context precision: proportion of retrieved contexts that are relevant.
        
        Academic definition: Precision@k for retrieval stage.
        """
        if self.use_llm_judge and self.client:
            return self._evaluate_context_precision_llm(retrieved_contexts, query, ground_truth)
        else:
            return self._evaluate_context_precision_heuristic(retrieved_contexts, query)
    
    def _evaluate_context_precision_llm(
        self,
        contexts: List[str],
        query: str,
        ground_truth: Optional[str]
    ) -> Dict[str, Any]:
        """Use LLM-as-judge to evaluate context precision."""
        contexts_text = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are evaluating retrieval quality for a RAG system.

QUERY: {query}

RETRIEVED CONTEXTS:
{contexts_text}

Task: For each context, determine if it is relevant to answering the query.
A context is relevant if it contains information that could help answer the query.

Respond in JSON format:
{{
    "relevant_contexts": [1, 3],  // list of relevant context numbers (1-indexed)
    "irrelevant_contexts": [2],
    "precision": <float 0-1>,  // proportion that are relevant
    "reasoning": "Brief explanation"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating document retrieval. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return {
                "score": result.get("precision", 0.0),
                "relevant_contexts": result.get("relevant_contexts", []),
                "irrelevant_contexts": result.get("irrelevant_contexts", []),
                "reasoning": result.get("reasoning", ""),
                "method": "llm_judge",
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            print(f"Warning: LLM judge failed, falling back to heuristic: {e}")
            return self._evaluate_context_precision_heuristic(contexts, query)
    
    def _evaluate_context_precision_heuristic(
        self,
        contexts: List[str],
        query: str
    ) -> Dict[str, Any]:
        """Heuristic context precision based on term overlap."""
        query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
        
        if not query_terms:
            return {"score": 0.5, "method": "heuristic"}
        
        relevant_count = 0
        for context in contexts:
            context_lower = context.lower()
            # Context is "relevant" if it contains >30% of query terms
            found = sum(1 for term in query_terms if term in context_lower)
            if found / len(query_terms) >= CONTEXT_RELEVANCE_THRESHOLD:
                relevant_count += 1
        
        score = relevant_count / len(contexts) if contexts else 0.0
        
        return {
            "score": score,
            "method": "heuristic",
            "reasoning": f"{relevant_count}/{len(contexts)} contexts deemed relevant"
        }
    
    def evaluate_context_recall(
        self,
        retrieved_contexts: List[str],
        ground_truth_contexts: Optional[List[str]] = None,
        answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate context recall: whether all needed information was retrieved.
        
        RAGAS definition: Proportion of ground truth contexts that were retrieved.
        This measures if the retrieval system found all the minimal necessary passages.
        
        Args:
            retrieved_contexts: List of retrieved document chunks
            ground_truth_contexts: List of ground truth context snippets (from golden set)
            answer: Generated answer (used as fallback if no ground truth)
            
        Returns:
            Dict with recall score and breakdown
        """
        if ground_truth_contexts:
            return self._evaluate_context_recall_with_ground_truth(retrieved_contexts, ground_truth_contexts)
        elif answer:
            # Fallback: estimate based on answer if no ground truth available
            return self._evaluate_context_recall_from_answer(retrieved_contexts, answer)
        else:
            return {"score": 0.5, "method": "unknown", "reasoning": "No ground truth or answer provided"}
    
    def _evaluate_context_recall_from_answer(
        self,
        contexts: List[str],
        answer: str
    ) -> Dict[str, Any]:
        """Estimate recall by checking if contexts support the answer."""
        combined_context = " ".join(contexts).lower()
        
        # Extract key facts from answer (sentences with specific info)
        sentences = re.split(r'[.!?]+', answer)
        fact_sentences = [s for s in sentences if (
            any(c.isdigit() for c in s) or 
            any(word in s.lower() for word in ['must', 'shall', 'will', 'covered', 'excluded', 'required'])
        ) and len(s.strip()) > 15]
        
        if not fact_sentences:
            return {"score": 0.7, "method": "heuristic", "reasoning": "No specific facts to check"}
        
        # Check if contexts support these facts
        supported = 0
        for fact in fact_sentences:
            key_terms = set(re.findall(WORD_PATTERN_5, fact.lower()))
            if key_terms:
                found = sum(1 for term in key_terms if term in combined_context)
                if found / len(key_terms) >= RECALL_SUPPORT_THRESHOLD:
                    supported += 1
        
        score = supported / len(fact_sentences) if fact_sentences else 0.5
        
        return {
            "score": score,
            "method": "heuristic",
            "reasoning": f"{supported}/{len(fact_sentences)} answer facts found in contexts"
        }
    
    def _evaluate_context_recall_with_ground_truth(
        self,
        contexts: List[str],
        ground_truth_contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate recall against ground truth contexts from golden set.
        
        This measures what fraction of the required ground truth context snippets
        were successfully retrieved by the RAG system.
        
        Args:
            contexts: Retrieved contexts from RAG pipeline
            ground_truth_contexts: List of minimal required context snippets
            
        Returns:
            Dict with score, found/missed contexts, and reasoning
        """
        if not ground_truth_contexts:
            return {"score": 1.0, "method": "ground_truth", "reasoning": "No ground truth contexts to check"}
        
        combined_retrieved = " ".join(contexts).lower()
        
        found_count = 0
        found_contexts = []
        missed_contexts = []
        
        for gt_context in ground_truth_contexts:
            # Extract key terms from ground truth context (words >= 4 chars)
            gt_terms = set(re.findall(WORD_PATTERN_4, gt_context.lower()))
            
            if not gt_terms:
                # If no key terms, do substring match
                if gt_context.lower() in combined_retrieved:
                    found_count += 1
                    found_contexts.append(gt_context)
                else:
                    missed_contexts.append(gt_context)
                continue
            
            # Check if majority of ground truth terms appear in retrieved contexts
            found_terms = sum(1 for term in gt_terms if term in combined_retrieved)
            term_coverage = found_terms / len(gt_terms)
            
            if term_coverage >= 0.6:  # 60% of terms must be found
                found_count += 1
                found_contexts.append(gt_context)
            else:
                missed_contexts.append(gt_context)
        
        score = found_count / len(ground_truth_contexts)
        
        return {
            "score": score,
            "method": "ground_truth",
            "reasoning": f"{found_count}/{len(ground_truth_contexts)} ground truth contexts retrieved",
            "found_contexts": found_contexts,
            "missed_contexts": missed_contexts
        }
    
    def evaluate_answer_correctness(
        self,
        query: str,
        answer: str,
        ground_truth_answer: str,
        retrieved_contexts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate answer correctness: semantic similarity with ground truth answer.
        
        This measures end-to-end quality by comparing the generated answer to
        the ideal ground truth answer from the golden set. Uses LLM-as-judge
        to assess semantic equivalence, factual accuracy, and completeness.
        
        Args:
            query: Original user query
            answer: Generated answer from RAG system
            ground_truth_answer: Ground truth answer from golden set
            retrieved_contexts: Optional retrieved contexts for additional evaluation
            
        Returns:
            Dict with correctness score (0-1), reasoning, and breakdown
        """
        if not self.client:
            # Fallback: simple heuristic based on word overlap
            return self._evaluate_answer_correctness_heuristic(answer, ground_truth_answer)
        
        contexts_text = ""
        if retrieved_contexts:
            contexts_text = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(retrieved_contexts)])
        
        prompt = f"""You are evaluating the correctness of an AI-generated answer against a ground truth answer.

QUERY: {query}

GROUND TRUTH ANSWER (the ideal correct answer):
{ground_truth_answer}

GENERATED ANSWER (from RAG system):
{answer}

{"RETRIEVED CONTEXTS:\n" + contexts_text if contexts_text else ""}

Task: Rate the correctness of the generated answer (0.0 to 1.0) by comparing it to the ground truth.

Scoring criteria:
- 1.0 = Semantically equivalent, all key facts present and accurate
- 0.9 = Minor differences in phrasing, all essential information correct
- 0.7-0.8 = Most key facts correct, minor factual discrepancies or missing details
- 0.5-0.6 = Partially correct, some key facts present but significant omissions or errors
- 0.3-0.4 = Minimal overlap, mostly incorrect or irrelevant
- 0.0-0.2 = Completely incorrect or contradicts ground truth

Consider:
1. Factual accuracy: Are numerical values, names, and specific details correct?
2. Completeness: Does it address all aspects covered in the ground truth?
3. Semantic equivalence: Does it convey the same meaning even if worded differently?

IMPORTANT: If the generated answer states "The provided contexts do not contain..." and this is factually correct (the information truly isn't in the contexts), you should evaluate based on whether this matches the ground truth expectation. If the ground truth expects a specific answer but the system correctly identifies missing information, this is still valuable but should be scored based on alignment with ground truth.

Respond in JSON format:
{{
    "correctness_score": <float 0-1>,
    "factual_accuracy": <float 0-1>,
    "completeness": <float 0-1>,
    "semantic_similarity": <float 0-1>,
    "key_differences": ["difference1", "difference2"],
    "reasoning": "Brief explanation of score"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for RAG systems. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
            
            result = json.loads(content)
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return {
                "score": result.get("correctness_score", 0.0),
                "factual_accuracy": result.get("factual_accuracy", 0.0),
                "completeness": result.get("completeness", 0.0),
                "semantic_similarity": result.get("semantic_similarity", 0.0),
                "key_differences": result.get("key_differences", []),
                "reasoning": result.get("reasoning", ""),
                "method": "llm_judge",
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            print(f"Warning: LLM judge failed for answer correctness, falling back to heuristic: {e}")
            return self._evaluate_answer_correctness_heuristic(answer, ground_truth_answer)
    
    def _evaluate_answer_correctness_heuristic(
        self,
        answer: str,
        ground_truth_answer: str
    ) -> Dict[str, Any]:
        """Heuristic answer correctness based on word overlap."""
        # Extract key terms from both answers
        answer_terms = set(re.findall(WORD_PATTERN_4, answer.lower()))
        gt_terms = set(re.findall(WORD_PATTERN_4, ground_truth_answer.lower()))
        
        if not gt_terms:
            return {"score": 0.5, "method": "heuristic", "reasoning": "No key terms in ground truth"}
        
        # Calculate Jaccard similarity
        intersection = answer_terms & gt_terms
        union = answer_terms | gt_terms
        
        if not union:
            return {"score": 0.0, "method": "heuristic", "reasoning": "No terms to compare"}
        
        score = len(intersection) / len(union)
        
        return {
            "score": score,
            "method": "heuristic",
            "reasoning": f"Term overlap: {len(intersection)}/{len(union)} terms in common"
        }
    
    def evaluate_rag_response(
        self,
        query: str,
        answer: str,
        retrieved_contexts: List[str],
        ground_truth_contexts: Optional[List[str]] = None,
        ground_truth_answer: Optional[str] = None
    ) -> StandardRAGMetrics:
        """
        Comprehensive evaluation using industry-standard metrics.
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_contexts: List of retrieved document chunks
            ground_truth_contexts: Optional list of ground truth context snippets for recall
            ground_truth_answer: Optional ground truth answer for correctness evaluation
            
        Returns:
            StandardRAGMetrics with all scores
        """
        print("  [EVAL] Evaluating with industry-standard metrics...")
        
        # Core RAGAS metrics
        faithfulness_result = self.evaluate_faithfulness(answer, retrieved_contexts, query)
        relevance_result = self.evaluate_answer_relevance(query, answer)
        precision_result = self.evaluate_context_precision(retrieved_contexts, query)
        recall_result = self.evaluate_context_recall(retrieved_contexts, ground_truth_contexts, answer)
        
        faithfulness_score = faithfulness_result['score']
        relevance_score = relevance_result['score']
        precision_score = precision_result['score']
        recall_score = recall_result['score']
        
        # Optional: Answer Correctness (if ground truth answer provided)
        answer_correctness_score = None
        if ground_truth_answer:
            correctness_result = self.evaluate_answer_correctness(query, answer, ground_truth_answer, retrieved_contexts)
            answer_correctness_score = correctness_result['score']
            print(f"  [EVAL] Answer Correctness: {answer_correctness_score:.3f}")
        
        # Calculate RAGAS score (harmonic mean of 4 core metrics)
        # Harmonic mean = n / (1/x1 + 1/x2 + ... + 1/xn)
        scores = [faithfulness_score, relevance_score, precision_score, recall_score]
        valid_scores = [s for s in scores if s > 0]
        if valid_scores:
            ragas_score = len(valid_scores) / sum(1/s for s in valid_scores)
        else:
            ragas_score = 0.0
        
        # Additional metrics
        # RAGAS standard: hallucination is the inverse of faithfulness
        # Low faithfulness = high hallucination (answer contains unsupported claims)
        hallucination_score = 1.0 - faithfulness_score  # Inverse of faithfulness
        
        # Citation quality (simple heuristic)
        citation_patterns = [r'\[source[^\]]*\]', r'\(source[^\)]*\)', r'according to', r'based on']
        citations = sum(len(re.findall(pattern, answer, re.IGNORECASE)) for pattern in citation_patterns)
        citation_quality = min(1.0, citations / 3.0)  # Normalize: 3+ citations = perfect
        
        # Sum tokens
        total_tokens = sum([
            faithfulness_result.get('tokens_used', 0),
            relevance_result.get('tokens_used', 0),
            precision_result.get('tokens_used', 0),
            recall_result.get('tokens_used', 0)
        ])
        
        method = "llm_judge" if self.use_llm_judge else "heuristic"
        
        return StandardRAGMetrics(
            faithfulness=faithfulness_score,
            answer_relevance=relevance_score,
            context_precision=precision_score,
            context_recall=recall_score,
            hallucination_score=hallucination_score,
            citation_quality=citation_quality,
            ragas_score=ragas_score,
            evaluation_method=method,
            answer_correctness=answer_correctness_score,
            tokens_used=total_tokens if total_tokens > 0 else None
        )
