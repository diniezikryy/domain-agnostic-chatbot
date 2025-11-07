"""
LLM-based Judge Evaluator

Compares experiment outputs and ranks them based on:
- Correctness: Does it answer the question accurately for this user?
- Source Fidelity: Does it cite sources and avoid hallucination?
- Personalization: Does it use the user's profile context appropriately?
"""

import json
from typing import Dict, List, Any, Optional
from openai import OpenAI
from config.settings import Settings


class LLMJudgeEvaluator:
    """Use GPT as a judge to score and rank experiment answers."""

    def __init__(self, user_profile: Optional[Dict[str, Any]] = None):
        settings = Settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.user_profile = user_profile or {}
        self.model = "gpt-4o-mini"

    def _format_profile_context(self) -> str:
        """Format user profile into a readable summary for the judge."""
        if not self.user_profile:
            return "No user profile provided."

        lines = []
        lines.append(f"User: {self.user_profile.get('name', 'Unknown')}")
        lines.append(f"Age: {self.user_profile.get('age_last_birthday', 'Unknown')}")
        lines.append(f"Location: {self.user_profile.get('location', 'Unknown')}")
        lines.append(f"Smoking Status: {self.user_profile.get('smoking_status', 'Unknown')}")

        policies = self.user_profile.get("policy_details", [])
        if policies:
            lines.append(f"Policies Owned:")
            for policy in policies:
                insurer = policy.get("insurer", "Unknown")
                product = policy.get("product_name", "Unknown")
                plan_tier = policy.get("plan_tier", {})
                tier_name = plan_tier.get("tier", "")
                if tier_name:
                    lines.append(f"  - {insurer} {product} ({tier_name})")
                else:
                    lines.append(f"  - {insurer} {product}")

        return "\n".join(lines)

    def judge_answers(
        self,
        query: str,
        answers: Dict[str, str],  # {experiment_name: answer_text}
        retrieved_contexts: List[str],  # contexts used for the answer
    ) -> Dict[str, Any]:
        """
        Judge multiple answers to a query and rank them.

        Args:
            query: The user's question
            answers: Dict mapping experiment names to answer texts
            retrieved_contexts: List of context strings (retrieved from RAG)

        Returns:
            Dict with rankings, scores, and detailed feedback
        """
        if not answers:
            return {"error": "No answers to judge"}

        # Build context string for judge
        profile_context = self._format_profile_context()

        context_str = "\n\n".join([f"[Context {i}]\n{c}" for i, c in enumerate(retrieved_contexts, 1)])
        if not context_str:
            context_str = "(No retrieved contexts available)"

        # Build the judging prompt
        answers_str = "\n\n".join([f"**{exp_name}**\n{answer}" for exp_name, answer in answers.items()])

        # Stronger prompt that requires explicit evidence-checking and hallucination detection.
        # The judge must only credit facts that are present in the retrieved contexts. Any claim
        # in an answer that cannot be supported by the contexts should be listed as a hallucinated
        # claim and reduce the answer's "document_consistency" score.
        judge_prompt = f"""You are an expert insurance advisor evaluating RAG system outputs.
You will judge which answer best serves the user by measuring these criteria for each answer:

1) correctness (0.0-1.0): whether the answer actually answers the user's question and the answer's
   factual claims are true given the retrieved contexts.
2) document_consistency (0.0-1.0): how closely the answer's factual content matches the provided
   retrieved contexts. Credit only facts that are explicitly present in the contexts; deduct points for
   any factual statements that are not supported.
3) extraneous_hallucination_penalty (0.0-1.0): penalty representing the fraction of important claims
   that are unsupported or invented (higher is worse). Include a list of hallucinated claims.
4) accuracy_detail (0.0-1.0): level of numeric / detail accuracy (e.g., correct premium numbers, dates,
   percentages) when those details are present in the contexts.
5) personalization (0.0-1.0): whether the answer correctly uses the user's profile (age, policies,
   location) when applicable.
6) completeness (0.0-1.0): whether the answer addresses all parts of the question.

Rules for evaluation:
- Only credit a factual claim if it can be matched to text in the provided retrieved contexts. If an
  answer infers or adds information that is not present, mark it as a hallucinated claim and include it
  in the "hallucinated_claims" list for that answer.
- For numeric or precise claims (premiums, sums, dates), check whether an exact or clearly-supported
  value appears in the contexts; if not present, treat as unsupported.
- Provide per-criterion numeric scores in [0.0, 1.0], an overall_score (weighted average; weights below),
  and a short reasoning comment for each answer.

Weights to compute overall_score: correctness 0.35, document_consistency 0.30, accuracy_detail 0.15,
personalization 0.10, completeness 0.10. Subtract extraneous_hallucination_penalty as a final adjustment
but do not produce negative overall scores (floor at 0.0).

USER PROFILE:
{profile_context}

RETRIEVED POLICY CONTEXTS:
{context_str}

USER QUESTION:
{query}

EXPERIMENT ANSWERS:
{answers_str}

Output strictly valid JSON (nothing else). Required schema:
{{
  "rankings": [
    {{
      "experiment": "name",
      "rank": 1,
      "overall_score": 0.95,
      "scores": {{
         "correctness": 0.9,
         "document_consistency": 0.9,
         "extraneous_hallucination_penalty": 0.0,
         "accuracy_detail": 0.9,
         "personalization": 0.5,
         "completeness": 0.9
      }},
      "hallucinated_claims": ["claim text 1", "claim text 2"],
      "reasoning": "Concise human-readable justification"
    }},
    ...
  ],
  "winner": "experiment_name",
  "winner_reasoning": "Why this answer is best for this user"
}}

Only output JSON. Do not wrap in markdown or other text.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=1500,
                temperature=0.3,  # Low temp for consistency
            )

            # Defensive extraction of text
            result_text = None
            try:
                result_text = response.choices[0].message.content
            except Exception:
                try:
                    result_text = str(response.choices[0])
                except Exception:
                    result_text = ''

            if result_text is None:
                result_text = ''

            result_text = result_text.strip()

            # Try to extract JSON from response
            import re
            json_match = re.search(r"\{[\s\S]*\}", result_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"raw_response": result_text, "error": "Could not parse JSON from judge"}

            return result

        except Exception as e:
            return {"error": f"Judge evaluation failed: {str(e)}"}

        # Fallback (shouldn't be reached) - return an explicit error dict to satisfy callers.
        return {"error": "Judge did not return a result"}

    def generate_summary_report(
        self,
        query: str,
        judge_result: Dict[str, Any],
        answers: Dict[str, str],
    ) -> str:
        """Generate a human-readable summary of judge findings."""
        if "error" in judge_result:
            return f"Judge Error: {judge_result['error']}"

        summary_lines = []
        summary_lines.append(f"\n{'='*80}")
        summary_lines.append(f"QUESTION: {query}")
        summary_lines.append(f"{'='*80}\n")

        winner = judge_result.get("winner", "Unknown")
        winner_reasoning = judge_result.get("winner_reasoning", "")
        summary_lines.append(f"BEST ANSWER: {winner}")
        if winner_reasoning:
            summary_lines.append(f"Reasoning: {winner_reasoning}\n")

        rankings = judge_result.get("rankings", [])
        if rankings:
            summary_lines.append("RANKING:")
            for rank_info in rankings:
                exp = rank_info.get("experiment", "Unknown")
                rank = rank_info.get("rank", "?")
                score = rank_info.get("score", 0)
                reasoning = rank_info.get("reasoning", "")
                summary_lines.append(f"  {rank}. {exp} (Score: {score:.2f})")
                if reasoning:
                    summary_lines.append(f"     {reasoning[:100]}...")

        return "\n".join(summary_lines)
