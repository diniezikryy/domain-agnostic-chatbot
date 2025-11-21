# intent_analyzer.py
from openai import OpenAI
import json
import os
import re 
from typing import List, Dict, Any, Optional

class ResponseAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
    def _analyze_response_quality(
        self, 
        query: str, 
        response: str, 
        user_profile: Optional[Dict]
    ) -> bool:
        """
        Checks if the RAG response adequately answered the question.
        Returns True if good, False if needs deep research.
        """
        import re as regex  # local import to avoid scope issues in comprehensions
        response_text = response or ""
        response_lower = response_text.lower()

        # Special case: clear, definitive negatives should be treated as satisfactory
        definitive_negative_phrases = [
            "you do not have",
            "you don't have",
            "you are not covered",
            "you aren't covered",
            "does not have any",
            "no document found",
            "no relevant documents",
            "no records found",
            "no policy found",
            "no coverage found",
            "no evidence of coverage",
            "policy does not cover",
            "couldn't find any relevant document",
            "could not find any relevant document",
            "couldn't find any relevant documents",
            "could not find any relevant documents",
        ]
        definitive_negative_patterns = [
            r"\bnot (eligible|covered)\b",
            r"\bno (plan|policy|coverage) on file\b",
            r"\bno (matching )?document\b",
            r"\bnothing (relevant )?found\b",
        ]
        if any(phrase in response_lower for phrase in definitive_negative_phrases) or any(
            regex.search(pattern, response_lower) for pattern in definitive_negative_patterns
        ):
            print("📊 Quality Check: ✅ Definitive negative response detected; treating as satisfactory.")
            return True

        # Quick checks first
        if not response_text or len(response_text.strip()) < 50:
            print("📊 Quality Check: ❌ Response too short")
            return False
        
        response_lower = response.lower()
        query_lower = query.lower()
        
        # ===== 1. Explicit "don't know" phrases =====
        dont_know_phrases = [
            "i couldn't find",
            "no relevant information",
            "i don't have information",
            "unable to find",
            "not available in",
            "couldn't locate",
            "amount not found",
            "not found in the provided documents"
        ]
        
        if any(phrase in response_lower for phrase in dont_know_phrases):
            print(f"📊 Quality Check: ❌ Contains 'don't know' phrase")
            return False
        
        # ===== 2. Deflecting to external sources =====
        deflecting_phrases = [
            "please refer to your policy details",
            "consider consulting with an insurance advisor",
            "you may want to refer",
            "you may want to contact",
            "contact your insurance provider",
            "consult your financial adviser",
            "for personalized advice",
            "for more information, please contact"
        ]
        
        if any(phrase in response_lower for phrase in deflecting_phrases):
            print(f"📊 Quality Check: ❌ Deflecting to external sources")
            return False
        
        # ===== 3. Generic recommendations without specifics =====
        generic_recommendation_phrases = [
            "you might consider purchasing",
            "you should consider",
            "you may want to purchase",
            "you could look into",
            "it may be worth considering",
            "you might want to explore"
        ]
        
        # Check if user asked for specific recommendations
        asks_for_recommendations = any(word in query_lower for word in [
            'what should i get',
            'what should i buy',
            'what insurance',
            'what options',
            'what alternatives',
            'recommend',
            'suggestion'
        ])
        
        if asks_for_recommendations:
            # Response must have specific product/company names
            has_specific_product = any(name in response for name in [
                'AIG', 'Allianz', 'AXA', 'Sompo', 'NTUC Income', 
                'FWD', 'Tokio Marine', 'Chubb', 'Singlife',
                'TravelShield', 'TravelCare', 'IncomeShield'
            ])
            
            if any(phrase in response_lower for phrase in generic_recommendation_phrases) and not has_specific_product:
                print(f"📊 Quality Check: ❌ Generic recommendation without specific products")
                return False
        
        # ===== 4. Vague comparison language =====
        vague_comparison_phrases = [
            "typically aligns with",
            "generally similar to",
            "comparable to industry standards",
            "typical offerings",
            "industry standard often includes"
        ]
        
        is_comparison_query = any(word in query_lower for word in [
            'compare', 'comparison', 'versus', 'vs', 'better than', 'switch', 'difference'
        ])
        
        if is_comparison_query and any(phrase in response_lower for phrase in vague_comparison_phrases):
            print(f"📊 Quality Check: ❌ Vague comparison without data")
            return False
        
        # ===== 5. Check for missing specifics when user asks for them =====
        asks_for_specifics = any(word in query_lower for word in [
            'exactly', 'specifically', 'which', 'what are the'
        ])
        
        if asks_for_specifics:
            # Response should have concrete details (numbers, names, amounts)
            import re
            has_numbers = bool(re.search(r'\$[\d,]+|\d+%|\d{3,}', response))
            has_specific_names = bool(re.search(r'[A-Z][a-zA-Z]+ (Insurance|Shield|Care|Plan)', response))
            
            if not (has_numbers or has_specific_names):
                print(f"📊 Quality Check: ❌ User asked for specifics but response is vague")
                return False
        
        # ===== 6. LLM analysis =====
        analysis_prompt = f"""You are a STRICT quality checker for insurance chatbot responses.

    USER QUESTION: {query}

    BOT RESPONSE: {response}

    The user asked: "{query}"

    Does the response FULLY answer this question with SPECIFIC, ACTIONABLE information?

    ❌ Mark as BAD if response:
    - Says "you might consider..." without naming specific products/companies
    - Says "consult an advisor" or "refer to your policy" (deflecting)
    - Gives generic advice like "buy travel insurance" without specifics
    - Uses vague terms like "typically" or "generally" for comparisons
    - User asked "exactly what options" but got generic categories instead

    ✅ Mark as GOOD if response:
    - Names specific insurance products (e.g., "AIG Travel Guard", "Allianz TravelCare")
    - Provides concrete numbers ($X coverage, Y% co-insurance)
    - Directly answers what user asked for
    - Has proper citations [Source X: ...]

    Examples:

    USER: "What alternative insurance should I get?"
    BAD: "You might consider purchasing travel insurance for full coverage."
    GOOD: "Consider AIG Travel Guard ($100k coverage, $50/month) or Allianz TravelCare Premium ($200k coverage, $80/month)."

    USER: "Does my plan cover X?"
    BAD: "Your plan may cover X, please consult your advisor."
    GOOD: "Yes, your plan covers X up to $5,000 [Source 1: Policy.pdf, Page 5]."

    Respond ONLY: "GOOD" or "BAD"
    """
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=10,
                temperature=0
            )
            
            verdict = result.choices[0].message.content.strip().upper()
            is_good = "GOOD" in verdict
            
            print(f"📊 Response Quality Check (LLM): {'✅ GOOD' if is_good else '❌ BAD - Triggering deep research'}")
            
            return is_good
            
        except Exception as e:
            print(f"⚠️ Analysis failed: {e}. Assuming response is GOOD (safe default).")
            return True
    
