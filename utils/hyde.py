"""
HyDE (Hypothetical Document Embeddings) Query Synthesis
Generates hypothetical answers to improve retrieval accuracy.
"""

import os
from typing import Optional
from openai import OpenAI


class HyDEQuerySynthesizer:
    """Generates hypothetical document for improved retrieval."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern with lazy initialization."""
        if cls._instance is None:
            cls._instance = super(HyDEQuerySynthesizer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize HyDE synthesizer (lazy loaded)."""
        if not HyDEQuerySynthesizer._initialized:
            self.client = None
            self.available = False
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                    self.available = True
                    print("HyDE synthesizer initialized successfully")
                else:
                    print("Warning: HyDE disabled - OPENAI_API_KEY not set")
            except Exception as e:
                print(f"Warning: HyDE initialization failed - {e}")
                self.available = False
            HyDEQuerySynthesizer._initialized = True
    
    def synthesize_hypothetical_document(self, query: str, max_tokens: int = 150) -> Optional[str]:
        """
        Generate a hypothetical document/answer for the query.
        
        Args:
            query: Original user query
            max_tokens: Maximum length of generated document
            
        Returns:
            Hypothetical document text or None if generation fails
        """
        if not self.available or not self.client:
            print("HyDE not available, using original query")
            return None
        
        try:
            prompt = f"""Generate a concise, factual answer to the following question as if you were 
an expert writing documentation. Focus on key terms and concepts that would appear in relevant documents.

Question: {query}

Hypothetical Answer:"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model for synthesis
                messages=[
                    {"role": "system", "content": "You are a documentation expert generating concise, factual answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            hypothetical_doc = response.choices[0].message.content.strip()
            print(f"HyDE: Generated hypothetical document ({len(hypothetical_doc)} chars)")
            return hypothetical_doc
            
        except Exception as e:
            print(f"Warning: HyDE synthesis failed - {e}. Using original query.")
            return None
    
    def enhance_query(self, original_query: str) -> str:
        """
        Enhance query with HyDE by combining original and hypothetical.
        
        Args:
            original_query: Original user query
            
        Returns:
            Enhanced query (original + hypothetical) or just original if HyDE fails
        """
        hypothetical = self.synthesize_hypothetical_document(original_query)
        
        if hypothetical:
            # Combine original query with hypothetical document
            enhanced = f"{original_query} {hypothetical}"
            return enhanced
        else:
            return original_query


def get_hyde_synthesizer() -> HyDEQuerySynthesizer:
    """Get singleton instance of HyDE synthesizer."""
    return HyDEQuerySynthesizer()
