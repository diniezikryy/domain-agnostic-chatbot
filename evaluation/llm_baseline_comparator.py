"""
Simple baseline LLM comparator for generating non-RAG responses.
"""

import time
from openai import OpenAI
from config.settings import Settings

class BaselineLLMComparator:
    """Generates baseline LLM responses without RAG."""
    
    def __init__(self):
        """Initialize the baseline comparator."""
        settings = Settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o-mini"
    
    def get_baseline_response(self, query: str) -> dict:
        """
        Get a baseline response from the LLM without RAG.
        
        Args:
            query: The user query
            
        Returns:
            dict with 'response' and 'time_seconds' keys
        """
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer the user's question directly and concisely."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            
            return {
                'response': answer,
                'time_seconds': elapsed_time
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                'response': f"Error generating baseline response: {str(e)}",
                'time_seconds': elapsed_time
            }
