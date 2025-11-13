"""
Embedding Generator
Handles text embedding generation for FAISS indexing.
"""
from dotenv import load_dotenv

# Load .env file
load_dotenv()

import os
import time
from typing import List
import numpy as np

class EmbeddingGenerator:
    """Generates embeddings for text chunks."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            print(f"DEBUG: API Key loaded: {api_key[:5]}...{api_key[-4:]}" if api_key else "DEBUG: API Key is None")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self.client = OpenAI(api_key=api_key)
            print(f"OpenAI client initialized with model: {self.model_name}")

        except ImportError:
            print("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")

    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """Generate embeddings for a list of texts with rate limiting."""
        if not self.client:
            print("OpenAI client not available")
            return []

        embeddings = []

        try:
            # Process in batches to avoid API limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                print(f"Generating embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

                # Retry logic with exponential backoff for rate limit errors
                max_retries = 5
                retry_delay = 1  # Start with 1 second
                
                for retry in range(max_retries):
                    try:
                        response = self.client.embeddings.create(
                            model=self.model_name,
                            input=batch # We assume the input is already clean
                        )

                        batch_embeddings = [np.array(data.embedding) for data in response.data]
                        embeddings.extend(batch_embeddings)
                        
                        # Add a small delay between batches to avoid rate limits
                        if i + batch_size < len(texts):
                            time.sleep(0.5)
                        
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        error_str = str(e)
                        if "429" in error_str or "rate_limit" in error_str.lower():
                            if retry < max_retries - 1:
                                print(f"  Rate limit hit. Retrying in {retry_delay}s... (attempt {retry + 1}/{max_retries})")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                print(f"  Rate limit exceeded after {max_retries} retries. Skipping batch.")
                                raise
                        else:
                            # Non-rate-limit error, raise immediately
                            raise

            print(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else np.array([])

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        # text-embedding-3-small has 1536 dimensions
        if "text-embedding-3-small" in self.model_name:
            return 1536
        elif "text-embedding-3-large" in self.model_name:
            return 3072
        else:
            # Default fallback
            return 1536